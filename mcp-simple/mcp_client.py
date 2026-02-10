#!/usr/bin/env python3
"""
MCP Client supporting both STDIO and HTTP streaming transports
"""

import asyncio
import json
import sys

from icecream import ic

try:
    import aiohttp as aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class MCPClient:
    """MCP client supporting both STDIO and HTTP streaming transports"""

    def __init__(self, server_command=None, base_url=None, transport="stdio"):
        """
        Initialize MCP client
        Args:
            server_command: Command to start STDIO server (for stdio transport)
            base_url: Base URL for HTTP server (for http transport)
            transport: Transport type - "stdio" or "http"
        """
        if transport == "stdio" and not server_command:
            raise ValueError("server_command required for stdio transport")
        if transport == "http" and not base_url:
            base_url = "http://localhost:8080"
        if transport == "http" and not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp package required for HTTP transport. Install with: pip install aiohttp")

        self.transport = transport
        self.server_command = server_command
        self.base_url = base_url
        self.process = None
        self.session = None
        self.request_id = 0

    def get_next_id(self):
        """Get next request ID"""
        self.request_id += 1
        return self.request_id

    async def start_server(self):
        """Start the MCP server or connect to HTTP server"""
        if self.transport == "stdio":
            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        elif self.transport == "http":
            import aiohttp

            self.session = aiohttp.ClientSession()
            # Test connection
            try:
                async with self.session.get(f"{self.base_url}/health") as response:
                    if response.status != 200:
                        raise ConnectionError(f"HTTP server not available: {response.status}")
            except Exception as e:
                await self.session.close()
                raise ConnectionError(f"Cannot connect to HTTP server at {self.base_url}: {e}")

    async def send_request(self, method, params=None):
        """Send a request to the server"""
        request = {"jsonrpc": "2.0", "id": self.get_next_id(), "method": method, "params": params or {}}

        if self.transport == "stdio":
            message = json.dumps(request) + "\n"
            self.process.stdin.write(message.encode())
            await self.process.stdin.drain()

            # Read response
            response_line = await self.process.stdout.readline()
            if response_line:
                try:
                    response = json.loads(response_line.decode().strip())
                    return response
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response: {e}")
                    print(f"Raw response: {response_line.decode().strip()}")
                    return None
            else:
                # Check if process has terminated
                if self.process.returncode is not None:
                    stderr = await self.process.stderr.read()
                    print(f"Server process terminated with code {self.process.returncode}")
                    if stderr:
                        print(f"Server stderr: {stderr.decode()}")
                return None

        elif self.transport == "http":
            async with self.session.post(
                f"{self.base_url}/message", json=request, headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    print(f"HTTP error {response.status}: {error_text}")
                    return None

    async def send_notification(self, method, params=None):
        """Send a notification to the server"""
        notification = {"jsonrpc": "2.0", "method": method, "params": params or {}}

        if self.transport == "stdio":
            message = json.dumps(notification) + "\n"
            self.process.stdin.write(message.encode())
            await self.process.stdin.drain()
        elif self.transport == "http":
            # For HTTP transport, notifications are sent as POST requests without expecting a response
            async with self.session.post(
                f"{self.base_url}/message", json=notification, headers={"Content-Type": "application/json"}
            ) as response:
                # Notifications don't expect responses, but we should handle errors
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Notification error {response.status}: {error_text}")

    async def initialize(self):
        """Initialize the MCP session"""
        # Send initialize request
        init_response = await self.send_request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "Test MCP Client", "version": "1.0.0"},
            },
        )

        if init_response and "result" in init_response:
            result = init_response["result"]
            print("✓ Server initialized successfully")

            # Safely access nested fields
            if "protocolVersion" in result:
                print(f"  Protocol version: {result['protocolVersion']}")

            if "serverInfo" in result:
                server_info = result["serverInfo"]
                name = server_info.get("name", "Unknown")
                version = server_info.get("version", "Unknown")
                print(f"  Server: {name} v{version}")

            # Send initialized notification
            await self.send_notification("initialized")
            return True
        else:
            print("✗ Failed to initialize server")
            if init_response and "error" in init_response:
                error = init_response["error"]
                print(f"  Error: {error.get('message', 'Unknown error')}")
            return False

    async def list_tools(self):
        """List available tools"""
        response = await self.send_request("tools/list")
        if response and "result" in response:
            tools = response["result"]["tools"]
            print(f"\n✓ Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description']}")
            return tools
        else:
            print("✗ Failed to list tools")
            return []

    async def call_tool(self, name, arguments):
        """Call a tool"""
        response = await self.send_request("tools/call", {"name": name, "arguments": arguments})

        if response and "result" in response:
            content = response["result"]["content"]
            print(f"\n✓ Tool '{name}' result:")
            for item in content:
                if item["type"] == "text":
                    print(f"  {item['text']}")
            return response["result"]
        elif response and "error" in response:
            error = response["error"]
            print(f"\n✗ Tool '{name}' failed: {error.get('message', 'Unknown error')}")
            return None
        else:
            print(f"\n✗ Tool '{name}' failed: No response received")
            return None

    async def list_resources(self):
        """List available resources"""
        response = await self.send_request("resources/list")
        if response and "result" in response:
            resources = response["result"]["resources"]
            print(f"\n✓ Found {len(resources)} resources:")
            for resource in resources:
                print(f"  - {resource['uri']}: {resource['name']}")
            return resources
        else:
            print("✗ Failed to list resources")
            return []

    async def read_resource(self, uri):
        """Read a resource"""
        response = await self.send_request("resources/read", {"uri": uri})

        if response and "result" in response:
            contents = response["result"]["contents"]
            print(f"\n✓ Resource '{uri}' content:")
            for content in contents:
                print(f"  {content['text']}")
            return response["result"]
        else:
            if response and "error" in response:
                error = response["error"]
                print(f"✗ Resource read failed: {error.get('message', 'Unknown error')}")
            else:
                print("✗ Resource read failed: No response received")
            return None

    async def close(self):
        """Close the client and server process/connection"""
        if self.transport == "stdio" and self.process:
            self.process.terminate()
            await self.process.wait()
        elif self.transport == "http" and self.session:
            await self.session.close()


async def run_tests(transport="stdio", server_path="mcp_server.py", base_url="http://localhost:8080"):
    """Run comprehensive tests of the MCP server"""
    print(f"🚀 Starting MCP Server Tests ({transport.upper()} transport)")
    print("=" * 60)

    # Create client based on transport
    if transport == "stdio":
        client = MCPClient(server_command=[sys.executable, server_path, "stdio"], transport="stdio")
    elif transport == "http":
        client = MCPClient(base_url=base_url, transport="http")
    else:
        raise ValueError(f"Unknown transport: {transport}")

    try:
        await client.start_server()
        print(f"✅ Connected via {transport.upper()} transport")

        # Initialize
        if not await client.initialize():
            return

        # List and test tools
        tools = await client.list_tools()

        if tools:
            # Test echo tool
            await client.call_tool("echo", {"text": "Hello, MCP!"})

            # Test calculate tool
            await client.call_tool("calculate", {"expression": "2 + 3 * 4"})

            # Test reverse tool
            await client.call_tool("reverse", {"text": "MCP Server"})

            # Test error case
            print("\n🔧 Testing error handling with nonexistent tool...")
            await client.call_tool("nonexistent", {})

        # List and test resources
        try:
            resources = await client.list_resources()

            if resources:
                for resource in resources:
                    await client.read_resource(resource["uri"])
        except Exception as e:
            print(f"❌ Resource test failed: {e}")

        print("\n✅ All tests completed!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")

    finally:
        await client.close()


async def main():
    """Main entry point with argument parsing"""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Client with STDIO and HTTP streaming support")
    parser.add_argument(
        "transport", choices=["stdio", "http"], default="stdio", nargs="?", help="Transport type to use"
    )
    parser.add_argument("--server", default="mcp_server.py", help="Path to MCP server for STDIO transport")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL for HTTP transport")

    args = parser.parse_args()
    ic(args)

    await run_tests(transport=args.transport, server_path=args.server, base_url=args.url)


if __name__ == "__main__":
    asyncio.run(main())
