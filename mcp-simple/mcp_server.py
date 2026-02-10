#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server Implementation from Scratch

This is a complete MCP server implementation that follows the MCP specification
without using any existing MCP packages.
"""

import asyncio
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Union

from icecream import ic


# MCP Protocol Types and Classes
@dataclass
class MCPRequest:
    """MCP request message"""

    jsonrpc: str
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None


@dataclass
class MCPResponse:
    """MCP response message"""

    jsonrpc: str
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class MCPNotification:
    """MCP notification message"""

    jsonrpc: str
    method: str
    params: Optional[Dict[str, Any]] = None


@dataclass
class MCPError:
    """MCP error structure"""

    code: int
    message: str
    data: Optional[Any] = None


@dataclass
class Tool:
    """Tool definition"""

    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class Resource:
    """Resource definition"""

    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None


class MCPServerError(Exception):
    """Base exception for MCP server errors"""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)


class MCPTransport(ABC):
    """Abstract transport layer for MCP communication"""

    @abstractmethod
    async def read_message(self) -> Optional[Dict[str, Any]]:
        """Read a message from the transport"""
        pass

    @abstractmethod
    async def write_message(self, message: Dict[str, Any]) -> None:
        """Write a message to the transport"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the transport"""
        pass


class StdioTransport(MCPTransport):
    """Standard I/O transport implementation"""

    def __init__(self):
        self.closed = False
        # Use line buffering for real-time communication
        sys.stdout.reconfigure(line_buffering=True)

    async def read_message(self) -> Optional[Dict[str, Any]]:
        """Read a JSON-RPC message from stdin"""
        if self.closed:
            return None

        try:
            # Use asyncio to read from stdin in a non-blocking way
            loop = asyncio.get_event_loop()
            line = await loop.run_in_executor(None, sys.stdin.readline)

            if not line:
                return None

            line = line.strip()
            if not line:
                return None

            # Parse JSON
            message = json.loads(line)
            return message

        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logging.error(f"Error reading message: {e}")
            return None

    async def write_message(self, message: Dict[str, Any]) -> None:
        """Write a JSON-RPC message to stdout"""
        if self.closed:
            return

        try:
            json_str = json.dumps(message, separators=(",", ":"))
            print(json_str, flush=True)
        except Exception as e:
            logging.error(f"Error writing message: {e}")

    async def close(self) -> None:
        """Close the transport"""
        self.closed = True


class HTTPStreamTransport(MCPTransport):
    """HTTP Streaming transport for modern streaming HTTP responses"""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.app = None
        self.runner = None
        self.site = None
        self.client_connections = {}
        self.message_queue = asyncio.Queue()
        self.response_queues = {}
        self.closed = False

    async def start_server(self):
        """Start the HTTP streaming server using aiohttp"""
        from aiohttp import web

        self.app = web.Application()

        # Add CORS middleware
        self.app.middlewares.append(self._create_cors_middleware())

        # Modern HTTP streaming endpoints
        self.app.router.add_post("/stream", self._handle_streaming_connection)
        self.app.router.add_post("/message", self._handle_message_post)
        self.app.router.add_options("/{path:.*}", self._handle_cors_preflight)
        self.app.router.add_get("/health", self._handle_health_check)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        print(f"HTTP Streaming transport listening on http://{self.host}:{self.port}")
        print("Endpoints: POST /stream, POST /message, GET /health")

    def _create_cors_middleware(self):
        """Create CORS middleware"""
        from aiohttp.web import middleware

        @middleware
        async def cors_middleware(request, handler):
            # Skip CORS middleware for OPTIONS requests (handled by preflight handler)
            if request.method == "OPTIONS":
                return await handler(request)

            response = await handler(request)
            response.headers.update(
                {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    "Access-Control-Max-Age": "86400",
                }
            )
            return response

        return cors_middleware

    async def _handle_streaming_connection(self, request):
        """Handle HTTP streaming connection for bidirectional MCP communication"""
        import json

        from aiohttp.web import StreamResponse

        # Create streaming response with proper headers for HTTP streaming
        response = StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "application/x-ndjson",  # Newline-delimited JSON
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
        )

        await response.prepare(request)

        client_id = id(request)
        client_queue = asyncio.Queue()
        self.client_connections[client_id] = response
        self.response_queues[client_id] = client_queue

        try:
            # Send initial connection confirmation
            connection_msg = {"type": "connection", "status": "established", "protocol": "MCP-HTTP-Stream"}
            await response.write((json.dumps(connection_msg) + "\n").encode())

            # Process initial request body if present
            if request.content_length and request.content_length > 0:
                request_data = await request.json()
                await self.message_queue.put(request_data)

            # Handle streaming responses
            while not self.closed:
                try:
                    # Wait for response messages
                    response_msg = await asyncio.wait_for(client_queue.get(), timeout=30.0)

                    # Send response as newline-delimited JSON
                    json_line = json.dumps(response_msg) + "\n"
                    await response.write(json_line.encode())

                except asyncio.TimeoutError:
                    # Send keep-alive ping
                    keepalive = {"type": "ping", "timestamp": asyncio.get_event_loop().time()}
                    await response.write((json.dumps(keepalive) + "\n").encode())

        except Exception as e:
            logging.error(f"HTTP streaming connection error: {e}")
        finally:
            if client_id in self.client_connections:
                del self.client_connections[client_id]
            if client_id in self.response_queues:
                del self.response_queues[client_id]

        return response

    async def _handle_message_post(self, request):
        """Handle POST requests with JSON messages"""
        from aiohttp.web import json_response

        try:
            # Read JSON message
            message = await request.json()

            # Process the message directly if we have an MCP server instance
            if hasattr(self, "mcp_server") and self.mcp_server:
                response = await self._process_mcp_message(message)
                if response:
                    return json_response(response)
                else:
                    return json_response({"status": "processed"})
            else:
                # Queue the message for processing (fallback)
                await self.message_queue.put(message)
                return json_response({"status": "queued"})

        except Exception as e:
            logging.error(f"Error handling POST message: {e}")
            return json_response({"error": str(e)}, status=500)

    async def _process_mcp_message(self, message_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process MCP message and return response"""
        try:
            # Validate JSON-RPC structure
            if message_dict.get("jsonrpc") != "2.0":
                return {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid JSON-RPC version"}}

            # Handle requests (expect response)
            if "id" in message_dict and "method" in message_dict:
                request = MCPRequest(
                    jsonrpc="2.0",
                    id=message_dict["id"],
                    method=message_dict["method"],
                    params=message_dict.get("params"),
                )
                response = await self.mcp_server.handle_request(request)

                # Create response dict with proper JSON-RPC structure
                response_dict = {"jsonrpc": response.jsonrpc, "id": response.id}
                if response.error is not None:
                    response_dict["error"] = response.error
                else:
                    response_dict["result"] = response.result
                return response_dict

            # Handle notifications (no response expected)
            elif "method" in message_dict:
                notification = MCPNotification(
                    jsonrpc="2.0",
                    method=message_dict["method"],
                    params=message_dict.get("params"),
                )
                await self.mcp_server.handle_notification(notification)
                return None  # No response for notifications

            else:
                return {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid request structure"}}

        except Exception as e:
            logging.error(f"Error processing MCP message: {e}")
            return {"jsonrpc": "2.0", "error": {"code": -32603, "message": f"Internal error: {str(e)}"}}

    async def _handle_cors_preflight(self, request):
        """Handle CORS preflight requests"""
        from aiohttp.web import Response

        return Response(
            status=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    async def read_message(self) -> Optional[Dict[str, Any]]:
        """Read a message from the queue"""
        if self.closed:
            return None

        try:
            # Wait for message with timeout
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logging.error(f"Error reading message: {e}")
            return None

    async def write_message(self, message: Dict[str, Any]) -> None:
        """Write a message to all connected HTTP streaming clients"""
        if self.closed:
            return

        try:
            # Send to all connected clients via their response queues
            for client_id, queue in list(self.response_queues.items()):
                try:
                    await queue.put(message)
                except Exception as e:
                    logging.error(f"Error queuing message for client {client_id}: {e}")
                    # Remove failed connection
                    if client_id in self.response_queues:
                        del self.response_queues[client_id]
                    if client_id in self.client_connections:
                        del self.client_connections[client_id]

        except Exception as e:
            logging.error(f"Error writing message: {e}")

    async def _handle_health_check(self, request):
        """Handle health check requests"""
        from aiohttp.web import json_response

        return json_response(
            {
                "status": "ok",
                "transport": "HTTP Streaming",
                "protocol": "MCP-HTTP-Stream",
                "connections": len(self.client_connections),
            }
        )

    async def close(self) -> None:
        """Close the transport"""
        self.closed = True

        # Close all client connections
        self.client_connections.clear()
        self.response_queues.clear()

        # Close server
        if self.site:
            await self.site.stop()

        if self.runner:
            await self.runner.cleanup()


class MCPServer:
    """MCP Server implementation"""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.transport: Optional[MCPTransport] = None
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Resource] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        self.resource_handlers: Dict[str, Callable] = {}
        self.running = False
        self.initialized = False

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("/tmp/mcp_server.log")],
        )
        self.logger = logging.getLogger(__name__)

    def add_tool(self, name: str, description: str, input_schema: Dict[str, Any], handler: Callable) -> None:
        """Add a tool to the server"""
        tool = Tool(name=name, description=description, inputSchema=input_schema)
        self.tools[name] = tool
        self.tool_handlers[name] = handler
        self.logger.info(f"Added tool: {name}")

    def add_resource(
        self, uri: str, name: str, description: str = "", mime_type: str = "text/plain", handler: Callable = None
    ) -> None:
        """Add a resource to the server"""
        resource = Resource(uri=uri, name=name, description=description, mimeType=mime_type)
        self.resources[uri] = resource
        if handler:
            self.resource_handlers[uri] = handler
        self.logger.info(f"Added resource: {uri}")

    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request"""
        self.logger.info("Handling initialize request")

        # Validate required parameters
        if "protocolVersion" not in params:
            raise MCPServerError(-32602, "Missing protocolVersion parameter")

        # Check protocol version compatibility
        protocol_version = params["protocolVersion"]
        # Accept 2024 and 2025 versions
        if not (protocol_version.startswith("2024-") or protocol_version.startswith("2025-")):
            raise MCPServerError(-32602, f"Unsupported protocol version: {protocol_version}")

        self.initialized = True

        return {
            "protocolVersion": "2025-06-18",
            "capabilities": {
                "tools": {"listChanged": False} if self.tools else {},
                "resources": {"subscribe": False, "listChanged": False} if self.resources else {},
            },
            "serverInfo": {"name": self.name, "version": self.version},
        }

    async def handle_initialized(self, params: Dict[str, Any]) -> None:
        """Handle initialized notification"""
        self.logger.info("Client initialized")

    async def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request"""
        if not self.initialized:
            raise MCPServerError(-32002, "Server not initialized")

        tools_list = [asdict(tool) for tool in self.tools.values()]
        return {"tools": tools_list}

    async def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        if not self.initialized:
            raise MCPServerError(-32002, "Server not initialized")

        if "name" not in params:
            raise MCPServerError(-32602, "Missing tool name")

        tool_name = params["name"]
        arguments = params.get("arguments", {})

        if tool_name not in self.tool_handlers:
            raise MCPServerError(-32601, f"Unknown tool: {tool_name}")

        try:
            handler = self.tool_handlers[tool_name]
            result = await handler(arguments) if asyncio.iscoroutinefunction(handler) else handler(arguments)

            return {"content": [{"type": "text", "text": str(result)}]}
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            raise MCPServerError(-32603, f"Tool execution failed: {str(e)}")

    async def handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request"""
        if not self.initialized:
            raise MCPServerError(-32002, "Server not initialized")

        resources_list = [asdict(resource) for resource in self.resources.values()]
        return {"resources": resources_list}

    async def handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request"""
        if not self.initialized:
            raise MCPServerError(-32002, "Server not initialized")

        if "uri" not in params:
            raise MCPServerError(-32602, "Missing resource URI")

        uri = params["uri"]

        if uri not in self.resources:
            raise MCPServerError(-32601, f"Unknown resource: {uri}")

        try:
            if uri in self.resource_handlers:
                handler = self.resource_handlers[uri]
                content = await handler(uri) if asyncio.iscoroutinefunction(handler) else handler(uri)
            else:
                content = f"Content for resource: {uri}"

            resource = self.resources[uri]
            return {"contents": [{"uri": uri, "mimeType": resource.mimeType, "text": str(content)}]}
        except Exception as e:
            self.logger.error(f"Error reading resource {uri}: {e}")
            raise MCPServerError(-32603, f"Resource read failed: {str(e)}")

    async def handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping request"""
        return {}

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle incoming MCP request"""
        self.logger.info(f"Handling request: {request.method}")

        try:
            # Route request to appropriate handler
            if request.method == "initialize":
                result = await self.handle_initialize(request.params or {})
            elif request.method == "tools/list":
                result = await self.handle_tools_list(request.params or {})
            elif request.method == "tools/call":
                result = await self.handle_tools_call(request.params or {})
            elif request.method == "resources/list":
                result = await self.handle_resources_list(request.params or {})
            elif request.method == "resources/read":
                result = await self.handle_resources_read(request.params or {})
            elif request.method == "ping":
                result = await self.handle_ping(request.params or {})
            else:
                raise MCPServerError(-32601, f"Unknown method: {request.method}")

            return MCPResponse(jsonrpc="2.0", id=request.id, result=result)

        except MCPServerError as e:
            error_dict = {"code": e.code, "message": e.message}
            if e.data is not None:
                error_dict["data"] = e.data

            return MCPResponse(jsonrpc="2.0", id=request.id, error=error_dict)

        except Exception as e:
            self.logger.error(f"Unexpected error handling request: {e}")
            return MCPResponse(
                jsonrpc="2.0", id=request.id, error={"code": -32603, "message": "Internal error", "data": str(e)}
            )

    async def handle_notification(self, notification: MCPNotification) -> None:
        """Handle incoming MCP notification"""
        self.logger.info(f"Handling notification: {notification.method}")

        if notification.method == "initialized":
            await self.handle_initialized(notification.params or {})
        elif notification.method == "notifications/cancelled":
            # Handle request cancellation
            self.logger.info("Request cancelled")
        else:
            self.logger.warning(f"Unknown notification method: {notification.method}")

    async def process_message(self, message_dict: Dict[str, Any]) -> None:
        """Process incoming message"""
        try:
            # Validate JSON-RPC structure
            if message_dict.get("jsonrpc") != "2.0":
                self.logger.error("Invalid JSON-RPC version")
                return

            # Determine message type
            if "id" in message_dict and "method" in message_dict:
                # Request
                request = MCPRequest(
                    jsonrpc="2.0",
                    id=message_dict["id"],
                    method=message_dict["method"],
                    params=message_dict.get("params"),
                )
                response = await self.handle_request(request)
                # Create response dict with proper JSON-RPC structure
                response_dict = {"jsonrpc": response.jsonrpc, "id": response.id}
                if response.error is not None:
                    response_dict["error"] = response.error
                else:
                    response_dict["result"] = response.result

                await self.transport.write_message(response_dict)

            elif "method" in message_dict:
                # Notification
                notification = MCPNotification(
                    jsonrpc="2.0", method=message_dict["method"], params=message_dict.get("params")
                )
                await self.handle_notification(notification)

            else:
                self.logger.error("Invalid message structure")

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    async def run(self, transport: MCPTransport) -> None:
        """Run the MCP server"""
        self.transport = transport
        self.running = True
        self.logger.info(f"Starting MCP server: {self.name}")

        try:
            while self.running:
                message = await transport.read_message()
                if message is None:
                    break

                await self.process_message(message)

        except KeyboardInterrupt:
            self.logger.info("Server interrupted")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            await transport.close()
            self.logger.info("Server stopped")

    def stop(self) -> None:
        """Stop the server"""
        self.running = False


# Example usage and demo tools
def create_demo_server() -> MCPServer:
    """Create a demo MCP server with example tools and resources"""
    server = MCPServer("Demo MCP Server", "1.0.0")

    # Add example tools
    server.add_tool(
        name="echo",
        description="Echo back the input text",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to echo back"}},
            "required": ["text"],
        },
        handler=lambda args: f"Echo: {args.get('text', '')}",
    )

    server.add_tool(
        name="calculate",
        description="Perform basic arithmetic calculations",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Mathematical expression to evaluate"}},
            "required": ["expression"],
        },
        handler=lambda args: str(eval(args.get("expression", "0"))),
    )

    server.add_tool(
        name="reverse",
        description="Reverse a string",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to reverse"}},
            "required": ["text"],
        },
        handler=lambda args: args.get("text", "")[::-1],
    )

    # Add example resources
    server.add_resource(
        uri="demo://greeting",
        name="Greeting Resource",
        description="A simple greeting message",
        mime_type="text/plain",
        handler=lambda uri: "Hello from the MCP server!",
    )

    server.add_resource(
        uri="demo://time",
        name="Current Time",
        description="Current server time",
        mime_type="text/plain",
        handler=lambda uri: f"Current time: {asyncio.get_event_loop().time()}",
    )

    return server


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Server with multiple transport options")
    parser.add_argument(
        "transport",
        choices=["stdio", "http"],
        nargs="?",
        default="stdio",
        help="Transport type to use (http for HTTP streaming)",
    )
    parser.add_argument("--host", default="localhost", help="Host for HTTP transports")
    parser.add_argument("--port", type=int, help="Port for HTTP transports")

    args = parser.parse_args()
    ic(args)

    server = create_demo_server()

    # Choose transport based on arguments
    if args.transport == "stdio":
        transport = StdioTransport()
        try:
            await server.run(transport)
        except KeyboardInterrupt:
            server.stop()
    elif args.transport == "http":
        port = args.port or 8080
        transport = HTTPStreamTransport(args.host, port)

        # Set the server instance in the transport so handlers can access it
        transport.mcp_server = server

        await transport.start_server()

        # For HTTP transport, just keep the server running
        try:
            while True:
                await asyncio.sleep(1)  # Keep the server alive
        except KeyboardInterrupt:
            await transport.close()
    else:
        raise ValueError(f"Unknown transport type: {args.transport}")


if __name__ == "__main__":
    asyncio.run(main())
