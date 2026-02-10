# MCP Server Implementation Guide

## What is MCP (Model Context Protocol)?

MCP is a protocol that allows AI assistants to securely connect to data sources and tools. It provides a standardized way to expose resources and tools to AI models through a JSON-RPC 2.0 interface.

## Core Requirements for MCP Server Implementation

### 1. JSON-RPC 2.0 Protocol
MCP servers MUST implement JSON-RPC 2.0 as the communication protocol:

```json
{
  "jsonrpc": "2.0",
  "method": "method_name",
  "params": { ... },
  "id": 1
}
```

### 2. Required Methods

#### Core Protocol Methods:
- **`initialize`** - Handshake and capability negotiation
- **`ping`** - Health check and keep-alive
- **`resources/list`** - List available resources
- **`tools/list`** - List available tools

#### Resource Methods:
- **`resources/read`** - Read resource content
- **`resources/subscribe`** - Subscribe to resource changes (optional)
- **`resources/unsubscribe`** - Unsubscribe from resource changes (optional)

#### Tool Methods:
- **`tools/call`** - Execute a tool with parameters

#### Optional Methods:
- **`notifications/list`** - List available notification types
- **`logging/setLevel`** - Configure logging levels
- **`prompts/list`** - List available prompt templates
- **`prompts/get`** - Get specific prompt template
- **`completion/complete`** - Text completion capabilities

#### Notification Methods (Server-to-Client):
- **`notifications/tools/list_changed`** - Tools list has changed
- **`notifications/resources/list_changed`** - Resources list has changed
- **`notifications/roots/list_changed`** - Root resources have changed

### 3. Protocol Flow

#### Initialization:
```json
Request:
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "roots": { "listChanged": true },
      "sampling": {}
    },
    "clientInfo": {
      "name": "example-client",
      "version": "1.0.0"
    }
  },
  "id": 1
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "logging": {},
      "tools": { "listChanged": true },
      "resources": { "subscribe": true, "listChanged": true }
    },
    "serverInfo": {
      "name": "example-server",
      "version": "1.0.0"
    }
  },
  "id": 1
}
```

#### Ping:
```json
Request:
{
  "jsonrpc": "2.0",
  "method": "ping",
  "id": 2
}

Response:
{
  "jsonrpc": "2.0",
  "result": {},
  "id": 2
}
```

#### List Resources:
```json
Request:
{
  "jsonrpc": "2.0",
  "method": "resources/list",
  "id": 3
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "resources": [
      {
        "uri": "file:///path/to/resource",
        "name": "Resource Name",
        "description": "Description of the resource",
        "mimeType": "text/plain"
      }
    ]
  },
  "id": 3
}
```

#### List Tools:
```json
Request:
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 4
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "tool_name",
        "description": "Tool description",
        "inputSchema": {
          "type": "object",
          "properties": {
            "param1": { "type": "string", "description": "Parameter description" }
          },
          "required": ["param1"]
        }
      }
    ]
  },
  "id": 4
}
```

#### Read Resource:
```json
Request:
{
  "jsonrpc": "2.0",
  "method": "resources/read",
  "params": {
    "uri": "file:///path/to/resource"
  },
  "id": 5
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "contents": [
      {
        "uri": "file:///path/to/resource",
        "mimeType": "text/plain",
        "text": "Resource content here"
      }
    ]
  },
  "id": 5
}
```

#### Call Tool:
```json
Request:
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": {
      "param1": "value1"
    }
  },
  "id": 6
}

Response:
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Tool execution result"
      }
    ]
  },
  "id": 6
}
```

## Implementation Examples

### Python Implementation (using FastMCP):
```python
from fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("My MCP Server")

# Add a tool
@mcp.tool()
def search_data(query: str) -> str:
    """Search for data based on query"""
    # Implement your search logic
    return f"Search results for: {query}"

# Add a resource
@mcp.resource("data://example")
def get_example_data() -> str:
    """Get example data"""
    return "This is example data"

# Run server
if __name__ == "__main__":
    mcp.run()
```

## Key Implementation Points

### 1. Transport Layer
- **HTTP/HTTPS**: For web-based servers
- **WebSocket**: For real-time communication
- **stdio**: For process-based communication
- **SSE (Server-Sent Events)**: For streaming

### 2. Error Handling
Always return proper JSON-RPC error responses:

#### Standard JSON-RPC Error Codes:
- **-32700**: Parse error (Invalid JSON)
- **-32600**: Invalid Request
- **-32601**: Method not found
- **-32602**: Invalid params
- **-32603**: Internal error

#### MCP-Specific Error Codes:
- **-32000**: Server error (generic MCP error)
- **-32001**: Resource not found
- **-32002**: Tool execution failed
- **-32003**: Permission denied
- **-32004**: Rate limit exceeded
- **-32005**: Resource unavailable

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32001,
    "message": "Resource not found",
    "data": {
      "uri": "file:///nonexistent/file.txt",
      "details": "The requested resource does not exist"
    }
  },
  "id": 1
}
```

### 3. Security Considerations

#### Authentication & Authorization:
```json
// Bearer token authentication
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "client",
      "version": "1.0.0"
    }
  },
  "id": 1,
  "meta": {
    "authorization": "Bearer your-token-here"
  }
}
```

#### Security Checklist:
- ✅ Validate all inputs against schemas
- ✅ Implement proper authentication (Bearer tokens, API keys)
- ✅ Sanitize resource URIs to prevent path traversal
- ✅ Rate limiting for tool calls and resource access
- ✅ Input validation and sanitization
- ✅ HTTPS/TLS encryption for network transport
- ✅ Role-based access control for tools/resources
- ✅ Audit logging for security events

### 4. Resource Types

#### Standard URI Schemes:
- **Files**: `file://` URIs
- **Database entries**: `db://` URIs  
- **Web resources**: `http://` or `https://` URIs
- **Custom schemes**: Define your own URI schemes

#### Enhanced Resource Response:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "contents": [
      {
        "uri": "file:///data/report.json",
        "mimeType": "application/json",
        "text": "{\"data\": \"content\"}",
        "size": 1024,
        "lastModified": "2024-11-05T10:30:00Z",
        "annotations": {
          "created": "2024-11-01T09:00:00Z",
          "author": "system",
          "tags": ["report", "data"]
        }
      }
    ]
  },
  "id": 5
}
```

### 5. Tool Design

#### Design Principles:
- Keep tools focused and atomic
- Provide clear parameter schemas
- Handle errors gracefully
- Return structured results
- Support progress indicators for long-running operations

#### Enhanced Tool Response Types:
```json
// Text response
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Processing completed successfully"
      }
    ]
  },
  "id": 6
}

// JSON/Structured response
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"status\": \"success\", \"results\": [...]}"
      }
    ]
  },
  "id": 6
}

// Binary/Image response
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "resource",
        "resource": {
          "uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
          "mimeType": "image/png"
        }
      }
    ]
  },
  "id": 6
}
```

## Testing Your MCP Server

### Basic Connectivity Test:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "ping", "id": 1}' \
  http://localhost:3000/mcp
```

### Tool Testing:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search",
      "arguments": {"query": "test"}
    },
    "id": 1
  }' \
  http://localhost:3000/mcp
```

## Protocol Version Management

### Version Negotiation:
```json
// Client requests specific version
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "clientInfo": {
      "name": "client",
      "version": "1.0.0"
    }
  },
  "id": 1
}

// Server responds with supported version
{
  "jsonrpc": "2.0",
  "result": {
    "protocolVersion": "2024-11-05",
    "serverInfo": {
      "name": "server",
      "version": "1.0.0"
    }
  },
  "id": 1
}
```

### Backward Compatibility Strategy:
- Support multiple protocol versions
- Graceful degradation of features
- Clear version deprecation timeline
- Migration guides for breaking changes

## Notification System

### Server-to-Client Notifications:
```json
// Tools list changed
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed",
  "params": {}
}

// Resource list changed
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/list_changed",
  "params": {}
}

// Custom notification
{
  "jsonrpc": "2.0",
  "method": "notifications/custom/data_updated",
  "params": {
    "resource": "data://user_data",
    "timestamp": "2024-11-05T10:30:00Z"
  }
}
```

## Connection Management

### Connection Lifecycle:
1. **Establish Transport** (HTTP, WebSocket, stdio)
2. **Initialize Protocol** (version negotiation)
3. **Active Communication** (requests/responses)
4. **Heartbeat/Ping** (keep-alive)
5. **Graceful Shutdown** (cleanup resources)

### Reconnection Strategy:
```python
import asyncio
import logging

class MCPClient:
    def __init__(self, max_retries=5, backoff_factor=2):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
    async def connect_with_retry(self):
        for attempt in range(self.max_retries):
            try:
                await self.connect()
                return
            except ConnectionError as e:
                wait_time = self.backoff_factor ** attempt
                logging.warning(f"Connection failed (attempt {attempt + 1}), retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
        raise ConnectionError("Max retries exceeded")
```

## Performance Optimization

### Pagination for Large Lists:
```json
// Request with pagination
{
  "jsonrpc": "2.0",
  "method": "resources/list",
  "params": {
    "cursor": "eyJvZmZzZXQiOjEwMH0=",
    "limit": 50
  },
  "id": 1
}

// Response with pagination info
{
  "jsonrpc": "2.0",
  "result": {
    "resources": [...],
    "nextCursor": "eyJvZmZzZXQiOjE1MH0=",
    "hasMore": true,
    "total": 1000
  },
  "id": 1
}
```

### Streaming for Large Responses:
```json
// Start streaming response
{
  "jsonrpc": "2.0",
  "result": {
    "isStream": true,
    "streamId": "stream_123",
    "totalSize": 1048576
  },
  "id": 1
}

// Stream chunks
{
  "jsonrpc": "2.0",
  "method": "notifications/stream/chunk",
  "params": {
    "streamId": "stream_123",
    "chunk": "base64-encoded-data",
    "offset": 0,
    "isLast": false
  }
}
```

## Configuration Management

### Server Configuration Example:
```yaml
# mcp-server.yaml
server:
  name: "PubChem MCP Server"
  version: "1.0.0"
  host: "0.0.0.0"
  port: 3000
  
protocol:
  version: "2024-11-05"
  
security:
  auth_required: true
  rate_limit:
    requests_per_minute: 1000
    burst_size: 100
    
logging:
  level: "info"
  format: "json"
  
resources:
  max_file_size: 10485760  # 10MB
  allowed_schemes: ["file", "http", "https", "data"]
  
tools:
  timeout: 30  # seconds
  max_concurrent: 10
```

### Environment Variable Support:
```python
import os
from dataclasses import dataclass

@dataclass
class MCPConfig:
    server_name: str = os.getenv("MCP_SERVER_NAME", "MCP Server")
    port: int = int(os.getenv("MCP_PORT", "3000"))
    auth_token: str = os.getenv("MCP_AUTH_TOKEN", "")
    log_level: str = os.getenv("MCP_LOG_LEVEL", "INFO")
```

## Deployment Patterns

### Docker Deployment:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

CMD ["python", "server.py"]
```

### Process Management (systemd):
```ini
[Unit]
Description=MCP Server
After=network.target

[Service]
Type=simple
User=mcp
WorkingDirectory=/opt/mcp-server
ExecStart=/opt/mcp-server/venv/bin/python server.py
Restart=always
RestartSec=10
Environment=MCP_PORT=3000
Environment=MCP_LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
```

## Best Practices

### Performance:
- Use connection pooling for database resources
- Implement caching for frequently accessed resources
- Use async/await for non-blocking operations
- Monitor memory usage for large tool responses

### Security:
- Never expose internal file paths in error messages
- Implement request size limits
- Use secure random tokens for authentication
- Regular security audits and dependency updates

### Monitoring:
```python
import time
import logging
from functools import wraps

def monitor_tool_execution(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.info(f"Tool {func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"Tool {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper
```

### Testing Framework:
```python
import pytest
import asyncio
from mcp_client import MCPClient

@pytest.fixture
async def mcp_client():
    client = MCPClient("http://localhost:3000")
    await client.connect()
    yield client
    await client.disconnect()

@pytest.mark.asyncio
async def test_ping(mcp_client):
    response = await mcp_client.ping()
    assert response["jsonrpc"] == "2.0"
    assert "result" in response

@pytest.mark.asyncio
async def test_tool_execution(mcp_client):
    response = await mcp_client.call_tool("search", {"query": "test"})
    assert response["jsonrpc"] == "2.0"
    assert "result" in response
    assert "content" in response["result"]
```

## Troubleshooting Guide

### Common Issues:

1. **Connection Refused**
   - Check server is running: `curl http://localhost:3000/health`
   - Verify port is not blocked by firewall
   - Check server logs for startup errors

2. **Method Not Found**
   - Ensure method names match exactly (case-sensitive)
   - Check protocol version compatibility
   - Verify server implements required methods

3. **Authentication Failures**
   - Validate token format and expiry
   - Check authorization headers
   - Verify server auth configuration

4. **Tool Execution Timeouts**
   - Increase timeout configuration
   - Check for blocking operations
   - Implement progress reporting for long tasks

5. **Resource Access Errors**
   - Validate URI format and scheme
   - Check file permissions
   - Verify resource exists and is accessible

### Debug Mode Configuration:
```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log all JSON-RPC messages
class DebugMCPServer(MCPServer):
    async def handle_request(self, request):
        logging.debug(f"Incoming request: {request}")
        response = await super().handle_request(request)
        logging.debug(f"Outgoing response: {response}")
        return response
```

## Common Pitfalls to Avoid

1. **Not following JSON-RPC 2.0 spec** - Must include `jsonrpc: "2.0"`
2. **Missing required methods** - ping, initialize, resources/list, tools/list
3. **Improper error handling** - Return JSON-RPC errors, not HTTP errors
4. **Schema validation** - Validate tool parameters against inputSchema
5. **URI handling** - Properly validate and sanitize resource URIs
6. **Method naming inconsistency** - Use `resources/list` not `list_resources`
7. **Blocking operations** - Use async/await for I/O operations
8. **Memory leaks** - Properly cleanup resources and connections
9. **Security vulnerabilities** - Validate all inputs, use HTTPS
10. **Poor error messages** - Provide helpful, actionable error information
