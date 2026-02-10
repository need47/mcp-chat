# MCP Server Implementation

A complete Model Context Protocol (MCP) server implementation built from scratch with modern HTTP streaming transport.

## Overview

This implementation provides a fully functional MCP server that follows the MCP 2024-11-05 specification without using any existing MCP packages. It supports multiple transport options including modern HTTP streaming to replace deprecated Server-Sent Events (SSE).

## Files

### Core Implementation

- **`mcp_server.py`** - Main comprehensive MCP server with multiple transport options
- **`modern_http_streaming_server.py`** - Clean, production-ready HTTP streaming server
- **`http_streaming_client.py`** - Modern HTTP streaming client
- **`mcp_client.py`** - Traditional STDIO MCP client
- **`README.py`** - Interactive documentation and file overview

## Transport Options

### ✅ Supported (Modern)
- **HTTP Streaming** - Uses `Transfer-Encoding: chunked` with NDJSON
- **STDIO** - Traditional MCP transport for command-line clients
- **WebSocket** - Real-time bidirectional communication

### ❌ Deprecated
- **Server-Sent Events (SSE)** - Replaced with HTTP streaming

## Features

- 🔧 **Full MCP Protocol Support** - Complete 2024-11-05 specification
- 🚀 **Multiple Transports** - STDIO, HTTP Streaming, WebSocket
- 🛠️ **Built-in Tools** - Echo, calculate, reverse, and more
- 📦 **Resource Management** - Dynamic resource handling
- 🌐 **CORS Support** - Web client compatibility
- 🏥 **Health Checks** - Monitoring endpoints
- 📝 **Comprehensive Logging** - Debug and monitoring
- ⚡ **Production Ready** - Error handling and connection management

## Quick Start

### HTTP Streaming Server (Recommended)

```bash
# Start the modern HTTP streaming server
python modern_http_streaming_server.py

# Test with curl
curl http://localhost:8080/health

# Send MCP message
curl -X POST http://localhost:8080/message \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "Test", "version": "1.0"}}}'
```

### Full Server with Multiple Transports

```bash
# STDIO transport (traditional)
python mcp_server.py --transport stdio

# HTTP streaming transport
python mcp_server.py --transport stream --port 8080

# WebSocket transport
python mcp_server.py --transport websocket --port 8081
```

### Client Testing

```bash
# Test STDIO transport
python mcp_client.py

# Test HTTP streaming
python http_streaming_client.py
```

## HTTP Streaming Protocol

The modern HTTP streaming transport uses:

- **Protocol**: MCP-HTTP-Stream v1.0
- **Content-Type**: `application/x-ndjson` (Newline-delimited JSON)
- **Transfer-Encoding**: `chunked`
- **Endpoints**:
  - `POST /stream` - Bidirectional streaming
  - `POST /message` - Single request/response
  - `GET /health` - Health check

### Response Format

```json
{"type": "connection", "status": "established", "protocol": "MCP-HTTP-Stream"}
{"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "Response"}]}}
{"type": "ping", "timestamp": 1695657600.123}
```

## MCP Protocol Support

### Implemented Methods

- `initialize` - Server initialization and capability exchange
- `tools/list` - List available tools
- `tools/call` - Execute tools
- `resources/list` - List available resources
- `resources/read` - Read resource content

### Built-in Tools

- **echo** - Echo back input text
- **calculate** - Perform basic arithmetic
- **reverse** - Reverse a string
- **uppercase** - Convert text to uppercase

## Development

### Requirements

- Python 3.8+
- `aiohttp` (for HTTP transports)
- `websockets` (for WebSocket transport, optional)

### Architecture

The implementation follows a clean architecture with:

- **Transport Layer** - Abstract base class with concrete implementations
- **Protocol Layer** - MCP message handling and routing
- **Tool Layer** - Extensible tool system
- **Resource Layer** - Dynamic resource management

## Testing

Run the interactive documentation to see all available options:

```bash
python README.py
```

## License

This implementation is provided as-is for educational and development purposes.

---

**Note**: This implementation replaces deprecated Server-Sent Events (SSE) with modern HTTP streaming using chunked transfer encoding and newline-delimited JSON for better performance and standards compliance.
