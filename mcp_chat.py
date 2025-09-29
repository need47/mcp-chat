import argparse
import asyncio
import json
import os
import readline
import sys
import textwrap
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Union

# Import pydantic for exception handling
try:
    from pydantic import ValidationError
    from pydantic_core import ValidationError as PydanticCoreValidationError
except ImportError:
    ValidationError = Exception
    PydanticCoreValidationError = Exception

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()  # Load .env file if it exists
except ImportError:
    # python-dotenv is optional, continue without it
    pass

# Import core LangChain components only
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError as e:
    print("Error: Missing LangChain core. Install with: pip install langchain-core")
    print(f"Specific error: {e}")
    sys.exit(1)

from fastmcp import Client
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def setup_readline():
    """Setup readline with history"""
    # Setup history
    history_file = os.path.expanduser("~/.mcp_chat_history")
    try:
        readline.read_history_file(history_file)
    except (FileNotFoundError, OSError):
        pass

    # Limit history size
    readline.set_history_length(1000)

    # Save history on exit
    import atexit

    def save_history():
        try:
            readline.write_history_file(history_file)
        except OSError:
            pass

    atexit.register(save_history)
    return True


def get_user_input(prompt_text: str = "You") -> str:
    """Get user input with colored prompt and readline history navigation"""
    try:
        # Force flush all output before prompting
        sys.stdout.flush()
        sys.stderr.flush()

        # Display colored prompt and get input
        # Readline history navigation works automatically when available
        console.print(f"\n[bold green]{prompt_text}[/bold green]: ", end="")
        sys.stdout.flush()
        result = input()
        return result

    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt from None


class LLMProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"


@dataclass
class LLMConfig:
    provider: LLMProvider
    model: str
    temperature: float = 0
    max_tokens: int = 2000


class LLMClient:
    """LangChain-based LLM client supporting multiple providers with dynamic imports"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = None
        try:
            try:
                self._initialize_llm()
            except ImportError as e:
                raise ImportError(f"Missing package for {self.config.provider.value}") from e
        except Exception as e:
            console.print(f"❌ Failed to initialize LLM: {e}")
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def _initialize_llm(self):
        """Initialize the appropriate LangChain LLM based on provider"""
        if self.config.provider == LLMProvider.OLLAMA:
            from langchain_community.chat_models import ChatOllama

            self.llm = ChatOllama(
                model=self.config.model,
                temperature=self.config.temperature,
                base_url="http://localhost:11434",
                num_predict=self.config.max_tokens,
            )

        elif self.config.provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        elif self.config.provider == LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            self.llm = ChatAnthropic(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        elif self.config.provider == LLMProvider.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )

        elif self.config.provider == LLMProvider.MISTRAL:
            from langchain_mistralai import ChatMistralAI

            self.llm = ChatMistralAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        elif self.config.provider == LLMProvider.GROQ:
            from langchain_groq import ChatGroq

            self.llm = ChatGroq(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    async def chat(self, messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) -> str:
        """Send chat messages to the LLM and return response"""
        try:
            # Messages are already LangChain message objects
            # Get response from LLM
            if hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke(messages)
            else:
                response = self.llm.invoke(messages)

            return response.content

        except Exception as e:
            return f"❌ Error: {str(e)}"


class MCPChatBot:
    """Enhanced MCP client with LLM chat capabilities"""

    def __init__(self, mcp_server_path: str, llm_config: LLMConfig):
        if mcp_server_path == "https://api.githubcopilot.com/mcp":
            self.mcp_client = Client(mcp_server_path, auth=os.getenv("GITHUB_PAT"))
        else:
            self.mcp_client = Client(mcp_server_path)
        self.llm_config = llm_config
        self.conversation_history: List[Union[SystemMessage, HumanMessage, AIMessage]] = []
        self.available_tools = []
        self.available_resources = []

    async def initialize(self):
        """Initialize MCP client and discover available tools/resources"""
        try:
            await self.mcp_client.ping()
        except Exception as e:
            raise ConnectionError(f"❌ Failed to connect to MCP server: {e}") from e

        # Get available tools
        try:
            self.available_tools = await self.mcp_client.list_tools()
        except Exception as e:
            if "ValidationError" in str(e) or "Invalid JSON" in str(e) or "JSONRPCMessage" in str(e):
                console.print("⚠️  Warning: Could not list tools - endpoint doesn't support MCP protocol")
                console.print(f"   Endpoint returned non-JSON response: {str(e)[:100]}...")
            else:
                console.print(f"⚠️  Warning: Could not list tools: {e}")
            self.available_tools = []
        tool_descriptions = []
        for tool in self.available_tools:
            desc = f"- {tool.name}: {tool.description}"
            if tool.inputSchema and tool.inputSchema.get("properties"):
                params = ", ".join(tool.inputSchema["properties"].keys())
                desc += f" (params: {params})"
            tool_descriptions.append(desc)

        # Get available resources
        try:
            self.available_resources = await self.mcp_client.list_resources()
        except Exception as e:
            if "ValidationError" in str(e) or "Invalid JSON" in str(e) or "JSONRPCMessage" in str(e):
                console.print("⚠️  Warning: Could not list resources - endpoint doesn't support MCP protocol")
                console.print(f"   Endpoint returned non-JSON response: {str(e)[:100]}...")
            else:
                console.print(f"⚠️  Warning: Could not list resources: {e}")
            self.available_resources = []
        resource_descriptions = []
        for resource in self.available_resources:
            resource_descriptions.append(f"- {resource.uri}: {resource.name} - {resource.description}")

        # Create system prompt with MCP context
        system_content = textwrap.dedent(
            f"""
            You are an AI assistant with access to MCP (Model Context Protocol) tools and resources.

            Available MCP Tools:
            {"\n".join(tool_descriptions) if tool_descriptions else "None"}

            Available MCP Resources:
            {"\n".join(resource_descriptions) if resource_descriptions else "None"}

            When a user asks for something that can be accomplished with these tools or resources, you should:
            1. Explain what you can do to help
            2. For tools: Suggest using the appropriate tool with this format: "I can use the `tool_name` tool with parameter_name=value"
            3. For resources: Suggest accessing the resource with this format: "I can access the `resource_uri` resource"
            4. Be specific about the parameters needed

            Examples of good suggestions:
            - "I can use the `search` tool with query=your search term"
            - "Let me use the `summarize` tool with text=your content"
            - "I can access the resource://pubchem_compound_property resource to get available property types"
            - "I'll access the file://data.json resource to get the content"

            After suggesting a tool or resource, I will automatically offer to execute it for you if you confirm.
            Be helpful and concise.

            ONLY suggest tools or resources when the user's question directly relates to what these tools can do.

            DO NOT mention tools or resources if the user is asking general questions unrelated to their functionality. In such cases, imply answer the question without mentioning tools.
        """
        ).strip()

        self.conversation_history = [SystemMessage(system_content)]

        # Setup readline for history management
        setup_readline()

    async def chat_interactive(self):
        """Start interactive chat session"""
        # Show input capabilities and mode
        console.print("✨ Enhanced interactive input enabled:")
        console.print("   • 🔄 arrows: Navigate command history")
        console.print("   • 🎨 Colored prompts and formatted output")
        console.print("   • 🛑 Ctrl+C: Exit chat")
        console.print("   • 📦 Install readline for command history")

        # Show welcome message
        print(
            Panel.fit(
                f"🚀 MCP ChatBot Ready!\n"
                f"🔧 Provider: {self.llm_config.provider.value}\n"
                f"🤖 Model: {self.llm_config.model}\n"
                f"🔨 Tools: {len(self.available_tools)}\n"
                f"📚 Resources: {len(self.available_resources)}\n\n"
                f"🚪 Type 'quit' to exit, ❓ '/help' for commands, 🔧 '/tools' for tools, 📚 '/resources' for resources",
                title="🤖 MCP ChatBot",
            )
        )

        # Start chat loop
        async with LLMClient(self.llm_config) as llm:
            while True:
                try:
                    user_input = get_user_input("You")

                    if user_input.lower() in ["quit", "exit", "q", "x"]:
                        console.print("👋 Goodbye!")
                        break
                    elif user_input == "/tools":
                        await self.show_tools()
                        continue
                    elif user_input == "/resources":
                        await self.show_resources()
                        continue
                    elif user_input in ["/help", "/h", "?"]:
                        self.show_help()
                        continue
                    elif user_input.startswith("/use_tool "):
                        tool_command = user_input[10:].strip()
                        await self.execute_tool_command(tool_command)
                        continue
                    elif user_input.startswith("/get_resource "):
                        resource_uri = user_input[14:].strip()
                        await self.get_resource_content(resource_uri)
                        continue

                    # Add user message to conversation
                    self.conversation_history.append(HumanMessage(user_input))

                    # Get LLM response
                    console.print("\n[bold blue]Assistant[/bold blue] is thinking...")
                    response = await llm.chat(self.conversation_history)

                    # Add assistant response to conversation
                    self.conversation_history.append(AIMessage(response))

                    # Display response with rich formatting
                    console.print(
                        Panel(Markdown(response), title="[bold blue]Assistant[/bold blue]", border_style="blue")
                    )

                    # Check for tool suggestions first, then resources if no tools executed
                    any_executed = await self.handle_tool_suggestions(response)
                    if not any_executed:
                        any_executed = await self.handle_resource_suggestions(response)

                    # If tools/resources were executed, get another LLM response to format the results
                    if any_executed:
                        formatted_response = await llm.chat(self.conversation_history)

                        # Add the formatted response to conversation
                        self.conversation_history.append(AIMessage(formatted_response))

                        # Display the formatted response
                        console.print(
                            Panel(
                                Markdown(formatted_response),
                                title="[bold blue]Assistant[/bold blue]",
                                border_style="blue",
                            )
                        )

                        # Do NOT check for additional tool/resource suggestions after tool execution
                        # The formatted response should just present the results

                except KeyboardInterrupt:
                    console.print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    console.print(f"❌ Error: {e}")

    async def execute_tool_command(self, command: str):
        """Execute MCP tool command"""
        try:
            # Parse command: tool_name param1=value1 param2=value2
            parts = command.split()
            if not parts:
                console.print("❌ Usage: /use_tool <tool_name> [param=value ...]")
                return

            tool_name = parts[0]

            # Parse parameters
            params = {}
            for part in parts[1:]:
                if "=" in part:
                    key, value = part.split("=", 1)
                    # Try to parse as JSON, fall back to string
                    try:
                        params[key] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        params[key] = value

            # Execute tool
            result = await self.mcp_client.call_tool(tool_name, params)

            console.print(
                Panel(
                    f"Tool: {tool_name}\nResult: {result}", title="[green]Tool Execution[/green]", border_style="green"
                )
            )

        except Exception as e:
            console.print(f"❌ Tool execution error: {e}")

    async def get_resource_content(self, uri: str):
        """Get MCP resource content"""
        try:
            result = await self.mcp_client.read_resource(uri)

            console.print(
                Panel(f"Resource: {uri}\nContent: {result}", title="[cyan]Resource Content[/cyan]", border_style="cyan")
            )

        except Exception as e:
            console.print(f"❌ Resource access error: {e}")

    async def handle_tool_suggestions(self, response: str):
        """Parse LLM response for tool suggestions and offer to execute them"""
        import re

        mentioned_tools = []

        # Simple tool detection - check if LLM mentions using any available tool
        for tool in self.available_tools:
            tool_patterns = [
                rf"use the `?{re.escape(tool.name)}`? tool",
                rf"call `?{re.escape(tool.name)}`? with",
                rf"using `?{re.escape(tool.name)}`? with",
            ]

            for pattern in tool_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    console.print(f"🔧 Tool detected: {tool.name}")
                    mentioned_tools.append(tool.name)
                    break

        # If no tools found, return False
        if not mentioned_tools:
            return False

        # Track if any tools were executed
        any_executed = False

        # For each mentioned tool, show it and ask for execution
        for tool_name in mentioned_tools:
            tool_schema = next((tool for tool in self.available_tools if tool.name == tool_name), None)
            if not tool_schema:
                continue

            console.print(
                Panel(
                    f"{tool_schema.description.strip()}",
                    title=f"[cyan]🔧 {tool_name}[/cyan]",
                    border_style="cyan",
                )
            )

            # Show tool schema details
            if tool_schema.inputSchema and tool_schema.inputSchema.get("properties"):
                console.print("Parameters:")
                for param, schema in tool_schema.inputSchema["properties"].items():
                    param_type = schema.get("type", "unknown")
                    param_desc = schema.get("description", "")
                    required = "required" if param in tool_schema.inputSchema.get("required", []) else "optional"
                    console.print(f"      • {param} ({param_type}, {required}): {param_desc}")

            # Extract parameters using simplified logic
            params = self._extract_parameters(response, tool_schema)

            if params is None:
                console.print("   ⚠️  Could not extract required parameters.")
                console.print("       💡 Please provide them manually or rephrase your request.")
                continue

            console.print(f"🎯 Auto-extracted parameters: {params}")

            try:
                confirm = get_user_input("🤔 Execute tool with these parameters? (y/n)").strip()
                if confirm.lower() in ["y", "yes"]:
                    await self.execute_tool(tool_name, params)
                    any_executed = True
                    console.print("   ✅ Tool executed successfully")
                else:
                    console.print("   ⏭️  Skipped.")
            except (KeyboardInterrupt, EOFError):
                console.print("\n🛑 Tool execution cancelled")
                continue
            except Exception as e:
                console.print(f"❌ Error during tool execution: {e}")
                continue

        return any_executed

    async def handle_resource_suggestions(self, response: str):
        """Parse LLM response for resource suggestions and offer to access them"""
        import re

        mentioned_resources = []

        # Simple resource detection - check if LLM mentions accessing any available resource
        for resource in self.available_resources:
            uri_str = str(resource.uri)
            # More specific patterns to avoid false positives - only match actual suggestions
            resource_patterns = [
                # Direct suggestions
                rf"I'll\s+access\s+(?:the\s+)?`?{re.escape(uri_str)}`?(?:\s+resource)?",
                rf"Let\s+me\s+access\s+(?:the\s+)?`?{re.escape(uri_str)}`?(?:\s+resource)?",
                rf"I\s+will\s+access\s+(?:the\s+)?`?{re.escape(uri_str)}`?(?:\s+resource)?",
                # Only match "I can access" when followed by action words or to get/find something
                rf"I\s+can\s+access\s+(?:the\s+)?`?{re.escape(uri_str)}`?(?:\s+resource)?\s+to\s+(?:get|find|retrieve|obtain)",
                # Imperative suggestions
                rf"access\s+(?:the\s+)?`?{re.escape(uri_str)}`?\s+resource\s+to\s+(?:get|find|retrieve|obtain)",
            ]

            for pattern in resource_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    mentioned_resources.append(uri_str)
                    break

        # If no resources found, return False
        if not mentioned_resources:
            return False

        # Track if any resources were accessed
        any_executed = False

        # Handle resource access suggestions
        for resource_uri in mentioned_resources:
            # Find the resource schema (handle both string and AnyUrl types)
            resource_schema = None
            for resource in self.available_resources:
                if str(resource.uri) == str(resource_uri):
                    resource_schema = resource
                    break

            if not resource_schema:
                continue

            console.print(
                Panel(
                    resource_schema.description,
                    title=f"[cyan]📄 {resource_uri}[/cyan]",
                    border_style="cyan",
                )
            )

            try:
                confirm = get_user_input("🤔 Access this resource? (y/n)").strip()
                if confirm.lower() in ["y", "yes"]:
                    await self.access_resource(resource_uri)
                    any_executed = True
                    console.print("   ✅ Resource accessed successfully")
                else:
                    console.print("   ⏭️  Skipped.")
            except (KeyboardInterrupt, EOFError):
                console.print("\n🛑 Resource access cancelled")
                continue
            except Exception as e:
                console.print(f"❌ Error during resource access: {e}")
                continue

        return any_executed

    def _extract_parameters(self, response: str, tool_schema) -> dict:
        """🔍 Extract parameters for a tool from the LLM response with robust parsing"""
        input_schema = tool_schema.inputSchema
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        params = {}

        # 🎯 Enhanced parameter extraction for each required parameter
        for param_name in required:
            param_schema = properties.get(param_name, {})
            param_type = param_schema.get("type", "string")
            param_desc = param_schema.get("description", "")

            # 🔄 Multiple extraction strategies
            extracted_value = self._extract_single_parameter(response, param_name, param_type, param_desc)

            if extracted_value is not None:
                params[param_name] = extracted_value

        # 🎯 Return None if we couldn't find all required parameters
        missing = [p for p in required if p not in params]
        return None if missing else params

    def _extract_single_parameter(self, response: str, param_name: str, param_type: str, param_desc: str):
        """🎯 Extract a single parameter using multiple strategies"""
        import re

        # 📋 Strategy 1: Direct assignment patterns
        direct_patterns = [
            # ✅ Standard formats: param=value, param=[array], param="string"
            rf"{re.escape(param_name)}\s*=\s*(\[[^\]]*\])",  # Arrays
            rf"{re.escape(param_name)}\s*=\s*([\"'])([^\"']*)\1",  # Quoted strings
            rf"{re.escape(param_name)}\s*=\s*([^\s,\]\)]+)",  # Unquoted values
            # 🔗 Colon formats: param: value, param: [array]
            rf"{re.escape(param_name)}\s*:\s*(\[[^\]]*\])",  # Arrays with colon
            rf"{re.escape(param_name)}\s*:\s*([\"'])([^\"']*)\1",  # Quoted with colon
            rf"{re.escape(param_name)}\s*:\s*([^\s,\]\)]+)",  # Unquoted with colon
        ]

        for pattern in direct_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:  # Quoted string pattern
                    raw_value = match.group(2)
                else:
                    raw_value = match.group(1)

                parsed_value = self._parse_parameter_value(raw_value, param_type)
                if parsed_value is not None:
                    return parsed_value

        # 💬 Strategy 2: Natural language patterns
        natural_patterns = [
            # 📝 "with param_name [values]" or "with param_name value"
            rf"with\s+{re.escape(param_name)}\s+(\[[^\]]*\])",
            rf"with\s+{re.escape(param_name)}\s+([\"'])([^\"']*)\1",
            rf"with\s+{re.escape(param_name)}\s+([^\s,\.\!]+)",
            # 🔧 "using param_name as [values]" or "using param_name as value"
            rf"using\s+{re.escape(param_name)}\s+(?:as\s+)?(\[[^\]]*\])",
            rf"using\s+{re.escape(param_name)}\s+(?:as\s+)?([\"'])([^\"']*)\1",
            rf"using\s+{re.escape(param_name)}\s+(?:as\s+)?([^\s,\.\!]+)",
            # 📊 "param_name of [values]" or "param_name of value"
            rf"{re.escape(param_name)}\s+of\s+(\[[^\]]*\])",
            rf"{re.escape(param_name)}\s+of\s+([\"'])([^\"']*)\1",
            rf"{re.escape(param_name)}\s+of\s+([^\s,\.\!]+)",
        ]

        for pattern in natural_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:  # Quoted string pattern
                    raw_value = match.group(2)
                else:
                    raw_value = match.group(1)

                parsed_value = self._parse_parameter_value(raw_value, param_type)
                if parsed_value is not None:
                    return parsed_value

        # 🧠 Strategy 3: Context-based extraction using parameter description
        if param_desc:
            context_value = self._extract_from_context(response, param_name, param_type, param_desc)
            if context_value is not None:
                return context_value

        # 🔍 Strategy 4: Fuzzy matching for common parameter names
        fuzzy_value = self._extract_fuzzy_match(response, param_name, param_type)
        if fuzzy_value is not None:
            return fuzzy_value

        return None

    def _parse_parameter_value(self, raw_value: str, param_type: str):
        """🔧 Parse a raw parameter value based on its expected type"""
        import json

        if not raw_value or raw_value.strip() == "":
            return None

        raw_value = raw_value.strip()

        try:
            # 📋 Handle array types
            if param_type == "array" or (raw_value.startswith("[") and raw_value.endswith("]")):
                if raw_value.startswith("[") and raw_value.endswith("]"):
                    # 🎯 Try JSON parsing first
                    try:
                        # Handle both single and double quotes
                        clean_value = raw_value.replace("'", '"')
                        return json.loads(clean_value)
                    except json.JSONDecodeError:
                        # 🔧 Advanced array parsing for complex cases
                        return self._parse_complex_array(raw_value)
                else:
                    # 🔄 Single value that should be an array
                    return [raw_value.strip("\"'")]

            # 📦 Handle object types
            elif param_type == "object":
                if raw_value.startswith("{") and raw_value.endswith("}"):
                    try:
                        clean_value = raw_value.replace("'", '"')
                        return json.loads(clean_value)
                    except json.JSONDecodeError:
                        return raw_value  # Return as string if can't parse

            # 🔢 Handle numeric types
            elif param_type in ["integer", "number"]:
                # Extract numbers from text
                import re

                number_match = re.search(r"-?\d+\.?\d*", raw_value)
                if number_match:
                    num_str = number_match.group()
                    return int(num_str) if param_type == "integer" and "." not in num_str else float(num_str)

            # ✅ Handle boolean types
            elif param_type == "boolean":
                lower_val = raw_value.lower()
                if lower_val in ["true", "yes", "1", "on", "enabled"]:
                    return True
                elif lower_val in ["false", "no", "0", "off", "disabled"]:
                    return False

            # 📄 Handle string types (default)
            else:
                # Remove surrounding quotes
                if (raw_value.startswith('"') and raw_value.endswith('"')) or (
                    raw_value.startswith("'") and raw_value.endswith("'")
                ):
                    return raw_value[1:-1]
                return raw_value

        except (ValueError, TypeError):
            pass

        # 🔄 Fallback: return as string
        return raw_value.strip("\"'")

    def _parse_complex_array(self, array_str: str) -> list:
        """🔧 Parse complex array strings that might contain nested structures"""

        # 🔧 Remove outer brackets
        content = array_str[1:-1].strip()

        if not content:
            return []

        # 📝 Handle simple comma-separated values
        if not any(char in content for char in ['"', "'", "[", "]", "{", "}"]):
            return [item.strip() for item in content.split(",") if item.strip()]

        # 🎯 Advanced parsing for nested structures
        items = []
        current_item = ""
        bracket_depth = 0
        quote_char = None

        for char in content:
            if quote_char:
                current_item += char
                if char == quote_char and current_item[-2:] != f"\\{quote_char}":
                    quote_char = None
            elif char in ['"', "'"]:
                quote_char = char
                current_item += char
            elif char in ["[", "{"]:
                bracket_depth += 1
                current_item += char
            elif char in ["]", "}"]:
                bracket_depth -= 1
                current_item += char
            elif char == "," and bracket_depth == 0:
                if current_item.strip():
                    items.append(current_item.strip().strip("\"'"))
                current_item = ""
            else:
                current_item += char

        # Add the last item
        if current_item.strip():
            items.append(current_item.strip().strip("\"'"))

        return items

    def _extract_from_context(self, response: str, param_name: str, param_type: str, param_desc: str):
        """Extract parameter value using context from parameter description"""
        import re

        # Look for keywords from parameter description in the response
        desc_words = re.findall(r"\w+", param_desc.lower())

        for word in desc_words:
            if len(word) > 3:  # Skip short words
                # Look for values near description keywords
                pattern = rf"{re.escape(word)}\s*[:\-]?\s*([^\s,\.\!]+)"
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return self._parse_parameter_value(match.group(1), param_type)

        return None

    def _extract_fuzzy_match(self, response: str, param_name: str, param_type: str):
        """Extract using fuzzy matching for common parameter name patterns"""
        import re

        # Common parameter aliases
        fuzzy_mappings = {
            "query": ["search", "term", "keyword", "q"],
            "text": ["content", "message", "body"],
            "id": ["identifier", "cid", "uid"],
            "ids": ["identifiers", "cids", "uids", "list"],
            "properties": ["props", "attributes", "fields"],
            "format": ["type", "output"],
            "limit": ["max", "maximum", "count"],
        }

        # Check if parameter name has known aliases
        aliases = fuzzy_mappings.get(param_name.lower(), [])
        all_names = [param_name] + aliases

        for name in all_names:
            # Try direct value extraction with fuzzy name
            pattern = rf"{re.escape(name)}\s*[=:]\s*([^\s,\]\)]+)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return self._parse_parameter_value(match.group(1), param_type)

        return None

    async def execute_tool(self, tool_name: str, params: dict):
        """Execute a tool silently and add results to conversation history for LLM formatting"""
        try:
            # Execute tool completely silently
            result = await self.mcp_client.call_tool(tool_name, params)

            # Add the tool result to conversation history so the LLM can format it
            if hasattr(result, "content") and result.content:
                content = result.content[0].text if result.content else str(result)
                tool_result_msg = f"The tool '{tool_name}' was executed with parameters {params} and returned the following data: {content}. Please format this data as requested by the user."
            else:
                tool_result_msg = f"The tool '{tool_name}' was executed with parameters {params} and returned: {result}. Please format this data as requested by the user."

            self.conversation_history.append(HumanMessage(tool_result_msg))

        except Exception as e:
            # Add error to conversation history silently (no console output)
            error_msg = f"Tool '{tool_name}' execution failed with error: {e}"
            self.conversation_history.append(HumanMessage(error_msg))

    async def access_resource(self, resource_uri: str):
        """Access a resource silently and add content to conversation history for LLM formatting"""
        try:
            # Ensure URI is properly formatted as string
            uri_str = str(resource_uri)

            # Access resource completely silently
            result = await self.mcp_client.read_resource(uri_str)

            # Extract content more safely
            content = "No content available"
            try:
                if hasattr(result, "contents") and result.contents and len(result.contents) > 0:
                    first_content = result.contents[0]
                    if hasattr(first_content, "text") and first_content.text:
                        content = str(first_content.text)
                    elif hasattr(first_content, "data") and first_content.data:
                        content = str(first_content.data)
                    else:
                        content = str(first_content)
                elif hasattr(result, "content") and result.content:
                    # Try alternative content structure
                    content = str(result.content)
                else:
                    content = str(result)
            except Exception as content_error:
                content = f"Error extracting content: {content_error}"

            resource_result_msg = f"The resource '{uri_str}' was accessed and returned the following data: {content}. Please format this data as requested by the user."
            self.conversation_history.append(HumanMessage(resource_result_msg))

        except Exception as e:
            # Add error to conversation history silently (no console output)
            error_msg = f"Resource '{resource_uri}' access failed with error: {e}"
            self.conversation_history.append(HumanMessage(error_msg))

    async def show_tools(self):
        """🔧 Display available MCP tools"""
        if not self.available_tools:
            console.print("❌ No tools available")
            return

        tool_info = []
        for tool in self.available_tools:
            info = f"🛠️ **{tool.name}**: {tool.description}"
            if tool.inputSchema and tool.inputSchema.get("properties"):
                params = []
                for param, schema in tool.inputSchema["properties"].items():
                    param_type = schema.get("type", "unknown")
                    param_desc = schema.get("description", "")
                    params.append(f"    📋 `{param}` ({param_type}): {param_desc}")
                if params:
                    info += f"\n{'\n'.join(params)}"
            tool_info.append(info)

        tools_text = "\n\n".join(tool_info)
        console.print(
            Panel(Markdown(tools_text), title="🔧 [yellow]Available MCP Tools[/yellow]", border_style="yellow")
        )

    async def show_resources(self):
        """📚 Display available MCP resources"""
        if not self.available_resources:
            console.print("❌ No resources available")
            return

        resource_info = []
        for resource in self.available_resources:
            info = f"📄 **{resource.uri}**\n"
            info += f"    📝 Name: {resource.name}\n"
            info += f"    📋 Description: {resource.description}\n"
            info += f"    🏷️  MIME Type: {resource.mimeType}"

            if hasattr(resource, "_meta") and resource._meta:
                fastmcp_meta = resource._meta.get("_fastmcp", {})
                tags = fastmcp_meta.get("tags", [])
                if tags:
                    info += f"\n    🏷️  Tags: {', '.join(tags)}"

            resource_info.append(info)

        resources_text = "\n\n".join(resource_info)
        console.print(
            Panel(Markdown(resources_text), title="📚 [cyan]Available MCP Resources[/cyan]", border_style="cyan")
        )

    def show_help(self):
        """❓ Display help information for chat commands"""
        help_text = textwrap.dedent(
            """
            **🎯 Available Commands:**

            - 💬 **Chat**: Type any message to chat with the AI assistant
            - 🚪 **`quit`**, **`exit`**, **`q`**, **`x`**: Exit the chat
            - 🔧 **`/tools`**: List all available MCP tools
            - 📚 **`/resources`**: List all available MCP resources
            - 🔨 **`/use_tool <tool_name> [param=value ...]`**: Execute an MCP tool manually
            - 📄 **`/get_resource <uri>`**: Get content from an MCP resource
            - ❓ **`/help`**, **`/h`**, **`?`**: Show this help message

            **🤖 Smart Tool Integration:**
            - 🧠 The AI assistant can suggest and execute MCP tools automatically
            - 🤔 When the AI suggests using a tool, you'll be asked to confirm
            - ✅ You can approve all suggestions (y), decline (n), or select specific ones (1,3)

            **📝 Examples:**
            ```
            💬 "Search for python tutorials" → 🤖 AI suggests search tool → ✅ Confirm → 🚀 Execute
            🔨 /use_tool search query="python programming"
            📄 /get_resource file://example.txt
            🔧 /tools
            ```

            **💡 Tips:**
            - 🔄 Use UP/DOWN arrow keys to navigate command history
            - 🎨 Rich-based input with colored prompts and clean formatting
            - 💾 Command history is saved between sessions
            - 🛑 Use **Ctrl+C** to exit the chat
        """
        ).strip()

        console.print(Panel(Markdown(help_text), title="❓ [yellow]Help[/yellow]", border_style="yellow"))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="FastMCP Chat Client - Interactive AI Assistant with MCP Tool Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Chat with Ollama (local)
              python fastmcp-client.py server.py --provider ollama --model llama3.2

              # Chat with OpenAI (uses API key from .env file)
              python fastmcp-client.py server.py --provider openai --model gpt-4o-mini

              # Chat with Anthropic (uses API key from .env file)
              python fastmcp-client.py server.py --provider anthropic --model claude-3-5-sonnet-20241022

              # Chat with Gemini (shortcut - sets provider=google model=gemini-2.5-flash)
              python fastmcp-client.py server.py --gemini

              # Chat with Claude (shortcut, uses default model)
              python fastmcp-client.py server.py --claude

              # Chat with OpenAI (shortcut - sets provider=openai model=gpt-4o-mini)
              python fastmcp-client.py server.py --openai

              # Chat with Groq (shortcut - sets provider=groq model=llama-3.3-70b-versatile)
              python fastmcp-client.py server.py --groq

              # Chat with Mistral (shortcut - sets provider=mistral model=mistral-large-latest)
              python fastmcp-client.py server.py --mistral
        """
        ).strip(),
    )

    parser.add_argument("server", help="MCP server URL or path")

    # LLM configuration arguments
    parser.add_argument(
        "--provider",
        "-p",
        choices=[p.value for p in LLMProvider],
        default="ollama",
        help="LLM provider (default: ollama)",
    )

    parser.add_argument(
        "--model",
        "-m",
        help="Model name (uses provider default if not specified: ollama=llama3.2, openai=gpt-4o-mini, anthropic=claude-sonnet-4-20250514, google=gemini-2.5-flash)",
    )

    parser.add_argument("--temperature", "-t", type=float, default=0, help="Temperature for LLM (0.0-1.0, default: 0)")

    parser.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens for LLM response (default: 2000)")

    # Custom action class for provider+model shortcuts
    class ProviderModelAction(argparse.Action):
        def __init__(self, option_strings, dest, provider, model=None, **kwargs):
            super().__init__(option_strings, dest, nargs=0, **kwargs)
            self.provider = provider
            self.model = model

        def __call__(self, parser, namespace, values, option_string=None):
            namespace.provider = self.provider
            if self.model:
                # Only set model if it hasn't been explicitly set by --model
                if not hasattr(namespace, "model") or namespace.model is None:
                    namespace.model = self.model

    parser.add_argument(
        "--google",
        "--gemini",
        action=partial(ProviderModelAction, provider="google", model="gemini-2.5-flash"),
        help="Use google gemini-2.5-flash",
    )
    parser.add_argument(
        "--openai",
        action=partial(ProviderModelAction, provider="openai", model="gpt-4o-mini"),
        help="Use openai gpt-4o-mini",
    )
    parser.add_argument(
        "--anthropic",
        "--claude",
        action=partial(ProviderModelAction, provider="anthropic", model="claude-sonnet-4-20250514"),
        help="Use anthropic claude-sonnet-4-20250514",
    )
    parser.add_argument(
        "--ollama",
        action=partial(ProviderModelAction, provider="ollama", model="llama3.2"),
        help="Use ollama llama3.2",
    )
    parser.add_argument(
        "--groq",
        action=partial(ProviderModelAction, provider="groq", model="llama-3.3-70b-versatile"),
        help="Use groq llama-3.3-70b-versatile",
    )
    parser.add_argument(
        "--mistral",
        action=partial(ProviderModelAction, provider="mistral", model="mistral-large-latest"),
        help="Use mistral mistral-large-latest",
    )

    return parser.parse_args()


def create_llm_config(args) -> LLMConfig:
    """Create LLM configuration from parsed arguments with .env fallback for API keys"""
    try:
        provider = LLMProvider(args.provider)
    except ValueError:
        console.print(f"❌ Invalid provider: {args.provider}")
        console.print("Available providers: " + ", ".join([p.value for p in LLMProvider]))
        sys.exit(1)

    # Default models for each provider
    default_models = {
        LLMProvider.OLLAMA: "llama3.2",
        LLMProvider.OPENAI: "gpt-4o-mini",
        LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
        LLMProvider.GOOGLE: "gemini-2.5-flash",
        LLMProvider.MISTRAL: "mistral-large-latest",
        LLMProvider.GROQ: "llama-3.3-70b-versatile",
        LLMProvider.HUGGINGFACE: "microsoft/DialoGPT-medium",
    }

    model = args.model or default_models[provider]

    # Show if using default model
    if not args.model:
        console.print(f"ℹ️  Using default model for {provider.value}: {model}")

    return LLMConfig(
        provider=provider,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


async def main():
    args = parse_args()

    # Ensure we're running in an interactive terminal
    if not sys.stdin.isatty():
        console.print("❌ Non-interactive mode detected.")
        console.print("   This application requires an interactive terminal.")
        console.print("   Please run in an interactive terminal.")
        sys.exit(1)

    # Create LLM configuration
    llm_config = create_llm_config(args)

    console.print("🚀 Initializing MCP ChatBot...")
    console.print(f"   🔧 Provider: {llm_config.provider.value}")
    console.print(f"   🤖 Model: {llm_config.model}\n")

    chatbot = MCPChatBot(args.server, llm_config)

    async with chatbot.mcp_client:
        console.print("🔌 Connecting to MCP server...")
        try:
            await chatbot.initialize()
            console.print("✅ MCP server connected!\n")
            await chatbot.chat_interactive()
        except ConnectionError as e:
            console.print(str(e))
            sys.exit(1)
        except Exception as e:
            console.print(f"❌ Unexpected error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
