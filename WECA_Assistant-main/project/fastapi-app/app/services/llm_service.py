"""LLM service for communicating with Ollama."""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict
import json
import httpx

from app.config import settings
from app.api.external_apis.tool import LLMResult2, Tool, ToolCall
logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """Result of an LLM completion."""

    text: str
    success: bool
    error: Optional[str] = None
    model: Optional[str] = None


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", or "assistant"
    content: str


class LLMService:
    """Service for generating text responses using Ollama."""

    def __init__(
        self,
        host: str = settings.ollama_host,
        port: int = settings.ollama_port,
        model: str = settings.ollama_model,
    ):
        self.host = host
        self.port = port
        self.model = model
        self.base_url = f"http://{host}:{port}"
        self.timeout = httpx.Timeout(60.0, connect=10.0)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResult:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user's input text
            system_prompt: Optional system prompt to set context
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            LLMResult with the generated text
        """
        system_prompt = system_prompt or settings.llm_system_prompt
        max_tokens = max_tokens or settings.llm_max_tokens
        temperature = temperature if temperature is not None else settings.llm_temperature

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Use chat endpoint for better conversation handling
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                    },
                )

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Ollama error: {response.status_code} - {error_text}")
                    return LLMResult(
                        text="",
                        success=False,
                        error=f"LLM request failed: {error_text}",
                    )

                data = response.json()
                message = data.get("message", {})
                generated_text = message.get("content", "").strip()

                if not generated_text:
                    return LLMResult(
                        text="",
                        success=False,
                        error="LLM returned empty response",
                    )

                logger.info(f"LLM generated {len(generated_text)} characters")
                return LLMResult(
                    text=generated_text,
                    success=True,
                    model=self.model,
                )

        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return LLMResult(
                text="",
                success=False,
                error=f"Cannot connect to LLM server at {self.base_url}",
            )
        except httpx.TimeoutException as e:
            logger.error(f"Ollama timeout: {e}")
            return LLMResult(
                text="",
                success=False,
                error="LLM request timed out",
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return LLMResult(
                text="",
                success=False,
                error=str(e),
            )
            
            
    async def generateforapi(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Tool]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> LLMResult2:
        """
        Generate a response from the LLM with optional tool calling and conversation history.
        
        Args:
            prompt: Current user input
            system_prompt: System prompt for LLM behavior
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            tools: Available tools for function calling
            conversation_history: Previous messages [{"role": "user/assistant", "content": "..."}]
        """
        system_prompt = system_prompt or "You are a helpful assistant."
        max_tokens = max_tokens or 500
        temperature = temperature if temperature is not None else 0.7

        # Build the enhanced system prompt with tool definitions
        if tools:
            tools_description = self._build_tools_prompt(tools)
            enhanced_system_prompt = f"""{system_prompt}

{tools_description}

RESPONSE FORMAT - STRICTLY FOLLOW:

For tool calls, respond with ONLY this exact JSON structure (no other text before or after):
{{"tool_call": {{"name": "TOOL_NAME", "arguments": {{"param": "value"}}}}}}

Examples:
- To get calendar: {{"tool_call": {{"name": "get_all_calendar_event", "arguments": {{}}}}}}
- To get weather: {{"tool_call": {{"name": "get_weather", "arguments": {{"place": "Berlin"}}}}}}
- To create event: {{"tool_call": {{"name": "create_calendar_event", "arguments": {{"title": "Meeting", "start_time": "2026-01-14T09:00", "end_time": "2026-01-14T10:00"}}}}}}
- To delete event: {{"tool_call": {{"name": "delete_calendar_event", "arguments": {{"id": "123"}}}}}}

For normal conversation (greetings, thanks, unrelated questions): respond with plain text only.

NEVER invent calendar events or weather data. ALWAYS use the appropriate tool."""
        else:
            enhanced_system_prompt = system_prompt

        # Build messages list with history
        messages = [{"role": "system", "content": enhanced_system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": prompt})

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                    },
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract the generated text
                generated_text = result.get("message", {}).get("content", "")
                
                if not generated_text:
                    raise ValueError("Empty response from LLM")
                
                # Log raw LLM output for debugging
                logger.info(f"LLM raw response: {generated_text[:500]}")
                
                # Check if this is a tool call
                tool_calls = self._parse_tool_calls(generated_text)
                
                if tool_calls:
                    logger.info(f"Tool calls detected: {[tc.name for tc in tool_calls]}")
                    return LLMResult2(
                        text=None,
                        tool_calls=tool_calls,
                        model=self.model,
                        tokens_used=result.get("eval_count", 0),
                        prompt_tokens=result.get("prompt_eval_count", 0),
                    )
                else:
                    return LLMResult2(
                        text=generated_text,
                        tool_calls=None,
                        model=self.model,
                        tokens_used=result.get("eval_count", 0),
                        prompt_tokens=result.get("prompt_eval_count", 0),
                    )
                
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return LLMResult2(
                text=f"Cannot connect to LLM server at {self.base_url}",
                tool_calls=None,
                model=self.model,
            )
        except httpx.TimeoutException as e:
            logger.error(f"Ollama timeout: {e}")
            return LLMResult2(
                text="LLM request timed out",
                tool_calls=None,
                model=self.model,
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
            return LLMResult2(
                text=f"LLM API error: {e.response.status_code}",
                tool_calls=None,
                model=self.model,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return LLMResult2(
                text=f"LLM generation failed: {str(e)}",
                tool_calls=None,
                model=self.model,
            )

    def _build_tools_prompt(self, tools: List[Tool]) -> str:
        """Build a concise description of available tools for the system prompt."""
        tools_text = "AVAILABLE TOOLS:\n"
        
        for tool in tools:
            # Get required params
            required = tool.parameters.get("required", [])
            props = tool.parameters.get("properties", {})
            
            param_list = []
            for param_name, param_info in props.items():
                req_marker = "*" if param_name in required else ""
                param_list.append(f"{param_name}{req_marker}")
            
            params_str = ", ".join(param_list) if param_list else "none"
            tools_text += f"- {tool.name}: {tool.description} (params: {params_str})\n"
        
        return tools_text

    def _parse_tool_calls(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from LLM response with improved extraction."""
        try:
            cleaned_text = text.strip()
            
            # Remove markdown code blocks if present
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text.replace("```", "").strip()
            
            # Try to find JSON object in the text (LLM might add extra text)
            import re
            json_match = re.search(r'\{[^{}]*"tool_call"[^{}]*\{[^{}]*\}[^{}]*\}', cleaned_text, re.DOTALL)
            if json_match:
                cleaned_text = json_match.group(0)
            
            # Also try to match simpler pattern where tool_call contains nested object
            if not json_match:
                json_match = re.search(r'\{"tool_call"\s*:\s*\{.*?\}\s*\}', cleaned_text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group(0)
            
            parsed = json.loads(cleaned_text)
            logger.info(f"Parsed tool call JSON: {parsed}")
            
            if "tool_call" in parsed:
                tool_call = parsed["tool_call"]
                return [ToolCall(
                    name=tool_call["name"],
                    arguments=tool_call.get("arguments", {})
                )]
            
            return None
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"No tool call found in response: {e}")
            return None
        
            

    async def chat(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResult:
        """
        Generate a response from the LLM with conversation history.

        Args:
            messages: List of Message objects representing the conversation
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            LLMResult with the generated text
        """
        max_tokens = max_tokens or settings.llm_max_tokens
        temperature = temperature if temperature is not None else settings.llm_temperature

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": m.role, "content": m.content} for m in messages
                        ],
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                    },
                )

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Ollama error: {response.status_code} - {error_text}")
                    return LLMResult(
                        text="",
                        success=False,
                        error=f"LLM request failed: {error_text}",
                    )

                data = response.json()
                message = data.get("message", {})
                generated_text = message.get("content", "").strip()

                if not generated_text:
                    return LLMResult(
                        text="",
                        success=False,
                        error="LLM returned empty response",
                    )

                return LLMResult(
                    text=generated_text,
                    success=True,
                    model=self.model,
                )

        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return LLMResult(
                text="",
                success=False,
                error=f"Cannot connect to LLM server at {self.base_url}",
            )
        except httpx.TimeoutException as e:
            logger.error(f"Ollama timeout: {e}")
            return LLMResult(
                text="",
                success=False,
                error="LLM request timed out",
            )
        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            return LLMResult(
                text="",
                success=False,
                error=str(e),
            )

    async def health_check(self) -> bool:
        """Check if Ollama server is reachable and model is available."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                # Check if server is up
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False

                # Check if our model is available
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]

                # Check for exact match or model without tag
                model_base = self.model.split(":")[0]
                for m in models:
                    if m == self.model or m.startswith(f"{model_base}:"):
                        return True

                logger.warning(f"Model {self.model} not found. Available: {models}")
                return False

        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def pull_model(self) -> bool:
        """Pull the configured model from Ollama registry."""
        try:
            logger.info(f"Pulling model {self.model}...")
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model, "stream": False},
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    async def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m.get("name", "") for m in data.get("models", [])]
                return []
        except Exception:
            return []
