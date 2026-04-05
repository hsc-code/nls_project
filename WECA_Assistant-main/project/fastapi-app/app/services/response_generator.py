"""
Response Generator Service - Stage 3 of the LLM pipeline.

Generates natural language responses from tool results.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    """Generated natural language response."""
    
    text: str
    success: bool = True
    error: Optional[str] = None


class ResponseGenerator:
    """
    Stage 3: Generate natural language responses from structured data.
    
    Takes tool execution results and converts them to human-friendly speech.
    """
    
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
        self.timeout = httpx.Timeout(30.0, connect=10.0)
    
    async def generate(
        self,
        intent_type: str,
        tool_result: Dict[str, Any],
        original_query: str,
        entities: Optional[Dict[str, Any]] = None,
    ) -> GeneratedResponse:
        """
        Generate a natural language response from tool results.
        
        Args:
            intent_type: The classified intent type
            tool_result: Result from tool execution
            original_query: The user's original question
            entities: Extracted entities from classification
            
        Returns:
            GeneratedResponse with natural language text
        """
        logger.info("=" * 60)
        logger.info("RESPONSE GENERATOR - Stage 3")
        logger.info("=" * 60)
        logger.info(f"Intent type: {intent_type}")
        logger.info(f"Original query: '{original_query}'")
        logger.info(f"Tool result success: {tool_result.get('success', False)}")
        
        # Build context-specific prompt
        system_prompt = self._build_system_prompt(intent_type)
        user_prompt = self._build_user_prompt(intent_type, tool_result, original_query, entities)
        
        logger.debug(f"System prompt: '{system_prompt}'")
        logger.debug(f"User prompt: '{user_prompt}'")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info("Sending request to Ollama for response generation...")
                
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": {
                            "num_predict": 150,  # Keep responses concise for TTS
                            "temperature": 0.6,  # Slightly creative but coherent
                        },
                    },
                )
                
                response.raise_for_status()
                result = response.json()
                
                generated_text = result.get("message", {}).get("content", "").strip()
                
                logger.info(f"Generated response: '{generated_text}'")
                logger.info("=" * 60)
                
                if not generated_text:
                    return GeneratedResponse(
                        text="I processed your request but couldn't generate a response.",
                        success=False,
                        error="Empty response from LLM",
                    )
                
                return GeneratedResponse(text=generated_text, success=True)
                
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return GeneratedResponse(
                text="I'm having trouble generating a response right now.",
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return GeneratedResponse(
                text="Something went wrong while generating the response.",
                success=False,
                error=str(e),
            )
    
    def _build_system_prompt(self, intent_type: str) -> str:
        """Build a minimal system prompt for response generation."""
        
        return """You are Richard, a friendly voice assistant. 
Generate a natural, conversational response in 1-2 sentences.
Keep it brief - this will be spoken aloud.
Be warm and helpful. Don't use emojis or special characters.

IMPORTANT FORMATTING RULES:
- Dates: Always format as "12th January, 2026" or "January 12th, 2026" (never "2026-01-12")
- Times: Format as "2 PM", "12 AM", "10:30 AM" (never "14:00" or "12:00 AM" - drop :00)
- Speak naturally: "tomorrow at 2 PM" not "2026-01-23T14:00"
- For calendar events, mention specific details if only one event matches."""
    
    def _build_user_prompt(
        self,
        intent_type: str,
        tool_result: Dict[str, Any],
        original_query: str,
        entities: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the user prompt with context and data."""
        
        success = tool_result.get("success", False)
        data = tool_result.get("data", {})
        error = tool_result.get("error", "")
        
        if not success:
            return f"""The user asked: "{original_query}"
But the request failed with error: {error}
Apologize briefly and offer to help differently."""
        
        # Format based on intent type
        if intent_type == "weather":
            return self._format_weather_prompt(data, original_query, entities)
        
        elif intent_type == "calendar_read":
            return self._format_calendar_read_prompt(data, original_query, entities)
        
        elif intent_type == "calendar_create":
            return self._format_calendar_create_prompt(data, original_query, entities)
        
        elif intent_type == "calendar_update":
            return self._format_calendar_update_prompt(data, original_query)
        
        elif intent_type == "calendar_delete":
            return self._format_calendar_delete_prompt(data, original_query)
        
        else:
            # Conversation or unknown
            return f"""The user said: "{original_query}"
Respond naturally and helpfully."""
    
    def _format_weather_prompt(
        self,
        data: Dict[str, Any],
        original_query: str,
        entities: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format weather data for response generation."""
        
        place = entities.get("place", "the location") if entities else "the location"
        
        # Handle different weather API response formats
        if isinstance(data, dict):
            # Try to extract key weather info
            temp = data.get("temperature") or data.get("temp") or data.get("main", {}).get("temp")
            condition = data.get("condition") or data.get("weather") or data.get("description")
            humidity = data.get("humidity") or data.get("main", {}).get("humidity")
            
            weather_info = []
            if temp is not None:
                weather_info.append(f"temperature: {temp}")
            if condition:
                weather_info.append(f"condition: {condition}")
            if humidity is not None:
                weather_info.append(f"humidity: {humidity}%")
            
            if weather_info:
                return f"""The user asked: "{original_query}"
Weather data for {place}: {', '.join(weather_info)}
Summarize this weather information naturally."""
        
        # Fallback for raw data
        return f"""The user asked: "{original_query}"
Weather data: {str(data)[:300]}
Summarize the key weather information for {place}."""
    
    def _format_calendar_read_prompt(self, data: Dict[str, Any], original_query: str, entities: Optional[Dict[str, Any]] = None) -> str:
        """Format calendar events for response generation."""
        
        events = data if isinstance(data, list) else data.get("events", [])
        
        if not events:
            return f"""The user asked: "{original_query}"
Result: No events found on the calendar.
Let them know their calendar is clear."""
        
        # Format events with human-readable dates/times
        event_summaries = []
        for i, event in enumerate(events[:5]):  # Limit to 5 events
            title = event.get("title", "Untitled")
            start = event.get("start_time", "")
            end = event.get("end_time", "")
            location = event.get("location", "")
            
            # Format dates and times for natural speech
            start_formatted = self._format_datetime_for_speech(start)
            end_formatted = self._format_datetime_for_speech(end) if end else ""
            
            event_info = f"{title}"
            if start_formatted:
                event_info += f" on {start_formatted}"
            if end_formatted and end_formatted != start_formatted:
                event_info += f" until {end_formatted}"
            if location:
                event_info += f" at {location}"
            
            event_summaries.append(f"- {event_info}")
        
        events_text = "\n".join(event_summaries)
        total = len(events)
        
        # If only one event, provide more detail
        if total == 1:
            return f"""The user asked: "{original_query}"
Found 1 matching event:
{events_text}
Provide the specific details of this event naturally."""
        
        return f"""The user asked: "{original_query}"
Found {total} event(s):
{events_text}
Summarize these events naturally. If there are many, mention the count."""
    
    def _format_calendar_create_prompt(
        self,
        data: Dict[str, Any],
        original_query: str,
        entities: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format calendar creation result."""
        
        title = entities.get("title", "the event") if entities else "the event"
        start_time = entities.get("start_time", "") if entities else ""
        
        return f"""The user asked: "{original_query}"
Result: Successfully created "{title}" {f'at {start_time}' if start_time else ''}.
Confirm the event was created."""
    
    def _format_calendar_update_prompt(self, data: Dict[str, Any], original_query: str) -> str:
        """Format calendar update result."""
        
        return f"""The user asked: "{original_query}"
Result: The event was successfully updated.
Confirm the change briefly."""
    
    def _format_calendar_delete_prompt(self, data: Dict[str, Any], original_query: str) -> str:
        """Format calendar deletion result."""
        
        return f"""The user asked: "{original_query}"
Result: The event was successfully deleted.
Confirm the deletion briefly."""
    
    async def generate_conversation_response(
        self,
        user_text: str,
        suggested_response: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> GeneratedResponse:
        """
        Generate a conversational response (no tool involved).
        
        Args:
            user_text: The user's input
            suggested_response: Optional pre-generated response from intent classification
            conversation_history: Recent conversation for context
            
        Returns:
            GeneratedResponse with natural language text
        """
        logger.info("=" * 60)
        logger.info("RESPONSE GENERATOR - Conversation Mode")
        logger.info("=" * 60)
        logger.info(f"User text: '{user_text}'")
        
        # If we have a suggested response from classification, use it
        if suggested_response:
            logger.info(f"Using suggested response: '{suggested_response}'")
            return GeneratedResponse(text=suggested_response, success=True)
        
        # Otherwise, generate a response
        system_prompt = """You are Richard, a friendly voice assistant.
Respond naturally in 1-2 sentences. Be helpful and warm.
Keep responses brief for speech. No emojis."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[-4:])
        
        messages.append({"role": "user", "content": user_text})
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": 100,
                            "temperature": 0.7,
                        },
                    },
                )
                
                response.raise_for_status()
                result = response.json()
                
                generated_text = result.get("message", {}).get("content", "").strip()
                
                logger.info(f"Generated conversation response: '{generated_text}'")
                logger.info("=" * 60)
                
                return GeneratedResponse(
                    text=generated_text or "I'm not sure how to respond to that.",
                    success=bool(generated_text),
                )
                
        except Exception as e:
            logger.error(f"Conversation response generation failed: {e}")
            return GeneratedResponse(
                text="I'm having trouble responding right now.",
                success=False,
                error=str(e),
            )
    
    def _format_datetime_for_speech(self, iso_string: str) -> str:
        """Convert ISO datetime string to natural speech format.
        
        Examples:
        - "2026-01-12T14:00" -> "12th January, 2026 at 2 PM"
        - "2026-01-23T10:30" -> "23rd January, 2026 at 10:30 AM"
        - "2026-01-12T00:00" -> "12th January, 2026 at 12 AM"
        """
        if not iso_string or iso_string.strip() == "":
            return ""
        
        try:
            # Parse ISO format (handle with/without timezone)
            iso_clean = iso_string.replace("Z", "").strip()
            
            # Try parsing with time
            try:
                dt = datetime.fromisoformat(iso_clean)
            except ValueError:
                # Try just date
                dt = datetime.strptime(iso_clean[:10], "%Y-%m-%d")
            
            # Format date with ordinal suffix
            day = dt.day
            if 4 <= day <= 20 or 24 <= day <= 30:
                suffix = "th"
            else:
                # Handle 1st, 2nd, 3rd, 21st, 22nd, 23rd, 31st
                last_digit = day % 10
                if last_digit == 1:
                    suffix = "st"
                elif last_digit == 2:
                    suffix = "nd"
                elif last_digit == 3:
                    suffix = "rd"
                else:
                    suffix = "th"
            
            date_str = f"{day}{suffix} {dt.strftime('%B, %Y')}"
            
            # Format time (drop :00, use AM/PM)
            hour = dt.hour
            minute = dt.minute
            
            if minute == 0:
                # Just hour, no minutes
                if hour == 0:
                    time_str = "12 AM"
                elif hour < 12:
                    time_str = f"{hour} AM"
                elif hour == 12:
                    time_str = "12 PM"
                else:
                    time_str = f"{hour - 12} PM"
            else:
                # Include minutes
                if hour == 0:
                    time_str = f"12:{minute:02d} AM"
                elif hour < 12:
                    time_str = f"{hour}:{minute:02d} AM"
                elif hour == 12:
                    time_str = f"12:{minute:02d} PM"
                else:
                    time_str = f"{hour - 12}:{minute:02d} PM"
            
            return f"{date_str} at {time_str}"
            
        except Exception as e:
            logger.debug(f"Failed to format datetime '{iso_string}': {e}")
            # Fallback: return cleaned version
            return iso_string.replace("T", " at ").replace("Z", "")
    
    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
