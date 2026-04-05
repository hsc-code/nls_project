"""
Intent Classifier Service - Stage 1 of the LLM pipeline.

Lightweight intent detection and entity extraction with minimal prompting.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Classified intent from user input."""
    
    intent_type: str  # weather, calendar_create, calendar_read, calendar_update, calendar_delete, conversation
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: str = "high"  # high, medium, low
    raw_response: str = ""  # For debugging


# Intent types
INTENT_WEATHER = "weather"
INTENT_CALENDAR_CREATE = "calendar_create"
INTENT_CALENDAR_READ = "calendar_read"
INTENT_CALENDAR_UPDATE = "calendar_update"
INTENT_CALENDAR_DELETE = "calendar_delete"
INTENT_CONVERSATION = "conversation"

VALID_INTENTS = [
    INTENT_WEATHER,
    INTENT_CALENDAR_CREATE,
    INTENT_CALENDAR_READ,
    INTENT_CALENDAR_UPDATE,
    INTENT_CALENDAR_DELETE,
    INTENT_CONVERSATION,
]


class IntentClassifier:
    """
    Stage 1: Classify user intent and extract entities.
    
    Uses a minimal prompt to reduce cognitive load on small models.
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
    
    def _build_system_prompt(self, has_history: bool = False) -> str:
        """Build a minimal, focused system prompt for intent classification."""
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        
        history_note = "\nIMPORTANT: Use conversation history above to understand context. 'that event', 'the meeting', 'it' refer to recent events mentioned." if has_history else ""
        
        # More explicit prompt with examples - small models need very clear instructions
        return f"""You are a JSON-only classifier. You MUST respond with ONLY valid JSON, nothing else.

Today: {now.strftime('%Y-%m-%d')}. Tomorrow: {tomorrow.strftime('%Y-%m-%d')}.{history_note}

Classify the user's intent into ONE of these types:
1. weather - user asks about weather/forecast/temperature
2. calendar_create - user wants to CREATE/schedule/book a new event/meeting
3. calendar_read - user wants to SEE/list/check/view existing events/meetings/calendar
4. calendar_update - user wants to EDIT/change/modify/reschedule an existing event
5. calendar_delete - user wants to CANCEL/delete/remove an event
6. conversation - greetings, small talk, or unclear requests

EXAMPLES:
User: "What's the weather in Paris?" → {{"intent":"weather","entities":{{"place":"Paris"}}}}
User: "Weather in Frankfurt" → {{"intent":"weather","entities":{{"place":"Frankfurt"}}}}
User: "Show my calendar" → {{"intent":"calendar_read","entities":{{}}}}
User: "Schedule a meeting tomorrow at 2pm" → {{"intent":"calendar_create","entities":{{"title":"meeting","start_time":"{tomorrow.strftime('%Y-%m-%d')}T14:00","end_time":"{tomorrow.strftime('%Y-%m-%d')}T15:00"}}}}
User: "Can you get me the meeting details?" → {{"intent":"calendar_read","entities":{{}}}}
User: "Create an event" → {{"intent":"calendar_create","entities":{{"title":"event"}}}}

IMPORTANT: For weather intent, ALWAYS use "place" as the entity key, not "city" or "location".

RULES:
- Output ONLY the JSON object, no other text
- Start with {{ and end with }}
- Extract dates/times and convert to ISO format (YYYY-MM-DDTHH:MM)
- For calendar_read: use empty entities {{}} if no filters
- For calendar_create: extract title, start_time, end_time, location if mentioned
- If unclear, use conversation intent

YOUR RESPONSE MUST BE VALID JSON ONLY. NO EXPLANATIONS. NO MARKDOWN CODE BLOCKS."""
    
    async def classify(
        self,
        user_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Intent:
        """
        Classify user intent and extract entities.
        
        Args:
            user_text: The user's input text
            conversation_history: Recent conversation for context (optional)
            
        Returns:
            Intent object with type and extracted entities
        """
        logger.info("=" * 60)
        logger.info("INTENT CLASSIFIER - Stage 1")
        logger.info("=" * 60)
        logger.info(f"Input text: '{user_text}'")
        
        has_history = bool(conversation_history and len(conversation_history) > 0)
        system_prompt = self._build_system_prompt(has_history=has_history)
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for context (last 4 exchanges = 8 messages max)
        if conversation_history:
            recent_history = conversation_history[-8:]  # Last 4 user + 4 assistant
            messages.extend(recent_history)
            logger.info(f"Added {len(recent_history)} history messages for context")
            # Log recent context for debugging
            for i, msg in enumerate(recent_history[-4:]):
                logger.debug(f"History[{i}]: {msg.get('role')}: {msg.get('content', '')[:100]}")
        
        # Enhance user message with context hints if needed
        enhanced_user_text = user_text
        if has_history and any(word in user_text.lower() for word in ["that", "it", "the", "this", "edit", "change", "update"]):
            enhanced_user_text = f"{user_text}\n\n[Context: Previous conversation above shows recent events and actions. Use it to resolve references like 'that event' or 'the meeting'.]"
        
        messages.append({"role": "user", "content": enhanced_user_text})
        
        logger.debug(f"System prompt length: {len(system_prompt)} chars")
        logger.debug(f"Total messages: {len(messages)}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info("Sending request to Ollama...")
                
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": 200,  # Short response expected
                            "temperature": 0.0,  # Zero temperature for most deterministic JSON output
                            "top_p": 0.9,  # Nucleus sampling for better consistency
                        },
                        "format": "json",  # Force JSON output if supported by Ollama
                    },
                )
                
                response.raise_for_status()
                result = response.json()
                
                raw_response = result.get("message", {}).get("content", "")
                logger.info(f"Raw LLM response: '{raw_response}'")
                
                # Parse the response
                intent = self._parse_response(raw_response, user_text)
                
                logger.info(f"Classified intent: {intent.intent_type}")
                logger.info(f"Extracted entities: {intent.entities}")
                logger.info(f"Confidence: {intent.confidence}")
                logger.info("=" * 60)
                
                return intent
                
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return Intent(
                intent_type=INTENT_CONVERSATION,
                entities={"response": "I'm having trouble connecting. Please try again."},
                confidence="low",
                raw_response=str(e),
            )
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return Intent(
                intent_type=INTENT_CONVERSATION,
                entities={"response": "I didn't understand that. Could you rephrase?"},
                confidence="low",
                raw_response=str(e),
            )
    
    def _parse_response(self, raw_response: str, original_text: str) -> Intent:
        """Parse LLM response into Intent object with improved JSON extraction."""
        
        logger.debug(f"Parsing response: '{raw_response[:200]}...'")
        
        # Clean up the response
        cleaned = raw_response.strip()
        
        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.replace("```", "").strip()
        
        # Try multiple JSON extraction strategies
        json_candidates = []
        
        # Strategy 1: Look for complete JSON object with nested entities
        nested_pattern = r'\{[^{}]*(?:"intent"[^{}]*"entities"[^{}]*\{[^{}]*\}[^{}]*)\}'
        match = re.search(nested_pattern, cleaned, re.DOTALL)
        if match:
            json_candidates.append(match.group(0))
        
        # Strategy 2: Look for JSON object with intent and entities (simpler)
        simple_pattern = r'\{[^{}]*"intent"[^{}]*"entities"[^{}]*[^}]*\}'
        match = re.search(simple_pattern, cleaned, re.DOTALL)
        if match:
            json_candidates.append(match.group(0))
        
        # Strategy 3: Find first { and last } and try to parse
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidates.append(cleaned[first_brace:last_brace+1])
        
        # Strategy 4: Try the whole cleaned string
        json_candidates.append(cleaned)
        
        # Try parsing each candidate
        for candidate in json_candidates:
            try:
                logger.debug(f"Trying to parse: '{candidate[:100]}...'")
                parsed = json.loads(candidate)
                
                intent_type = parsed.get("intent", INTENT_CONVERSATION)
                entities = parsed.get("entities", {})
                
                # Validate intent type
                if intent_type not in VALID_INTENTS:
                    logger.warning(f"Invalid intent type '{intent_type}', defaulting to conversation")
                    intent_type = INTENT_CONVERSATION
                
                # Normalize entity keys for consistency
                # For weather, convert "city" or "location" to "place"
                if intent_type == INTENT_WEATHER:
                    if "city" in entities and "place" not in entities:
                        entities["place"] = entities.pop("city")
                        logger.debug("Normalized 'city' to 'place' for weather intent")
                    elif "location" in entities and "place" not in entities:
                        entities["place"] = entities.pop("location")
                        logger.debug("Normalized 'location' to 'place' for weather intent")
                
                # Determine confidence based on entity completeness
                confidence = self._assess_confidence(intent_type, entities)
                
                logger.info(f"Successfully parsed JSON: intent={intent_type}, entities={entities}")
                
                return Intent(
                    intent_type=intent_type,
                    entities=entities,
                    confidence=confidence,
                    raw_response=raw_response,
                )
                
            except json.JSONDecodeError:
                continue
        
        # If all parsing attempts failed, use fallback
        logger.warning(f"JSON parse error: Could not extract valid JSON from: '{raw_response[:200]}...'")
        logger.warning(f"Failed to parse: '{raw_response}'")
        
        # Fallback: try keyword-based classification
        return self._fallback_classification(original_text, raw_response)
    
    def _assess_confidence(self, intent_type: str, entities: Dict[str, Any]) -> str:
        """Assess confidence level based on extracted entities."""
        
        if intent_type == INTENT_WEATHER:
            # Check for place, city, or location (flexible)
            has_location = bool(entities.get("place") or entities.get("city") or entities.get("location"))
            return "high" if has_location else "low"
        
        elif intent_type == INTENT_CALENDAR_CREATE:
            has_title = bool(entities.get("title"))
            has_time = bool(entities.get("start_time"))
            if has_title and has_time:
                return "high"
            elif has_title or has_time:
                return "medium"
            return "low"
        
        elif intent_type in [INTENT_CALENDAR_READ, INTENT_CONVERSATION]:
            return "high"
        
        elif intent_type in [INTENT_CALENDAR_UPDATE, INTENT_CALENDAR_DELETE]:
            return "high" if entities.get("id") or entities.get("event_hint") else "medium"
        
        return "medium"
    
    def _fallback_classification(self, user_text: str, raw_response: str) -> Intent:
        """Improved keyword-based fallback when JSON parsing fails."""
        
        logger.info("Using fallback keyword classification")
        
        text_lower = user_text.lower()
        
        # Weather keywords - highest priority
        weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot", "climate"]
        if any(kw in text_lower for kw in weather_keywords):
            place = self._extract_place(user_text)
            return Intent(
                intent_type=INTENT_WEATHER,
                entities={"place": place} if place else {},
                confidence="medium",
                raw_response=raw_response,
            )
        
        # Calendar READ keywords - check BEFORE create (more specific patterns first)
        # Questions about existing events - EXPANDED patterns
        read_patterns = [
            "what's on", "what is on", "my calendar", "my events", "my schedule",
            "show events", "list events", "any meetings", "any events",
            "do i have", "have i got", "anything scheduled", "anything on",
            "tell me about", "about the", "scheduled meeting", "meeting details",
            "what meetings", "what events", "check calendar", "check my",
            "get me", "get the", "can you get", "show me", "tell me more",
            "calendar details", "meeting information", "event information",
            "all the", "all my", "all events", "all meetings"
        ]
        if any(pattern in text_lower for pattern in read_patterns):
            # Additional check: if it's asking about details/info, it's likely read
            if any(word in text_lower for word in ["details", "information", "about", "tell me", "show me", "get me"]):
                return Intent(
                    intent_type=INTENT_CALENDAR_READ,
                    entities={},
                    confidence="medium",
                    raw_response=raw_response,
                )
        
        # Calendar UPDATE keywords - check BEFORE delete/create
        update_keywords = ["edit", "change", "update", "modify", "reschedule", "move", "adjust"]
        if any(kw in text_lower for kw in update_keywords):
            return Intent(
                intent_type=INTENT_CALENDAR_UPDATE,
                entities={"event_hint": user_text},  # Will be resolved in action executor
                confidence="medium",
                raw_response=raw_response,
            )
        
        # Calendar DELETE keywords
        delete_keywords = ["cancel", "delete", "remove event", "clear event"]
        if any(kw in text_lower for kw in delete_keywords):
            return Intent(
                intent_type=INTENT_CALENDAR_DELETE,
                entities={"event_hint": user_text},  # Will be resolved in action executor
                confidence="low",
                raw_response=raw_response,
            )
        
        # Calendar CREATE keywords - EXPANDED patterns
        # Must have ACTION verbs for creating OR clear time/date indicators
        create_action_verbs = [
            "schedule", "book", "create", "add", "set up", "plan", "make",
            "create me", "create an", "schedule a", "book a", "add a",
            "set up a", "plan a", "make a", "put", "insert"
        ]
        has_create_verb = any(verb in text_lower for verb in create_action_verbs)
        
        # Time/date indicators which suggest creation
        has_time_indicator = any(t in text_lower for t in [
            "at ", "from ", "to ", " am", " pm", "o'clock", "tomorrow", 
            "next ", "on the", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "january", "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "2nd", "3rd", "4th", "5th", "1st"
        ])
        has_event_noun = any(n in text_lower for n in [
            "meeting", "appointment", "event", "call", "conference"
        ])
        
        # If it has create verb OR (time indicator AND event noun AND not a question)
        if has_create_verb or (has_time_indicator and has_event_noun and "?" not in user_text):
            # Try to extract basic info for calendar_create
            entities = {"raw_request": user_text}
            
            # Try to extract title
            title_match = re.search(r'(?:title|meeting|event|call)\s+(?:is|:)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', user_text, re.IGNORECASE)
            if title_match:
                entities["title"] = title_match.group(1)
            
            return Intent(
                intent_type=INTENT_CALENDAR_CREATE,
                entities=entities,
                confidence="low",
                raw_response=raw_response,
            )
        
        # Default to conversation - use LLM's response if it looks reasonable
        response = raw_response if raw_response and len(raw_response) < 300 else "I'm not sure how to help with that."
        return Intent(
            intent_type=INTENT_CONVERSATION,
            entities={"response": response},
            confidence="low",
            raw_response=raw_response,
        )
    
    def _extract_place(self, text: str) -> Optional[str]:
        """Try to extract a place name from text."""
        
        # Common patterns
        patterns = [
            r"weather (?:in|at|for) ([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+weather",
            r"(?:in|at|for)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
