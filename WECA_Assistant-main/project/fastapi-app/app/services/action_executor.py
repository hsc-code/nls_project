"""
Action Executor Service - Stage 2 of the LLM pipeline.

Executes actions based on classified intents. No LLM involved - pure code routing.
"""

import logging
from typing import Dict, Any, Optional, List

import httpx

from app.services.intent_classifier import (
    Intent,
    INTENT_WEATHER,
    INTENT_CALENDAR_CREATE,
    INTENT_CALENDAR_READ,
    INTENT_CALENDAR_UPDATE,
    INTENT_CALENDAR_DELETE,
    INTENT_CONVERSATION,
)

logger = logging.getLogger(__name__)


# API Configuration
WEATHER_API_URL = "https://api.responsible-nlp.net/weather.php"
CALENDAR_API_URL = "https://api.responsible-nlp.net/calendar.php"
CALENDAR_ID = "255"  # TODO: Make this configurable


class ActionExecutor:
    """
    Stage 2: Execute actions based on classified intents.
    
    This is pure code - no LLM calls. Routes intents to appropriate APIs.
    """
    
    def __init__(self):
        self.timeout = httpx.Timeout(15.0, connect=5.0)
    
    async def execute(
        self,
        intent: Intent,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute action based on intent.
        
        Args:
            intent: Classified intent with entities
            conversation_history: For context when needed (e.g., "delete that event")
            
        Returns:
            Dict with 'success', 'data', and optional 'error' keys
        """
        logger.info("=" * 60)
        logger.info("ACTION EXECUTOR - Stage 2")
        logger.info("=" * 60)
        logger.info(f"Intent type: {intent.intent_type}")
        logger.info(f"Entities: {intent.entities}")
        logger.info(f"Confidence: {intent.confidence}")
        
        # Route to appropriate handler
        handlers = {
            INTENT_WEATHER: self._execute_weather,
            INTENT_CALENDAR_CREATE: self._execute_calendar_create,
            INTENT_CALENDAR_READ: self._execute_calendar_read,
            INTENT_CALENDAR_UPDATE: self._execute_calendar_update,
            INTENT_CALENDAR_DELETE: self._execute_calendar_delete,
            INTENT_CONVERSATION: self._handle_conversation,
        }
        
        handler = handlers.get(intent.intent_type, self._handle_conversation)
        
        try:
            result = await handler(intent.entities, conversation_history)
            logger.info(f"Execution result: success={result.get('success')}")
            if result.get('error'):
                logger.warning(f"Execution error: {result.get('error')}")
            logger.info("=" * 60)
            return result
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": None,
            }
    
    async def _execute_weather(
        self,
        entities: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Execute weather API call."""
        
        # Try multiple possible entity keys (place, city, location) for flexibility
        place = entities.get("place") or entities.get("city") or entities.get("location")
        
        if not place:
            logger.warning("No place specified for weather request")
            logger.debug(f"Available entities: {entities}")
            return {
                "success": False,
                "error": "No location specified. Please tell me which city you want weather for.",
                "data": None,
            }
        
        logger.info(f"Fetching weather for: {place}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    WEATHER_API_URL,
                    data={"place": place},
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Weather API response: {data}")
                
                return {
                    "success": True,
                    "data": data,
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Weather API HTTP error: {e.response.status_code}")
            return {
                "success": False,
                "error": f"Weather service returned error: {e.response.status_code}",
                "data": None,
            }
        except Exception as e:
            logger.error(f"Weather API call failed: {e}")
            return {
                "success": False,
                "error": f"Could not fetch weather: {str(e)}",
                "data": None,
            }
    
    async def _execute_calendar_read(
        self,
        entities: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Fetch calendar events, optionally filtered by entities."""
        
        logger.info("Fetching calendar events")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{CALENDAR_API_URL}?calenderid={CALENDAR_ID}",
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                all_events = response.json()
                
                logger.info(f"Calendar API response: {len(all_events) if isinstance(all_events, list) else 'unknown'} events")
                
                # Filter events if specific criteria provided
                filtered_events = self._filter_calendar_events(all_events, entities)
                
                logger.info(f"Filtered to {len(filtered_events)} matching event(s)")
                
                return {
                    "success": True,
                    "data": filtered_events,
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Calendar API HTTP error: {e.response.status_code}")
            return {
                "success": False,
                "error": f"Calendar service returned error: {e.response.status_code}",
                "data": None,
            }
        except Exception as e:
            logger.error(f"Calendar read failed: {e}")
            return {
                "success": False,
                "error": f"Could not fetch calendar: {str(e)}",
                "data": None,
            }
    
    async def _execute_calendar_create(
        self,
        entities: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Create a calendar event."""
        
        title = entities.get("title")
        start_time = entities.get("start_time")
        end_time = entities.get("end_time")
        
        # Validate required fields
        missing = []
        if not title:
            missing.append("title")
        if not start_time:
            missing.append("start time")
        if not end_time:
            # Default end time to 1 hour after start
            if start_time:
                end_time = self._add_hour_to_time(start_time)
            else:
                missing.append("end time")
        
        if missing:
            logger.warning(f"Missing required fields for calendar create: {missing}")
            return {
                "success": False,
                "error": f"Missing required information: {', '.join(missing)}. Please provide these details.",
                "data": None,
            }
        
        payload = {
            "title": title,
            "description": entities.get("description", ""),
            "start_time": start_time,
            "end_time": end_time,
            "location": entities.get("location", ""),
        }
        
        logger.info(f"Creating calendar event: {payload}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{CALENDAR_API_URL}?calenderid={CALENDAR_ID}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Calendar create response: {data}")
                
                return {
                    "success": True,
                    "data": data,
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Calendar create HTTP error: {e.response.status_code}")
            return {
                "success": False,
                "error": f"Could not create event: {e.response.status_code}",
                "data": None,
            }
        except Exception as e:
            logger.error(f"Calendar create failed: {e}")
            return {
                "success": False,
                "error": f"Could not create event: {str(e)}",
                "data": None,
            }
    
    async def _resolve_event_id_from_context(
        self,
        entities: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """
        Resolve event ID from context by:
        1. Checking if ID is directly provided
        2. Fetching all events and matching against conversation history
        3. Returning the most recently created/mentioned event ID
        """
        # If ID is directly provided, use it
        event_id = entities.get("id")
        if event_id:
            logger.info(f"Event ID provided directly: {event_id}")
            return str(event_id)
        
        # Fetch all events
        logger.info("No event ID provided, fetching all events to resolve from context")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{CALENDAR_API_URL}?calenderid={CALENDAR_ID}",
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                all_events = response.json()
                
                if not all_events or len(all_events) == 0:
                    logger.warning("No events found in calendar")
                    return None
                
                logger.info(f"Fetched {len(all_events)} events from calendar")
                
                # If we have conversation history, try to match
                if history:
                    # Look for mentions of event titles, times, or IDs in recent history
                    recent_text = " ".join([
                        msg.get("content", "") for msg in history[-6:]  # Last 3 exchanges
                    ]).lower()
                    
                    logger.debug(f"Searching history for event references: {recent_text[:200]}")
                    
                    # Find the most recently mentioned event
                    # Check for event IDs first
                    for event in all_events:
                        event_id_str = str(event.get("id", ""))
                        if event_id_str in recent_text:
                            logger.info(f"Found event ID {event_id_str} mentioned in history")
                            return event_id_str
                    
                    # Then check for event titles
                    for event in reversed(all_events):  # Most recent first
                        title = event.get("title", "").lower()
                        if title and title in recent_text:
                            logger.info(f"Found event '{title}' (ID: {event.get('id')}) mentioned in history")
                            return str(event.get("id"))
                
                # Fallback: return the most recent event (highest ID or latest start_time)
                most_recent = max(all_events, key=lambda e: int(e.get("id", 0)))
                logger.info(f"Using most recent event: ID {most_recent.get('id')} - {most_recent.get('title')}")
                return str(most_recent.get("id"))
                
        except Exception as e:
            logger.error(f"Failed to fetch events for context resolution: {e}")
            return None
    
    async def _execute_calendar_update(
        self,
        entities: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Update a calendar event."""
        
        # Resolve event ID from context if not provided
        event_id = await self._resolve_event_id_from_context(entities, history)
        
        if not event_id:
            return {
                "success": False,
                "error": "I couldn't determine which event to update. Please specify the event ID or say 'show my events' first.",
                "data": None,
            }
        
        logger.info(f"Updating event ID: {event_id}")
        
        payload = {}
        if entities.get("title"):
            payload["title"] = entities["title"]
        if entities.get("description"):
            payload["description"] = entities["description"]
        if entities.get("start_time"):
            payload["start_time"] = entities["start_time"]
        if entities.get("end_time"):
            payload["end_time"] = entities["end_time"]
        if entities.get("location"):
            payload["location"] = entities["location"]
        
        if not payload:
            return {
                "success": False,
                "error": "No changes specified. What would you like to update?",
                "data": None,
            }
        
        logger.info(f"Updating event {event_id} with: {payload}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.put(
                    f"{CALENDAR_API_URL}?calenderid={CALENDAR_ID}&id={event_id}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Calendar update response: {data}")
                
                return {
                    "success": True,
                    "data": data,
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Calendar update HTTP error: {e.response.status_code}")
            return {
                "success": False,
                "error": f"Could not update event: {e.response.status_code}",
                "data": None,
            }
        except Exception as e:
            logger.error(f"Calendar update failed: {e}")
            return {
                "success": False,
                "error": f"Could not update event: {str(e)}",
                "data": None,
            }
    
    async def _execute_calendar_delete(
        self,
        entities: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Delete a calendar event."""
        
        # Resolve event ID from context if not provided
        event_id = await self._resolve_event_id_from_context(entities, history)
        
        if not event_id:
            return {
                "success": False,
                "error": "I couldn't determine which event to delete. Please specify the event ID or say 'show my events' first.",
                "data": None,
            }
        
        logger.info(f"Deleting event ID: {event_id}")
        
        logger.info(f"Deleting event: {event_id}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(
                    f"{CALENDAR_API_URL}?calenderid={CALENDAR_ID}&id={event_id}",
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Calendar delete response: {data}")
                
                return {
                    "success": True,
                    "data": data,
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Calendar delete HTTP error: {e.response.status_code}")
            return {
                "success": False,
                "error": f"Could not delete event: {e.response.status_code}",
                "data": None,
            }
        except Exception as e:
            logger.error(f"Calendar delete failed: {e}")
            return {
                "success": False,
                "error": f"Could not delete event: {str(e)}",
                "data": None,
            }
    
    async def _handle_conversation(
        self,
        entities: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Handle conversation intent - no API call needed."""
        
        logger.info("Handling conversation intent - no API call needed")
        
        # Pass through any suggested response from intent classification
        return {
            "success": True,
            "data": {
                "type": "conversation",
                "suggested_response": entities.get("response"),
            },
        }
    
    def _add_hour_to_time(self, time_str: str) -> str:
        """Add one hour to an ISO time string."""
        
        try:
            from datetime import datetime, timedelta
            
            # Parse the time
            dt = datetime.fromisoformat(time_str)
            dt += timedelta(hours=1)
            return dt.strftime("%Y-%m-%dT%H:%M")
        except Exception:
            # Fallback: just return the same time
            return time_str
    
    def _filter_calendar_events(self, events: List[Dict[str, Any]], entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter calendar events based on provided entities."""
        
        if not isinstance(events, list):
            return []
        
        # If no filter criteria provided, return all events
        title_filter = entities.get("title", "").strip().lower()
        start_time_filter = entities.get("start_time", "").strip()
        end_time_filter = entities.get("end_time", "").strip()
        
        if not title_filter and not start_time_filter and not end_time_filter:
            logger.debug("No filter criteria, returning all events")
            return events
        
        filtered = []
        for event in events:
            matches = True
            
            # Filter by title (case-insensitive, partial match)
            if title_filter:
                event_title = event.get("title", "").strip().lower()
                if title_filter not in event_title and event_title not in title_filter:
                    matches = False
            
            # Filter by start_time (exact match or same day)
            if matches and start_time_filter:
                event_start = event.get("start_time", "").strip()
                # Compare dates (ignore time if only date specified)
                try:
                    from datetime import datetime
                    filter_date = datetime.fromisoformat(start_time_filter.replace("Z", "")).date()
                    event_date = datetime.fromisoformat(event_start.replace("Z", "")).date()
                    if filter_date != event_date:
                        matches = False
                except Exception:
                    # If parsing fails, do string comparison
                    if start_time_filter[:10] not in event_start:
                        matches = False
            
            # Filter by end_time (exact match or same day)
            if matches and end_time_filter:
                event_end = event.get("end_time", "").strip()
                try:
                    from datetime import datetime
                    filter_date = datetime.fromisoformat(end_time_filter.replace("Z", "")).date()
                    event_date = datetime.fromisoformat(event_end.replace("Z", "")).date()
                    if filter_date != event_date:
                        matches = False
                except Exception:
                    # If parsing fails, do string comparison
                    if end_time_filter[:10] not in event_end:
                        matches = False
            
            if matches:
                filtered.append(event)
        
        return filtered