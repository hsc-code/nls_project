from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import httpx

@dataclass
class Tool:
    """Definition of a tool/function the LLM can call"""
    name: str
    description: str
    parameters: Dict[str, Any]

@dataclass
class ToolCall:
    """Represents a tool call decision from the LLM"""
    name: str
    arguments: Dict[str, Any]

@dataclass
class LLMResult2:
    """Result from LLM generation"""
    text: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    model: str = ""
    tokens_used: int = 0
    prompt_tokens: int = 0

# Tool Definitions
WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get weather forecast for a specific location. Use this when the user asks about weather, temperature, forecast, or climate conditions.",
    parameters={
        "type": "object",
        "properties": {
            "place": {
                "type": "string",
                "description": "The city or location name (e.g., 'Marburg', 'Berlin', 'London')"
            }
        },
        "required": ["place"]
    }
)

CALENDAR_CREATE_TOOL = Tool(
    name="create_calendar_event",
    description="Create a new calendar event, meeting, or appointment. Use this when the user wants to schedule, book, or plan something.",
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title or name of the event"
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the event (optional)"
            },
            "start_time": {
                "type": "string",
                "description": "Start time in ISO 8601 format (YYYY-MM-DDTHH:MM)"
            },
            "end_time": {
                "type": "string",
                "description": "End time in ISO 8601 format (YYYY-MM-DDTHH:MM)"
            },
            "location": {
                "type": "string",
                "description": "Location of the event (optional)"
            }
        },
        "required": ["title", "start_time", "end_time"]
    }
)
GET_LIST_CALENDAR_TOOL = Tool(
    name="get_all_calendar_event",
    description="Get all calendar event, meeting, or appointment available. Use this when the user wants to get all scheduled events",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    }
)
CALENDAR_UPDATE_TOOL = Tool(
    name="update_calendar_event",
    description="Update a new calendar event, meeting, or appointment. Use this when the user wants to update, book, or plan something.",
    parameters={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "id of the event"
            },
            "title": {
                "type": "string",
                "description": "Title or name of the event"
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the event (optional)"
            },
            "start_time": {
                "type": "string",
                "description": "Start time in ISO 8601 format (YYYY-MM-DDTHH:MM)"
            },
            "end_time": {
                "type": "string",
                "description": "End time in ISO 8601 format (YYYY-MM-DDTHH:MM)"
            },
            "location": {
                "type": "string",
                "description": "Location of the event (optional)"
            }
        },
        "required": ["id"]
    }
)
CALENDAR_DELETE_TOOL = Tool(
    name="delete_calendar_event",
    description="Delete a calendar event by its ID. Use this when the user wants to cancel or remove an event.",
    parameters={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "ID of the calendar event to delete"
            }
        },
        "required": ["id"]
    }
)
tools = [WEATHER_TOOL, CALENDAR_CREATE_TOOL, GET_LIST_CALENDAR_TOOL, CALENDAR_UPDATE_TOOL, CALENDAR_DELETE_TOOL]


# Tool Executor
class ToolExecutor:
    """Executes tool calls by making actual API requests"""
    
    def __init__(self):
        self.tools_map = {
            "get_weather": self.execute_weather,
            "create_calendar_event": self.execute_calendar_create,
            "get_all_calendar_event": self.execute_calendar_get_all,
            "update_calendar_event": self.execute_calendar_update,
            "delete_calendar_event": self.execute_calendar_delete, 
        }
    
    async def execute(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute a tool call and return the result"""
        if tool_call.name not in self.tools_map:
            return {"error": f"Unknown tool: {tool_call.name}"}
        
        handler = self.tools_map[tool_call.name]
        try:
            result = await handler(tool_call.arguments)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_weather(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute weather API call"""
        place = args.get("place")
        if not place:
            raise ValueError("Missing required parameter: place")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.responsible-nlp.net/weather.php",
                data={"place": place}
            )
            response.raise_for_status()
            return response.json()
    
    async def execute_calendar_create(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute calendar create API call"""
        required = ["title", "start_time", "end_time"]
        for field in required:
            if field not in args:
                raise ValueError(f"Missing required parameter: {field}")
        
        payload = {
            "title": args["title"],
            "description": args.get("description", ""),
            "start_time": args["start_time"],
            "end_time": args["end_time"],
            "location": args.get("location", "")
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.responsible-nlp.net/calendar.php?calenderid=255",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
    
    async def execute_calendar_update(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute calendar update API call"""
        required = ["id"]
        for field in required:
            if field not in args:
                raise ValueError(f"Missing required parameter: {field}")
        id = args["id"]
        payload = {
            "title": args .get("title", ""),
            "description": args.get("description", ""),
            "start_time": args .get("start_time", ""),
            "end_time": args .get("end_time", ""),
            "location": args.get("location", "")
        }
        
        url = f"https://api.responsible-nlp.net/calendar.php?calenderid=255&id={id}"

        async with httpx.AsyncClient() as client:
            response = await client.put(url,
                json=payload,
                headers={"Content-Type": "application/json"},
                params={"id": id}
            )
            response.raise_for_status()
            return response.json()
    async def execute_calendar_get_all(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute calendar get API call"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.responsible-nlp.net/calendar.php?calenderid=255",
                
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        
    async def execute_calendar_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
      """Execute calendar delete API call"""
      if "id" not in args:
        raise ValueError("Missing required parameter: id")

      event_id = args["id"]
      url = f"https://api.responsible-nlp.net/calendar.php?calenderid=255&id={event_id}"
      async with httpx.AsyncClient() as client:
        response = await client.delete( 
            url,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
