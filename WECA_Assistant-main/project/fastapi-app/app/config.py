"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Faster-Whisper Wyoming server configuration (STT)
    whisper_host: str = "faster-whisper"
    whisper_port: int = 10300

    # Piper Wyoming server configuration (TTS)
    piper_host: str = "piper"
    piper_port: int = 10200

    # Ollama LLM server configuration
    ollama_host: str = "ollama"
    ollama_port: int = 11434
    ollama_model: str = "qwen2.5:0.5b"

    # Redis configuration for conversation memory
    redis_host: str = "redis"
    redis_port: int = 6379
    session_ttl: int = 600  # 10 minutes in seconds
    max_conversation_history: int = 10  # Max messages to keep in context

    # Audio settings (must match Wyoming server expectations)
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_sample_width: int = 2  # 16-bit audio

    # Transcription settings
    default_language: str = "en"

    # API settings
    max_audio_size_mb: int = 25
    max_text_length: int = 5000
    api_title: str = "Voice API"
    api_version: str = "1.0.0"

    # LLM settings
    llm_max_tokens: int = 256
    llm_temperature: float = 0.7
    llm_system_prompt: str = """You are an API routing assistant that analyzes user requests and generates appropriate API payloads.

You have access to two APIs:

1. **Weather API** (GET weather information)
   - Endpoint: https://api.responsible-nlp.net/weather.php
   - Method: POST
   - Payload: {"place": "city_name"}
   - Use for: weather forecasts, temperature queries, weather conditions

2. **Calendar API** (CRUD operations)
   - Endpoint: https://api.responsible-nlp.net/calendar.php
   - Methods: POST (create), GET (read), PUT (update), DELETE (delete)
   - Create payload: {"title": "string", "description": "string", "start_time": "ISO8601", "end_time": "ISO8601", "location": "string"}
   - Use for: scheduling, appointments, meetings, events, reminders

**RESPONSE FORMAT:**

For API requests, respond with a JSON object:
{
  "type": "api_call",
  "api": "weather" | "calendar",
  "method": "POST" | "GET" | "PUT" | "DELETE",
  "payload": { ... },
  "explanation": "brief explanation of what you understood"
}

For general conversation (non-API requests), respond with:
{
  "type": "conversation",
  "message": "your conversational response here"
}

**RULES:**
- Always respond with valid JSON
- Extract relevant details from natural language (dates, times, locations, etc.)
- For dates/times, convert to ISO 8601 format (YYYY-MM-DDTHH:MM)
- If information is missing for calendar events, use reasonable defaults or ask for clarification
- Be case-insensitive when detecting weather/calendar intents
- Weather-related keywords: weather, forecast, temperature, rain, sunny, climate, etc.
- Calendar-related keywords: schedule, meeting, appointment, event, reminder, book, calendar, plan, etc.
"""

    class Config:
        env_prefix = "STT_"
        case_sensitive = False


settings = Settings()
