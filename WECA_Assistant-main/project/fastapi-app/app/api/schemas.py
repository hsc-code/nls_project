"""Pydantic schemas for API requests and responses."""

from typing import Optional

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """Response schema for transcription endpoint."""

    text: str = Field(..., description="Transcribed text from the audio")
    language: Optional[str] = Field(None, description="Language of the transcription")
    success: bool = Field(..., description="Whether transcription was successful")
    error: Optional[str] = Field(None, description="Error message if transcription failed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Hello, how are you today?",
                    "language": "en",
                    "success": True,
                    "error": None,
                }
            ]
        }
    }


class TTSRequest(BaseModel):
    """Request schema for TTS endpoint."""

    text: str = Field(..., description="Text to convert to speech", max_length=5000)
    voice: Optional[str] = Field(None, description="Voice name (optional)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Hello, how are you today?",
                    "voice": None,
                }
            ]
        }
    }


class LLMRequest(BaseModel):
    """Request schema for LLM chat endpoint."""

    prompt: str = Field(..., description="User prompt/question", max_length=5000)
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response", ge=1, le=2048)
    temperature: Optional[float] = Field(None, description="Sampling temperature", ge=0.0, le=2.0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "What is the capital of France?",
                    "system_prompt": None,
                    "max_tokens": 256,
                    "temperature": 0.7,
                }
            ]
        }
    }


class LLMResponse(BaseModel):
    """Response schema for LLM chat endpoint."""

    text: str = Field(..., description="Generated response text")
    success: bool = Field(..., description="Whether generation was successful")
    model: Optional[str] = Field(None, description="Model used for generation")
    error: Optional[str] = Field(None, description="Error message if generation failed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "The capital of France is Paris.",
                    "success": True,
                    "model": "qwen2.5:0.5b",
                    "error": None,
                }
            ]
        }
    }


class ConversationResponse(BaseModel):
    """Response schema for the voice conversation pipeline."""

    user_text: str = Field(..., description="Transcribed user speech")
    assistant_text: str = Field(..., description="LLM-generated response")
    success: bool = Field(..., description="Whether the full pipeline succeeded")
    error: Optional[str] = Field(None, description="Error message if any step failed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_text": "What time is it?",
                    "assistant_text": "I don't have access to the current time, but you can check your device!",
                    "success": True,
                    "error": None,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Overall health status")
    whisper_server: bool = Field(..., description="Whether whisper server is reachable")
    piper_server: bool = Field(..., description="Whether piper TTS server is reachable")
    ollama_server: bool = Field(..., description="Whether ollama LLM server is reachable")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "whisper_server": True,
                    "piper_server": True,
                    "ollama_server": True,
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Response schema for error responses."""

    detail: str = Field(..., description="Error message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": "Audio file is required",
                }
            ]
        }
    }


class ModelsResponse(BaseModel):
    """Response schema for listing available LLM models."""

    models: list[str] = Field(..., description="List of available model names")
    current_model: str = Field(..., description="Currently configured model")
