"""FastAPI application for voice services (STT and TTS)."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.config import settings

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Voice API...")
    logger.info(f"STT server: {settings.whisper_host}:{settings.whisper_port}")
    logger.info(f"TTS server: {settings.piper_host}:{settings.piper_port}")
    logger.info(f"LLM server: {settings.ollama_host}:{settings.ollama_port} (model: {settings.ollama_model})")
    yield
    logger.info("Shutting down Voice API...")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="""
## Voice API

This API provides a complete voice assistant pipeline: Speech-to-Text (STT), 
Large Language Model (LLM), and Text-to-Speech (TTS).

### Features
- **STT**: Transcribe speech using faster-whisper
- **LLM**: Generate responses using Ollama
- **TTS**: Synthesize speech using Piper
- **Conversation**: Full voice-to-voice pipeline

### Endpoints
- `POST /api/v1/transcribe` - Convert speech to text
- `POST /api/v1/synthesize` - Convert text to speech
- `POST /api/v1/chat` - Chat with the LLM
- `POST /api/v1/converse` - Full voice conversation (audio in -> audio out)
- `GET /api/v1/health` - Check service health
- `GET /api/v1/models` - List available LLM models
    """,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["transcription"])

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", tags=["root"])
async def root():
    """Serve the frontend application."""
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
