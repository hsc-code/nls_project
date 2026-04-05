"""API routes for voice services (STT, TTS, and LLM)."""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from app.api.schemas import (
    ErrorResponse,
    HealthResponse,
    TranscriptionResponse,
    TTSRequest,
    LLMRequest,
    LLMResponse,
    ModelsResponse,
)
from app.config import settings
from app.services import (
    TranscriptionService,
    TTSService,
    LLMService,
    ConversationManager,
    IntentClassifier,
    ActionExecutor,
    ResponseGenerator,
)
from app.services.intent_classifier import INTENT_CONVERSATION

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton service instances
_transcription_service: Optional[TranscriptionService] = None
_tts_service: Optional[TTSService] = None
_llm_service: Optional[LLMService] = None
_conversation_manager: Optional[ConversationManager] = None
_intent_classifier: Optional[IntentClassifier] = None
_action_executor: Optional[ActionExecutor] = None
_response_generator: Optional[ResponseGenerator] = None


def get_transcription_service() -> TranscriptionService:
    """Get or create the transcription service singleton."""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service


def get_tts_service() -> TTSService:
    """Get or create the TTS service singleton."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service


def get_llm_service() -> LLMService:
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def get_conversation_manager() -> ConversationManager:
    """Get or create the conversation manager singleton."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager


def get_intent_classifier() -> IntentClassifier:
    """Get or create the intent classifier singleton."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier


def get_action_executor() -> ActionExecutor:
    """Get or create the action executor singleton."""
    global _action_executor
    if _action_executor is None:
        _action_executor = ActionExecutor()
    return _action_executor


def get_response_generator() -> ResponseGenerator:
    """Get or create the response generator singleton."""
    global _response_generator
    if _response_generator is None:
        _response_generator = ResponseGenerator()
    return _response_generator


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Whisper server unavailable"},
    },
    summary="Transcribe audio to text",
    description="Upload an audio file to transcribe it to text using faster-whisper.",
)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(
        None,
        description="Language code (e.g., 'en', 'es', 'fr'). If not provided, defaults to English.",
    ),
) -> TranscriptionResponse:
    """
    Transcribe an audio file to text.

    Supported audio formats: WAV, MP3, OGG, FLAC, WebM, MP4, M4A

    The audio will be automatically converted to the format required by the
    transcription engine (16kHz, mono, 16-bit PCM).
    """
    # Validate file presence
    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is required",
        )

    # Read file content
    content = await audio.read()

    # Check file size
    max_size = settings.max_audio_size_mb * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file too large. Maximum size: {settings.max_audio_size_mb}MB",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty",
        )

    # Determine content type
    content_type = audio.content_type or "audio/wav"
    logger.info(
        f"Received audio file: {audio.filename}, "
        f"size: {len(content)} bytes, "
        f"content_type: {content_type}"
    )

    # Transcribe
    service = get_transcription_service()
    result = await service.transcribe(
        audio_bytes=content,
        content_type=content_type,
        language=language,
    )

    if not result.success:
        # Check if it's a connection error
        if "Cannot connect" in (result.error or ""):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.error,
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error or "Transcription failed",
        )

    return TranscriptionResponse(
        text=result.text,
        language=result.language,
        success=result.success,
        error=result.error,
    )


@router.post(
    "/synthesize",
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Piper server unavailable"},
    },
    summary="Convert text to speech",
    description="Convert text to speech audio using Piper TTS.",
)
async def synthesize_speech(request: TTSRequest) -> Response:
    """
    Convert text to speech.

    Returns a WAV audio file containing the synthesized speech.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty",
        )

    if len(request.text) > settings.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text too long. Maximum length: {settings.max_text_length} characters",
        )

    logger.info(f"TTS request: {len(request.text)} characters")

    service = get_tts_service()
    result = await service.synthesize(
        text=request.text,
        voice=request.voice,
        output_format="wav",
    )

    if not result.success:
        if "Cannot connect" in (result.error or ""):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.error,
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error or "TTS failed",
        )

    return Response(
        content=result.audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=speech.wav",
        },
    )


@router.post(
    "/chat",
    response_model=LLMResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "LLM server unavailable"},
    },
    summary="Chat with LLM",
    description="Send a text prompt to the LLM and get a response.",
)
async def chat_with_llm(request: LLMRequest) -> LLMResponse:
    """
    Send a prompt to the LLM and get a text response.
    """
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt cannot be empty",
        )

    logger.info(f"LLM chat request: {len(request.prompt)} characters")

    service = get_llm_service()
    result = await service.generate(
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
   
    if not result.success:
        if "Cannot connect" in (result.error or ""):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.error,
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error or "LLM generation failed",
        )

    return LLMResponse(
        text=result.text,
        success=result.success,
        model=result.model,
        error=result.error,
    )


@router.post(
    "/converse",
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Voice conversation pipeline",
    description="Full pipeline: Audio -> STT -> Intent Classification -> Action Execution -> Response Generation -> TTS -> Audio",
)
async def converse(
    audio: UploadFile = File(..., description="Audio file with user speech"),
    language: Optional[str] = Form(None, description="Language code for STT"),
    session_id: Optional[str] = Form(None, description="Session ID for conversation continuity"),
) -> Response:
    """
    Full voice conversation pipeline with 3-stage LLM processing:
    
    1. STT: Transcribe user audio to text
    2. STAGE 1 - Intent Classification: Classify intent and extract entities (LLM call #1)
    3. STAGE 2 - Action Execution: Execute API calls based on intent (no LLM)
    4. STAGE 3 - Response Generation: Generate natural language response (LLM call #2)
    5. TTS: Convert response to speech
    
    The response includes headers with the transcribed user text, assistant response, and session ID.
    """
    logger.info("=" * 80)
    logger.info("CONVERSATION PIPELINE START")
    logger.info("=" * 80)
    
    # Validate file
    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is required",
        )

    content = await audio.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty",
        )

    max_size = settings.max_audio_size_mb * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file too large. Maximum size: {settings.max_audio_size_mb}MB",
        )

    content_type = audio.content_type or "audio/wav"
    
    # Generate or use provided session ID
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"New session created: {session_id}")
    else:
        logger.info(f"Continuing session: {session_id}")
    
    logger.info(f"Audio received: {len(content)} bytes, content_type: {content_type}")

    # ==================== STEP 1: STT ====================
    logger.info("-" * 40)
    logger.info("STEP 1: Speech-to-Text")
    logger.info("-" * 40)
    
    stt_service = get_transcription_service()
    stt_result = await stt_service.transcribe(
        audio_bytes=content,
        content_type=content_type,
        language=language,
    )

    if not stt_result.success:
        logger.error(f"STT failed: {stt_result.error}")
        if "Cannot connect" in (stt_result.error or ""):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"STT service unavailable: {stt_result.error}",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {stt_result.error}",
        )

    user_text = stt_result.text
    logger.info(f"STT Result: '{user_text}'")

    if not user_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No speech detected in audio",
        )

    # ==================== Get Conversation History ====================
    conversation_manager = get_conversation_manager()
    conversation_history = await conversation_manager.get_conversation_context(session_id)
    logger.info(f"Retrieved {len(conversation_history)} messages from session history")

    # ==================== STEP 2: Intent Classification (LLM Call #1) ====================
    logger.info("-" * 40)
    logger.info("STEP 2: Intent Classification (LLM Call #1)")
    logger.info("-" * 40)
    
    intent_classifier = get_intent_classifier()
    intent = await intent_classifier.classify(
        user_text=user_text,
        conversation_history=conversation_history,
    )
    
    logger.info(f"Intent: {intent.intent_type}")
    logger.info(f"Entities: {intent.entities}")
    logger.info(f"Confidence: {intent.confidence}")

    # ==================== STEP 3: Action Execution (No LLM) ====================
    logger.info("-" * 40)
    logger.info("STEP 3: Action Execution")
    logger.info("-" * 40)
    
    action_executor = get_action_executor()
    action_result = await action_executor.execute(
        intent=intent,
        conversation_history=conversation_history,
    )
    
    logger.info(f"Action result success: {action_result.get('success')}")
    if action_result.get('error'):
        logger.warning(f"Action error: {action_result.get('error')}")

    # ==================== STEP 4: Response Generation (LLM Call #2) ====================
    logger.info("-" * 40)
    logger.info("STEP 4: Response Generation (LLM Call #2)")
    logger.info("-" * 40)
    
    response_generator = get_response_generator()
    
    if intent.intent_type == INTENT_CONVERSATION:
        # For conversation, use the conversation response generator
        suggested_response = intent.entities.get("response")
        generated = await response_generator.generate_conversation_response(
            user_text=user_text,
            suggested_response=suggested_response,
            conversation_history=conversation_history,
        )
    else:
        # For tool-based intents, generate response from action result
        generated = await response_generator.generate(
            intent_type=intent.intent_type,
            tool_result=action_result,
            original_query=user_text,
            entities=intent.entities,
        )
    
    assistant_text = generated.text
    logger.info(f"Generated response: '{assistant_text}'")

    if not assistant_text:
        assistant_text = "I'm sorry, I couldn't generate a response. Please try again."
        logger.warning("Empty response generated, using fallback")

    # ==================== Store Conversation History ====================
    try:
        await conversation_manager.add_message(session_id, "user", user_text)
        await conversation_manager.add_message(session_id, "assistant", assistant_text)
        logger.info("Conversation history updated")
    except Exception as e:
        logger.warning(f"Failed to store conversation history: {e}")

    # ==================== STEP 5: TTS ====================
    logger.info("-" * 40)
    logger.info("STEP 5: Text-to-Speech")
    logger.info("-" * 40)
    
    tts_service = get_tts_service()
    tts_result = await tts_service.synthesize(text=assistant_text)

    if not tts_result.success:
        logger.error(f"TTS failed: {tts_result.error}")
        if "Cannot connect" in (tts_result.error or ""):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"TTS service unavailable: {tts_result.error}",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS synthesis failed: {tts_result.error}",
        )

    logger.info(f"TTS generated {len(tts_result.audio_bytes)} bytes of audio")
    
    logger.info("=" * 80)
    logger.info("CONVERSATION PIPELINE COMPLETE")
    logger.info(f"User: '{user_text}'")
    logger.info(f"Assistant: '{assistant_text}'")
    logger.info("=" * 80)

    # Return audio with metadata in headers
    # Sanitize text for HTTP headers (ASCII only, no special chars)
    def sanitize_header(text: str) -> str:
        """Remove non-ASCII characters and normalize for HTTP headers."""
        import unicodedata
        # Normalize unicode (convert smart quotes to regular quotes, etc.)
        text = unicodedata.normalize('NFKD', text)
        # Encode to ASCII, ignoring non-ASCII chars
        return text.encode('ascii', 'ignore').decode('ascii').replace('\n', ' ')[:500]
    
    return Response(
        content=tts_result.audio_bytes,
        media_type="audio/wav",
        headers={
            "X-User-Text": sanitize_header(user_text),
            "X-Assistant-Text": sanitize_header(assistant_text),
            "X-Session-ID": session_id,
            "X-Intent": intent.intent_type,
            "X-Confidence": intent.confidence,
            "Content-Disposition": "attachment; filename=response.wav",
            "Access-Control-Expose-Headers": "X-User-Text, X-Assistant-Text, X-Session-ID, X-Intent, X-Confidence",
        },
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available LLM models",
    description="List all models available in Ollama.",
)
async def list_models() -> ModelsResponse:
    """List available LLM models."""
    service = get_llm_service()
    models = await service.list_models()
    return ModelsResponse(
        models=models,
        current_model=settings.ollama_model,
    )


@router.post(
    "/models/pull",
    summary="Pull an LLM model",
    description="Download/pull a model to Ollama. This may take several minutes.",
)
async def pull_model(model_name: Optional[str] = None):
    """Pull a model from Ollama registry."""
    service = get_llm_service()
    model_to_pull = model_name or settings.ollama_model

    logger.info(f"Pulling model: {model_to_pull}")
    success = await service.pull_model()

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pull model {model_to_pull}",
        )

    return {"status": "success", "model": model_to_pull}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health of the API and its dependencies.",
)
async def health_check() -> HealthResponse:
    """
    Check the health of the API.

    Returns the status of the API and whether all servers are reachable.
    """
    stt_service = get_transcription_service()
    tts_service = get_tts_service()
    llm_service = get_llm_service()
    conversation_manager = get_conversation_manager()

    whisper_healthy = await stt_service.health_check()
    piper_healthy = await tts_service.health_check()
    ollama_healthy = await llm_service.health_check()
    redis_healthy = await conversation_manager.health_check()

    all_healthy = whisper_healthy and piper_healthy and ollama_healthy and redis_healthy
    status_str = "healthy" if all_healthy else "degraded"
    
    if not redis_healthy:
        logger.warning("Redis is unavailable - session memory will not work")

    return HealthResponse(
        status=status_str,
        whisper_server=whisper_healthy,
        piper_server=piper_healthy,
        ollama_server=ollama_healthy,
    )


@router.post(
    "/session/new",
    summary="Create new conversation session",
    description="Generate a new session ID for conversation continuity.",
)
async def create_session():
    """Create a new conversation session and return the session ID."""
    session_id = str(uuid.uuid4())
    logger.info(f"Created new session: {session_id}")
    return {"session_id": session_id}


@router.delete(
    "/session/{session_id}",
    summary="Delete conversation session",
    description="Clear conversation history for a session.",
)
async def delete_session(session_id: str):
    """Delete a conversation session and its history."""
    conversation_manager = get_conversation_manager()
    success = await conversation_manager.delete_session(session_id)
    
    if success:
        logger.info(f"Deleted session: {session_id}")
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session",
        )


@router.get(
    "/session/{session_id}/history",
    summary="Get conversation history",
    description="Retrieve the conversation history for a session.",
)
async def get_session_history(session_id: str):
    """Get the conversation history for a session."""
    conversation_manager = get_conversation_manager()
    history = await conversation_manager.get_conversation_context(session_id)
    
    return {
        "session_id": session_id,
        "message_count": len(history),
        "messages": history,
    }
