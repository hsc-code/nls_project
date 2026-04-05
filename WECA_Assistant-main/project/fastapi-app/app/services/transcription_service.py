"""
Transcription service using Wyoming protocol to communicate with faster-whisper.

The Wyoming protocol uses the following event sequence for transcription:
1. AudioStart - signals beginning of audio stream with format info
2. AudioChunk - sends raw PCM audio data in chunks
3. AudioStop - signals end of audio stream
4. Transcript - receives the transcribed text
"""

import asyncio
import io
import logging
from dataclasses import dataclass
from typing import Optional

from pydub import AudioSegment
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.asr import Transcribe, Transcript
from wyoming.client import AsyncClient
from wyoming.event import Event

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""

    text: str
    language: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class TranscriptionService:
    """
    Service for transcribing audio using the Wyoming protocol.

    Connects to a faster-whisper Wyoming server and sends audio for transcription.
    """

    def __init__(
        self,
        host: str = settings.whisper_host,
        port: int = settings.whisper_port,
        sample_rate: int = settings.audio_sample_rate,
        channels: int = settings.audio_channels,
        sample_width: int = settings.audio_sample_width,
    ):
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self._uri = f"tcp://{host}:{port}"

    def _convert_audio_to_pcm(self, audio_bytes: bytes, content_type: str) -> bytes:
        """
        Convert audio file to raw PCM format required by Wyoming protocol.

        Args:
            audio_bytes: Raw bytes of the audio file
            content_type: MIME type of the audio (e.g., 'audio/wav', 'audio/mp3')

        Returns:
            Raw PCM audio bytes in the required format (16kHz, mono, 16-bit)
        """
        # Determine format from content type
        format_map = {
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/wave": "wav",
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "audio/ogg": "ogg",
            "audio/flac": "flac",
            "audio/webm": "webm",
            "audio/mp4": "mp4",
            "audio/m4a": "m4a",
        }

        # Strip codec info from content type (e.g., "audio/webm;codecs=opus" -> "audio/webm")
        base_content_type = content_type.split(";")[0].strip()

        audio_format = format_map.get(base_content_type)
        if not audio_format:
            # Try to detect from content type suffix
            if "/" in base_content_type:
                audio_format = base_content_type.split("/")[-1]
            else:
                audio_format = "wav"  # Default fallback

        logger.debug(f"Converting audio from format: {audio_format}")

        # Load audio with pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)

        # Convert to required format: 16kHz, mono, 16-bit
        audio = audio.set_frame_rate(self.sample_rate)
        audio = audio.set_channels(self.channels)
        audio = audio.set_sample_width(self.sample_width)

        # Export as raw PCM
        return audio.raw_data

    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_bytes: Raw bytes of the audio file
            content_type: MIME type of the audio file
            language: Language code for transcription (optional)

        Returns:
            TranscriptionResult with the transcribed text or error
        """
        language = language or settings.default_language

        try:
            # Convert audio to PCM format
            pcm_audio = await asyncio.get_event_loop().run_in_executor(
                None, self._convert_audio_to_pcm, audio_bytes, content_type
            )

            logger.info(
                f"Converted audio: {len(pcm_audio)} bytes PCM "
                f"({len(pcm_audio) / (self.sample_rate * self.sample_width):.2f}s)"
            )

            # Connect to Wyoming server and transcribe
            async with AsyncClient.from_uri(self._uri) as client:
                # Send AudioStart event
                audio_start = AudioStart(
                    rate=self.sample_rate,
                    width=self.sample_width,
                    channels=self.channels,
                )
                await client.write_event(audio_start.event())
                logger.debug("Sent AudioStart event")

                # Send audio chunks (4096 bytes per chunk is typical)
                chunk_size = 4096
                for i in range(0, len(pcm_audio), chunk_size):
                    chunk_data = pcm_audio[i : i + chunk_size]
                    audio_chunk = AudioChunk(
                        rate=self.sample_rate,
                        width=self.sample_width,
                        channels=self.channels,
                        audio=chunk_data,
                    )
                    await client.write_event(audio_chunk.event())

                logger.debug(f"Sent {len(pcm_audio) // chunk_size + 1} audio chunks")

                # Send AudioStop event
                await client.write_event(AudioStop().event())
                logger.debug("Sent AudioStop event")

                # Wait for transcript
                transcript_text = ""
                timeout = 60.0  # 60 second timeout for transcription

                try:
                    async with asyncio.timeout(timeout):
                        while True:
                            event = await client.read_event()
                            if event is None:
                                logger.warning("Connection closed before transcript received")
                                break

                            if Transcript.is_type(event.type):
                                transcript = Transcript.from_event(event)
                                transcript_text = transcript.text
                                logger.info(f"Received transcript: {transcript_text[:100]}...")
                                break
                            else:
                                logger.debug(f"Received event type: {event.type}")

                except asyncio.TimeoutError:
                    return TranscriptionResult(
                        text="",
                        language=language,
                        success=False,
                        error=f"Transcription timeout after {timeout}s",
                    )

                return TranscriptionResult(
                    text=transcript_text.strip(),
                    language=language,
                    success=True,
                )

        except ConnectionRefusedError:
            error_msg = f"Cannot connect to Wyoming server at {self._uri}"
            logger.error(error_msg)
            return TranscriptionResult(
                text="",
                success=False,
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Transcription failed: {type(e).__name__}: {str(e)}"
            logger.exception(error_msg)
            return TranscriptionResult(
                text="",
                success=False,
                error=error_msg,
            )

    async def health_check(self) -> bool:
        """
        Check if the Wyoming server is reachable.

        Returns:
            True if server is reachable, False otherwise
        """
        try:
            async with asyncio.timeout(5.0):
                async with AsyncClient.from_uri(self._uri) as client:
                    # Just connecting successfully is enough
                    return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
