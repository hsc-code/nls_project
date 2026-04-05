"""
TTS service using Wyoming protocol to communicate with Piper.

The Wyoming protocol uses the following event sequence for TTS:
1. Synthesize - sends text to convert to speech
2. AudioStart - receives audio format info
3. AudioChunk - receives raw PCM audio data in chunks
4. AudioStop - signals end of audio stream
"""

import asyncio
import io
import logging
import wave
from dataclasses import dataclass
from typing import Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.tts import Synthesize

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result of a TTS operation."""

    audio_bytes: bytes
    sample_rate: int
    channels: int
    sample_width: int
    success: bool = True
    error: Optional[str] = None


class TTSService:
    """
    Service for text-to-speech using the Wyoming protocol.

    Connects to a Piper Wyoming server and sends text for synthesis.
    """

    def __init__(
        self,
        host: str = settings.piper_host,
        port: int = settings.piper_port,
    ):
        self.host = host
        self.port = port
        self._uri = f"tcp://{host}:{port}"

    def _pcm_to_wav(
        self, pcm_data: bytes, sample_rate: int, channels: int, sample_width: int
    ) -> bytes:
        """
        Convert raw PCM data to WAV format.

        Args:
            pcm_data: Raw PCM audio bytes
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            sample_width: Bytes per sample (2 for 16-bit)

        Returns:
            WAV file bytes
        """
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        wav_buffer.seek(0)
        return wav_buffer.read()

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        output_format: str = "wav",
    ) -> TTSResult:
        """
        Convert text to speech.

        Args:
            text: Text to convert to speech
            voice: Voice name (optional, uses server default)
            output_format: Output format ('wav' or 'pcm')

        Returns:
            TTSResult with audio bytes or error
        """
        if not text or not text.strip():
            return TTSResult(
                audio_bytes=b"",
                sample_rate=0,
                channels=0,
                sample_width=0,
                success=False,
                error="Text cannot be empty",
            )

        try:
            async with AsyncClient.from_uri(self._uri) as client:
                # Send Synthesize event with text
                synthesize = Synthesize(text=text.strip(), voice=voice)
                await client.write_event(synthesize.event())
                logger.debug(f"Sent Synthesize event for text: {text[:50]}...")

                # Collect audio chunks
                audio_chunks: list[bytes] = []
                sample_rate = 22050  # Piper default
                channels = 1
                sample_width = 2

                timeout = 60.0  # 60 second timeout

                try:
                    async with asyncio.timeout(timeout):
                        while True:
                            event = await client.read_event()
                            if event is None:
                                logger.warning("Connection closed before audio complete")
                                break

                            if AudioStart.is_type(event.type):
                                audio_start = AudioStart.from_event(event)
                                sample_rate = audio_start.rate
                                channels = audio_start.channels
                                sample_width = audio_start.width
                                logger.debug(
                                    f"AudioStart: rate={sample_rate}, "
                                    f"channels={channels}, width={sample_width}"
                                )

                            elif AudioChunk.is_type(event.type):
                                audio_chunk = AudioChunk.from_event(event)
                                audio_chunks.append(audio_chunk.audio)

                            elif AudioStop.is_type(event.type):
                                logger.debug("Received AudioStop")
                                break

                            else:
                                logger.debug(f"Received event type: {event.type}")

                except asyncio.TimeoutError:
                    return TTSResult(
                        audio_bytes=b"",
                        sample_rate=0,
                        channels=0,
                        sample_width=0,
                        success=False,
                        error=f"TTS timeout after {timeout}s",
                    )

                if not audio_chunks:
                    return TTSResult(
                        audio_bytes=b"",
                        sample_rate=0,
                        channels=0,
                        sample_width=0,
                        success=False,
                        error="No audio received from TTS server",
                    )

                # Combine all chunks
                pcm_audio = b"".join(audio_chunks)
                logger.info(
                    f"Synthesized audio: {len(pcm_audio)} bytes PCM "
                    f"({len(pcm_audio) / (sample_rate * sample_width):.2f}s)"
                )

                # Convert to requested format
                if output_format == "wav":
                    audio_bytes = self._pcm_to_wav(
                        pcm_audio, sample_rate, channels, sample_width
                    )
                else:
                    audio_bytes = pcm_audio

                return TTSResult(
                    audio_bytes=audio_bytes,
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_width=sample_width,
                    success=True,
                )

        except ConnectionRefusedError:
            error_msg = f"Cannot connect to Piper server at {self._uri}"
            logger.error(error_msg)
            return TTSResult(
                audio_bytes=b"",
                sample_rate=0,
                channels=0,
                sample_width=0,
                success=False,
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"TTS failed: {type(e).__name__}: {str(e)}"
            logger.exception(error_msg)
            return TTSResult(
                audio_bytes=b"",
                sample_rate=0,
                channels=0,
                sample_width=0,
                success=False,
                error=error_msg,
            )

    async def health_check(self) -> bool:
        """
        Check if the Piper server is reachable.

        Returns:
            True if server is reachable, False otherwise
        """
        try:
            async with asyncio.timeout(5.0):
                async with AsyncClient.from_uri(self._uri) as client:
                    return True
        except Exception as e:
            logger.warning(f"Piper health check failed: {e}")
            return False
