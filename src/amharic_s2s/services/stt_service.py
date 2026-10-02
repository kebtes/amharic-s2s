"""
Pipecat STT Service implementation for Amharic CTC ASR (snapwre/hohe-asr-amharic).
"""

import asyncio
import time
from typing import Optional
import numpy as np
import torch
from loguru import logger

try:
    from pipecat.services.ai_services import STTService
    from pipecat.frames.frames import (
        Frame,
        AudioRawFrame,
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        InterruptionFrame,
    )
except ImportError:
    # Graceful fallback mock classes for environments without pipecat installed
    class Frame:
        pass
    class AudioRawFrame(Frame):
        def __init__(self, audio=b"", sample_rate=16000, num_channels=1):
            self.audio = audio
            self.sample_rate = sample_rate
            self.num_channels = num_channels
    class TranscriptionFrame(Frame):
        def __init__(self, text="", user_id="", timestamp=""):
            self.text = text
            self.user_id = user_id
            self.timestamp = timestamp
    class UserStartedSpeakingFrame(Frame): pass
    class UserStoppedSpeakingFrame(Frame): pass
    class InterruptionFrame(Frame): pass
    class STTService:
        def __init__(self, **kwargs):
            self._downstream = None
        async def push_frame(self, frame, direction=None):
            pass
        async def process_frame(self, frame, direction=None):
            pass


class AmharicSTTService(STTService):
    """
    Custom Pipecat STT service wrapping snapwre/hohe-asr-amharic.
    Buffers incoming raw audio frames during user speech and performs single-pass CTC inference
    when user stops speaking (or when forced).
    """

    def __init__(
        self,
        model,
        processor,
        device: Optional[torch.device] = None,
        sample_rate: int = 16000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.processor = processor
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.sample_rate = sample_rate
        self._audio_buffer = bytearray()
        self._is_speaking = False

    def reset_buffer(self):
        self._audio_buffer.clear()

    def transcribe_pcm(self, pcm_bytes: bytes) -> str:
        """Runs synchronous CTC transcription on raw 16-bit PCM bytes."""
        if not pcm_bytes:
            return ""

        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        waveform = audio_int16.astype(np.float32) / 32768.0

        if len(waveform) < self.sample_rate * 0.1:  # ignore tiny glitches (<100ms)
            return ""

        inputs = self.processor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0].strip()

        return transcription

    async def process_frame(self, frame: Frame, direction=None):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._is_speaking = True
            self.reset_buffer()
            await self.push_frame(frame, direction)

        elif isinstance(frame, AudioRawFrame):
            # Accumulate raw audio frames
            self._audio_buffer.extend(frame.audio)
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._is_speaking = False
            pcm_bytes = bytes(self._audio_buffer)
            self.reset_buffer()

            if pcm_bytes:
                loop = asyncio.get_running_loop()
                t0 = time.perf_counter()
                text = await loop.run_in_executor(None, self.transcribe_pcm, pcm_bytes)
                latency = time.perf_counter() - t0
                logger.debug(f"[ASR] Transcription in {latency*1000:.1f}ms: '{text}'")

                if text:
                    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    await self.push_frame(
                        TranscriptionFrame(text=text, user_id="user", timestamp=timestamp),
                        direction
                    )

            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            self.reset_buffer()
            self._is_speaking = False
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
