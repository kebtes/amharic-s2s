"""
Pipecat TTS Service for Amharic Speech Synthesis using OmniVoice.
"""

import asyncio
import time
from typing import Optional
import numpy as np
import torch
from loguru import logger

from amharic_s2s.audio_utils import float32_to_pcm

try:
    from pipecat.services.ai_services import TTSService
    from pipecat.frames.frames import (
        Frame,
        TextFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
        TTSAudioRawFrame,
        AudioRawFrame,
        InterruptionFrame,
        CancelFrame,
    )
except ImportError:
    class Frame: pass
    class TextFrame(Frame):
        def __init__(self, text=""): self.text = text
    class TTSStartedFrame(Frame): pass
    class TTSStoppedFrame(Frame): pass
    class TTSAudioRawFrame(Frame):
        def __init__(self, audio=b"", sample_rate=24000, num_channels=1):
            self.audio = audio
            self.sample_rate = sample_rate
            self.num_channels = num_channels
    class AudioRawFrame(TTSAudioRawFrame): pass
    class InterruptionFrame(Frame): pass
    class CancelFrame(Frame): pass
    class TTSService:
        def __init__(self, **kwargs): pass
        async def push_frame(self, frame, direction=None): pass
        async def process_frame(self, frame, direction=None): pass


class AmharicTTSService(TTSService):
    """
    Pipecat TTS service wrapping gheero-Leyu/amharic-omnivoice-tts.
    Converts aggregated Amharic text frames into streaming PCM audio frames.
    """

    def __init__(
        self,
        tts_model,
        diffusion_steps: int = 16,
        sample_rate: int = 24000,
        chunk_size_samples: int = 2400,  # 100ms per audio packet
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tts_model = tts_model
        self.diffusion_steps = diffusion_steps
        self.sample_rate = sample_rate
        self.chunk_size_samples = chunk_size_samples
        self._interrupted = False

    def synthesize_wav(self, text: str) -> np.ndarray:
        """Synchronous synthesis call to OmniVoice."""
        if not text.strip():
            return np.array([], dtype=np.float32)

        try:
            with torch.no_grad():
                out = self.tts_model.generate(
                    text=text,
                    num_steps=self.diffusion_steps,
                )
                if isinstance(out, list):
                    audio = out[0]
                elif hasattr(out, "cpu"):
                    audio = out.cpu().numpy()
                else:
                    audio = np.array(out)
                
                if isinstance(audio, np.ndarray) and audio.ndim > 1:
                    audio = audio.squeeze()
                return audio.astype(np.float32)
        except Exception as e:
            logger.error(f"[TTS] Synthesis failed for '{text}': {e}")
            return np.array([], dtype=np.float32)

    async def _process_text_and_stream_audio(self, text: str, direction=None):
        self._interrupted = False
        text = text.strip()
        if not text:
            return

        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()

        # Generate audio array in worker thread
        audio_array = await loop.run_in_executor(None, self.synthesize_wav, text)
        tts_latency = time.perf_counter() - t0

        if len(audio_array) == 0 or self._interrupted:
            return

        duration_sec = len(audio_array) / self.sample_rate
        logger.debug(
            f"[TTS] Synthesized {duration_sec:.2f}s audio in {tts_latency*1000:.1f}ms "
            f"(RTF: {tts_latency/max(duration_sec, 0.001):.2f}) for '{text}'"
        )

        await self.push_frame(TTSStartedFrame(), direction)

        # Chunk into streaming frames
        pcm_data = float32_to_pcm(audio_array)
        bytes_per_sample = 2  # 16-bit PCM
        chunk_bytes = self.chunk_size_samples * bytes_per_sample

        for i in range(0, len(pcm_data), chunk_bytes):
            if self._interrupted:
                logger.info("[TTS] Synthesis playback interrupted by user barge-in.")
                break
            chunk = pcm_data[i : i + chunk_bytes]
            try:
                frame = TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1
                )
            except Exception:
                frame = AudioRawFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1
                )
            await self.push_frame(frame, direction)
            # Yield control briefly to ensure smooth async pipeline execution
            await asyncio.sleep(0.001)

        await self.push_frame(TTSStoppedFrame(), direction)

    async def process_frame(self, frame: Frame, direction=None):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self._process_text_and_stream_audio(frame.text, direction)

        elif isinstance(frame, (InterruptionFrame, CancelFrame)):
            self._interrupted = True
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
