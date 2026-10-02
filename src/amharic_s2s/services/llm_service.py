"""
Pipecat LLM Service for Amharic conversational generation with streaming token emission.
"""

import asyncio
import threading
import time
from typing import List, Dict, Optional
from loguru import logger
from transformers import TextIteratorStreamer

try:
    from pipecat.services.ai_services import LLMService
    from pipecat.frames.frames import (
        Frame,
        TextFrame,
        TranscriptionFrame,
        LLMResponseStartFrame,
        LLMResponseEndFrame,
        InterruptionFrame,
        CancelFrame,
    )
except ImportError:
    class Frame: pass
    class TextFrame(Frame):
        def __init__(self, text=""): self.text = text
    class TranscriptionFrame(Frame):
        def __init__(self, text="", user_id="", timestamp=""): self.text = text
    class LLMResponseStartFrame(Frame): pass
    class LLMResponseEndFrame(Frame): pass
    class InterruptionFrame(Frame): pass
    class CancelFrame(Frame): pass
    class LLMService:
        def __init__(self, **kwargs): pass
        async def push_frame(self, frame, direction=None): pass
        async def process_frame(self, frame, direction=None): pass


class AmharicLLMService(LLMService):
    """
    Pipecat LLM service wrapping fine-tuned Amharic Gemma 2B or Qwen 2.5.
    Streams output tokens asynchronously using HuggingFace TextIteratorStreamer.
    """

    def __init__(
        self,
        model,
        tokenizer,
        system_prompt: str = "You are a helpful Amharic assistant. Respond politely and concisely in Amharic.",
        max_new_tokens: int = 64,
        temperature: float = 0.3,
        top_p: float = 0.9,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self._current_task: Optional[asyncio.Task] = None
        self._interrupted = False

    def reset_conversation(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def _stream_llm_response(self, user_text: str, direction=None):
        self._interrupted = False
        self.messages.append({"role": "user", "content": user_text})

        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            prompt = f"{self.system_prompt}\nUser: {user_text}\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0.0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        # Run model.generate in a background thread
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        await self.push_frame(LLMResponseStartFrame(), direction)

        accumulated_text = []
        loop = asyncio.get_running_loop()

        def get_next_token():
            try:
                return next(streamer)
            except StopIteration:
                return None

        t_start = time.perf_counter()
        first_token_emitted = False

        while not self._interrupted:
            token = await loop.run_in_executor(None, get_next_token)
            if token is None:
                break

            if not first_token_emitted:
                ttft = time.perf_counter() - t_start
                logger.debug(f"[LLM] Time-To-First-Token: {ttft*1000:.1f}ms")
                first_token_emitted = True

            accumulated_text.append(token)
            await self.push_frame(TextFrame(text=token), direction)

        full_reply = "".join(accumulated_text).strip()
        if full_reply:
            self.messages.append({"role": "assistant", "content": full_reply})
            logger.debug(f"[LLM] Completed response ({len(accumulated_text)} chunks): '{full_reply}'")

        await self.push_frame(LLMResponseEndFrame(), direction)

    async def process_frame(self, frame: Frame, direction=None):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text.strip()
            if user_text:
                if self._current_task and not self._current_task.done():
                    self._interrupted = True
                    self._current_task.cancel()

                self._current_task = asyncio.create_task(
                    self._stream_llm_response(user_text, direction)
                )

        elif isinstance(frame, (InterruptionFrame, CancelFrame)):
            self._interrupted = True
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
