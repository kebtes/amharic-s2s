"""
Pipeline construction and assembly for Amharic Voice Bot.
"""

import re
from typing import Any, List, Optional
from loguru import logger

from amharic_s2s.models.loader import ModelBundle
from amharic_s2s.services.stt_service import AmharicSTTService
from amharic_s2s.services.llm_service import AmharicLLMService
from amharic_s2s.services.tts_service import AmharicTTSService

try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    from pipecat.processors.aggregators.sentence import SentenceAggregator
    from pipecat.frames.frames import Frame, TextFrame, LLMResponseEndFrame
    from pipecat.processors.frame_processor import FrameProcessor
except ImportError:
    class Frame: pass
    class TextFrame(Frame):
        def __init__(self, text=""): self.text = text
    class LLMResponseEndFrame(Frame): pass
    class FrameProcessor:
        def __init__(self): pass
        async def push_frame(self, frame, direction=None): pass
        async def process_frame(self, frame, direction=None): pass
    class Pipeline:
        def __init__(self, processors): self.processors = processors
    class PipelineTask:
        def __init__(self, pipeline, params=None):
            self.pipeline = pipeline
            self.params = params
    class PipelineParams:
        def __init__(self, **kwargs): pass


class AmharicSentenceAggregator(FrameProcessor):
    """
    Aggregates streamed tokens into full Amharic sentences before forwarding to TTS.
    Recognizes Amharic full stop ('።'), question mark ('?'), exclamation mark ('!'),
    standard period ('.'), and newline ('\n').
    """

    def __init__(self, min_chars: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.min_chars = min_chars
        self._buffer = ""
        # Amharic punctuation and standard sentence endings
        self._sentence_end_regex = re.compile(r'([።!?.\n]+)')

    async def process_frame(self, frame: Frame, direction=None):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            self._buffer += frame.text
            parts = self._sentence_end_regex.split(self._buffer)
            
            # If parts has completed sentences
            if len(parts) > 1:
                # Re-join sentence with its punctuation
                complete_sentence = "".join(parts[:-1]).strip()
                self._buffer = parts[-1]  # Remainder
                
                if len(complete_sentence) >= self.min_chars:
                    await self.push_frame(TextFrame(text=complete_sentence), direction)
                else:
                    self._buffer = complete_sentence + " " + self._buffer

        elif isinstance(frame, LLMResponseEndFrame):
            remainder = self._buffer.strip()
            self._buffer = ""
            if remainder:
                await self.push_frame(TextFrame(text=remainder), direction)
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)


def build_amharic_pipeline(
    bundle: ModelBundle,
    transport_input=None,
    transport_output=None,
    diffusion_steps: int = 16,
    enable_interruptions: bool = True,
) -> Any:
    """
    Builds and wires up the end-to-end Pipecat pipeline:
    Transport Input -> Amharic ASR -> Amharic LLM -> Amharic Sentence Aggregator -> Amharic TTS -> Transport Output
    """
    logger.info("Assembling Amharic Pipecat Pipeline...")

    # 1. Initialize Services
    stt_service = AmharicSTTService(
        model=bundle.asr.model,
        processor=bundle.asr.processor,
        device=bundle.asr.device,
        sample_rate=bundle.config.asr.sample_rate,
    )

    llm_service = AmharicLLMService(
        model=bundle.llm.model,
        tokenizer=bundle.llm.tokenizer,
        system_prompt=bundle.config.llm.system_prompt,
        max_new_tokens=bundle.config.llm.max_new_tokens,
        temperature=bundle.config.llm.temperature,
        top_p=bundle.config.llm.top_p,
    )

    sentence_aggregator = AmharicSentenceAggregator()

    tts_service = AmharicTTSService(
        tts_model=bundle.tts.model,
        diffusion_steps=diffusion_steps or bundle.config.tts.diffusion_steps,
        sample_rate=bundle.config.tts.sample_rate,
    )

    # 2. Build Pipeline Chain
    processors: List[Any] = []
    if transport_input is not None:
        processors.append(transport_input)

    processors.extend([
        stt_service,
        llm_service,
        sentence_aggregator,
        tts_service,
    ])

    if transport_output is not None:
        processors.append(transport_output)

    pipeline = Pipeline(processors)
    logger.success(f"Pipeline assembled with {len(processors)} stages.")

    params = PipelineParams(
        allow_interruptions=enable_interruptions,
        enable_metrics=True,
    )

    return PipelineTask(pipeline, params=params)
