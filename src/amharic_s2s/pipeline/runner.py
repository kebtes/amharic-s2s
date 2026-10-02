"""
Pipeline execution runners for both interactive live audio and batch file processing.
"""

import asyncio
import time
from typing import Dict, Any, Optional
import numpy as np
from loguru import logger

from amharic_s2s.models.loader import ModelBundle
from amharic_s2s.audio_utils import load_audio_file, float32_to_pcm, pcm_to_float32
from amharic_s2s.services.stt_service import AmharicSTTService
from amharic_s2s.services.llm_service import AmharicLLMService
from amharic_s2s.services.tts_service import AmharicTTSService

try:
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.transports.local.audio import LocalAudioTransport
    from pipecat.audio.vad.silero import SileroVADAnalyzer
except ImportError:
    class PipelineRunner:
        async def run(self, task):
            pass
    class LocalAudioTransport:
        def __init__(self, **kwargs): pass
        def input(self): return None
        def output(self): return None
    class SileroVADAnalyzer:
        def __init__(self, **kwargs): pass


async def run_pipeline_with_audio_file(
    bundle: ModelBundle,
    audio_path_or_array: Any,
    orig_sr: int = 16000,
    diffusion_steps: int = 16
) -> Dict[str, Any]:
    """
    Executes an end-to-end streamed S2S run given an audio file or array,
    tracking detailed stage latencies (ASR, TTFT, TTFA, Total S2S Latency).
    """
    if isinstance(audio_path_or_array, str):
        audio_np, sr, duration = load_audio_file(audio_path_or_array, target_sr=16000)
    elif isinstance(audio_path_or_array, np.ndarray):
        audio_np = audio_path_or_array
        duration = len(audio_np) / orig_sr
    else:
        raise ValueError("Unsupported audio input type")

    pcm_bytes = float32_to_pcm(audio_np)
    metrics: Dict[str, Any] = {
        "input_duration_sec": duration,
        "asr_latency_sec": 0.0,
        "ttft_sec": 0.0,
        "ttfa_sec": 0.0,
        "total_latency_sec": 0.0,
        "transcript": "",
        "llm_response": "",
        "audio_out": None,
        "output_sr": bundle.config.tts.sample_rate,
    }

    t_start = time.perf_counter()

    # 1. ASR Stage
    stt_service = AmharicSTTService(
        model=bundle.asr.model,
        processor=bundle.asr.processor,
        device=bundle.asr.device,
        sample_rate=bundle.config.asr.sample_rate,
    )
    t_asr_start = time.perf_counter()
    loop = asyncio.get_running_loop()
    transcript = await loop.run_in_executor(None, stt_service.transcribe_pcm, pcm_bytes)
    t_asr_end = time.perf_counter()
    metrics["asr_latency_sec"] = t_asr_end - t_asr_start
    metrics["transcript"] = transcript

    if not transcript:
        logger.warning("ASR produced an empty transcript.")
        metrics["total_latency_sec"] = time.perf_counter() - t_start
        return metrics

    # 2. LLM & TTS Streamed Execution
    llm_service = AmharicLLMService(
        model=bundle.llm.model,
        tokenizer=bundle.llm.tokenizer,
        system_prompt=bundle.config.llm.system_prompt,
        max_new_tokens=bundle.config.llm.max_new_tokens,
        temperature=bundle.config.llm.temperature,
    )

    tts_service = AmharicTTSService(
        tts_model=bundle.tts.model,
        diffusion_steps=diffusion_steps,
        sample_rate=bundle.config.tts.sample_rate,
    )

    # Stream LLM tokens and feed first sentence immediately to TTS
    t_llm_start = time.perf_counter()
    from transformers import TextIteratorStreamer
    import threading

    prompt = f"{bundle.config.llm.system_prompt}\nUser: {transcript}\nAssistant:"
    inputs = bundle.llm.tokenizer(prompt, return_tensors="pt").to(bundle.llm.model.device)
    streamer = TextIteratorStreamer(bundle.llm.tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=bundle.config.llm.max_new_tokens,
        temperature=bundle.config.llm.temperature,
        top_p=bundle.config.llm.top_p,
        do_sample=bundle.config.llm.temperature > 0.0,
    )

    thread = threading.Thread(target=bundle.llm.model.generate, kwargs=gen_kwargs)
    thread.start()

    collected_tokens = []
    generated_audio_chunks = []
    first_token_time = None
    first_audio_time = None

    def get_token():
        try:
            return next(streamer)
        except StopIteration:
            return None

    current_sentence_buffer = ""
    while True:
        token = await loop.run_in_executor(None, get_token)
        if token is None:
            break

        if first_token_time is None:
            first_token_time = time.perf_counter()
            metrics["ttft_sec"] = first_token_time - t_llm_start

        collected_tokens.append(token)
        current_sentence_buffer += token

        # If sentence ends (Amharic full stop or punctuation)
        if any(punct in current_sentence_buffer for punct in ["።", "?", "!", "\n", "."]):
            sentence_to_synth = current_sentence_buffer.strip()
            current_sentence_buffer = ""
            if sentence_to_synth:
                t_synth_start = time.perf_counter()
                audio_chunk = await loop.run_in_executor(None, tts_service.synthesize_wav, sentence_to_synth)
                if first_audio_time is None and len(audio_chunk) > 0:
                    first_audio_time = time.perf_counter()
                    metrics["ttfa_sec"] = first_audio_time - t_start
                if len(audio_chunk) > 0:
                    generated_audio_chunks.append(audio_chunk)

    # Flush any remaining text in buffer
    if current_sentence_buffer.strip():
        audio_chunk = await loop.run_in_executor(None, tts_service.synthesize_wav, current_sentence_buffer.strip())
        if first_audio_time is None and len(audio_chunk) > 0:
            first_audio_time = time.perf_counter()
            metrics["ttfa_sec"] = first_audio_time - t_start
        if len(audio_chunk) > 0:
            generated_audio_chunks.append(audio_chunk)

    metrics["llm_response"] = "".join(collected_tokens).strip()
    metrics["total_latency_sec"] = time.perf_counter() - t_start

    if generated_audio_chunks:
        metrics["audio_out"] = np.concatenate(generated_audio_chunks)
    else:
        metrics["audio_out"] = np.array([], dtype=np.float32)

    return metrics


async def run_interactive_pipeline(bundle: ModelBundle):
    """
    Runs the live microphone + speaker interactive voice agent using LocalAudioTransport and Silero VAD.
    """
    logger.info("Initializing Local Microphone + Speaker Transport with Silero VAD...")
    vad = SileroVADAnalyzer()
    transport = LocalAudioTransport(vad_analyzer=vad)

    from amharic_s2s.pipeline.builder import build_amharic_pipeline
    task = build_amharic_pipeline(
        bundle=bundle,
        transport_input=transport.input(),
        transport_output=transport.output(),
    )

    runner = PipelineRunner()
    logger.success("Amharic Voice Assistant is LIVE! Speak into your microphone in Amharic...")
    await runner.run(task)
