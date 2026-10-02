"""
Configuration dataclasses and default parameters for Amharic S2S pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ASRConfig:
    model_id: str = "snapwre/hohe-asr-amharic"
    sample_rate: int = 16000
    device: Optional[str] = None  # None = auto-detect cuda/cpu


@dataclass
class LLMConfig:
    model_id: str = "yosefw/gemma-2-2b-it-finetuned-amharic"
    fallback_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens: int = 64
    temperature: float = 0.3
    top_p: float = 0.9
    quantization: str = "4bit"  # "4bit", "8bit", or "none"
    system_prompt: str = "You are a helpful, respectful, and concise Amharic conversational assistant. Always respond fluently in Amharic script (Ge'ez)."
    device_map: str = "auto"


@dataclass
class TTSConfig:
    model_id: str = "gheero-Leyu/amharic-omnivoice-tts"
    sample_rate: int = 24000
    diffusion_steps: int = 16  # 16 steps provides optimal latency/quality balance for real-time
    baseline_steps: int = 32
    device: Optional[str] = None


@dataclass
class VADConfig:
    confidence_threshold: float = 0.5
    start_speech_ms: int = 150
    min_silence_duration_ms: int = 600
    sample_rate: int = 16000


@dataclass
class S2SConfig:
    experiment_name: str = "amharic_s2s_pipecat"
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    random_seed: int = 42
    output_dir: str = "results"


default_config = S2SConfig()
