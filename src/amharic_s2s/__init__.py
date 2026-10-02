"""
Amharic Speech-to-Speech (S2S) Pipecat Package.
"""

__version__ = "0.1.0"

from amharic_s2s.config import S2SConfig, default_config
from amharic_s2s.models.loader import ModelBundle, load_all_models
from amharic_s2s.pipeline.builder import build_amharic_pipeline

__all__ = [
    "S2SConfig",
    "default_config",
    "ModelBundle",
    "load_all_models",
    "build_amharic_pipeline",
]
