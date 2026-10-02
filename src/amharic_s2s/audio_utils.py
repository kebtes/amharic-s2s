"""
Audio utilities for resampling, format conversion, and frame packaging.
"""

import io
from typing import Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torchaudio


def pcm_to_float32(pcm_data: bytes, dtype=np.int16) -> np.ndarray:
    """Convert raw PCM byte stream to normalized float32 numpy array (-1.0 to 1.0)."""
    audio_int = np.frombuffer(pcm_data, dtype=dtype)
    return audio_int.astype(np.float32) / 32768.0


def float32_to_pcm(audio: np.ndarray, dtype=np.int16) -> bytes:
    """Convert normalized float32 numpy array to raw PCM byte stream."""
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int = (audio_clipped * 32767.0).astype(dtype)
    return audio_int.tobytes()


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D float32 audio numpy array from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return audio
    tensor = torch.from_numpy(audio).unsqueeze(0).float()
    resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()


def load_audio_file(file_path_or_bytes: Union[str, bytes], target_sr: int = 16000) -> Tuple[np.ndarray, int, float]:
    """
    Loads audio from file path or bytes, converts to mono, resamples to target_sr.
    Returns: (audio_array, sample_rate, duration_seconds)
    """
    if isinstance(file_path_or_bytes, bytes):
        data, sr = sf.read(io.BytesIO(file_path_or_bytes), dtype="float32")
    else:
        data, sr = sf.read(file_path_or_bytes, dtype="float32")

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != target_sr:
        data = resample_audio(data, sr, target_sr)
        sr = target_sr

    duration = float(len(data)) / float(sr)
    return data, sr, duration


def save_audio_file(file_path: str, audio: np.ndarray, sample_rate: int = 24000) -> str:
    """Saves float32 audio to a WAV file."""
    sf.write(file_path, audio, sample_rate, subtype="PCM_16")
    return file_path
