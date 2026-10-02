# Amharic Speech-to-Speech Baseline Research Report

## 1. Executive Summary
Sequential baseline using:
- ASR: `snapwre/hohe-asr-amharic`
- LLM: `google/gemma-2-2b-it` (float16 on GPU)
- TTS: `gheero-Leyu/amharic-omnivoice-tts` (32 diffusion steps)

Dataset: 12 real Amharic speech samples selected from a larger candidate pool using duration stratification. Completed 60 of 60 planned runs after warm-up.

This benchmark is sequential. TTS latency means speech-end to complete waveform availability; true first-audio latency is not measured.

## 2. Environment
- GPU: Tesla T4 (14.56 GB VRAM)
- RAM: 12.67 GB
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Python: 3.13.15

## 3. Results
| Metric | Result |
|---|---:|
| ASR preprocessing | 0.1802s median (p90 0.2613s) |
| ASR inference | 2.3611s median (p90 3.1751s) |
| ASR decoding | 0.0016s median (p90 0.0026s) |
| ASR total | 2.5747s median (p90 3.3681s) |
| ASR RTF | 0.0586s median (p90 0.0649s) |
| ASR WER | 0.2177s median (p90 0.3000s) |
| ASR CER | 0.0582s median (p90 0.0729s) |
| LLM first streamed text | 0.1491s median (p90 0.1547s) |
| LLM generation | 3.4447s median (p90 4.0285s) |
| LLM tokens/sec | 18.5800s median (p90 19.6220s) |
| TTS generation | 2.4890s median (p90 3.6969s) |
| TTS RTF | 0.2114s median (p90 0.2236s) |
| E2E speech-end → complete audio | 9.0220s median (p90 10.2350s) |

## 4. Bottleneck
The largest measured median stage is **LLM generation**, accounting for approximately 40.5% of the sum of median ASR, LLM-generation, and TTS stage latencies. This is a descriptive baseline finding, not an optimization conclusion.

## 5. ASR Quality
WER/CER are reference-based sanity metrics using the dataset transcripts after minimal Unicode NFC and whitespace normalization. They are not presented as a formal held-out model evaluation.

## 6. Limitations
- Only 12 speech samples were benchmarked.
- Repeated runs are not independent speakers.
- The pipeline is sequential, not true streaming.
- TTS first-audio latency is unavailable from the current blocking generation API.
- WER/CER depend on the supplied reference transcripts.
- Hardware, quantization, offloading, and software versions affect latency.

## 7. Next Step
Use this frozen baseline to choose one targeted optimization experiment based on the measured bottleneck. Optimization is intentionally separated from this notebook.
