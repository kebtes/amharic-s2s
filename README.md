# 🇪🇹 Amharic Speech-to-Speech (S2S) Pipeline with Pipecat

A modular, real-time, bidirectional conversational voice agent for Amharic built with **[Pipecat](https://github.com/pipecat-ai/pipecat)**, Hugging Face Transformers, and OmniVoice.

---

## 🏗️ Architecture

```
[ User Mic / WebRTC / Audio File ]
               ↓
    [ Silero VAD / User Mute ]
               ↓
[ AmharicSTTService: snapwre/hohe-asr-amharic (CTC ASR) ]
               ↓ Amharic Transcript Frame
[ AmharicLLMService: yosefw/gemma-2-2b-it-finetuned-amharic (4-bit Streaming LLM) ]
               ↓ Streamed Token Frames
[ AmharicSentenceAggregator: Boundary Chunking (።, ?, !, \n) ]
               ↓ Sentence Text Frames
[ AmharicTTSService: gheero-Leyu/amharic-omnivoice-tts (Streaming PCM) ]
               ↓ 24 kHz PCM Audio Frames
[ WebRTC / Speaker Playback Output ]
```

---

## 📁 Project Structure

```text
amharic-s2s/
├── src/
│   └── amharic_s2s/
│       ├── __init__.py               # Package exports
│       ├── config.py                 # Hyperparameters & model configurations
│       ├── audio_utils.py            # Audio format conversions & resampling
│       ├── models/
│       │   ├── __init__.py
│       │   └── loader.py             # Model loaders (ASR, LLM, TTS) & VRAM tracking
│       ├── services/                 # Pipecat AI Service classes
│       │   ├── __init__.py
│       │   ├── stt_service.py        # AmharicSTTService (CTC ASR)
│       │   ├── llm_service.py        # AmharicLLMService (Streaming Gemma-2B)
│       │   └── tts_service.py        # AmharicTTSService (OmniVoice TTS)
│       └── pipeline/
│           ├── __init__.py
│           ├── builder.py            # Pipecat pipeline assembler & sentence aggregator
│           └── runner.py             # Interactive & benchmark runners
├── notebooks/
│   └── Amharic_S2S_Pipecat.ipynb     # Lightweight notebook for loading & testing
├── scripts/
│   ├── run_local.py                  # Live microphone/speaker CLI runner
│   └── benchmark_pipeline.py         # Automated TTFT, TTFA, and latency benchmarks
├── pyproject.toml                    # Package build & dependencies
├── requirements.txt                  # PIP requirements
└── README.md                         # Documentation
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kebtes/amharic-s2s.git
cd amharic-s2s

# Install dependencies and package in editable mode
pip install -e .
```

### 2. Run Interactive Live Voice Assistant (Mic & Speaker)

```bash
python scripts/run_local.py
```

### 3. Run Pipeline Latency Benchmark

```bash
python scripts/benchmark_pipeline.py
```

---

## 📓 Running in Google Colab / Jupyter

Open [`notebooks/Amharic_S2S_Pipecat.ipynb`](file:///c:/Users/CompUser/Documents/VSCode%20files/gheero/amharic-s2s/notebooks/Amharic_S2S_Pipecat.ipynb) (or [`Amharic_S2S_Pipecat.ipynb`](file:///c:/Users/CompUser/Documents/VSCode%20files/gheero/amharic-s2s/Amharic_S2S_Pipecat.ipynb)) in Google Colab. The notebook is purely a **thin loading harness**:

```python
from amharic_s2s import load_all_models, default_config
from amharic_s2s.pipeline.runner import run_pipeline_with_audio_file

# 1. Load models
bundle = load_all_models(default_config)

# 2. Run streamed S2S with audio
res = await run_pipeline_with_audio_file(bundle, "sample.wav")
print("Response:", res["llm_response"])
```
