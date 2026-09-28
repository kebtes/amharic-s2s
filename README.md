# Amharic Speech-to-Speech Baseline Research Pipeline

This repository contains the complete Google Colab research baseline notebook for measuring and optimizing end-to-end latency in an Amharic Speech-to-Speech (ASR → LLM → TTS) pipeline.

## Baseline Pipeline Architecture

```
[Audio Input] (WAV)
       ↓
[ASR] snapwre/hohe-asr-amharic (Single-pass CTC)
       ↓ Amharic Transcript
[LLM] b1n1yam/gemma-2-27b-amharic-alpaca-sft (4-bit NF4 Quantized)
       ↓ Amharic Response
[TTS] gheero-Leyu/amharic-omnivoice-tts (32 Diffusion Steps)
       ↓
[Audio Output] (24 kHz WAV)
```

## Primary Deliverable

* **[`Amharic_S2S_Baseline.ipynb`](file:///c:/Users/CompUser/Documents/VSCode%20files/gheero/amharic-s2s/Amharic_S2S_Baseline.ipynb)**: Executable Google Colab research notebook with 22 structured sections.

## Quickstart in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Amharic_S2S_Baseline.ipynb`.
3. Select **Runtime → Change runtime type → T4 GPU** (or A100/L4 GPU).
4. Run all cells from top to bottom (**Runtime → Run all**).

## Output Artifacts

Running the notebook automatically generates the following structured research directory:

```text
results/
├── raw_runs.jsonl          # Raw per-run structured timing records
├── summary.json            # Descriptive statistics (Mean, Median, Std, Min, Max, P90)
├── environment.json        # Hardware, VRAM, and library versions
├── model_config.json       # Documented model card metadata
├── benchmark_config.json   # 12-sample test dataset definitions
├── baseline_report.md      # Comprehensive research report
├── generated_audio/        # Synthesized output WAV files
└── figures/
    └── latency_breakdown.png  # 4-panel latency breakdown & distribution plots
```
