# Amharic Speech-to-Speech Baseline Research Pipeline

This repository contains the complete Google Colab research baseline notebook for measuring and optimizing end-to-end latency in an Amharic Speech-to-Speech (ASR → LLM → TTS) pipeline.

## Baseline Pipeline Architecture

```
[Audio Input] (WAV)
       ↓
[ASR] snapwre/hohe-asr-amharic (Single-pass CTC)
       ↓ Amharic Transcript
[RAG Engine] AmharicBankRAG (Hybrid FAISS + BM25, < 15ms Latency)
       ↓ Grounded Amharic Prompt & Bank Context
[LLM] yosefw/gemma-2-2b-it-finetuned-amharic (2.6B SFT, Low-Latency)
       ↓ Amharic Response
[TTS] gheero-Leyu/amharic-omnivoice-tts (32 Diffusion Steps)
       ↓
[Audio Output] (24 kHz WAV)
```

## Amharic Bank RAG System

The RAG module (`rag/`) indexes **82 banking documents** (44 FAQ documents and 38 general text guides) to provide low-latency, grounded knowledge retrieval for bank domain queries.

### Quickstart for RAG System

1. **Build Vector Store Index**:
   ```bash
   python rag/build_index.py
   ```
2. **Run Latency & Retrieval Benchmarks**:
   ```bash
   python rag/eval_rag.py
   ```
3. **Start FastAPI Service**:
   ```bash
   python rag/server.py
   ```
4. **Detailed Integration & Handoff Docs**: See [`HANDOFF_RAG.md`](file:///home/vini/01-projects/work-01-gheero-projects/phase-2/amharic-s2s/amharic-s2s/HANDOFF_RAG.md) for full Python API and REST endpoint details.

## Primary Deliverables

* **[`Amharic_S2S_Baseline.ipynb`](file:///c:/Users/CompUser/Documents/VSCode%20files/gheero/amharic-s2s/Amharic_S2S_Baseline.ipynb)**: Executable Google Colab research notebook.
* **[`HANDOFF_RAG.md`](file:///home/vini/01-projects/work-01-gheero-projects/phase-2/amharic-s2s/amharic-s2s/HANDOFF_RAG.md)**: Teammate handoff documentation for RAG S2S integration.

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
