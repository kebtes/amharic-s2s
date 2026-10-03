# Amharic Bank Document RAG System - Teammate Handoff Guide

## Overview

This documentation explains how to integrate the **Amharic Bank Document RAG System** into the overall Speech-to-Speech (S2S) low-latency conversational pipeline.

The RAG system parses Bank of Abyssinia FAQ files and general banking text documents, chunks them into semantically meaningful blocks, and performs ultra-fast hybrid retrieval (< 15ms) combining **FAISS Dense Vector Search** and **BM25 Sparse Keyword Search** with Reciprocal Rank Fusion (RRF).

---

## Architecture Flow

```text
[User Audio Input] (WAV)
       ↓
[ASR] snapwre/hohe-asr-amharic
       ↓ Amharic Transcript
[RAG Engine] AmharicBankRAG (Hybrid FAISS + BM25) <--- Reads 'vector_store/' (82 Banking Docs)
       ↓ Grounded Amharic Prompt & Bank Context (< 15ms)
[LLM] yosefw/gemma-2-2b-it-finetuned-amharic (Streaming Tokens)
       ↓ Amharic Text Stream
[TTS] gheero-Leyu/amharic-omnivoice-tts
       ↓
[Audio Output] (24 kHz WAV)
```

---

## Key Modules in `rag/`

| File | Purpose |
|---|---|
| [`rag/document_loader.py`](file:///home/vini/01-projects/work-01-gheero-projects/phase-2/amharic-s2s/amharic-s2s/rag/document_loader.py) | Parses `.docx` and `.pdf` files from `Amharic FAQ/` and `amharic text/`. |
| [`rag/text_processor.py`](file:///home/vini/01-projects/work-01-gheero-projects/phase-2/amharic-s2s/amharic-s2s/rag/text_processor.py) | Ethiopic text cleaning, sentence splitting, and Question-Answer pair extraction. |
| [`rag/embeddings.py`](file:///home/vini/01-projects/work-01-gheero-projects/phase-2/amharic-s2s/amharic-s2s/rag/embeddings.py) | Multilingual vector embeddings using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. |
| [`rag/retriever.py`](file:///home/vini/01-projects/work-01-gheero-projects/phase-2/amharic-s2s/amharic-s2s/rag/retriever.py) | Hybrid search combining FAISS Inner Product & BM25 with RRF scoring. |
| [`rag/rag_engine.py`](file:///home/vini/01-projects/work-01-gheero-projects/phase-2/amharic-s2s/amharic-s2s/rag/rag_engine.py) | Main `AmharicBankRAG` class for query retrieval and grounded prompt formatting. |
| [`rag/build_index.py`](file:///home/vini/01-projects/work-01-gheero-projects/phase-2/amharic-s2s/amharic-s2s/rag/build_index.py) | CLI script to build and serialize vector indices to `vector_store/`. |
| [`rag/eval_rag.py`](file:///home/vini/01-projects/work-01-gheero-projects/phase-2/amharic-s2s/amharic-s2s/rag/eval_rag.py) | Measures retrieval latency (P50, P90, Mean) and context stats. |
| [`rag/server.py`](file:///home/vini/01-projects/work-01-gheero-projects/phase-2/amharic-s2s/amharic-s2s/rag/server.py) | FastAPI service exposing `/retrieve` endpoint for REST integration. |

---

## How to Integrate with your S2S Pipeline

### Option A: In-Process Python Import (Recommended for Lowest Latency < 15ms)

Import `AmharicBankRAG` directly in your S2S pipeline script:

```python
from rag.rag_engine import AmharicBankRAG

# 1. Initialize RAG engine once at model load time
rag = AmharicBankRAG(index_dir="vector_store", top_k=3)

# 2. When ASR returns Amharic transcript:
asr_transcript = "የሞባይል ባንኪንግ አገልግሎት ለመጠቀም ምን ያስፈልጋል?"

# 3. Fetch context and formatted LLM prompt
rag_result = rag.query(asr_transcript)

# 4. Pass prompt to LLM streaming generator
llm_input_prompt = rag_result["formatted_prompt"]
print("Retrieval Latency (ms):", rag_result["latency_ms"])
print("LLM Input Prompt:\n", llm_input_prompt)
```

---

### Option B: FastAPI REST Microservice

1. **Start Server**:
   ```bash
   ./.venv/bin/python rag/server.py
   ```

2. **Send POST Request from S2S Service**:
   - Endpoint: `POST http://localhost:8000/retrieve`
   - Request Body:
     ```json
     {
       "query": "የካርድ አገልግሎት ክፍያ ስንት ነው?",
       "top_k": 3
     }
     ```
   - Response Body:
     ```json
     {
       "query": "የካርድ አገልግሎት ክፍያ ስንት ነው?",
       "context": "[1] ... \n\n [2] ...",
       "formatted_prompt": "አንተ የባንክ መረጃ ረዳት ነህ...",
       "sources": ["Card Product Features FAQ.pdf"],
       "latency_ms": 12.4,
       "chunks": [...]
     }
     ```

---

## Grounded Amharic System Prompt Template

The RAG engine automatically formats queries into the following grounded prompt template to prevent LLM hallucinations:

```text
አንተ የባንክ መረጃ ረዳት ነህ። ከታች የተሰጠውን የባንክ ሰነድ መረጃ ብቻ በመጠቀም ለደንበኛው ጥያቄ ግልጽ፣ አጭር እና ትክክለኛ ምላሽ ስጥ። 
ሰነዱ ውስጥ የሌለ መረጃ በፍጹም አትጨምር።

[የተገኘ የባንክ መረጃ]:
{context}

[የደንበኛ ጥያቄ]:
{query}

[ምላሽ]:
```

---

## Index Maintenance

If new bank `.docx` or `.pdf` documents are added to `Amharic FAQ/` or `amharic text/`, update the index by running:

```bash
./.venv/bin/python rag/build_index.py
```
