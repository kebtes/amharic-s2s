import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_engine import AmharicBankRAG

app = FastAPI(
    title="Amharic Bank Document RAG API",
    description="Low-Latency RAG service for Speech-to-Speech (S2S) Amharic conversational AI",
    version="1.0.0"
)

# Global RAG engine instance
rag_engine: Optional[AmharicBankRAG] = None


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    query: str
    context: str
    formatted_prompt: str
    sources: List[str]
    latency_ms: float
    chunks: List[Dict[str, Any]]


@app.on_event("startup")
def startup_event():
    global rag_engine
    index_dir = os.getenv("VECTOR_STORE_DIR", "vector_store")
    print(f"Initializing Amharic RAG engine from '{index_dir}'...")
    if not os.path.exists(index_dir):
        print(f"Warning: Index directory '{index_dir}' not found. Please run build_index.py.")
        rag_engine = None
    else:
        rag_engine = AmharicBankRAG(index_dir=index_dir)
        print("✓ Amharic RAG engine loaded successfully.")


@app.get("/health")
def healthcheck():
    return {
        "status": "healthy" if rag_engine and rag_engine.is_ready else "not_ready",
        "index_ready": rag_engine.is_ready if rag_engine else False
    }


@app.post("/retrieve", response_model=QueryResponse)
def retrieve_context(req: QueryRequest):
    if not rag_engine or not rag_engine.is_ready:
        raise HTTPException(status_code=503, detail="RAG engine index not loaded.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    res = rag_engine.query(req.query, top_k=req.top_k)
    return QueryResponse(
        query=res["query"],
        context=res["context"],
        formatted_prompt=res["formatted_prompt"],
        sources=res["sources"],
        latency_ms=res["latency_ms"],
        chunks=res["chunks"]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag.server:app", host="0.0.0.0", port=8000, reload=False)
