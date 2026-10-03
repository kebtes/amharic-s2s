import os
import time
from typing import List, Dict, Any, Optional
from .retriever import HybridRetriever
from .embeddings import AmharicEmbeddingModel


DEFAULT_SYSTEM_PROMPT = """አንተ የባንክ መረጃ ረዳት ነህ። ከታች የተሰጠውን የባንክ ሰነድ መረጃ ብቻ በመጠቀም ለደንበኛው ጥያቄ ግልጽ፣ አጭር እና ትክክለኛ ምላሽ ስጥ። 
ሰነዱ ውስጥ የሌለ መረጃ በፍጹም አትጨምር።

[የተገኘ የባንክ መረጃ]:
{context}

[የደንበኛ ጥያቄ]:
{query}

[ምላሽ]:"""


class AmharicBankRAG:
    def __init__(
        self,
        index_dir: str = "vector_store",
        embedder: Optional[AmharicEmbeddingModel] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        top_k: int = 3
    ):
        self.index_dir = index_dir
        self.embedder = embedder or AmharicEmbeddingModel()
        self.retriever = HybridRetriever(embedder=self.embedder)
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.is_ready = False

        if os.path.exists(index_dir):
            try:
                self.retriever.load(index_dir)
                self.is_ready = True
            except Exception as e:
                print(f"Warning: Index loading failed from '{index_dir}': {e}")

    def get_context(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve context for a user query.
        Returns context text, retrieved metadata, and latency.
        """
        if not self.is_ready:
            raise RuntimeError("RAG index is not loaded. Run build_index.py first.")

        k = top_k or self.top_k
        t0 = time.time()
        chunks = self.retriever.retrieve(query, top_k=k)
        latency_ms = (time.time() - t0) * 1000.0

        context_passages = []
        sources = []

        for idx, chunk in enumerate(chunks, 1):
            src = chunk.get("metadata", {}).get("source_file", "unknown")
            sources.append(src)
            context_passages.append(f"[{idx}] {chunk['content']}")

        context_str = "\n\n".join(context_passages)

        return {
            "query": query,
            "context": context_str,
            "chunks": chunks,
            "sources": list(set(sources)),
            "latency_ms": round(latency_ms, 2)
        }

    def format_prompt(self, query: str, context: str) -> str:
        """Format prompt for downstream LLM generation."""
        return self.system_prompt.format(context=context, query=query)

    def query(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """End-to-end query method returning context and LLM-ready formatted prompt."""
        result = self.get_context(query, top_k=top_k)
        formatted_prompt = self.format_prompt(query, result["context"])
        result["formatted_prompt"] = formatted_prompt
        return result

    def benchmark_latency(self, queries: List[str]) -> Dict[str, float]:
        """Run latency benchmarks across a set of test queries."""
        latencies = []
        for q in queries:
            t0 = time.time()
            self.get_context(q)
            latencies.append((time.time() - t0) * 1000.0)

        import numpy as np
        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "p90_latency_ms": float(np.percentile(latencies, 90)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "total_queries": len(queries)
        }
