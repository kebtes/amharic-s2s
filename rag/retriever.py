import os
import json
import pickle
import time
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple
from .embeddings import AmharicEmbeddingModel
from .text_processor import Chunk


def amharic_tokenizer(text: str) -> List[str]:
    """Simple whitespace and punctuation tokenizer for Amharic BM25 search."""
    import re
    # Clean non-alphanumeric except Ethiopic script range \u1200-\u137F
    clean = re.sub(r'[^\w\u1200-\u137F\s]', ' ', text)
    tokens = [t.strip().lower() for t in clean.split() if len(t.strip()) > 1]
    return tokens


class HybridRetriever:
    def __init__(
        self,
        embedder: AmharicEmbeddingModel = None,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        self.embedder = embedder or AmharicEmbeddingModel()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self.faiss_index: faiss.IndexFlatIP = None
        self.bm25: BM25Okapi = None
        self.chunks: List[Dict[str, Any]] = []
        self.is_indexed = False

    def build_index(self, chunks: List[Chunk], batch_size: int = 32):
        """Build FAISS vector index and BM25 index from document chunks."""
        if not chunks:
            raise ValueError("No chunks provided to build_index.")

        print(f"Building index for {len(chunks)} chunks...")
        t0 = time.time()

        self.chunks = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "metadata": c.metadata
            }
            for c in chunks
        ]

        texts = [c.content for c in chunks]

        # 1. Build Dense FAISS Index
        embeddings = self.embedder.encode(texts, is_query=False, batch_size=batch_size)
        dimension = embeddings.shape[1]

        # Inner Product on normalized vectors = Cosine Similarity
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

        # 2. Build Sparse BM25 Index
        corpus_tokens = [amharic_tokenizer(t) for t in texts]
        self.bm25 = BM25Okapi(corpus_tokens)

        self.is_indexed = True
        t_elapsed = time.time() - t0
        print(f"✓ Indexing completed in {t_elapsed:.2f}s (FAISS dim={dimension}, BM25 corpus={len(corpus_tokens)}).")

    def search_dense(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Dense embedding search returning (chunk_idx, score)."""
        q_vec = self.embedder.encode(query, is_query=True)
        scores, indices = self.faiss_index.search(q_vec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:
                results.append((int(idx), float(score)))
        return results

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Sparse BM25 search returning (chunk_idx, score)."""
        q_tokens = amharic_tokenizer(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        return results

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion (RRF).
        Returns top_k retrieved chunks with relevance metadata and timing.
        """
        if not self.is_indexed:
            raise RuntimeError("Index is not built or loaded.")

        t0 = time.time()

        # Fetch more candidates for fusion
        fetch_k = max(top_k * 4, 15)
        dense_results = self.search_dense(query, top_k=fetch_k)
        bm25_results = self.search_bm25(query, top_k=fetch_k)

        # RRF Scoring map: chunk_idx -> score
        rrf_scores: Dict[int, float] = {}

        # Dense RRF
        for rank, (idx, score) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self.dense_weight * (1.0 / (self.rrf_k + rank + 1))

        # BM25 RRF
        for rank, (idx, score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self.bm25_weight * (1.0 / (self.rrf_k + rank + 1))

        # Sort by final RRF score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        retrieved_chunks = []
        search_time_ms = (time.time() - t0) * 1000.0

        for rank, (idx, rrf_score) in enumerate(sorted_indices):
            chunk_data = dict(self.chunks[idx])
            chunk_data["rrf_score"] = rrf_score
            chunk_data["rank"] = rank + 1
            chunk_data["retrieval_latency_ms"] = round(search_time_ms, 2)
            retrieved_chunks.append(chunk_data)

        return retrieved_chunks

    def save(self, output_dir: str):
        """Save FAISS index, BM25 index, and metadata to disk."""
        os.makedirs(output_dir, exist_ok=True)
        faiss_path = os.path.join(output_dir, "faiss_index.bin")
        bm25_path = os.path.join(output_dir, "bm25_index.pkl")
        chunks_path = os.path.join(output_dir, "chunks_metadata.json")

        faiss.write_index(self.faiss_index, faiss_path)
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved hybrid vector index to '{output_dir}'.")

    def load(self, input_dir: str):
        """Load pre-built indices and metadata from disk."""
        faiss_path = os.path.join(input_dir, "faiss_index.bin")
        bm25_path = os.path.join(input_dir, "bm25_index.pkl")
        chunks_path = os.path.join(input_dir, "chunks_metadata.json")

        if not (os.path.exists(faiss_path) and os.path.exists(bm25_path) and os.path.exists(chunks_path)):
            raise FileNotFoundError(f"Index files missing in '{input_dir}'.")

        t0 = time.time()
        self.faiss_index = faiss.read_index(faiss_path)
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.is_indexed = True
        t_elapsed = time.time() - t0
        print(f"✓ Loaded hybrid index from '{input_dir}' in {t_elapsed:.3f}s ({len(self.chunks)} chunks).")
