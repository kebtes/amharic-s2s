import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_engine import AmharicBankRAG


BENCHMARK_QUERIES = [
    "የአቢሲኒያ ኦንላይን ባንክ አገልግሎት ምንድነው?",
    "ATM ካርድ ቢጠፋብኝ የት ነው ማመልከት ያለብኝ?",
    "የሞባይል ባንኪንግ አገልግሎት ለመጠቀም ምን ያስፈልጋል?",
    "የመኖሪያ ቤት ብድር (Mortgage) ለማግኘት መስፈርቶቹ ምንድናቸው?",
    "የውጭ ሀገር ገንዘብ (Foreign Currency) አካውንት እንዴት ይከፈታል?",
    "የኢስላሚክ ባንክ (IFB Waddiah) ምርቶች ምንድናቸው?",
    "POS ማሽን ለማግኘት የሚያስፈልጉ ቅድመ ሁኔታዎች ምንድናቸው?",
    "የብድር አገልግሎት አሰጣጥ ሂደት ምን ይመስላል?",
    "የካርድ ክፍያ እና የካርድ ገደብ (Limit) ስንት ነው?",
    "የንግድ እና ኮርፖሬት ባንክ አገልግሎት ጥቅሞች ምንድናቸው?"
]


def run_eval(index_dir: str = "vector_store", output_json: str = "results/rag_evaluation.json"):
    print("=== Amharic Bank RAG System Evaluation ===")
    
    if not os.path.exists(index_dir):
        print(f"Error: Vector store index directory '{index_dir}' does not exist. Run build_index.py first.")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    rag = AmharicBankRAG(index_dir=index_dir, top_k=3)

    query_results = []
    latencies = []
    context_lengths = []

    print(f"\nEvaluating {len(BENCHMARK_QUERIES)} banking benchmark queries...")

    for i, q in enumerate(BENCHMARK_QUERIES, 1):
        res = rag.query(q)
        lat = res["latency_ms"]
        ctx_len = len(res["context"])
        
        latencies.append(lat)
        context_lengths.append(ctx_len)

        query_results.append({
            "id": i,
            "query": q,
            "latency_ms": lat,
            "context_char_length": ctx_len,
            "num_chunks_retrieved": len(res["chunks"]),
            "sources": res["sources"],
            "top_chunk_snippet": res["chunks"][0]["content"][:120] if res["chunks"] else ""
        })
        print(f"[{i}/{len(BENCHMARK_QUERIES)}] Query: '{q[:30]}...' -> Latency: {lat:.2f}ms | Sources: {res['sources']}")

    import numpy as np
    summary = {
        "num_test_queries": len(BENCHMARK_QUERIES),
        "latency_stats": {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "median_ms": round(float(np.median(latencies)), 2),
            "p90_ms": round(float(np.percentile(latencies, 90)), 2),
            "min_ms": round(float(np.min(latencies)), 2),
            "max_ms": round(float(np.max(latencies)), 2)
        },
        "context_stats": {
            "avg_context_length_chars": round(float(np.mean(context_lengths)), 1),
            "max_context_length_chars": int(np.max(context_lengths))
        },
        "detailed_runs": query_results
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Evaluation complete! Summary report saved to '{output_json}'.")
    print(f"📊 Latency Summary: Mean = {summary['latency_stats']['mean_ms']}ms | P90 = {summary['latency_stats']['p90_ms']}ms")
    return summary


if __name__ == "__main__":
    run_eval()
