"""
Benchmark script to measure Time-To-First-Token (TTFT), Time-To-First-Audio (TTFA),
ASR latency, and total latency using the streaming Pipecat pipeline.
"""

import sys
import os
import asyncio
import json
import numpy as np
from loguru import logger

# Add src to sys.path so script can be invoked from any working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from amharic_s2s.models.loader import load_all_models
from amharic_s2s.pipeline.runner import run_pipeline_with_audio_file
from amharic_s2s.audio_utils import save_audio_file


async def run_benchmark():
    logger.info("Initializing Amharic S2S Pipeline Benchmark...")
    bundle = load_all_models()

    os.makedirs("results/pipecat_benchmarks", exist_ok=True)

    # Synthetic sample prompts
    test_samples = [
        {"id": "sample_01", "text_hint": "ሰላም እንደምን ነህ?", "duration_sec": 1.5, "f0": 220},
        {"id": "sample_02", "text_hint": "ዛሬ የአየር ሁኔታው እንዴት ነው?", "duration_sec": 2.2, "f0": 260},
        {"id": "sample_03", "text_hint": "ስለ ኢትዮጵያ ታሪክ ንገረኝ።", "duration_sec": 2.8, "f0": 200},
    ]

    records = []
    print("\n" + "=" * 80)
    print(f"{'Sample ID':<12} | {'ASR (s)':<9} | {'TTFT (s)':<9} | {'TTFA (s)':<9} | {'Total (s)':<9}")
    print("-" * 80)

    for item in test_samples:
        # Synthesize a simple test tone audio
        sr = 16000
        dur = item["duration_sec"]
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        audio = (0.2 * np.sin(2 * np.pi * item["f0"] * t)).astype(np.float32)

        res = await run_pipeline_with_audio_file(bundle, audio, orig_sr=sr, diffusion_steps=16)

        records.append({
            "id": item["id"],
            "asr_latency_sec": res["asr_latency_sec"],
            "ttft_sec": res["ttft_sec"],
            "ttfa_sec": res["ttfa_sec"],
            "total_latency_sec": res["total_latency_sec"],
            "transcript": res["transcript"],
            "llm_response": res["llm_response"],
        })

        print(
            f"{item['id']:<12} | "
            f"{res['asr_latency_sec']:<9.3f} | "
            f"{res['ttft_sec']:<9.3f} | "
            f"{res['ttfa_sec']:<9.3f} | "
            f"{res['total_latency_sec']:<9.3f}"
        )

        if res["audio_out"] is not None and len(res["audio_out"]) > 0:
            out_path = f"results/pipecat_benchmarks/{item['id']}_output.wav"
            save_audio_file(out_path, res["audio_out"], sample_rate=res["output_sr"])

    print("=" * 80)
    summary_path = "results/pipecat_benchmarks/benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.success(f"Benchmark completed! Results saved to {summary_path}")


def main():
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
