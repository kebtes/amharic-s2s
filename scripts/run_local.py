"""
CLI entry point to launch the interactive live microphone/speaker Amharic Voice Assistant.
"""

import sys
import os
import asyncio

# Add src to sys.path so script can be invoked from any working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from amharic_s2s.models.loader import load_all_models
from amharic_s2s.pipeline.runner import run_interactive_pipeline


def main():
    print("=" * 60)
    print("🇪🇹 Amharic Speech-to-Speech (S2S) Live Voice Assistant (Pipecat)")
    print("=" * 60)

    # 1. Load models
    bundle = load_all_models()

    # 2. Start interactive session
    asyncio.run(run_interactive_pipeline(bundle))


if __name__ == "__main__":
    main()
