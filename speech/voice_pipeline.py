"""
speech/voice_pipeline.py

Voice -> Sarvam STT -> existing RAG pipeline.

Pipeline:
    Audio file
        ↓
    Sarvam STT
        ↓
    Transcript
        ↓
    RAGOrchestrator
        ↓
    Grounded answer
"""

import argparse
import time
from pathlib import Path

from speech.sarvam import SarvamSTTClient
from pipeline.orchestrator import RAGOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Voice RAG pipeline: Sarvam STT -> RAG"
    )

    parser.add_argument(
        "--audio-file",
        type=Path,
        required=True,
        help="Path to audio file"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed_token"
    )

    args = parser.parse_args()

    total_start = time.perf_counter()

    # ---------------------------------------------------------
    # 1. Speech-to-text
    # ---------------------------------------------------------
    print("\n🎤 STEP 1: Sarvam STT")

    stt_start = time.perf_counter()

    stt_client = SarvamSTTClient(
        model="saaras:v3"
    )

    stt_result = stt_client.transcribe(
        str(args.audio_file),
        mode="translate",   # Returns English text for the English-only corpus.
        language_code=None, # Auto-detect source language.
    )

    stt_wall_ms = (time.perf_counter() - stt_start) * 1000

    query = stt_result.transcript.strip()

    print(f"Transcript: {query}")
    print(f"Detected language: {stt_result.language_code}")
    print(f"STT reported latency: {stt_result.latency_ms:.2f} ms")
    print(f"STT wall latency: {stt_wall_ms:.2f} ms")

    if not query:
        print("\n❌ STT returned an empty transcript.")
        return

    # ---------------------------------------------------------
    # 2. Existing RAG pipeline
    # ---------------------------------------------------------
    print("\n🧠 STEP 2: RAG")

    pipeline = RAGOrchestrator(
        strategy=args.strategy
    )

    rag_start = time.perf_counter()

    response = pipeline.answer(query)

    rag_wall_ms = (time.perf_counter() - rag_start) * 1000

    # ---------------------------------------------------------
    # 3. Final result
    # ---------------------------------------------------------
    total_ms = (time.perf_counter() - total_start) * 1000

    print("\n" + "=" * 70)
    print("VOICE RAG RESULT")
    print("=" * 70)

    print(f"Query: {query}")
    print(f"Answer: {response.answer}")
    print(f"Grounded: {response.grounded}")

    if response.refusal_reason:
        print(f"Refusal reason: {response.refusal_reason}")

    print(f"Sources: {[s['chunk_id'] for s in response.sources]}")

    print("\nLatency:")
    print(f"  STT:        {stt_wall_ms:.2f} ms")
    print(f"  RAG:        {rag_wall_ms:.2f} ms")
    print(f"  TOTAL:      {total_ms:.2f} ms")

    print("\nRAG internal latency:")
    for stage, ms in response.latency_ms.items():
        print(f"  {stage}: {ms:.2f} ms")

    print("=" * 70)


if __name__ == "__main__":
    main()