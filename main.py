"""
main.py — Entry point for the Voice-Enabled RAG System.

Starts the FastAPI server (via uvicorn) which:
  - Pre-loads the FAISS/BM25 indexes and embedding model at startup
  - Serves the REST API at /health, /query, /voice
  - Serves the frontend SPA at /

Usage:
    python main.py
    python main.py --host 0.0.0.0 --port 8000 --strategy fixed_token

Environment variables:
    GROQ_API_KEY     — required for LLM generation
    SARVAM_API_KEY   — required for voice input (optional for text-only mode)
    RAG_STRATEGY     — chunking strategy override (default: fixed_token)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Voice-Enabled RAG System — FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--strategy", type=str, default="fixed_token", help="Chunking strategy (default: fixed_token)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    # Set strategy as environment variable so app/api.py startup event picks it up
    os.environ["RAG_STRATEGY"] = args.strategy

    print("=" * 60)
    print("  Voice-Enabled RAG System — HH Goa 2026")
    print("=" * 60)
    print(f"  Strategy:  {args.strategy}")
    print(f"  Server:    http://{args.host}:{args.port}")
    print(f"  Frontend:  http://localhost:{args.port}/")
    print(f"  API Docs:  http://localhost:{args.port}/docs")
    print(f"  Groq API:  {'✅ set' if os.environ.get('GROQ_API_KEY') else '❌ not set'}")
    print(f"  Sarvam:    {'✅ set' if os.environ.get('SARVAM_API_KEY') else '❌ not set (voice disabled)'}")
    print("=" * 60)
    print()

    import uvicorn
    uvicorn.run(
        "app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
