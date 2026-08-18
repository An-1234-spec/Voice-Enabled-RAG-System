"""
app/api.py

FastAPI application for the Voice-Enabled RAG system.

Endpoints:
  GET  /health  — confirms indexes are loaded and models are ready
  POST /query   — text query -> structured RAG response
  POST /voice   — audio file upload -> Sarvam STT -> RAG -> structured response

Design decisions:
  - RAGOrchestrator is initialized once at startup (pre-loads embedding model,
    FAISS index, BM25 index, guardrails). First request is fast.
  - CORS is wide open (allow all origins) for local dev / HF Spaces demo.
    Tighten for production.
  - Frontend is served as static files from the frontend/ directory.
  - Voice endpoint accepts multipart file upload (WAV/WebM/MP3), writes to a
    temp file for Sarvam SDK (which needs a file path), then cleans up.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import (
    QueryRequest,
    RAGResponse,
    VoiceResponse,
    HealthResponse,
    SourceInfo,
    LatencyBreakdown,
)
from pipeline.orchestrator import RAGOrchestrator
from speech.sarvam import SarvamSTTClient

# ── App setup ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice-Enabled RAG System",
    description="HH Goa 2026 — Hybrid retrieval + reranking + grounded LLM generation with voice input via Sarvam STT",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (initialized at startup) ────────────────────────────────

_orchestrator: RAGOrchestrator | None = None
_stt_client: SarvamSTTClient | None = None
_strategy: str = "fixed_token"


def _get_orchestrator() -> RAGOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Server is still starting up.")
    return _orchestrator


def _get_stt_client() -> SarvamSTTClient:
    global _stt_client
    if _stt_client is None:
        try:
            _stt_client = SarvamSTTClient()
        except EnvironmentError:
            raise HTTPException(
                status_code=503,
                detail="SARVAM_API_KEY not set. Voice input is unavailable.",
            )
    return _stt_client


# ── Startup event ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _orchestrator, _strategy

    _strategy = os.environ.get("RAG_STRATEGY", "fixed_token")
    print(f"[startup] Initializing RAG pipeline with strategy='{_strategy}'...")

    try:
        _orchestrator = RAGOrchestrator(strategy=_strategy)
        print("[startup] Pipeline ready.")
    except Exception as e:
        print(f"[startup] WARNING: Pipeline initialization failed: {e}")
        print("[startup] Server will start but /query and /voice will return 503.")


# ── Helper: convert orchestrator RAGResponse -> API RAGResponse ──────────

def _convert_response(orch_response, stt_ms: float = 0.0) -> dict:
    """Convert pipeline.orchestrator.RAGResponse dataclass to API schema dict."""
    latency = orch_response.latency_ms

    # Sum generation attempt keys for a single "generation_ms" number
    gen_ms = sum(v for k, v in latency.items() if k.startswith("generation_ms_attempt"))
    grounding_ms = sum(v for k, v in latency.items() if k.startswith("grounding_ms_attempt"))

    # Compute a simple confidence score:
    # - 1.0 if grounded and answer exists
    # - 0.5 if answer exists but not fully grounded
    # - 0.0 if refused
    if orch_response.answer and orch_response.grounded:
        confidence = 1.0
    elif orch_response.answer:
        confidence = 0.5
    else:
        confidence = 0.0

    return {
        "request_id": orch_response.request_id,
        "query": orch_response.query,
        "answer": orch_response.answer,
        "grounded": orch_response.grounded,
        "confidence": confidence,
        "sources": [
            SourceInfo(
                chunk_id=s["chunk_id"],
                text=s["text"],
                score=s.get("score"),
            )
            for s in orch_response.sources
        ],
        "refusal_reason": orch_response.refusal_reason,
        "stage_reached": orch_response.stage_reached,
        "latency": LatencyBreakdown(
            safety_ms=latency.get("safety_ms"),
            relevance_ms=latency.get("relevance_ms"),
            retrieval_ms=latency.get("retrieval_ms"),
            generation_ms=gen_ms if gen_ms > 0 else None,
            grounding_ms=grounding_ms if grounding_ms > 0 else None,
            total_ms=latency.get("total_ms"),
            stt_ms=stt_ms if stt_ms > 0 else None,
        ),
    }


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — confirms indexes are loaded and models are ready."""
    return HealthResponse(
        status="ok" if _orchestrator is not None else "initializing",
        strategy=_strategy,
        index_loaded=_orchestrator is not None,
        model_loaded=_orchestrator is not None,
    )


@app.post("/query", response_model=RAGResponse)
async def query(request: QueryRequest):
    """Text query → structured RAG response."""
    orchestrator = _get_orchestrator()

    try:
        orch_response = orchestrator.answer(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return RAGResponse(**_convert_response(orch_response))


@app.post("/voice", response_model=VoiceResponse)
async def voice(
    file: UploadFile = File(..., description="Audio file (WAV, WebM, MP3, OGG) up to 30 seconds"),
    language_code: str | None = Form(default=None, description="Language code (e.g. hi-IN). Omit for auto-detect."),
):
    """Audio file upload → Sarvam STT → RAG → structured response."""
    orchestrator = _get_orchestrator()
    stt_client = _get_stt_client()

    # Validate file type
    allowed_types = {
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/webm", "audio/mp3", "audio/mpeg",
        "audio/ogg", "audio/flac", "application/octet-stream",
    }
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file.content_type}. Use WAV, WebM, MP3, OGG, or FLAC.",
        )

    # Save uploaded audio to a temp file (Sarvam SDK needs a file path)
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            content = await file.read()
            tmp.write(content)

        # STT
        stt_result = stt_client.transcribe(
            tmp_path,
            mode="translate",  # Returns English text for our English-only corpus
            language_code=language_code,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Audio file could not be processed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {str(e)}")
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

    transcript = stt_result.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="STT returned an empty transcript. Try speaking more clearly or check the audio file.")

    # RAG pipeline
    try:
        orch_response = orchestrator.answer(transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    base = _convert_response(orch_response, stt_ms=stt_result.latency_ms)
    return VoiceResponse(
        **base,
        transcript=transcript,
        detected_language=stt_result.language_code,
        stt_latency_ms=stt_result.latency_ms,
    )


# ── Serve frontend ──────────────────────────────────────────────────────

frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend SPA."""
        index_path = frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        raise HTTPException(status_code=404, detail="Frontend not found")
