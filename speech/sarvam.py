"""
speech/sarvam.py

Sarvam STT client (REST endpoint, saaras:v3/v4). Accepts an audio file,
returns transcribed/translated text. Reports STT latency separately, per
plan spec ("Reports STT latency separately").

MODE DEFAULT - READ BEFORE CHANGING: this client defaults to mode="translate",
NOT the Sarvam API's own default of "transcribe". Reason: this project's
indexed corpus is English-only (English_passages/Eng_Query), but the plan
calls for Hindi/Indic voice input. mode="translate" returns spoken Indic
language audio ALREADY IN ENGLISH, ready to feed straight into
pipeline.orchestrator.RAGOrchestrator.answer(). mode="transcribe" would
return Hindi text with no path into an English-only corpus without a
separate translation step that doesn't exist in this project. Override to
"transcribe" only if you specifically want native-language text back.

REQUIRES: SARVAM_API_KEY set in the environment.
    PowerShell:  $env:SARVAM_API_KEY="your-key-here"

LIMITS: REST endpoint only supports audio up to 30 seconds. Longer clips
need Sarvam's async Batch API (not implemented here - out of scope for a
live voice-query demo).

Usage (CLI smoke test - requires a real API key and a real audio file):
    python -m speech.sarvam --audio-file path\to\clip.wav
    python -m speech.sarvam --audio-file path\to\clip.wav --mode transcribe --language-code hi-IN
"""

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

from sarvamai import SarvamAI

DEFAULT_MODEL = "saaras:v3"
DEFAULT_MODE = "translate"  # see module docstring - deliberate override of the API's own default


@dataclass
class STTResult:
    transcript: str
    language_code: str | None
    request_id: str | None
    latency_ms: float
    mode: str


class SarvamSTTClient:
    """Wraps Sarvam's REST speech-to-text endpoint."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        key = api_key or os.environ.get("SARVAM_API_KEY")
        if not key:
            raise EnvironmentError(
                "SARVAM_API_KEY not set. Set it in your environment before running "
                '(PowerShell: $env:SARVAM_API_KEY="your-key-here"), or pass api_key= explicitly.'
            )
        self.model = model
        self.client = SarvamAI(api_subscription_key=key)

    def transcribe(
        self,
        audio_path: Path,
        mode: str = DEFAULT_MODE,
        language_code: str | None = None,
    ) -> STTResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t0 = time.perf_counter()
        with open(audio_path, "rb") as f:
            kwargs = {"file": f, "model": self.model, "mode": mode}
            if language_code:
                kwargs["language_code"] = language_code
            response = self.client.speech_to_text.transcribe(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        return STTResult(
            transcript=response.transcript,
            language_code=response.language_code,
            request_id=response.request_id,
            latency_ms=latency_ms,
            mode=mode,
        )


def main():
    parser = argparse.ArgumentParser(description="Sarvam STT smoke test.")
    parser.add_argument("--audio-file", type=Path, required=True)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=["transcribe", "translate", "verbatim", "translit", "codemix"])
    parser.add_argument("--language-code", type=str, default=None, help="e.g. hi-IN, ta-IN. Omit to auto-detect.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=["saaras:v3", "saaras:v4"])
    args = parser.parse_args()

    client = SarvamSTTClient(model=args.model)
    result = client.transcribe(args.audio_file, mode=args.mode, language_code=args.language_code)

    print(f"\nAudio file: {args.audio_file}")
    print(f"Mode: {result.mode}")
    print(f"Transcript: {result.transcript}")
    print(f"Detected language: {result.language_code}")
    print(f"STT latency: {result.latency_ms:.2f} ms")


if __name__ == "__main__":
    main()