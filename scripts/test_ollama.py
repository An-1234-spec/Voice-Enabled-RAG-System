"""
scripts/test_ollama.py

Standalone test of local Ollama inference latency — deliberately NOT wired
into generation/llm.py yet. Run this first to confirm (a) Ollama is
reachable, (b) the model responds with valid JSON, and (c) latency is
actually in the <200ms range before we touch your working pipeline code.

Uses raw HTTP (requests) against Ollama's REST API rather than the ollama
pip package — one fewer dependency, and full control over `keep_alive` /
`format` params.

Run:
    python -m scripts.test_ollama
"""

from __future__ import annotations

import json
import time

import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"  # NOT "localhost" — on Windows, requests/urllib3
# resolves "localhost" and tries IPv6 (::1) first; if Ollama only listens on IPv4, that attempt
# has to time out before falling back to IPv4, adding a fixed ~2s tax to every call. Using the
# literal IP skips DNS resolution entirely.
MODEL = "llama3.2"
KEEP_ALIVE = "30m"  # prevents Ollama unloading the model between requests
NUM_PREDICT = 60  # tight output cap — your system prompt asks for 1-3 sentences, generation
# time scales ~linearly with token count, and 150 tokens was most of your latency budget

SYSTEM_PROMPT = """You are a factual question-answering assistant. Answer ONLY using the \
provided context. If the context doesn't contain enough information, set \
"insufficient_context" to true. Respond with ONLY a JSON object with these exact fields: \
{"answer": "<string>", "confidence": "<high|medium|low>", "insufficient_context": \
<true|false>, "used_source_indices": [<int>, ...]}"""

CONTEXTS = [
    "Corporation definition: an association of individuals, created by law or under authority of law, having a continuous existence independent of its members.",
    "A corporation is owned by shareholders who share in profits and losses.",
]


def build_user_prompt(query: str) -> str:
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(CONTEXTS))
    return f"Context:\n{context_block}\n\nQuestion: {query}"


def call_ollama(query: str, model: str = MODEL, warm: bool = False, timeout: float = 30) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query)},
        ],
        "format": "json",  # Ollama grammar-constrains output to valid JSON
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.0,
            "num_predict": NUM_PREDICT,
        },
    }

    start = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    wall_ms = (time.perf_counter() - start) * 1000

    resp.raise_for_status()
    data = resp.json()

    raw_answer = data["message"]["content"]

    # Ollama reports its own timing in nanoseconds — convert for comparison.
    load_ms = data.get("load_duration", 0) / 1e6
    prompt_eval_ms = data.get("prompt_eval_duration", 0) / 1e6
    eval_ms = data.get("eval_duration", 0) / 1e6
    total_ms = data.get("total_duration", 0) / 1e6

    result = {
        "wall_clock_ms": round(wall_ms, 1),
        "ollama_load_ms": round(load_ms, 1),
        "ollama_prompt_eval_ms": round(prompt_eval_ms, 1),
        "ollama_eval_ms": round(eval_ms, 1),
        "ollama_total_ms": round(total_ms, 1),
        "raw_answer": raw_answer,
        "parsed_ok": False,
    }

    try:
        parsed = json.loads(raw_answer)
        result["parsed_ok"] = True
        result["parsed"] = parsed
    except json.JSONDecodeError:
        pass

    return result


def main():
    print("Checking Ollama is reachable...")
    try:
        requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
    except requests.exceptions.ConnectionError:
        print(
            "ERROR: Can't reach Ollama at 127.0.0.1:11434. Is the Ollama "
            "service running? Try `ollama serve` in another terminal, or "
            "check it started after `winget install`."
        )
        return

    query = "what is a corporation?"
    candidates = ["llama3.2", "llama3.2:1b", "qwen2.5:1.5b"]  # 3B, 1B, and a 1.5B alternative
    # family — Qwen2.5 is often reported as stronger at instruction-following per
    # parameter than Llama 3.2 at similar size, worth checking given llama3.2:1b's
    # answer collapsed to a near-useless fragment despite being fast enough.

    all_results = {}

    for model in candidates:
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")

        print(f"\n--- Warmup call (loads model into VRAM, will be slow — this can genuinely ---")
        print(f"--- take 30-90s+ for CUDA init + first load, that's expected, not a bug) ---")
        try:
            warmup = call_ollama(query, model=model, warm=True, timeout=120)
        except requests.exceptions.HTTPError as e:
            print(f"  FAILED: {e}")
            print(f"  (If 404: run `ollama pull {model}` first, this model isn't downloaded yet.)")
            continue

        print(f"  wall_clock_ms   : {warmup['wall_clock_ms']}")
        print(f"  ollama_load_ms  : {warmup['ollama_load_ms']}  <- one-time cost, ignore for real benchmark")
        print(f"  ollama_eval_ms  : {warmup['ollama_eval_ms']}")

        print(f"\n--- Real calls (model already warm, this is the number that matters) ---")
        model_results = []
        for i in range(3):
            result = call_ollama(query, model=model)
            print(f"\n  Run {i+1}:")
            print(f"    wall_clock_ms      : {result['wall_clock_ms']}")
            print(f"    ollama_prompt_eval : {result['ollama_prompt_eval_ms']}")
            print(f"    ollama_eval_ms     : {result['ollama_eval_ms']}  <- generation compute")
            print(f"    ollama_total_ms    : {result['ollama_total_ms']}")
            print(f"    parsed_ok          : {result['parsed_ok']}")
            if result["parsed_ok"]:
                print(f"    answer             : {result['parsed'].get('answer', '')[:120]}")
            else:
                print(f"    raw_answer (unparsed): {result['raw_answer'][:200]}")
            model_results.append(result)
        all_results[model] = model_results

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY (avg wall_clock_ms across 3 real runs, sorted fastest first)")
        print(f"{'='*60}")
        summary = []
        for model, results in all_results.items():
            avg_wall = sum(r["wall_clock_ms"] for r in results) / len(results)
            avg_eval = sum(r["ollama_eval_ms"] for r in results) / len(results)
            summary.append((model, avg_wall, avg_eval))
        for model, avg_wall, avg_eval in sorted(summary, key=lambda x: x[1]):
            flag = "  <-- under 200ms target!" if avg_wall < 200 else ""
            print(f"  {model:15s}  wall={avg_wall:7.1f}ms  eval={avg_eval:7.1f}ms{flag}")


if __name__ == "__main__":
    main()