# Voice-Enabled RAG System — HH Goa 2026

A production-quality, voice-enabled Retrieval-Augmented Generation (RAG) system built for the HH Goa 2026 hackathon. Processes voice input → Sarvam STT → hybrid retrieval (FAISS + BM25) → cross-encoder reranking → grounded LLM generation → guardrails → structured answer, backed by the [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset.

## Architecture

```
🎤 Voice Input ──→ Sarvam STT ──→ Query
                                    │
📝 Text Input ─────────────────────→│
                                    ▼
                            Input Safety Check
                                    │
                            ┌───────┴───────┐
                            │unsafe         │safe
                            ▼               ▼
                        🚫 Refuse    Relevance Check
                                        │
                                ┌───────┴───────┐
                                │out-of-domain  │in-domain
                                ▼               ▼
                            🚫 Refuse    Hybrid Retrieval
                                        ┌───┴───┐
                                    Dense     BM25
                                    FAISS     Lexical
                                        └───┬───┘
                                        Score Fusion
                                            │
                                    Cross-Encoder Reranker
                                            │
                                    Context Selection (Top 5)
                                            │
                                    LLM Generation (Groq)
                                            │
                                    Output Validation
                                            │
                                    Grounding Check
                                            │
                                    ✅ Structured Response
```

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **STT** | Sarvam API (Saaras v3) | 22 Indian languages support |
| **Embedding** | all-MiniLM-L6-v2 (384d) | Fast (~5ms), strong MS MARCO performance |
| **Vector Index** | FAISS IndexFlatIP | Exact search, best recall for <50K chunks |
| **BM25** | rank_bm25 | Simple, effective, in-memory |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Small, fast, trained on MS MARCO |
| **LLM** | Groq API + openai/gpt-oss-20b | Low latency, free tier, OpenAI-compatible |
| **API** | FastAPI + Uvicorn | Async, fast, auto-generated docs |
| **Frontend** | Vanilla HTML/CSS/JS | No build step, lightweight |

## Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd RAG-1

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
# PowerShell
$env:GROQ_API_KEY="your-groq-key"
$env:SARVAM_API_KEY="your-sarvam-key"

# Bash
export GROQ_API_KEY="your-groq-key"
export SARVAM_API_KEY="your-sarvam-key"
```

Or create a `.env` file (copy from `.env.example`):
```
GROQ_API_KEY=your-groq-key
SARVAM_API_KEY=your-sarvam-key
```

Get keys:
- **Groq**: [console.groq.com](https://console.groq.com/)
- **Sarvam**: [console.sarvam.ai](https://console.sarvam.ai/)

### 3. Build Offline Pipeline (One-Time)

```bash
# Download dataset subset (100 records, ~1.2 MB)
python -m data.download --size 100

# Preprocess passages
python -m data.preprocess

# Generate chunks for all strategies
python -m chunking.generate --data data\processed\passages.jsonl --output-dir data\processed\chunks --chunk-size 80

# Embed chunks (downloads all-MiniLM-L6-v2 on first run, ~80 MB)
# Run for each strategy:
python -c "
from embeddings.embedder import Embedder
from chunking.base import Chunk
import json

for strategy in ['fixed_token', 'passage', 'sentence', 'semantic', 'parent_child']:
    print(f'Embedding {strategy}...')
    chunks = []
    with open(f'data/processed/chunks/{strategy}.jsonl') as f:
        for line in f:
            rec = json.loads(line)
            chunks.append(Chunk(**rec))
    embedder = Embedder()
    result = embedder.embed_chunks(chunks, show_progress=True)
    result.save(f'data/processed/embeddings/{strategy}')
    # Write meta JSONL for FAISS lookup
    with open(f'data/processed/embeddings/{strategy}_meta.jsonl', 'w') as mf:
        for c in chunks:
            mf.write(json.dumps({'chunk_id': c.chunk_id, 'passage_id': c.passage_id, 'document_id': c.document_id, 'num_tokens': c.token_count}) + '\n')
    print(f'  Done: {len(chunks)} chunks embedded')
"

# Build FAISS indexes
python -m indexing.build_faiss
```

### 4. Start the Server

```bash
python main.py
```

Then open:
- **Frontend**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs

## API Endpoints

### `GET /health`
Health check — confirms indexes are loaded.

```json
{"status": "ok", "strategy": "fixed_token", "index_loaded": true, "model_loaded": true}
```

### `POST /query`
Text query → structured RAG response.

**Request:**
```json
{"query": "What is a corporation?", "strategy": "fixed_token"}
```

**Response:**
```json
{
  "request_id": "uuid",
  "query": "What is a corporation?",
  "answer": "A corporation is a legal entity...",
  "grounded": true,
  "confidence": 1.0,
  "sources": [{"chunk_id": "...", "text": "...", "score": 0.85}],
  "refusal_reason": null,
  "stage_reached": "structured_response",
  "latency": {
    "safety_ms": 0.1,
    "relevance_ms": 5.2,
    "retrieval_ms": 45.3,
    "generation_ms": 500.0,
    "grounding_ms": 0.3,
    "total_ms": 551.2
  }
}
```

### `POST /voice`
Audio file → Sarvam STT → RAG → structured response.

**Request:** Multipart form with `file` (audio) and optional `language_code`.

**Response:** Same as `/query` plus `transcript`, `detected_language`, `stt_latency_ms`.

## Project Structure

```
RAG-1/
├── app/
│   ├── api.py              # FastAPI application
│   └── schemas.py           # Pydantic request/response models
├── chunking/
│   ├── base.py              # Abstract Chunker + Chunk dataclass
│   ├── fixed_token.py       # Fixed-size sliding window
│   ├── sentence.py          # Sentence-aware with overlap
│   ├── passage.py           # Paragraph-aware
│   ├── semantic.py          # Embedding similarity breakpoints
│   ├── parent_child.py      # Two-level: index children, expand to parents
│   ├── generate.py          # Run all strategies over passages
│   └── benchmark.py         # Chunk-level statistics benchmark
├── config/
│   └── settings.py          # Pydantic centralized configuration
├── data/
│   ├── dataset_inspect.py   # Stream & inspect MSMARCO-XI
│   ├── download.py          # Controlled subset download
│   └── preprocess.py        # Deduplicate, clean, assign IDs
├── embeddings/
│   └── embedder.py          # Batch embedding with save/load
├── evaluation/
│   ├── chunking_eval.py     # Combined chunk stats + retrieval quality
│   ├── retrieval_eval.py    # Recall@k, Precision@k, MRR
│   ├── latency_eval.py      # End-to-end pipeline latency (P50/P70/P100)
│   └── rerank_tradeoff.py   # Candidate pool size vs quality/latency
├── frontend/
│   └── index.html           # Single-page voice + text demo
├── generation/
│   ├── llm.py               # Groq API client (structured JSON output)
│   └── prompts.py           # RAG prompt templates with grounding rules
├── guardrails/
│   ├── safety.py            # Input safety (keyword + regex)
│   ├── relevance.py         # Domain relevance (corpus centroid cosine)
│   ├── grounding.py         # Answer grounding (lexical overlap)
│   └── output_validator.py  # Structural validation of LLM output
├── indexing/
│   └── build_faiss.py       # Build FAISS indexes per strategy
├── pipeline/
│   └── orchestrator.py      # Full pipeline: safety→retrieve→generate→validate
├── retrieval/
│   ├── vector_retriever.py  # Dense retrieval (FAISS)
│   ├── bm25_retriever.py    # BM25 lexical retrieval
│   ├── hybrid_retriever.py  # Weighted score fusion
│   └── reranker.py          # Cross-encoder reranking
├── scripts/
│   ├── benchmark.py         # Combined benchmark runner
│   ├── test_pipeline.py     # 8 demo scenarios
│   ├── diagnose_latency.py  # Groq model latency diagnosis
│   └── compare_models.py    # Side-by-side model comparison
├── speech/
│   ├── sarvam.py            # Sarvam STT client
│   └── voice_pipeline.py    # Voice → STT → RAG pipeline
├── tests/
│   ├── test_chunking.py     # Chunking unit tests
│   ├── test_guardrails.py   # Guardrails unit tests
│   └── test_orchestrator.py # Orchestrator unit tests
├── main.py                  # Entry point (starts FastAPI server)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
└── .gitignore
```

## Chunking Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **fixed_token** | Pure sliding window, no linguistic awareness | Baseline comparison |
| **sentence** | Sentence-boundary splitting with token overlap | General-purpose RAG |
| **passage** | Paragraph-aware, merges short paragraphs | Multi-paragraph docs |
| **semantic** | Embedding similarity to detect topic shifts | Topic-diverse passages |
| **parent_child** | Small children for retrieval, large parents for context | Precision + context |

## Guardrails

1. **Input Safety** — Keyword + regex blocking for unsafe/inappropriate queries
2. **Domain Relevance** — Cosine similarity to corpus centroid; rejects out-of-domain
3. **Output Validation** — Structural consistency of LLM JSON output
4. **Grounding Check** — Lexical overlap between answer and cited sources

## Latency Budget

| Stage | Target | Notes |
|-------|--------|-------|
| Query preprocessing | ~2ms | Tokenization, normalization |
| Dense retrieval (FAISS) | ~5ms | Embedding + FAISS search |
| BM25 retrieval | ~3ms | In-memory tokenized search |
| Score fusion | ~1ms | Normalize + combine |
| Guardrails (pre-gen) | ~5ms | Safety + relevance |
| LLM generation (Groq) | ~200-800ms | Model-dependent, network-bound |
| Guardrails (post-gen) | ~5ms | Grounding + output validation |
| **Total RAG** | **~250-850ms** | Text query → answer |
| Sarvam STT | ~300-800ms | Network-dependent |
| **Total E2E Voice** | **~600-1600ms** | Voice → answer |

> **Note:** LLM generation latency depends heavily on the Groq model and queue
> time. The original <200ms target was designed for llama-3.1-8b-instant (since
> deprecated by Groq). Current model (openai/gpt-oss-20b) includes reasoning
> overhead. Per-stage breakdown is reported transparently in every response.

## Running Tests

```bash
# Unit tests
python -m pytest tests/ -v

# Full pipeline demo scenarios (requires API keys)
python scripts/test_pipeline.py --strategy fixed_token

# Combined benchmark
python scripts/benchmark.py --strategy fixed_token --max-queries 50

# Individual evaluations
python -m evaluation.chunking_eval --strategies all
python -m evaluation.retrieval_eval --strategies all --modes dense,bm25,hybrid
python -m evaluation.latency_eval --strategy fixed_token --num-queries 50
```

## Dataset

**ai4bharat/MSMARCO-XI** — A multilingual extension of MS MARCO with 14 Indic languages + English.

- We use streaming mode to download a controlled subset (default: 100 records)
- English passages and queries are used for retrieval (best embedding model compatibility)
- Sarvam STT supports Hindi/Indic voice input, translated to English for querying
- Each record contains ~10 passages with binary relevance labels (`is_selected`)

## Known Limitations

1. **Safety guardrail** is keyword-based — easily bypassed by rephrasing; not production-grade
2. **Grounding check** is lexical overlap — doesn't detect semantic hallucination
3. **Relevance guardrail** has weak discrimination on broad corpora — calibrate with `--calibrate`
4. **Corpus size** is intentionally small (100 records, ~1000-1500 chunks) for demo; scale to 5K-10K for better coverage
5. **LLM latency** is network-bound and depends on Groq queue times; report separates our pipeline time from Groq's server time
6. **Voice** is limited to 30-second clips (Sarvam REST API limit)

## License

Built for HH Goa 2026 hackathon shortlisting.
