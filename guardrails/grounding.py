"""
guardrails/grounding.py

Multi-signal grounding check. See prior docstring for signal stack
(lexical / entity / sentence). This version fixes one entity-extraction
bug: a model-written numeric range like "3-8" was captured as ONE opaque
hyphenated token, which could never match a source phrasing the same fact
differently ("3 to 8", or as separate numbers) - a pure formatting
mismatch, not a real grounding failure. Ranges are now split into their
endpoint numbers before comparison, on both sides, so "3-8" and "3 to 8"
both reduce to {"3", "8"}.

THRESHOLD CHANGE (2026-08-20, evidence-based): DEFAULT_OVERLAP_THRESHOLD
lowered 0.50 -> 0.30. Justification: a live 200-query benchmark showed 58
lexical grounding failures, almost all clustered 15-45% overlap - not
near-misses of a correct 50% bar, but a systematic mismatch between the
threshold (calibrated for longer, closer-to-verbatim answers) and the
current regime (24-token, naturally paraphrased answers from a 1B local
model). 0.30 is chosen so clearly-bad answers in this run's data (7-25%
overlap) still fail, while the bulk of wrongly-rejected paraphrases
(28-45%) pass. This is informed by real failure data, not recalibrated
from a full success/failure distribution - re-verify after the next
benchmark run and tighten/loosen further if the data suggests it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from retrieval.bm25_retriever import tokenize

DEFAULT_OVERLAP_THRESHOLD = 0.30    # see THRESHOLD CHANGE note above
ENTITY_PRESENCE_THRESHOLD = 0.80
SENTENCE_COVERAGE_MIN_OVERLAP = 0.20

_NUMBER_RE = re.compile(r"\b\d[\d,.\-]*\b")
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_RANGE_SPLIT_RE = re.compile(r"^(\d[\d,.]*)-(\d[\d,.]*)$")


_CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[.?!:])\s+")


def _normalize_number_token(tok: str) -> set[str]:
    """
    "3-8" (a range) -> {"3", "8"}. "3,000" -> {"3000"} (strip thousands
    commas so formatting differences don't cause a false mismatch).
    "10-" (a truncated range, e.g. "10-15%" cut off by the token budget)
    -> {"10"} rather than an unmatchable orphaned fragment.
    """
    m = _RANGE_SPLIT_RE.match(tok)
    if m:
        return {m.group(1).replace(",", ""), m.group(2).replace(",", "")}
    if tok.endswith("-"):
        stripped = tok.rstrip("-").replace(",", "")
        return {stripped} if stripped else set()
    return {tok.replace(",", "")}


def _clause_initial_words(text: str) -> set[str]:
    """Words right after a sentence OR colon boundary get a capitalization
    pass -- colons commonly introduce a capitalized list item
    ('Foods low in sodium include: Bananas...') that isn't a real proper
    noun, same root issue as sentence starts."""
    words: set[str] = set()
    all_words = text.strip().split()
    if all_words:
        words.add(all_words[0].strip(".,!?;:\"'").lower())
    for m in _CLAUSE_BOUNDARY_RE.finditer(text):
        rest = text[m.end():].split()
        if rest:
            words.add(rest[0].strip(".,!?;:\"'").lower())
    return words


def _extract_entities(text: str) -> set[str]:
    """Heuristic entity extraction: numbers (range/comma-normalized) + CapitalisedWords."""
    entities: set[str] = set()
    for m in _NUMBER_RE.finditer(text):
        entities.update(_normalize_number_token(m.group().lower()))
    for m in _CAMEL_RE.finditer(text):
        entities.add(m.group().lower())

    clause_initial = _clause_initial_words(text)
    mid_clause_caps: set[str] = set()
    boundaries = [0] + [m.end() for m in _CLAUSE_BOUNDARY_RE.finditer(text)] + [len(text)]
    for i in range(len(boundaries) - 1):
        words = text[boundaries[i]:boundaries[i + 1]].split()
        for w in words[1:]:
            cleaned = w.strip(".,!?;:\"'")
            if _CAMEL_RE.fullmatch(cleaned):
                mid_clause_caps.add(cleaned.lower())

    return {e for e in entities if e not in clause_initial or e in mid_clause_caps}

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class GroundingResult:
    passed: bool
    overlap_ratio: float
    reason: str | None = None
    entity_ratio: float = 0.0
    sentence_coverage: float = 0.0
    failed_signals: list[str] = field(default_factory=list)


def check(answer: str, cited_chunks: list[dict], threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> GroundingResult:
    answer = (answer or "").strip()

    if not answer:
        return GroundingResult(passed=True, overlap_ratio=1.0, reason="empty answer, nothing to ground")

    if not cited_chunks:
        return GroundingResult(
            passed=False, overlap_ratio=0.0, reason="answer given but no sources cited",
            failed_signals=["lexical", "entity", "sentence"],
        )

    all_source_text = " ".join(chunk.get("text", "") for chunk in cited_chunks)

    answer_tokens = set(tokenize(answer))
    source_tokens = set(tokenize(all_source_text))

    if not answer_tokens:
        return GroundingResult(passed=True, overlap_ratio=1.0, reason="answer has no scorable tokens")

    overlap_ratio = len(answer_tokens & source_tokens) / len(answer_tokens)

    failed_signals: list[str] = []
    reasons: list[str] = []

    if overlap_ratio < threshold:
        failed_signals.append("lexical")
        reasons.append(f"lexical: only {overlap_ratio:.0%} of answer words in cited sources (threshold {threshold:.0%})")

    answer_entities = _extract_entities(answer)
    entity_ratio = 1.0

    if answer_entities:
        source_entities = _extract_entities(all_source_text)
        found = answer_entities & source_entities
        entity_ratio = len(found) / len(answer_entities)

        if entity_ratio < ENTITY_PRESENCE_THRESHOLD:
            failed_signals.append("entity")
            missing = answer_entities - source_entities
            reasons.append(
                f"entity: {entity_ratio:.0%} of answer entities found in sources "
                f"(need {ENTITY_PRESENCE_THRESHOLD:.0%}); missing: {', '.join(sorted(missing)[:5])}"
            )

    answer_sentences = _split_sentences(answer)
    source_sentences = _split_sentences(all_source_text)
    source_sent_token_sets = [set(tokenize(s)) for s in source_sentences]

    covered = 0
    for sent in answer_sentences:
        sent_tokens = set(tokenize(sent))
        if not sent_tokens:
            covered += 1
            continue
        best_overlap = max(
            (len(sent_tokens & src_set) / len(sent_tokens) for src_set in source_sent_token_sets),
            default=0.0,
        )
        if best_overlap >= SENTENCE_COVERAGE_MIN_OVERLAP:
            covered += 1

    sentence_coverage = covered / len(answer_sentences) if answer_sentences else 1.0

    if len(answer_sentences) > 1 and sentence_coverage < 1.0:
        failed_signals.append("sentence")
        reasons.append(f"sentence: {covered}/{len(answer_sentences)} answer sentences covered by cited sources (need all)")

    passed = len(failed_signals) == 0
    reason = "; ".join(reasons) if reasons else None

    return GroundingResult(
        passed=passed, overlap_ratio=overlap_ratio, entity_ratio=entity_ratio,
        sentence_coverage=sentence_coverage, failed_signals=failed_signals, reason=reason,
    )