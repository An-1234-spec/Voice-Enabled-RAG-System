"""
guardrails/output_validator.py

Structural validation of an LLMResult (generation/llm.py) - separate from
JSON parsing itself (already handled in llm.py's parse_error flag). Checks
the parsed fields are internally consistent: sources_used tags actually
resolve to retrieved chunks, grounded answers have sources, refused
answers have a reason.
"""

from dataclasses import dataclass, field

from generation.llm import LLMResult
from generation.prompts import tag_to_chunk_id


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


def validate(result: LLMResult, retrieved_chunks: list[dict]) -> ValidationResult:
    errors = []

    if result.parse_error:
        reason = result.refusal_reason or "LLM output parsing or generation failed"
        errors.append(reason)
        return ValidationResult(valid=False, errors=errors)  # nothing else to check if parsing failed

    if not isinstance(result.answer, str):
        errors.append("'answer' is not a string")
    if not isinstance(result.grounded, bool):
        errors.append("'grounded' is not a boolean")

    if not isinstance(result.sources_used, list):
        errors.append("'sources_used' is not a list")
    else:
        # FIX: one hallucinated/out-of-range tag used to fail the ENTIRE
        # response, forcing a full retry even when e.g. [S1] was correct
        # and only [S4] was invented. Filter unresolvable tags instead --
        # grounding.py's own signals still verify whatever valid citations
        # remain, so this doesn't weaken correctness, just stops
        # double-penalizing a partially-good citation list.
        valid_tags = [t for t in result.sources_used if tag_to_chunk_id(retrieved_chunks, t) is not None]
        if len(valid_tags) != len(result.sources_used):
            result.sources_used = valid_tags  # mutate in place; grounding sees the cleaned list

    if result.grounded and not result.sources_used:
        errors.append("grounded=true but sources_used is empty")
    if not result.grounded and not result.refusal_reason:
        errors.append("grounded=false but refusal_reason is missing")

    return ValidationResult(valid=len(errors) == 0, errors=errors)