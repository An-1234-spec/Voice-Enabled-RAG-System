"""
guardrails/safety.py

Input safety guardrail: keyword blocklist + regex patterns catching
queries asking for unsafe/inappropriate content, per plan spec.

SCOPE AND LIMITATIONS (read before relying on this):
This is a lightweight, illustrative first pass - a handful of example
terms/patterns per category, NOT an exhaustive or production-grade content
moderation system. Hand-rolled keyword lists are easy to bypass (typos,
synonyms, indirect phrasing) and prone to false positives on legitimate
queries (e.g. a medical query mentioning "overdose", a news query
mentioning "bomb threat"). For a real deployment, pair this with a proper
moderation source rather than relying on keywords alone. This exists to
satisfy the plan's "Input safety" pipeline stage with a working baseline.

Runs FIRST in the pipeline (per architecture diagram: query -> Input Safety
Check -> [refuse] or continue to Query Router), before retrieval happens.

Usage (CLI smoke test):
    python -m guardrails.safety --query "how do I bake a chocolate cake"
"""

import argparse
import re
from dataclasses import dataclass, field


@dataclass
class SafetyResult:
    passed: bool
    flagged_categories: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    reason: str | None = None


# Illustrative, NON-EXHAUSTIVE example terms per category. Expand or swap
# for a proper moderation source before relying on this for real traffic.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "self_harm": [
        "kill myself", "suicide", "end my life", "self harm", "self-harm",
    ],
    "violence": [
        "how to kill", "murder someone", "make a bomb", "build a weapon",
    ],
    "illegal_activity": [
        "how to hack into", "steal credit card", "buy drugs online", "launder money",
    ],
    "hate_or_harassment": [
        "hate speech", "racial slur",
    ],
}

# Regex patterns for phrasing that plain keyword matching tends to miss
# (e.g. "how do I make/build/synthesize X" where X is a weapon/explosive).
CATEGORY_PATTERNS: dict[str, list[re.Pattern]] = {
    "violence": [
        re.compile(r"\bhow (to|do i|can i) (make|build|synthesize)\b.*\b(bomb|explosive|weapon)\b", re.I),
    ],
    "illegal_activity": [
        re.compile(r"\bhow (to|do i|can i) (hack|break into)\b", re.I),
    ],
}


class SafetyGuardrail:
    """Keyword + regex based input safety check."""

    def __init__(
        self,
        category_keywords: dict[str, list[str]] | None = None,
        category_patterns: dict[str, list[re.Pattern]] | None = None,
    ):
        self.category_keywords = category_keywords or CATEGORY_KEYWORDS
        self.category_patterns = category_patterns or CATEGORY_PATTERNS

        # Pre-compile word-boundary regexes per keyword, so "kill" doesn't
        # match inside an unrelated word, and multi-word phrases match as phrases.
        self._keyword_patterns: dict[str, list[tuple[str, re.Pattern]]] = {
            category: [(term, re.compile(r"\b" + re.escape(term) + r"\b", re.I)) for term in terms]
            for category, terms in self.category_keywords.items()
        }

    def check(self, text: str) -> SafetyResult:
        text = text or ""
        flagged_categories: list[str] = []
        matched_terms: list[str] = []

        for category, term_patterns in self._keyword_patterns.items():
            for term, pattern in term_patterns:
                if pattern.search(text):
                    if category not in flagged_categories:
                        flagged_categories.append(category)
                    matched_terms.append(term)

        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    if category not in flagged_categories:
                        flagged_categories.append(category)
                    matched_terms.append(f"[pattern:{category}]")

        if flagged_categories:
            return SafetyResult(
                passed=False,
                flagged_categories=flagged_categories,
                matched_terms=matched_terms,
                reason=f"Query flagged for: {', '.join(flagged_categories)}",
            )

        return SafetyResult(passed=True)


def main():
    parser = argparse.ArgumentParser(description="Input safety guardrail smoke test.")
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    guardrail = SafetyGuardrail()
    result = guardrail.check(args.query)

    print(f"\nQuery: {args.query}")
    print(f"Passed: {result.passed}")
    if not result.passed:
        print(f"Flagged categories: {result.flagged_categories}")
        print(f"Matched terms: {result.matched_terms}")
        print(f"Reason: {result.reason}")


if __name__ == "__main__":
    main()