"""Retrieve relevant, cited user statements from older active-branch turns."""

from __future__ import annotations

import re
from typing import Any

from .state import Store


RECENT_TURNS = 20
MAX_EVIDENCE = 5
MAX_QUOTE_CHARS = 700
MAX_TOTAL_CHARS = 3000
WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP_WORDS = frozenset("a an and are as at be been by can do for from had has have how i if in is it its me my of on or our should that the their them there these this to us was we were what when where which who why will with would you your".split())


def _terms(value: str) -> set[str]:
    return {word for word in WORD.findall(value.lower()) if len(word) > 2 and word not in STOP_WORDS}


def retrieve_history(store: Store, branch: str, current: str,
                     context: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Rank older user turns by lexical relevance, with strict branch and size bounds.

    The recent turns already appear verbatim in the conversational input. Older
    statements are quotations for the model to evaluate, never instructions or
    numerical evidence for the deterministic decision engine.
    """
    turns = [(object_id, obj["payload"]["user"]) for object_id, obj in store.log(branch)
             if obj["kind"] == "turn" and isinstance(obj["payload"].get("user"), str)]
    if len(turns) <= RECENT_TURNS:
        return []
    current_terms = _terms(current)
    context_terms = _terms(" ".join(str(context.get(field, "")) for field in ("goal", "options", "known"))) if context else set()
    if not current_terms and not context_terms:
        return []
    ranked: list[tuple[int, int, str, str]] = []
    for age, (object_id, statement) in enumerate(turns[RECENT_TURNS:]):
        terms = _terms(statement)
        score = 3 * len(terms & current_terms) + len(terms & context_terms)
        if score:
            ranked.append((score, -age, object_id, statement))
    ranked.sort(reverse=True)
    selected: list[dict[str, str]] = []
    remaining = MAX_TOTAL_CHARS
    for _, _, object_id, statement in ranked:
        if len(selected) >= MAX_EVIDENCE or remaining <= 0:
            break
        quote = statement[:min(MAX_QUOTE_CHARS, remaining)].strip()
        if quote:
            selected.append({"commit": object_id, "quote": quote})
            remaining -= len(quote)
    return selected


def verified_conflict(conflict: dict[str, str] | None,
                      evidence: list[dict[str, str]]) -> dict[str, str] | None:
    """Accept a proposed challenge only when its citation and quote are real."""
    if not isinstance(conflict, dict):
        return None
    commit, quote, challenge = (conflict.get(key) for key in ("commit", "quote", "challenge"))
    if (not all(isinstance(value, str) for value in (commit, quote, challenge))
            or not quote.strip() or len(quote) > 220 or not challenge.strip() or len(challenge) > 400):
        return None
    for item in evidence:
        if item["commit"] == commit and quote in item["quote"]:
            return {"commit": commit, "quote": quote, "challenge": challenge.strip()}
    return None
