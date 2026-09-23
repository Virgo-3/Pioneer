"""Retrieve relevant, cited user statements from older active-branch turns."""

from __future__ import annotations

import re
from typing import Any

from .state import Store


RECENT_TURNS = 20
MAX_RECENT_CHARS = 24_000
MAX_EVIDENCE = 5
MAX_QUOTE_CHARS = 700
MAX_TOTAL_CHARS = 3000
WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
RECALL_CUE = re.compile(r"\b(remember|recall|earlier|previously|originally|beginning|what did i say|what have i said)\b", re.IGNORECASE)
EARLIEST_CUE = re.compile(r"\b(first|earliest|start|beginning|originally)\b", re.IGNORECASE)
RECALL_TERMS = frozenset("remember recall earlier previously originally beginning first earliest start said say history conversation before".split())
STOP_WORDS = frozenset("a an and are as at be been by can do for from had has have how i if in is it its me my of on or our should that the their them there these this to us was we were what when where which who why will with would you your".split())


def _terms(value: str) -> set[str]:
    return {word for word in WORD.findall(value.lower()) if len(word) > 2 and word not in STOP_WORDS}


def recent_messages(store: Store, branch: str) -> tuple[list[dict[str, str]], int]:
    """Keep complete recent turn pairs within both turn and character budgets."""
    messages = store.messages(branch)[-2 * RECENT_TURNS:]
    selected: list[list[dict[str, str]]] = []
    remaining = MAX_RECENT_CHARS
    for index in range(len(messages) - 2, -1, -2):
        pair = messages[index:index + 2]
        size = sum(len(item["content"]) for item in pair)
        if size > remaining:
            break
        selected.append(pair)
        remaining -= size
    selected.reverse()
    return [message for pair in selected for message in pair], len(selected)


def _excerpt(statement: str, current_terms: set[str], context_terms: set[str], limit: int) -> str:
    if len(statement) <= limit:
        return statement.strip()
    positions = [match.start() for match in WORD.finditer(statement)
                 if match.group().lower() in current_terms | context_terms]
    if not positions:
        return statement[:limit].strip()
    # Search a bounded spread of candidate windows, including matches near the end.
    if len(positions) > 64:
        positions = positions[:24] + positions[-24:] + [positions[index * (len(positions) - 1) // 15]
                                                    for index in range(16)]
    best_start, best_score = 0, -1
    for position in positions:
        start = max(0, min(position - limit // 2, len(statement) - limit))
        terms = _terms(statement[start:start + limit])
        score = 3 * len(terms & current_terms) + len(terms & context_terms)
        if score > best_score:
            best_start, best_score = start, score
    return statement[best_start:best_start + limit].strip()


def retrieve_history(store: Store, branch: str, current: str,
                     context: dict[str, Any] | None = None,
                     recent_turns: int = RECENT_TURNS) -> list[dict[str, str]]:
    """Rank older user turns by lexical relevance, with strict branch and size bounds.

    The recent turns already appear verbatim in the conversational input. Older
    statements are quotations for the model to evaluate, never instructions or
    numerical evidence for the deterministic decision engine.
    """
    turns = [(object_id, obj["payload"]["user"]) for object_id, obj in store.log(branch)
             if obj["kind"] == "turn" and isinstance(obj["payload"].get("user"), str)]
    if len(turns) <= recent_turns:
        return []
    current_terms = _terms(current) - RECALL_TERMS
    context_terms = (_terms(" ".join(str(context.get(field, "")) for field in ("goal", "options", "known")))
                     if context and current_terms else set())
    recall = bool(RECALL_CUE.search(current))
    if not current_terms and not context_terms and not recall:
        return []
    ranked: list[tuple[int, int, str, str]] = []
    older = turns[recent_turns:]
    for age, (object_id, statement) in enumerate(older):
        terms = _terms(statement)
        score = 3 * len(terms & current_terms) + len(terms & context_terms)
        if score:
            ranked.append((score, -age, object_id, statement))
    ranked.sort(reverse=True)
    if not ranked and recall:
        source = reversed(older) if EARLIEST_CUE.search(current) else older
        ranked = [(1, -age, object_id, statement) for age, (object_id, statement) in enumerate(source)]
    selected: list[dict[str, str]] = []
    remaining = MAX_TOTAL_CHARS
    seen: set[str] = set()
    for _, _, object_id, statement in ranked:
        if len(selected) >= MAX_EVIDENCE or remaining <= 0:
            break
        quote = _excerpt(statement, current_terms, context_terms, min(MAX_QUOTE_CHARS, remaining))
        if quote and quote not in seen:
            selected.append({"commit": object_id, "quote": quote})
            remaining -= len(quote)
            seen.add(quote)
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
