"""Retrieve cited conversation claims from the active branch."""

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
RECALL_CUE = re.compile(
    r"\b(?:remember|recall|earlier|previously|originally|beginning|history|"
    r"(?:what|why) did (?:i|you|we) say|what have (?:i|you|we) said|"
    r"(?:i|you|we) (?:said|claimed|argued)|(?:didn't|did not) you say|"
    r"challenge (?:that|this|it|your|my)|where did (?:that|this|it) come from|"
    r"(?:did (?:you|we|pioneer)|what did (?:you|we|pioneer)) "
    r"(?:save|record|log|store|track)|"
    r"have (?:you|we|pioneer) (?:saved|recorded|logged|stored|tracked)|"
    r"(?:was|is) (?:my|our|the) (?:target|goal|forecast|prediction|outcome|result|decision|record) "
    r"(?:saved|recorded|logged|stored|tracked))\b",
    re.IGNORECASE,
)
EARLIEST_CUE = re.compile(r"\b(first|earliest|start|beginning|originally)\b", re.IGNORECASE)
USER_RECALL = re.compile(r"\b(?:(?:what|why) did i say|what have i said|i (?:said|claimed|argued)|my (?:claim|answer))\b", re.IGNORECASE)
ASSISTANT_RECALL = re.compile(r"\b(?:(?:what|why) did you say|what have you said|you (?:said|claimed|argued)|"
                              r"your (?:claim|answer|advice)|where did (?:that|this|it) come from|"
                              r"challenge (?:that|this|it))\b", re.IGNORECASE)
FOLLOWUP_CUE = re.compile(r"""
    ^\s*(?:
        (?:and\s+)?now(?:\s+what)? | what\s+now | (?:and\s+)?then |
        what\s+about\s+(?:now|that|this|it) |
        what\s+should\s+we\s+do\s+now |
        do\s+you\s+still\s+think\s+so |
        is\s+that\s+still\s+(?:true|your\s+view) |
        do\s+it | go\s+ahead | let['’]s\s+do\s+it |
        what\s+changed | why(?:\s+not)? | how\s+so | really | are\s+you\s+sure
    )[\s?!.]*$
""", re.IGNORECASE | re.VERBOSE)
RECALL_TERMS = frozenset("remember recall earlier previously originally beginning first earliest start said say history conversation before claimed challenge did come".split())
STOP_WORDS = frozenset("a an and are as at be been by can do for from had has have how i if in is it its me my of on or our should that the their them there these this to us was we were what when where which who why will with would you your".split())


def _terms(value: str) -> set[str]:
    return {word for word in WORD.findall(value.lower()) if len(word) > 2 and word not in STOP_WORDS}


def _query_terms(value: str) -> set[str]:
    terms = _terms(value) - RECALL_TERMS
    if terms & {"target", "goal", "objective"}:
        terms.update({"target", "goal", "objective"})
    return terms


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
                     recent_turns: int = RECENT_TURNS,
                     include_recent: bool = False,
                     followup_text: str | None = None) -> list[dict[str, str]]:
    """Rank user and assistant claims with branch, citation, and size bounds.

    Ordinary topic retrieval favors turns outside the recent prompt. Explicit
    recall or challenge requests can cite recent turns too. Quotes are data for
    evaluation, never instructions or verified facts.
    """
    turns = [(object_id, obj["payload"]) for object_id, obj in store.log(branch)
             if obj["kind"] == "turn" and isinstance(obj.get("payload"), dict)]
    # During post-answer review, only the user's words can open a zero-overlap
    # fallback. The authored reply itself must not cause a history review.
    cue_text = followup_text if followup_text is not None else current
    recall = bool(RECALL_CUE.search(cue_text))
    followup = bool(include_recent and followup_text is not None
                    and FOLLOWUP_CUE.fullmatch(followup_text))
    allow_recent = recall or include_recent
    candidates = turns if allow_recent else turns[max(0, recent_turns):]
    if not candidates:
        return []
    current_terms = _query_terms(current)
    context_terms = (_terms(" ".join(str(context.get(field, "")) for field in ("goal", "options", "known")))
                     if context and current_terms else set())
    if not current_terms and not context_terms and not allow_recent:
        return []
    preferred_role = ("user" if USER_RECALL.search(cue_text) else
                      "assistant" if ASSISTANT_RECALL.search(cue_text) or followup else None)
    earliest = bool(EARLIEST_CUE.search(cue_text))
    ranked: list[tuple[int, int, int, str, str, str, str | None]] = []
    for age, (object_id, payload) in enumerate(candidates):
        for role in ("user", "assistant"):
            statement = payload.get(role)
            if not isinstance(statement, str) or not statement.strip():
                continue
            terms = _terms(statement)
            score = 3 * len(terms & current_terms) + len(terms & context_terms)
            role_match = int(role == preferred_role)
            provider = payload.get("provider") if role == "assistant" else None
            ranked.append((score, role_match, age, object_id, role, statement,
                           provider if isinstance(provider, str) and provider else None))
    if not ranked:
        return []
    if any(item[0] for item in ranked):
        ranked = [item for item in ranked if item[0]]
        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2] if earliest else item[2]))
    elif recall:
        ranked.sort(key=lambda item: (-item[2] if earliest else -item[1],
                                      -item[1] if earliest else item[2]))
    elif followup:
        recent_count = min(len(turns), max(1, recent_turns))
        ranked = [item for item in ranked if item[2] < recent_count]
        ranked.sort(key=lambda item: (item[2], -item[1]))
    else:
        return []

    # A request to inspect the conversation needs a citation for at least one
    # recent turn, even if older matches dominate the lexical ranking.
    if recall or followup:
        recent_count = min(len(turns), max(1, recent_turns))
        if not any(item[2] < recent_count for item in ranked[:MAX_EVIDENCE]):
            recent = [item for item in ranked if item[2] < recent_count]
            if not recent:
                recent = []
                for age, (object_id, payload) in enumerate(turns[:recent_count]):
                    for role in ("user", "assistant"):
                        statement = payload.get(role)
                        if isinstance(statement, str) and statement.strip():
                            provider = payload.get("provider") if role == "assistant" else None
                            recent.append((0, int(role == preferred_role), age, object_id, role,
                                           statement, provider if isinstance(provider, str) and provider else None))
                recent.sort(key=lambda item: (-item[1], item[2]))
            if recent:
                ranked = ranked[:MAX_EVIDENCE - 1] + [recent[0]] + ranked[MAX_EVIDENCE:]
    selected: list[dict[str, str]] = []
    remaining = MAX_TOTAL_CHARS
    seen: set[tuple[str, str]] = set()
    for _, _, _, object_id, role, statement, provider in ranked:
        if len(selected) >= MAX_EVIDENCE or remaining <= 0:
            break
        quote = _excerpt(statement, current_terms, context_terms, min(MAX_QUOTE_CHARS, remaining))
        if quote and (role, quote) not in seen:
            item = {"commit": object_id, "role": role, "quote": quote}
            if provider is not None:
                item["provider"] = provider
            selected.append(item)
            remaining -= len(quote)
            seen.add((role, quote))
    return selected


def verified_conflict(conflict: dict[str, str] | None,
                      evidence: list[dict[str, str]]) -> dict[str, str] | None:
    """Verify provenance for a challenge, not its interpretation or truth."""
    if not isinstance(conflict, dict):
        return None
    commit, role, quote, challenge = (conflict.get(key) for key in ("commit", "role", "quote", "challenge"))
    if (not all(isinstance(value, str) for value in (commit, role, quote, challenge))
            or role not in {"user", "assistant"} or not quote.strip() or len(quote) > 220
            or not challenge.strip() or len(challenge) > 400):
        return None
    for item in evidence:
        if item.get("commit") == commit and item.get("role") == role and quote in item.get("quote", ""):
            verified = {"commit": commit, "role": role, "quote": quote,
                        "challenge": challenge.strip()}
            if isinstance(item.get("provider"), str):
                verified["provider"] = item["provider"]
            return verified
    return None
