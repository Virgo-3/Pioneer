"""Turn-level routing and interpretation of optional Jev decision signals.

Jev's noul values describe its confidence in *questions about the situation*.
They are never probabilities of a user's outcome or a substitute for the local
decision engine. This module keeps its influence small enough for Dao to
give one ordinary conversational response.
"""

from __future__ import annotations

import math
import re
from typing import Any


_DIRECT_CHOICE = re.compile(
    r"\b(?:should\s+(?:i|we)|what\s+should\s+(?:i|we)|which\s+(?:option|path|approach)"
    r"|(?:i|we)\s+should\s+\w+"
    r"|would\s+you\s+(?:recommend|choose|do)|is\s+it\s+worth\s+it"
    r"|(?:would|will)\s+it\s+be\s+better\s+to|is\s+it\s+better\s+to"
    r"|is\s+(?:it|this|that)\s+(?:a\s+)?(?:good|smart|wise)\s+idea"
    r"|is\s+(?:buying|selling|launching|waiting|acting|committing)\b.{0,32}\b(?:good|smart|wise)\s+idea"
    r"|(?:should|recommend)\s+(?:buying|selling|launching|waiting|acting|committing)"
    r"|(?:i|we)\s+(?:need|have)\s+to\s+(?:decide|choose)"
    r"|(?:decide|choos\w*|choice|trade[ -]?offs?|pros\s+and\s+cons)"
    r"|(?:act\s+now|wait\s+or\s+act|go\s+ahead|hold\s+off))\b",
    re.IGNORECASE,
)
_CONSEQUENTIAL_ACTION = (
    r"(?:quit|quitting|resign|resigning|leave\s+(?:my|our|the)\s+job|"
    r"leaving\s+(?:my|our|the)\s+job|pilot|piloting|launch|launching|"
    r"invest|investing|borrow|borrowing|deploy|deploying|release|releasing|"
    r"buy(?:ing)?\s+(?:a|the)\s+(?:house|home|business)|"
    r"sell(?:ing)?\s+(?:a|the)\s+(?:house|home|business)|"
    r"sign(?:ing)?\s+(?:a|the)\s+contract)"
)
_IMPLICIT_CHOICE = re.compile(
    r"\b(?:i(?:['’]m|\s+am)|we(?:['’]re|\s+are))\s+"
    r"(?:thinking\s+(?:about|of)|considering)\s+" + _CONSEQUENTIAL_ACTION + r"\b"
    r"|\b(?:i|we)\s+(?:might|may|could)\s+" + _CONSEQUENTIAL_ACTION + r"\b",
    re.IGNORECASE,
)
_DEFINITIONAL = re.compile(
    r"^\s*(?:what\s+does\b.+\bmean\b|what\s+is\s+the\s+meaning\s+of\b|"
    r"(?:can\s+you\s+)?define\b|(?:can\s+you\s+)?explain\s+the\s+(?:term|phrase)\b)",
    re.IGNORECASE,
)
_CASE_NUMBER = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?%?|\.\d+)(?!\w)")
_CASE_PAYOFF = re.compile(r"\b(?:pays?|payoffs?|outcomes?|costs?|profits?|losses?)\b", re.IGNORECASE)
_CASE_CHOICE = re.compile(r"\b(?:wait|act|invest|hold|launch|pilot|buy|sell)\b", re.IGNORECASE)
_CASE_STATES = re.compile(r"\b(?:good|bad|success|failure|positive|negative)\b", re.IGNORECASE)
_FOLLOWUP_CUE = re.compile(
    r"\b(?:what\s+if|instead|actually|deadline|due\s+(?:by|on)|cost|payoff|risk|"
    r"undo|reverse|reversible|irreversible|delay|wait|learn|evidence|review|approved?|"
    r"days?|weeks?|months?|years?|hours?|minutes?|"
    r"\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|hours?|minutes?|%|dollars?))\b",
    re.IGNORECASE,
)
_ACKNOWLEDGMENT = re.compile(
    r"(?:hi|hello|hey|thanks?|thank\s+you|ok(?:ay)?|got\s+it|sounds\s+good|"
    r"makes\s+sense|bye|goodbye)[.!\s]*",
    re.IGNORECASE,
)
_STOPWORDS = {
    "about", "after", "again", "and", "are", "before", "could", "for", "from", "have",
    "into", "more", "need", "next", "our", "should", "that", "the", "their", "there",
    "these", "this", "those", "what", "when", "which", "with", "would", "your",
}
_TOPIC_STOPWORDS = _STOPWORDS | {
    "accept", "accepting", "act", "acting", "again", "best", "better", "borrow",
    "borrowing", "buy", "buying", "choice", "choose", "choosing", "considering",
    "cost", "costs", "could", "day", "days", "deadline",
    "decide", "decision", "delay", "evidence", "friday", "good", "hold", "hour",
    "hours", "information", "invest", "investing", "later", "leave", "leaving",
    "maybe", "might", "month", "months", "move", "moving", "next",
    "option", "options", "payoff", "payoffs", "plan", "plans", "quit", "quitting",
    "resign", "resigning", "result", "results", "risk", "risks", "sell", "selling",
    "should", "sign", "signing", "someday", "switch", "switching", "take",
    "taking", "thinking", "time", "timing", "today", "tomorrow",
    "wait", "waiting", "week", "weeks", "year", "years", "yesterday",
}
_ATTENTION_NAMES = ("assumption_tension", "urgency", "reversibility", "information_value")
_SCORE_NAMES = {
    "assumption_tension": "assumption_tension",
    "urgency": "time_sensitive",
    "reversibility": "hard_to_reverse",
    "information_value": "missing_information",
}


def _terms(value: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z0-9]+", value.lower())
            if len(word) > 3 and word not in _STOPWORDS}


def _context_terms(context: dict[str, Any]) -> set[str]:
    pieces = [str(context.get("goal") or "")]
    for key in ("options", "known", "uncertain", "next_questions"):
        values = context.get(key)
        if isinstance(values, list):
            pieces.extend(value for value in values if isinstance(value, str))
    return _terms(" ".join(pieces))


def _topic_terms(value: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z0-9]*", value.lower())
            if len(word) >= 3 and word not in _TOPIC_STOPWORDS}


def _same_topic_word(left: str, right: str) -> bool:
    # Inflected actions such as launch/launching and pilot/piloting still refer
    # to the same subject. Requiring four letters avoids car/career matches.
    return left == right or (min(len(left), len(right)) >= 4
                             and (left.startswith(right) or right.startswith(left)))


def _decision_case(text: str) -> bool:
    actions = {item.lower() for item in _CASE_CHOICE.findall(text)}
    return (len(_CASE_NUMBER.findall(text)) >= 2 and bool(_CASE_PAYOFF.search(text))
            and (len(actions) >= 2 or bool(actions and _CASE_STATES.search(text))))


def select_jev_context(text: str, context: dict | None) -> dict | None:
    """Keep active context unless a clear choice introduces another subject.

    A short question such as "Should I wait?" and a bare answer do not contain
    enough topic evidence to discard the ongoing decision. The lexical check is
    deliberately conservative; ambiguous turns retain context so Jev can read
    the exchange rather than treat each message as a fresh decision.
    """
    if not isinstance(context, dict) or context.get("status") not in {"active", "resolved"}:
        return None
    message = text.strip()
    if not (_DIRECT_CHOICE.search(message) or _IMPLICIT_CHOICE.search(message)
            or _decision_case(message)):
        return context
    latest = _topic_terms(message)
    if not latest:
        return context
    pieces = [str(context.get("goal") or "")]
    for key in ("options", "known", "uncertain", "next_questions"):
        values = context.get(key)
        if isinstance(values, list):
            pieces.extend(value for value in values if isinstance(value, str))
    previous = _topic_terms(" ".join(pieces))
    if any(_same_topic_word(new, old) for new in latest for old in previous):
        return context
    return None


def should_consult_jev(text: str, context: dict | None) -> bool:
    """Route decision turns to Jev while leaving ordinary chat alone.

    This is deliberately lexical and conservative. A missed case still reaches
    OpenAI's normal decision instructions; a false positive costs an extra API
    call. The active-context check lets new facts continue an existing decision
    without requiring the user to repeat "should I" every turn.
    """
    message = text.strip()
    if not message or _ACKNOWLEDGMENT.fullmatch(message):
        return False
    if _DEFINITIONAL.search(message):
        return False
    if _DIRECT_CHOICE.search(message) or _IMPLICIT_CHOICE.search(message):
        return True
    # A complete-looking prose case is a decision turn even if it contains no
    # question. Two figures, a payoff/cost term, and an action term avoid
    # routing every casual mention of a number to Jev.
    if _decision_case(message):
        return True
    if not isinstance(context, dict) or context.get("status") not in {"active", "resolved"}:
        return False

    if _terms(message) & _context_terms(context):
        return True
    if _FOLLOWUP_CUE.search(message):
        return True
    # A bare answer is meaningful when Dao has a pending decision question.
    questions = context.get("next_questions")
    if (isinstance(questions, list) and questions and len(message.split()) <= 8
            and re.fullmatch(r"(?:yes|no|probably|maybe)[.!\s]*", message, re.IGNORECASE)):
        return True
    return False


def _attention(score: Any) -> str:
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        return "background"
    # These are routing thresholds, not outcome-probability cutoffs. A signal
    # below 0.55 does not steer the reply; 0.55-0.74 invites a check; 0.75+
    # can become one of at most two focal points.
    if score >= 0.75:
        return "focus"
    if score >= 0.55:
        return "check"
    return "background"


def make_jev_guidance(scores: dict[str, float], context: dict | None) -> dict[str, Any]:
    """Summarize Jev scores as bounded attention data for this turn.

    An assumption-tension score does not establish a contradiction or citation;
    those need the conversation and source verification.
    """
    active_decision = isinstance(context, dict) and context.get("status") in {"active", "resolved"}
    # A lexical router can be fooled by wording such as "which option" in an
    # explanatory question. Jev's low decision-request signal suppresses its
    # attention for a new topic. During an active decision, factual
    # followups remain eligible even when they do not ask for a choice again.
    eligible_turn = active_decision or _attention(scores.get("decision_request")) != "background"
    attention = {name: _attention(scores.get(source)) if eligible_turn else "background"
                 for name, source in _SCORE_NAMES.items()}
    eligible = [name for name in _ATTENTION_NAMES if attention[name] != "background"]
    # Verified prior statements are easy to lose in a long conversation. If
    # tension is plausible, examine it first, then the other strongest cues.
    eligible.sort(key=lambda name: (
        name != "assumption_tension",
        attention[name] != "focus",
        -float(scores[_SCORE_NAMES[name]]),
    ))
    priorities = eligible[:2]
    return {
        "attention": attention,
        "priorities": priorities,
    }
