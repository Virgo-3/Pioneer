"""One conversational turn: attend to the decision, plan, calculate, and commit."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

from .calibration import (CalibrationError, calibration_context, calibration_report,
                          forecast_records, validate_forecast)
from .decision import DecisionError, analyze
from .history import RECALL_CUE, recent_messages, retrieve_history, verified_conflict
from .jev import make_jev_guidance, select_jev_context, should_consult_jev
from .objectives import (ObjectiveError, compare_objective, objective_records,
                         outcome_report, validate_objective)
from .outcomes import (OutcomeError, outcome_report as unified_outcome_report,
                       outcome_threads, validate_outcome_forecast)
from .providers import ProviderError, assess_jev, compose_turn, review_history
from .state import Store, StoreError


NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?|(?<![\w\d])[-+]?\.\d+%?")
WAIT_WORD = re.compile(r"\b(wait|waiting|delay|defer|postpone|hold off)\b", re.IGNORECASE)
UNDO_WORD = re.compile(r"\b(undo|revert|reverse|reversible|rollback|roll back)\b", re.IGNORECASE)
PERSISTENCE_CLAIM = re.compile(
    r"\b(?:(?:i|we|pioneer)(?:['’]ve| have)?\s+"
    r"(?:(?:already|just|now)\s+)?(?:saved|logged|stored|tracked|recorded|marked)\s+"
    r"(?:(?:your|my|the|this|that|a|an)\s+)?"
    r"(?:(?:new|revised|updated|changed|latest|current|another|different|old|previous|prior)\s+)?"
    r"(?P<active_kind>forecast outcome|reported outcome|forecast|prediction|target|goal|"
    r"actual|result|outcome|record)\b|"
    r"(?:your|my|the|this)\s+"
    r"(?:(?:new|revised|updated|changed|latest|current|another|different|old|previous|prior)\s+)?"
    r"(?P<passive_kind>forecast outcome|reported outcome|forecast|prediction|target|goal|"
    r"actual|result|outcome|record)\s+(?:has been|was|is)\s+"
    r"(?:(?:already|just|now)\s+)?(?:saved|logged|stored|tracked|recorded|marked)\b|"
    r"(?:i|we|pioneer)(?:['’]ve| have)?\s+(?:(?:already|just|now)\s+)?"
    r"(?:saved|logged|stored)\s+(?P<generic_kind>it|that|this|one)\b)",
    re.IGNORECASE,
)
DECISION_RECALL = re.compile(
    r"\b(?:what did (?:you|we) (?:recommend|decide|choose|calculate)|"
    r"what was (?:your|our|the) (?:recommendation|decision|analysis|calculation)|"
    r"why did (?:you|we) (?:recommend|choose|decide|wait|act|pick|prefer)|"
    r"(?:remind me|recap|repeat|explain).{0,60}(?:decision|recommendation|analysis|calculation))\b",
    re.IGNORECASE,
)
DECISION_RECALL_TOPIC = re.compile(
    r"\b(?:for|about|regarding|on)\s+(?:the|my|our|that|this)?\s*([^?.!,;]{2,80})",
    re.IGNORECASE,
)
RECALL_GENERIC_TERMS = {"acti", "case", "choi", "comp", "deci", "opti", "plan", "resu"}
SAVE_HISTORY_QUERY = re.compile(
    r"\b(?:did (?:you|we|pioneer)|have (?:you|we)|was|is|do (?:you|we)|"
    r"are (?:you|we))\b.{0,50}\b(?:save|saved|record|recorded|track|tracked|"
    r"log|logged|store|stored)\b",
    re.IGNORECASE,
)
SPECIFIC_SAVE_CUE = re.compile(r"\b(?:new|revised|updated|changed|latest|current|another|different)\b",
                               re.IGNORECASE)
FORECAST_REQUEST = re.compile(
    r"\b(?:how likely|how confident|what(?:'s| is| are) (?:the )?(?:chances?|odds|probability)|"
    r"what (?:chances?|odds|probability) do you give|"
    r"(?:give|make) (?:me )?(?:your |a )?(?:forecast|probability estimate|odds)|"
    r"(?:give me|what is|what's) (?:a |your |the )?(?:percentage|percent chance)|"
    r"(?:estimate|predict) (?:the )?(?:chances?|odds|probability)|"
    r"(?:can|could|would) you (?:predict|forecast|estimate)|"
    r"your (?:forecast|probability|odds|estimate))\b", re.IGNORECASE)
OUTCOME_CALIBRATION_QUERY = re.compile(
    r"\b(?:calibrat\w*|desired.{0,30}actual|target.{0,30}actual|outcome gap|"
    r"how (?:did|have) (?:i|we) (?:do|done)|how close (?:did|were) (?:i|we))\b", re.IGNORECASE)
FORECAST_ACCURACY_QUERY = re.compile(
    r"\b(?:forecast accuracy|prediction accuracy|brier|"
    r"how accurate.{0,35}(?:forecasts?|predictions?))\b", re.IGNORECASE)
DESIRED_INTENT = re.compile(
    r"\b(?:i|we) (?:want|need|aim|hope|plan|intend|target)|"
    r"\b(?:our|my|the) (?:goal|target|desired outcome|aim)\b|"
    r"\b(?:goal|target) (?:is|was|of|to)\b", re.IGNORECASE)
ACTUAL_REPORT = re.compile(
    r"\b(?:actual|result|outcome|got|reached|achieved|measured|recorded|"
    r"ended|finished|came in|turned out|ended up|we had|i had|we have|i have|"
    r"we're at|i'm at)\b", re.IGNORECASE)
OTHER_MEASURE = re.compile(
    r"\b(?:dollars?|usd|revenue|percent(?:age)?s?|users?|customers?|signups?|"
    r"orders?|sales|leads?|visits?|sessions?|hours?|days?|weeks?|months?)\b|[$%]",
    re.IGNORECASE)
VISIBLE_PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
VISIBLE_PROBABILITY = re.compile(r"(?<![\w.])(?:0\.\d+|1\.0+)(?![\w.%])")
CLAUSE_BREAK = re.compile(r"[.,](?!\d)|[!?;\n]|\bbut\b", re.IGNORECASE)
CORRECTION_CUE = re.compile(
    r"(?:^\s*no\b|\b(?:actually|correction|correct that|wrong|i meant|instead|rather than)\b)",
    re.IGNORECASE)
NEGATIVE_OUTCOME = re.compile(
    r"\b(?:failed|never happened|did not happen|didn't happen|"
    r"(?:did not|didn't) (?:launch|ship|arrive|deliver|pass|win|release|finish|start|open|close|succeed)|"
    r"(?:was not|wasn't|were not|weren't) (?:launched|shipped|delivered|released|finished|started|opened|closed)|"
    r"was cancelled|was canceled|it didn't|missed|did not|didn't|never)\b", re.IGNORECASE)
POSITIVE_OUTCOME = re.compile(
    r"\b(?:happened|occurred|succeeded|completed|launched|shipped|arrived|delivered|"
    r"passed|won|released|finished|started|opened|closed|came true|it did)\b", re.IGNORECASE)
EVENT_STOPWORDS = {"about", "after", "before", "could", "deadline", "event", "from", "happen",
                   "happens", "have", "into", "might", "occur", "should", "their", "there",
                   "these", "those", "would", "with", "will", "when", "that", "this", "they",
                   "then", "than", "were", "your", "chance", "likely", "probability", "forecast",
                   "timing", "decision", "question", "choose", "whether", "options"}


@dataclass(frozen=True)
class TurnOutcome:
    text: str
    commit: str
    model: str
    usage: list[dict[str, Any]]
    decision: dict[str, Any] | None
    branch: str
    branched_from: str | None = None
    notices: tuple[str, ...] = ()
    history_challenge: dict[str, str] | None = None


def _usage(provider: str, result: Any) -> dict[str, Any]:
    return {"provider": provider, "model": result.model if provider == "openai" else result["model"],
            "input_tokens": result.input_tokens if provider == "openai" else result["input_tokens"],
            "output_tokens": result.output_tokens if provider == "openai" else result["output_tokens"],
            **({"response_id": result.response_id} if provider == "openai" else {})}


def _user_numbers(messages: list[dict[str, str]]) -> list[float]:
    found: list[float] = []
    for message in messages:
        if message["role"] != "user":
            continue
        for match in NUMBER.finditer(message["content"]):
            raw = match.group().replace(",", "")
            percent = raw.endswith("%")
            value = float(raw.rstrip("%"))
            found.append(value)
            if percent:
                found.append(value / 100)
    return found


def _case_numbers(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [number for item in value.values() for number in _case_numbers(item)]
    if isinstance(value, list):
        return [number for item in value for number in _case_numbers(item)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return []


def _check_explicit_numbers(case: dict[str, Any], messages: list[dict[str, str]]) -> None:
    supplied = _user_numbers(messages)
    unsupported = [value for value in _case_numbers(case)
                   if not any(math.isclose(value, given, rel_tol=1e-8, abs_tol=1e-8) for given in supplied)]
    if unsupported:
        distinct = list(dict.fromkeys(unsupported))
        raise DecisionError("The drafted case contains numbers you did not provide: "
                            + ", ".join(f"{value:g}" for value in distinct[:8]))


def _check_decision_scope(case: dict[str, Any], messages: list[dict[str, str]]) -> None:
    text = "\n".join(message["content"] for message in messages if message["role"] == "user")
    actions = case.get("actions")
    if not isinstance(actions, dict) or len(actions) < 2:
        raise DecisionError("Please provide at least two actions, including doing nothing if it is a real option")
    if WAIT_WORD.search(text) and "wait" not in case:
        raise DecisionError("To compare waiting, provide a possible signal, its likelihood under each state, and the costs of delay and information")
    if "wait" in case and (not isinstance(case["wait"], dict)
                           or "delay_cost" not in case["wait"]
                           or "information_cost" not in case["wait"]):
        raise DecisionError("Please state both delay cost and information cost explicitly, using 0 if a cost is truly zero")
    if UNDO_WORD.search(text):
        has_undo = any(isinstance(action, dict) and isinstance(action.get("outcomes"), dict)
                       and any(isinstance(outcome, dict) and "undo" in outcome
                               for outcome in action["outcomes"].values())
                       for action in actions.values())
        if not has_undo:
            raise DecisionError("To account for reversal, provide the payoff after undo and its cost in the relevant outcomes")


def _direct_case(text: str) -> dict[str, Any] | None:
    if not text.lstrip().startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and "states" in data and "actions" in data else None


def format_analysis(case: dict[str, Any], result: dict[str, Any]) -> str:
    def label(value: str) -> str:
        return value.replace("_", " ")

    units = result["units"]
    states = ", ".join(f"{label(name)} {probability:.1%}" for name, probability in case["states"].items())
    best_now = result["best_now"]
    now_value = result["best_now_value"]
    if "wait" in result:
        wait = result["wait"]
        information_value = wait["value_of_information"]
        wait_cost = wait["delay_cost"] + wait["information_cost"]
        if result["recommendation"]["kind"] == "wait":
            opening = (f"Given your estimates, the checked calculation favors waiting for the signal. "
                       f"Its expected payoff is "
                       f"{wait['value']:,.2f} {units}, versus {now_value:,.2f} from the best move now ({label(best_now)}).")
        else:
            opening = (f"Given your estimates, the checked calculation favors {label(best_now)} now: "
                       f"{now_value:,.2f} {units}, "
                       f"versus {wait['value']:,.2f} from waiting for the signal.")
        next_moves = "; ".join(f"{label(signal)} would favor {label(info['best_action'])}"
                                for signal, info in wait["signals"].items())
        lines = [opening, f"The option to change course after learning is worth {information_value:,.2f} "
                 f"before {wait_cost:,.2f} in waiting costs. If you wait, {next_moves}."]
    else:
        lines = [f"Given your estimates, the checked calculation favors {label(best_now)}, with the highest expected payoff "
                 f"({now_value:,.2f} {units})."]
    reversals = [(action, state) for action, cells in result["outcomes"].items()
                 for state, cell in cells.items() if cell["reversed"]]
    if reversals:
        lines.append("That includes the option to undo " + ", ".join(f"{label(action)} after {label(state)}" for action, state in reversals) + ".")
    lines.append(f"This depends on your estimates ({states}); we can change any of them and compare.")
    return " ".join(lines)


def _last_context(store: Store, branch: str) -> dict[str, Any] | None:
    for _, obj in store.log(branch):
        if obj["kind"] == "turn":
            context = obj["payload"].get("context")
            if isinstance(context, dict):
                return context
    return None


def _recent_verified_decision(store: Store, branch: str,
                              context: dict[str, Any] | None, text: str) -> str | None:
    """Carry the latest checked result as data without rewriting model history."""
    active = isinstance(context, dict) and context.get("status") in {"active", "resolved"}
    recalling = bool(DECISION_RECALL.search(text))
    if not active and not recalling:
        return None
    goal = context.get("goal") if active else None
    topic_match = DECISION_RECALL_TOPIC.search(text) if recalling else None
    topic_terms = (_salient_terms(topic_match.group(1)) - RECALL_GENERIC_TERMS
                   if topic_match else set())
    best: tuple[tuple[int, int], str] | None = None
    for _, obj in store.log(branch):
        if obj["kind"] != "turn":
            continue
        payload = obj["payload"]
        turn_context = payload.get("context")
        if active and not recalling and isinstance(turn_context, dict) and turn_context.get("goal") != goal:
            return None
        decision = payload.get("decision")
        if isinstance(decision, dict) and isinstance(decision.get("case"), dict) \
                and isinstance(decision.get("analysis"), dict):
            summary = format_analysis(decision["case"], decision["analysis"])[:2000]
            if not topic_terms:
                return summary
            case_title = str(decision["case"].get("title", ""))
            context_goal = str(turn_context.get("goal", "")) if isinstance(turn_context, dict) else ""
            strong = len(topic_terms & _salient_terms(case_title + " " + context_goal))
            weak = len(topic_terms & _salient_terms(str(payload.get("user", ""))))
            score = (strong, weak)
            if score > (0, 0) and (best is None or score > best[0]):
                best = (score, summary)
    return best[1] if best else None


def _last_jev_assessment(store: Store, branch: str, context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Carry decision attention only within the same active branch goal."""
    if not context or context.get("status") not in {"active", "resolved"}:
        return None
    goal = context.get("goal")
    for _, obj in store.log(branch):
        if obj["kind"] != "turn":
            continue
        payload = obj["payload"]
        turn_context = payload.get("context")
        if not isinstance(turn_context, dict) or turn_context.get("goal") != goal:
            return None
        jev = payload.get("jev")
        if isinstance(jev, dict) and jev.get("goal") == goal:
            guidance = jev.get("guidance")
            return {"scores": jev.get("scores", {}),
                    "attention": guidance.get("attention", {}) if isinstance(guidance, dict) else {}}
    return None


def _decision_evidence(store: Store, branch: str, current: str, previous_context: dict[str, Any] | None,
                       next_context: dict[str, Any] | None, recent_turns: int) -> list[dict[str, str]]:
    """Keep numerical and scope checks inside the recent decision conversation."""
    if not recent_turns:
        return [{"role": "user", "content": current}]
    if (previous_context and next_context and previous_context.get("goal")
            and next_context.get("goal") and previous_context["goal"] != next_context["goal"]):
        return [{"role": "user", "content": current}]
    prior: list[dict[str, str]] = []
    for _, obj in store.log(branch):
        if obj["kind"] != "turn":
            continue
        payload = obj["payload"]
        context = payload.get("context")
        if isinstance(context, dict) and context.get("status") == "none":
            break
        prior.append({"role": "user", "content": payload["user"]})
        if len(prior) >= recent_turns:
            break
    return list(reversed(prior)) + [{"role": "user", "content": current}]


def _context_from_analysis(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    choice = result["recommendation"]
    view = "Wait for the signal" if choice["kind"] == "wait" else f"Act now with {choice['action']}"
    return {"status": "resolved", "goal": str(case.get("title", "Compare the available actions")),
            "options": list(case["actions"]),
            "known": [f"Assumed {state}: {probability:.1%}" for state, probability in case["states"].items()],
            "uncertain": [], "provisional_view": view, "next_questions": []}


def _branch_name(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:30].strip("-") or "alternative"
    return f"explore-{slug}-{uuid.uuid4().hex[:6]}"


def _unsupported_save_claims(reply: str, saved_now: dict[str, bool],
                             saved_before: dict[str, bool], *,
                             allow_previous: bool = False) -> set[str]:
    """Identify checkable claims without changing the model's words."""
    unsupported: set[str] = set()
    for match in PERSISTENCE_CLAIM.finditer(reply):
        kind = (match.group("active_kind") or match.group("passive_kind") or
                match.group("generic_kind")).casefold()
        if kind in {"forecast", "prediction"}:
            keys, label = ("forecast",), "forecast"
        elif kind == "forecast outcome":
            keys, label = ("forecast outcome",), "forecast outcome"
        elif kind in {"target", "goal"}:
            keys, label = ("target",), "target"
        elif kind in {"reported outcome", "actual", "result"}:
            keys, label = ("reported outcome",), "reported outcome"
        elif kind == "outcome":
            keys, label = ("forecast outcome", "reported outcome"), "outcome"
        else:
            keys, label = tuple(saved_now), "record"
        if not any(saved_now[key] or (allow_previous and saved_before[key]) for key in keys):
            unsupported.add(label)
    return unsupported


def _deadline_mentioned(text: str, deadline: str) -> bool:
    normalized = " ".join(text.casefold().split())
    target = " ".join(deadline.casefold().split())
    return target in normalized or (target.startswith("by ") and target[3:] in normalized)


def _event_mentioned(text: str, event: str, deadline: str) -> bool:
    """Require all event words in one clause, allowing ordinary word order changes."""
    event_terms = _salient_terms(re.sub(re.escape(deadline), " ", event, flags=re.IGNORECASE)
                                 if deadline else event)
    return bool(event_terms) and any(event_terms <= _salient_terms(clause)
                                     for clause in re.split(r"[.!?;,\n]", text))


def _salient_words(text: str) -> list[str]:
    return [word[:4] for word in re.findall(r"[a-z]{4,}", text.casefold())
            if word not in EVENT_STOPWORDS]


def _salient_terms(text: str) -> set[str]:
    return set(_salient_words(text))


def _matching_calibration_topic(store: Store, records: list[dict], text: str) -> dict | None:
    """Select feedback only when one recorded topic clearly matches this question."""
    question_terms = _salient_terms(text)
    topic_scores = {topic: len(_salient_terms(topic))
                    for topic in {item["forecast"]["topic"] for item in records
                                  if item["forecast"]["source"] == "pioneer"
                                  and item["resolution"] is not None and item["forecast"]["topic"]}
                    if _salient_terms(topic) and _salient_terms(topic) <= question_terms}
    best = max(topic_scores.values(), default=0)
    if best == 0:
        return None
    matches = [topic for topic, score in topic_scores.items() if score == best]
    if len(matches) != 1:
        return None
    return calibration_context(store, matches[0])


def _visible_forecast_probability(reply: str, forecast: dict[str, Any]) -> bool:
    """Tie the displayed percentage to this event and deadline, not another estimate."""
    percentages = list(VISIBLE_PERCENT.finditer(reply))
    if len(percentages) != 1:
        return False
    breaks = list(CLAUSE_BREAK.finditer(reply))
    for match in percentages:
        if not math.isclose(float(match.group(1)) / 100, forecast["probability"], abs_tol=0.0001):
            continue
        left = max((item.end() for item in breaks if item.end() <= match.start()), default=0)
        right = min((item.start() for item in breaks if item.start() >= match.end()), default=len(reply))
        clause = reply[left:right]
        if (_deadline_mentioned(clause, forecast["deadline"])
                and _event_mentioned(clause, forecast["event"], forecast["deadline"])):
            return True
    return False


def _linked_probability_visible(quote: str, probability: float) -> bool:
    """Require an explicit probability in the attributed words."""
    percentages = [float(match.group(1)) / 100 for match in VISIBLE_PERCENT.finditer(quote)]
    decimals = [float(match.group()) for match in VISIBLE_PROBABILITY.finditer(quote)]
    stated = percentages + decimals
    return len(stated) == 1 and math.isclose(stated[0], probability, abs_tol=0.0001)


def _linked_probability_clause(source_text: str, quote: str) -> str:
    """Return the clause containing the attributed probability, not another event."""
    position = source_text.find(quote)
    if position < 0:
        return ""
    end = position + len(quote)
    breaks = list(CLAUSE_BREAK.finditer(source_text))
    left = max((item.end() for item in breaks if item.end() <= position), default=0)
    right = min((item.start() for item in breaks if item.start() >= end),
                default=len(source_text))
    return source_text[left:right].strip()


def _linked_author_supported(clause: str, source: str) -> bool:
    """A quoted estimate must belong to the speaker whose score it affects."""
    if source == "openai":
        return not bool(re.search(
            r"\b(?:you (?:said|estimated|predicted|put|gave)|your (?:estimate|forecast|"
            r"prediction|number|view|belief)|according to you|as you said)\b",
            clause, re.IGNORECASE))
    return not bool(re.search(
        r"\b(?:you (?:said|estimated|predicted|put|gave)|your (?:estimate|forecast|"
        r"prediction|number|view|belief)|openai'?s? (?:estimate|forecast|prediction))\b",
        clause, re.IGNORECASE))


def _bare_estimate_clause(clause: str) -> bool:
    """Allow a short direct answer when the target is clear from the question."""
    return bool(re.fullmatch(
        r"\s*(?:(?:i|we)\s+(?:think|estimate|guess|say|put it at)\s+|"
        r"(?:my|our)\s+(?:estimate|forecast|probability)\s+(?:is|would be)\s+)?"
        r"(?:about\s+|roughly\s+|around\s+|a\s+)?"
        r"(?:\d+(?:\.\d+)?\s*(?:%|percent)|0\.\d+|1\.0+)"
        r"(?:\s+chance)?\s*", clause, re.IGNORECASE))


def _linked_polarity_supported(source_text: str, quote: str,
                               objective: dict[str, Any]) -> bool:
    """Do not record odds of missing a target as odds of meeting it."""
    clause = _linked_probability_clause(source_text, quote).casefold()
    if not clause:
        return False
    if re.search(r"\b(?:miss(?:es|ed|ing)?|fail(?:s|ed|ing)?|fall(?:s|ing)? short)\b"
                 r".{0,50}\b(?:target|goal|checkpoint)\b", clause):
        return False
    if re.search(r"\b(?:don'?t|not)\s+(?:think|believe)\b", clause):
        return False
    if (re.search(r"\b(?:not|never|won't|can't|cannot|don't|doesn't)\s+"
                  r"(?:hit|meet|reach|achieve|attain)\b", clause)
            or re.search(r"\b(?:target|goal)\s+(?:won't|will not|is not|isn't)\s+"
                         r"(?:be\s+)?(?:met|reached|achieved)\b", clause)):
        return False
    negated = bool(re.search(r"\b(?:not|never|won't|wouldn't|can't|cannot|don't|doesn't|didn't)\b",
                             clause))
    if objective["kind"] == "numeric":
        inverse = (r"\b(?:below|under|less than|fewer than|short of)\b"
                   if objective["direction"] == "at_least" else
                   r"\b(?:above|over|more than|greater than|exceed\w*)\b"
                   if objective["direction"] == "at_most" else
                   r"\b(?:below|under|less than|fewer than|above|over|more than|"
                   r"greater than|exceed\w*)\b")
        if re.search(inverse, clause) and not negated:
            return False
    elif objective["desired"] is True and negated:
        return False
    elif objective["desired"] is False:
        if not negated and not re.search(r"\b(?:meet|hit|reach|achieve)\s+(?:the|our|my|this|that)?\s*"
                                         r"(?:target|goal)\b", clause):
            return False
    return True


def _linked_target_grounded(text: str, objective: dict[str, Any], objective_id: str,
                            *, ambiguous: bool, prior_question: str = "") -> bool:
    """Keep a probability attached to the target the user actually identified."""
    if objective_id and any(objective_id.startswith(match.group())
                            for match in re.finditer(r"\b[0-9a-f]{8,64}\b", text.casefold())):
        return True
    terms = _salient_terms(text)
    metric = _salient_terms(objective["metric"])
    goal = _salient_terms(objective["goal"])
    named_metric = bool(metric) and metric <= terms
    named_goal = bool(goal) and goal <= terms
    if ambiguous:
        return named_metric and (named_goal or _deadline_mentioned(text, objective["deadline"]))
    if named_metric or named_goal:
        return True
    if (objective["kind"] == "numeric" and _number_supported(objective["desired"],
                                                               _numbers_in_clause(text))
            and _unit_pattern(objective["unit"]).search(text)
            and _deadline_mentioned(text, objective["deadline"])):
        return True
    if re.search(r"\b(?:that|the|my|our)\s+(?:target|goal|outcome)\b", text, re.IGNORECASE):
        return True
    if re.search(r"\b(?:hit|reach|meet|achieve)(?:ing)?\s+(?:it|that)\b", text,
                 re.IGNORECASE):
        return True
    if re.search(r"\b(?:give|assign|put)\s+(?:it|that)\s+(?:at|a)\b", text,
                 re.IGNORECASE):
        return True
    return ("?" in prior_question
            and bool(re.search(r"\b(?:chance|likely|probability|odds)\b", prior_question,
                               re.IGNORECASE))
            and (_metric_in_clause(prior_question, objective)
                 or bool(re.search(r"\b(?:it|that|target|goal|outcome)\b", prior_question,
                                   re.IGNORECASE))))


def _reported_outcome(text: str, forecast: dict[str, Any]) -> bool | None:
    """Score a report only when it identifies the saved event and its deadline."""
    if text.rstrip().endswith("?") and not re.search(r"[.!]\s*[^?]*\?\s*$", text):
        return None
    forecast_id = forecast["id"]
    for match in re.finditer(r"\b[0-9a-f]{8,64}\b", text.casefold()):
        if forecast_id.startswith(match.group()):
            answer = re.match(r"\s*(?:was|is|:)?\s*(yes|no|true|false)\b",
                              text[match.end():], re.IGNORECASE)
            if answer:
                return answer.group(1).casefold() in {"yes", "true"}
    if (not _deadline_mentioned(text, forecast["deadline"])
            or not _event_mentioned(text, forecast["event"], forecast["deadline"])):
        return None
    bare_deadline = re.sub(r"^by\s+", "", forecast["deadline"], flags=re.IGNORECASE)
    late = bool(re.search(r"\b(?:after|not(?:\s+by|\s+on)?|instead\s+of|rather\s+than)\s+"
                          + re.escape(bare_deadline) + r"\b", text, re.IGNORECASE))
    if late:
        return False
    matching_clauses = [clause for clause in re.split(r"[.!?;,\n]", text)
                        if _deadline_mentioned(clause, forecast["deadline"])
                        and _event_mentioned(clause, forecast["event"], forecast["deadline"])]
    if len(matching_clauses) != 1:
        return None
    clause = matching_clauses[0]
    negative = bool(NEGATIVE_OUTCOME.search(clause))
    remaining = NEGATIVE_OUTCOME.sub(" ", clause)
    positive = bool(POSITIVE_OUTCOME.search(remaining))
    if negative and positive:
        return None
    if negative:
        return False
    return True if positive else None


def _numbers_in_clause(clause: str) -> list[float]:
    """Find explicit values, allowing a date to appear before the measured value."""
    found: list[float] = []
    for number in NUMBER.finditer(clause):
        raw = number.group().replace(",", "")
        try:
            value = float(raw.rstrip("%"))
        except (OverflowError, ValueError):
            continue
        found.append(value)
        if raw.endswith("%"):
            found.append(value / 100)
    return found


def _number_supported(number: float, choices: list[float]) -> bool:
    return any(math.isclose(number, candidate, rel_tol=1e-8, abs_tol=1e-8)
               for candidate in choices)


def _unit_pattern(unit: str) -> re.Pattern[str]:
    if unit.casefold() in {"usd", "dollars", "$"}:
        return re.compile(r"\$|\b(?:usd|dollars?)\b", re.IGNORECASE)
    if unit == "%":
        return re.compile(r"%|\bpercent(?:age)?\b", re.IGNORECASE)
    return re.compile(r"\b" + re.escape(unit) + r"\b", re.IGNORECASE)


def _metric_in_clause(clause: str, objective: dict[str, Any]) -> bool:
    metric_terms = _salient_terms(objective["metric"])
    if not metric_terms or not metric_terms <= _salient_terms(clause):
        return False
    return bool(_unit_pattern(objective["unit"]).search(clause))


def _value_tied_to_measure(reported: str, value: float, objective: dict[str, Any],
                           timing: str) -> bool:
    unit = _unit_pattern(objective["unit"])
    metric_terms = _salient_terms(objective["metric"])
    timing_spans = [(item.start(), item.end()) for item in re.finditer(
        re.escape(timing), reported, re.IGNORECASE)] if timing else []
    for number in NUMBER.finditer(reported):
        if any(start <= number.start() < end for start, end in timing_spans):
            continue
        if not _number_supported(value, _numbers_in_clause(number.group())):
            continue
        before = reported[max(0, number.start() - 50):number.start()]
        after = reported[number.end():number.end() + 55]
        if metric_terms <= _salient_terms(before) and unit.search(before):
            return True
        unit_after = unit.search(after)
        if unit_after is None:
            continue
        intervening = after[:unit_after.start()]
        if NUMBER.search(intervening) or OTHER_MEASURE.search(intervening):
            continue
        return True
    return False


def _measured_clause(text: str, cue: re.Pattern[str], objective: dict[str, Any],
                     value: float, *, timing: str = "") -> bool:
    """Tie a number to the intended measure in the same reported clause."""
    for clause in CLAUSE_BREAK.split(text):
        for marker in cue.finditer(clause):
            if (cue is ACTUAL_REPORT and re.search(r"\b(?:desired|target|goal)\s+$",
                                                   clause[:marker.start()], re.IGNORECASE)):
                continue
            reported = clause[marker.start():]
            if (_metric_in_clause(reported, objective)
                    and _value_tied_to_measure(reported, value, objective, timing)):
                return True
    return False


def _objective_grounded(objective: dict[str, Any], text: str,
                        messages: list[dict[str, str]]) -> bool:
    if not _deadline_mentioned(text, objective["deadline"]):
        return False
    if DESIRED_INTENT.search(text):
        source = text
    else:
        # A short answer to Pioneer's deadline question may complete a target
        # from the immediately preceding user turn. Do not borrow from older
        # unrelated goals or from a turn that was not asking for the deadline.
        if len(messages) < 3 or messages[-2]["role"] != "assistant" or not re.search(
                r"\b(?:when|deadline|by what date|date|checkpoint)\b",
                messages[-2]["content"], re.IGNORECASE):
            return False
        prior_users = [message["content"] for message in messages[:-2]
                       if message["role"] == "user"]
        if not prior_users:
            return False
        source = prior_users[-1]
    if objective["kind"] == "binary":
        return bool(DESIRED_INTENT.search(source)
                    and _event_mentioned(source, objective["goal"], objective["deadline"]))
    return _measured_clause(source, DESIRED_INTENT, objective, objective["desired"],
                            timing=objective["deadline"])


def _actual_grounded(actual: dict[str, Any], objective: dict[str, Any], text: str,
                     messages: list[dict[str, str]], *, same_turn: bool,
                     ambiguous_targets: bool) -> bool:
    prior_question = (messages[-2]["content"] if len(messages) >= 3
                      and messages[-2]["role"] == "assistant" else "")
    direct_answer = (not same_turn and not ambiguous_targets and "?" in prior_question
                     and _deadline_mentioned(prior_question, actual["as_of"]))
    if not _deadline_mentioned(text, actual["as_of"]) and not direct_answer:
        return False
    if objective["kind"] == "binary":
        if (direct_answer and _event_mentioned(prior_question, objective["goal"], objective["deadline"])
                and re.fullmatch(r"\s*(?:yes|no|true|false)\s*[.!]?\s*", text, re.IGNORECASE)):
            return (text.strip().rstrip(".!").casefold() in {"yes", "true"}) is actual["value"]
        reported = _reported_outcome(text, {"id": actual["objective_id"] or "0" * 64,
                                            "event": objective["goal"],
                                            "deadline": actual["as_of"]})
        return reported is actual["value"]
    if (direct_answer and _metric_in_clause(prior_question, objective)
            and re.fullmatch(r"\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?\s*[.!]?\s*", text)
            and _number_supported(actual["value"], _numbers_in_clause(text))):
        return True
    if not _measured_clause(text, ACTUAL_REPORT, objective, actual["value"],
                            timing=actual["as_of"]):
        return False
    if same_turn:
        return True
    if actual["objective_id"] in text.casefold():
        return True
    metric_terms = _salient_terms(objective["metric"])
    if not metric_terms or not metric_terms <= _salient_terms(text):
        return not ambiguous_targets
    if ambiguous_targets:
        goal_terms = _salient_terms(objective["goal"])
        return bool(goal_terms) and goal_terms <= _salient_terms(text)
    return True


def _objective_comparison_text(comparison: dict[str, Any]) -> str:
    if comparison["status"] == "unobserved":
        return ""
    when = comparison["as_of"]
    if comparison["kind"] == "binary":
        actual = "yes" if comparison["actual"] else "no"
        desired = "yes" if comparison["desired"] else "no"
        result = f"As of {when}, your reported result is {actual} against a desired {desired}."
    else:
        unit = comparison["unit"]
        result = (f"As of {when}, you reported {comparison['actual']:g} {unit} against "
                  f"a target of {comparison['desired']:g} {unit} "
                  f"({comparison['direction'].replace('_', ' ')}).")
        if comparison["status"] == "final":
            if comparison["shortfall"]:
                result += f" The gap from meeting it is {comparison['shortfall']:g} {unit}."
            else:
                result += " That meets the target."
        else:
            result += f" The current gap from meeting it is {comparison['shortfall']:g} {unit}."
    if comparison["status"] == "progress":
        result += f" This is a progress reading; the target checkpoint is {comparison['deadline']}."
    return result


def _retracts_checkpoint(text: str, old_as_of: str, new_as_of: str) -> bool:
    """Recognize an explicit correction of a result's checkpoint timing."""
    if not CORRECTION_CUE.search(text) or old_as_of.casefold() == new_as_of.casefold():
        return False
    old = re.sub(r"^by\s+", "", old_as_of.strip(), flags=re.IGNORECASE)
    return bool(re.search(r"\b(?:not|instead of|rather than)\s+(?:(?:on|by)\s+)?"
                          + re.escape(old) + r"\b", text, re.IGNORECASE)
                or re.search(re.escape(old) + r"\s+(?:was|is)\s+wrong\b", text,
                             re.IGNORECASE))


def run_turn(store: Store, text: str, *, model: str | None = None) -> TurnOutcome:
    store.require()
    if not text.strip():
        raise StoreError("Message cannot be empty")
    starting_branch = store.current_branch()
    head = store.resolve(starting_branch)
    previous_context = _last_context(store, starting_branch)
    recent, recent_turns = recent_messages(store, starting_branch)
    messages = recent + [{"role": "user", "content": text}]
    direct = _direct_case(text)
    if direct is not None:
        result = analyze(direct)
        answer = format_analysis(direct, result)
        object_id = store.commit("turn", {"user": text, "assistant": answer, "provider": "local",
                                          "context": _context_from_analysis(direct, result),
                                          "decision": {"case": direct, "analysis": result}},
                                 expected_head=head, expected_branch=starting_branch)
        return TurnOutcome(answer, object_id, "local", [], result, starting_branch)

    # A Jev assessment is billable. Check the conversation provider before
    # consulting it so an absent OpenAI key cannot leave an unusable result.
    if not os.environ.get("OPENAI_API_KEY"):
        raise ProviderError("Set OPENAI_API_KEY to chat with OpenAI.")

    asking_about_history = bool(RECALL_CUE.search(text) or SAVE_HISTORY_QUERY.search(text))
    history_evidence = (retrieve_history(store, starting_branch, text, previous_context, recent_turns)
                        if asking_about_history else [])
    records = forecast_records(store, starting_branch)
    available_forecasts = [{"id": item["id"], **item["forecast"]}
                           for item in reversed(records) if item["resolution"] is None]
    resolved_by_id = {item["id"]: item for item in records if item["resolution"] is not None}
    resolved_forecasts: list[dict[str, Any]] = []
    seen_resolutions: set[str] = set()
    for _, obj in store.log(starting_branch):
        resolution = obj.get("payload", {}).get("resolution")
        forecast_id = resolution.get("forecast_id") if isinstance(resolution, dict) else None
        if forecast_id in resolved_by_id and forecast_id not in seen_resolutions:
            item = resolved_by_id[forecast_id]
            resolved_forecasts.append({"id": item["id"], **item["forecast"],
                                       "reported_outcome": item["resolution"]["outcome"]})
            seen_resolutions.add(forecast_id)
    mentioned_ids = re.findall(r"\b[0-9a-f]{8,64}\b", text.lower())
    mentioned = [item for item in available_forecasts
                 if any(item["id"].startswith(prefix) for prefix in mentioned_ids)]
    open_context = (mentioned + [item for item in available_forecasts if item not in mentioned])[:5]
    resolved_context: list[dict[str, Any]] = []
    if CORRECTION_CUE.search(text):
        mentioned_resolved = [item for item in resolved_forecasts
                              if any(item["id"].startswith(prefix) for prefix in mentioned_ids)]
        resolved_context = (mentioned_resolved + [item for item in resolved_forecasts
                                                   if item not in mentioned_resolved])[:3]
    calibration_evidence: dict[str, Any] | None = None
    if FORECAST_ACCURACY_QUERY.search(text):
        calibration_evidence = calibration_report(store, starting_branch, source="pioneer")
    elif FORECAST_REQUEST.search(text):
        calibration_evidence = _matching_calibration_topic(store, records, text)
    elif previous_context and select_jev_context(text, previous_context) is not None:
        topic = str(previous_context.get("goal", "")).strip()[:120]
        if topic:
            calibration_evidence = calibration_context(store, topic)
    objective_data = objective_records(store, starting_branch)
    objective_comparisons = [compare_objective(item) for item in objective_data]
    linked_threads = outcome_threads(store, starting_branch)
    linked_by_id = {item["id"]: item for item in linked_threads}
    text_terms = _salient_terms(text)
    id_threads = [item for item in linked_threads if any(
        item["id"].startswith(prefix) for prefix in mentioned_ids)]
    named_threads = [item for item in reversed(linked_threads)
                     if item not in id_threads and any(
                         terms and terms <= text_terms for terms in (
                             _salient_terms(item["objective"]["goal"]),
                             _salient_terms(item["objective"]["metric"]))) ]
    prioritized_threads = (id_threads + named_threads
                           + [item for item in reversed(linked_threads)
                              if item["comparison"]["status"] != "final"
                              and item not in id_threads and item not in named_threads]
                           + [item for item in reversed(linked_threads)
                              if item["comparison"]["status"] == "final"
                              and item not in id_threads and item not in named_threads])
    linked_context: list[dict[str, Any]] = []
    for thread in prioritized_threads[:5]:
        comparison = thread["comparison"]
        latest_by_source = {item["source"]: item for item in thread["forecasts"]}
        linked_context.append({
            "id": thread["id"],
            "goal": comparison["goal"], "metric": comparison["metric"],
            "kind": comparison["kind"], "desired": comparison["desired"],
            "direction": comparison["direction"], "unit": comparison["unit"],
            "deadline": comparison["deadline"], "status": comparison["status"],
            "reported_actual": comparison["actual"],
            "forecasts": [{"source": source, "probability": forecast["probability"],
                           "recorded_at": forecast["timestamp"]}
                          for source, forecast in latest_by_source.items()],
        })
    available_objectives = [item for item in reversed(objective_comparisons)
                            if item["status"] != "final"]
    mentioned_objectives = [item for item in available_objectives
                            if any(item["id"].startswith(prefix) for prefix in mentioned_ids)]
    objective_context = (mentioned_objectives + [item for item in available_objectives
                                                 if item not in mentioned_objectives])[:5]
    finished_objectives = [item for item in reversed(objective_comparisons)
                           if item["status"] == "final"]
    recent_objective_context: list[dict[str, Any]] = []
    if CORRECTION_CUE.search(text) or OUTCOME_CALIBRATION_QUERY.search(text):
        mentioned_finished = [item for item in finished_objectives
                              if any(item["id"].startswith(prefix) for prefix in mentioned_ids)]
        recent_objective_context = (mentioned_finished + [item for item in finished_objectives
                                                           if item not in mentioned_finished])[:3]
    outcome_evidence: dict[str, Any] | None = None
    if OUTCOME_CALIBRATION_QUERY.search(text):
        full_report = outcome_report(store, starting_branch)
        outcome_evidence = {**full_report, "comparisons": full_report["comparisons"][-8:]}
    else:
        question_terms = _salient_terms(text)
        if previous_context and select_jev_context(text, previous_context) is not None:
            question_terms |= _salient_terms(str(previous_context.get("goal", "")))
        relevant = [item for item in finished_objectives
                    if _salient_terms(item["goal"]) and _salient_terms(item["goal"]) <= question_terms]
        if relevant:
            outcome_evidence = {"comparisons": relevant[:3], "final_count": len(relevant),
                                "caveat": "These are user-reported outcomes, not proven causes."}
    if OUTCOME_CALIBRATION_QUERY.search(text) or FORECAST_ACCURACY_QUERY.search(text):
        linked_report = unified_outcome_report(store, starting_branch)
        linked_scores = {"scored_count": linked_report["scored_count"],
                         "mean_brier": linked_report["mean_brier"],
                         "by_source": linked_report["by_source"],
                         "caveat": linked_report["caveat"]}
        outcome_evidence = {**(outcome_evidence or {}), "linked_forecast_accuracy": linked_scores}
    jev_assessment: dict[str, Any] | None = None
    jev_guidance: dict[str, Any] | None = None
    jev_error: str | None = None
    usage: list[dict[str, Any]] = []
    if os.environ.get("TYPESAFE_API_KEY") and should_consult_jev(text, previous_context):
        try:
            jev_context = select_jev_context(text, previous_context)
            recent_user_messages = [message["content"] for message in recent if message["role"] == "user"][-4:]
            if previous_context and previous_context.get("status") in {"active", "resolved"} and jev_context is None:
                recent_user_messages = []
            jev_assessment = assess_jev(
                text, context=jev_context,
                recent_user_messages=recent_user_messages,
                history_evidence=[],
                previous_assessment=_last_jev_assessment(store, starting_branch, jev_context))
            jev_guidance = make_jev_guidance(jev_assessment["scores"], jev_context)
            usage.append(_usage("jev", jev_assessment))
        except ProviderError as exc:
            jev_error = str(exc)
            if exc.usage:
                usage.append(exc.usage)
    try:
        plan = compose_turn(messages, model=model, jev_guidance=jev_guidance, context=previous_context,
                            history_evidence=history_evidence, open_forecasts=open_context,
                            recent_resolutions=resolved_context, calibration=calibration_evidence,
                            open_objectives=objective_context, open_outcomes=linked_context,
                            recent_objectives=recent_objective_context,
                            outcome_history=outcome_evidence,
                            verified_decision=_recent_verified_decision(
                                store, starting_branch, previous_context, text))
    except ProviderError as exc:
        if exc.usage:
            usage.append(exc.usage)
        if usage:
            store.commit("note", {"title": "Incomplete turn", "text": text, "jev": jev_assessment},
                         expected_head=head, expected_branch=starting_branch, usage=usage)
        raise
    usage.append(_usage("openai", plan))
    if plan.state_usage:
        usage.append({**plan.state_usage, "purpose": "state_check"})
    decision_requested = plan.decision_requested or bool(plan.case_json)
    result: dict[str, Any] | None = None
    case: dict[str, Any] | None = None
    validation_error: str | None = None
    reply = plan.reply
    notices: list[str] = []
    if plan.state_error:
        notices.append("Pioneer's state check was unavailable; no new records were saved from this reply.")
    if not reply.strip():
        store.commit("note", {"title": "Incomplete turn", "text": text, "jev": jev_assessment},
                     expected_head=head, expected_branch=starting_branch, usage=usage)
        raise ProviderError("OpenAI returned an empty conversational reply.")
    review_evidence = retrieve_history(store, starting_branch, text + " " + reply,
                                       previous_context, recent_turns, include_recent=True,
                                       followup_text=text)
    history_challenge: dict[str, str] | None = None
    history_review_error: str | None = None
    if review_evidence:
        try:
            review = review_history(reply, review_evidence, user_text=text, model=model)
            usage.append({**_usage("openai", review), "purpose": "history_review"})
            if review.conflict:
                history_challenge = verified_conflict(review.conflict, review_evidence)
                if history_challenge is None:
                    notices.append("Pioneer could not verify a proposed history citation.")
        except ProviderError as exc:
            history_review_error = str(exc)
            if exc.usage:
                usage.append({**exc.usage, "purpose": "history_review"})
            notices.append("Pioneer's retrospective history check was unavailable for this turn.")
    if decision_requested and plan.case_json:
        try:
            candidate = json.loads(plan.case_json)
            if not isinstance(candidate, dict):
                raise DecisionError("Decision case must be a JSON object")
            evidence = _decision_evidence(store, starting_branch, text, previous_context, plan.context,
                                          recent_turns)
            _check_explicit_numbers(candidate, evidence)
            _check_decision_scope(candidate, evidence)
            result = analyze(candidate)
            case = candidate
            summary = format_analysis(candidate, result)
            notices.append(summary)
        except (json.JSONDecodeError, DecisionError) as exc:
            validation_error = str(exc)
            notices.append(f"Proposed calculation was not verified: {validation_error}")
    stored_forecast: dict[str, Any] | None = None
    if (plan.forecast and plan.outcome_forecast is None and FORECAST_REQUEST.search(text)
            and case is None and not validation_error):
        topic = str((plan.context or {}).get("goal", "")).strip()[:120]
        try:
            candidate = validate_forecast(plan.forecast, source="pioneer", topic=topic, model=plan.model)
            user_evidence = " ".join(message["content"] for message in messages[-7:]
                                     if message["role"] == "user")
            if (_deadline_mentioned(user_evidence, candidate["deadline"])
                    and _event_mentioned(user_evidence, candidate["event"], candidate["deadline"])
                    and _visible_forecast_probability(reply, candidate)):
                stored_forecast = candidate
                notices.append(f"Forecast saved: {candidate['probability']:.1%} — "
                               f"{candidate['event']}. Resolution: {candidate['deadline']}.")
        except CalibrationError:
            pass
    stored_resolution: dict[str, Any] | None = None
    if plan.resolution:
        forecast_id = plan.resolution["forecast_id"]
        target = next((item for item in open_context if item["id"] == forecast_id), None)
        correction = False
        if target is None and CORRECTION_CUE.search(text):
            target = next((item for item in resolved_context if item["id"] == forecast_id), None)
            correction = (target is not None and target["reported_outcome"] is not plan.resolution["outcome"])
        if (target is not None and (target in open_context or correction)
                and _reported_outcome(text, target) is plan.resolution["outcome"]):
            stored_resolution = {"forecast_id": forecast_id, "outcome": plan.resolution["outcome"],
                                 "note": text.strip()[:1000]}
            action = "corrected" if correction else "recorded"
            notices.append(f"Forecast outcome {action} for {forecast_id[:12]}.")
    stored_objective: dict[str, Any] | None = None
    if plan.objective and case is None and not validation_error:
        try:
            candidate = validate_objective(plan.objective)
            if _objective_grounded(candidate, text, messages):
                user_evidence = " ".join(message["content"] for message in messages[-7:]
                                         if message["role"] == "user")
                if candidate["action"] and not _salient_terms(candidate["action"]) <= _salient_terms(user_evidence):
                    candidate["action"] = ""
                stored_objective = candidate
        except ObjectiveError:
            pass
    stored_actual: dict[str, Any] | None = None
    actual_preview: dict[str, Any] | None = None
    comparison_text = ""
    if plan.actual and not validation_error:
        reported = plan.actual
        same_turn = reported["objective_id"] == "" and stored_objective is not None
        target = None if same_turn else next(
            (item for item in objective_context + recent_objective_context
             if item["id"] == reported["objective_id"]), None)
        correction = target is not None and target["status"] == "final"
        mentioned_target = any(reported["objective_id"].startswith(prefix) for prefix in mentioned_ids)
        if (same_turn or target is not None and (not correction or CORRECTION_CUE.search(text) or mentioned_target)):
            objective = stored_objective if same_turn else {key: target[key] for key in (
                "goal", "metric", "kind", "desired", "direction", "unit", "deadline", "action")}
            prospective_id = "" if same_turn else target["id"]
            normalized = {"objective_id": prospective_id, "value": reported["value"],
                          "as_of": reported["as_of"].strip(), "note": text.strip()[:1000]}
            ambiguous = (not same_turn and len(available_objectives) > 1 and not mentioned_target)
            if _actual_grounded(normalized, objective, text, messages, same_turn=same_turn,
                                ambiguous_targets=ambiguous):
                try:
                    previous = (next((item for item in objective_data
                                      if item["id"] == prospective_id), None)
                                if not same_turn else None)
                    replacement_id = (target["observation_id"] if correction and target
                                      and _retracts_checkpoint(text, target["as_of"],
                                                               normalized["as_of"]) else None)
                    observations = [dict(item) for item in previous["observations"]] if previous else []
                    if replacement_id:
                        replaced = next((item for item in observations
                                         if item["id"] == replacement_id), None)
                        if replaced is None or replaced.get("superseded_by"):
                            replacement_id = None
                        else:
                            replaced["superseded_by"] = "$pending"
                    observations.append({"id": "", **normalized})
                    preview = compare_objective({"id": prospective_id,
                                                 "objective": objective,
                                                 "observations": observations})
                    if not correction or target["actual"] != preview["actual"] or target["as_of"] != preview["as_of"]:
                        stored_actual = {**normalized, "objective_id": "$self" if same_turn else prospective_id}
                        if replacement_id:
                            stored_actual["replaces_observation_id"] = replacement_id
                        actual_preview = preview
                        comparison_text = _objective_comparison_text(preview)
                except ObjectiveError:
                    pass
    stored_outcome_forecast: dict[str, Any] | None = None
    if plan.outcome_forecast and case is None and not validation_error:
        proposed = dict(plan.outcome_forecast)
        if proposed.get("objective_id") == "" and stored_objective is not None:
            proposed["objective_id"] = "$self"
        source = proposed.get("source")
        source_text = text if source == "user" else reply if source == "openai" else ""
        try:
            candidate = validate_outcome_forecast(
                {**proposed, "model": plan.model if source == "openai" else None},
                source_text=source_text)
            source_clause = _linked_probability_clause(source_text, candidate["quote"])
            objective_id = candidate["objective_id"]
            thread = linked_by_id.get(objective_id)
            same_turn = objective_id == "$self" and stored_objective is not None
            objective = stored_objective if same_turn else thread["objective"] if thread else None
            final_known = (actual_preview is not None and actual_preview["status"] == "final"
                           and (same_turn or stored_actual is not None
                                and stored_actual["objective_id"] == objective_id))
            if (objective is not None and (same_turn or thread["comparison"]["status"] != "final")
                    and not final_known and _linked_probability_visible(candidate["quote"],
                                                                        candidate["probability"])
                    and _linked_polarity_supported(source_text, candidate["quote"], objective)
                    and _linked_author_supported(source_clause, source)
                    and (source == "user" or FORECAST_REQUEST.search(text))):
                prior_question = (messages[-2]["content"] if len(messages) >= 3
                                  and messages[-2]["role"] == "assistant" else "")
                ambiguous = (not same_turn and len(available_objectives) > 1)
                grounding_text = source_clause if source == "user" else text
                matching_targets = [item["id"] for item in available_objectives
                                    if _linked_target_grounded(
                                        grounding_text, item, item["id"], ambiguous=ambiguous,
                                        prior_question=prior_question if source == "user" else "")]
                cited_ids = [item["id"] for item in available_objectives
                             if any(item["id"].startswith(match.group()) for match in
                                    re.finditer(r"\b[0-9a-f]{8,64}\b", grounding_text.casefold()))]
                antecedent = (same_turn and (
                    _linked_target_grounded(source_clause, objective, "", ambiguous=False)
                    or _bare_estimate_clause(source_clause))
                    or not same_turn and (objective_id in cited_ids and len(cited_ids) == 1
                                          or matching_targets == [objective_id]))
                reply_scope = (source == "user" or _linked_target_grounded(
                    source_clause, objective, objective_id, ambiguous=False)
                    or _bare_estimate_clause(source_clause))
                if antecedent and reply_scope:
                    stored_outcome_forecast = candidate
        except OutcomeError:
            pass
    if stored_actual:
        notices.append(comparison_text)
        if actual_preview and actual_preview["status"] == "final":
            thread = linked_by_id.get(stored_actual["objective_id"])
            if thread:
                latest_eligible: dict[str, dict[str, Any]] = {}
                for forecast in thread["forecasts"]:
                    if thread["comparison"]["status"] != "final" or forecast["brier"] is not None:
                        latest_eligible[forecast["source"]] = forecast
                for source, forecast in latest_eligible.items():
                    brier = (forecast["probability"] - int(actual_preview["met"])) ** 2
                    notices.append(f"{source.capitalize()} forecast of {forecast['probability']:.1%} "
                                   f"scored {brier:.3f} Brier against your reported result.")
    elif stored_objective:
        if stored_objective["kind"] == "numeric":
            target_value = (f"{stored_objective['direction'].replace('_', ' ')} "
                            f"{stored_objective['desired']:g} {stored_objective['unit']}")
        else:
            target_value = "yes" if stored_objective["desired"] else "no"
        deadline = stored_objective["deadline"]
        when = deadline if deadline.casefold().startswith("by ") else f"by {deadline}"
        notices.append(f"Target saved for {stored_objective['metric']}: {target_value} {when}.")
    if stored_outcome_forecast:
        source = "Your" if stored_outcome_forecast["source"] == "user" else "OpenAI's"
        notices.append(f"{source} {stored_outcome_forecast['probability']:.1%} forecast was "
                       "attached to this outcome.")
    rejected_records = {label for proposed, saved, label in (
        (plan.forecast if plan.outcome_forecast is None else None, stored_forecast, "forecast"),
        (plan.resolution, stored_resolution, "forecast outcome"),
        (plan.objective, stored_objective, "target"),
        (plan.actual, stored_actual, "reported outcome"),
        (plan.outcome_forecast, stored_outcome_forecast, "forecast"))
        if proposed is not None and saved is None}
    if rejected_records:
        notices.append("Pioneer did not save the proposed " +
                       ", ".join(sorted(rejected_records)) + " record.")
    saved_now = {
        "forecast": stored_forecast is not None or stored_outcome_forecast is not None,
        "forecast outcome": stored_resolution is not None,
        "target": stored_objective is not None,
        "reported outcome": stored_actual is not None,
    }
    saved_before = {
        "forecast": bool(records) or any(item["forecasts"] for item in linked_threads),
        "forecast outcome": any(
            item["resolution"] is not None for item in records),
        "target": bool(objective_data),
        "reported outcome": any(
            item["observations"] for item in objective_data),
    }
    save_history_query = bool(SAVE_HISTORY_QUERY.search(text))
    specific_save_claim = bool(SPECIFIC_SAVE_CUE.search(text + " " + reply)
                               or NUMBER.search(text) or NUMBER.search(reply))
    unsupported_saves = _unsupported_save_claims(
        reply, saved_now, saved_before,
        allow_previous=save_history_query and not specific_save_claim) - rejected_records
    if unsupported_saves:
        subject = ", ".join(sorted(unsupported_saves))
        if save_history_query and any(saved_before.get(key, False) for key in unsupported_saves):
            notices.append(f"OpenAI claimed a save for {subject}. Pioneer found an earlier record "
                           "of that type, but could not verify it matches this specific claim.")
        else:
            notices.append(f"OpenAI claimed a save for {subject}, but Pioneer did not save "
                           "that record this turn.")
    explore = plan.explore_alternative and previous_context and previous_context.get("status") in {"active", "resolved"}
    payload: dict[str, Any] = {"user": text, "assistant": reply, "provider": "openai", "model": plan.model,
                               "decision_requested": decision_requested}
    if case is not None and result is not None:
        payload["context"] = _context_from_analysis(case, result)
    elif plan.context:
        payload["context"] = plan.context
    if explore:
        payload["branched_from"] = starting_branch
    if jev_assessment and jev_guidance:
        payload["jev"] = {"goal": (plan.context or previous_context or {}).get("goal", ""),
                          "model": jev_assessment["model"], "scores": jev_assessment["scores"],
                          "guidance": jev_guidance}
    if jev_error:
        payload["jev_error"] = jev_error
    if history_challenge:
        payload["history_challenge"] = history_challenge
    if history_review_error:
        payload["history_review_error"] = history_review_error
    if plan.state_error:
        payload["state_error"] = plan.state_error
    if notices:
        payload["notices"] = notices
    if stored_forecast:
        payload["forecast"] = stored_forecast
    if stored_outcome_forecast:
        payload["outcome_forecast"] = stored_outcome_forecast
    if stored_resolution:
        payload["resolution"] = stored_resolution
    if stored_objective:
        payload["objective"] = stored_objective
    if stored_actual:
        payload["actual"] = stored_actual
    if case is not None:
        payload["decision"] = {"case": case, "analysis": result}
    elif decision_requested:
        payload["missing"] = plan.missing
        if validation_error:
            payload["draft_case"] = plan.case_json
            payload["validation_error"] = validation_error
    if explore:
        branch_name = _branch_name(text)
        object_id = store.fork_and_commit(branch_name, "turn", payload, expected_head=head,
                                          expected_branch=starting_branch, usage=usage)
        return TurnOutcome(reply, object_id, plan.model, usage, result, branch_name, starting_branch,
                           tuple(notices), history_challenge)
    object_id = store.commit("turn", payload, expected_head=head,
                             expected_branch=starting_branch, usage=usage)
    return TurnOutcome(reply, object_id, plan.model, usage, result, starting_branch,
                       notices=tuple(notices), history_challenge=history_challenge)
