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
from .history import recent_messages, retrieve_history, verified_conflict
from .jev import make_jev_guidance, select_jev_context, should_consult_jev
from .providers import ProviderError, assess_jev, compose_turn
from .state import Store, StoreError


NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?|(?<![\w\d])[-+]?\.\d+%?")
WAIT_WORD = re.compile(r"\b(wait|waiting|delay|defer|postpone|hold off)\b", re.IGNORECASE)
UNDO_WORD = re.compile(r"\b(undo|revert|reverse|reversible|rollback|roll back)\b", re.IGNORECASE)
CONCLUSION_WORD = re.compile(
    r"\b(recommend\w*|should|would|better|best|prefer\w*|highest|lowest|payoff|expected|wait|waiting|act|choose)\b",
    re.IGNORECASE,
)
FORECAST_REQUEST = re.compile(
    r"\b(?:how likely|how confident|what(?:'s| is| are) (?:the )?(?:chances?|odds|probability)|"
    r"(?:give|make) (?:me )?(?:your |a )?(?:forecast|probability estimate|odds)|"
    r"(?:give me|what is|what's) (?:a |your |the )?(?:percentage|percent chance)|"
    r"(?:estimate|predict) (?:the )?(?:chances?|odds|probability)|"
    r"(?:can|could|would) you (?:predict|forecast|estimate)|"
    r"your (?:forecast|probability|odds|estimate))\b", re.IGNORECASE)
CALIBRATION_QUERY = re.compile(r"\b(?:calibrat\w*|forecast accuracy|how accurate|brier)\b", re.IGNORECASE)
VISIBLE_PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
CLAUSE_BREAK = re.compile(r"(?<!\d)[.!?;,\n](?!\d)|\bbut\b", re.IGNORECASE)
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
            opening = (f"Given your estimates, I would wait for the signal. Its expected payoff is "
                       f"{wait['value']:,.2f} {units}, versus {now_value:,.2f} from the best move now ({label(best_now)}).")
        else:
            opening = (f"Given your estimates, {label(best_now)} looks better now: {now_value:,.2f} {units}, "
                       f"versus {wait['value']:,.2f} from waiting for the signal.")
        next_moves = "; ".join(f"{label(signal)} would favor {label(info['best_action'])}"
                                for signal, info in wait["signals"].items())
        lines = [opening, f"The option to change course after learning is worth {information_value:,.2f} "
                 f"before {wait_cost:,.2f} in waiting costs. If you wait, {next_moves}."]
    else:
        lines = [f"Given your estimates, {label(best_now)} has the highest expected payoff "
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


def _validation_reply(error: str) -> str:
    if "numbers you did not provide" in error:
        return ("I can work with those possibilities, but I can't trace every number in the draft to "
                "one you gave me. Which estimate would you use for the uncertain outcome?")
    if "To compare waiting" in error or "delay cost" in error:
        return ("Waiting could change the choice. What could you learn by waiting, and what would "
                "the delay cost you? A rough range is fine for discussing it; I need explicit numbers "
                "only for a calculation.")
    if "reversal" in error:
        return "If you changed course afterward, what would remain lost, and what would undoing it cost?"
    return "I can talk through the tradeoff, but I need to clear up one assumption before calculating: " + error


def _analysis_framing(reply: str) -> str:
    """Use only a short, nonnumeric preface before the verified calculation."""
    framing = " ".join(reply.split())
    if len(framing) > 180 or NUMBER.search(framing) or CONCLUSION_WORD.search(framing):
        return ""
    return framing


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
    late = bool(re.search(r"\b(?:after|not(?:\s+by|\s+on)?|instead\s+of|rather\s+than)\s+"
                          + re.escape(forecast["deadline"]) + r"\b", text, re.IGNORECASE))
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

    history_evidence = retrieve_history(store, starting_branch, text, previous_context, recent_turns)
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
    if CALIBRATION_QUERY.search(text):
        calibration_evidence = calibration_report(store, starting_branch, source="pioneer")
    elif FORECAST_REQUEST.search(text):
        calibration_evidence = _matching_calibration_topic(store, records, text)
    elif previous_context and select_jev_context(text, previous_context) is not None:
        topic = str(previous_context.get("goal", "")).strip()[:120]
        if topic:
            calibration_evidence = calibration_context(store, topic)
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
                history_evidence=history_evidence,
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
                            recent_resolutions=resolved_context, calibration=calibration_evidence)
    except ProviderError as exc:
        if exc.usage:
            usage.append(exc.usage)
        if usage:
            store.commit("note", {"title": "Incomplete turn", "text": text, "jev": jev_assessment},
                         expected_head=head, expected_branch=starting_branch, usage=usage)
        raise
    usage.append(_usage("openai", plan))
    decision_requested = plan.decision_requested or bool(plan.case_json)
    result: dict[str, Any] | None = None
    case: dict[str, Any] | None = None
    validation_error: str | None = None
    reply = plan.reply.strip()
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
            framing = _analysis_framing(reply)
            reply = f"{framing} {summary}" if framing else summary
        except (json.JSONDecodeError, DecisionError) as exc:
            validation_error = str(exc)
            reply = _validation_reply(validation_error)
    if not reply:
        questions = (plan.context or {}).get("next_questions", [])
        reply = " ".join(questions) if questions else "What part of this would you like to explore next?"
    conflict = verified_conflict(plan.history_conflict, history_evidence) if not validation_error else None
    if conflict:
        reply += (f"\n\nEarlier you said “{conflict['quote']}” (history {conflict['commit'][:10]}). "
                  f"{conflict['challenge']}")
    stored_forecast: dict[str, Any] | None = None
    if plan.forecast and FORECAST_REQUEST.search(text) and case is None and not validation_error:
        topic = str((plan.context or {}).get("goal", "")).strip()[:120]
        try:
            candidate = validate_forecast(plan.forecast, source="pioneer", topic=topic, model=plan.model)
            user_evidence = " ".join(message["content"] for message in messages[-7:]
                                     if message["role"] == "user")
            if (_deadline_mentioned(user_evidence, candidate["deadline"])
                    and _event_mentioned(user_evidence, candidate["event"], candidate["deadline"])
                    and _visible_forecast_probability(reply, candidate)):
                stored_forecast = candidate
                if not re.search(r"\b(?:track|record|sav)(?:ed|ing)?\b", reply, re.IGNORECASE):
                    reply += "\n\nI'll track that forecast so we can score it when the outcome is known."
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
            if not re.search(r"\b(?:record|mark|sav)(?:ed|ing)?\b", reply, re.IGNORECASE):
                action = "corrected" if correction else "recorded"
                reply += f"\n\nI've {action} the reported outcome for forecast {forecast_id[:12]}."
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
    if conflict:
        payload["history_conflict"] = conflict
    if stored_forecast:
        payload["forecast"] = stored_forecast
    if stored_resolution:
        payload["resolution"] = stored_resolution
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
        return TurnOutcome(reply, object_id, plan.model, usage, result, branch_name, starting_branch)
    object_id = store.commit("turn", payload, expected_head=head,
                             expected_branch=starting_branch, usage=usage)
    return TurnOutcome(reply, object_id, plan.model, usage, result, starting_branch)
