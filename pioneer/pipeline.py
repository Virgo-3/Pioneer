"""One conversational turn: triage, plan, calculate, explain, and commit."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

from .decision import DecisionError, analyze
from .providers import ProviderError, compose_turn, triage_jev
from .state import Store, StoreError


NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?|(?<![\w\d])[-+]?\.\d+%?")
WAIT_WORD = re.compile(r"\b(wait|waiting|delay|defer|postpone|hold off)\b", re.IGNORECASE)
UNDO_WORD = re.compile(r"\b(undo|revert|reverse|reversible|rollback|roll back)\b", re.IGNORECASE)


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


def _check_decision_scope(case: dict[str, Any], text: str) -> None:
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


def _last_context(store: Store) -> dict[str, Any] | None:
    for _, obj in store.log():
        if obj["kind"] == "turn":
            context = obj["payload"].get("context")
            if isinstance(context, dict):
                return context
    return None


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


def run_turn(store: Store, text: str, *, model: str | None = None) -> TurnOutcome:
    store.require()
    if not text.strip():
        raise StoreError("Message cannot be empty")
    head = store.resolve()
    previous_context = _last_context(store)
    messages = store.messages() + [{"role": "user", "content": text}]
    direct = _direct_case(text)
    if direct is not None:
        result = analyze(direct)
        answer = format_analysis(direct, result)
        object_id = store.commit("turn", {"user": text, "assistant": answer, "provider": "local",
                                          "context": _context_from_analysis(direct, result),
                                          "decision": {"case": direct, "analysis": result}}, expected_head=head)
        return TurnOutcome(answer, object_id, "local", [], result, store.current_branch())

    triage: dict[str, Any] | None = None
    triage_error: str | None = None
    usage: list[dict[str, Any]] = []
    if os.environ.get("TYPESAFE_API_KEY"):
        try:
            triage = triage_jev(text, context=previous_context)
            usage.append(_usage("jev", triage))
        except ProviderError as exc:
            triage_error = str(exc)
    try:
        plan = compose_turn(messages, model=model, triage=triage, context=previous_context)
    except ProviderError:
        if usage:
            store.commit("note", {"title": "Incomplete turn", "text": text, "triage": triage},
                         expected_head=head, usage=usage)
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
            _check_explicit_numbers(candidate, messages)
            _check_decision_scope(candidate, text)
            result = analyze(candidate)
            case = candidate
            reply = format_analysis(candidate, result)
        except (json.JSONDecodeError, DecisionError) as exc:
            validation_error = str(exc)
            reply = _validation_reply(validation_error)
    if not reply:
        questions = (plan.context or {}).get("next_questions", [])
        reply = " ".join(questions) if questions else "What part of this would you like to explore next?"
    source_branch: str | None = None
    if plan.explore_alternative and previous_context and previous_context.get("status") in {"active", "resolved"}:
        branch_name = _branch_name(text)
        source_branch = store.fork_and_switch(branch_name, head)
    payload: dict[str, Any] = {"user": text, "assistant": reply, "provider": "openai", "model": plan.model,
                               "decision_requested": decision_requested}
    if case is not None and result is not None:
        payload["context"] = _context_from_analysis(case, result)
    elif plan.context:
        payload["context"] = plan.context
    if source_branch:
        payload["branched_from"] = source_branch
    if triage:
        payload["triage"] = triage
    if triage_error:
        payload["triage_error"] = triage_error
    if case is not None:
        payload["decision"] = {"case": case, "analysis": result}
    elif decision_requested:
        payload["missing"] = plan.missing
        if validation_error:
            payload["draft_case"] = plan.case_json
            payload["validation_error"] = validation_error
    object_id = store.commit("turn", payload, expected_head=head, usage=usage)
    return TurnOutcome(reply, object_id, plan.model, usage, result, store.current_branch(), source_branch)
