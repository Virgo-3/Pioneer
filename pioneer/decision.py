"""Finite-state decision analysis with an explicit option to wait for a signal."""

from __future__ import annotations

import math
from typing import Any


class DecisionError(Exception):
    pass


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DecisionError(f"{label} must be a finite number")
    return float(value)


def _probability(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0 <= result <= 1:
        raise DecisionError(f"{label} must be between 0 and 1")
    return result


def analyze(case: dict[str, Any]) -> dict[str, Any]:
    """Compare acting now with waiting, then choosing after an informative signal.

    A cell's `undo` value is the utility remaining after reversal. The engine uses
    reversal only when that value less `undo_cost` beats the original payoff.
    All utilities, costs, and delay cost must share the user's chosen unit.
    """
    if not isinstance(case, dict):
        raise DecisionError("Case must be a JSON object")
    states = case.get("states")
    actions = case.get("actions")
    if not isinstance(states, dict) or not states:
        raise DecisionError("states must be a nonempty map of names to prior probabilities")
    if not isinstance(actions, dict) or not actions:
        raise DecisionError("actions must be a nonempty map")
    priors = {str(name): _probability(p, f"states.{name}") for name, p in states.items()}
    if any(not name for name in priors) or abs(sum(priors.values()) - 1) > 1e-6:
        raise DecisionError("State names must be nonempty and probabilities must sum to 1")

    action_cells: dict[str, dict[str, dict[str, float | bool]]] = {}
    for name, action in actions.items():
        if not isinstance(name, str) or not name or not isinstance(action, dict):
            raise DecisionError("Each action needs a name and object definition")
        cost = _number(action.get("cost", 0), f"actions.{name}.cost")
        if cost < 0:
            raise DecisionError(f"actions.{name}.cost cannot be negative")
        outcomes = action.get("outcomes")
        if not isinstance(outcomes, dict) or set(outcomes) != set(priors):
            raise DecisionError(f"actions.{name}.outcomes must contain exactly the named states")
        cells: dict[str, dict[str, float | bool]] = {}
        for state, outcome in outcomes.items():
            if isinstance(outcome, (int, float)) and not isinstance(outcome, bool):
                outcome = {"payoff": outcome}
            if not isinstance(outcome, dict) or "payoff" not in outcome:
                raise DecisionError(f"actions.{name}.outcomes.{state} needs a payoff")
            payoff = _number(outcome["payoff"], f"actions.{name}.outcomes.{state}.payoff")
            reversal = outcome.get("undo")
            undo_cost = _number(outcome.get("undo_cost", 0), f"actions.{name}.outcomes.{state}.undo_cost")
            if undo_cost < 0:
                raise DecisionError("undo_cost cannot be negative")
            if reversal is None:
                effective, reversed_now = payoff - cost, False
            else:
                undo_value = _number(reversal, f"actions.{name}.outcomes.{state}.undo") - undo_cost
                effective, reversed_now = max(payoff, undo_value) - cost, undo_value > payoff
            cells[state] = {"payoff": payoff, "effective": effective, "reversed": reversed_now}
        action_cells[name] = cells

    values_now = {name: sum(priors[state] * cell["effective"] for state, cell in cells.items())
                  for name, cells in action_cells.items()}
    best_now = max(values_now, key=values_now.get)
    now_value = values_now[best_now]
    result: dict[str, Any] = {
        "title": str(case.get("title", "Decision")),
        "units": str(case.get("units", "utility points")),
        "action_values": values_now,
        "best_now": best_now,
        "best_now_value": now_value,
        "outcomes": action_cells,
        "recommendation": {"kind": "act", "action": best_now},
    }

    wait = case.get("wait")
    if wait is None:
        return result
    if not isinstance(wait, dict):
        raise DecisionError("wait must be an object")
    delay_cost = _number(wait.get("delay_cost", 0), "wait.delay_cost")
    info_cost = _number(wait.get("information_cost", 0), "wait.information_cost")
    if delay_cost < 0 or info_cost < 0:
        raise DecisionError("Waiting costs cannot be negative")
    likelihoods = wait.get("signals")
    if not isinstance(likelihoods, dict) or not likelihoods:
        raise DecisionError("wait.signals must map signal names to likelihoods by state")
    if any(not isinstance(row, dict) or set(row) != set(priors) for row in likelihoods.values()):
        raise DecisionError("Every signal must give a likelihood for every state")
    signals = {str(signal): {state: _probability(value, f"wait.signals.{signal}.{state}")
                             for state, value in row.items()} for signal, row in likelihoods.items()}
    for state in priors:
        if abs(sum(row[state] for row in signals.values()) - 1) > 1e-6:
            raise DecisionError(f"Signal likelihoods for state {state} must sum to 1")
    after_signal: dict[str, Any] = {}
    before_costs = 0.0
    for signal, row in signals.items():
        probability = sum(priors[state] * row[state] for state in priors)
        if probability == 0:
            continue
        posterior = {state: priors[state] * row[state] / probability for state in priors}
        values = {name: sum(posterior[state] * cells[state]["effective"] for state in priors)
                  for name, cells in action_cells.items()}
        best = max(values, key=values.get)
        before_costs += probability * values[best]
        after_signal[signal] = {"probability": probability, "posterior": posterior,
                                "action_values": values, "best_action": best}
    wait_value = before_costs - delay_cost - info_cost
    result["wait"] = {
        "value": wait_value,
        "value_of_information": before_costs - now_value,
        "delay_cost": delay_cost,
        "information_cost": info_cost,
        "maximum_total_wait_cost": before_costs - now_value,
        "signals": after_signal,
    }
    if wait_value > now_value + 1e-9:
        result["recommendation"] = {"kind": "wait", "action": None}
    return result
