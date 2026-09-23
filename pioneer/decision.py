"""Finite-state decision analysis with an explicit option to wait for a signal."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


class DecisionError(Exception):
    pass


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise DecisionError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise DecisionError(f"{label} must be a finite number")
    return result


def _probability(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0 <= result <= 1:
        raise DecisionError(f"{label} must be between 0 and 1")
    return result


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise DecisionError(f"{label} exceeds the supported numeric range")
    return value


def _weighted_sum(terms: Iterable[tuple[float, float]], label: str) -> float:
    try:
        return _finite(math.fsum(_finite(weight * value, label) for weight, value in terms), label)
    except OverflowError as exc:
        raise DecisionError(f"{label} exceeds the supported numeric range") from exc


def _check_fields(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise DecisionError(f"{label} has unknown field(s): {', '.join(sorted(map(str, unknown)))}")


def _check_names(value: dict[Any, Any], label: str) -> None:
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise DecisionError(f"{label} names must be nonempty strings")


def analyze(case: dict[str, Any]) -> dict[str, Any]:
    """Compare acting now with waiting, then choosing after an informative signal.

    A cell's `undo` value is the utility remaining after reversal. The engine uses
    reversal only when that value less `undo_cost` beats the original payoff.
    All utilities, costs, and delay cost must share the user's chosen unit.
    """
    if not isinstance(case, dict):
        raise DecisionError("Case must be a JSON object")
    _check_fields(case, {"title", "units", "states", "actions", "wait"}, "Case")
    states = case.get("states")
    actions = case.get("actions")
    if not isinstance(states, dict) or not states:
        raise DecisionError("states must be a nonempty map of names to prior probabilities")
    if not isinstance(actions, dict) or not actions:
        raise DecisionError("actions must be a nonempty map")
    _check_names(states, "State")
    _check_names(actions, "Action")
    priors = {name: _probability(p, f"states.{name}") for name, p in states.items()}
    prior_total = math.fsum(priors.values())
    if abs(prior_total - 1) > 1e-6:
        raise DecisionError("State probabilities must sum to 1")
    priors = {state: probability / prior_total for state, probability in priors.items()}

    action_cells: dict[str, dict[str, dict[str, float | bool]]] = {}
    for name, action in actions.items():
        if not isinstance(action, dict):
            raise DecisionError("Each action needs a name and object definition")
        _check_fields(action, {"cost", "outcomes"}, f"actions.{name}")
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
            _check_fields(outcome, {"payoff", "undo", "undo_cost"},
                          f"actions.{name}.outcomes.{state}")
            payoff = _number(outcome["payoff"], f"actions.{name}.outcomes.{state}.payoff")
            reversal = outcome.get("undo")
            undo_cost = _number(outcome.get("undo_cost", 0), f"actions.{name}.outcomes.{state}.undo_cost")
            if undo_cost < 0:
                raise DecisionError("undo_cost cannot be negative")
            if reversal is None:
                if "undo_cost" in outcome:
                    raise DecisionError(f"actions.{name}.outcomes.{state}.undo_cost requires undo")
                effective, reversed_now = _finite(payoff - cost,
                                                   f"actions.{name}.outcomes.{state}.effective"), False
            else:
                undo_value = _finite(_number(reversal, f"actions.{name}.outcomes.{state}.undo") - undo_cost,
                                     f"actions.{name}.outcomes.{state}.undo_value")
                effective, reversed_now = _finite(max(payoff, undo_value) - cost,
                                                   f"actions.{name}.outcomes.{state}.effective"), undo_value > payoff
            cells[state] = {"payoff": payoff, "effective": effective, "reversed": reversed_now}
        action_cells[name] = cells

    values_now = {name: _weighted_sum(((priors[state], cell["effective"]) for state, cell in cells.items()),
                                      f"actions.{name}.expected_value")
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
    _check_fields(wait, {"delay_cost", "information_cost", "signals"}, "wait")
    delay_cost = _number(wait.get("delay_cost", 0), "wait.delay_cost")
    info_cost = _number(wait.get("information_cost", 0), "wait.information_cost")
    if delay_cost < 0 or info_cost < 0:
        raise DecisionError("Waiting costs cannot be negative")
    likelihoods = wait.get("signals")
    if not isinstance(likelihoods, dict) or not likelihoods:
        raise DecisionError("wait.signals must map signal names to likelihoods by state")
    _check_names(likelihoods, "Signal")
    if any(not isinstance(row, dict) or set(row) != set(priors) for row in likelihoods.values()):
        raise DecisionError("Every signal must give a likelihood for every state")
    signals = {signal: {state: _probability(value, f"wait.signals.{signal}.{state}")
                             for state, value in row.items()} for signal, row in likelihoods.items()}
    for state in priors:
        signal_total = math.fsum(row[state] for row in signals.values())
        if abs(signal_total - 1) > 1e-6:
            raise DecisionError(f"Signal likelihoods for state {state} must sum to 1")
        for row in signals.values():
            row[state] /= signal_total
    after_signal: dict[str, Any] = {}
    signal_values: list[tuple[float, float]] = []
    for signal, row in signals.items():
        probability = _weighted_sum(((priors[state], row[state]) for state in priors),
                                    f"wait.signals.{signal}.probability")
        if probability == 0:
            continue
        posterior = {state: _finite(priors[state] * row[state] / probability,
                                    f"wait.signals.{signal}.posterior.{state}") for state in priors}
        values = {name: _weighted_sum(((posterior[state], cells[state]["effective"])
                                       for state in priors), f"wait.signals.{signal}.actions.{name}.expected_value")
                  for name, cells in action_cells.items()}
        best = max(values, key=values.get)
        signal_values.append((probability, values[best]))
        after_signal[signal] = {"probability": probability, "posterior": posterior,
                                "action_values": values, "best_action": best}
    before_costs = _weighted_sum(signal_values, "wait.expected_value_before_costs")
    scale = max(abs(cell["effective"]) for cells in action_cells.values() for cell in cells.values())
    scale = max(scale, abs(delay_cost), abs(info_cost))
    roundoff = 16 * (len(priors) + len(signals) + 2) * math.ulp(scale)
    if abs(before_costs - now_value) <= roundoff:
        before_costs = now_value
    value_of_information = _finite(before_costs - now_value, "wait.value_of_information")
    wait_value = _finite(before_costs - delay_cost - info_cost, "wait.value")
    result["wait"] = {
        "value": wait_value,
        "value_of_information": value_of_information,
        "delay_cost": delay_cost,
        "information_cost": info_cost,
        "maximum_total_wait_cost": value_of_information,
        "signals": after_signal,
    }
    if wait_value > now_value + roundoff:
        result["recommendation"] = {"kind": "wait", "action": None}
    return result
