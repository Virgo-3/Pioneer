"""Compare desired outcomes with results reported on the current branch.

An objective records a goal, a measurable target, and when to judge it. Later
observations are separate commits. All observations remain in history. A
correction supersedes the earlier report; comparisons use the current
checkpoint result or, without one, the latest progress report on the branch.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .state import Store, StoreError


class ObjectiveError(StoreError):
    """An objective or reported result cannot be recorded or compared."""


_HEX_PREFIX = re.compile(r"^[0-9a-f]{4,64}$")
_MAX_GOAL = 500
_MAX_METRIC = 240
_MAX_UNIT = 80
_MAX_DEADLINE = 160
_MAX_ACTION = 500
_MAX_NOTE = 1000
_MAX_AS_OF = 160


def _text(value: Any, name: str, limit: int, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ObjectiveError(f"{name} must be text.")
    result = value.strip()
    if required and not result:
        raise ObjectiveError(f"{name} cannot be blank.")
    if len(result) > limit:
        raise ObjectiveError(f"{name} must be {limit} characters or fewer.")
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObjectiveError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ObjectiveError(f"{name} must be a finite number.") from exc
    # Leave room for a finite actual-minus-desired comparison as well.
    if not math.isfinite(result) or abs(result) > 1e150:
        raise ObjectiveError(f"{name} must be a finite number within the supported range.")
    return result


def validate_objective(value: dict) -> dict:
    """Return a bounded, canonical target for a desired outcome.

    ``deadline`` may be a date or an explicit resolution condition. Dao
    compares results only when the user reports one; it does not verify them.
    """
    if not isinstance(value, dict):
        raise ObjectiveError("Objective must be an object.")
    goal = _text(value.get("goal"), "Goal", _MAX_GOAL, required=True)
    metric = _text(value.get("metric"), "Metric", _MAX_METRIC, required=True)
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in {"numeric", "binary"}:
        raise ObjectiveError("Kind must be 'numeric' or 'binary'.")
    direction = value.get("direction")
    if not isinstance(direction, str) or direction not in {"at_least", "at_most", "exact"}:
        raise ObjectiveError("Direction must be 'at_least', 'at_most', or 'exact'.")
    if kind == "binary" and direction != "exact":
        raise ObjectiveError("Binary objectives must use 'exact' direction.")
    desired = value.get("desired")
    if kind == "numeric":
        desired = _finite_number(desired, "Desired value")
    elif not isinstance(desired, bool):
        raise ObjectiveError("Desired value must be yes or no for a binary objective.")
    unit_value = value.get("unit", "")
    if unit_value is None and kind == "binary":
        unit_value = ""
    unit = _text(unit_value, "Unit", _MAX_UNIT, required=kind == "numeric")
    deadline = _text(value.get("deadline"), "Deadline", _MAX_DEADLINE, required=True)
    action_value = value.get("action", "")
    action = _text("" if action_value is None else action_value, "Action", _MAX_ACTION)
    return {
        "goal": goal,
        "metric": metric,
        "kind": kind,
        "desired": desired,
        "direction": direction,
        "unit": unit,
        "deadline": deadline,
        "action": action,
    }


def add_objective(store: Store, value: dict) -> str:
    """Save a desired outcome on the current branch and return its commit ID."""
    return store.commit("note", {"objective": validate_objective(value)})


def _actual_value(value: Any, objective: dict) -> float | bool:
    if objective["kind"] == "binary":
        if not isinstance(value, bool):
            raise ObjectiveError("Actual value must be yes or no for a binary objective.")
        return value
    return _finite_number(value, "Actual value")


def _normalized_time(value: str) -> str:
    # Conservative equality prevents a progress report from becoming final just
    # because it was recorded after a date. No calendar inference is made.
    return re.sub(r"^by\s+", "", " ".join(value.split()).casefold())


def objective_records(store: Store, ref: str | None = None) -> list[dict]:
    """Read objectives and every reported result visible from a branch/ref."""
    records: list[dict] = []
    by_id: dict[str, dict] = {}
    for commit_id, obj in reversed(store.log(ref)):
        if obj.get("kind") not in {"turn", "note"}:
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        if "objective" in payload:
            objective = validate_objective(payload["objective"])
            record = {
                "id": commit_id,
                "objective": objective,
                "observations": [],
                "timestamp": obj.get("timestamp"),
            }
            records.append(record)
            by_id[commit_id] = record
        if "actual" in payload:
            observation = payload["actual"]
            if not isinstance(observation, dict):
                raise ObjectiveError(f"Invalid result in commit {commit_id[:8]}.")
            objective_id = observation.get("objective_id")
            if objective_id == "$self" and "objective" in payload:
                objective_id = commit_id
            record = by_id.get(objective_id) if isinstance(objective_id, str) else None
            if record is None:
                raise ObjectiveError(f"Result {commit_id[:8]} refers to an unavailable objective.")
            actual = _actual_value(observation.get("value"), record["objective"])
            as_of = _text(observation.get("as_of"), "As-of", _MAX_AS_OF, required=True)
            note = _text(observation.get("note", ""), "Result note", _MAX_NOTE)
            replaces = observation.get("replaces_observation_id")
            prior = None
            if replaces is not None:
                if not isinstance(replaces, str):
                    raise ObjectiveError(f"Result {commit_id[:8]} has an invalid correction reference.")
                prior = next((item for item in record["observations"]
                              if item["id"] == replaces), None)
                if prior is None or prior.get("superseded_by"):
                    raise ObjectiveError(f"Result {commit_id[:8]} cannot replace that observation.")
            if _normalized_time(as_of) == _normalized_time(record["objective"]["deadline"]):
                # A checkpoint has one current result. This also folds legacy
                # same-checkpoint corrections that predate explicit references.
                for earlier in record["observations"]:
                    if (not earlier.get("superseded_by")
                            and _normalized_time(earlier["as_of"]) == _normalized_time(as_of)):
                        earlier["superseded_by"] = commit_id
            entry = {
                "id": commit_id,
                "objective_id": objective_id,
                "value": actual,
                "as_of": as_of,
                "note": note,
                "timestamp": obj.get("timestamp"),
            }
            if prior is not None:
                prior["superseded_by"] = commit_id
                entry["replaces_observation_id"] = replaces
            record["observations"].append(entry)
    return records


def record_actual(store: Store, ref: str, value: float | bool, as_of: str, *,
                  note: str = "", replaces: str | None = None) -> str:
    """Record progress, a final result, or a correction to an earlier report."""
    if not isinstance(ref, str) or not _HEX_PREFIX.fullmatch(ref):
        raise ObjectiveError("Use an objective ID or a unique prefix of at least four hex characters.")
    canonical_as_of = _text(as_of, "As-of", _MAX_AS_OF, required=True)
    canonical_note = _text(note, "Result note", _MAX_NOTE)
    branch = store.current_branch()
    head = store.resolve()
    matching = [record for record in objective_records(store, head) if record["id"].startswith(ref)]
    if not matching:
        raise ObjectiveError("No objective with that ID is visible on this branch.")
    if len(matching) > 1:
        raise ObjectiveError("Objective ID prefix is ambiguous; use more characters.")
    record = matching[0]
    actual = _actual_value(value, record["objective"])
    replaced_id = None
    if replaces is not None:
        if not isinstance(replaces, str) or not _HEX_PREFIX.fullmatch(replaces):
            raise ObjectiveError("Use an observation ID or a unique prefix of at least four hex characters.")
        prior = [item for item in record["observations"] if item["id"].startswith(replaces)]
        if len(prior) != 1 or prior[0].get("superseded_by"):
            raise ObjectiveError("A replaceable observation with that ID is not visible on this target.")
        replaced_id = prior[0]["id"]
    at_checkpoint = [item for item in record["observations"]
                     if not item.get("superseded_by")
                     and _normalized_time(item["as_of"]) == _normalized_time(canonical_as_of)]
    latest = at_checkpoint[-1] if at_checkpoint else None
    if (replaced_id is None and latest and latest["value"] == actual
            and latest["note"] == canonical_note):
        raise ObjectiveError("That result is already recorded.")
    observation = {"objective_id": record["id"], "value": actual,
                   "as_of": canonical_as_of, "note": canonical_note}
    if replaced_id is None and latest is not None:
        replaced_id = latest["id"]
    if replaced_id is not None:
        observation["replaces_observation_id"] = replaced_id
    return store.commit(
        "note",
        {"actual": observation},
        expected_head=head,
        expected_branch=branch,
    )


def compare_objective(record: dict) -> dict:
    """Compare the checkpoint result or latest progress, without mixing units.

    Gap is actual minus desired. Shortfall is the nonnegative distance from
    meeting the target, measured in the objective's own unit (0/1 for binary).
    Progress comparisons are provisional; only a matching deadline is final.
    """
    objective = validate_objective(record["objective"])
    observations = record.get("observations")
    if not isinstance(observations, list):
        raise ObjectiveError("Invalid objective observations.")
    deadline = _normalized_time(objective["deadline"])
    active_observations = [item for item in observations
                           if isinstance(item, dict) and not item.get("superseded_by")]
    checkpoint = [item for item in active_observations
                  if isinstance(item, dict) and isinstance(item.get("as_of"), str)
                  and _normalized_time(item["as_of"]) == deadline]
    latest = checkpoint[-1] if checkpoint else active_observations[-1] if active_observations else None
    result = {
        "id": record.get("id"),
        **objective,
        "actual": None,
        "as_of": None,
        "observation_id": None,
        "gap": None,
        "shortfall": None,
        "met": None,
        "status": "unobserved",
        "provisional": False,
    }
    if latest is None:
        return result
    if not isinstance(latest, dict):
        raise ObjectiveError("Invalid objective observation.")
    actual = _actual_value(latest.get("value"), objective)
    as_of = _text(latest.get("as_of"), "As-of", _MAX_AS_OF, required=True)
    final = _normalized_time(as_of) == deadline
    if objective["kind"] == "binary":
        gap = int(actual) - int(objective["desired"])
        shortfall = 0 if actual is objective["desired"] else 1
    else:
        gap = actual - objective["desired"]
        if objective["direction"] == "at_least":
            shortfall = max(-gap, 0.0)
        elif objective["direction"] == "at_most":
            shortfall = max(gap, 0.0)
        else:
            shortfall = abs(gap)
    result.update({
        "actual": actual,
        "as_of": as_of,
        "observation_id": latest.get("id"),
        "gap": gap,
        "shortfall": shortfall,
        "met": shortfall == 0 if final else None,
        "status": "final" if final else "progress",
        "provisional": not final,
    })
    return result


def open_objectives(store: Store, ref: str | None = None) -> list[dict]:
    """Return objectives without a final result, including progress to date."""
    comparisons = [compare_objective(record) for record in objective_records(store, ref)]
    return [item for item in comparisons if item["status"] != "final"]


def outcome_report(store: Store, ref: str | None = None, *, goal: str | None = None) -> dict:
    """Report each desired-versus-actual comparison and unit-free counts."""
    selected_goal = None if goal is None else _text(goal, "Goal", _MAX_GOAL, required=True)
    comparisons = [compare_objective(record) for record in objective_records(store, ref)]
    if selected_goal is not None:
        comparisons = [item for item in comparisons
                       if item["goal"].casefold() == selected_goal.casefold()]
    final = [item for item in comparisons if item["status"] == "final"]
    return {
        "goal": selected_goal,
        "count": len(comparisons),
        "final_count": len(final),
        "met_count": sum(item["met"] is True for item in final),
        "missed_count": sum(item["met"] is False for item in final),
        "progress_count": sum(item["status"] == "progress" for item in comparisons),
        "unobserved_count": sum(item["status"] == "unobserved" for item in comparisons),
        "comparisons": comparisons,
        "caveat": "Results are user reported and have not been independently verified.",
    }
