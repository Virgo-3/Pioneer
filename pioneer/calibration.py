"""Branch-local forecasts, observed outcomes, and calibration feedback.

Forecasts are binary claims with an explicit probability and resolution deadline.
Their IDs are the content-addressed commits that first recorded them. Resolutions
are later commits, so correcting an outcome preserves the earlier record.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .state import Store, StoreError


class CalibrationError(StoreError):
    """A forecast or outcome cannot be recorded or summarized."""


_HEX_PREFIX = re.compile(r"^[0-9a-f]{4,64}$")
_MAX_EVENT = 500
_MAX_DEADLINE = 160
_MAX_TOPIC = 120
_MAX_MODEL = 120
_MAX_NOTE = 1000


def _bounded_text(value: Any, name: str, limit: int, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise CalibrationError(f"{name} must be text.")
    result = value.strip()
    if required and not result:
        raise CalibrationError(f"{name} cannot be blank.")
    if len(result) > limit:
        raise CalibrationError(f"{name} must be {limit} characters or fewer.")
    return result


def validate_forecast(value: dict, *, source: str, topic: str = "",
                      model: str | None = None) -> dict:
    """Return a canonical, bounded binary forecast for storage.

    ``deadline`` is a human-readable resolution date or condition. The app does
    not claim to know an outcome merely because this deadline has passed.
    """
    if not isinstance(value, dict):
        raise CalibrationError("Forecast must be an object.")
    event = _bounded_text(value.get("event"), "Event", _MAX_EVENT, required=True)
    deadline = _bounded_text(value.get("deadline"), "Deadline", _MAX_DEADLINE, required=True)
    probability = value.get("probability")
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1 or not math.isfinite(float(probability))):
        raise CalibrationError("Probability must be a finite number from 0 to 1.")
    if source not in {"pioneer", "user"}:
        raise CalibrationError("Forecast source must be 'pioneer' or 'user'.")
    canonical_topic = _bounded_text(topic, "Topic", _MAX_TOPIC)
    canonical_model = None if model is None else _bounded_text(model, "Model", _MAX_MODEL, required=True)
    return {
        "event": event,
        "probability": float(probability),
        "deadline": deadline,
        "topic": canonical_topic,
        "source": source,
        "model": canonical_model,
    }


def add_forecast(store: Store, event: str, probability: float, deadline: str, *,
                 topic: str = "", source: str = "user", model: str | None = None) -> str:
    """Record a forecast on the current branch and return its full commit ID."""
    forecast = validate_forecast(
        {"event": event, "probability": probability, "deadline": deadline},
        source=source, topic=topic, model=model,
    )
    return store.commit("note", {"forecast": forecast})


def forecast_records(store: Store, ref: str | None = None) -> list[dict]:
    """Read forecasts and their latest outcomes from one branch ancestry."""
    records: list[dict] = []
    by_id: dict[str, dict] = {}
    for commit_id, obj in reversed(store.log(ref)):
        if obj.get("kind") not in {"turn", "note"}:
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        if "forecast" in payload:
            raw = payload["forecast"]
            if not isinstance(raw, dict):
                raise CalibrationError(f"Invalid forecast in commit {commit_id[:8]}.")
            forecast = validate_forecast(raw, source=raw.get("source"),
                                         topic=raw.get("topic", ""), model=raw.get("model"))
            record = {"id": commit_id, "forecast": forecast, "resolution": None,
                      "timestamp": obj.get("timestamp")}
            records.append(record)
            by_id[commit_id] = record
        if "resolution" in payload:
            resolution = payload["resolution"]
            if not isinstance(resolution, dict):
                raise CalibrationError(f"Invalid resolution in commit {commit_id[:8]}.")
            forecast_id = resolution.get("forecast_id")
            outcome = resolution.get("outcome")
            if not isinstance(forecast_id, str) or not isinstance(outcome, bool):
                raise CalibrationError(f"Invalid resolution in commit {commit_id[:8]}.")
            record = by_id.get(forecast_id)
            if record is None:
                raise CalibrationError(f"Resolution {commit_id[:8]} refers to an unavailable forecast.")
            note = _bounded_text(resolution.get("note", ""), "Resolution note", _MAX_NOTE)
            record["resolution"] = {"id": commit_id, "forecast_id": forecast_id,
                                    "outcome": outcome, "note": note,
                                    "timestamp": obj.get("timestamp")}
    return records


def open_forecasts(store: Store, ref: str | None = None) -> list[dict]:
    """List unresolved forecasts visible from a branch or commit."""
    return [{"id": record["id"], **record["forecast"]}
            for record in forecast_records(store, ref) if record["resolution"] is None]


def resolve_forecast(store: Store, ref: str, outcome: bool, *, note: str = "") -> str:
    """Record an outcome, or a later correction, on the current branch."""
    if not isinstance(ref, str) or not _HEX_PREFIX.fullmatch(ref):
        raise CalibrationError("Use a forecast ID or a unique prefix of at least four hex characters.")
    if not isinstance(outcome, bool):
        raise CalibrationError("Outcome must be yes or no.")
    canonical_note = _bounded_text(note, "Resolution note", _MAX_NOTE)
    branch = store.current_branch()
    head = store.resolve()
    matching = [record for record in forecast_records(store, head) if record["id"].startswith(ref)]
    if not matching:
        raise CalibrationError("No forecast with that ID is visible on this branch.")
    if len(matching) > 1:
        raise CalibrationError("Forecast ID prefix is ambiguous; use more characters.")
    record = matching[0]
    previous = record["resolution"]
    if previous is not None and previous["outcome"] is outcome:
        raise CalibrationError("That outcome is already recorded. Use the opposite outcome to correct it.")
    return store.commit("note", {"resolution": {"forecast_id": record["id"],
                                                 "outcome": outcome, "note": canonical_note}},
                        expected_head=head, expected_branch=branch)


def calibration_report(store: Store, ref: str | None = None, *, source: str = "pioneer",
                       topic: str | None = None) -> dict:
    """Score resolved binary forecasts; an empty sample has null metrics."""
    if source not in {"pioneer", "user", "all"}:
        raise CalibrationError("Source must be 'pioneer', 'user', or 'all'.")
    selected_topic = None if topic is None else _bounded_text(topic, "Topic", _MAX_TOPIC)
    selected = []
    for record in forecast_records(store, ref):
        forecast, resolution = record["forecast"], record["resolution"]
        if resolution is None or (source != "all" and forecast["source"] != source):
            continue
        if selected_topic is not None and forecast["topic"].casefold() != selected_topic.casefold():
            continue
        selected.append((forecast["probability"], int(resolution["outcome"])))

    buckets = []
    for index in range(5):
        lower, upper = index / 5, (index + 1) / 5
        values = [(probability, outcome) for probability, outcome in selected
                  if lower <= probability < upper or (index == 4 and probability == 1)]
        count = len(values)
        buckets.append({
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_predicted": sum(p for p, _ in values) / count if count else None,
            "observed_rate": sum(y for _, y in values) / count if count else None,
        })
    count = len(selected)
    return {
        "source": source,
        "topic": selected_topic,
        "count": count,
        "mean_predicted": sum(p for p, _ in selected) / count if count else None,
        "observed_rate": sum(y for _, y in selected) / count if count else None,
        "brier_score": sum((p - y) ** 2 for p, y in selected) / count if count else None,
        "buckets": buckets,
    }


def calibration_context(store: Store, topic: str, *, min_count: int = 5) -> dict | None:
    """Offer same-topic Pioneer feedback only when there is enough local data.

    The summary is evidence for human/model reflection, never a multiplier or
    automatic rewrite of a new forecast probability.
    """
    canonical_topic = _bounded_text(topic, "Topic", _MAX_TOPIC, required=True)
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise CalibrationError("Minimum count must be a positive integer.")
    report = calibration_report(store, source="pioneer", topic=canonical_topic)
    if report["count"] < min_count:
        return None
    return {**report, "caveat": (
        "Limited evidence from resolved forecasts on this branch and topic. "
        "Selection and small samples may distort the pattern; do not automatically adjust a new probability."
    )}
