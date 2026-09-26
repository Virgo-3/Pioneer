"""Connect a desired outcome, attributed forecasts, and reported results.

An outcome thread is a projection of commits on one branch ancestry. The
objective commit anchors the thread; forecasts remain attributed statements,
and observations remain separate, correctable commits. A forecast predicts
whether the objective will be met at its checkpoint, not the numeric result.
Legacy free-text forecasts have no objective link and remain in calibration.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .objectives import compare_objective, objective_records
from .state import Store, StoreError


class OutcomeError(StoreError):
    """An attached outcome forecast is invalid or has no visible objective."""


_OBJECT_ID = re.compile(r"^[0-9a-f]{64}$")
_MAX_QUOTE = 1000
_MAX_MODEL = 120


def validate_outcome_forecast(value: dict, *, source_text: str | None = None) -> dict:
    """Validate an attributed forecast attached to an objective thread.

    ``$self`` is resolved only while projecting a commit that also creates an
    objective. If source text is available, the quote must occur in it exactly.
    A note commit can carry an attributed quote without a turn message.
    """
    if not isinstance(value, dict):
        raise OutcomeError("Outcome forecast must be an object.")
    objective_id = value.get("objective_id")
    if objective_id != "$self" and (not isinstance(objective_id, str)
                                    or not _OBJECT_ID.fullmatch(objective_id)):
        raise OutcomeError("Outcome forecast needs a full objective ID or '$self'.")
    probability = value.get("probability")
    try:
        number = float(probability) if not isinstance(probability, bool) else float("nan")
    except (TypeError, ValueError, OverflowError):
        number = float("nan")
    if (not isinstance(probability, (int, float)) or not math.isfinite(number)
            or not 0 <= number <= 1):
        raise OutcomeError("Probability must be a finite number from 0 to 1.")
    source = value.get("source")
    if not isinstance(source, str) or source not in {"user", "openai"}:
        raise OutcomeError("Forecast source must be 'user' or 'openai'.")
    quote = value.get("quote")
    if not isinstance(quote, str) or not quote.strip() or len(quote) > _MAX_QUOTE:
        raise OutcomeError("Forecast quote must be nonblank text of at most 1000 characters.")
    if source_text is not None and quote not in source_text:
        raise OutcomeError("Forecast quote must exactly match its source turn.")
    model = value.get("model")
    if model is not None:
        if not isinstance(model, str) or not model.strip() or len(model) > _MAX_MODEL:
            raise OutcomeError("Forecast model must be nonblank text of at most 120 characters.")
        model = model.strip()
    return {
        "objective_id": objective_id,
        "probability": number,
        "source": source,
        "quote": quote,
        "model": model,
    }


def _checkpoint(value: str) -> str:
    # Match the conservative comparison rule in objectives: only an explicitly
    # matching checkpoint is final, regardless of when the commit was made.
    return re.sub(r"^by\s+", "", " ".join(value.split()).casefold())


def outcome_threads(store: Store, ref: str | None = None) -> list[dict]:
    """Project branch-visible objectives into unified outcome threads.

    Each forecast keeps its own score if it precedes the first final result.
    Later corrections change that score's outcome, while forecasts made at or
    after the first final result remain unscored, including a same-commit one.
    """
    commits = list(reversed(store.log(ref)))
    order = {commit_id: index for index, (commit_id, _) in enumerate(commits)}
    records = objective_records(store, ref)
    threads = [{
        "id": record["id"],
        "objective": record["objective"],
        "comparison": compare_objective(record),
        "forecasts": [],
    } for record in records]
    by_id = {thread["id"]: thread for thread in threads}
    first_final: dict[str, int] = {}
    for record in records:
        deadline = _checkpoint(record["objective"]["deadline"])
        final_positions = [order[item["id"]] for item in record["observations"]
                           if _checkpoint(item["as_of"]) == deadline]
        if final_positions:
            first_final[record["id"]] = min(final_positions)

    visible: set[str] = set()
    for index, (commit_id, obj) in enumerate(commits):
        if obj.get("kind") not in {"turn", "note"}:
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        if "objective" in payload:
            visible.add(commit_id)
        if "outcome_forecast" not in payload:
            continue
        raw = payload["outcome_forecast"]
        # Turn quotations are checkable against the recorded speaker. Notes
        # have no conversational source text to check against.
        source = raw.get("source") if isinstance(raw, dict) else None
        source_text = None
        if obj["kind"] == "turn":
            source_text = payload.get("user" if source == "user" else "assistant")
            if not isinstance(source_text, str):
                raise OutcomeError(f"Forecast {commit_id[:8]} has no source turn text.")
        forecast = validate_outcome_forecast(raw, source_text=source_text)
        objective_id = forecast["objective_id"]
        if objective_id == "$self":
            if commit_id not in visible:
                raise OutcomeError(f"Forecast {commit_id[:8]} has no same-commit objective.")
            objective_id = commit_id
        if objective_id not in visible or objective_id not in by_id:
            raise OutcomeError(f"Forecast {commit_id[:8]} refers to an unavailable objective.")
        thread = by_id[objective_id]
        final_position = first_final.get(objective_id)
        outcome = (thread["comparison"]["met"] if final_position is not None
                   and index < final_position else None)
        brier = (forecast["probability"] - int(outcome)) ** 2 if outcome is not None else None
        thread["forecasts"].append({
            "id": commit_id,
            "objective_id": objective_id,
            "probability": forecast["probability"],
            "source": forecast["source"],
            "quote": forecast["quote"],
            "model": forecast["model"],
            "timestamp": obj.get("timestamp"),
            "outcome": outcome,
            "brier": brier,
        })
    return threads


def outcome_report(store: Store, ref: str | None = None) -> dict:
    """Summarize targets and score the latest eligible forecast per source.

    Revisions of one source's forecast about one objective are one sample in
    aggregate calibration. All revisions remain visible in the thread.
    """
    threads = outcome_threads(store, ref)
    comparisons = [thread["comparison"] for thread in threads]
    final = [item for item in comparisons if item["status"] == "final"]
    latest_scored: dict[tuple[str, str], dict] = {}
    for thread in threads:
        for forecast in thread["forecasts"]:
            if forecast["brier"] is not None:
                latest_scored[(thread["id"], forecast["source"])] = forecast
    scores = [item["brier"] for item in latest_scored.values()]
    by_source = {}
    for source in ("user", "openai"):
        source_scores = [item["brier"] for (_, author), item in latest_scored.items()
                         if author == source]
        by_source[source] = {
            "scored_count": len(source_scores),
            "mean_brier": sum(source_scores) / len(source_scores) if source_scores else None,
        }
    return {
        "threads": threads,
        "count": len(threads),
        "final_count": len(final),
        "progress_count": sum(item["status"] == "progress" for item in comparisons),
        "unobserved_count": sum(item["status"] == "unobserved" for item in comparisons),
        "met_count": sum(item["met"] is True for item in final),
        "missed_count": sum(item["met"] is False for item in final),
        "scored_count": len(scores),
        "mean_brier": sum(scores) / len(scores) if scores else None,
        "by_source": by_source,
        "caveat": "Results are user reported; forecasts at or after the first final result are not scored.",
    }
