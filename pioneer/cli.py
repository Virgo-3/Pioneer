"""Pioneer's command-line and interactive terminal interface."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .calibration import add_forecast, calibration_report, open_forecasts, resolve_forecast
from .decision import DecisionError, analyze
from .objectives import (add_objective, compare_objective, objective_records,
                         outcome_report, record_actual)
from .pipeline import run_turn
from .providers import ProviderError, triage_jev
from .state import Store, StoreError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pioneer", description="Branchable conversational AI terminal")
    parser.add_argument("--repo", default=".", help="Pioneer workspace directory (default: current directory)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Initialize a Pioneer workspace")
    chat = sub.add_parser("chat", help="Open the interactive terminal")
    chat.add_argument("--model", help="OpenAI model for this session")
    ask = sub.add_parser("ask", help="Run one cohesive conversation turn")
    ask.add_argument("text", nargs="+", help="Message text")
    ask.add_argument("--model", help="OpenAI model")
    branch = sub.add_parser("branch", help="Create or list branches")
    branch.add_argument("name", nargs="?")
    branch.add_argument("--from", dest="from_ref", help="Existing branch or full commit ID")
    switch = sub.add_parser("switch", help="Switch the active branch")
    switch.add_argument("name")
    rewind = sub.add_parser("rewind", help="Move this branch to an earlier commit; creates a rescue branch")
    rewind.add_argument("ref")
    reset = sub.add_parser("reset", help="Restart main or another branch, preserving its previous history")
    reset.add_argument("branch", nargs="?", default="main")
    log = sub.add_parser("log", help="Show commit history")
    log.add_argument("--ref")
    log.add_argument("--limit", type=int, default=20)
    show = sub.add_parser("show", help="Show one commit as JSON")
    show.add_argument("ref", nargs="?", default="HEAD")
    sub.add_parser("status", help="Show current branch and commit")
    usage = sub.add_parser("usage", help="Show recorded token usage")
    usage.add_argument("--prices", help="Optional JSON price card for estimated USD cost")
    decide = sub.add_parser("decide", help="Analyze and save a decision case from JSON")
    decide.add_argument("file")
    decide.add_argument("--no-save", action="store_true", help="Analyze without creating a commit")
    triage = sub.add_parser("triage", help="Ask Jev for four narrow uncertainty judgments")
    triage.add_argument("text", nargs="+")
    forecast = sub.add_parser("forecast", help="Record a yes/no prediction with a resolution condition")
    forecast.add_argument("probability", help="Probability such as 0.7 or 70%%")
    forecast.add_argument("event", nargs="+", help="Specific event to check later")
    forecast.add_argument("--by", dest="deadline", required=True, help="Resolution date or condition")
    forecast.add_argument("--topic", default="", help="Optional topic for grouped calibration")
    sub.add_parser("forecasts", help="List open predictions on this branch")
    resolution = sub.add_parser("resolve", help="Report whether a predicted event happened")
    resolution.add_argument("id", help="Forecast ID or unique prefix")
    resolution.add_argument("outcome", choices=("yes", "no"))
    target = sub.add_parser("target", help="Set a desired outcome to compare with reality later")
    target.add_argument("goal", help="What you are trying to achieve")
    target.add_argument("metric", help="What will be measured")
    target.add_argument("--desired", required=True, help="Target number or yes/no")
    target.add_argument("--by", dest="deadline", required=True, help="Check date or condition")
    target.add_argument("--direction", choices=("at_least", "at_most", "exact"),
                        help="Numeric success rule (default: at_least)")
    target.add_argument("--unit", default="", help="Unit for numeric targets, such as users or dollars")
    target.add_argument("--action", default="", help="Action associated with the target, if any")
    sub.add_parser("targets", help="List outcome targets on this branch")
    observe = sub.add_parser("observe", help="Record an actual value for a target")
    observe.add_argument("id", help="Target ID or unique prefix")
    observe.add_argument("value", help="Actual number or yes/no")
    observe.add_argument("--at", dest="as_of", required=True, help="Observation date or condition")
    observe.add_argument("--note", default="", help="Optional explanation of what happened")
    calibration = sub.add_parser("calibration", help="Compare desired outcomes with reported actual outcomes")
    calibration.add_argument("--goal", help="Show only one goal")
    accuracy = sub.add_parser("forecast-accuracy", help="Compare forecast probabilities with reported events")
    accuracy.add_argument("--topic", help="Show only one forecast topic")
    accuracy.add_argument("--source", choices=("pioneer", "user", "all"), default="pioneer")
    sub.add_parser("verify", help="Check stored object hashes, refs, and ledger links")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = Store(args.repo)
    try:
        if args.command == "init":
            print(f"Initialized Pioneer at {store.root} ({store.init()[:12]}).")
        elif args.command == "chat":
            _chat(store, args.model)
        elif args.command == "ask":
            _ask(store, " ".join(args.text), args.model)
        elif args.command == "branch":
            if args.name:
                object_id = store.create_branch(args.name, args.from_ref)
                print(f"Created {args.name} at {object_id[:12]}.")
            else:
                for name, object_id in store.branches().items():
                    print(f"{'*' if name == store.current_branch() else ' '} {name:<20} {object_id[:12]}")
        elif args.command == "switch":
            print(f"Switched to {args.name} at {store.switch(args.name)[:12]}.")
        elif args.command == "rewind":
            old = store.resolve()
            rescue = f"rescue-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{old[:6]}"
            store.create_branch(rescue, old)
            print(f"Moved {store.current_branch()} to {store.rewind(args.ref)[:12]}; prior head saved as {rescue}.")
        elif args.command == "reset":
            _reset(store, args.branch)
        elif args.command == "log":
            if args.limit < 1:
                raise StoreError("--limit must be positive")
            for object_id, obj in store.log(args.ref)[:args.limit]:
                payload = obj["payload"]
                title = payload.get("user") or payload.get("title") or payload.get("text") or obj["kind"]
                print(f"{object_id[:12]}  {obj['timestamp']}  {obj['kind']:<8}  {str(title).splitlines()[0][:68]}")
        elif args.command == "show":
            object_id = store.resolve(args.ref)
            print(json.dumps({"id": object_id, **store.read_object(object_id)}, indent=2, ensure_ascii=False))
        elif args.command == "status":
            print(f"{store.current_branch()} @ {store.resolve()[:12]}  ({len(store.log()) - 1} commits)")
        elif args.command == "usage":
            _usage(store, args.prices)
        elif args.command == "decide":
            _decide(store, args.file, save=not args.no_save)
        elif args.command == "triage":
            _triage(store, " ".join(args.text))
        elif args.command == "forecast":
            _forecast(store, args.probability, " ".join(args.event), args.deadline, args.topic)
        elif args.command == "forecasts":
            _print_forecasts(store)
        elif args.command == "resolve":
            _resolve_forecast(store, args.id, args.outcome)
        elif args.command == "target":
            _target(store, args.goal, args.metric, args.desired, args.deadline,
                    args.direction, args.unit, args.action)
        elif args.command == "targets":
            _print_targets(store)
        elif args.command == "observe":
            _observe(store, args.id, args.value, args.as_of, args.note)
        elif args.command == "calibration":
            _print_outcome_calibration(store, goal=args.goal)
        elif args.command == "forecast-accuracy":
            _print_forecast_accuracy(store, source=args.source, topic=args.topic)
        elif args.command == "verify":
            counts = store.verify()
            print(f"Verified {counts['objects']} objects, {counts['branches']} branches, {counts['usage_entries']} usage entries.")
        return 0
    except (StoreError, ProviderError, DecisionError, OSError, json.JSONDecodeError) as exc:
        print(f"Pioneer: {exc}", file=sys.stderr)
        return 1


def _ask(store: Store, text: str, model: str | None = None, *, interactive: bool = False) -> None:
    outcome = run_turn(store, text, model=model)
    if outcome.branched_from:
        print(f"Exploring on {outcome.branch}. Your conversation on {outcome.branched_from} is still there.")
    if outcome.text:
        print(f"Pioneer: {outcome.text}" if interactive else outcome.text)
    for notice in outcome.notices:
        print(f"Pioneer record: {notice}")


def _triage(store: Store, text: str) -> None:
    store.require()
    if not text.strip():
        raise StoreError("Proposed action cannot be empty")
    head = store.resolve()
    result = triage_jev(text)
    object_id = store.commit("note", {"title": "Jev triage", "text": text, "result": result},
                             expected_head=head,
                             usage={"provider": "jev", "model": result["model"],
                                    "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"]})
    print(f"Jev triage | {object_id[:12]}")
    for label, probability in result["scores"].items():
        print(f"  {label.replace('_', ' '):<22} {probability:.1%}")
    print("These are judgments about the description, not measured outcome probabilities. Use 'decide' for an explicit decision case.")


def _parse_probability(value: str) -> float:
    raw = value.strip()
    percentage = raw.endswith("%")
    try:
        probability = float(raw[:-1].strip() if percentage else raw)
    except ValueError as exc:
        raise StoreError("Probability must be between 0 and 1, or a percentage such as 70%.") from exc
    if not math.isfinite(probability):
        raise StoreError("Probability must be a finite number.")
    if percentage:
        probability /= 100
    elif probability > 1:
        raise StoreError("Use 0.7 or 70%, not 70; bare numbers above 1 are ambiguous.")
    if not 0 <= probability <= 1:
        raise StoreError("Probability must be between 0 and 1, or a percentage from 0% to 100%.")
    return probability


def _percentage(value: float) -> str:
    return f"{value * 100:g}%"


def _measurement(value: str) -> bool | float:
    raw = value.strip()
    if raw.casefold() in {"yes", "true"}:
        return True
    if raw.casefold() in {"no", "false"}:
        return False
    try:
        number = float(raw)
    except ValueError as exc:
        raise StoreError("Value must be a finite number or yes/no.") from exc
    if not math.isfinite(number):
        raise StoreError("Value must be a finite number or yes/no.")
    return number


def _target(store: Store, goal: str, metric: str, desired: str, deadline: str,
            direction: str | None = None, unit: str = "", action: str = "") -> None:
    wanted = _measurement(desired)
    kind = "binary" if isinstance(wanted, bool) else "numeric"
    rule = direction or ("exact" if kind == "binary" else "at_least")
    if kind == "binary" and rule != "exact":
        raise StoreError("A yes/no target uses the exact direction.")
    if kind == "numeric" and not unit.strip():
        raise StoreError("Numeric targets need a unit: --unit UNIT or /target ... | DIRECTION | UNIT.")
    target_id = add_objective(store, {"goal": goal.strip(), "metric": metric.strip(),
                                      "kind": kind, "desired": wanted, "direction": rule,
                                      "unit": unit.strip(), "deadline": deadline.strip(),
                                      "action": action.strip()})
    print(f"Saved target {target_id[:12]} on {store.current_branch()}.")
    when = deadline.strip() if deadline.strip().casefold().startswith("by ") else f"by {deadline.strip()}"
    print(f"  Desired: {metric.strip()} {rule.replace('_', ' ')} {_format_value(wanted, unit)} "
          f"{when}")
    print(f"  Goal: {goal.strip()}")
    print(f"  ID: {target_id}")


def _format_value(value: bool | float, unit: str = "") -> str:
    measured = "yes" if value is True else "no" if value is False else f"{value:g}"
    return measured + (f" {unit}" if unit else "")


def _chat_target(store: Store, argument: str) -> None:
    parts = [part.strip() for part in argument.split("|")]
    if len(parts) not in {4, 5, 6, 7} or not all(parts[:4]):
        raise StoreError("Use /target GOAL | METRIC | DESIRED | WHEN [| DIRECTION | UNIT | ACTION].")
    _target(store, parts[0], parts[1], parts[2], parts[3],
            (parts[4] or None) if len(parts) >= 5 else None,
            parts[5] if len(parts) >= 6 else "",
            parts[6] if len(parts) == 7 else "")


def _chat_observe(store: Store, argument: str) -> None:
    parts = [part.strip() for part in argument.split("|")]
    if len(parts) not in {3, 4} or not all(parts[:3]):
        raise StoreError("Use /observe ID | ACTUAL | WHEN [| NOTE]. Find IDs with /targets.")
    _observe(store, parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else "")


def _print_target(comparison: dict[str, Any]) -> None:
    unit = comparison["unit"]
    deadline = comparison["deadline"]
    when = deadline if deadline.casefold().startswith("by ") else f"by {deadline}"
    print(f"  {comparison['id'][:12]}  {_brief(comparison['goal'], 90)}")
    print(f"    {comparison['metric']}: target {_format_value(comparison['desired'], unit)} "
          f"({comparison['direction'].replace('_', ' ')}) {when}")
    if comparison["action"]:
        print(f"    Action: {_brief(comparison['action'], 90)}")
    if comparison["status"] != "unobserved":
        qualifier = "final" if comparison["status"] == "final" else "progress only"
        print(f"    Reported actual: {_format_value(comparison['actual'], unit)} "
              f"at {comparison['as_of']} ({qualifier})")
        if comparison["kind"] == "numeric":
            prefix = "Gap" if comparison["status"] == "final" else "Provisional gap"
            print(f"    {prefix} (actual - desired): {comparison['gap']:+g} {unit}")
        if comparison["status"] == "final":
            print(f"    Target {'met' if comparison['met'] else 'missed'}")
    print(f"    ID: {comparison['id']}")


def _print_targets(store: Store) -> None:
    records = objective_records(store)
    if not records:
        print(f"No outcome targets on {store.current_branch()}.")
        return
    print(f"Outcome targets on {store.current_branch()} ({len(records)}):")
    for record in records:
        _print_target(compare_objective(record))


def _observe(store: Store, ref: str, value: str, as_of: str, note: str = "") -> None:
    actual = _measurement(value)
    observation_id = record_actual(store, ref.strip(), actual, as_of.strip(), note=note.strip())
    payload = store.read_object(observation_id)["payload"]["actual"]
    target_id = payload["objective_id"]
    comparison = next(compare_objective(record) for record in objective_records(store)
                      if record["id"] == target_id)
    print(f"Recorded actual {_format_value(actual, comparison['unit'])} for target {target_id[:12]} "
          f"at {as_of.strip()}.")
    if comparison["observation_id"] != observation_id:
        print(f"The final comparison still uses the result reported at {comparison['deadline']}.")
    elif comparison["status"] == "final":
        print(f"Target {'met' if comparison['met'] else 'missed'}."
              + (f" Gap (actual - desired): {comparison['gap']:+g} {comparison['unit']}."
                 if comparison["kind"] == "numeric" else ""))
    else:
        print(f"Progress recorded; compare the final outcome at {comparison['deadline']}."
              + (f" Provisional gap: {comparison['gap']:+g} {comparison['unit']}."
                 if comparison["kind"] == "numeric" else ""))
    print(f"Target ID: {target_id}")
    print(f"Observation entry: {observation_id}")


def _print_outcome_calibration(store: Store, *, goal: str | None = None) -> None:
    report = outcome_report(store, goal=goal)
    if report["count"] == 0:
        print(f"No outcome targets" + (f" for {goal}" if goal else "") +
              f" on {store.current_branch()} yet. Set one with 'target' or /target.")
        return
    print(f"Outcome calibration on {store.current_branch()}" +
          (f" | goal: {goal}" if goal else ""))
    print(f"  Targets: {report['count']} | final: {report['final_count']} "
          f"| met: {report['met_count']} | missed: {report['missed_count']}")
    print(f"  In progress: {report['progress_count']} | not observed: {report['unobserved_count']}")
    for comparison in report["comparisons"]:
        _print_target(comparison)
    print("Actual outcomes come from your reports. A gap alone cannot prove why it happened or whether an action caused it.")


def _forecast(store: Store, probability: str, event: str, deadline: str, topic: str = "") -> None:
    chance = _parse_probability(probability)
    forecast_id = add_forecast(store, event.strip(), chance, deadline.strip(),
                               topic=topic.strip(), source="user")
    print(f"Saved forecast {forecast_id[:12]} on {store.current_branch()}: "
          f"{_percentage(chance)} that {event.strip()}.")
    print(f"Resolution condition: {deadline.strip()}")
    print(f"ID: {forecast_id}")


def _print_forecasts(store: Store) -> None:
    items = open_forecasts(store)
    if not items:
        print(f"No open forecasts on {store.current_branch()}.")
        return
    print(f"Open forecasts on {store.current_branch()} ({len(items)}):")
    for item in items:
        probability = item["probability"]
        print(f"  {item['id'][:12]}  {_percentage(probability)}  "
              f"[{item['source']}] {_brief(item['event'], 110)}")
        print(f"    Resolution condition: {_brief(item['deadline'], 110)}")
        if item.get("topic"):
            print(f"    Topic: {_brief(item['topic'], 110)}")
        print(f"    ID: {item['id']}")


def _resolve_forecast(store: Store, ref: str, outcome: str) -> None:
    resolution_id = resolve_forecast(store, ref, outcome == "yes")
    forecast_id = store.read_object(resolution_id)["payload"]["resolution"]["forecast_id"]
    print(f"Recorded your outcome for forecast {forecast_id[:12]}: "
          f"{'happened' if outcome == 'yes' else 'did not happen'}.")
    print(f"Forecast ID: {forecast_id}")
    print(f"Resolution entry: {resolution_id}")


def _print_forecast_accuracy(store: Store, *, source: str = "pioneer", topic: str | None = None) -> None:
    report = calibration_report(store, source=source, topic=topic)
    count = report["count"]
    label = "Pioneer" if source == "pioneer" else "your" if source == "user" else "all"
    if count == 0:
        subject = ("Pioneer forecasts" if source == "pioneer" else
                   "forecasts you entered" if source == "user" else "forecasts")
        print(f"No resolved {subject}" + (f" for {topic}" if topic else "") +
              f" on {store.current_branch()} yet.")
        if source == "pioneer":
            print("For your forecasts, use 'forecast-accuracy --source user' or '/forecast-accuracy user'.")
        return
    print(f"Forecast accuracy on {store.current_branch()} | {label} forecasts" +
          (f" | topic: {topic}" if topic else ""))
    print("  Outcomes were reported by a user; Pioneer has not verified them.")
    print(f"  Resolved: {count}")
    print(f"  Average forecast: {report['mean_predicted']:.1%}")
    print(f"  Reported event rate: {report['observed_rate']:.1%}")
    print(f"  Brier score: {report['brier_score']:.3f} (lower is better against reported outcomes)")
    buckets = report.get("buckets", [])
    if buckets:
        print("  Reported outcomes by confidence range:")
        for bucket in buckets:
            if not isinstance(bucket, dict) or not bucket.get("count"):
                continue
            label = (bucket.get("label") or bucket.get("range") or
                     f"{bucket['lower']:.0%}-{bucket['upper']:.0%}")
            expected = bucket.get("mean_predicted")
            observed = bucket.get("observed_rate")
            if isinstance(expected, (float, int)) and isinstance(observed, (float, int)):
                print(f"    {label}: {bucket['count']} forecast(s), "
                      f"predicted {expected:.1%}, reported {observed:.1%}")


def _chat_forecast(store: Store, argument: str) -> None:
    parts = [part.strip() for part in argument.split("|")]
    if len(parts) not in {3, 4} or not all(parts[:3]):
        raise StoreError("Use /forecast PROBABILITY | EVENT | DEADLINE [| TOPIC].")
    _forecast(store, parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else "")


def _chat_resolve(store: Store, argument: str) -> None:
    parts = argument.split()
    if len(parts) != 2 or parts[1].lower() not in {"yes", "no"}:
        raise StoreError("Use /resolve ID yes|no. Find IDs with /forecasts.")
    _resolve_forecast(store, parts[0], parts[1].lower())


def _decide(store: Store, file: str, *, save: bool = True) -> None:
    case = json.loads(Path(file).read_text(encoding="utf-8"))
    result = analyze(case)
    print(f"{result['title']}  [{result['units']}]")
    print("Act now:")
    for name, value in result["action_values"].items():
        print(f"  {name:<22} {value:,.3f}")
    if "wait" in result:
        wait = result["wait"]
        print(f"Wait and observe:         {wait['value']:,.3f}")
        print(f"Value of information:     {wait['value_of_information']:,.3f}")
        print(f"Maximum combined wait cost before acting wins: {wait['maximum_total_wait_cost']:,.3f}")
        for signal, info in wait["signals"].items():
            print(f"  If {signal} ({info['probability']:.1%}): {info['best_action']}")
    recommendation = result["recommendation"]
    print("Recommendation:", "wait for the signal" if recommendation["kind"] == "wait" else f"act now: {recommendation['action']}")
    if save:
        store.require()
        object_id = store.commit("decision", {"title": result["title"], "case": case, "analysis": result})
        print(f"Saved as {object_id[:12]} on {store.current_branch()}.")


def _usage(store: Store, prices_file: str | None) -> None:
    prices: dict[str, Any] = {}
    if prices_file:
        prices = json.loads(Path(prices_file).read_text(encoding="utf-8"))
        if not isinstance(prices, dict):
            raise StoreError("Price card must be a JSON object keyed by model")
    totals: dict[tuple[str, str], list[float]] = {}
    for item in store.usage():
        key = (str(item.get("provider", "?")), str(item.get("model", "?")))
        row = totals.setdefault(key, [0.0, 0.0, 0.0])
        row[0] += int(item.get("input_tokens", 0))
        row[1] += int(item.get("output_tokens", 0))
        row[2] += 1
    if not totals:
        print("No API usage recorded.")
        return
    for (provider, model), (input_tokens, output_tokens, calls) in sorted(totals.items()):
        line = f"{provider}/{model}: {int(calls)} calls | {int(input_tokens)} input | {int(output_tokens)} output tokens"
        rate = prices.get(model)
        if rate is not None:
            if not isinstance(rate, dict):
                raise StoreError(f"Invalid price card entry for {model}")
            input_rate = float(rate.get("input_per_million", 0))
            output_rate = float(rate.get("output_per_million", 0))
            if any(not (0 <= value < float("inf")) for value in (input_rate, output_rate)):
                raise StoreError("Price rates must be finite nonnegative numbers")
            estimate = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
            line += f" | estimated ${estimate:.6f}"
        print(line)
    if not prices_file:
        print("Cost estimate unavailable. Pass --prices FILE with rates per million tokens.")


def _brief(value: Any, limit: int = 88) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _latest_context(store: Store, ref: str | None = None) -> dict[str, Any] | None:
    for _, obj in store.log(ref):
        if obj["kind"] == "turn":
            context = obj["payload"].get("context")
            if isinstance(context, dict):
                return context
    return None


def _latest_jev_guidance(store: Store, goal: str) -> dict[str, Any] | None:
    for _, obj in store.log():
        if obj["kind"] != "turn":
            continue
        payload = obj["payload"]
        context = payload.get("context")
        if not isinstance(context, dict) or context.get("goal") != goal:
            return None
        if payload.get("jev_error"):
            return None
        jev = payload.get("jev")
        if isinstance(jev, dict) and jev.get("goal") == goal:
            guidance = jev.get("guidance")
            return guidance if isinstance(guidance, dict) else None
    return None


def _topic(store: Store, ref: str | None = None) -> str | None:
    for _, obj in store.log(ref):
        payload = obj["payload"]
        if obj["kind"] == "turn":
            context = payload.get("context")
            if isinstance(context, dict) and context.get("goal"):
                return _brief(context["goal"])
            return _brief(payload.get("user", "")) or None
        if obj["kind"] == "decision":
            return _brief(payload.get("title", "")) or None
    return None


def _orientation(store: Store) -> None:
    branch = store.current_branch()
    topic = _topic(store)
    print(f"On {branch}" + (f" | {topic}" if topic else " | new conversation"))
    context = _latest_context(store)
    if context and context.get("provisional_view"):
        print(f"Current view: {_brief(context['provisional_view'])}")


def _print_context(store: Store) -> None:
    context = _latest_context(store)
    if context is None:
        print("No working context on this branch yet. Tell Pioneer what you are considering.")
        return
    print(f"Working context on {store.current_branch()}:")
    for key, label in (("goal", "Goal"), ("provisional_view", "Current view")):
        if context.get(key):
            print(f"  {label}: {_brief(context[key], 160)}")
    for key, label in (("options", "Options"), ("known", "Known"),
                       ("uncertain", "Still uncertain"), ("next_questions", "Possible follow-ups")):
        values = context.get(key)
        if isinstance(values, list) and values:
            print(f"  {label}:")
            for value in values:
                print(f"    - {_brief(value, 160)}")
    guidance = _latest_jev_guidance(store, str(context.get("goal", "")))
    if guidance:
        labels = {"assumption_tension": "earlier assumptions", "urgency": "timing",
                  "reversibility": "what can be undone", "information_value": "what waiting could reveal"}
        priorities = guidance.get("priorities")
        if isinstance(priorities, list) and priorities:
            checks = [labels[item] for item in priorities if item in labels]
            if checks:
                print("Decision checks: " + ", ".join(checks) + " (Jev guidance)")


def _required(argument: str, usage: str) -> str:
    value = argument.strip()
    if not value or any(character.isspace() for character in value):
        raise StoreError(f"Use {usage}.")
    return value


def _file_argument(argument: str) -> str:
    value = argument.strip()
    if not value:
        raise StoreError("Use /decide FILE.")
    if value[0] in {'"', "'"}:
        if len(value) < 2 or value[-1] != value[0]:
            raise StoreError("Use /decide FILE (close the path's quote).")
        value = value[1:-1]
    return value


def _clear_terminal() -> None:
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def _reset(store: Store, branch: str, *, clear: bool = False) -> None:
    rescue, _ = store.reset_branch(branch)
    if clear:
        _clear_terminal()
    print(f"Started a fresh conversation on {branch}.")
    if rescue:
        print(f"Previous history is on {rescue}. Use /switch {rescue} to revisit it.")
    else:
        print(f"{branch} was already at its starting point.")


def _chat(store: Store, model: str | None) -> None:
    store.require()
    print("Pioneer | Talk through a choice, or type /help for commands.")
    _orientation(store)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY for conversation. Complete decision JSON still works locally.")
    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        try:
            parts = line.split(maxsplit=1)
            command = parts[0]
            argument = parts[1] if len(parts) == 2 else ""
            if command in {"/exit", "/quit"} and not argument:
                return
            if command == "/help" and not argument:
                print("Type a message to continue the conversation. Commands:")
                print("  /status                 Show the current branch and topic")
                print("  /context                Show the saved working context")
                print("  /branches               List conversations and their topics")
                print("  /branch NAME            Copy this conversation to a new branch")
                print("  /switch NAME            Continue on another branch")
                print("  /clear                  Clear the terminal display")
                print("  /reset [BRANCH]         Restart main or a named branch; save its old history")
                print("  /log                    Show recent saved turns")
                print("  /analysis               Show the latest full calculation")
                print("  /usage                  Show recorded API token usage")
                print("  /target GOAL | METRIC | DESIRED | WHEN [| DIRECTION | UNIT | ACTION]")
                print("                           Save a desired outcome (DESIRED: number or yes/no)")
                print("  /targets                List desired outcomes on this branch")
                print("  /observe ID | ACTUAL | WHEN [| NOTE]  Report what actually happened")
                print("  /calibration [GOAL]     Compare desired and actual outcomes")
                print("  /forecast P | EVENT | WHEN [| TOPIC]  Save a prediction (P: 0.7 or 70%)")
                print("  /forecasts              List open predictions and their IDs")
                print("  /resolve ID yes|no      Report whether a predicted event happened")
                print("  /forecast-accuracy [SOURCE]  Score probability estimates")
                print("  /decide FILE            Calculate a decision from JSON")
                print("  /triage ACTION          Ask Jev to assess an action")
                print("  /model MODEL            Select a model for this session")
                print("  /exit                   Leave the chat")
            elif command == "/model":
                if argument:
                    model = _required(argument, "/model MODEL")
                    print(f"Session model: {model}")
                else:
                    print(f"Session model: {model or os.environ.get('PIONEER_OPENAI_MODEL', 'gpt-6-astra')}")
            elif command == "/branch":
                name = _required(argument, "/branch NAME")
                object_id = store.create_branch(name)
                print(f"Created {name} at {object_id[:12]}. Type /switch {name} to continue there.")
            elif command == "/branches" and not argument:
                current = store.current_branch()
                for name, object_id in store.branches().items():
                    topic = _topic(store, name)
                    print(f"{'*' if name == current else ' '} {name:<20} {object_id[:12]}"
                          + (f"  {_brief(topic, 55)}" if topic else ""))
            elif command == "/switch":
                name = _required(argument, "/switch NAME")
                store.switch(name)
                _orientation(store)
            elif command == "/clear" and not argument:
                _clear_terminal()
                _orientation(store)
            elif command == "/reset":
                name = _required(argument or "main", "/reset [BRANCH]")
                _reset(store, name, clear=True)
                _orientation(store)
            elif command == "/log" and not argument:
                for object_id, obj in store.log()[:12]:
                    payload = obj["payload"]
                    title = payload.get("user") or payload.get("title") or payload.get("text") or obj["kind"]
                    print(f"{object_id[:12]}  {obj['kind']:<8}  {_brief(title, 72)}")
            elif command == "/status" and not argument:
                print(f"{store.current_branch()} @ {store.resolve()[:12]}  ({len(store.log()) - 1} saved entries)")
                topic = _topic(store)
                if topic:
                    print(f"Topic: {topic}")
                context = _latest_context(store)
                if context and context.get("provisional_view"):
                    print(f"Current view: {_brief(context['provisional_view'])}")
            elif command == "/context" and not argument:
                _print_context(store)
            elif command == "/usage" and not argument:
                _usage(store, None)
            elif command == "/target":
                _chat_target(store, argument)
            elif command == "/targets" and not argument:
                _print_targets(store)
            elif command == "/observe":
                _chat_observe(store, argument)
            elif command == "/calibration":
                _print_outcome_calibration(store, goal=argument.strip() or None)
            elif command == "/forecast":
                _chat_forecast(store, argument)
            elif command == "/forecasts" and not argument:
                _print_forecasts(store)
            elif command == "/resolve":
                _chat_resolve(store, argument)
            elif command == "/forecast-accuracy":
                source = argument.strip().lower() or "pioneer"
                if source not in {"pioneer", "user", "all"}:
                    raise StoreError("Use /forecast-accuracy [pioneer|user|all].")
                _print_forecast_accuracy(store, source=source)
            elif command == "/analysis" and not argument:
                for _, obj in store.log():
                    payload = obj["payload"]
                    decision = payload.get("decision")
                    if obj["kind"] == "decision":
                        decision = {"case": payload["case"], "analysis": payload["analysis"]}
                    if decision:
                        print(json.dumps(decision, indent=2, ensure_ascii=False))
                        break
                else:
                    print("No calculated decision on this branch yet.")
            elif command == "/decide":
                _decide(store, _file_argument(argument))
            elif command == "/triage":
                if not argument.strip():
                    raise StoreError("Use /triage ACTION.")
                _triage(store, argument)
            elif command.startswith("/"):
                print("Unknown command or extra argument. Type /help for commands.", file=sys.stderr)
            else:
                _ask(store, line, model, interactive=True)
        except (StoreError, ProviderError, DecisionError, OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Pioneer: {exc}", file=sys.stderr)
