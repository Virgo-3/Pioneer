"""Pioneer's command-line and interactive terminal interface."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision import DecisionError, analyze
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
    triage = sub.add_parser("triage", help="Ask Jev for three narrow uncertainty judgments")
    triage.add_argument("text", nargs="+")
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
        elif args.command == "verify":
            counts = store.verify()
            print(f"Verified {counts['objects']} objects, {counts['branches']} branches, {counts['usage_entries']} usage entries.")
        return 0
    except (StoreError, ProviderError, DecisionError, OSError, json.JSONDecodeError) as exc:
        print(f"Pioneer: {exc}", file=sys.stderr)
        return 1


def _ask(store: Store, text: str, model: str | None = None) -> None:
    outcome = run_turn(store, text, model=model)
    print(f"\nPioneer | {outcome.model} | {outcome.commit[:12]}\n{outcome.text}\n")
    for record in outcome.usage:
        print(f"{record['provider']}: {record['input_tokens']} input / {record['output_tokens']} output tokens")


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


def _chat(store: Store, model: str | None) -> None:
    store.require()
    print("PIONEER | conversational terminal")
    print(f"Branch {store.current_branch()} @ {store.resolve()[:12]} | /help for commands | /exit to leave")
    while True:
        try:
            line = input(f"\n{store.current_branch()} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        try:
            if line in {"/exit", "/quit"}:
                return
            if line == "/help":
                print("/branch NAME  /branches  /switch NAME  /log  /status  /usage")
                print("/decide FILE  /triage ACTION  /model MODEL  /exit")
                continue
            if line.startswith("/model "):
                model = line[7:].strip()
                print(f"Session model: {model}")
            elif line.startswith("/branch "):
                name = line[8:].strip()
                print(f"Created {name} at {store.create_branch(name)[:12]}.")
            elif line == "/branches":
                for name, object_id in store.branches().items():
                    print(f"{'*' if name == store.current_branch() else ' '} {name:<20} {object_id[:12]}")
            elif line.startswith("/switch "):
                name = line[8:].strip()
                print(f"Switched to {name} at {store.switch(name)[:12]}.")
            elif line == "/log":
                for object_id, obj in store.log()[:12]:
                    print(f"{object_id[:12]}  {obj['kind']:<8}  {obj['timestamp']}")
            elif line == "/status":
                print(f"{store.current_branch()} @ {store.resolve()[:12]}")
            elif line == "/usage":
                _usage(store, None)
            elif line.startswith("/decide "):
                _decide(store, shlex.split(line[8:])[0])
            elif line.startswith("/triage "):
                _triage(store, line[8:])
            elif line.startswith("/"):
                print("Unknown command. Type /help.")
            else:
                _ask(store, line, model)
        except (StoreError, ProviderError, DecisionError, OSError, json.JSONDecodeError, ValueError, IndexError) as exc:
            print(f"Pioneer: {exc}", file=sys.stderr)
