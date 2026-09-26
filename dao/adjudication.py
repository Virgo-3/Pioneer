"""Branch-local, append-only records for reviewing a natural-language question."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .state import Store, StoreError


class AdjudicationError(StoreError):
    """Invalid case, evidence, citation, or finding."""


ID_PREFIX = re.compile(r"^[0-9a-f]{8,64}$")
CONCLUSIONS = {"supported", "refuted", "disputed", "unresolved"}
MAX_TEXT = 20_000


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdjudicationError(f"{label} must be nonempty text.")
    result = value.strip()
    if len(result) > MAX_TEXT:
        raise AdjudicationError(f"{label} is too long (maximum {MAX_TEXT} characters).")
    return result


def _id(records: Iterable[dict[str, Any]], prefix: str, label: str) -> str:
    if not isinstance(prefix, str) or not ID_PREFIX.fullmatch(prefix.lower()):
        raise AdjudicationError(f"{label} needs at least eight hexadecimal ID characters.")
    matches = [item["id"] for item in records if item["id"].startswith(prefix.lower())]
    if not matches:
        raise AdjudicationError(f"{label} is not on this branch.")
    if len(matches) != 1:
        raise AdjudicationError(f"{label} ID is ambiguous; use more characters.")
    return matches[0]


def _all_cases(store: Store, ref: str | None = None) -> list[dict[str, Any]]:
    """Rebuild the view from commits reachable through one branch or ref."""
    by_id: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for object_id, obj in reversed(store.log(ref)):
        if obj["kind"] != "note":
            continue
        record = obj["payload"].get("adjudication")
        if not isinstance(record, dict):
            continue
        kind = record.get("type")
        if kind == "case":
            case = {"id": object_id, "question": record["question"],
                    "note": record.get("note", ""), "evidence": [], "findings": [],
                    "current_finding": None}
            by_id[object_id] = case
            ordered.append(case)
        elif kind in {"evidence", "finding"}:
            case = by_id.get(record.get("case_id"))
            if case is None:
                raise AdjudicationError("A case record refers to a case outside its history.")
            entry = {"id": object_id, **{key: value for key, value in record.items()
                                          if key != "type"}}
            if kind == "evidence":
                case["evidence"].append(entry)
            else:
                case["findings"].append(entry)
                case["current_finding"] = entry
    return ordered


def cases(store: Store, ref: str | None = None) -> list[dict[str, Any]]:
    """List cases reachable on the selected branch, oldest first."""
    return _all_cases(store, ref)


def case_record(store: Store, case_id: str, ref: str | None = None) -> dict[str, Any]:
    """Show one case with all its evidence and successive findings."""
    available = _all_cases(store, ref)
    return next(case for case in available
                if case["id"] == _id(available, case_id, "Case"))


def open_case(store: Store, question: str, *, note: str = "",
              expected_head: str | None = None) -> str:
    question = _text(question, "Question")
    if not isinstance(note, str) or len(note) > MAX_TEXT:
        raise AdjudicationError("Note must be text no longer than 20,000 characters.")
    branch = store.current_branch()
    head = store.resolve()
    if expected_head is not None and expected_head != head:
        raise AdjudicationError("Branch changed while preparing the case. Please retry.")
    return store.commit("note", {"title": question,
                                 "adjudication": {"type": "case", "question": question,
                                                  "note": note.strip()}},
                        expected_head=head, expected_branch=branch)


def _citation(store: Store, citation: dict[str, str] | None,
              ref: str) -> dict[str, str] | None:
    if citation is None:
        return None
    if not isinstance(citation, dict) or set(citation) != {"commit", "role", "quote"}:
        raise AdjudicationError("Citation needs commit, role, and quote.")
    commit = citation["commit"]
    role = citation["role"]
    quote = _text(citation["quote"], "Quote")
    if role not in {"user", "assistant"}:
        raise AdjudicationError("Citation role must be user or assistant.")
    turns = [{"id": object_id, "payload": obj["payload"]}
             for object_id, obj in store.log(ref) if obj["kind"] == "turn"]
    full_id = _id(turns, commit, "Cited turn")
    source = next(item for item in turns if item["id"] == full_id)
    statement = source["payload"].get(role)
    if not isinstance(statement, str) or quote not in statement:
        raise AdjudicationError("The exact quote does not appear for that speaker in the cited turn.")
    return {"commit": full_id, "role": role, "quote": quote}


def add_evidence(store: Store, case_id: str, text: str, *,
                 citation: dict[str, str] | None = None,
                 expected_head: str | None = None) -> str:
    branch = store.current_branch()
    head = store.resolve()
    if expected_head is not None and expected_head != head:
        raise AdjudicationError("Branch changed while preparing evidence. Please retry.")
    case = case_record(store, case_id, head)
    text = _text(text, "Evidence")
    checked_citation = _citation(store, citation, head)
    record = {"type": "evidence", "case_id": case["id"], "text": text,
              "citation": checked_citation, "authored_by": "user"}
    return store.commit("note", {"title": f"Evidence: {text[:100]}", "adjudication": record},
                        expected_head=head, expected_branch=branch)


def make_finding(store: Store, case_id: str, conclusion: str, rationale: str, *,
                 support: Iterable[str] = (), oppose: Iterable[str] = (),
                 supersedes: str | None = None,
                 expected_head: str | None = None) -> str:
    branch = store.current_branch()
    head = store.resolve()
    if expected_head is not None and expected_head != head:
        raise AdjudicationError("Branch changed while preparing the finding. Please retry.")
    case = case_record(store, case_id, head)
    if conclusion not in CONCLUSIONS:
        raise AdjudicationError("Conclusion must be supported, refuted, disputed, or unresolved.")
    rationale = _text(rationale, "Rationale")
    evidence = case["evidence"]
    supporting = [_id(evidence, value, "Supporting evidence") for value in support]
    opposing = [_id(evidence, value, "Opposing evidence") for value in oppose]
    if len(set(supporting)) != len(supporting) or len(set(opposing)) != len(opposing):
        raise AdjudicationError("Each evidence record may be cited only once per side.")
    if set(supporting) & set(opposing):
        raise AdjudicationError("The same evidence cannot support and oppose one finding.")
    if conclusion == "supported" and not supporting:
        raise AdjudicationError("A supported finding needs supporting evidence.")
    if conclusion == "refuted" and not opposing:
        raise AdjudicationError("A refuted finding needs opposing evidence.")
    if conclusion == "disputed" and (not supporting or not opposing):
        raise AdjudicationError("A disputed finding needs evidence on both sides.")
    previous = case["current_finding"]
    if previous is None and supersedes is not None:
        raise AdjudicationError("There is no earlier finding to supersede.")
    if previous is not None and (supersedes is None or
                                 _id(case["findings"], supersedes, "Superseded finding") != previous["id"]):
        raise AdjudicationError("Name the current finding with --supersedes to revise it.")
    record = {"type": "finding", "case_id": case["id"], "conclusion": conclusion,
              "rationale": rationale, "support": supporting, "oppose": opposing,
              "supersedes": previous["id"] if previous else None, "authored_by": "user"}
    return store.commit("note", {"title": f"Finding: {case['question'][:100]}",
                                 "adjudication": record}, expected_head=head,
                        expected_branch=branch)
