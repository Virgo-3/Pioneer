"""Small content-addressed conversation store, independent of Git internals."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


BRANCH_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")
OBJECT_ID = re.compile(r"^[0-9a-f]{64}$")


class StoreError(Exception):
    pass


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Store:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.data = self.root / ".pioneer"

    @property
    def exists(self) -> bool:
        return (self.data / "HEAD").is_file()

    def require(self) -> None:
        if not self.exists:
            raise StoreError(f"No Pioneer workspace in {self.root}. Run 'pioneer init' first.")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.require()
        lock = self.data / "LOCK"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise StoreError("Workspace is locked by another operation. Retry in a moment.") from exc
        try:
            with os.fdopen(fd, "w", encoding="ascii") as stream:
                stream.write(str(os.getpid()))
            yield
        finally:
            lock.unlink(missing_ok=True)

    def init(self) -> str:
        if self.exists:
            raise StoreError("Pioneer workspace already exists here.")
        (self.data / "objects").mkdir(parents=True, exist_ok=True)
        (self.data / "refs").mkdir(parents=True, exist_ok=True)
        root_id = self._write_object({"schema": 1, "parent": None, "kind": "root", "timestamp": _now(), "payload": {"title": "Pioneer"}})
        _atomic_write(self.data / "refs" / "main", (root_id + "\n").encode("ascii"))
        _atomic_write(self.data / "HEAD", b"main\n")
        return root_id

    def _write_object(self, obj: dict[str, Any]) -> str:
        data = _json_bytes(obj)
        object_id = hashlib.sha256(data).hexdigest()
        path = self.data / "objects" / f"{object_id}.json"
        if not path.exists():
            _atomic_write(path, data)
        return object_id

    def read_object(self, object_id: str) -> dict[str, Any]:
        self.require()
        if not OBJECT_ID.fullmatch(object_id):
            raise StoreError("Invalid object ID; use a full 64-character hash.")
        path = self.data / "objects" / f"{object_id}.json"
        try:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != object_id:
                raise StoreError(f"Corrupt object: {object_id}")
            obj = json.loads(raw)
            if not isinstance(obj, dict) or obj.get("schema") != 1:
                raise StoreError(f"Unsupported object: {object_id}")
            return obj
        except FileNotFoundError as exc:
            raise StoreError(f"Object not found: {object_id}") from exc
        except json.JSONDecodeError as exc:
            raise StoreError(f"Invalid object JSON: {object_id}") from exc

    def current_branch(self) -> str:
        self.require()
        branch = (self.data / "HEAD").read_text(encoding="ascii").strip()
        if not BRANCH_NAME.fullmatch(branch):
            raise StoreError("Invalid HEAD branch")
        return branch

    def branches(self) -> dict[str, str]:
        self.require()
        return {p.name: self._read_ref(p.name) for p in sorted((self.data / "refs").iterdir()) if p.is_file() and BRANCH_NAME.fullmatch(p.name)}

    def _read_ref(self, branch: str) -> str:
        if not BRANCH_NAME.fullmatch(branch):
            raise StoreError("Invalid branch name")
        try:
            object_id = (self.data / "refs" / branch).read_text(encoding="ascii").strip()
        except FileNotFoundError as exc:
            raise StoreError(f"Unknown branch: {branch}") from exc
        if not OBJECT_ID.fullmatch(object_id):
            raise StoreError(f"Invalid branch pointer: {branch}")
        self.read_object(object_id)
        return object_id

    def resolve(self, ref: str | None = None) -> str:
        self.require()
        if ref is None or ref == "HEAD":
            return self._read_ref(self.current_branch())
        if BRANCH_NAME.fullmatch(ref) and (self.data / "refs" / ref).exists():
            return self._read_ref(ref)
        self.read_object(ref)
        return ref

    def create_branch(self, name: str, from_ref: str | None = None) -> str:
        self.require()
        if not BRANCH_NAME.fullmatch(name):
            raise StoreError("Branch names must start with a letter and contain only letters, digits, ., _, or - (max 64).")
        with self._locked():
            if (self.data / "refs" / name).exists():
                raise StoreError(f"Branch already exists: {name}")
            object_id = self.resolve(from_ref)
            _atomic_write(self.data / "refs" / name, (object_id + "\n").encode("ascii"))
        return object_id

    def switch(self, name: str) -> str:
        with self._locked():
            object_id = self._read_ref(name)
            _atomic_write(self.data / "HEAD", (name + "\n").encode("ascii"))
        return object_id

    def rewind(self, ref: str) -> str:
        """Move the current branch pointer; old objects and ledger entries stay intact."""
        with self._locked():
            target = self.resolve(ref)
            _atomic_write(self.data / "refs" / self.current_branch(), (target + "\n").encode("ascii"))
        return target

    def commit(self, kind: str, payload: dict[str, Any], *, expected_head: str | None = None,
               usage: dict[str, Any] | list[dict[str, Any]] | None = None) -> str:
        if kind not in {"turn", "decision", "note"}:
            raise StoreError("Unsupported commit kind")
        with self._locked():
            branch = self.current_branch()
            parent = self._read_ref(branch)
            usage_records = [] if usage is None else usage if isinstance(usage, list) else [usage]
            if expected_head is not None and parent != expected_head:
                for record in usage_records:
                    self._append_usage({**record, "orphaned": True})
                raise StoreError("Branch changed during the API call. Usage was recorded; retry on the current branch.")
            object_id = self._write_object({"schema": 1, "parent": parent, "kind": kind, "timestamp": _now(), "payload": payload})
            for record in usage_records:
                self._append_usage({**record, "commit": object_id, "branch": branch})
            _atomic_write(self.data / "refs" / branch, (object_id + "\n").encode("ascii"))
        return object_id

    def _append_usage(self, record: dict[str, Any]) -> None:
        with (self.data / "usage.jsonl").open("ab") as stream:
            stream.write(_json_bytes({"timestamp": _now(), **record}))
            stream.flush()
            os.fsync(stream.fileno())

    def log(self, ref: str | None = None) -> list[tuple[str, dict[str, Any]]]:
        cursor = self.resolve(ref)
        result: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        while cursor is not None:
            if cursor in seen:
                raise StoreError("Cycle in commit history")
            seen.add(cursor)
            obj = self.read_object(cursor)
            result.append((cursor, obj))
            cursor = obj.get("parent")
        return result

    def messages(self, ref: str | None = None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for _, obj in reversed(self.log(ref)):
            if obj["kind"] == "turn":
                payload = obj["payload"]
                messages.extend([{"role": "user", "content": payload["user"]}, {"role": "assistant", "content": payload["assistant"]}])
        return messages

    def usage(self) -> list[dict[str, Any]]:
        self.require()
        path = self.data / "usage.jsonl"
        if not path.exists():
            return []
        try:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        except json.JSONDecodeError as exc:
            raise StoreError("Usage ledger is corrupt") from exc

    def verify(self) -> dict[str, int]:
        self.require()
        objects: set[str] = set()
        for path in (self.data / "objects").glob("*.json"):
            self.read_object(path.stem)
            objects.add(path.stem)
        for name in self.branches():
            self.log(name)
        entries = self.usage()
        for entry in entries:
            if "commit" in entry and entry["commit"] not in objects:
                raise StoreError(f"Usage ledger references missing commit: {entry['commit']}")
        return {"objects": len(objects), "branches": len(self.branches()), "usage_entries": len(entries)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
