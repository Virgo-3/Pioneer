"""Double-click entry point for the Windows executable."""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

from .cli import main as cli_main
from .state import Store, StoreError
from .windows_credentials import load_openai_key, read_clipboard_text, save_openai_key


def workspace_path() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    current = local / "Dao" / "workspace"
    legacy = local / "Pioneer" / "workspace"
    if (current / ".dao" / "HEAD").is_file():
        return current
    if (legacy / ".pioneer" / "HEAD").is_file() or (legacy / ".dao" / "HEAD").is_file():
        return legacy
    return current


def _load_saved_key(*, report_error: bool = False) -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    try:
        saved = load_openai_key()
    except (OSError, UnicodeError) as exc:
        if report_error:
            print(f"Saved OpenAI key could not be opened ({exc}). Enter a new one below.")
        return False
    if saved and not any(character.isspace() for character in saved):
        os.environ["OPENAI_API_KEY"] = saved
        return True
    return False


def _setup_key() -> None:
    if _load_saved_key(report_error=True):
        return
    print("To connect OpenAI, copy your API key first. Dao will not print it.")
    print("Press Enter for clipboard, H to type hidden, V to paste visibly, or O for offline tools.")
    print("You can also paste an sk- key here directly; your terminal may display it as you paste.")
    while True:
        try:
            entry = input("Key entry [clipboard]: ").strip()
            choice = entry.lower()
            if choice in {"", "p"}:
                try:
                    key = (read_clipboard_text() or "").strip()
                except OSError as exc:
                    print(f"Could not read the clipboard ({exc}). Try V for visible paste.")
                    continue
                if not key:
                    print("No text was on the clipboard. Copy the key, then press Enter again.")
                    continue
            elif choice == "h":
                print("Type or paste the key, then press Enter. Characters will not appear.")
                key = getpass.getpass("OpenAI API key: ").strip()
            elif choice == "v":
                print("Your key will be visible while you type or paste it.")
                key = input("OpenAI API key: ").strip()
            elif choice in {"o", "offline"}:
                return
            elif entry.startswith("sk-"):
                key = entry
            else:
                print("Choose Enter, H, V, or O, or paste a key beginning with sk-.")
                continue
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not key or any(character.isspace() for character in key):
            print("No single key was received. Try again, or choose O for offline tools.")
            continue
        print("Key received.")
        break
    os.environ["OPENAI_API_KEY"] = key
    try:
        answer = input("Save it for future launches with Windows account encryption? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        answer = "n"
    if answer in {"", "y", "yes"}:
        try:
            save_openai_key(key)
            print("Key saved for this Windows account.")
        except OSError as exc:
            print(f"Could not save key ({exc}). It will work for this session.")


def _needs_openai_key(args: list[str]) -> bool:
    """Identify the CLI command without mistaking a --repo path for one."""
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--":
            return index + 1 < len(args) and args[index + 1] in {"chat", "ask"}
        if argument == "--repo":
            index += 2
            continue
        if argument.startswith("--repo="):
            index += 1
            continue
        return argument in {"chat", "ask"}
    return False


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        # Conversational CLI commands use the same saved credential as double-click chat.
        if _needs_openai_key(args):
            _load_saved_key()
        return cli_main(args)
    try:
        store = Store(workspace_path())
        if not store.exists:
            store.init()
            print(f"Your Dao workspace is ready at {store.root}.")
    except (OSError, StoreError) as exc:
        print(f"Dao could not open its workspace: {exc}", file=sys.stderr)
        return 1
    _setup_key()
    return cli_main(["--repo", str(store.root), "chat"])


if __name__ == "__main__":
    raise SystemExit(main())
