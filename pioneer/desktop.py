"""Double-click entry point for the Windows executable."""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

from .cli import main as cli_main
from .state import Store
from .windows_credentials import load_openai_key, read_clipboard_text, save_openai_key


def workspace_path() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    return local / "Pioneer" / "workspace"


def _setup_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    try:
        saved = load_openai_key()
    except (OSError, UnicodeError) as exc:
        print(f"Saved OpenAI key could not be opened ({exc}). Enter a new one below.")
        saved = None
    if saved:
        os.environ["OPENAI_API_KEY"] = saved
        return
    print("To connect OpenAI, copy your API key first. Pioneer will not display it.")
    print("Press Enter to use the clipboard, H to type hidden, V for visible paste, or O for offline tools.")
    while True:
        try:
            choice = input("Key entry [clipboard]: ").strip().lower()
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
            else:
                print("Choose Enter, H, V, or O.")
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


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        return cli_main(args)
    store = Store(workspace_path())
    if not store.exists:
        store.init()
        print(f"Your Pioneer workspace is ready at {store.root}.")
    _setup_key()
    return cli_main(["--repo", str(store.root), "chat"])


if __name__ == "__main__":
    raise SystemExit(main())
