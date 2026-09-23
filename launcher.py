"""PyInstaller entry point. Run Pioneer.exe with no arguments to start chatting."""

from pioneer.desktop import main


if __name__ == "__main__":
    raise SystemExit(main())
