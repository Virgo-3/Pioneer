"""PyInstaller entry point. Run Dao.exe with no arguments to start chatting."""

from dao.desktop import main


if __name__ == "__main__":
    raise SystemExit(main())
