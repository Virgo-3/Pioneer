"""Store an optional OpenAI key with Windows' per-user data protection API."""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from pathlib import Path

from .state import _atomic_write


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def credential_path() -> Path:
    roaming = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return roaming / "Dao" / "openai-key.dpapi"


def _legacy_credential_path() -> Path:
    roaming = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return roaming / "Pioneer" / "openai-key.dpapi"


def _open_clipboard(user32: object) -> None:
    # A copy operation can briefly leave the Windows clipboard locked.
    for attempt in range(5):
        if user32.OpenClipboard(None):
            return
        if attempt < 4:
            time.sleep(0.05)
    raise ctypes.WinError(ctypes.get_last_error())


def read_clipboard_text() -> str | None:
    """Read Unicode text only after the user asks to use their clipboard."""
    if os.name != "nt":
        raise OSError("Clipboard key setup is available only on Windows")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = ctypes.c_void_p
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    _open_clipboard(user32)
    try:
        if not user32.IsClipboardFormatAvailable(13):  # CF_UNICODETEXT
            return None
        handle = user32.GetClipboardData(13)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.wstring_at(pointer)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _crypt(data: bytes, *, protect: bool) -> bytes:
    if os.name != "nt":
        raise OSError("Windows account encryption is available only on Windows")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    function = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    function.argtypes = [ctypes.POINTER(_DataBlob), ctypes.c_wchar_p,
                         ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.c_void_p,
                         wintypes.DWORD, ctypes.POINTER(_DataBlob)] if protect else [
                             ctypes.POINTER(_DataBlob), ctypes.POINTER(ctypes.c_wchar_p),
                             ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.c_void_p,
                             wintypes.DWORD, ctypes.POINTER(_DataBlob)]
    function.restype = wintypes.BOOL
    buffer = ctypes.create_string_buffer(data)
    source = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = _DataBlob()
    try:
        if not function(ctypes.byref(source), None, None, None, None, 1,
                        ctypes.byref(result)):
            raise ctypes.WinError(ctypes.get_last_error())
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        ctypes.memset(buffer, 0, len(buffer))
        if result.pbData:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.LocalFree.argtypes = [ctypes.c_void_p]
            kernel32.LocalFree.restype = ctypes.c_void_p
            kernel32.LocalFree(ctypes.cast(result.pbData, ctypes.c_void_p))


def save_openai_key(key: str, path: Path | None = None) -> None:
    key = key.strip()
    if not key:
        raise ValueError("OpenAI key cannot be empty")
    _atomic_write(path or credential_path(), _crypt(key.encode("utf-8"), protect=True))


def load_openai_key(path: Path | None = None) -> str | None:
    if path is None:
        current = credential_path()
        path = current if current.is_file() or not _legacy_credential_path().is_file() else _legacy_credential_path()
    try:
        encrypted = path.read_bytes()
    except FileNotFoundError:
        return None
    return _crypt(encrypted, protect=False).decode("utf-8")
