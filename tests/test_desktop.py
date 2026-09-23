import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from pioneer.desktop import _setup_key, main, workspace_path
from pioneer.state import Store
from pioneer.windows_credentials import _open_clipboard, load_openai_key, save_openai_key


class DesktopTests(unittest.TestCase):
    def test_no_arguments_creates_per_user_workspace_then_opens_chat(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {"LOCALAPPDATA": temp, "OPENAI_API_KEY": "fake"}):
                with patch("pioneer.desktop.cli_main", return_value=0) as cli:
                    self.assertEqual(main([]), 0)
                    self.assertEqual(main([]), 0)
                workspace = workspace_path()
                self.assertEqual(workspace, Path(temp) / "Pioneer" / "workspace")
                self.assertTrue(Store(workspace).exists)
                cli.assert_called_with(["--repo", str(Store(workspace).root), "chat"])

    def test_arguments_retain_regular_cli(self):
        with patch("pioneer.desktop.cli_main", return_value=0) as cli, \
             patch("pioneer.desktop.load_openai_key") as load:
            self.assertEqual(main(["--help"]), 0)
            cli.assert_called_once_with(["--help"])
            load.assert_not_called()

    def test_explicit_cli_uses_saved_key(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("pioneer.desktop.load_openai_key", return_value="saved-key"):
            def run_cli(args):
                self.assertEqual(os.environ.get("OPENAI_API_KEY"), "saved-key")
                return 0

            with patch("pioneer.desktop.cli_main", side_effect=run_cli) as cli:
                self.assertEqual(main(["--repo", "workspace", "chat"]), 0)
                cli.assert_called_once_with(["--repo", "workspace", "chat"])

    def test_local_cli_command_does_not_load_saved_key(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("pioneer.desktop.load_openai_key") as load, \
             patch("pioneer.desktop.cli_main", return_value=0) as cli:
            self.assertEqual(main(["--repo", "ask", "status"]), 0)
            cli.assert_called_once_with(["--repo", "ask", "status"])
            load.assert_not_called()

    def test_explicit_ask_loads_saved_key(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("pioneer.desktop.load_openai_key", return_value="saved-key") as load, \
             patch("pioneer.desktop.cli_main", return_value=0) as cli:
            self.assertEqual(main(["--repo=workspace", "ask", "hello"]), 0)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "saved-key")
            load.assert_called_once_with()
            cli.assert_called_once_with(["--repo=workspace", "ask", "hello"])

    def test_clipboard_entry_saves_only_when_chosen(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("pioneer.desktop.load_openai_key", return_value=None), \
                 patch("pioneer.desktop.read_clipboard_text", return_value="  fake-key  "), \
                 patch("builtins.input", side_effect=["", "y"]), \
                 patch("pioneer.desktop.save_openai_key") as save:
                _setup_key()
                self.assertEqual(os.environ["OPENAI_API_KEY"], "fake-key")
                save.assert_called_once_with("fake-key")

    def test_visible_mode_accepts_typed_or_pasted_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("pioneer.desktop.load_openai_key", return_value=None), \
                 patch("builtins.input", side_effect=["v", "  fake-key  ", "n"]), \
                 patch("pioneer.desktop.save_openai_key") as save:
                _setup_key()
                self.assertEqual(os.environ["OPENAI_API_KEY"], "fake-key")
                save.assert_not_called()

    def test_direct_paste_at_first_prompt_preserves_key_case(self):
        key = "sk-proj-MiXeDcase123"
        output = StringIO()
        with patch.dict(os.environ, {}, clear=True), \
             patch("pioneer.desktop.load_openai_key", return_value=None), \
             patch("builtins.input", side_effect=[f"  {key}  ", "n"]), \
             patch("pioneer.desktop.save_openai_key") as save, \
             redirect_stdout(output):
            _setup_key()
            self.assertEqual(os.environ["OPENAI_API_KEY"], key)
            save.assert_not_called()
        self.assertNotIn(key, output.getvalue())

    def test_empty_clipboard_can_be_retried_or_skipped(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("pioneer.desktop.load_openai_key", return_value=None), \
                 patch("pioneer.desktop.read_clipboard_text", return_value=None), \
                 patch("builtins.input", side_effect=["", "o"]):
                _setup_key()
                self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_saved_key_skips_prompt(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("pioneer.desktop.load_openai_key", return_value="saved-key"), \
                 patch("pioneer.desktop.getpass.getpass") as prompt:
                _setup_key()
                self.assertEqual(os.environ["OPENAI_API_KEY"], "saved-key")
                prompt.assert_not_called()

    def test_clipboard_open_retries_after_transient_lock(self):
        class Clipboard:
            attempts = 0

            def OpenClipboard(self, owner):
                self.assert_owner(owner)
                self.attempts += 1
                return self.attempts == 3

            @staticmethod
            def assert_owner(owner):
                if owner is not None:
                    raise AssertionError("expected null clipboard owner")

        clipboard = Clipboard()
        with patch("pioneer.windows_credentials.time.sleep") as sleep:
            _open_clipboard(clipboard)
        self.assertEqual(clipboard.attempts, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_workspace_initialization_error_is_reported(self):
        error = StringIO()
        with patch("pioneer.desktop.Store", side_effect=OSError("disk unavailable")), \
             patch("pioneer.desktop.cli_main") as cli, redirect_stderr(error):
            self.assertEqual(main([]), 1)
            cli.assert_not_called()
        self.assertIn("disk unavailable", error.getvalue())

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI test")
    def test_windows_key_is_encrypted_and_can_be_reopened(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "key.dpapi"
            save_openai_key("fake-secret", path)
            self.assertNotIn(b"fake-secret", path.read_bytes())
            self.assertEqual(load_openai_key(path), "fake-secret")


if __name__ == "__main__":
    unittest.main()
