import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pioneer.desktop import _setup_key, main, workspace_path
from pioneer.state import Store
from pioneer.windows_credentials import load_openai_key, save_openai_key


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
                cli.assert_called_with(["--repo", str(workspace), "chat"])

    def test_arguments_retain_regular_cli(self):
        with patch("pioneer.desktop.cli_main", return_value=0) as cli:
            self.assertEqual(main(["--help"]), 0)
            cli.assert_called_once_with(["--help"])

    def test_first_run_key_prompt_saves_only_when_chosen(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("pioneer.desktop.load_openai_key", return_value=None), \
                 patch("pioneer.desktop.getpass.getpass", return_value="  fake-key  "), \
                 patch("builtins.input", return_value="y"), \
                 patch("pioneer.desktop.save_openai_key") as save:
                _setup_key()
                self.assertEqual(os.environ["OPENAI_API_KEY"], "fake-key")
                save.assert_called_once_with("fake-key")

    def test_saved_key_skips_prompt(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("pioneer.desktop.load_openai_key", return_value="saved-key"), \
                 patch("pioneer.desktop.getpass.getpass") as prompt:
                _setup_key()
                self.assertEqual(os.environ["OPENAI_API_KEY"], "saved-key")
                prompt.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI test")
    def test_windows_key_is_encrypted_and_can_be_reopened(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "key.dpapi"
            save_openai_key("fake-secret", path)
            self.assertNotIn(b"fake-secret", path.read_bytes())
            self.assertEqual(load_openai_key(path), "fake-secret")


if __name__ == "__main__":
    unittest.main()
