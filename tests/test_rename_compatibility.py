"""Old workspaces and saved records remain accessible after the Dao rename."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dao.calibration import calibration_report, validate_forecast
from dao.desktop import workspace_path
from dao.state import Store, StoreError
from dao.windows_credentials import load_openai_key


class RenameCompatibilityTests(unittest.TestCase):
    def test_existing_pioneer_store_is_opened_in_place(self):
        with tempfile.TemporaryDirectory() as temp:
            old = Store(temp)
            old.init()
            turn = old.commit("turn", {"user": "Earlier words", "assistant": "Earlier reply"})
            (Path(temp) / ".dao").rename(Path(temp) / ".pioneer")
            reopened = Store(temp)
            self.assertEqual(reopened.data.name, ".pioneer")
            self.assertEqual(reopened.resolve(), turn)
            self.assertFalse((Path(temp) / ".dao").exists())
            with self.assertRaisesRegex(StoreError, "already exists"):
                reopened.init()

    def test_desktop_finds_old_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            old_root = Path(temp) / "Pioneer" / "workspace"
            Store(old_root).init()
            (old_root / ".dao").rename(old_root / ".pioneer")
            with patch.dict(os.environ, {"LOCALAPPDATA": temp}):
                self.assertEqual(workspace_path(), old_root)

    def test_saved_key_falls_back_to_old_path(self):
        with tempfile.TemporaryDirectory() as temp:
            old = Path(temp) / "Pioneer" / "openai-key.dpapi"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"legacy-example")
            with patch.dict(os.environ, {"APPDATA": temp}), \
                 patch("dao.windows_credentials._crypt", side_effect=lambda data, protect: data):
                self.assertEqual(load_openai_key(), "legacy-example")

    def test_old_forecasts_are_included_in_dao_report(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(temp)
            store.init()
            forecast = validate_forecast(
                {"event": "Pilot ships", "probability": 0.7, "deadline": "October"},
                source="pioneer")
            forecast_id = store.commit("note", {"forecast": forecast})
            store.commit("note", {"resolution": {"forecast_id": forecast_id,
                                                   "outcome": True, "note": ""}})
            self.assertEqual(calibration_report(store, source="dao")["count"], 1)
