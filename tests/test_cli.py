import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pioneer.cli import _chat
from pioneer.pipeline import TurnOutcome
from pioneer.state import Store


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.store.init()

    def tearDown(self):
        self.temp.cleanup()

    def chat(self, lines):
        output, errors = io.StringIO(), io.StringIO()
        with patch("builtins.input", side_effect=lines), redirect_stdout(output), redirect_stderr(errors):
            _chat(self.store, None)
        return output.getvalue(), errors.getvalue()

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""})
    @patch("pioneer.cli.run_turn")
    def test_chat_reply_stays_conversational_and_usage_is_on_demand(self, run_turn):
        run_turn.return_value = TurnOutcome("A pilot buys you time to learn.", "a" * 64, "test-model",
                                            [{"provider": "openai", "input_tokens": 10, "output_tokens": 6}],
                                            None, "main")
        output, errors = self.chat(["Should we launch?", "/exit"])
        self.assertIn("Pioneer: A pilot buys you time to learn.", output)
        self.assertIn("Set OPENAI_API_KEY for conversation", output)
        self.assertNotIn("10 input", output)
        self.assertNotIn("test-model", output)
        self.assertEqual(errors, "")
        run_turn.assert_called_once_with(self.store, "Should we launch?", model=None)

    def test_context_and_branch_orientation_follow_saved_history(self):
        self.store.commit("turn", {"user": "Should we launch?", "assistant": "Try a pilot.",
                                   "context": {"status": "active", "goal": "Launch timing",
                                               "options": ["launch", "pilot"],
                                               "known": ["Launch is hard to undo"],
                                               "uncertain": ["Pilot duration"],
                                               "provisional_view": "Pilot first",
                                               "next_questions": ["How long would a pilot take?"]}})
        output, errors = self.chat(["/status", "/context", "/branch alternate",
                                    "/branches", "/switch alternate", "/exit"])
        self.assertIn("Topic: Launch timing", output)
        self.assertIn("Current view: Pilot first", output)
        self.assertIn("Still uncertain:\n    - Pilot duration", output)
        self.assertIn("Type /switch alternate to continue there", output)
        self.assertIn("On alternate | Launch timing", output)
        self.assertEqual(errors, "")

    def test_command_errors_are_actionable_and_do_not_leave_chat(self):
        output, errors = self.chat(["/branch", "/switch two words", "/decide",
                                    "/triage", "/unknown", "/help", "/exit"])
        self.assertIn("Use /branch NAME", errors)
        self.assertIn("Use /switch NAME", errors)
        self.assertIn("Use /decide FILE", errors)
        self.assertIn("Use /triage ACTION", errors)
        self.assertIn("Unknown command", errors)
        self.assertIn("/context", output)

    def test_quoted_decision_path_and_analysis_command(self):
        case = {"title": "Test choice", "states": {"yes": 1},
                "actions": {"do": {"outcomes": {"yes": 2}}}}
        file = Path(self.temp.name) / "case with spaces.json"
        file.write_text(json.dumps(case), encoding="utf-8")
        output, errors = self.chat([f'/decide "{file}"', "/analysis", "/exit"])
        self.assertIn("Recommendation: act now: do", output)
        self.assertIn('"case":', output)
        self.assertIn('"analysis":', output)
        self.assertEqual(errors, "")

    def test_clear_and_reset_main_restart_chat_without_losing_history(self):
        old_tip = self.store.commit("turn", {"user": "old goal", "assistant": "old reply"})
        output, errors = self.chat(["/clear", "/reset main", "/status", "/exit"])
        self.assertIn("Started a fresh conversation on main", output)
        self.assertIn("On main | new conversation", output)
        self.assertEqual(self.store.resolve(), self.store.log()[-1][0])
        rescue = next(name for name in self.store.branches() if name.startswith("before-main-"))
        self.assertEqual(self.store.resolve(rescue), old_tip)
        self.assertEqual(errors, "")


if __name__ == "__main__":
    unittest.main()
