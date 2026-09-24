import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pioneer.cli import _ask, _chat
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
        self.assertIn("OpenAI: A pilot buys you time to learn.", output)
        self.assertIn("Set OPENAI_API_KEY for conversation", output)
        self.assertNotIn("10 input", output)
        self.assertNotIn("test-model", output)
        self.assertEqual(errors, "")
        run_turn.assert_called_once_with(self.store, "Should we launch?", model=None)

    @patch("pioneer.cli.run_turn")
    def test_chat_displays_local_records_after_the_model_reply(self, run_turn):
        run_turn.return_value = TurnOutcome(
            "I would try the pilot first.", "a" * 64, "test-model", [], None, "main",
            notices=("Saved the desired outcome: 100 users by Friday.",
                     "Current gap from the target: 30 users."))
        output, errors = self.chat(["What should we do?", "/exit"])
        lines = output.splitlines()
        self.assertIn("OpenAI: I would try the pilot first.", lines)
        self.assertIn("Pioneer check: Saved the desired outcome: 100 users by Friday.", lines)
        self.assertIn("Pioneer check: Current gap from the target: 30 users.", lines)
        self.assertLess(lines.index("OpenAI: I would try the pilot first."),
                        lines.index("Pioneer check: Saved the desired outcome: 100 users by Friday."))
        self.assertEqual(errors, "")

    @patch("pioneer.cli.run_turn")
    def test_ask_displays_local_record_separately(self, run_turn):
        run_turn.return_value = TurnOutcome(
            "That plan could work if the pilot is cheap.", "a" * 64, "test-model", [],
            None, "main", notices=("Saved forecast for Friday.",))
        output = io.StringIO()
        with redirect_stdout(output):
            _ask(self.store, "Should we pilot?")
        self.assertEqual(output.getvalue().splitlines(),
                         ["OpenAI: That plan could work if the pilot is cheap.",
                          "Pioneer check: Saved forecast for Friday."])

    @patch("pioneer.cli.run_turn")
    def test_ask_attributes_source_and_challenge_separately(self, run_turn):
        source = "a" * 64
        run_turn.return_value = TurnOutcome(
            "I think we can launch now.", "b" * 64, "test-model", [], None, "main",
            history_challenge={"commit": source, "role": "assistant", "provider": "openai",
                               "quote": "We should wait for legal review.",
                               "challenge": "What changed your view?"})
        output = io.StringIO()
        with redirect_stdout(output):
            _ask(self.store, "Should we launch?")
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0], "OpenAI: I think we can launch now.")
        self.assertIn(f"Pioneer source {source[:12]}: OpenAI said", lines[1])
        self.assertEqual(lines[2], "OpenAI challenge: What changed your view?")

    def test_source_inspects_both_sides_on_current_branch(self):
        source = self.store.commit("turn", {"user": "Would a pilot preserve our options?",
                                            "assistant": "Yes, if the cost is small.",
                                            "provider": "openai"})
        output, errors = self.chat([f"/source {source[:12]}", "/exit"])
        self.assertIn(f"Source {source} on main:", output)
        self.assertIn("You: Would a pilot preserve our options?", output)
        self.assertIn("OpenAI: Yes, if the cost is small.", output)
        self.assertEqual(errors, "")

    def test_source_rejects_turn_outside_current_branch(self):
        self.store.create_branch("alternate")
        source = self.store.commit("turn", {"user": "Private main branch premise.",
                                            "assistant": "I hear you."})
        self.store.switch("alternate")
        output, errors = self.chat([f"/source {source[:12]}", "/exit"])
        self.assertNotIn("Private main branch premise.", output)
        self.assertIn("That source is not on the current branch", errors)

    def test_context_and_branch_orientation_follow_saved_history(self):
        self.store.commit("turn", {"user": "Should we launch?", "assistant": "Try a pilot.",
                                   "context": {"status": "active", "goal": "Launch timing",
                                               "options": ["launch", "pilot"],
                                               "known": ["Launch is hard to undo"],
                                               "uncertain": ["Pilot duration"],
                                               "provisional_view": "Pilot first",
                                               "next_questions": ["How long would a pilot take?"]}})
        output, errors = self.chat(["/context", "/branch alternate",
                                    "/branches", "/switch alternate", "/exit"])
        self.assertIn("Goal: Launch timing", output)
        self.assertIn("Current view: Pilot first", output)
        self.assertIn("Still uncertain:\n    - Pilot duration", output)
        self.assertIn("Type /switch alternate to continue there", output)
        self.assertIn("On alternate | Launch timing", output)
        self.assertEqual(errors, "")

    def test_context_shows_optional_jev_decision_checks_without_scores(self):
        self.store.commit("turn", {"user": "Should we launch?", "assistant": "Check the deadline.",
                                   "context": {"status": "active", "goal": "Launch timing",
                                               "options": ["launch", "pilot"], "known": [], "uncertain": [],
                                               "provisional_view": "Pilot first", "next_questions": []},
                                   "jev": {"goal": "Launch timing", "scores": {"time_sensitive": 0.91},
                                           "guidance": {"priorities": ["urgency", "reversibility"]}}})
        output, errors = self.chat(["/context", "/exit"])
        self.assertIn("Decision checks: timing, what can be undone (Jev guidance)", output)
        self.assertNotIn("0.91", output)
        self.assertEqual(errors, "")

    def test_context_hides_prior_jev_checks_after_latest_failure(self):
        context = {"status": "active", "goal": "Launch timing", "options": ["launch", "pilot"],
                   "known": [], "uncertain": [], "provisional_view": "Pilot first", "next_questions": []}
        self.store.commit("turn", {"user": "Should we launch?", "assistant": "Check the deadline.",
                                   "context": context,
                                   "jev": {"goal": "Launch timing", "guidance": {"priorities": ["urgency"]}}})
        self.store.commit("turn", {"user": "The deadline moved.", "assistant": "That changes the timing.",
                                   "context": context, "jev_error": "Jev unavailable"})
        output, errors = self.chat(["/context", "/exit"])
        self.assertNotIn("Decision checks:", output)
        self.assertEqual(errors, "")

    def test_command_errors_are_actionable_and_do_not_leave_chat(self):
        output, errors = self.chat(["/branch", "/switch two words", "/decide",
                                    "/status", "/triage launch", "/unknown", "/help", "/exit"])
        self.assertIn("Use /branch NAME", errors)
        self.assertIn("Use /switch NAME", errors)
        self.assertIn("Use /decide FILE", errors)
        self.assertEqual(errors.count("Unknown command"), 3)
        self.assertIn("/branches", output)
        self.assertIn("/source ID", output)
        self.assertNotIn("/context", output)
        self.assertNotIn("/status", output)
        self.assertNotIn("/triage", output)

    def test_advanced_help_keeps_exact_controls_available(self):
        output, errors = self.chat(["/help advanced", "/help other", "/exit"])
        self.assertIn("/context", output)
        self.assertIn("/target GOAL", output)
        self.assertIn("/forecast P", output)
        self.assertIn("/decide FILE", output)
        self.assertIn("/model MODEL", output)
        self.assertIn("Use /help or /help advanced.", errors)

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
        output, errors = self.chat(["/clear", "/reset main", "/exit"])
        self.assertIn("Started a fresh conversation on main", output)
        self.assertIn("On main | new conversation", output)
        self.assertEqual(self.store.resolve(), self.store.log()[-1][0])
        rescue = next(name for name in self.store.branches() if name.startswith("before-main-"))
        self.assertEqual(self.store.resolve(rescue), old_tip)
        self.assertEqual(errors, "")


if __name__ == "__main__":
    unittest.main()
