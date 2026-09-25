"""Exact and interactive controls for the linked outcome thread."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from pioneer.cli import _chat, main
from pioneer.objectives import add_objective, record_actual
from pioneer.outcomes import outcome_report
from pioneer.state import Store


class UnifiedOutcomeCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.store.init()

    def command(self, *args: str) -> tuple[int, str, str]:
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(["--repo", self.temp.name, *args])
        return status, output.getvalue(), errors.getvalue()

    def chat(self, *lines: str) -> tuple[str, str]:
        output, errors = io.StringIO(), io.StringIO()
        with patch("builtins.input", side_effect=[*lines, "/exit"]), \
                redirect_stdout(output), redirect_stderr(errors):
            _chat(self.store, None)
        return output.getvalue(), errors.getvalue()

    def target(self) -> str:
        return add_objective(self.store, {
            "goal": "Pilot", "metric": "weekly active users", "kind": "numeric",
            "desired": 100, "direction": "at_least", "unit": "users",
            "deadline": "2026-10-31", "action": "",
        })

    def test_predict_links_user_belief_to_one_target_and_outcome_scores_it(self) -> None:
        target_id = self.target()
        status, output, errors = self.command("predict", target_id[:12], "70%")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Saved your 70% forecast", output)
        forecast = self.store.read_object(self.store.resolve())["payload"]["outcome_forecast"]
        self.assertEqual(forecast["objective_id"], target_id)
        self.assertEqual(forecast["source"], "user")
        self.assertEqual(forecast["quote"], f"predict {target_id[:12]} 70%")
        record_actual(self.store, target_id, 92, "2026-10-31")
        status, output, errors = self.command("outcomes")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Desired: weekly active users at least 100 users by 2026-10-31", output)
        self.assertIn("Expected: 70% chance target is met (You,", output)
        self.assertIn("Observed: 92 users at 2026-10-31 (final)", output)
        self.assertIn("Gap (actual - desired): -8 users", output)
        self.assertIn("Target missed", output)
        self.assertIn("Forecast score: Brier 0.490", output)
        self.assertAlmostEqual(outcome_report(self.store)["mean_brier"], 0.49)

    def test_chat_predict_and_outcomes_keep_source_wording(self) -> None:
        target_id = self.target()
        output, errors = self.chat(f"/predict {target_id[:8]} | 65%", "/outcomes")
        self.assertEqual(errors, "")
        self.assertIn("Saved your 65% forecast", output)
        self.assertIn("Expected: 65% chance target is met (You,", output)
        self.assertIn("Observed: no result reported", output)
        forecast = self.store.read_object(self.store.resolve())["payload"]["outcome_forecast"]
        self.assertEqual(forecast["quote"], f"/predict {target_id[:8]} | 65%")

    def test_outcomes_show_latest_revision_with_attribution_without_double_counting(self) -> None:
        target_id = self.target()
        self.command("predict", target_id[:12], "70%")
        self.command("predict", target_id[:12], "80%")
        record_actual(self.store, target_id, 92, "2026-10-31")
        _, output, _ = self.command("outcomes")
        self.assertIn("Expected: 80% chance target is met", output)
        self.assertIn("2 recorded estimates", output)
        self.assertIn("Forecast score: Brier 0.640", output)
        self.assertIn("Scored forecasts: 1 | mean Brier: 0.640", output)

    def test_predict_rejects_target_on_other_branch_and_ambiguous_probability(self) -> None:
        self.store.create_branch("other")
        target_id = self.target()
        count = len(self.store.log())
        status, _, errors = self.command("predict", target_id[:12], "70")
        self.assertEqual(status, 1)
        self.assertIn("bare numbers above 1 are ambiguous", errors)
        self.assertEqual(len(self.store.log()), count)
        self.store.switch("other")
        status, _, errors = self.command("predict", target_id[:12], "70%")
        self.assertEqual(status, 1)
        self.assertIn("No target with that ID is visible on this branch", errors)
        _, output, _ = self.command("outcomes")
        self.assertIn("No outcome threads on other yet", output)

    def test_post_result_prediction_is_visible_but_unscored(self) -> None:
        target_id = self.target()
        record_actual(self.store, target_id, 100, "2026-10-31")
        status, output, errors = self.command("predict", target_id[:12], "90%")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("recorded but cannot be scored prospectively", output)
        _, output, _ = self.command("outcomes")
        self.assertIn("Forecast score: unscored", output)
        self.assertNotIn("mean Brier", output)

    def test_outcomes_keep_both_authors_and_earlier_eligible_scores_visible(self) -> None:
        target_id = self.target()
        self.command("predict", target_id[:12], "70%")
        self.store.commit("note", {"outcome_forecast": {
            "objective_id": target_id, "probability": 0.6, "source": "openai",
            "quote": "I estimate 60%.", "model": "test-model"}})
        record_actual(self.store, target_id, 92, "2026-10-31")
        self.command("predict", target_id[:12], "90%")
        _, output, _ = self.command("outcomes")
        self.assertIn("Forecast score: unscored", output)
        self.assertIn("Eligible your forecast: 70%, Brier 0.490", output)
        self.assertIn("Eligible OpenAI's forecast: 60%, Brier 0.360", output)
        self.assertIn("OpenAI's latest: 60%", output)
        self.assertIn("Your forecasts: 1 | mean Brier: 0.490", output)
        self.assertIn("OpenAI forecasts: 1 | mean Brier: 0.360", output)

    def test_observe_can_correct_a_checkpoint_date(self) -> None:
        target_id = self.target()
        first = record_actual(self.store, target_id, 92, "2026-10-31")
        status, output, errors = self.command(
            "observe", target_id[:12], "92", "--at", "2026-10-30",
            "--corrects", first[:12])
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Progress recorded", output)
        self.assertEqual(outcome_report(self.store)["final_count"], 0)


if __name__ == "__main__":
    unittest.main()
