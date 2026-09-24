"""End-to-end terminal flows for desired-versus-actual outcome calibration."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from pioneer.cli import _chat, main
from pioneer.objectives import objective_records, outcome_report
from pioneer.state import Store


class OutcomeCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.store.init()

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, *args):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(["--repo", self.temp.name, *args])
        return status, output.getvalue(), errors.getvalue()

    def chat(self, lines):
        output, errors = io.StringIO(), io.StringIO()
        with patch("builtins.input", side_effect=lines), redirect_stdout(output), redirect_stderr(errors):
            _chat(self.store, None)
        return output.getvalue(), errors.getvalue()

    def test_numeric_target_progress_final_and_correction(self):
        status, output, errors = self.cli(
            "target", "Grow the pilot", "weekly active users", "--desired", "100",
            "--by", "2027-01-31", "--unit", "users", "--action", "run a pilot")
        self.assertEqual((status, errors), (0, ""))
        target_id = objective_records(self.store)[0]["id"]
        self.assertIn(target_id, output)
        self.assertIn("weekly active users at least 100 users", output)

        status, output, errors = self.cli("targets")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn(target_id, output)
        self.assertIn("run a pilot", output)

        status, output, errors = self.cli("observe", target_id[:12], "80", "--at", "2027-01-15")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Progress recorded", output)
        self.assertEqual(outcome_report(self.store)["progress_count"], 1)

        status, output, errors = self.cli("observe", target_id[:12], "90", "--at", "2027-01-31")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Target missed", output)
        self.assertIn("-10", output)

        status, output, errors = self.cli("calibration", "--goal", "Grow the pilot")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Outcome calibration", output)
        self.assertIn("final: 1 | met: 0 | missed: 1", output)
        self.assertIn("Gap (actual - desired): -10 users", output)
        self.assertNotIn("Brier score", output)

        status, output, errors = self.cli("observe", target_id[:12], "120", "--at", "2027-01-31",
                                          "--note", "corrected report")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Target met", output)
        self.assertEqual(len(objective_records(self.store)[0]["observations"]), 3)
        self.assertEqual(outcome_report(self.store)["met_count"], 1)

        status, output, errors = self.cli("observe", target_id[:12], "130", "--at", "2027-02-05")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("final comparison still uses the result reported at 2027-01-31", output)
        self.assertEqual(outcome_report(self.store)["met_count"], 1)

    def test_interactive_binary_target_and_forecast_accuracy_is_separate(self):
        output, errors = self.chat(["/target Launch well | pilot accepted | yes | 2027-02-01",
                                    "/targets", "/calibration", "/forecast-accuracy",
                                    "/observe bad", "/exit"])
        self.assertEqual(len(objective_records(self.store)), 1)
        target_id = objective_records(self.store)[0]["id"]
        self.assertIn(target_id, output)
        self.assertIn("No resolved Pioneer forecasts", output)
        self.assertIn("Use /observe ID | ACTUAL | WHEN", errors)

        output, errors = self.chat([f"/observe {target_id[:12]} | no | 2027-02-01",
                                    "/calibration Launch well", "/exit"])
        self.assertEqual(errors, "")
        self.assertIn("Target missed", output)
        self.assertIn("final: 1 | met: 0 | missed: 1", output)

    def test_numeric_target_requires_explicit_unit_and_finite_value(self):
        status, _, errors = self.cli("target", "Grow", "users", "--desired", "100",
                                     "--by", "2027-01-31")
        self.assertEqual(status, 1)
        self.assertIn("Numeric targets need a unit", errors)
        status, _, errors = self.cli("target", "Grow", "users", "--desired", "NaN",
                                     "--by", "2027-01-31", "--unit", "users")
        self.assertEqual(status, 1)
        self.assertIn("finite number", errors)
        self.assertEqual(objective_records(self.store), [])


if __name__ == "__main__":
    unittest.main()
