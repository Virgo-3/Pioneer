"""End-to-end terminal commands for recording and reviewing forecast accuracy."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from pioneer.calibration import open_forecasts
from pioneer.cli import _chat, main
from pioneer.state import Store


class CalibrationCommandTests(unittest.TestCase):
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

    def test_cli_forecast_resolution_and_user_calibration(self):
        status, output, errors = self.cli("forecast", "70%", "Pilot finishes on time",
                                          "--by", "2027-01-01", "--topic", "launch")
        self.assertEqual((status, errors), (0, ""))
        forecast = open_forecasts(self.store)[0]
        self.assertIn(forecast["id"], output)
        self.assertIn("70% that Pilot finishes on time", output)
        self.assertIn("Resolution condition: 2027-01-01", output)

        status, output, errors = self.cli("forecasts")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn(forecast["id"], output)
        self.assertIn("launch", output)

        status, output, errors = self.cli("resolve", forecast["id"][:12], "yes")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn(f"Recorded your outcome for forecast {forecast['id'][:12]}", output)
        self.assertIn(f"Forecast ID: {forecast['id']}", output)
        self.assertEqual(open_forecasts(self.store), [])

        status, output, errors = self.cli("calibration", "--source", "user", "--topic", "launch")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Resolved: 1", output)
        self.assertIn("Average forecast: 70.0%", output)
        self.assertIn("Reported event rate: 100.0%", output)
        self.assertIn("Brier score: 0.090", output)
        self.assertIn("Pioneer has not verified them", output)

        status, output, errors = self.cli("calibration")
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("No resolved Pioneer forecasts", output)

    def test_bare_percent_number_is_rejected_without_recording_forecast(self):
        status, _, errors = self.cli("forecast", "70", "Pilot finishes on time", "--by", "2027-01-01")
        self.assertEqual(status, 1)
        self.assertIn("Use 0.7 or 70%, not 70", errors)
        self.assertEqual(open_forecasts(self.store), [])

    def test_interactive_forecast_commands_and_errors(self):
        output, errors = self.chat(["/forecast 55% | Pilot hits target | 2027-01-01 | launch",
                                    "/forecasts", "/resolve missing maybe", "/calibration user", "/exit"])
        forecast = open_forecasts(self.store)[0]
        self.assertIn(forecast["id"], output)
        self.assertIn("55%", output)
        self.assertIn("No resolved forecasts you entered", output)
        self.assertIn("Use /resolve ID yes|no", errors)

        output, errors = self.chat([f"/resolve {forecast['id'][:12]} no", "/forecasts",
                                    "/calibration user", "/exit"])
        self.assertEqual(errors, "")
        self.assertIn("did not happen", output)
        self.assertIn("No open forecasts", output)
        self.assertIn("Resolved: 1", output)
        self.assertIn("Brier score: 0.303", output)


if __name__ == "__main__":
    unittest.main()
