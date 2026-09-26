"""Outcome-based calibration uses only recorded, branch-visible evidence."""

import tempfile
import unittest

from pioneer.calibration import (
    CalibrationError,
    add_forecast,
    calibration_context,
    calibration_report,
    forecast_records,
    open_forecasts,
    resolve_forecast,
    validate_forecast,
)
from pioneer.state import Store, StoreError


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.store.init()

    def tearDown(self):
        self.temp.cleanup()

    def test_forecast_validation_rejects_unscorable_or_unbounded_claims(self):
        value = {"event": "  The pilot ships  ", "probability": 0.7,
                 "deadline": "  2027-01-01  "}
        self.assertEqual(validate_forecast(value, source="pioneer", topic=" Launch ", model="m"), {
            "event": "The pilot ships", "probability": 0.7, "deadline": "2027-01-01",
            "topic": "Launch", "source": "pioneer", "model": "m",
        })
        for bad in (True, float("nan"), float("inf"), -0.01, 1.01, 10 ** 1000, "70%"):
            with self.subTest(probability=bad), self.assertRaises(CalibrationError):
                validate_forecast({**value, "probability": bad}, source="pioneer")
        for field, replacement in (("event", " "), ("deadline", ""),
                                   ("event", "x" * 501), ("deadline", "x" * 161)):
            with self.subTest(field=field, replacement=replacement[:12]), self.assertRaises(CalibrationError):
                validate_forecast({**value, field: replacement}, source="user")
        with self.assertRaises(CalibrationError):
            validate_forecast(value, source="unknown")
        with self.assertRaises(CalibrationError):
            validate_forecast(value, source="user", topic="x" * 121)
        self.assertTrue(issubclass(CalibrationError, StoreError))

    def test_branch_isolation_and_ancestor_visibility(self):
        shared = add_forecast(self.store, "Pilot ships", 0.7, "2027-01-01", topic="Launch")
        self.store.create_branch("alternate")
        self.store.switch("alternate")
        resolve_forecast(self.store, shared[:8], True)
        only_alternate = add_forecast(self.store, "Trial succeeds", 0.4, "2027-02-01")
        self.assertEqual([item["id"] for item in open_forecasts(self.store)], [only_alternate])
        self.store.switch("main")
        self.assertEqual([item["id"] for item in open_forecasts(self.store)], [shared])
        self.assertIsNone(forecast_records(self.store)[0]["resolution"])
        with self.assertRaisesRegex(CalibrationError, "visible"):
            resolve_forecast(self.store, only_alternate, True)
        self.assertEqual(calibration_report(self.store, source="all")["count"], 0)
        self.assertEqual(calibration_report(self.store, "alternate", source="all")["count"], 1)

    def test_scoring_and_five_probability_buckets(self):
        forecasts = [
            ("A", 0.0, False), ("B", 0.2, False), ("C", 0.5, True),
            ("D", 0.8, True), ("E", 1.0, True),
        ]
        for event, probability, outcome in forecasts:
            ref = add_forecast(self.store, event, probability, "2027-01-01",
                               topic="launch", source="pioneer", model="test-model")
            resolve_forecast(self.store, ref, outcome)
        user_ref = add_forecast(self.store, "F", 0.9, "2027-01-01", source="user")
        resolve_forecast(self.store, user_ref, False)
        report = calibration_report(self.store, topic="Launch")
        self.assertEqual(report["count"], 5)
        self.assertAlmostEqual(report["mean_predicted"], 0.5)
        self.assertAlmostEqual(report["observed_rate"], 0.6)
        self.assertAlmostEqual(report["brier_score"], (0 + 0.04 + 0.25 + 0.04 + 0) / 5)
        self.assertEqual(len(report["buckets"]), 5)
        self.assertEqual([bucket["count"] for bucket in report["buckets"]], [1, 1, 1, 0, 2])
        self.assertEqual(calibration_report(self.store, source="all")["count"], 6)
        self.assertEqual(calibration_report(self.store, source="user")["count"], 1)
        self.assertIsNone(calibration_report(self.store, topic="unknown")["brier_score"])

    def test_resolution_correction_preserves_history_and_recomputes_score(self):
        ref = add_forecast(self.store, "It rains", 0.8, "Tomorrow")
        first = resolve_forecast(self.store, ref[:8], True, note="Initial report")
        self.assertAlmostEqual(calibration_report(self.store, source="user")["brier_score"], 0.04)
        with self.assertRaisesRegex(CalibrationError, "already recorded"):
            resolve_forecast(self.store, ref, True)
        corrected = resolve_forecast(self.store, ref, False, note="Official result")
        self.assertNotEqual(first, corrected)
        self.assertEqual(self.store.read_object(first)["payload"]["resolution"]["outcome"], True)
        record = forecast_records(self.store)[0]
        self.assertEqual(record["resolution"]["id"], corrected)
        self.assertEqual(record["resolution"]["note"], "Official result")
        self.assertAlmostEqual(calibration_report(self.store, source="user")["brier_score"], 0.64)
        with self.assertRaises(CalibrationError):
            resolve_forecast(self.store, ref, 1)

    def test_turn_forecast_is_counted_and_low_sample_is_not_context(self):
        first = self.store.commit("turn", {
            "user": "Will it happen?", "assistant": "I estimate 60%.",
            "forecast": validate_forecast({"event": "Shipment arrives", "probability": 0.6,
                                           "deadline": "Friday"}, source="pioneer", topic="shipping"),
        })
        self.store.commit("turn", {
            "user": "It arrived", "assistant": "Recorded.",
            "resolution": {"forecast_id": first, "outcome": True, "note": "User reported delivery"},
        })
        self.assertEqual(forecast_records(self.store)[0]["resolution"]["outcome"], True)
        self.assertIsNone(calibration_context(self.store, "shipping"))
        for index in range(4):
            ref = add_forecast(self.store, f"Shipment {index} arrives", 0.6, "Friday",
                               topic="shipping", source="pioneer")
            resolve_forecast(self.store, ref, index % 2 == 0)
        context = calibration_context(self.store, "Shipping")
        self.assertEqual(context["count"], 5)
        self.assertIn("Limited evidence", context["caveat"])
        self.assertIsNone(calibration_context(self.store, "another topic"))
        with self.assertRaises(CalibrationError):
            calibration_context(self.store, "shipping", min_count=0)


if __name__ == "__main__":
    unittest.main()
