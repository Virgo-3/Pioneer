"""Unified, branch-local outcome threads and calibrated predictions."""

import math
import tempfile
import unittest

from dao.calibration import add_forecast
from dao.objectives import add_objective, record_actual, validate_objective
from dao.outcomes import (
    OutcomeError,
    outcome_report,
    outcome_threads,
    validate_outcome_forecast,
)
from dao.state import Store


def target(**overrides):
    return {
        "goal": "Reach 100 users",
        "metric": "Weekly active users",
        "kind": "numeric",
        "desired": 100,
        "direction": "at_least",
        "unit": "users",
        "deadline": "2027-01-31",
        "action": "Invite the pilot group",
        **overrides,
    }


class OutcomeThreadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.store.init()

    def tearDown(self):
        self.temp.cleanup()

    def test_same_turn_self_reference_and_forecast_provenance(self):
        claim = "There is a 70% chance we reach 100 weekly users."
        commit = self.store.commit("turn", {
            "user": f"I want 100 users by January 31. {claim}",
            "assistant": "Let's watch the count.",
            "objective": validate_objective(target()),
            "outcome_forecast": {
                "objective_id": "$self", "probability": 0.7,
                "source": "user", "quote": claim,
            },
        })
        thread = outcome_threads(self.store)[0]
        self.assertEqual(thread["id"], commit)
        self.assertEqual(thread["objective"]["desired"], 100.0)
        forecast = thread["forecasts"][0]
        self.assertEqual((forecast["id"], forecast["objective_id"]), (commit, commit))
        self.assertEqual((forecast["source"], forecast["quote"]), ("user", claim))
        self.assertIsNone(forecast["outcome"])
        self.assertIsNone(forecast["brier"])
        record_actual(self.store, commit, 108, "2027-01-31")
        forecast = outcome_threads(self.store)[0]["forecasts"][0]
        self.assertTrue(forecast["outcome"])
        self.assertAlmostEqual(forecast["brier"], 0.09)
        report = outcome_report(self.store)
        self.assertEqual((report["count"], report["final_count"], report["scored_count"]),
                         (1, 1, 1))
        self.assertAlmostEqual(report["mean_brier"], 0.09)

    def test_progress_revisions_correction_and_post_result_forecast(self):
        objective_id = add_objective(self.store, target())
        first = self.store.commit("note", {"outcome_forecast": {
            "objective_id": objective_id, "probability": 0.7,
            "source": "user", "quote": "I think the chance is 70%.",
        }})
        latest_user = self.store.commit("note", {"outcome_forecast": {
            "objective_id": objective_id, "probability": 0.8,
            "source": "user", "quote": "I now think the chance is 80%.",
        }})
        openai = self.store.commit("note", {"outcome_forecast": {
            "objective_id": objective_id, "probability": 0.6,
            "source": "openai", "quote": "I estimate a 60% chance.", "model": "test-model",
        }})
        before_result = self.store.resolve()
        record_actual(self.store, objective_id, 90, "2027-01-15")
        report = outcome_report(self.store)
        self.assertEqual(report["progress_count"], 1)
        self.assertEqual(report["scored_count"], 0)
        self.assertIsNone(report["mean_brier"])
        self.assertTrue(all(item["outcome"] is None and item["brier"] is None
                            for item in report["threads"][0]["forecasts"]))

        record_actual(self.store, objective_id, 110, "2027-01-31")
        report = outcome_report(self.store)
        forecasts = report["threads"][0]["forecasts"]
        self.assertEqual([item["id"] for item in forecasts], [first, latest_user, openai])
        self.assertTrue(all(item["outcome"] is True for item in forecasts))
        self.assertAlmostEqual(forecasts[0]["brier"], 0.09)
        self.assertAlmostEqual(forecasts[1]["brier"], 0.04)
        self.assertAlmostEqual(forecasts[2]["brier"], 0.16)
        # The first user's revision remains visible, but does not add a second
        # calibration sample for the same source and objective.
        self.assertEqual(report["scored_count"], 2)
        self.assertAlmostEqual(report["mean_brier"], 0.10)
        self.assertAlmostEqual(report["by_source"]["user"]["mean_brier"], 0.04)
        self.assertAlmostEqual(report["by_source"]["openai"]["mean_brier"], 0.16)
        self.assertEqual(outcome_report(self.store, before_result)["scored_count"], 0)

        late = self.store.commit("note", {"outcome_forecast": {
            "objective_id": objective_id, "probability": 0.9,
            "source": "user", "quote": "Now I say 90%.",
        }})
        record_actual(self.store, objective_id, 80, "2027-01-31", note="Corrected count")
        report = outcome_report(self.store)
        forecasts = report["threads"][0]["forecasts"]
        self.assertEqual(forecasts[-1]["id"], late)
        self.assertIsNone(forecasts[-1]["outcome"])
        self.assertIsNone(forecasts[-1]["brier"])
        self.assertFalse(forecasts[0]["outcome"])
        self.assertAlmostEqual(forecasts[0]["brier"], 0.49)
        self.assertAlmostEqual(forecasts[1]["brier"], 0.64)
        self.assertAlmostEqual(forecasts[2]["brier"], 0.36)
        self.assertEqual((report["met_count"], report["missed_count"], report["scored_count"]),
                         (0, 1, 2))
        self.assertAlmostEqual(report["mean_brier"], 0.50)

    def test_same_commit_final_result_cannot_be_predicted_after_the_fact(self):
        claim = "I give it an 80% chance."
        commit = self.store.commit("turn", {
            "user": f"We aimed for 100 users and had 110. {claim}",
            "assistant": "The target was met.",
            "objective": validate_objective(target()),
            "actual": {"objective_id": "$self", "value": 110,
                       "as_of": "2027-01-31", "note": "User report"},
            "outcome_forecast": {"objective_id": "$self", "probability": 0.8,
                                 "source": "user", "quote": claim},
        })
        thread = outcome_threads(self.store)[0]
        self.assertEqual(thread["id"], commit)
        self.assertEqual(thread["comparison"]["status"], "final")
        self.assertTrue(thread["comparison"]["met"])
        self.assertIsNone(thread["forecasts"][0]["outcome"])
        self.assertIsNone(thread["forecasts"][0]["brier"])
        self.assertEqual(outcome_report(self.store)["scored_count"], 0)
        record_actual(self.store, commit, 95, "2027-01-31", note="Correction")
        self.assertIsNone(outcome_threads(self.store)[0]["forecasts"][0]["brier"])

    def test_branch_isolation_and_legacy_target_only(self):
        shared = add_objective(self.store, target())
        # Old calibration forecasts remain independent; matching words do not
        # silently attach them to an outcome thread.
        add_forecast(self.store, "Reach 100 users", 0.7, "2027-01-31", source="user")
        self.assertEqual(outcome_report(self.store)["threads"][0]["forecasts"], [])
        self.store.create_branch("alternate")
        self.store.switch("alternate")
        forecast_id = self.store.commit("note", {"outcome_forecast": {
            "objective_id": shared, "probability": 0.6,
            "source": "user", "quote": "60% chance",
        }})
        record_actual(self.store, shared, 105, "2027-01-31")
        alternate = outcome_report(self.store)
        self.assertEqual(alternate["threads"][0]["forecasts"][0]["id"], forecast_id)
        self.assertEqual(alternate["scored_count"], 1)
        self.store.switch("main")
        main = outcome_report(self.store)
        self.assertEqual(main["count"], 1)
        self.assertEqual(main["unobserved_count"], 1)
        self.assertEqual(main["scored_count"], 0)
        self.assertEqual(main["threads"][0]["forecasts"], [])
        self.assertEqual(outcome_report(self.store, "alternate")["scored_count"], 1)

    def test_validation_and_unavailable_source(self):
        objective_id = add_objective(self.store, target())
        base = {"objective_id": objective_id, "probability": 0.7,
                "source": "user", "quote": "I say 70%."}
        self.assertEqual(validate_outcome_forecast(base, source_text="I say 70%."),
                         {**base, "model": None})
        for override in (
            {"objective_id": "abcd"}, {"objective_id": "$bad"},
            {"probability": -0.1}, {"probability": 1.1},
            {"probability": True}, {"probability": math.nan},
            {"probability": 10 ** 1000}, {"source": "dao"},
            {"source": []}, {"quote": " "}, {"model": " "},
        ):
            with self.subTest(override=override), self.assertRaises(OutcomeError):
                validate_outcome_forecast({**base, **override})
        with self.assertRaisesRegex(OutcomeError, "exactly match"):
            validate_outcome_forecast(base, source_text="No probability here.")
        self.store.commit("note", {"outcome_forecast": {
            **base, "objective_id": "0" * 64,
        }})
        with self.assertRaisesRegex(OutcomeError, "unavailable objective"):
            outcome_threads(self.store)

    def test_corrected_checkpoint_timing_retracts_score_without_hindsight(self):
        objective_id = add_objective(self.store, target(deadline="by Friday"))
        self.store.commit("note", {"outcome_forecast": {
            "objective_id": objective_id, "probability": 0.7,
            "source": "user", "quote": "70% chance",
        }})
        first = record_actual(self.store, objective_id, 80, "Friday")
        self.assertEqual(outcome_report(self.store)["final_count"], 1)
        self.store.commit("note", {"outcome_forecast": {
            "objective_id": objective_id, "probability": 0.9,
            "source": "openai", "quote": "90% chance", "model": "test",
        }})
        record_actual(self.store, objective_id, 80, "Thursday", replaces=first)
        retracted = outcome_report(self.store)
        self.assertEqual((retracted["final_count"], retracted["progress_count"],
                          retracted["scored_count"]), (0, 1, 0))
        record_actual(self.store, objective_id, 105, "Friday")
        final = outcome_report(self.store)
        self.assertEqual(final["final_count"], 1)
        self.assertEqual(final["scored_count"], 1)
        self.assertAlmostEqual(final["threads"][0]["forecasts"][0]["brier"], 0.09)
        self.assertIsNone(final["threads"][0]["forecasts"][1]["brier"])

    def test_timing_correction_does_not_expose_an_older_final_result(self):
        objective_id = add_objective(self.store, target())
        first = record_actual(self.store, objective_id, 80, "2027-01-31")
        second = record_actual(self.store, objective_id, 90, "2027-01-31")
        observations = outcome_threads(self.store)[0]["comparison"]
        self.assertEqual(observations["observation_id"], second)
        self.assertNotEqual(first, second)
        record_actual(self.store, objective_id, 90, "2027-01-30", replaces=second)
        comparison = outcome_threads(self.store)[0]["comparison"]
        self.assertEqual(comparison["status"], "progress")
        self.assertEqual(comparison["actual"], 90)


if __name__ == "__main__":
    unittest.main()
