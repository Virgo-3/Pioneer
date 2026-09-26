"""Desired outcomes are compared with reported results on each branch."""

import tempfile
import unittest

from dao.objectives import (
    ObjectiveError,
    add_objective,
    compare_objective,
    objective_records,
    open_objectives,
    outcome_report,
    record_actual,
    validate_objective,
)
from dao.state import Store, StoreError


def numeric(**overrides):
    value = {
        "goal": "Ship the pilot",
        "metric": "Active users",
        "kind": "numeric",
        "desired": 100,
        "direction": "at_least",
        "unit": "users",
        "deadline": "2027-01-31",
        "action": "Invite the pilot group",
    }
    return {**value, **overrides}


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.store.init()

    def tearDown(self):
        self.temp.cleanup()

    def test_validation_and_numeric_bounds(self):
        self.assertEqual(validate_objective(numeric(goal="  Ship the pilot  "))["goal"], "Ship the pilot")
        self.assertTrue(issubclass(ObjectiveError, StoreError))
        for field, bad in (
            ("goal", " "), ("metric", ""), ("deadline", " "),
            ("unit", ""), ("kind", "other"), ("kind", []),
            ("direction", "more"), ("direction", []),
            ("desired", True), ("desired", float("nan")),
            ("desired", float("inf")), ("desired", 10 ** 1000),
            ("goal", "x" * 501),
        ):
            with self.subTest(field=field, bad=str(bad)[:12]), self.assertRaises(ObjectiveError):
                validate_objective(numeric(**{field: bad}))
        binary = numeric(kind="binary", desired=True, direction="exact", unit="")
        self.assertEqual(validate_objective(binary)["desired"], True)
        with self.assertRaises(ObjectiveError):
            validate_objective({**binary, "direction": "at_least"})
        with self.assertRaises(ObjectiveError):
            validate_objective({**binary, "desired": 1})

    def test_progress_is_provisional_and_final_only_on_matching_deadline(self):
        target = add_objective(self.store, numeric())
        self.assertEqual(outcome_report(self.store)["unobserved_count"], 1)
        progress = record_actual(self.store, target[:8], 90, "2027-01-15", note="Weekly check")
        comparison = compare_objective(objective_records(self.store)[0])
        self.assertEqual(comparison["status"], "progress")
        self.assertTrue(comparison["provisional"])
        self.assertIsNone(comparison["met"])
        self.assertEqual(comparison["gap"], -10)
        self.assertEqual(comparison["shortfall"], 10)
        self.assertEqual(comparison["observation_id"], progress)
        self.assertEqual([item["id"] for item in open_objectives(self.store)], [target])
        with self.assertRaisesRegex(ObjectiveError, "already recorded"):
            record_actual(self.store, target, 90, "2027-01-15", note="Weekly check")
        final = record_actual(self.store, target, 112, "2027-01-31")
        comparison = compare_objective(objective_records(self.store)[0])
        self.assertEqual(comparison["observation_id"], final)
        self.assertEqual(comparison["status"], "final")
        self.assertFalse(comparison["provisional"])
        self.assertEqual(comparison["gap"], 12)
        self.assertEqual(comparison["shortfall"], 0)
        self.assertTrue(comparison["met"])
        self.assertEqual(open_objectives(self.store), [])
        self.assertEqual(len(objective_records(self.store)[0]["observations"]), 2)

    def test_correction_preserves_history_and_recomputes_gap(self):
        target = add_objective(self.store, numeric(direction="at_most", desired=40,
                                                   metric="Hours spent", unit="hours"))
        first = record_actual(self.store, target, 35, "2027-01-31")
        self.assertEqual(outcome_report(self.store)["met_count"], 1)
        correction = record_actual(self.store, target, 55, "2027-01-31", note="Corrected timesheet")
        record = objective_records(self.store)[0]
        self.assertEqual([item["id"] for item in record["observations"]], [first, correction])
        self.assertEqual(self.store.read_object(first)["payload"]["actual"]["value"], 35)
        comparison = compare_objective(record)
        self.assertEqual((comparison["gap"], comparison["shortfall"], comparison["met"]), (15, 15, False))
        report = outcome_report(self.store)
        self.assertEqual((report["met_count"], report["missed_count"]), (0, 1))
        with self.assertRaises(ObjectiveError):
            record_actual(self.store, target, float("inf"), "2027-01-31")
        with self.assertRaises(ObjectiveError):
            record_actual(self.store, target, True, "2027-01-31")

    def test_later_off_checkpoint_report_does_not_erase_final(self):
        target = add_objective(self.store, numeric())
        first = record_actual(self.store, target, 105, "2027-01-31")
        progress = record_actual(self.store, target, 120, "2027-02-01")
        comparison = compare_objective(objective_records(self.store)[0])
        self.assertEqual(comparison["observation_id"], first)
        self.assertEqual(comparison["status"], "final")
        self.assertTrue(comparison["met"])
        self.assertEqual(open_objectives(self.store), [])
        with self.assertRaisesRegex(ObjectiveError, "already recorded"):
            record_actual(self.store, target, 105, "2027-01-31")
        correction = record_actual(self.store, target, 80, "2027-01-31", note="Revised count at checkpoint")
        record = objective_records(self.store)[0]
        self.assertEqual([item["id"] for item in record["observations"]], [first, progress, correction])
        comparison = compare_objective(record)
        self.assertEqual(comparison["observation_id"], correction)
        self.assertEqual((comparison["shortfall"], comparison["met"]), (20, False))

    def test_branch_isolation_and_ancestor_visibility(self):
        shared = add_objective(self.store, numeric())
        self.store.create_branch("alternate")
        self.store.switch("alternate")
        record_actual(self.store, shared, 105, "2027-01-31")
        alternate = add_objective(self.store, numeric(goal="Reduce cost", metric="Spend",
                                                      desired=50, direction="at_most", unit="USD"))
        self.assertEqual(outcome_report(self.store)["final_count"], 1)
        self.store.switch("main")
        self.assertEqual([item["id"] for item in open_objectives(self.store)], [shared])
        self.assertEqual(outcome_report(self.store)["final_count"], 0)
        with self.assertRaisesRegex(ObjectiveError, "visible"):
            record_actual(self.store, alternate, 30, "2027-01-31")
        self.assertEqual(outcome_report(self.store, "alternate")["count"], 2)

    def test_binary_and_mixed_units_stay_separate(self):
        users = add_objective(self.store, numeric())
        money = add_objective(self.store, numeric(goal="Stay within budget", metric="Spend",
                                                       desired=500, direction="at_most", unit="USD"))
        launched = add_objective(self.store, numeric(goal="Launch publicly", metric="Public release",
                                                          kind="binary", desired=True, direction="exact",
                                                          unit=""))
        record_actual(self.store, users, 80, "2027-01-31")
        record_actual(self.store, money, 450, "2027-01-31")
        record_actual(self.store, launched, False, "2027-01-31")
        report = outcome_report(self.store)
        self.assertEqual(report["count"], 3)
        self.assertEqual((report["met_count"], report["missed_count"]), (1, 2))
        self.assertEqual([item["unit"] for item in report["comparisons"]], ["users", "USD", ""])
        self.assertEqual([item["shortfall"] for item in report["comparisons"]], [20, 0, 1])
        self.assertEqual(report["comparisons"][2]["gap"], -1)
        self.assertNotIn("average_gap", report)
        self.assertEqual(outcome_report(self.store, goal="LAUNCH PUBLICLY")["count"], 1)

    def test_turn_payloads_and_condition_normalization(self):
        target = self.store.commit("turn", {
            "user": "I want to ship.", "assistant": "Let's target launch.",
            "objective": validate_objective(numeric(deadline="At  launch")),
        })
        self.store.commit("turn", {
            "user": "At launch, we had 102 users.", "assistant": "Target met.",
            "actual": {"objective_id": target, "value": 102, "as_of": "at launch", "note": "User report"},
        })
        record = objective_records(self.store)[0]
        self.assertEqual(compare_objective(record)["status"], "final")
        self.assertTrue(compare_objective(record)["met"])

    def test_same_turn_objective_and_actual_uses_self_reference(self):
        commit = self.store.commit("turn", {
            "user": "We wanted 100 paid users by Friday and got 70.",
            "assistant": "That is 30 users below the target.",
            "objective": validate_objective(numeric(metric="Paid users", deadline="Friday")),
            "actual": {"objective_id": "$self", "value": 70, "as_of": "Friday", "note": "User report"},
        })
        record = objective_records(self.store)[0]
        self.assertEqual(record["id"], commit)
        self.assertEqual(record["observations"][0]["objective_id"], commit)
        comparison = compare_objective(record)
        self.assertEqual((comparison["status"], comparison["gap"], comparison["shortfall"]),
                         ("final", -30, 30))
        self.assertFalse(comparison["met"])
        self.store.commit("note", {"actual": {"objective_id": "$self", "value": 80,
                                               "as_of": "Friday", "note": "Invalid"}})
        with self.assertRaisesRegex(ObjectiveError, "unavailable objective"):
            objective_records(self.store)


if __name__ == "__main__":
    unittest.main()
