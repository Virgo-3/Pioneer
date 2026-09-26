"""Edge cases where a plausible looking decision result would be misleading."""

import copy
import math
import unittest

from dao.decision import DecisionError, analyze


class DecisionEdgeCaseTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "states": {"good": 0.5, "bad": 0.5},
            "actions": {
                "act": {"outcomes": {"good": 10, "bad": -10}},
                "hold": {"outcomes": {"good": 0, "bad": 0}},
            },
        }

    def test_small_but_real_value_of_information_recommends_waiting(self):
        case = copy.deepcopy(self.case)
        case["actions"]["act"]["outcomes"] = {"good": 1e-9, "bad": -1e-9}
        case["wait"] = {"signals": {
            "good_news": {"good": 1, "bad": 0},
            "bad_news": {"good": 0, "bad": 1},
        }}

        result = analyze(case)

        self.assertEqual(result["recommendation"], {"kind": "wait", "action": None})
        self.assertAlmostEqual(result["wait"]["value_of_information"], 5e-10, places=20)

    def test_uninformative_signal_has_no_option_value(self):
        case = copy.deepcopy(self.case)
        case["wait"] = {"signals": {
            "heads": {"good": 0.5, "bad": 0.5},
            "tails": {"good": 0.5, "bad": 0.5},
        }}

        result = analyze(case)

        self.assertEqual(result["recommendation"]["kind"], "act")
        self.assertEqual(result["wait"]["value_of_information"], 0)

    def test_decimal_roundoff_does_not_make_uninformative_signal_valuable(self):
        case = {"states": {"a": 0.1, "b": 0.9}, "actions": {
            "act": {"outcomes": {"a": 0.1, "b": 0.2}},
            "hold": {"outcomes": {"a": 0, "b": 0}},
        }, "wait": {"signals": {
            "heads": {"a": 0.1, "b": 0.1},
            "tails": {"a": 0.9, "b": 0.9},
        }}}

        result = analyze(case)

        self.assertEqual(result["recommendation"]["kind"], "act")
        self.assertEqual(result["wait"]["value_of_information"], 0)

    def test_near_unit_likelihood_is_normalized_within_tolerance(self):
        case = {"states": {"only": 1}, "actions": {
            "act": {"outcomes": {"only": 1000}},
        }, "wait": {"signals": {"news": {"only": 0.9999995}}}}

        result = analyze(case)

        self.assertEqual(result["wait"]["value_of_information"], 0)
        self.assertEqual(result["wait"]["signals"]["news"]["probability"], 1)

    def test_rejects_silent_decision_field_typos(self):
        case = copy.deepcopy(self.case)
        case["waiting"] = {"delay_cost": 1}
        with self.assertRaisesRegex(DecisionError, "unknown field.*waiting"):
            analyze(case)

        case = copy.deepcopy(self.case)
        case["actions"]["act"]["outcomes"]["bad"] = {"payoff": -10, "undo-cost": 2}
        with self.assertRaisesRegex(DecisionError, "unknown field.*undo-cost"):
            analyze(case)

    def test_undo_cost_requires_an_undo_value(self):
        case = copy.deepcopy(self.case)
        case["actions"]["act"]["outcomes"]["bad"] = {"payoff": -10, "undo_cost": 2}
        with self.assertRaisesRegex(DecisionError, "undo_cost requires undo"):
            analyze(case)

    def test_rejects_nonstring_or_empty_signal_names(self):
        for name in (1, " "):
            with self.subTest(name=name):
                case = copy.deepcopy(self.case)
                case["wait"] = {"signals": {name: {"good": 1, "bad": 1}}}
                with self.assertRaisesRegex(DecisionError, "Signal names"):
                    analyze(case)

    def test_huge_integer_raises_decision_error(self):
        case = copy.deepcopy(self.case)
        case["actions"]["act"]["outcomes"]["good"] = 10 ** 1000
        with self.assertRaisesRegex(DecisionError, "finite number"):
            analyze(case)

    def test_finite_inputs_cannot_produce_nonfinite_action_value(self):
        case = {"states": {"only": 1}, "actions": {
            "act": {"cost": 1e308, "outcomes": {"only": -1e308}},
        }}
        with self.assertRaisesRegex(DecisionError, "numeric range"):
            analyze(case)

    def test_finite_inputs_cannot_produce_nonfinite_wait_value(self):
        case = {"states": {"only": 1}, "actions": {
            "act": {"outcomes": {"only": -1e308}},
        }, "wait": {"delay_cost": 1e308, "signals": {"news": {"only": 1}}}}
        with self.assertRaisesRegex(DecisionError, "numeric range"):
            analyze(case)

    def test_all_returned_decision_numbers_are_finite(self):
        case = copy.deepcopy(self.case)
        case["actions"]["act"]["outcomes"]["bad"] = {
            "payoff": -10, "undo": -2, "undo_cost": 1,
        }
        case["wait"] = {"delay_cost": 1, "signals": {
            "positive": {"good": 0.9, "bad": 0.1},
            "negative": {"good": 0.1, "bad": 0.9},
        }}

        def walk(value):
            if isinstance(value, dict):
                for nested in value.values():
                    walk(nested)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                self.assertTrue(math.isfinite(value))

        walk(analyze(case))


if __name__ == "__main__":
    unittest.main()
