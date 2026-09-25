"""Provider contracts for target-linked forecasts in the unified outcome loop."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from pioneer.providers import TURN_SCHEMA, compose_turn


OBJECTIVE_ID = "a" * 64
EMPTY_CONTEXT = {"status": "none", "goal": "", "options": [], "known": [],
                 "uncertain": [], "provisional_view": "", "next_questions": []}


def _payload(outcome_forecast=None):
    return {"reply": "I think the target has a 70% chance.",
            "decision_requested": False, "case_json": None, "missing": [],
            "context": EMPTY_CONTEXT, "explore_alternative": False,
            "forecast": None, "resolution": None, "objective": None,
            "actual": None, "outcome_forecast": outcome_forecast}


def _response(payload):
    return {"id": "resp_outcome", "model": "test-model",
            "output": [{"content": [{"type": "output_text", "text": json.dumps(payload)}]}],
            "usage": {"input_tokens": 13, "output_tokens": 7}}


class ProviderOutcomeTests(unittest.TestCase):
    def test_strict_schema_requires_nullable_attributed_linked_forecast(self):
        schema = TURN_SCHEMA["properties"]["outcome_forecast"]
        self.assertIn("outcome_forecast", TURN_SCHEMA["required"])
        self.assertEqual(schema["type"], ["object", "null"])
        self.assertEqual(set(schema["required"]),
                         {"objective_id", "probability", "source", "quote"})
        self.assertEqual(schema["properties"]["source"]["enum"], ["user", "openai"])
        self.assertEqual(schema["properties"]["probability"]["minimum"], 0)
        self.assertEqual(schema["properties"]["probability"]["maximum"], 1)
        self.assertFalse(schema["additionalProperties"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_combined_open_outcomes_replaces_duplicate_objective_context(self, post):
        post.return_value = _response(_payload({"objective_id": OBJECTIVE_ID,
                                                "probability": .7, "source": "user",
                                                "quote": "70% chance"}))
        outcomes = [{"objective_id": OBJECTIVE_ID,
                     "predicate": "weekly users at least 100",
                     "checkpoint": "2026-10-31"}] * 6
        plan = compose_turn([{"role": "user", "content": "I think there is a 70% chance."}],
                            open_outcomes=outcomes,
                            open_objectives=[{"objective_id": OBJECTIVE_ID}])

        self.assertEqual(plan.outcome_forecast["source"], "user")
        request = post.call_args.args[2]
        self.assertTrue(request["text"]["format"]["strict"])
        context = json.loads(request["input"][-2]["content"].split(": ", 1)[1])
        self.assertEqual(len(context["open_outcomes"]), 5)
        self.assertNotIn("open_objectives", context)
        self.assertIn("latest user message", request["instructions"])
        self.assertIn("Do not attribute your estimate to the user", request["instructions"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_legacy_objective_context_and_no_forced_probability(self, post):
        post.return_value = _response(_payload())
        plan = compose_turn([{"role": "user", "content": "I want 100 users by October."}],
                            open_objectives=[{"objective_id": OBJECTIVE_ID}])
        self.assertIsNone(plan.outcome_forecast)
        context = json.loads(post.call_args.args[2]["input"][-2]["content"].split(": ", 1)[1])
        self.assertIn("open_objectives", context)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_openai_attributed_forecast_accepts_same_turn_objective(self, post):
        post.return_value = _response(_payload({"objective_id": "", "probability": .7,
                                                "source": "openai", "quote": "70% chance"}))
        plan = compose_turn([{"role": "user", "content": "What are my odds of reaching 100 users?"}])
        self.assertEqual(plan.outcome_forecast["objective_id"], "")
        self.assertEqual(plan.outcome_forecast["source"], "openai")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_invalid_linked_forecast_does_not_discard_authored_reply(self, post):
        bad_records = [
            {"objective_id": OBJECTIVE_ID, "probability": 1.2, "source": "user", "quote": "70%"},
            {"objective_id": OBJECTIVE_ID, "probability": True, "source": "user", "quote": "70%"},
            {"objective_id": OBJECTIVE_ID, "probability": .7, "source": "unknown", "quote": "70%"},
            {"objective_id": OBJECTIVE_ID, "probability": .7, "source": [], "quote": "70%"},
            {"objective_id": OBJECTIVE_ID, "probability": .7, "source": "user", "quote": " "},
            {"objective_id": "short", "probability": .7, "source": "user", "quote": "70%"},
        ]
        for record in bad_records:
            with self.subTest(record=record):
                post.return_value = _response(_payload(record))
                plan = compose_turn([{"role": "user", "content": "70% chance"}])
                self.assertEqual(plan.reply, "I think the target has a 70% chance.")
                self.assertIsNone(plan.outcome_forecast)
                self.assertEqual(plan.input_tokens, 13)


if __name__ == "__main__":
    unittest.main()
