"""Provider contracts for retrospective, source-labeled history challenges."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from pioneer.providers import ProviderError, compose_turn, review_history


def _response(payload: dict[str, object], *, input_tokens: int = 17) -> dict[str, object]:
    return {
        "id": "resp_review",
        "model": "test-model",
        "output": [{"content": [{"type": "output_text", "text": json.dumps(payload)}]}],
        "usage": {"input_tokens": input_tokens, "output_tokens": 9},
    }


class ProviderReviewTests(unittest.TestCase):
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_history_review_uses_finished_reply_and_exact_source_data(self, post):
        evidence = [{"commit": "a" * 64, "role": "assistant", "quote": "The pilot has no cost."}]
        conflict = {"commit": "a" * 64, "role": "assistant", "quote": "pilot has no cost",
                    "challenge": "You previously said the pilot had no cost. What changed?"}
        post.return_value = _response({"conflict": conflict})

        review = review_history("The pilot costs $500.", evidence,
                                user_text="Would a pilot be worth it?", model="test-model")

        self.assertEqual(review.conflict, conflict)
        self.assertEqual((review.input_tokens, review.output_tokens, review.response_id),
                         (17, 9, "resp_review"))
        request = post.call_args.args[2]
        self.assertEqual(request["text"]["format"]["name"], "pioneer_history_review")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertFalse(request["store"])
        review_input = json.loads(request["input"][0]["content"])
        self.assertEqual(review_input["finished_assistant_reply"], "The pilot costs $500.")
        self.assertEqual(review_input["retrieved_branch_statements"], evidence)
        self.assertEqual(review_input["latest_user_message"], "Would a pilot be worth it?")
        self.assertIn("does not establish what is true", request["instructions"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_no_tension_returns_no_conflict(self, post):
        post.return_value = _response({"conflict": None})
        review = review_history("I still prefer a pilot.", [], user_text="What now?")
        self.assertIsNone(review.conflict)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_invalid_review_retains_usage_for_accounting(self, post):
        post.return_value = _response({"conflict": {"commit": "a" * 64, "role": "system",
                                                    "quote": "quoted", "challenge": "Why?"}},
                                      input_tokens=31)
        with self.assertRaises(ProviderError) as caught:
            review_history("Final reply", [], user_text="Question")
        self.assertEqual(caught.exception.usage["input_tokens"], 31)
        self.assertEqual(caught.exception.usage["response_id"], "resp_review")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_turn_schema_only_authors_reply_and_state(self, post):
        context = {"status": "none", "goal": "", "options": [], "known": [], "uncertain": [],
                   "provisional_view": "", "next_questions": []}
        post.return_value = _response({
            "reply": "The pilot costs $500, and I think it may be worth that.",
            "decision_requested": True, "case_json": None, "missing": [], "context": context,
            "explore_alternative": False, "forecast": None, "resolution": None,
            "objective": None, "actual": None,
        })
        evidence = [{"commit": "b" * 64, "role": "assistant", "quote": "The pilot has no cost."}]

        plan = compose_turn([{"role": "user", "content": "Would a pilot be worth it?"}],
                            history_evidence=evidence)

        self.assertEqual(plan.reply, "The pilot costs $500, and I think it may be worth that.")
        self.assertIsNone(plan.history_conflict)
        request = post.call_args.args[2]
        schema = request["text"]["format"]["schema"]
        self.assertNotIn("history_conflict", schema["properties"])
        self.assertNotIn("history_conflict", schema["required"])
        self.assertEqual(json.loads(request["input"][-2]["content"].split(": ", 1)[1])
                         ["retrieved_branch_statements"], evidence)
        self.assertNotIn("keep reply free", request["instructions"])
        self.assertNotIn("do not claim", request["instructions"])


if __name__ == "__main__":
    unittest.main()
