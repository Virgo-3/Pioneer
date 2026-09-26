"""A native OpenAI reply remains the turn even when Dao's state check fails."""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch

from dao.pipeline import run_turn
from dao.providers import ProviderError, TurnPlan, compose_turn
from dao.state import Store


EMPTY_CONTEXT = {
    "status": "none", "goal": "", "options": [], "known": [], "uncertain": [],
    "provisional_view": "", "next_questions": [],
}
NATIVE_REPLY = "I would pause for a week.\nWhat could you learn in that time?\n"


def _response(text: str, *, response_id: str, model: str,
              input_tokens: int, output_tokens: int) -> dict:
    return {
        "id": response_id,
        "model": model,
        "output": [{"content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _state() -> dict:
    return {
        "decision_requested": False, "case_json": None, "missing": [],
        "context": EMPTY_CONTEXT, "explore_alternative": False,
        "forecast": None, "resolution": None, "objective": None,
        "actual": None, "outcome_forecast": None,
    }


class NativeTurnProviderTests(unittest.TestCase):
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("dao.providers._post")
    def test_native_answer_is_authored_before_separate_state_extraction(self, post):
        post.side_effect = [
            _response(NATIVE_REPLY, response_id="answer_1", model="answer-model",
                      input_tokens=17, output_tokens=18),
            _response(json.dumps(_state()), response_id="state_1", model="state-model",
                      input_tokens=24, output_tokens=9),
        ]

        plan = compose_turn([{"role": "user", "content": "Should I wait?"}], model="answer-model")

        self.assertEqual(post.call_count, 2)
        answer_request = post.call_args_list[0].args[2]
        state_request = post.call_args_list[1].args[2]
        self.assertNotIn("text", answer_request)
        self.assertEqual(answer_request["input"][-1]["content"], "Should I wait?")
        self.assertEqual(state_request["text"]["format"]["type"], "json_schema")
        self.assertTrue(state_request["text"]["format"]["strict"])
        schema = state_request["text"]["format"]["schema"]
        self.assertNotIn("reply", schema["properties"])
        self.assertNotIn("reply", schema["required"])
        self.assertEqual(plan.reply, NATIVE_REPLY)
        self.assertEqual(plan.model, "answer-model")
        self.assertEqual(plan.response_id, "answer_1")
        self.assertEqual((plan.input_tokens, plan.output_tokens), (17, 18))
        self.assertEqual(plan.state_usage["model"], "state-model")
        self.assertEqual(plan.state_usage["response_id"], "state_1")
        self.assertEqual((plan.state_usage["input_tokens"],
                          plan.state_usage["output_tokens"]), (24, 9))
        self.assertIsNone(plan.state_error)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("dao.providers._post")
    def test_invalid_state_json_preserves_answer_and_extraction_usage(self, post):
        post.side_effect = [
            _response(NATIVE_REPLY, response_id="answer_2", model="answer-model",
                      input_tokens=17, output_tokens=18),
            _response("{broken", response_id="state_bad", model="state-model",
                      input_tokens=23, output_tokens=2),
        ]

        plan = compose_turn([{"role": "user", "content": "Should I wait?"}], model="answer-model")

        self.assertEqual(plan.reply, NATIVE_REPLY)
        self.assertEqual(plan.model, "answer-model")
        self.assertEqual(plan.response_id, "answer_2")
        self.assertTrue(plan.state_error)
        self.assertEqual(plan.state_usage["response_id"], "state_bad")
        self.assertEqual(plan.state_usage["input_tokens"], 23)
        self.assertFalse(plan.decision_requested)
        self.assertIsNone(plan.case_json)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("dao.providers._post")
    def test_state_request_failure_preserves_answer_without_fabricating_usage(self, post):
        post.side_effect = [
            _response(NATIVE_REPLY, response_id="answer_3", model="answer-model",
                      input_tokens=17, output_tokens=18),
            ProviderError("API returned HTTP 503"),
        ]

        plan = compose_turn([{"role": "user", "content": "Should I wait?"}], model="answer-model")

        self.assertEqual(plan.reply, NATIVE_REPLY)
        self.assertEqual((plan.input_tokens, plan.output_tokens), (17, 18))
        self.assertTrue(plan.state_error)
        self.assertIsNone(plan.state_usage)


class NativeTurnPipelineTests(unittest.TestCase):
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test", "TYPESAFE_API_KEY": ""})
    @patch("dao.pipeline.compose_turn")
    def test_failed_state_check_commits_exact_answer_and_accounts_for_both_calls(self, compose):
        state_usage = {
            "provider": "openai", "model": "state-model", "input_tokens": 23,
            "output_tokens": 2, "response_id": "state_bad",
        }
        compose.return_value = TurnPlan(
            NATIVE_REPLY, False, None, [], "answer-model", 17, 18, "answer_4",
            state_usage=state_usage, state_error="Invalid state JSON",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            store.init()

            outcome = run_turn(store, "Help me think about this.")

            payload = store.read_object(outcome.commit)["payload"]
            self.assertEqual(outcome.text, NATIVE_REPLY)
            self.assertEqual(payload["assistant"], NATIVE_REPLY)
            self.assertEqual(payload["model"], "answer-model")
            self.assertEqual(store.messages()[-1]["content"], NATIVE_REPLY)
            self.assertTrue(any("state" in notice.lower() or "check" in notice.lower()
                                for notice in outcome.notices))
            usage = store.usage()
            self.assertEqual(len(usage), 2)
            self.assertEqual({item["response_id"] for item in usage}, {"answer_4", "state_bad"})
            self.assertEqual(next(item for item in usage if item["response_id"] == "state_bad")
                             ["input_tokens"], 23)


if __name__ == "__main__":
    unittest.main()
