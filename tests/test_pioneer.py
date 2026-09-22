import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pioneer.decision import DecisionError, analyze
from pioneer.providers import ProviderError, ask_openai, triage_jev
from pioneer.state import Store, StoreError


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.root = self.store.init()

    def tearDown(self):
        self.temp.cleanup()

    def test_branch_preserves_distinct_conversation_histories(self):
        first = self.store.commit("turn", {"user": "Question", "assistant": "First"})
        self.store.create_branch("alternate", first)
        main_tip = self.store.commit("turn", {"user": "Main follow-up", "assistant": "Main answer"})
        self.store.switch("alternate")
        other_tip = self.store.commit("turn", {"user": "Alternate follow-up", "assistant": "Alternate answer"})
        self.assertNotEqual(main_tip, other_tip)
        self.assertEqual([m["content"] for m in self.store.messages("main")],
                         ["Question", "First", "Main follow-up", "Main answer"])
        self.assertEqual([m["content"] for m in self.store.messages("alternate")],
                         ["Question", "First", "Alternate follow-up", "Alternate answer"])

    def test_rewind_keeps_old_objects_and_usage(self):
        first = self.store.commit("turn", {"user": "one", "assistant": "one"},
                                  usage={"provider": "openai", "model": "test", "input_tokens": 3, "output_tokens": 2})
        second = self.store.commit("turn", {"user": "two", "assistant": "two"},
                                   usage={"provider": "openai", "model": "test", "input_tokens": 4, "output_tokens": 2})
        self.store.rewind(first)
        self.assertEqual(self.store.resolve(), first)
        self.assertEqual(self.store.read_object(second)["payload"]["user"], "two")
        self.assertEqual(len(self.store.usage()), 2)
        self.assertEqual(self.store.verify()["usage_entries"], 2)

    def test_tamper_detection_and_concurrent_head_guard(self):
        with self.assertRaises(StoreError):
            self.store.commit("note", {"text": "stale"}, expected_head="0" * 64)
        object_id = self.store.commit("note", {"text": "test"})
        path = Path(self.temp.name) / ".pioneer" / "objects" / f"{object_id}.json"
        path.write_text('{"tampered":true}', encoding="utf-8")
        with self.assertRaises(StoreError):
            self.store.verify()


class DecisionTests(unittest.TestCase):
    def test_information_can_make_waiting_best(self):
        case = {
            "states": {"good": 0.5, "bad": 0.5},
            "actions": {
                "invest": {"outcomes": {"good": 10, "bad": -10}},
                "hold": {"outcomes": {"good": 0, "bad": 0}},
            },
            "wait": {"delay_cost": 1, "signals": {
                "positive": {"good": 0.9, "bad": 0.1},
                "negative": {"good": 0.1, "bad": 0.9},
            }},
        }
        result = analyze(case)
        self.assertEqual(result["recommendation"]["kind"], "wait")
        self.assertAlmostEqual(result["wait"]["value_of_information"], 4)
        self.assertAlmostEqual(result["wait"]["value"], 3)
        self.assertEqual(result["wait"]["signals"]["positive"]["best_action"], "invest")
        case["wait"]["delay_cost"] = 5
        self.assertEqual(analyze(case)["recommendation"]["kind"], "act")

    def test_undo_is_an_option_not_an_automatic_penalty(self):
        case = {"states": {"bad": 1}, "actions": {"try": {
            "cost": 1, "outcomes": {"bad": {"payoff": -10, "undo": -2, "undo_cost": 1}}}}}
        result = analyze(case)
        self.assertEqual(result["action_values"]["try"], -4)
        self.assertTrue(result["outcomes"]["try"]["bad"]["reversed"])

    def test_rejects_incoherent_probabilities(self):
        with self.assertRaises(DecisionError):
            analyze({"states": {"yes": 0.8}, "actions": {"a": {"outcomes": {"yes": 1}}}})


class ProviderTests(unittest.TestCase):
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_openai_uses_branch_messages_and_records_usage(self, post):
        post.return_value = {"id": "resp_1", "model": "test-model", "output": [
            {"content": [{"type": "output_text", "text": "Hello"}]}],
            "usage": {"input_tokens": 12, "output_tokens": 3}}
        result = ask_openai([{"role": "user", "content": "Hi"}], model="test-model")
        self.assertEqual(result.text, "Hello")
        self.assertEqual((result.input_tokens, result.output_tokens), (12, 3))
        self.assertEqual(post.call_args.args[2]["input"], [{"role": "user", "content": "Hi"}])
        self.assertFalse(post.call_args.args[2]["store"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_jev_schema_and_probability_validation(self, post):
        post.return_value = {"model": "jev-test", "answers": {
            name: {"type": "noul", "noul": 0.5}
            for name in ("time_sensitive", "hard_to_reverse", "missing_information")},
            "usage": {"input_tokens": 50, "output_tokens": 3}}
        result = triage_jev("Launch tomorrow")
        self.assertEqual(result["input_tokens"], 50)
        self.assertEqual(post.call_args.args[2]["questions"]["time_sensitive"]["type"], "noul")
        post.return_value["answers"]["time_sensitive"]["noul"] = 1.5
        with self.assertRaises(ProviderError):
            triage_jev("Launch tomorrow")


if __name__ == "__main__":
    unittest.main()
