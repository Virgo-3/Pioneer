import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pioneer.decision import DecisionError, analyze
from pioneer.pipeline import run_turn
from pioneer.providers import ProviderError, TurnPlan, ask_openai, compose_turn, triage_jev
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
            "wait": {"delay_cost": 1, "information_cost": 0, "signals": {
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
            for name in ("decision_request", "time_sensitive", "hard_to_reverse", "missing_information")},
            "usage": {"input_tokens": 50, "output_tokens": 3}}
        result = triage_jev("Launch tomorrow")
        self.assertEqual(result["input_tokens"], 50)
        self.assertEqual(post.call_args.args[2]["questions"]["time_sensitive"]["type"], "noul")
        post.return_value["answers"]["time_sensitive"]["noul"] = 1.5
        with self.assertRaises(ProviderError):
            triage_jev("Launch tomorrow")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_structured_turn_contract(self, post):
        post.return_value = {"id": "resp_2", "model": "test-model", "output": [
            {"content": [{"type": "output_text", "text": json.dumps({
                "reply": "Tell me the states.", "decision_requested": True,
                "case_json": None, "missing": ["state probabilities"]})}]}],
            "usage": {"input_tokens": 20, "output_tokens": 5}}
        plan = compose_turn([{"role": "user", "content": "Should I launch?"}], model="test-model")
        self.assertTrue(plan.decision_requested)
        self.assertEqual(plan.missing, ["state probabilities"])
        self.assertEqual(post.call_args.args[2]["text"]["format"]["type"], "json_schema")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.store.init()

    def tearDown(self):
        self.temp.cleanup()

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.pipeline.compose_turn")
    @patch("pioneer.pipeline.triage_jev")
    def test_one_turn_integrates_triage_analysis_and_usage(self, triage, compose):
        case = {"states": {"good": 0.5, "bad": 0.5}, "actions": {
            "invest": {"outcomes": {"good": 10, "bad": -10}},
            "hold": {"outcomes": {"good": 0, "bad": 0}}},
            "wait": {"delay_cost": 1, "information_cost": 0, "signals": {
                "positive": {"good": 0.9, "bad": 0.1},
                "negative": {"good": 0.1, "bad": 0.9}}}}
        text = "Good 50%, bad 50%. Invest pays 10 or -10; hold pays 0. Wait delay costs 1 and information costs 0. A positive signal is 90% likely in good and 10% in bad; a negative signal is 10% in good and 90% in bad."
        triage.return_value = {"model": "jev-test", "scores": {"decision_request": 0.99,
            "time_sensitive": 0.2, "hard_to_reverse": 0.8, "missing_information": 0.9},
            "input_tokens": 30, "output_tokens": 4}
        compose.return_value = TurnPlan("Here is the comparison.", True, json.dumps(case), [],
                                       "openai-test", 50, 20, "resp_3")
        outcome = run_turn(self.store, text)
        self.assertEqual(outcome.decision["recommendation"]["kind"], "wait")
        self.assertIn("Recommendation: wait", outcome.text)
        self.assertEqual(len(self.store.log()), 2)
        self.assertEqual(len(self.store.usage()), 2)
        self.assertEqual({record["commit"] for record in self.store.usage()}, {outcome.commit})
        self.assertEqual(self.store.read_object(outcome.commit)["payload"]["triage"]["model"], "jev-test")
        self.assertIsNotNone(compose.call_args.kwargs["triage"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_unprovided_number_blocks_calculation(self, compose):
        case = {"states": {"good": 0.9, "bad": 0.1}, "actions": {"act": {
            "outcomes": {"good": 10, "bad": -10}}}}
        compose.return_value = TurnPlan("I used your figures.", True, json.dumps(case), [],
                                       "openai-test", 10, 5, "resp_4")
        outcome = run_turn(self.store, "Act pays 10 if good and -10 if bad. What should I do?")
        self.assertIsNone(outcome.decision)
        self.assertIn("numbers you did not provide", outcome.text)
        self.assertNotIn("decision", self.store.read_object(outcome.commit)["payload"])
        self.assertEqual(len(self.store.usage()), 1)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_wait_question_never_becomes_act_only_calculation(self, compose):
        case = {"states": {"good": 0.5, "bad": 0.5}, "actions": {
            "invest": {"outcomes": {"good": 10, "bad": -10}},
            "hold": {"outcomes": {"good": 0, "bad": 0}}}}
        compose.return_value = TurnPlan("Invest now.", True, json.dumps(case), [],
                                       "openai-test", 10, 5, "resp_5")
        outcome = run_turn(self.store, "Good 50%, bad 50%. Invest pays 10 or -10; hold pays 0. Should I wait?")
        self.assertIsNone(outcome.decision)
        self.assertIn("To compare waiting", outcome.text)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    def test_direct_case_is_analyzed_and_saved_without_api_keys(self):
        case = {"states": {"yes": 1}, "actions": {"do": {"outcomes": {"yes": 2}}}}
        outcome = run_turn(self.store, json.dumps(case))
        self.assertEqual(outcome.model, "local")
        self.assertEqual(outcome.decision["recommendation"]["action"], "do")
        self.assertEqual(self.store.usage(), [])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.pipeline.compose_turn", side_effect=ProviderError("OpenAI unavailable"))
    @patch("pioneer.pipeline.triage_jev")
    def test_jev_usage_survives_later_openai_failure(self, triage, _compose):
        triage.return_value = {"model": "jev-test", "scores": {"decision_request": 0.9},
                               "input_tokens": 30, "output_tokens": 4}
        with self.assertRaises(ProviderError):
            run_turn(self.store, "Should I launch?")
        self.assertEqual(len(self.store.usage()), 1)
        self.assertEqual(self.store.log()[0][1]["kind"], "note")


if __name__ == "__main__":
    unittest.main()
