import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pioneer.calibration import add_forecast, calibration_report, forecast_records, resolve_forecast
from pioneer.decision import DecisionError, analyze
from pioneer.history import MAX_EVIDENCE, MAX_RECENT_CHARS, MAX_TOTAL_CHARS, recent_messages, retrieve_history
from pioneer.objectives import objective_records, outcome_report
from pioneer.pipeline import (_last_jev_assessment, _reported_outcome,
                              _visible_forecast_probability, run_turn)
from pioneer.providers import (SYSTEM_INSTRUCTIONS, STATE_INSTRUCTIONS, ProviderError,
                               HistoryReview, TurnPlan, ask_openai, assess_jev,
                               compose_turn, review_history, triage_jev)
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

    def test_verify_rejects_missing_current_branch(self):
        (self.store.data / "refs" / "main").unlink()
        with self.assertRaisesRegex(StoreError, "Current branch is missing: main"):
            self.store.verify()

    def test_reset_main_preserves_history_on_rescue_branch(self):
        first = self.store.commit("turn", {"user": "old question", "assistant": "old answer"},
                                  usage={"provider": "openai", "model": "test", "input_tokens": 3,
                                         "output_tokens": 2})
        self.store.create_branch("alternate", first)
        old_main = self.store.commit("turn", {"user": "follow-up", "assistant": "second answer"})
        self.store.switch("alternate")
        rescue, root = self.store.reset_branch("main")
        self.assertEqual(root, self.root)
        self.assertEqual(self.store.current_branch(), "main")
        self.assertEqual(self.store.resolve(), self.root)
        self.assertEqual(self.store.messages(), [])
        self.assertEqual(self.store.resolve(rescue), old_main)
        self.assertEqual(self.store.resolve("alternate"), first)
        self.assertEqual(len(self.store.usage()), 1)
        self.assertEqual(self.store.verify()["usage_entries"], 1)
        self.assertEqual(self.store.reset_branch("main"), (None, self.root))

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
    @staticmethod
    def _openai_response(text, *, response_id="answer_1", model="test-model",
                         input_tokens=20, output_tokens=5):
        return {"id": response_id, "model": model,
                "output": [{"content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}

    @staticmethod
    def _state(**updates):
        state = {"decision_requested": False, "case_json": None, "missing": [],
                 "context": {"status": "none", "goal": "", "options": [], "known": [],
                             "uncertain": [], "provisional_view": "", "next_questions": []},
                 "explore_alternative": False, "forecast": None, "resolution": None,
                 "objective": None, "actual": None, "outcome_forecast": None}
        state.update(updates)
        return state

    def _two_calls(self, post, reply, state):
        post.side_effect = [self._openai_response(reply),
                            self._openai_response(json.dumps(state), response_id="state_1",
                                                  input_tokens=22, output_tokens=7)]

    def test_openai_instructions_do_not_script_disagreement(self):
        instructions = (SYSTEM_INSTRUCTIONS + "\n" + STATE_INSTRUCTIONS).casefold()
        self.assertNotIn("do not agree merely to be agreeable", instructions)
        self.assertNotIn("don't be agreeable", instructions)

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
        with self.assertRaises(ProviderError) as caught:
            triage_jev("Launch tomorrow")
        self.assertEqual(caught.exception.usage["input_tokens"], 50)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_integrated_jev_receives_branch_decision_state(self, post):
        post.return_value = {"model": "jev-test", "answers": {
            name: {"type": "noul", "noul": 0.7}
            for name in ("decision_request", "time_sensitive", "hard_to_reverse",
                         "missing_information", "assumption_tension")},
            "usage": {"input_tokens": 40, "output_tokens": 5}}
        history = [{"commit": "a" * 64, "quote": "Wait for legal review."}]
        result = assess_jev("Launch now?", context={"goal": "Launch timing"},
                            recent_user_messages=["We could pilot first."], history_evidence=history,
                            previous_assessment={"scores": {"hard_to_reverse": 0.4}})
        state = post.call_args.args[2]["state"]
        self.assertEqual(state["working_context"]["goal"], "Launch timing")
        self.assertEqual(state["older_user_statements"], history)
        self.assertEqual(state["previous_jev_assessment"]["scores"]["hard_to_reverse"], 0.4)
        self.assertEqual(result["scores"]["assumption_tension"], 0.7)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_integrated_jev_asks_about_recent_tension(self, post):
        post.return_value = {"model": "jev-test", "answers": {
            name: {"type": "noul", "noul": 0.6}
            for name in ("decision_request", "time_sensitive", "hard_to_reverse",
                         "missing_information", "assumption_tension")},
            "usage": {"input_tokens": 25, "output_tokens": 4}}
        assess_jev("Should we launch?", recent_user_messages=["Old 1", "Old 2", "Old 3", "Old 4", "Old 5"])
        request = post.call_args.args[2]
        self.assertIn("assumption_tension", request["questions"])
        self.assertEqual(request["state"]["recent_user_messages"], ["Old 2", "Old 3", "Old 4", "Old 5"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_integrated_jev_omits_tension_question_without_prior_statements(self, post):
        post.return_value = {"model": "jev-test", "answers": {
            name: {"type": "noul", "noul": 0.6}
            for name in ("decision_request", "time_sensitive", "hard_to_reverse", "missing_information")},
            "usage": {"input_tokens": 25, "output_tokens": 4}}
        assess_jev("Should we wait?")
        request = post.call_args.args[2]
        self.assertNotIn("assumption_tension", request["questions"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_structured_turn_contract(self, post):
        state = self._state(
            decision_requested=True, missing=["state probabilities"],
            context={"status": "active", "goal": "Decide whether to launch", "options": ["launch"],
                     "known": [], "uncertain": ["demand"], "provisional_view": "Try a pilot",
                     "next_questions": ["What would a pilot cost?"]})
        self._two_calls(post, "Tell me the states.", state)
        plan = compose_turn([{"role": "user", "content": "Should I launch?"}], model="test-model")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(plan.reply, "Tell me the states.")
        self.assertTrue(plan.decision_requested)
        self.assertEqual(plan.missing, ["state probabilities"])
        self.assertEqual(plan.context["provisional_view"], "Try a pilot")
        self.assertNotIn("text", post.call_args_list[0].args[2])
        schema = post.call_args_list[1].args[2]["text"]["format"]["schema"]
        self.assertNotIn("reply", schema["required"])
        self.assertNotIn("history_conflict", schema["required"])
        state_input = json.loads(post.call_args_list[1].args[2]["input"][0]["content"])
        self.assertEqual(state_input["finished_assistant_reply"], "Tell me the states.")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_structured_forecast_fields_and_calibration_context(self, post):
        context = {"status": "active", "goal": "Launch timing", "options": [], "known": [],
                   "uncertain": [], "provisional_view": "Wait", "next_questions": []}
        forecast = {"event": "Launch by Friday", "probability": 0.7, "deadline": "Friday"}
        response = self._state(context=context, forecast=forecast)
        reply = "I estimate a 70% chance of launching by Friday."
        self._two_calls(post, reply, response)
        plan = compose_turn([{"role": "user", "content": "What are the chances?"}],
                            calibration={"count": 5, "observed_rate": 0.4})
        self.assertEqual(plan.reply, reply)
        self.assertEqual(plan.forecast, forecast)
        self.assertIsNone(plan.resolution)
        self.assertIn("forecast", post.call_args.args[2]["text"]["format"]["schema"]["required"])
        self.assertIn("forecast_accuracy_history", post.call_args_list[0].args[2]["input"][-2]["content"])
        self.assertIn("forecast_accuracy_history", post.call_args.args[2]["input"][0]["content"])
        response["forecast"]["probability"] = 1.5
        self._two_calls(post, reply, response)
        invalid = compose_turn([{"role": "user", "content": "What are the chances?"}])
        self.assertEqual(invalid.reply, reply)
        self.assertIsNone(invalid.forecast)
        self.assertTrue(invalid.state_error)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_structured_desired_and_actual_outcomes(self, post):
        context = {"status": "none", "goal": "", "options": [], "known": [],
                   "uncertain": [], "provisional_view": "", "next_questions": []}
        objective = {"goal": "Launch", "metric": "paid users", "kind": "numeric",
                     "desired": 100, "direction": "at_least", "unit": "users",
                     "deadline": "Friday", "action": ""}
        actual = {"objective_id": "", "value": 70, "as_of": "Friday", "note": ""}
        response = self._state(context=context, objective=objective, actual=actual)
        reply = "Let's compare the result with your goal."
        self._two_calls(post, reply, response)
        plan = compose_turn([{"role": "user", "content": "We wanted 100 users; got 70."}],
                            outcome_history={"final_count": 2})
        self.assertEqual(plan.reply, reply)
        self.assertEqual(plan.objective, objective)
        self.assertEqual(plan.actual, actual)
        schema = post.call_args.args[2]["text"]["format"]["schema"]
        self.assertIn("objective", schema["required"])
        self.assertIn("actual", schema["required"])
        self.assertIn("outcome_history", post.call_args_list[0].args[2]["input"][-2]["content"])
        response["actual"]["value"] = float("inf")
        self._two_calls(post, reply, response)
        invalid = compose_turn([{"role": "user", "content": "We got an impossible number."}])
        self.assertEqual(invalid.reply, reply)
        self.assertIsNone(invalid.actual)
        self.assertTrue(invalid.state_error)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_older_history_is_input_data_without_same_turn_challenge(self, post):
        context = {"status": "active", "goal": "Launch", "options": [], "known": [],
                   "uncertain": [], "provisional_view": "Wait", "next_questions": []}
        evidence = [{"commit": "a" * 64, "quote": "Wait for legal review before launch."}]
        guidance = {"attention": {"reversibility": "focus"}, "priorities": ["reversibility"]}
        self._two_calls(post, "A launch now may be premature.",
                        self._state(decision_requested=True, context=context))
        plan = compose_turn([{"role": "user", "content": "Launch now?"}], context=context,
                            history_evidence=evidence, jev_guidance=guidance)
        payload = post.call_args_list[0].args[2]
        self.assertEqual(payload["input"][-1]["content"], "Launch now?")
        self.assertIn("retrieved_branch_statements", payload["input"][-2]["content"])
        self.assertIn("decision_attention", payload["input"][-2]["content"])
        self.assertNotIn("Wait for legal review", payload["instructions"])
        self.assertIsNone(plan.history_conflict)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_history_review_is_separate_and_attributes_the_quoted_speaker(self, post):
        evidence = [{"commit": "a" * 64, "role": "assistant", "provider": "openai",
                     "quote": "I thought the pilot would take one week."}]
        conflict = {"commit": evidence[0]["commit"], "role": "assistant",
                    "quote": "pilot would take one week",
                    "challenge": "What changed your estimate?"}
        post.return_value = {"id": "review_1", "model": "test", "output": [{"content": [
            {"type": "output_text", "text": json.dumps({"conflict": conflict})}]}],
            "usage": {"input_tokens": 11, "output_tokens": 5}}
        review = review_history("A pilot may take a month.", evidence,
                                user_text="What do you think now?", model="test")
        self.assertEqual(review.conflict, conflict)
        self.assertEqual((review.input_tokens, review.output_tokens), (11, 5))
        payload = post.call_args.args[2]
        self.assertEqual(payload["input"][0]["role"], "user")
        sent = json.loads(payload["input"][0]["content"])
        self.assertEqual(sent["finished_assistant_reply"], "A pilot may take a month.")
        self.assertEqual(sent["retrieved_branch_statements"][0]["role"], "assistant")
        self.assertNotIn("history_conflict", payload["text"]["format"]["schema"]["required"])
        self.assertIn("conflict", payload["text"]["format"]["schema"]["required"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_unusable_openai_response_exposes_returned_usage(self, post):
        post.return_value = {"id": "resp_bad", "model": "test-model", "output": [
            {"content": []}],
            "usage": {"input_tokens": 19, "output_tokens": 4}}
        with self.assertRaises(ProviderError) as caught:
            compose_turn([{"role": "user", "content": "Hello"}], model="test-model")
        self.assertEqual(caught.exception.usage["input_tokens"], 19)
        self.assertEqual(caught.exception.usage["response_id"], "resp_bad")
        self.assertEqual(post.call_count, 1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.store.init()
        self.review_patcher = patch("pioneer.pipeline.review_history")
        self.review = self.review_patcher.start()
        self.addCleanup(self.review_patcher.stop)
        self.review.return_value = HistoryReview(None, "test", 3, 2, "history_1")

    def tearDown(self):
        self.temp.cleanup()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "", "TYPESAFE_API_KEY": "test"})
    @patch("pioneer.pipeline.assess_jev")
    def test_missing_openai_key_does_not_spend_on_jev(self, jev):
        with self.assertRaisesRegex(ProviderError, "OPENAI_API_KEY"):
            run_turn(self.store, "Should I launch tomorrow?")
        jev.assert_not_called()
        self.assertEqual(self.store.usage(), [])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.pipeline.compose_turn")
    @patch("pioneer.pipeline.assess_jev")
    def test_one_turn_integrates_jev_attention_analysis_and_usage(self, jev, compose):
        case = {"states": {"good": 0.5, "bad": 0.5}, "actions": {
            "invest": {"outcomes": {"good": 10, "bad": -10}},
            "hold": {"outcomes": {"good": 0, "bad": 0}}},
            "wait": {"delay_cost": 1, "information_cost": 0, "signals": {
                "positive": {"good": 0.9, "bad": 0.1},
                "negative": {"good": 0.1, "bad": 0.9}}}}
        text = "Good 50%, bad 50%. Invest pays 10 or -10; hold pays 0. Wait delay costs 1 and information costs 0. A positive signal is 90% likely in good and 10% in bad; a negative signal is 10% in good and 90% in bad."
        jev.return_value = {"model": "jev-test", "scores": {"decision_request": 0.99,
            "time_sensitive": 0.2, "hard_to_reverse": 0.8, "missing_information": 0.9},
            "input_tokens": 30, "output_tokens": 4}
        compose.return_value = TurnPlan("Here is the comparison.", True, json.dumps(case), [],
                                       "openai-test", 50, 20, "resp_3")
        outcome = run_turn(self.store, text)
        self.assertEqual(outcome.decision["recommendation"]["kind"], "wait")
        self.assertEqual(outcome.text, "Here is the comparison.")
        self.assertIn("favors waiting", "\n".join(outcome.notices))
        self.assertEqual(self.store.read_object(outcome.commit)["payload"]["assistant"], outcome.text)
        self.assertEqual(self.store.read_object(outcome.commit)["payload"]["notices"], list(outcome.notices))
        self.assertEqual(len(self.store.log()), 2)
        self.assertEqual(len(self.store.usage()), 2)
        self.assertEqual({record["commit"] for record in self.store.usage()}, {outcome.commit})
        saved_jev = self.store.read_object(outcome.commit)["payload"]["jev"]
        self.assertEqual(saved_jev["model"], "jev-test")
        self.assertIn("attention", saved_jev["guidance"])
        self.assertEqual(compose.call_args.kwargs["jev_guidance"], saved_jev["guidance"])

        compose.return_value = TurnPlan("The signal is still the deciding factor.", True,
                                        None, [], "openai-test", 14, 7, "resp_4")
        run_turn(self.store, "For those available actions, which choice makes sense after the signal?")
        sent = compose.call_args.args[0]
        self.assertEqual(sent[-2], {"role": "assistant", "content": "Here is the comparison."})
        checked = compose.call_args.kwargs["verified_decision"]
        self.assertTrue(checked)
        self.assertIn("wait", json.dumps(checked, sort_keys=True))

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.pipeline.compose_turn")
    @patch("pioneer.pipeline.assess_jev")
    def test_ordinary_chat_skips_optional_jev_call(self, jev, compose):
        compose.return_value = TurnPlan("Hello.", False, None, [], "test", 8, 3, "r")
        outcome = run_turn(self.store, "Hello, Pioneer.")
        jev.assert_not_called()
        self.assertEqual(outcome.text, "Hello.")
        self.assertEqual(outcome.notices, ())
        self.assertEqual(compose.call_args.kwargs["jev_guidance"], None)
        self.review.assert_not_called()
        self.assertNotIn("jev", self.store.read_object(outcome.commit)["payload"])
        self.assertEqual(len(self.store.usage()), 1)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_ordinary_recorded_wording_is_not_treated_as_a_persistence_claim(self, compose):
        reply = "We recorded 80 users on Friday. That gives us a useful baseline."
        compose.return_value = TurnPlan(reply, False, None, [], "test", 10, 5, "r")
        outcome = run_turn(self.store, "How would you phrase our customer update?")
        payload = self.store.read_object(outcome.commit)["payload"]
        self.assertEqual(outcome.text, reply)
        self.assertEqual(payload["assistant"], reply)
        self.assertEqual(outcome.notices, ())
        self.assertNotIn("notices", payload)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_explicit_recall_recovers_checked_decision_after_neutral_turn(self, compose):
        case = {"states": {"good": 0.5, "bad": 0.5}, "actions": {
            "invest": {"outcomes": {"good": 10, "bad": -10}},
            "hold": {"outcomes": {"good": 0, "bad": 0}}},
            "wait": {"delay_cost": 1, "information_cost": 0, "signals": {
                "positive": {"good": 0.9, "bad": 0.1},
                "negative": {"good": 0.1, "bad": 0.9}}}}
        case_text = ("Good 50%, bad 50%. Invest pays 10 or -10; hold pays 0. "
                     "Wait delay costs 1 and information costs 0. A positive signal is 90% "
                     "likely in good and 10% in bad; a negative signal is 10% in good and 90% in bad.")
        neutral = {"status": "none", "goal": "", "options": [], "known": [],
                   "uncertain": [], "provisional_view": "", "next_questions": []}
        compose.side_effect = [
            TurnPlan("I can compare those choices.", True, json.dumps(case), [], "test", 10, 5, "r1"),
            TurnPlan("You're welcome.", False, None, [], "test", 10, 5, "r2", neutral),
            TurnPlan("The checked comparison favored waiting.", True, None, [], "test", 10, 5, "r3"),
        ]
        first = run_turn(self.store, case_text)
        self.assertEqual(first.decision["recommendation"]["kind"], "wait")
        neutral_turn = run_turn(self.store, "Thanks.")
        self.assertEqual(self.store.read_object(neutral_turn.commit)["payload"]["context"]["status"], "none")
        run_turn(self.store, "What did you recommend?")
        checked = compose.call_args.kwargs["verified_decision"]
        self.assertTrue(checked)
        self.assertIn("favors waiting", checked)
        assistant_history = [item["content"] for item in compose.call_args.args[0]
                             if item["role"] == "assistant"]
        self.assertEqual(assistant_history, ["I can compare those choices.", "You're welcome."])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_recall_chooses_named_checked_decision_over_newer_other_topic(self, compose):
        launch_case = {"title": "Launch timing", "states": {"ready": 1.0}, "actions": {
            "launch": {"outcomes": {"ready": 8}}, "hold": {"outcomes": {"ready": 0}}}}
        budget_case = {"title": "Budget allocation", "states": {"available": 1.0}, "actions": {
            "spend": {"outcomes": {"available": -5}}, "hold": {"outcomes": {"available": 0}}}}
        launch = run_turn(self.store, json.dumps(launch_case))
        budget = run_turn(self.store, json.dumps(budget_case))
        self.assertEqual(launch.decision["recommendation"]["action"], "launch")
        self.assertEqual(budget.decision["recommendation"]["action"], "hold")
        neutral = {"status": "none", "goal": "", "options": [], "known": [],
                   "uncertain": [], "provisional_view": "", "next_questions": []}
        compose.side_effect = [
            TurnPlan("You're welcome.", False, None, [], "test", 10, 5, "r1", neutral),
            TurnPlan("Let me pull up the launch comparison.", True, None, [], "test", 10, 5, "r2"),
        ]
        run_turn(self.store, "Thanks.")
        run_turn(self.store, "What did you recommend for the launch?")
        checked = compose.call_args.kwargs["verified_decision"]
        self.assertTrue(checked)
        self.assertIn("checked calculation favors launch", checked)
        self.assertNotIn("checked calculation favors hold", checked)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_local_record_is_not_replayed_as_assistant_speech(self, compose):
        objective = {"goal": "Launch", "metric": "paid users", "kind": "numeric",
                     "desired": 100, "direction": "at_least", "unit": "users",
                     "deadline": "Friday", "action": ""}
        compose.side_effect = [
            TurnPlan("That is a measurable aim.", False, None, [], "test", 10, 5, "r1",
                     objective=objective),
            TurnPlan("We could test demand before committing.", False, None, [],
                     "test", 10, 5, "r2"),
        ]
        first = run_turn(self.store, "Our goal is at least 100 paid users by Friday.")
        self.assertTrue(first.notices)
        self.assertEqual(self.store.messages()[-1]["content"], "That is a measurable aim.")
        run_turn(self.store, "How could we reach it?")
        sent = compose.call_args.args[0]
        self.assertEqual(sent[1], {"role": "assistant", "content": "That is a measurable aim."})
        self.assertNotIn("Target saved", sent[1]["content"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_conversation_compares_target_progress_and_final_result(self, compose):
        objective = {"goal": "Launch", "metric": "paid users", "kind": "numeric",
                     "desired": 100, "direction": "at_least", "unit": "users",
                     "deadline": "Friday", "action": ""}
        compose.return_value = TurnPlan("That gives us a clear target.", False, None, [],
                                        "test", 10, 5, "r1", objective=objective)
        target = run_turn(self.store, "Our goal is at least 100 paid users by Friday.")
        self.assertEqual(self.store.read_object(target.commit)["payload"]["objective"]["desired"], 100)
        self.assertEqual(target.text, "That gives us a clear target.")
        self.assertIn("Target saved for paid users", "\n".join(target.notices))
        self.assertEqual(self.store.read_object(target.commit)["payload"]["assistant"], target.text)
        self.assertEqual(self.store.read_object(target.commit)["payload"]["notices"], list(target.notices))

        compose.return_value = TurnPlan("We are making progress.", False, None, [],
                                        "test", 10, 5, "r2",
                                        actual={"objective_id": target.commit, "value": 70,
                                                "as_of": "today", "note": ""})
        progress = run_turn(self.store, "We have 70 paid users today.")
        self.assertEqual(progress.text, "We are making progress.")
        self.assertIn("progress reading", "\n".join(progress.notices))
        self.assertEqual(outcome_report(self.store)["progress_count"], 1)

        compose.return_value = TurnPlan("Here is the result.", False, None, [],
                                        "test", 10, 5, "r3",
                                        actual={"objective_id": target.commit, "value": 80,
                                                "as_of": "Friday", "note": ""})
        final = run_turn(self.store, "We got 80 paid users by Friday.")
        self.assertEqual(final.text, "Here is the result.")
        self.assertIn("20 users", "\n".join(final.notices))
        report = outcome_report(self.store)
        self.assertEqual((report["final_count"], report["missed_count"]), (1, 1))
        self.assertEqual(report["comparisons"][0]["gap"], -20)

        compose.return_value = TurnPlan("Let's review the gap.", False, None, [],
                                        "test", 10, 5, "r4")
        run_turn(self.store, "How calibrated were we?")
        self.assertEqual(compose.call_args.kwargs["outcome_history"]["final_count"], 1)
        self.assertIsNone(compose.call_args.kwargs["calibration"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_retrospective_target_and_actual_share_one_turn(self, compose):
        objective = {"goal": "Launch", "metric": "paid users", "kind": "numeric",
                     "desired": 100, "direction": "at_least", "unit": "users",
                     "deadline": "Friday", "action": ""}
        compose.return_value = TurnPlan("That is a useful result to compare.", False, None, [],
                                        "test", 10, 5, "r", objective=objective,
                                        actual={"objective_id": "", "value": 70,
                                                "as_of": "Friday", "note": ""})
        result = run_turn(self.store, "We wanted at least 100 paid users by Friday, but got 70 paid users by Friday.")
        payload = self.store.read_object(result.commit)["payload"]
        self.assertEqual(payload["actual"]["objective_id"], "$self")
        self.assertEqual(objective_records(self.store)[0]["observations"][0]["objective_id"], result.commit)
        self.assertEqual(result.text, "That is a useful result to compare.")
        self.assertIn("30 users", "\n".join(result.notices))
        self.assertEqual(outcome_report(self.store)["missed_count"], 1)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_date_before_value_and_deadline_followup_save_target_and_result(self, compose):
        objective = {"goal": "Grow engagement", "metric": "weekly active users",
                     "kind": "numeric", "desired": 100, "direction": "at_least",
                     "unit": "users", "deadline": "Jan 31", "action": ""}
        compose.return_value = TurnPlan("By when should we check?", False, None, [],
                                        "test", 10, 5, "r1")
        first = run_turn(self.store, "I want at least 100 weekly active users.")
        self.assertNotIn("objective", self.store.read_object(first.commit)["payload"])
        compose.return_value = TurnPlan("I'll track that.", False, None, [],
                                        "test", 10, 5, "r2", objective=objective)
        target = run_turn(self.store, "By Jan 31.")
        self.assertIn("objective", self.store.read_object(target.commit)["payload"])
        compose.return_value = TurnPlan("Here's the comparison.", False, None, [],
                                        "test", 10, 5, "r3",
                                        actual={"objective_id": target.commit, "value": 90,
                                                "as_of": "Jan 31", "note": ""})
        result = run_turn(self.store, "The result on Jan 31 was 90 weekly active users.")
        self.assertEqual(self.store.read_object(result.commit)["payload"]["actual"]["value"], 90)
        compose.return_value = TurnPlan("The target is clear.", False, None, [],
                                        "test", 10, 5, "r4", objective=objective)
        date_first = run_turn(self.store, "My target by Jan 31 is 100 weekly active users.")
        self.assertIn("objective", self.store.read_object(date_first.commit)["payload"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_different_metric_cannot_be_saved_as_target_result(self, compose):
        objective = {"goal": "Grow engagement", "metric": "weekly active users",
                     "kind": "numeric", "desired": 100, "direction": "at_least",
                     "unit": "users", "deadline": "Jan 31", "action": ""}
        compose.return_value = TurnPlan("I'll track that.", False, None, [],
                                        "test", 10, 5, "r1", objective=objective)
        target = run_turn(self.store, "Our goal is 100 weekly active users by Jan 31.")
        compose.return_value = TurnPlan("Let me check which measure you mean.", False, None, [],
                                        "test", 10, 5, "r2",
                                        actual={"objective_id": target.commit, "value": 70,
                                                "as_of": "Jan 31", "note": ""})
        wrong = run_turn(self.store, "We got 70 dollars in revenue on Jan 31; weekly active users are unknown.")
        self.assertNotIn("actual", self.store.read_object(wrong.commit)["payload"])
        compose.return_value = TurnPlan("Those measures differ.", False, None, [],
                                        "test", 10, 5, "r3", objective=objective,
                                        actual={"objective_id": "", "value": 70,
                                                "as_of": "Jan 31", "note": ""})
        same_turn = run_turn(self.store, "Our goal is 100 weekly active users by Jan 31. We got 70 dollars in revenue on Jan 31.")
        self.assertNotIn("actual", self.store.read_object(same_turn.commit)["payload"])
        compose.return_value = TurnPlan("That still does not report active users.", False, None, [],
                                        "test", 10, 5, "r4",
                                        actual={"objective_id": target.commit, "value": 70,
                                                "as_of": "Jan 31", "note": ""})
        mixed = run_turn(self.store, "We got 70 dollars in revenue and weekly active users are unknown on Jan 31.")
        self.assertNotIn("actual", self.store.read_object(mixed.commit)["payload"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_rejected_target_preserves_openai_claim_and_marks_missing_record(self, compose):
        objective = {"goal": "Launch", "metric": "paid users", "kind": "numeric",
                     "desired": 100, "direction": "at_least", "unit": "users",
                     "deadline": "Friday", "action": ""}
        reply = "I've saved your target of 100 paid users by Friday."
        compose.return_value = TurnPlan(reply,
                                        False, None, [], "test", 10, 5, "r", objective=objective)
        outcome = run_turn(self.store, "I'm considering a launch, but haven't settled on a target.")
        payload = self.store.read_object(outcome.commit)["payload"]
        self.assertNotIn("objective", payload)
        self.assertEqual(outcome.text, reply)
        self.assertEqual(payload["assistant"], reply)
        self.assertTrue(any("target" in item.casefold() for item in outcome.notices))
        self.assertEqual(payload["notices"], list(outcome.notices))

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_unproposed_target_claim_is_preserved_with_separate_check(self, compose):
        reply = "I have already saved your target. We can decide timing next."
        compose.return_value = TurnPlan(
            reply,
            False, None, [], "test", 10, 5, "r")
        outcome = run_turn(self.store, "I may launch, but haven't set a target yet.")
        payload = self.store.read_object(outcome.commit)["payload"]
        self.assertNotIn("objective", payload)
        self.assertEqual(outcome.text, reply)
        self.assertEqual(payload["assistant"], reply)
        self.assertTrue(any("target" in item.casefold() for item in outcome.notices))

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_false_forecast_claim_stays_attributed_while_target_check_remains(self, compose):
        objective = {"goal": "Launch", "metric": "paid users", "kind": "numeric",
                     "desired": 100, "direction": "at_least", "unit": "users",
                     "deadline": "Friday", "action": ""}
        reply = "I saved your forecast of 70% for Friday. The target is clear."
        compose.return_value = TurnPlan(
            reply,
            False, None, [], "test", 10, 5, "r", objective=objective)
        outcome = run_turn(self.store, "Our goal is at least 100 paid users by Friday.")
        payload = self.store.read_object(outcome.commit)["payload"]
        self.assertIn("objective", payload)
        self.assertNotIn("forecast", payload)
        self.assertEqual(outcome.text, reply)
        self.assertEqual(payload["assistant"], reply)
        records = "\n".join(outcome.notices)
        self.assertIn("Target saved", records)
        self.assertIn("forecast", records.casefold())
        self.assertEqual(payload["notices"], list(outcome.notices))

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_conflicting_save_description_remains_authored_with_correct_record(self, compose):
        objective = {"goal": "Launch", "metric": "paid users", "kind": "numeric",
                     "desired": 100, "direction": "at_least", "unit": "users",
                     "deadline": "Friday", "action": ""}
        reply = "I saved your target of 200 users Monday. Let's review the path."
        compose.return_value = TurnPlan(
            reply,
            False, None, [], "test", 10, 5, "r", objective=objective)
        outcome = run_turn(self.store, "Our goal is at least 100 paid users by Friday.")
        payload = self.store.read_object(outcome.commit)["payload"]
        self.assertEqual(payload["objective"]["desired"], 100)
        self.assertEqual(payload["objective"]["deadline"], "Friday")
        self.assertEqual(outcome.text, reply)
        self.assertEqual(payload["assistant"], reply)
        records = "\n".join(outcome.notices)
        self.assertIn("Target saved", records)
        self.assertIn("100 users", records)
        self.assertIn("Friday", records)
        self.assertNotIn("200 users", records)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_truthful_confirmation_of_prior_target_is_not_filtered(self, compose):
        objective = {"goal": "Launch", "metric": "paid users", "kind": "numeric",
                     "desired": 100, "direction": "at_least", "unit": "users",
                     "deadline": "Friday", "action": ""}
        reply = "Yes, your target has been saved."
        compose.side_effect = [
            TurnPlan("That's a clear target.", False, None, [], "test", 10, 5, "r1",
                     objective=objective),
            TurnPlan(reply, False, None, [], "test", 10, 5, "r2"),
        ]
        first = run_turn(self.store, "Our goal is at least 100 paid users by Friday.")
        self.assertIn("objective", self.store.read_object(first.commit)["payload"])
        confirmed = run_turn(self.store, "Did you save my target?")
        payload = self.store.read_object(confirmed.commit)["payload"]
        self.assertEqual(confirmed.text, reply)
        self.assertEqual(payload["assistant"], reply)
        self.assertEqual(confirmed.notices, ())
        self.assertNotIn("notices", payload)
        self.assertNotIn("objective", payload)
        self.assertEqual(len(objective_records(self.store)), 1)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_old_target_does_not_confirm_false_new_save_claim(self, compose):
        old_target = {"goal": "Launch", "metric": "paid users", "kind": "numeric",
                      "desired": 100, "direction": "at_least", "unit": "users",
                      "deadline": "Friday", "action": ""}
        compose.return_value = TurnPlan("That's a useful target.", False, None, [],
                                        "test", 10, 5, "r1", objective=old_target)
        first = run_turn(self.store, "Our goal is at least 100 paid users by Friday.")
        self.assertIn("objective", self.store.read_object(first.commit)["payload"])

        reply = "I saved your target of 200 paid users by Monday."
        compose.return_value = TurnPlan(reply, False, None, [], "test", 10, 5, "r2")
        second = run_turn(self.store, "I'm considering a new target for Monday, but haven't chosen it.")
        payload = self.store.read_object(second.commit)["payload"]
        self.assertEqual(second.text, reply)
        self.assertEqual(payload["assistant"], reply)
        self.assertNotIn("objective", payload)
        self.assertEqual(len(objective_records(self.store)), 1)
        self.assertTrue(any("target" in item.casefold() and "this turn" in item.casefold()
                            for item in second.notices))

        new_target = {**old_target, "desired": 200, "deadline": "Monday"}
        compose.return_value = TurnPlan(reply, False, None, [], "test", 10, 5, "r3",
                                        objective=new_target)
        third = run_turn(self.store, "I still haven't chosen a new target.")
        third_payload = self.store.read_object(third.commit)["payload"]
        self.assertEqual(third.text, reply)
        self.assertNotIn("objective", third_payload)
        self.assertEqual(len(objective_records(self.store)), 1)
        self.assertTrue(any("target" in item.casefold() and "did not save" in item.casefold()
                            for item in third.notices))

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_old_target_does_not_verify_revised_target_history_claim(self, compose):
        old_target = {"goal": "Launch", "metric": "paid users", "kind": "numeric",
                      "desired": 100, "direction": "at_least", "unit": "users",
                      "deadline": "Friday", "action": ""}
        compose.return_value = TurnPlan("That's a clear target.", False, None, [],
                                        "test", 10, 5, "r1", objective=old_target)
        run_turn(self.store, "Our goal is at least 100 paid users by Friday.")
        reply = "I have saved your revised target."
        compose.return_value = TurnPlan(reply, False, None, [], "test", 10, 5, "r2")
        outcome = run_turn(self.store, "Did you save my revised target?")
        self.assertEqual(outcome.text, reply)
        self.assertTrue(any("could not verify" in notice and "target" in notice
                            for notice in outcome.notices))
        self.assertEqual(len(objective_records(self.store)), 1)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_short_answer_to_one_measure_question_records_result(self, compose):
        objective = {"goal": "Launch", "metric": "paid users", "kind": "numeric",
                     "desired": 100, "direction": "at_least", "unit": "users",
                     "deadline": "Friday", "action": ""}
        compose.return_value = TurnPlan("How many paid users did you have by Friday?",
                                        False, None, [], "test", 10, 5, "r1",
                                        objective=objective)
        target = run_turn(self.store, "Our goal is 100 paid users by Friday.")
        compose.return_value = TurnPlan("That missed the target.", False, None, [],
                                        "test", 10, 5, "r2",
                                        actual={"objective_id": target.commit, "value": 70,
                                                "as_of": "Friday", "note": ""})
        result = run_turn(self.store, "70.")
        self.assertEqual(self.store.read_object(result.commit)["payload"]["actual"]["value"], 70)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_binary_result_after_by_deadline_is_not_a_success(self, compose):
        objective = {"goal": "We launch by Friday", "metric": "completion", "kind": "binary",
                     "desired": True, "direction": "exact", "unit": "", "deadline": "by Friday", "action": ""}
        compose.return_value = TurnPlan("I'll track that.", False, None, [],
                                        "test", 10, 5, "r1", objective=objective)
        target = run_turn(self.store, "Our goal is to launch by Friday.")
        compose.return_value = TurnPlan("That was late.", False, None, [],
                                        "test", 10, 5, "r2",
                                        actual={"objective_id": target.commit, "value": True,
                                                "as_of": "by Friday", "note": ""})
        late = run_turn(self.store, "We launched after Friday.")
        self.assertNotIn("actual", self.store.read_object(late.commit)["payload"])
        self.assertIs(_reported_outcome("We launched after Friday.",
                      {"id": "a" * 64, "event": "We launch by Friday", "deadline": "by Friday"}), False)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_conversation_binary_target_requires_reported_result(self, compose):
        objective = {"goal": "Launch", "metric": "completion", "kind": "binary",
                     "desired": True, "direction": "exact", "unit": "",
                     "deadline": "Friday", "action": ""}
        compose.return_value = TurnPlan("I'll track that goal.", False, None, [],
                                        "test", 10, 5, "r1", objective=objective)
        target = run_turn(self.store, "Our goal is to launch by Friday.")
        compose.return_value = TurnPlan("That remains a plan.", False, None, [],
                                        "test", 10, 5, "r2",
                                        actual={"objective_id": target.commit, "value": True,
                                                "as_of": "Friday", "note": ""})
        planned = run_turn(self.store, "We plan to launch Friday.")
        self.assertNotIn("actual", self.store.read_object(planned.commit)["payload"])
        compose.return_value = TurnPlan("Thanks for the report.", False, None, [],
                                        "test", 10, 5, "r3",
                                        actual={"objective_id": target.commit, "value": True,
                                                "as_of": "Friday", "note": ""})
        final = run_turn(self.store, "We launched Friday.")
        self.assertTrue(self.store.read_object(final.commit)["payload"]["actual"]["value"])
        self.assertEqual(outcome_report(self.store)["met_count"], 1)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_conversation_records_and_scores_its_own_forecast(self, compose):
        context = {"status": "active", "goal": "Launch timing", "options": ["launch", "wait"],
                   "known": [], "uncertain": [], "provisional_view": "Wait", "next_questions": []}
        compose.return_value = TurnPlan(
            "I estimate a 70% chance we launch by Friday.", False, None, [], "test-model", 10, 5,
            "r1", context, forecast={"event": "We launch by Friday", "probability": 0.7,
                                     "deadline": "Friday"})
        first = run_turn(self.store, "What are the chances we launch by Friday?")
        saved = self.store.read_object(first.commit)["payload"]["forecast"]
        self.assertEqual((saved["probability"], saved["source"], saved["topic"]),
                         (0.7, "pioneer", "Launch timing"))
        self.assertEqual(first.text, "I estimate a 70% chance we launch by Friday.")
        self.assertIn("Forecast saved: 70.0%", "\n".join(first.notices))

        compose.return_value = TurnPlan("That remains a plan.", False, None, [], "test-model", 10, 5,
                                        "r2", context, resolution={"forecast_id": first.commit,
                                                                    "outcome": True})
        planned = run_turn(self.store, "We plan to launch Friday.")
        self.assertNotIn("resolution", self.store.read_object(planned.commit)["payload"])
        self.assertIsNone(forecast_records(self.store)[0]["resolution"])

        compose.return_value = TurnPlan("That was after the deadline.", False, None, [],
                                        "test-model", 10, 5, "r-late", context,
                                        resolution={"forecast_id": first.commit, "outcome": True})
        late = run_turn(self.store, "We launched Saturday.")
        self.assertNotIn("resolution", self.store.read_object(late.commit)["payload"])
        self.assertIsNone(forecast_records(self.store)[0]["resolution"])

        compose.return_value = TurnPlan("Thanks for reporting the result.", False, None, [],
                                        "test-model", 10, 5, "r3", context,
                                        resolution={"forecast_id": first.commit, "outcome": True})
        resolved = run_turn(self.store, "We launched Friday.")
        self.assertEqual(self.store.read_object(resolved.commit)["payload"]["resolution"]["forecast_id"],
                         first.commit)
        report = calibration_report(self.store)
        self.assertEqual(report["count"], 1)
        self.assertAlmostEqual(report["brier_score"], 0.09)

        compose.return_value = TurnPlan("Thanks for the correction.", False, None, [],
                                        "test-model", 10, 5, "r-correct", context,
                                        resolution={"forecast_id": first.commit, "outcome": False})
        corrected = run_turn(self.store, "Actually, we did not launch by Friday.")
        self.assertEqual(self.store.read_object(corrected.commit)["payload"]["resolution"]["outcome"], False)
        self.assertEqual(compose.call_args.kwargs["recent_resolutions"][0]["id"], first.commit)
        self.assertAlmostEqual(calibration_report(self.store)["brier_score"], 0.49)

        compose.return_value = TurnPlan("One result is too little to calibrate me.", False, None, [],
                                        "test-model", 10, 5, "r4", context)
        run_turn(self.store, "How accurate are your forecasts? Give me the Brier score.")
        self.assertEqual(compose.call_args.kwargs["calibration"]["count"], 1)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_forecast_requires_matching_visible_event_and_deadline(self, compose):
        forecast = {"event": "We launch by Friday", "probability": 0.7, "deadline": "Friday"}
        for reply in ("Rain is 70%, but launch by Friday is 30%.",
                      "I estimate a 70% chance of rain."):
            compose.return_value = TurnPlan(reply, False, None, [], "test-model", 10, 5,
                                            "r", forecast=forecast)
            result = run_turn(self.store, "Give me odds we launch by Friday.")
            self.assertNotIn("forecast", self.store.read_object(result.commit)["payload"])
        self.assertEqual(forecast_records(self.store), [])

        compose.return_value = TurnPlan("I estimate a 70% chance we launch by Friday.",
                                        False, None, [], "test-model", 10, 5,
                                        "r", forecast=forecast)
        result = run_turn(self.store, "How confident are you that we launch by Friday?")
        self.assertIn("forecast", self.store.read_object(result.commit)["payload"])

        beta = {"event": "Project Beta launch by Friday", "probability": 0.7,
                "deadline": "Friday"}
        compose.return_value = TurnPlan(
            "Project Alpha launch by Friday: 70%, Project Beta launch by Friday: 30%.",
            False, None, [], "test-model", 10, 5, "r", forecast=beta)
        result = run_turn(self.store, "What are the chances Project Beta launches by Friday?")
        self.assertNotIn("forecast", self.store.read_object(result.commit)["payload"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_late_report_does_not_become_a_forecast_success(self, compose):
        forecast_id = add_forecast(self.store, "Project Beta launch by Friday", 0.7,
                                   "Friday", source="pioneer")
        report = "Project Beta launched Saturday, not Friday."
        compose.return_value = TurnPlan("That missed Friday.", False, None, [], "test", 10, 5,
                                        "r", resolution={"forecast_id": forecast_id,
                                                         "outcome": True})
        wrong = run_turn(self.store, report)
        self.assertNotIn("resolution", self.store.read_object(wrong.commit)["payload"])
        compose.return_value = TurnPlan("I'll record that as a miss.", False, None, [],
                                        "test", 10, 5, "r", resolution={"forecast_id": forecast_id,
                                                                         "outcome": False})
        corrected = run_turn(self.store, report)
        self.assertFalse(self.store.read_object(corrected.commit)["payload"]["resolution"]["outcome"])

    def test_forecast_identity_allows_ordinary_word_order_without_crossing_events(self):
        forecast = {"id": "a" * 64, "event": "Project Beta launch by Friday",
                    "probability": 0.7, "deadline": "Friday"}
        self.assertTrue(_visible_forecast_probability(
            "I estimate a 70% chance we launch Project Beta by Friday.", forecast))
        self.assertTrue(_visible_forecast_probability(
            "I estimate a 70% chance Project Beta successfully launches by Friday.", forecast))
        self.assertFalse(_visible_forecast_probability(
            "Project Alpha launch by Friday: 70%, Project Beta launch by Friday: 30%.", forecast))
        self.assertIs(_reported_outcome("We launched Project Beta by Friday.", forecast), True)
        self.assertIs(_reported_outcome("Project Beta successfully launched by Friday.", forecast), True)
        self.assertIsNone(_reported_outcome(
            "Project Alpha launched Friday; Project Beta launched Saturday.", forecast))
        self.assertIs(_reported_outcome(
            "Project Alpha launched Friday; Project Beta did not launch Friday.", forecast), False)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_similar_forecast_history_reaches_future_decision(self, compose):
        for index in range(5):
            forecast_id = add_forecast(self.store, f"Launch trial {index}", 0.6,
                                       "Friday", topic="Launch timing", source="pioneer", model="test")
            resolve_forecast(self.store, forecast_id, index < 3)
        context = {"status": "active", "goal": "Launch timing", "options": ["launch", "wait"],
                   "known": [], "uncertain": [], "provisional_view": "Wait", "next_questions": []}
        self.store.commit("turn", {"user": "Should we launch?", "assistant": "Consider waiting.",
                                   "context": context})
        compose.return_value = TurnPlan("Let's examine the timing.", True, None, [], "test", 10, 5,
                                        "r", context)
        run_turn(self.store, "Should we launch next week?")
        evidence = compose.call_args.kwargs["calibration"]
        self.assertEqual(evidence["count"], 5)
        self.assertEqual(evidence["observed_rate"], 0.6)
        self.assertIn("Limited evidence", evidence["caveat"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_forecast_question_receives_only_matching_topic_history(self, compose):
        for index in range(5):
            forecast_id = add_forecast(self.store, f"Launch trial {index}", 0.6,
                                       "Friday", topic="Launch timing", source="pioneer")
            resolve_forecast(self.store, forecast_id, index < 3)
        compose.return_value = TurnPlan("I need the current evidence to estimate that.",
                                        False, None, [], "test", 10, 5, "r")
        run_turn(self.store, "What are the chances we launch by Friday?")
        self.assertEqual(compose.call_args.kwargs["calibration"]["count"], 5)
        run_turn(self.store, "What are the chances of rain by Friday?")
        self.assertIsNone(compose.call_args.kwargs["calibration"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_calibration_history_does_not_cross_project_subjects(self, compose):
        for index in range(5):
            forecast_id = add_forecast(self.store, f"Project Alpha launch trial {index}", 0.6,
                                       "Friday", topic="Project Alpha launch", source="pioneer")
            resolve_forecast(self.store, forecast_id, index < 3)
        compose.return_value = TurnPlan("Let's estimate Beta separately.", False, None, [],
                                        "test", 10, 5, "r")
        run_turn(self.store, "What are the chances Project Beta launches by Friday?")
        self.assertIsNone(compose.call_args.kwargs["calibration"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.pipeline.compose_turn")
    @patch("pioneer.pipeline.assess_jev")
    def test_jev_attention_carries_forward_with_same_branch_goal(self, jev, compose):
        context = {"status": "active", "goal": "Launch timing", "options": ["launch", "pilot"],
                   "known": [], "uncertain": ["deadline"], "provisional_view": "Pilot first",
                   "next_questions": []}
        jev.return_value = {"model": "jev-test", "scores": {
            "decision_request": 0.9, "time_sensitive": 0.8, "hard_to_reverse": 0.7,
            "missing_information": 0.8}, "input_tokens": 20, "output_tokens": 4}
        compose.return_value = TurnPlan("A pilot looks safer.", True, None, [], "test", 10, 5,
                                        "r", context)
        run_turn(self.store, "Should we launch now or pilot first?")
        run_turn(self.store, "The deadline moved to Friday.")
        self.assertEqual(jev.call_count, 2)
        previous = jev.call_args.kwargs["previous_assessment"]
        self.assertEqual(previous["scores"]["time_sensitive"], 0.8)
        self.assertIn("urgency", previous["attention"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.pipeline.compose_turn")
    @patch("pioneer.pipeline.assess_jev")
    def test_new_decision_does_not_inherit_unrelated_jev_state(self, jev, compose):
        launch = {"status": "active", "goal": "Launch timing", "options": ["launch", "pilot"],
                  "known": [], "uncertain": [], "provisional_view": "Pilot first", "next_questions": []}
        house = {"status": "active", "goal": "House purchase", "options": ["buy", "wait"],
                 "known": [], "uncertain": [], "provisional_view": "Need a budget", "next_questions": []}
        jev.return_value = {"model": "jev-test", "scores": {
            "decision_request": 0.9, "time_sensitive": 0.8, "hard_to_reverse": 0.7,
            "missing_information": 0.8}, "input_tokens": 20, "output_tokens": 4}
        compose.side_effect = [
            TurnPlan("Pilot first.", True, None, [], "test", 10, 5, "r1", launch),
            TurnPlan("Let's examine your budget.", True, None, [], "test", 10, 5, "r2", house),
        ]
        run_turn(self.store, "Should we launch now or pilot first?")
        outcome = run_turn(self.store, "Should I buy a house?")
        self.assertIsNone(jev.call_args.kwargs["context"])
        self.assertIsNone(jev.call_args.kwargs["previous_assessment"])
        self.assertEqual(jev.call_args.kwargs["recent_user_messages"], [])
        self.assertEqual(self.store.read_object(outcome.commit)["payload"]["jev"]["goal"], "House purchase")

    def test_jev_decision_state_stays_on_its_branch(self):
        context = {"status": "active", "goal": "Launch timing"}
        self.store.commit("turn", {"user": "Should we launch?", "assistant": "Consider a pilot.",
                                   "context": context})
        self.store.create_branch("alternate")
        self.store.commit("turn", {"user": "Main deadline", "assistant": "Noted.",
                                   "context": context,
                                   "jev": {"goal": "Launch timing", "scores": {"time_sensitive": 0.9},
                                           "guidance": {"attention": {"urgency": "focus"}}}})
        self.store.switch("alternate")
        self.store.commit("turn", {"user": "Alternate deadline", "assistant": "Noted.",
                                   "context": context,
                                   "jev": {"goal": "Launch timing", "scores": {"time_sensitive": 0.1},
                                           "guidance": {"attention": {"urgency": "background"}}}})
        self.assertEqual(_last_jev_assessment(self.store, "main", context)["scores"]["time_sensitive"], 0.9)
        self.assertEqual(_last_jev_assessment(self.store, "alternate", context)["scores"]["time_sensitive"], 0.1)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.pipeline.compose_turn")
    @patch("pioneer.pipeline.assess_jev")
    def test_jev_failure_keeps_conversation_and_records_usage(self, jev, compose):
        jev.side_effect = ProviderError("Jev unavailable", usage={"provider": "jev", "model": "jev-test",
                                                                 "input_tokens": 7, "output_tokens": 1})
        compose.return_value = TurnPlan("A pilot is worth considering.", True, None, [], "test", 10, 5, "r")
        outcome = run_turn(self.store, "Should we launch now or pilot?")
        self.assertEqual(outcome.text, "A pilot is worth considering.")
        self.assertIsNone(compose.call_args.kwargs["jev_guidance"])
        self.assertIn("Jev unavailable", self.store.read_object(outcome.commit)["payload"]["jev_error"])
        self.assertEqual([record["provider"] for record in self.store.usage()], ["jev", "openai"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_unprovided_number_blocks_calculation(self, compose):
        case = {"states": {"good": 0.9, "bad": 0.1}, "actions": {"act": {
            "outcomes": {"good": 10, "bad": -10}}}}
        compose.return_value = TurnPlan("I used your figures.", True, json.dumps(case), [],
                                       "openai-test", 10, 5, "resp_4")
        outcome = run_turn(self.store, "Act pays 10 if good and -10 if bad. What should I do?")
        self.assertIsNone(outcome.decision)
        self.assertEqual(outcome.text, "I used your figures.")
        payload = self.store.read_object(outcome.commit)["payload"]
        self.assertEqual(payload["assistant"], outcome.text)
        self.assertIn("Proposed calculation was not verified", "\n".join(outcome.notices))
        self.assertNotIn("decision", payload)
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
        self.assertEqual(outcome.text, "Invest now.")
        self.assertIn("Proposed calculation was not verified", "\n".join(outcome.notices))

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_wait_question_from_prior_turn_still_guards_calculation(self, compose):
        context = {"status": "active", "goal": "Investment timing", "options": ["invest", "hold"],
                   "known": [], "uncertain": ["outcome"], "provisional_view": "Wait for information",
                   "next_questions": []}
        case = {"states": {"good": 0.5, "bad": 0.5}, "actions": {
            "invest": {"outcomes": {"good": 10, "bad": -10}},
            "hold": {"outcomes": {"good": 0, "bad": 0}}}}
        compose.side_effect = [
            TurnPlan("What information could arrive?", True, None, [], "test", 10, 5, "r1", context),
            TurnPlan("I have the payoffs.", True, json.dumps(case), [], "test", 20, 6, "r2", context),
        ]
        run_turn(self.store, "Should I wait before investing?")
        outcome = run_turn(self.store, "Good 50%, bad 50%; invest pays 10 or -10, hold pays 0.")
        self.assertIsNone(outcome.decision)
        self.assertEqual(outcome.text, "I have the payoffs.")
        self.assertIn("Proposed calculation was not verified", "\n".join(outcome.notices))

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_new_goal_cannot_borrow_old_numbers(self, compose):
        old = {"status": "active", "goal": "Old project", "options": ["go", "hold"],
               "known": [], "uncertain": [], "provisional_view": "Unsure", "next_questions": []}
        new = {**old, "goal": "New project"}
        case = {"states": {"good": 0.9, "bad": 0.1}, "actions": {
            "go": {"outcomes": {"good": 10, "bad": -10}},
            "hold": {"outcomes": {"good": 0, "bad": 0}}}}
        compose.side_effect = [
            TurnPlan("Tell me more.", True, None, [], "test", 10, 5, "r1", old),
            TurnPlan("I can calculate.", True, json.dumps(case), [], "test", 20, 6, "r2", new),
        ]
        run_turn(self.store, "For the old project, good is 90%, bad 10%; go pays 10 or -10 and hold pays 0.")
        outcome = run_turn(self.store, "For a new project, should I go?")
        self.assertIsNone(outcome.decision)
        self.assertEqual(outcome.text, "I can calculate.")
        self.assertIn("Proposed calculation was not verified", "\n".join(outcome.notices))

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_chat_sends_recent_turns_with_working_context(self, compose):
        context = {"status": "active", "goal": "Timing", "options": [], "known": [],
                   "uncertain": [], "provisional_view": "Unsure", "next_questions": []}
        for index in range(25):
            self.store.commit("turn", {"user": f"Old user {index}", "assistant": f"Old reply {index}",
                                       "context": context})
        compose.return_value = TurnPlan("Let's continue.", True, None, [], "test", 10, 5, "r3", context)
        run_turn(self.store, "What now?")
        sent = compose.call_args.args[0]
        self.assertEqual(len(sent), 41)
        self.assertEqual(sent[0]["content"], "Old user 5")
        self.assertEqual(compose.call_args.kwargs["context"], context)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_history_review_cites_prior_user_without_rewriting_reply(self, compose):
        older = "We must finish legal review before the public launch."
        source = self.store.commit("turn", {"user": older, "assistant": "Understood."})
        for index in range(20):
            self.store.commit("turn", {"user": f"Unrelated update {index}", "assistant": "Okay."})
        conflict = {"commit": source, "role": "user",
                    "quote": "finish legal review before the public launch",
                    "challenge": "Has legal review finished, or are you changing that condition?"}
        self.review.return_value = HistoryReview(conflict, "test", 14, 6, "history_1")
        reply = "Launching now seems worthwhile."
        compose.return_value = TurnPlan(reply, True, None, [], "test", 10, 5, "r")
        outcome = run_turn(self.store, "Let's do the public launch before legal review is finished.")
        self.assertFalse(compose.call_args.kwargs["history_evidence"])
        self.review.assert_called_once()
        self.assertEqual(self.review.call_args.args[0], reply)
        self.assertIn(source, {item["commit"] for item in self.review.call_args.args[1]})
        self.assertEqual(outcome.text, reply)
        self.assertEqual(outcome.history_challenge, conflict)
        payload = self.store.read_object(outcome.commit)["payload"]
        self.assertEqual(payload["assistant"], reply)
        self.assertEqual(payload["history_challenge"], conflict)
        self.assertEqual([item["purpose"] for item in self.store.usage() if "purpose" in item],
                         ["history_review"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_history_review_cites_prior_assistant_with_correct_role(self, compose):
        source = self.store.commit("turn", {
            "user": "How long might a pilot take?", "assistant": "The pilot might take one week.",
            "provider": "openai"})
        for index in range(20):
            self.store.commit("turn", {"user": f"Unrelated update {index}", "assistant": "Okay."})
        conflict = {"commit": source, "role": "assistant", "quote": "pilot might take one week",
                    "challenge": "What changed your estimate?"}
        self.review.return_value = HistoryReview(conflict, "test", 14, 6, "history_2")
        reply = "I now think it may take a month."
        compose.return_value = TurnPlan(reply, False, None, [], "test", 10, 5, "r")
        outcome = run_turn(self.store, "What did you say earlier about the pilot's duration?")
        self.assertIn(source, {item["commit"] for item in
                               compose.call_args.kwargs["history_evidence"]})
        self.assertEqual(outcome.text, reply)
        self.assertEqual(outcome.history_challenge["role"], "assistant")
        self.assertEqual(outcome.history_challenge["provider"], "openai")
        self.assertEqual(outcome.history_challenge["commit"], source)
        self.assertEqual(self.store.read_object(outcome.commit)["payload"]["assistant"], reply)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_fabricated_history_citation_is_not_shown(self, compose):
        self.store.commit("turn", {"user": "Wait for legal review before launch.", "assistant": "Okay."})
        for index in range(20):
            self.store.commit("turn", {"user": f"Update {index}", "assistant": "Okay."})
        self.review.return_value = HistoryReview(
            {"commit": "f" * 64, "role": "user", "quote": "Wait for legal review",
             "challenge": "Why did you change your mind?"}, "test", 14, 6, "history_3")
        compose.return_value = TurnPlan("Let's examine it.", True, None, [], "test", 10, 5, "r")
        outcome = run_turn(self.store, "Launch before legal review?")
        self.assertEqual(outcome.text, "Let's examine it.")
        self.assertIsNone(outcome.history_challenge)
        self.assertTrue(any("history citation" in item for item in outcome.notices))
        self.assertNotIn("history_challenge", self.store.read_object(outcome.commit)["payload"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_history_quote_cannot_be_attributed_to_wrong_speaker(self, compose):
        source = self.store.commit("turn", {
            "user": "We should wait for legal review.",
            "assistant": "I agree that review matters.", "provider": "openai"})
        self.review.return_value = HistoryReview(
            {"commit": source, "role": "assistant", "quote": "wait for legal review",
             "challenge": "Did you change your view?"}, "test", 9, 4, "history_4")
        reply = "I think launch now is reasonable."
        compose.return_value = TurnPlan(reply, False, None, [], "test", 10, 5, "r")
        outcome = run_turn(self.store, "Should we wait for legal review or launch?")
        self.assertEqual(outcome.text, reply)
        self.assertIsNone(outcome.history_challenge)
        self.assertNotIn("history_challenge", self.store.read_object(outcome.commit)["payload"])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_history_review_failure_preserves_reply_and_records_usage(self, compose):
        self.store.commit("turn", {"user": "We discussed a pilot.", "assistant": "A week may suffice."})
        reply = "The pilot may take longer than a week."
        compose.return_value = TurnPlan(reply, False, None, [], "test", 10, 5, "answer_1")
        self.review.side_effect = ProviderError("Review failed", usage={
            "provider": "openai", "model": "test", "input_tokens": 12,
            "output_tokens": 4, "response_id": "review_bad"})
        outcome = run_turn(self.store, "What now about the pilot?")
        payload = self.store.read_object(outcome.commit)["payload"]
        self.assertEqual(outcome.text, reply)
        self.assertEqual(payload["assistant"], reply)
        self.assertIsNone(outcome.history_challenge)
        self.assertIn("Review failed", payload["history_review_error"])
        self.assertTrue(any("history check" in item for item in outcome.notices))
        self.assertEqual([entry["input_tokens"] for entry in self.store.usage()], [10, 12])

    def test_retrieval_is_bounded_and_stays_on_active_branch(self):
        self.store.create_branch("alternate")
        secret = self.store.commit("turn", {"user": "The alternate launch is confidential.", "assistant": "Okay."})
        self.store.switch("alternate")
        shared = self.store.commit("turn", {"user": "A launch requires legal review.", "assistant": "Okay."})
        for index in range(30):
            self.store.commit("turn", {"user": f"Launch note {index} " + "x" * 800,
                                       "assistant": "Okay."})
        evidence = retrieve_history(self.store, "alternate", "Should we launch before legal review?")
        self.assertIn(shared, {item["commit"] for item in evidence})
        self.assertNotIn(secret, {item["commit"] for item in evidence})
        self.assertLessEqual(len(evidence), MAX_EVIDENCE)
        self.assertLessEqual(sum(len(item["quote"]) for item in evidence), MAX_TOTAL_CHARS)

    def test_history_excerpt_includes_relevant_end_of_long_statement(self):
        source_text = "Launch idea. " + "Background details. " * 100 + "Legal review must finish before launch."
        source = self.store.commit("turn", {"user": source_text, "assistant": "Okay."})
        for index in range(20):
            self.store.commit("turn", {"user": f"Update {index}", "assistant": "Okay."})
        evidence = retrieve_history(self.store, "main", "Has legal review finished before launch?")
        item = next(item for item in evidence if item["commit"] == source)
        self.assertIn("Legal review must finish", item["quote"])
        self.assertIn(item["quote"], source_text)

    def test_explicit_recall_can_reach_earliest_branch_turn(self):
        source = self.store.commit("turn", {"user": "My first principle was to preserve reversibility.",
                                           "assistant": "Understood."})
        for index in range(20):
            self.store.commit("turn", {"user": f"Unrelated update {index}", "assistant": "Okay."})
        evidence = retrieve_history(self.store, "main", "What did I say at the beginning?")
        self.assertEqual(evidence[0]["commit"], source)

    def test_recent_context_has_character_budget_and_complete_pairs(self):
        first = self.store.commit("turn", {"user": "Earlier launch budget was 97.",
                                           "assistant": "A" * (MAX_RECENT_CHARS // 2)})
        self.store.commit("turn", {"user": "Current plan is a pilot.",
                                   "assistant": "B" * (MAX_RECENT_CHARS // 2)})
        messages, count = recent_messages(self.store, "main")
        self.assertEqual(count, 1)
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"])
        self.assertEqual(messages[0]["content"], "Current plan is a pilot.")
        self.assertLessEqual(sum(len(item["content"]) for item in messages), MAX_RECENT_CHARS)
        evidence = retrieve_history(self.store, "main", "What was the earlier launch budget?", recent_turns=count)
        self.assertEqual(evidence[0]["commit"], first)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_omitted_oversize_turn_does_not_supply_decision_numbers(self, compose):
        self.store.commit("turn", {"user": "For launch, good is 90% and bad 10%.",
                                   "assistant": "A" * (MAX_RECENT_CHARS + 1)})
        case = {"states": {"good": 0.9, "bad": 0.1}, "actions": {
            "launch": {"outcomes": {"good": 5, "bad": -5}},
            "hold": {"outcomes": {"good": 0, "bad": 0}}}}
        compose.return_value = TurnPlan("Here is the result.", True, json.dumps(case), [],
                                        "test", 10, 5, "r")
        current = "Should we launch? Launch pays 5 if good, -5 if bad; hold pays 0."
        outcome = run_turn(self.store, current)
        self.assertEqual(compose.call_args.args[0], [{"role": "user", "content": current}])
        self.assertFalse(compose.call_args.kwargs["history_evidence"])
        self.assertTrue(self.review.call_args.args[1])
        self.assertIsNone(outcome.decision)
        self.assertEqual(outcome.text, "Here is the result.")
        self.assertIn("Proposed calculation was not verified", "\n".join(outcome.notices))

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_retrieved_old_numbers_cannot_authorize_a_calculation(self, compose):
        self.store.commit("turn", {"user": "For launch, good is 90%, bad 10%; launch pays 10 or -10.",
                                   "assistant": "Understood."})
        for index in range(20):
            self.store.commit("turn", {"user": f"Update {index}", "assistant": "Okay."})
        case = {"states": {"good": 0.9, "bad": 0.1}, "actions": {
            "launch": {"outcomes": {"good": 10, "bad": -10}},
            "hold": {"outcomes": {"good": 0, "bad": 0}}}}
        compose.return_value = TurnPlan("Here is the result.", True, json.dumps(case), [],
                                        "test", 10, 5, "r")
        outcome = run_turn(self.store, "Should we launch now?")
        self.assertFalse(compose.call_args.kwargs["history_evidence"])
        self.assertTrue(self.review.call_args.args[1])
        self.assertIsNone(outcome.decision)
        self.assertEqual(outcome.text, "Here is the result.")
        self.assertIn("Proposed calculation was not verified", "\n".join(outcome.notices))

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "", "OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_failed_openai_turn_still_records_returned_usage(self, post):
        post.return_value = {"id": "resp_bad", "model": "test-model", "output": [],
                             "usage": {"input_tokens": 18, "output_tokens": 2}}
        with self.assertRaises(ProviderError):
            run_turn(self.store, "Should I launch?")
        self.assertEqual(len(self.store.usage()), 1)
        self.assertEqual(self.store.usage()[0]["input_tokens"], 18)
        self.assertEqual(self.store.log()[0][1]["kind"], "note")

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_provisional_reply_is_preserved_and_context_carried_forward(self, compose):
        context = {"status": "active", "goal": "Launch timing", "options": ["launch", "pilot"],
                   "known": ["Launch is hard to undo"], "uncertain": ["Pilot duration"],
                   "provisional_view": "Pilot first", "next_questions": ["How long would a pilot take?"]}
        compose.side_effect = [
            TurnPlan("A pilot seems safer because you can change course. How long would it take?",
                     True, None, ["Pilot duration"], "openai-test", 10, 5, "resp_6", context),
            TurnPlan("A week sounds short enough to learn before launch.", True, None, [],
                     "openai-test", 14, 6, "resp_7", {**context, "known": context["known"] + ["Pilot takes a week"],
                                                     "next_questions": []}),
        ]
        first = run_turn(self.store, "Should we launch now or pilot?")
        self.assertIn("pilot seems safer", first.text)
        self.assertNotIn("possible states and their probabilities", first.text)
        second = run_turn(self.store, "A pilot would take a week.")
        self.assertEqual(second.text, "A week sounds short enough to learn before launch.")
        self.assertEqual(compose.call_args.kwargs["context"], context)
        self.assertEqual(self.store.read_object(second.commit)["payload"]["context"]["known"][-1],
                         "Pilot takes a week")

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_counterfactual_creates_branch_and_keeps_original(self, compose):
        context = {"status": "active", "goal": "Launch timing", "options": ["launch", "pilot"],
                   "known": [], "uncertain": [], "provisional_view": "Pilot", "next_questions": []}
        compose.side_effect = [
            TurnPlan("A pilot seems reversible.", True, None, [], "openai-test", 10, 5, "resp_8", context),
            TurnPlan("If we launch anyway, we gain speed but accept more downside.", True, None, [],
                     "openai-test", 12, 5, "resp_9", context, True),
        ]
        first = run_turn(self.store, "Should we pilot?")
        second = run_turn(self.store, "What if we launch anyway?")
        self.assertEqual(second.branched_from, "main")
        self.assertTrue(second.branch.startswith("explore-"))
        self.assertEqual(self.store.resolve("main"), first.commit)
        self.assertEqual(self.store.resolve(), second.commit)
        self.assertEqual(len(self.store.messages("main")), 2)
        self.assertEqual(len(self.store.messages(second.branch)), 4)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "", "OPENAI_API_KEY": ""})
    def test_direct_case_is_analyzed_and_saved_without_api_keys(self):
        case = {"states": {"yes": 1}, "actions": {"do": {"outcomes": {"yes": 2}}}}
        outcome = run_turn(self.store, json.dumps(case))
        self.assertEqual(outcome.model, "local")
        self.assertEqual(outcome.decision["recommendation"]["action"], "do")
        self.assertEqual(self.store.usage(), [])

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"})
    @patch("pioneer.pipeline.compose_turn", side_effect=ProviderError("OpenAI unavailable"))
    @patch("pioneer.pipeline.assess_jev")
    def test_jev_usage_survives_later_openai_failure(self, jev, _compose):
        jev.return_value = {"model": "jev-test", "scores": {"decision_request": 0.9},
                               "input_tokens": 30, "output_tokens": 4}
        with self.assertRaises(ProviderError):
            run_turn(self.store, "Should I launch?")
        self.assertEqual(len(self.store.usage()), 1)
        self.assertEqual(self.store.log()[0][1]["kind"], "note")


if __name__ == "__main__":
    unittest.main()
