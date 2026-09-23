import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pioneer.decision import DecisionError, analyze
from pioneer.history import MAX_EVIDENCE, MAX_TOTAL_CHARS, retrieve_history
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

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_structured_turn_contract(self, post):
        post.return_value = {"id": "resp_2", "model": "test-model", "output": [
            {"content": [{"type": "output_text", "text": json.dumps({
                "reply": "Tell me the states.", "decision_requested": True,
                "case_json": None, "missing": ["state probabilities"],
                "context": {"status": "active", "goal": "Decide whether to launch", "options": ["launch"],
                            "known": [], "uncertain": ["demand"], "provisional_view": "Try a pilot",
                            "next_questions": ["What would a pilot cost?"]},
                "explore_alternative": False, "history_conflict": None})}]}],
            "usage": {"input_tokens": 20, "output_tokens": 5}}
        plan = compose_turn([{"role": "user", "content": "Should I launch?"}], model="test-model")
        self.assertTrue(plan.decision_requested)
        self.assertEqual(plan.missing, ["state probabilities"])
        self.assertEqual(plan.context["provisional_view"], "Try a pilot")
        self.assertEqual(post.call_args.args[2]["text"]["format"]["type"], "json_schema")
        self.assertIn("history_conflict", post.call_args.args[2]["text"]["format"]["schema"]["required"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_older_history_is_input_data_and_conflict_is_structured(self, post):
        context = {"status": "active", "goal": "Launch", "options": [], "known": [],
                   "uncertain": [], "provisional_view": "Wait", "next_questions": []}
        evidence = [{"commit": "a" * 64, "quote": "Wait for legal review before launch."}]
        post.return_value = {"model": "test", "output": [{"content": [{"type": "output_text", "text": json.dumps({
            "reply": "A launch now may be premature.", "decision_requested": True, "case_json": None,
            "missing": [], "context": context, "explore_alternative": False,
            "history_conflict": {"commit": "a" * 64, "quote": "Wait for legal review",
                                 "challenge": "Has the review finished?"}})}]}]}
        plan = compose_turn([{"role": "user", "content": "Launch now?"}], context=context,
                            history_evidence=evidence)
        payload = post.call_args.args[2]
        self.assertEqual(payload["input"][-1]["content"], "Launch now?")
        self.assertIn("retrieved_older_user_statements", payload["input"][-2]["content"])
        self.assertNotIn("Wait for legal review", payload["instructions"])
        self.assertEqual(plan.history_conflict["commit"], "a" * 64)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("pioneer.providers._post")
    def test_unusable_openai_response_exposes_returned_usage(self, post):
        post.return_value = {"id": "resp_bad", "model": "test-model", "output": [
            {"content": [{"type": "output_text", "text": "not valid JSON"}]}],
            "usage": {"input_tokens": 19, "output_tokens": 4}}
        with self.assertRaises(ProviderError) as caught:
            compose_turn([{"role": "user", "content": "Hello"}], model="test-model")
        self.assertEqual(caught.exception.usage["input_tokens"], 19)
        self.assertEqual(caught.exception.usage["response_id"], "resp_bad")


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
        self.assertTrue(outcome.text.startswith("Here is the comparison."))
        self.assertIn("I would wait", outcome.text)
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
        self.assertIn("can't trace every number", outcome.text)
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
        self.assertIn("Waiting could change the choice", outcome.text)

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
        self.assertIn("Waiting could change the choice", outcome.text)

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
        self.assertIn("can't trace every number", outcome.text)

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
    def test_older_statement_can_ground_a_challenge(self, compose):
        older = "We must finish legal review before the public launch."
        source = self.store.commit("turn", {"user": older, "assistant": "Understood."})
        for index in range(20):
            self.store.commit("turn", {"user": f"Unrelated update {index}", "assistant": "Okay."})
        conflict = {"commit": source, "quote": "finish legal review before the public launch",
                    "challenge": "Has legal review finished, or are you changing that condition?"}
        compose.return_value = TurnPlan("Launching now has a condition to resolve.", True, None, [],
                                        "test", 10, 5, "r", None, False, conflict)
        outcome = run_turn(self.store, "Let's do the public launch before legal review is finished.")
        evidence = compose.call_args.kwargs["history_evidence"]
        self.assertEqual(evidence[0]["commit"], source)
        self.assertIn("Earlier you said", outcome.text)
        self.assertIn(source[:10], outcome.text)
        self.assertIn("Has legal review finished", outcome.text)
        self.assertEqual(self.store.read_object(outcome.commit)["payload"]["history_conflict"], conflict)

    @patch.dict("os.environ", {"TYPESAFE_API_KEY": ""})
    @patch("pioneer.pipeline.compose_turn")
    def test_fabricated_history_citation_is_not_shown(self, compose):
        self.store.commit("turn", {"user": "Wait for legal review before launch.", "assistant": "Okay."})
        for index in range(20):
            self.store.commit("turn", {"user": f"Update {index}", "assistant": "Okay."})
        compose.return_value = TurnPlan("Let's examine it.", True, None, [], "test", 10, 5,
                                        "r", None, False,
                                        {"commit": "f" * 64, "quote": "Wait for legal review",
                                         "challenge": "Why did you change your mind?"})
        outcome = run_turn(self.store, "Launch before legal review?")
        self.assertEqual(outcome.text, "Let's examine it.")
        self.assertNotIn("history_conflict", self.store.read_object(outcome.commit)["payload"])

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
        self.assertTrue(compose.call_args.kwargs["history_evidence"])
        self.assertIsNone(outcome.decision)
        self.assertIn("can't trace every number", outcome.text)

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
