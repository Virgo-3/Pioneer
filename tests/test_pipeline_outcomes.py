"""Conversation records one attributable forecast against one desired outcome."""

import tempfile
import unittest
from unittest.mock import patch

from dao.objectives import add_objective
from dao.outcomes import outcome_report, outcome_threads
from dao.pipeline import run_turn
from dao.providers import HistoryReview, TurnPlan
from dao.state import Store


TARGET = {
    "goal": "Pilot adoption", "metric": "active users", "kind": "numeric",
    "desired": 100, "direction": "at_least", "unit": "users",
    "deadline": "Friday", "action": "",
}


def plan(reply, *, objective=None, actual=None, forecast=None):
    return TurnPlan(reply, False, None, [], "openai-test", 10, 5, "response-test",
                    objective=objective, actual=actual, outcome_forecast=forecast)


@patch.dict("os.environ", {"OPENAI_API_KEY": "test", "TYPESAFE_API_KEY": ""})
class ConversationalOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.store.init()
        self.addCleanup(self.temp.cleanup)
        self.review = patch("dao.pipeline.review_history")
        self.review.start().return_value = HistoryReview(None, "openai-test", 3, 2, "review-test")
        self.addCleanup(self.review.stop)

    @patch("dao.pipeline.compose_turn")
    def test_user_forecast_same_turn_target_then_progress_final_and_correction(self, compose):
        user = ("I want at least 100 active users by Friday for the pilot. "
                "I think there is a 70% chance we hit the target.")
        compose.return_value = plan("That is a clear checkpoint.", objective=TARGET, forecast={
            "objective_id": "", "probability": 0.7, "source": "user",
            "quote": "70% chance we hit the target"})
        first = run_turn(self.store, user)
        self.assertEqual(first.text, "That is a clear checkpoint.")
        self.assertIn("outcome_forecast", self.store.read_object(first.commit)["payload"])
        thread = outcome_threads(self.store)[0]
        self.assertEqual((thread["id"], thread["forecasts"][0]["source"]), (first.commit, "user"))
        self.assertIsNone(thread["forecasts"][0]["brier"])

        compose.return_value = plan("Progress is useful evidence.", actual={
            "objective_id": first.commit, "value": 80, "as_of": "Thursday", "note": ""})
        progress = run_turn(self.store, "We have 80 active users on Thursday.")
        self.assertIn("progress reading", "\n".join(progress.notices))
        self.assertIsNone(outcome_threads(self.store)[0]["forecasts"][0]["brier"])

        compose.return_value = plan("We missed the checkpoint.", actual={
            "objective_id": first.commit, "value": 80, "as_of": "Friday", "note": ""})
        final = run_turn(self.store, "We got 80 active users on Friday.")
        self.assertEqual(final.text, "We missed the checkpoint.")
        self.assertAlmostEqual(outcome_threads(self.store)[0]["forecasts"][0]["brier"], 0.49)
        self.assertEqual(outcome_report(self.store)["scored_count"], 1)
        self.assertIn("Brier", "\n".join(final.notices))

        compose.return_value = plan("That changes the outcome.", actual={
            "objective_id": first.commit, "value": 105, "as_of": "Friday", "note": ""})
        corrected = run_turn(self.store, "Correction: we got 105 active users on Friday.")
        self.assertEqual(corrected.text, "That changes the outcome.")
        self.assertAlmostEqual(outcome_threads(self.store)[0]["forecasts"][0]["brier"], 0.09)
        self.assertIn("Brier", "\n".join(corrected.notices))

    @patch("dao.pipeline.compose_turn")
    def test_openai_estimate_is_attributed_to_openai_and_branch_local(self, compose):
        target = add_objective(self.store, TARGET)
        reply = "I estimate a 60% chance of meeting it."
        compose.return_value = plan(reply, forecast={
            "objective_id": target, "probability": 0.6, "source": "openai",
            "quote": "60% chance of meeting it"})
        outcome = run_turn(self.store, "How likely are we to hit our active users target by Friday?")
        self.assertEqual(outcome.text, reply)
        self.assertEqual(outcome_threads(self.store)[0]["forecasts"][0]["source"], "openai")
        self.assertEqual(outcome_threads(self.store)[0]["forecasts"][0]["model"], "openai-test")
        self.store.create_branch("before-estimate", from_ref=target)
        self.assertEqual(outcome_threads(self.store, "before-estimate")[0]["forecasts"], [])

    @patch("dao.pipeline.compose_turn")
    def test_wrong_attribution_or_probability_does_not_create_a_record(self, compose):
        target = add_objective(self.store, TARGET)
        compose.return_value = plan("I can reason about the target.", forecast={
            "objective_id": target, "probability": 0.7, "source": "user", "quote": "70%"})
        first = run_turn(self.store, "What would help our active users target by Friday?")
        self.assertEqual(first.text, "I can reason about the target.")
        self.assertNotIn("outcome_forecast", self.store.read_object(first.commit)["payload"])
        self.assertIn("did not save", "\n".join(first.notices))

        compose.return_value = plan("Maybe 60% is reasonable.", forecast={
            "objective_id": target, "probability": 0.8, "source": "openai", "quote": "60%"})
        second = run_turn(self.store, "How likely is our active users target by Friday?")
        self.assertNotIn("outcome_forecast", self.store.read_object(second.commit)["payload"])
        self.assertEqual(outcome_threads(self.store)[0]["forecasts"], [])

    @patch("dao.pipeline.compose_turn")
    def test_known_result_cannot_be_predicted_after_the_fact(self, compose):
        target = add_objective(self.store, TARGET)
        compose.return_value = plan("The checkpoint was missed.", actual={
            "objective_id": target, "value": 80, "as_of": "Friday", "note": ""})
        run_turn(self.store, "We got 80 active users on Friday.")
        compose.return_value = plan("I'd have said 70%.", forecast={
            "objective_id": target, "probability": 0.7, "source": "openai", "quote": "70%"})
        late = run_turn(self.store, "What probability would you give our active users target by Friday?")
        self.assertNotIn("outcome_forecast", self.store.read_object(late.commit)["payload"])
        self.assertEqual(outcome_report(self.store)["scored_count"], 0)

    @patch("dao.pipeline.compose_turn")
    def test_ambiguous_target_is_not_linked_by_shared_name(self, compose):
        first = add_objective(self.store, TARGET)
        add_objective(self.store, {**TARGET, "desired": 200, "deadline": "Sunday"})
        compose.return_value = plan("Let's clarify the checkpoint.", forecast={
            "objective_id": first, "probability": 0.7, "source": "user",
            "quote": "70% chance"})
        ambiguous = run_turn(self.store, "I think there is a 70% chance for pilot adoption active users.")
        self.assertNotIn("outcome_forecast", self.store.read_object(ambiguous.commit)["payload"])
        self.assertEqual(outcome_report(self.store)["scored_count"], 0)

        explicit = run_turn(self.store, f"I give target {first[:12]} a 70% chance.")
        self.assertIn("outcome_forecast", self.store.read_object(explicit.commit)["payload"])

    @patch("dao.pipeline.compose_turn")
    def test_odds_of_missing_cannot_be_saved_as_odds_of_meeting(self, compose):
        target = add_objective(self.store, TARGET)
        compose.return_value = plan("That sounds concerning.", forecast={
            "objective_id": target, "probability": 0.7, "source": "user", "quote": "70%"})
        user = run_turn(self.store, "I think there's a 70% chance we'll miss the active users target by Friday.")
        self.assertNotIn("outcome_forecast", self.store.read_object(user.commit)["payload"])

        compose.return_value = plan("I estimate a 70% chance we'll miss the active users target.",
                                    forecast={"objective_id": target, "probability": 0.7,
                                              "source": "openai", "quote": "70%"})
        model = run_turn(self.store, "How likely is our active users target by Friday?")
        self.assertNotIn("outcome_forecast", self.store.read_object(model.commit)["payload"])

    @patch("dao.pipeline.compose_turn")
    def test_probability_must_belong_to_the_same_clause_and_speaker(self, compose):
        target = add_objective(self.store, TARGET)
        compose.return_value = plan("Rain is a separate event.", forecast={
            "objective_id": target, "probability": 0.3, "source": "user",
            "quote": "30% chance"})
        user = run_turn(self.store, "I give our active users target by Friday a 70% chance, "
                                    "and rain tomorrow a 30% chance.")
        self.assertNotIn("outcome_forecast", self.store.read_object(user.commit)["payload"])

        compose.return_value = plan("You put it at 70%; I cannot estimate it yet.", forecast={
            "objective_id": target, "probability": 0.7, "source": "openai",
            "quote": "70%"})
        model = run_turn(self.store, "I say 70% for the active users target by Friday. "
                                     "How likely is it in your view?")
        self.assertNotIn("outcome_forecast", self.store.read_object(model.commit)["payload"])

    @patch("dao.pipeline.compose_turn")
    def test_short_answer_to_probability_question_links_one_open_target(self, compose):
        target = add_objective(self.store, TARGET)
        self.store.commit("turn", {"user": "What is our target?",
                                   "assistant": "What chance do you give it?"})
        compose.return_value = plan("That is your estimate.", forecast={
            "objective_id": target, "probability": 0.7, "source": "user", "quote": "70%"})
        answer = run_turn(self.store, "70%")
        self.assertIn("outcome_forecast", self.store.read_object(answer.commit)["payload"])

    @patch("dao.pipeline.compose_turn")
    def test_explicit_timing_correction_retracts_a_final_score(self, compose):
        target = add_objective(self.store, TARGET)
        self.store.commit("note", {"outcome_forecast": {
            "objective_id": target, "probability": 0.7,
            "source": "user", "quote": "70% chance"}})
        compose.return_value = plan("The target was missed.", actual={
            "objective_id": target, "value": 80, "as_of": "Friday", "note": ""})
        run_turn(self.store, "We got 80 active users on Friday.")
        self.assertEqual(outcome_report(self.store)["scored_count"], 1)
        compose.return_value = plan("Thanks for correcting the date.", actual={
            "objective_id": target, "value": 80, "as_of": "Thursday", "note": ""})
        correction = run_turn(self.store, "Correction: we got 80 active users on Thursday, not Friday.")
        actual = self.store.read_object(correction.commit)["payload"]["actual"]
        self.assertIn("replaces_observation_id", actual)
        self.assertEqual(outcome_report(self.store)["progress_count"], 1)
        self.assertEqual(outcome_report(self.store)["scored_count"], 0)

    @patch("dao.pipeline.compose_turn")
    def test_negative_wording_can_mean_meeting_a_negative_target(self, compose):
        binary = add_objective(self.store, {
            "goal": "Project launch", "metric": "Project launch", "kind": "binary",
            "desired": False, "direction": "exact", "unit": "", "deadline": "Friday",
            "action": ""})
        compose.return_value = plan("That is a chance of meeting the no-launch target.", forecast={
            "objective_id": binary, "probability": 0.7, "source": "user",
            "quote": "70% chance the project won't launch by Friday"})
        no_launch = run_turn(self.store, "I give it a 70% chance the project won't launch by Friday.")
        self.assertIn("outcome_forecast", self.store.read_object(no_launch.commit)["payload"])

        compose.return_value = plan("That would miss your no-launch target.", forecast={
            "objective_id": binary, "probability": 0.7, "source": "user",
            "quote": "70% chance we won't meet the project launch target"})
        inverted = run_turn(self.store, "I see a 70% chance we won't meet the project launch target.")
        self.assertNotIn("outcome_forecast", self.store.read_object(inverted.commit)["payload"])

        spending = add_objective(self.store, {**TARGET, "goal": "Control spending",
                                              "metric": "budget spend", "desired": 100,
                                              "direction": "at_most", "unit": "USD"})
        compose.return_value = plan("That is your budget forecast.", forecast={
            "objective_id": spending, "probability": 0.7, "source": "user",
            "quote": "70% chance we don't exceed 100 USD budget spend by Friday"})
        budget = run_turn(self.store, "I give us a 70% chance we don't exceed 100 USD "
                                      "budget spend by Friday.")
        self.assertIn("outcome_forecast", self.store.read_object(budget.commit)["payload"])

    @patch("dao.pipeline.compose_turn")
    def test_accuracy_question_supplies_source_split_to_openai(self, compose):
        target = add_objective(self.store, TARGET)
        self.store.commit("note", {"outcome_forecast": {
            "objective_id": target, "probability": 0.1,
            "source": "user", "quote": "10% chance"}})
        self.store.commit("note", {"outcome_forecast": {
            "objective_id": target, "probability": 0.9,
            "source": "openai", "quote": "90% chance", "model": "test"}})
        from dao.objectives import record_actual
        record_actual(self.store, target, 110, "Friday")
        compose.return_value = plan("I was closer on that result.")
        run_turn(self.store, "How accurate are your forecasts?")
        split = compose.call_args.kwargs["outcome_history"]["linked_forecast_accuracy"]["by_source"]
        self.assertAlmostEqual(split["user"]["mean_brier"], 0.81)
        self.assertAlmostEqual(split["openai"]["mean_brier"], 0.01)

    @patch("dao.pipeline.compose_turn")
    def test_named_older_target_is_in_bounded_conversation_context(self, compose):
        older = add_objective(self.store, TARGET)
        for index in range(6):
            add_objective(self.store, {**TARGET, "goal": f"Project {index} adoption",
                                            "metric": f"Project {index} users"})
        compose.return_value = plan("Let's review that target.")
        run_turn(self.store, "What is the chance of meeting the Pilot adoption target "
                             "for active users by Friday?")
        self.assertEqual(compose.call_args.kwargs["open_outcomes"][0]["id"], older)


if __name__ == "__main__":
    unittest.main()
