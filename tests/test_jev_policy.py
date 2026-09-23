"""The optional Jev layer should steer attention, not choose for the user."""

import unittest

from pioneer.jev import make_jev_guidance, select_jev_context, should_consult_jev


ACTIVE = {
    "status": "active",
    "goal": "Public launch timing",
    "options": ["launch", "run a pilot"],
    "known": ["Legal review is incomplete"],
    "uncertain": ["How long a pilot would take"],
    "next_questions": ["How long would the pilot take?"],
}


class JevRoutingTests(unittest.TestCase):
    def test_clear_new_decisions_are_consulted(self):
        for text in (
            "Should we launch next week?",
            "Do you think I should launch next week?",
            "Is buying now a good idea?",
            "I need to choose between a pilot and a full launch.",
            "Would you recommend waiting or acting now?",
        ):
            with self.subTest(text=text):
                self.assertTrue(should_consult_jev(text, None))

    def test_prose_case_with_states_and_payoffs_is_consulted(self):
        text = ("Good 50%, bad 50%. Invest pays 10 or -10; hold pays 0. "
                "Wait delay costs 1 and information costs 0.")
        self.assertTrue(should_consult_jev(text, None))
        self.assertFalse(should_consult_jev("The launch costs 2026 dollars and 2030 hours.", None))

    def test_implicit_consequential_first_person_choices_are_consulted(self):
        for text in (
            "I’m thinking about quitting my job.",
            "We might pilot before launch.",
            "I'm considering signing a contract.",
            "I could buy a house next month.",
        ):
            with self.subTest(text=text):
                self.assertTrue(should_consult_jev(text, None))

    def test_ordinary_first_person_explanations_do_not_trigger_jev(self):
        for text in (
            "I'm thinking about the definition of expected value.",
            "We might explain expected value with an example.",
            "I might buy a sandwich.",
            "What does 'quitting my job' mean?",
        ):
            with self.subTest(text=text):
                self.assertFalse(should_consult_jev(text, None))

    def test_substantive_active_decision_followups_are_consulted(self):
        for text in (
            "The pilot would take a week.",
            "A week.",
            "Actually, legal review is complete.",
            "What if we launch anyway?",
            "Yes.",
        ):
            with self.subTest(text=text):
                self.assertTrue(should_consult_jev(text, ACTIVE))

    def test_ordinary_chat_and_unrelated_questions_are_skipped(self):
        for text, context in (
            ("Hi, how are you?", None),
            ("Can you explain expected value?", None),
            ("What does 'which option' mean?", None),
            ("Thanks.", ACTIVE),
            ("What is a terminal?", ACTIVE),
            ("", ACTIVE),
        ):
            with self.subTest(text=text):
                self.assertFalse(should_consult_jev(text, context))


class JevContextSelectionTests(unittest.TestCase):
    def test_new_explicit_or_implicit_choice_drops_unrelated_context(self):
        for text in (
            "Should I buy a car?",
            "I’m thinking about quitting my job.",
            "Should I move to Chicago?",
        ):
            with self.subTest(text=text):
                self.assertIsNone(select_jev_context(text, ACTIVE))

    def test_same_topic_choice_and_decision_followup_keep_context(self):
        for text in (
            "Should we launch next week?",
            "We might pilot before launch.",
            "Should I wait for legal review?",
            "Should I wait?",
            "Should I move?",
            "A week.",
            "Yes.",
        ):
            with self.subTest(text=text):
                self.assertIs(select_jev_context(text, ACTIVE), ACTIVE)

    def test_inactive_context_is_not_carried_into_jev(self):
        self.assertIsNone(select_jev_context("Should I wait?", None))
        self.assertIsNone(select_jev_context("Should I wait?", {**ACTIVE, "status": "none"}))

    def test_resolved_decision_can_be_reopened_for_a_counterfactual(self):
        resolved = {**ACTIVE, "status": "resolved"}
        self.assertTrue(should_consult_jev("What if we launch anyway?", resolved))
        self.assertIs(select_jev_context("What if we launch anyway?", resolved), resolved)
        guidance = make_jev_guidance({"decision_request": 0.2, "hard_to_reverse": 0.9}, resolved)
        self.assertEqual(guidance["priorities"], ["reversibility"])


class JevGuidanceTests(unittest.TestCase):
    def test_guidance_prioritizes_tension_and_limits_response_cues(self):
        guidance = make_jev_guidance({
            "decision_request": 0.95,
            "time_sensitive": 0.86,
            "hard_to_reverse": 0.78,
            "missing_information": 0.83,
            "assumption_tension": 0.65,
        }, ACTIVE)
        self.assertEqual(guidance["attention"], {
            "assumption_tension": "check",
            "urgency": "focus",
            "reversibility": "focus",
            "information_value": "focus",
        })
        self.assertEqual(guidance["priorities"], ["assumption_tension", "urgency"])
        self.assertEqual(len(guidance["response_cues"]), 2)
        self.assertIn("verified", guidance["response_cues"][0])
        self.assertIn("recent user statements", guidance["response_cues"][0])
        self.assertNotIn("recommendation", guidance)
        self.assertNotIn("probability", guidance)

    def test_below_threshold_scores_do_not_force_a_checklist(self):
        guidance = make_jev_guidance({
            "time_sensitive": 0.54,
            "hard_to_reverse": 0.30,
            "missing_information": 0.20,
        }, ACTIVE)
        self.assertEqual(guidance["priorities"], [])
        self.assertEqual(guidance["response_cues"], [])
        self.assertTrue(all(value == "background" for value in guidance["attention"].values()))

    def test_missing_and_malformed_optional_scores_are_neutral(self):
        guidance = make_jev_guidance({"decision_request": 0.9,
                                      "time_sensitive": float("nan"),
                                      "hard_to_reverse": True}, None)
        self.assertEqual(guidance["priorities"], [])
        self.assertTrue(all(value == "background" for value in guidance["attention"].values()))

    def test_guidance_names_checks_without_asserting_facts(self):
        guidance = make_jev_guidance({
            "time_sensitive": 0.9,
            "hard_to_reverse": 0.85,
            "missing_information": 0.8,
        }, ACTIVE)
        self.assertEqual(guidance["priorities"], ["urgency", "reversibility"])
        cues = " ".join(guidance["response_cues"])
        self.assertIn("Check whether", cues)
        self.assertIn("Identify what", cues)
        self.assertNotIn("You should", cues)
        self.assertNotIn("likely to succeed", cues)

    def test_low_decision_request_suppresses_new_topic_guidance(self):
        signals = {"decision_request": 0.2, "time_sensitive": 0.9,
                   "hard_to_reverse": 0.85, "missing_information": 0.8}
        cold = make_jev_guidance(signals, None)
        self.assertEqual(cold["priorities"], [])
        self.assertTrue(all(value == "background" for value in cold["attention"].values()))
        active = make_jev_guidance(signals, ACTIVE)
        self.assertEqual(active["priorities"], ["urgency", "reversibility"])


if __name__ == "__main__":
    unittest.main()
