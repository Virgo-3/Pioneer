"""Conversation history is a citation source, not a truth oracle."""

import tempfile
import unittest

from dao.history import MAX_EVIDENCE, MAX_TOTAL_CHARS, retrieve_history, verified_conflict
from dao.state import Store


class RetrospectiveHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.store.init()

    def tearDown(self):
        self.temp.cleanup()

    def test_both_speakers_have_distinct_provenance_in_one_commit(self):
        commit = self.store.commit("turn", {
            "user": "I want a reversible launch.",
            "assistant": "A reversible pilot would let us learn first.",
            "provider": "openai",
        })
        evidence = retrieve_history(self.store, "main", "What did we say about a reversible launch?")
        by_role = {item["role"]: item for item in evidence}
        self.assertEqual(set(by_role), {"user", "assistant"})
        self.assertEqual(by_role["user"], {
            "commit": commit, "role": "user", "quote": "I want a reversible launch.",
        })
        self.assertEqual(by_role["assistant"], {
            "commit": commit, "role": "assistant",
            "quote": "A reversible pilot would let us learn first.", "provider": "openai",
        })

    def test_challenge_requires_exact_role_commit_and_quote(self):
        commit = self.store.commit("turn", {
            "user": "I favored waiting for a pilot.",
            "assistant": "I favored launching the pilot first.",
            "provider": "openai",
        })
        evidence = retrieve_history(self.store, "main", "What did you say about the pilot?")
        proposal = {"commit": commit, "role": "assistant", "quote": "launching the pilot first",
                    "challenge": "What changed your view?", "provider": "forged"}
        self.assertEqual(verified_conflict(proposal, evidence), {
            "commit": commit, "role": "assistant", "quote": "launching the pilot first",
            "challenge": "What changed your view?", "provider": "openai",
        })
        self.assertIsNone(verified_conflict({**proposal, "role": "user"}, evidence))
        self.assertIsNone(verified_conflict({**proposal, "commit": "f" * 64}, evidence))
        self.assertIsNone(verified_conflict({**proposal, "quote": "launching without a pilot"}, evidence))
        self.assertIsNone(verified_conflict({key: value for key, value in proposal.items()
                                             if key != "role"}, evidence))

        user_proposal = {"commit": commit, "role": "user", "quote": "waiting for a pilot",
                         "challenge": "Do you still favor waiting?"}
        self.assertEqual(verified_conflict(user_proposal, evidence), user_proposal)

    def test_explicit_challenge_and_recall_can_cite_recent_assistant(self):
        self.store.commit("turn", {"user": "What should we do?", "assistant": "Try a pilot."})
        latest = self.store.commit("turn", {
            "user": "Did the launch date come from sales?",
            "assistant": "Launch immediately to gain speed.",
            "provider": "openai",
        })
        self.assertEqual(retrieve_history(self.store, "main", "Could speed help?"), [])
        for query in ("What did you say?", "You claimed speed mattered.",
                      "Challenge that.", "Where did that come from?"):
            with self.subTest(query=query):
                evidence = retrieve_history(self.store, "main", query)
                self.assertTrue(evidence)
                self.assertEqual(evidence[0]["commit"], latest)
                self.assertEqual(evidence[0]["role"], "assistant")

    def test_post_answer_review_can_include_recent_without_recall_cue(self):
        source = self.store.commit("turn", {
            "user": "We can run a pilot.", "assistant": "A pilot can preserve reversibility.",
            "provider": "openai",
        })
        evidence = retrieve_history(self.store, "main", "reversibility", include_recent=True)
        self.assertIn((source, "assistant"), {(item["commit"], item["role"]) for item in evidence})

    def test_post_answer_review_skips_unrelated_small_talk(self):
        self.store.commit("turn", {"user": "Our pilot needs legal review.",
                                   "assistant": "We can discuss the pilot timeline."})
        self.assertEqual(retrieve_history(self.store, "main", "Thanks.", include_recent=True), [])
        self.assertEqual(retrieve_history(self.store, "main", "Thanks. What did you say?",
                                          include_recent=True, followup_text="Thanks."), [])

    def test_short_user_followup_recovers_recent_ai_claim_without_shared_words(self):
        source = self.store.commit("turn", {"user": "Should we launch?",
                                             "assistant": "We should wait.", "provider": "openai"})
        query = "And now? Do it."
        self.assertEqual(retrieve_history(self.store, "main", query, include_recent=True), [])
        for user_text in ("And now?", "Do it.", "Go ahead.", "Let's do it.", "Let’s do it."):
            with self.subTest(user_text=user_text):
                evidence = retrieve_history(self.store, "main", query, include_recent=True,
                                            followup_text=user_text)
                self.assertEqual(evidence[0], {"commit": source, "role": "assistant",
                                               "quote": "We should wait.", "provider": "openai"})
                self.assertIn({"commit": source, "role": "user", "quote": "Should we launch?"},
                              evidence)

    def test_saved_target_recall_finds_older_goal_wording(self):
        source = self.store.commit("turn", {
            "user": "Our goal is 100 paid users by Friday.",
            "assistant": "That gives us a measurable checkpoint.", "provider": "openai",
        })
        for index in range(20):
            self.store.commit("turn", {"user": f"Unrelated note {index}",
                                       "assistant": "Okay."})
        evidence = retrieve_history(self.store, "main", "Did you save my target?")
        self.assertIn({"commit": source, "role": "user",
                       "quote": "Our goal is 100 paid users by Friday."}, evidence)

    def test_older_assistant_and_earliest_user_are_retrievable_with_bounds(self):
        source = self.store.commit("turn", {
            "user": "My first principle was reversibility.",
            "assistant": "I said a reversible pilot protects optionality.",
            "provider": "openai",
        })
        updates = []
        for index in range(25):
            updates.append(self.store.commit("turn", {
                "user": f"Update {index} " + "filler " * 150,
                "assistant": f"Acknowledged {index}.",
            }))
        topic = retrieve_history(self.store, "main", "Does a pilot protect optionality?")
        self.assertIn({"commit": source, "role": "assistant",
                       "quote": "I said a reversible pilot protects optionality.",
                       "provider": "openai"}, topic)
        earliest = retrieve_history(self.store, "main", "What did I say at the beginning?")
        self.assertEqual((earliest[0]["commit"], earliest[0]["role"]), (source, "user"))
        self.assertTrue(set(updates[-20:]) & {item["commit"] for item in earliest})
        self.assertLessEqual(len(earliest), MAX_EVIDENCE)
        self.assertLessEqual(sum(len(item["quote"]) for item in earliest), MAX_TOTAL_CHARS)

    def test_sibling_branch_claims_never_appear(self):
        shared = self.store.commit("turn", {"user": "We share a pilot premise.",
                                            "assistant": "A pilot is possible."})
        self.store.create_branch("alternative", shared)
        main_only = self.store.commit("turn", {"user": "Main asks about launch.",
                                               "assistant": "Main recommends launch."})
        self.store.switch("alternative")
        alternate_only = self.store.commit("turn", {"user": "Alternative asks about launch.",
                                                    "assistant": "Alternative recommends waiting."})
        evidence = retrieve_history(self.store, "alternative", "What did you say about launch?",
                                    include_recent=True)
        cited = {item["commit"] for item in evidence}
        self.assertIn(alternate_only, cited)
        self.assertNotIn(main_only, cited)


if __name__ == "__main__":
    unittest.main()
