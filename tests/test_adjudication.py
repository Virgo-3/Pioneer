"""The record view must follow branch history and reject invented citations."""

import tempfile
import unittest
from pathlib import Path

from dao.adjudication import (AdjudicationError, add_evidence, case_record, cases,
                              make_finding, open_case)
from dao.pipeline import _case_context
from dao.state import Store


class AdjudicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name))
        self.store.init()

    def test_case_evidence_and_revision_are_append_only(self):
        case = open_case(self.store, "Was the pilot successful?")
        positive = add_evidence(self.store, case, "18 of 20 participants completed it.")
        negative = add_evidence(self.store, case, "Two participants could not sign in.")
        first = make_finding(self.store, case, "supported", "Most participants completed it.",
                             support=[positive[:12]])
        second = make_finding(self.store, case, "disputed",
                              "Completion was high, but access failures remain unresolved.",
                              support=[positive], oppose=[negative], supersedes=first[:12])
        record = case_record(self.store, case[:12])
        self.assertEqual(record["current_finding"]["id"], second)
        self.assertEqual([item["id"] for item in record["findings"]], [first, second])
        self.assertEqual(record["findings"][1]["supersedes"], first)
        self.assertEqual(self.store.verify()["branches"], 1)

    def test_citation_proves_exact_provenance_on_this_branch(self):
        turn = self.store.commit("turn", {"user": "The pilot included 20 people.",
                                           "assistant": "I heard 20 people."})
        case = open_case(self.store, "How many people were included?")
        evidence = add_evidence(self.store, case, "The speaker reported 20 people.",
                                citation={"commit": turn[:12], "role": "user",
                                          "quote": "included 20 people"})
        source = case_record(self.store, case)["evidence"][0]["citation"]
        self.assertEqual(source["commit"], turn)
        self.assertEqual(source["role"], "user")
        self.assertEqual(source["quote"], "included 20 people")
        with self.assertRaisesRegex(AdjudicationError, "exact quote"):
            add_evidence(self.store, case, "Invented quote",
                         citation={"commit": turn, "role": "user", "quote": "21 people"})
        with self.assertRaisesRegex(AdjudicationError, "exact quote"):
            add_evidence(self.store, case, "Wrong speaker",
                         citation={"commit": turn, "role": "assistant",
                                   "quote": "included 20 people"})
        self.assertEqual(case_record(self.store, case)["evidence"][0]["id"], evidence)

    def test_branch_does_not_inherit_later_records(self):
        case = open_case(self.store, "Is it ready?")
        self.store.create_branch("alternate")
        later = add_evidence(self.store, case, "The first branch saw a failure.")
        self.store.switch("alternate")
        self.assertEqual(case_record(self.store, case)["evidence"], [])
        with self.assertRaisesRegex(AdjudicationError, "branch"):
            make_finding(self.store, case, "supported", "This cites another branch.",
                         support=[later])
        own = add_evidence(self.store, case, "The alternate branch has a report.")
        self.assertEqual(case_record(self.store, case)["evidence"][0]["id"], own)
        self.assertEqual(len(cases(self.store)), 1)

    def test_finding_requires_recorded_evidence_and_explicit_revision(self):
        case = open_case(self.store, "Does this claim hold?")
        with self.assertRaisesRegex(AdjudicationError, "supporting evidence"):
            make_finding(self.store, case, "supported", "No source yet.")
        positive = add_evidence(self.store, case, "One observation supports it.")
        first = make_finding(self.store, case, "supported", "The observation supports it.",
                             support=[positive])
        with self.assertRaisesRegex(AdjudicationError, "supersedes"):
            make_finding(self.store, case, "unresolved", "Need another measurement.")
        with self.assertRaisesRegex(AdjudicationError, "same evidence"):
            make_finding(self.store, case, "disputed", "Contradictory use.",
                         support=[positive], oppose=[positive], supersedes=first)

    def test_conversation_receives_bounded_record_context(self):
        case = open_case(self.store, "Should we launch?")
        evidence = add_evidence(self.store, case, "One test passed.")
        finding = make_finding(self.store, case, "unresolved", "A second test is needed.",
                               support=[evidence])
        context = _case_context(self.store, "main")
        self.assertEqual(context[0]["id"], case)
        self.assertEqual(context[0]["evidence"][0]["id"], evidence)
        self.assertEqual(context[0]["current_finding"]["id"], finding)
        self.assertIn("truth", context[0]["caveat"])
