import tempfile
import unittest
from unittest.mock import patch

from dao.state import Store, StoreError


class BranchIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.head = self.store.init()
        self.store.create_branch("alternate")

    def tearDown(self):
        self.temp.cleanup()

    def test_commit_rejects_same_tip_on_another_branch_and_records_usage(self):
        self.store.switch("alternate")
        with self.assertRaisesRegex(StoreError, "Branch changed during the API call"):
            self.store.commit(
                "turn", {"user": "question", "assistant": "answer"},
                expected_head=self.head, expected_branch="main",
                usage={"provider": "openai", "input_tokens": 3, "output_tokens": 2},
            )

        self.assertEqual(self.store.resolve("main"), self.head)
        self.assertEqual(self.store.resolve("alternate"), self.head)
        entries = self.store.usage()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["provider"], "openai")
        self.assertTrue(entries[0]["orphaned"])
        self.assertNotIn("commit", entries[0])
        self.assertEqual(self.store.verify()["usage_entries"], 1)

    def test_fork_rejects_same_tip_on_another_branch(self):
        self.store.switch("alternate")
        with self.assertRaisesRegex(StoreError, "Branch changed while planning this turn"):
            self.store.fork_and_switch("exploration", self.head, expected_branch="main")

        self.assertEqual(self.store.current_branch(), "alternate")
        self.assertNotIn("exploration", self.store.branches())
        self.store.switch("main")
        self.assertEqual(
            self.store.fork_and_switch("exploration", self.head, expected_branch="main"),
            "main",
        )
        self.assertEqual(self.store.current_branch(), "exploration")

    def test_fork_and_commit_creates_a_populated_branch(self):
        payload = {"user": "what if", "assistant": "try this", "branched_from": "main"}
        commit = self.store.fork_and_commit(
            "exploration", "turn", payload, expected_head=self.head, expected_branch="main",
            usage={"provider": "openai", "input_tokens": 4},
        )

        self.assertEqual(self.store.current_branch(), "exploration")
        self.assertEqual(self.store.resolve("main"), self.head)
        self.assertEqual(self.store.resolve("exploration"), commit)
        self.assertEqual(self.store.read_object(commit)["parent"], self.head)
        self.assertEqual(self.store.read_object(commit)["payload"], payload)
        self.assertEqual(self.store.usage()[0]["branch"], "exploration")
        self.assertEqual(self.store.usage()[0]["commit"], commit)

    def test_fork_and_commit_rejects_same_tip_on_another_branch(self):
        self.store.switch("alternate")
        with self.assertRaisesRegex(StoreError, "Branch changed during the API call"):
            self.store.fork_and_commit(
                "exploration", "turn", {"user": "what if", "assistant": "try this"},
                expected_head=self.head, expected_branch="main", usage={"provider": "openai"},
            )

        self.assertEqual(self.store.current_branch(), "alternate")
        self.assertNotIn("exploration", self.store.branches())
        self.assertTrue(self.store.usage()[0]["orphaned"])
        self.assertNotIn("commit", self.store.usage()[0])

    def test_fork_and_commit_does_not_enter_an_empty_branch_if_usage_fails(self):
        with patch.object(self.store, "_append_usage", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.store.fork_and_commit(
                    "exploration", "turn", {"user": "what if", "assistant": "try this"},
                    expected_head=self.head, expected_branch="main", usage={"provider": "openai"},
                )

        self.assertEqual(self.store.current_branch(), "main")
        self.assertNotIn("exploration", self.store.branches())


if __name__ == "__main__":
    unittest.main()
