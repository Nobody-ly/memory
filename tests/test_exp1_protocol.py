import tempfile
import unittest
from pathlib import Path

from src.experiments.exp1_protocol import (
    build_message_level_points,
    build_profile_corpus,
    merge_consecutive_utterances,
    resolve_chat_roles,
    select_realtalk_splits,
)


def _chat():
    return {
        "name": {"speaker_1": "User", "speaker_2": "Agent"},
        "session_1": [
            {"speaker": "User", "clean_text": "one", "dia_id": "1"},
            {"speaker": "User", "clean_text": "two", "dia_id": "2"},
            {"speaker": "Agent", "clean_text": "reply", "dia_id": "3"},
        ],
        "session_2": [
            {"speaker": "Agent", "clean_text": "older context", "dia_id": "4"},
            {"speaker": "User", "clean_text": "second target", "dia_id": "5"},
        ],
        "session_3": [
            {"speaker": "Agent", "clean_text": "hello", "dia_id": "6"},
            {"speaker": "User", "clean_text": "third target", "dia_id": "7"},
            {"speaker": "Agent", "clean_text": "future", "dia_id": "8"},
        ],
    }


class Exp1ProtocolTests(unittest.TestCase):
    def test_merges_adjacent_bubbles_within_session(self):
        turns = merge_consecutive_utterances(_chat())
        self.assertEqual(turns[0]["content"], "one\ntwo")
        self.assertEqual(turns[0]["dia_ids"], ["1", "2"])

    def test_profile_uses_only_first_ca_sessions(self):
        chat = _chat()
        chat["session_4"] = [
            {"speaker": "User", "clean_text": "future profile leak", "dia_id": "9"}
        ]
        corpus = build_profile_corpus(chat, "User", profile_sessions=3)
        self.assertEqual(
            corpus["sessions"], ["session_1", "session_2", "session_3"]
        )
        self.assertNotIn("future profile leak", corpus["text"])
        self.assertIn("third target", corpus["text"])

    def test_message_targets_roll_real_history_without_future_leakage(self):
        points = build_message_level_points(
            _chat(), "User", test_sessions=3
        )
        self.assertEqual(
            [point["target_message"] for point in points],
            ["one\ntwo", "second target", "third target"],
        )
        self.assertEqual(points[0]["context_text"], "")
        self.assertNotIn("second target", points[1]["context_text"])
        self.assertIn("one\ntwo", points[1]["context_text"])
        self.assertIn("second target", points[2]["context_text"])
        self.assertNotIn("third target", points[2]["context_text"])
        self.assertNotIn("future", points[2]["context_text"])

    def test_context_cap_drops_oldest_complete_turns(self):
        point = build_message_level_points(
            _chat(), "User", test_sessions=3, max_context_chars=30
        )[-1]
        self.assertTrue(point["context_truncated"])
        self.assertIn("hello", point["context_text"])
        self.assertNotIn("one", point["context_text"])

    def test_realtalk_split_selection_requires_both_files(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            (dataset / "Chat_4_Emi_Paola.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                select_realtalk_splits(str(dataset), speaker_filter=["Emi"])
            (dataset / "Chat_1_Emi_Elise.json").write_text("{}", encoding="utf-8")
            splits = select_realtalk_splits(
                str(dataset), speaker_filter=["emi"]
            )
            self.assertEqual(len(splits), 1)
            self.assertEqual(splits[0]["speaker"], "Emi")

    def test_role_metadata_can_recover_declared_user_only(self):
        chat = _chat()
        chat["name"]["speaker_1"] = "Wrong"
        user, agent, warnings = resolve_chat_roles(chat)
        self.assertEqual((user, agent), ("User", "Agent"))
        self.assertEqual(len(warnings), 1)

    def test_missing_declared_agent_is_fatal(self):
        chat = _chat()
        chat["name"]["speaker_2"] = "Wrong"
        with self.assertRaises(ValueError):
            resolve_chat_roles(chat)


if __name__ == "__main__":
    unittest.main()
