import unittest

from src.experiments.exp1_protocol import (
    build_session_boundary_points,
    merge_consecutive_utterances,
    resolve_chat_roles,
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
            {"speaker": "User", "clean_text": "old target", "dia_id": "5"},
        ],
        "session_3": [
            {"speaker": "Agent", "clean_text": "hello", "dia_id": "6"},
            {"speaker": "User", "clean_text": "CURRENT_TARGET", "dia_id": "7"},
            {"speaker": "Agent", "clean_text": "future", "dia_id": "8"},
        ],
    }


class Exp1ProtocolTests(unittest.TestCase):
    def test_merges_adjacent_bubbles_within_session(self):
        turns = merge_consecutive_utterances(_chat())
        self.assertEqual(turns[0]["content"], "one\ntwo")
        self.assertEqual(turns[0]["dia_ids"], ["1", "2"])

    def test_boundary_target_and_profile_are_strictly_causal(self):
        point = build_session_boundary_points(
            _chat(), "User", min_context_sessions=2, context_sessions=1
        )[0]
        self.assertEqual(point["target_message"], "CURRENT_TARGET")
        self.assertEqual(point["completed_sessions"], ["session_1", "session_2"])
        self.assertNotIn("CURRENT_TARGET", point["profile_text"])
        self.assertNotIn("future", point["profile_text"])
        self.assertIn("older context", point["profile_text"])
        self.assertIn("hello", point["context_text"])
        self.assertNotIn("future", point["context_text"])

    def test_context_cap_drops_oldest_complete_turns(self):
        point = build_session_boundary_points(
            _chat(), "User", min_context_sessions=2, context_sessions=None,
            max_context_chars=30,
        )[0]
        self.assertTrue(point["context_truncated"])
        self.assertIn("hello", point["context_text"])
        self.assertNotIn("one", point["context_text"])

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
