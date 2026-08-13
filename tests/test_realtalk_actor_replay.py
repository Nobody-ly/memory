from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.personaemp.client import ChatResult
from src.experiments.realtalk_actor_replay import run
from src.experiments.realtalk_ours import EXPECTED_MODEL


class FakeBackend:
    model = EXPECTED_MODEL
    token_usage = {}

    def chat(self, *args, **kwargs):
        return ChatResult("short reply?", self.model, 1, 1, 0.01, 1, "")


class RealTalkActorReplayTests(unittest.TestCase):
    def test_replays_only_reciprocal_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            dataset = Path("dataset")
            rows = [
                {
                    "result_id": "akib:message_0:session_1:turn_0",
                    "speaker": "Akib",
                    "test_chat": "Chat_8_Akib_Muhhamed.json",
                    "target_session": "session_1",
                    "context_turn_ids": [],
                    "generated_message": "old reciprocal",
                    "situation": {},
                    "next_action": {
                        "primary_move": "open",
                        "continuation_move": "reciprocal-question",
                    },
                },
                {
                    "result_id": "akib:message_1:session_1:turn_2",
                    "speaker": "Akib",
                    "test_chat": "Chat_8_Akib_Muhhamed.json",
                    "target_session": "session_1",
                    "context_turn_ids": ["session_1:turn_0", "session_1:turn_1"],
                    "generated_message": "preserve exactly",
                    "situation": {},
                    "next_action": {
                        "primary_move": "answer",
                        "continuation_move": "none",
                    },
                },
            ]
            (source / "predictions.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            (source / "self_domains.json").write_text(
                json.dumps({"Akib": {
                    "communication_signature": {}, "interaction_policy_prior": {},
                    "affective_social_signature": {}, "observable_statistics": {},
                }}),
                encoding="utf-8",
            )
            (source / "run_manifest.json").write_text(
                json.dumps({"ours_model": EXPECTED_MODEL}), encoding="utf-8"
            )
            with patch(
                "src.experiments.realtalk_actor_replay._backend_from_env",
                return_value=FakeBackend(),
            ):
                manifest = run(source, dataset, root / "output")
            output = [
                json.loads(line)
                for line in (root / "output" / "predictions.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(manifest["replayed_records"], 1)
            self.assertEqual(output[0]["generated_message"], "short reply?")
            self.assertEqual(output[1]["generated_message"], "preserve exactly")


if __name__ == "__main__":
    unittest.main()
