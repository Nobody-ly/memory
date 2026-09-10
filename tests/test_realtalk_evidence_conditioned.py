from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.exp1_protocol import REALTALK_PERSONA_SPLITS
from src.experiments.personaemp.client import ChatResult
from src.experiments.realtalk_evidence_conditioned import (
    ACTOR_SYSTEM_TEMPLATE,
    DECISION_SYSTEM_PROMPT,
    EvidenceConditionedConfig,
    actor_prompt,
    build_gate_manifests,
    build_generation_input,
    decision_prompt,
    evidence_ids,
    prepare_ca_dev,
    prepare_formal_cb,
    run,
)
from src.experiments.realtalk_evidence_schemas import (
    DECISION_SCHEMA,
    SELF_DOMAIN_SCHEMA,
    USER_DOMAIN_SCHEMA,
    empty_user_domain,
    normalize_decision,
    normalize_self_domain,
    normalize_user_domain,
    validate_evidence_ids,
)


DATASET = Path(__file__).resolve().parents[1] / "dataset"


def _extract_json_after(prompt: str, marker: str) -> list[str]:
    tail = prompt.split(marker, 1)[1].lstrip()
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(tail)
    return value


def _self_value(evidence_id: str) -> dict:
    return {
        "self_claims": [{
            "value": "The target mentions an ordinary personal detail.",
            "temporal_scope": "observed in the reference sessions",
            "evidence_ids": [evidence_id],
            "confidence": 0.8,
        }],
        "voice": [{
            "observation": "Uses casual conversational English.",
            "evidence_ids": [evidence_id],
            "confidence": 0.7,
        }],
        "social_dispositions": [{
            "observation": "Participates actively in this conversation.",
            "observed_context": "with the source partner",
            "evidence_ids": [evidence_id],
            "confidence": 0.6,
        }],
        "uncertainties": ["Cross-partner transfer is uncertain."],
    }


def _user_value(evidence_id: str) -> dict:
    value = empty_user_domain()
    value["behavior"] = [{
        "value": "The partner participates in casual conversation.",
        "evidence_ids": [evidence_id],
        "confidence": 0.6,
    }]
    value["update_summary"]["added"] = ["casual participation"]
    return value


def _decision_value(evidence_id: str | None = None) -> dict:
    return {
        "scene": {
            "conversation_threads": ["the current everyday topic"],
            "target_current_context": None,
            "partner_current_state": "The partner is continuing the exchange.",
            "relationship_context": "An ongoing casual chat.",
            "uncertainty": "The exact next content is uncertain.",
        },
        "alignment": {
            "self_tendency": "Continue in the target's ordinary voice.",
            "partner_expectation": "A natural continuation.",
            "lambda_trace": 0.35,
            "tradeoff": "Keep the person's voice while staying responsive.",
        },
        "policy": {
            "intent": "Continue the active topic naturally in character.",
            "evidence_ids": [evidence_id] if evidence_id else [],
        },
    }


class FakeBackend:
    model = "deepseek-v4-flash"
    base_url = "https://example.invalid/v1"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.token_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
            "network_attempts": 0, "network_retries": 0,
        }

    def available_models(self):
        return [self.model]

    def chat(
        self, system_prompt, user_prompt, *, temperature, max_tokens,
        top_p=0.9, response_schema=None, enable_thinking=None,
    ):
        self.calls.append({
            "system": system_prompt, "user": user_prompt,
            "schema": response_schema["name"] if response_schema else None,
            "temperature": temperature, "max_tokens": max_tokens,
            "top_p": top_p, "enable_thinking": enable_thinking,
        })
        if max_tokens == 8:
            content = "READY"
        elif response_schema and response_schema["name"] == SELF_DOMAIN_SCHEMA["name"]:
            allowed = _extract_json_after(
                user_prompt, "TARGET-SPEAKER EVIDENCE ID WHITELIST:"
            )
            content = json.dumps(_self_value(allowed[0]))
        elif response_schema and response_schema["name"] == USER_DOMAIN_SCHEMA["name"]:
            allowed = _extract_json_after(
                user_prompt, "CUMULATIVE PARTNER EVIDENCE ID WHITELIST:"
            )
            content = json.dumps(_user_value(allowed[0]))
        elif response_schema and response_schema["name"] == DECISION_SCHEMA["name"]:
            allowed = _extract_json_after(
                user_prompt, "VISIBLE SOURCE ID WHITELIST FOR POLICY EVIDENCE:"
            )
            content = json.dumps(_decision_value(allowed[0] if allowed else None))
        else:
            content = "That sounds interesting. I have been thinking about it too."
        self.token_usage["calls"] += 1
        self.token_usage["network_attempts"] += 1
        return ChatResult(
            content=content, model=self.model, prompt_tokens=10,
            completion_tokens=5, latency_seconds=0.01, attempts=1,
            reasoning_content="", finish_reason="stop", response_id="fake-id",
        )


class EvidenceSchemaTests(unittest.TestCase):
    def test_schemas_are_strict_and_accept_sparse_profiles(self):
        self_value = normalize_self_domain(_self_value("D1:1"))
        user_value = normalize_user_domain(empty_user_domain())
        decision = normalize_decision(_decision_value())
        self.assertEqual(self_value["self_claims"][0]["confidence"], 0.8)
        self.assertEqual(user_value["core"], [])
        self.assertEqual(decision["alignment"]["lambda_trace"], 0.35)

        invalid = _decision_value()
        invalid["alignment"]["score_hack"] = True
        with self.assertRaisesRegex(ValueError, "extra=.*score_hack"):
            normalize_decision(invalid)

    def test_evidence_validation_rejects_wrong_speaker_or_future_ids(self):
        with self.assertRaisesRegex(ValueError, "invalid evidence IDs"):
            validate_evidence_ids(
                normalize_self_domain(_self_value("FUTURE:1")), {"D1:1"}
            )


class EvidenceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev, cls.dev_manifest = prepare_ca_dev(DATASET)
        cls.formal, cls.formal_manifest = prepare_formal_cb(DATASET)

    def test_formal_protocol_matches_reconstructed_counts(self):
        self.assertEqual(self.formal_manifest["targets"], 519)
        self.assertEqual(self.formal_manifest["formal_raw_target_bubbles"], 1076)
        self.assertEqual(self.formal_manifest["source_raw_message_count"], 8944)
        self.assertEqual(self.formal_manifest["source_session_count"], 219)
        self.assertEqual(len(self.formal), 10)
        self.assertEqual(
            [item["speaker"] for item in self.formal],
            [item["speaker"] for item in REALTALK_PERSONA_SPLITS],
        )
        self.assertTrue(all(not point["context_truncated"] for item in self.formal for point in item["points"]))

    def test_development_files_do_not_overlap_any_formal_cb_file(self):
        formal_cb = {split["test_chat"] for split in REALTALK_PERSONA_SPLITS}
        self.assertEqual(sum(len(item["points"]) for item in self.dev), 68)
        self.assertTrue(all(item["current_file"] not in formal_cb for item in self.dev))

    def test_gate_manifests_are_nested_and_exact(self):
        dev = build_gate_manifests(self.dev, "ca-dev")
        formal = build_gate_manifests(self.formal, "cb")
        self.assertEqual({key: len(value) for key, value in dev.items()}, {"6": 6, "24": 24, "68": 68})
        self.assertEqual({key: len(value) for key, value in formal.items()}, {"10": 10, "30": 30, "90": 90, "519": 519})
        self.assertLessEqual(set(dev["6"]), set(dev["24"]))
        self.assertLessEqual(set(formal["30"]), set(formal["90"]))

    def test_prompts_do_not_receive_target_or_future_and_dev_text_is_not_duplicated(self):
        item = self.dev[0]
        point = item["points"][0]
        future = dict(item["current_turns"][-1])
        future["content"] = "FUTURE_SENTINEL"
        target = dict(point["target"])
        target["content"] = "GROUND_TRUTH_SENTINEL"
        safe_point = {**point, "target": target}
        generation_input = build_generation_input(item, safe_point, empty_user_domain())
        self_domain = _self_value(next(iter(evidence_ids(item["reference_turns"], item["speaker"]))))
        decision_text = decision_prompt(generation_input, self_domain)
        actor_text = actor_prompt(generation_input, self_domain, _decision_value())
        joined = decision_text + actor_text
        self.assertNotIn("GROUND_TRUTH_SENTINEL", joined)
        self.assertNotIn("FUTURE_SENTINEL", joined)
        first_reference_content = item["reference_turns"][0]["content"]
        self.assertEqual(decision_text.count(first_reference_content), 1)
        self.assertEqual(actor_text.count(first_reference_content), 1)
        self.assertNotIn('"lambda_trace"', actor_text)
        self.assertNotIn('"alignment"', actor_text)
        self.assertNotIn('"update_summary"', actor_text)

    def test_formal_actor_has_reference_and_current_history_once_each(self):
        item = self.formal[0]
        point = next(point for point in item["points"] if point["context_turns"])
        generation_input = build_generation_input(item, point, empty_user_domain())
        self_domain = _self_value(next(iter(evidence_ids(item["reference_turns"], item["speaker"]))))
        prompt = actor_prompt(generation_input, self_domain, _decision_value())
        self.assertEqual(prompt.count(item["reference_turns"][0]["content"]), 1)
        self.assertEqual(prompt.count(point["context_turns"][-1]["content"]), 1)


class EvidencePipelineTests(unittest.TestCase):
    def test_dev6_end_to_end_uses_fresh_domains_and_all_nonthinking(self):
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dev"
            result = run(EvidenceConditionedConfig(
                dataset_dir=str(DATASET), output_dir=str(output),
                mode="ca-dev", gate=6, fresh=True,
            ), backend)
            self.assertEqual(result["status"], "generation_complete")
            predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(predictions), 6)
            self.assertTrue((output / "GENERATION_COMPLETE").exists())
            self.assertFalse(result["manifest"]["legacy_self_user_decision_reused"])
            self.assertTrue(all(call["enable_thinking"] is False for call in backend.calls))
            self.assertEqual(sum(call["schema"] == SELF_DOMAIN_SCHEMA["name"] for call in backend.calls), 6)
            self.assertEqual(sum(call["schema"] == USER_DOMAIN_SCHEMA["name"] for call in backend.calls), 12)
            self.assertEqual(sum(call["schema"] == DECISION_SCHEMA["name"] for call in backend.calls), 6)
            actor_calls = [
                call for call in backend.calls
                if call["schema"] is None and call["max_tokens"] == 1024
            ]
            self.assertEqual(len(actor_calls), 6)
            self.assertTrue(all(call["max_tokens"] == 1024 for call in actor_calls))
            self.assertTrue(all("GROUND_TRUTH" not in call["user"] for call in backend.calls))

            continuation = FakeBackend()
            result24 = run(EvidenceConditionedConfig(
                dataset_dir=str(DATASET), output_dir=str(output),
                mode="ca-dev", gate=24, resume=True,
            ), continuation)
            self.assertEqual(result24["status"], "generation_complete")
            self.assertEqual(len((output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()), 24)
            self.assertEqual(sum(call["schema"] == DECISION_SCHEMA["name"] for call in continuation.calls), 18)
            self.assertEqual(len([
                call for call in continuation.calls
                if call["schema"] is None and call["max_tokens"] == 1024
            ]), 18)

    def test_actor_prompt_is_identity_driven_and_metric_free(self):
        system = ACTOR_SYSTEM_TEMPLATE.format(speaker="Emi")
        self.assertIn("You are Emi", system)
        for forbidden in ("Reflectiveness", "Grounding", "Intimacy", "Empathy"):
            self.assertNotIn(forbidden, system)
            self.assertNotIn(forbidden, DECISION_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
