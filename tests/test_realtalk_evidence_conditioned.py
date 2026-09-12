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
    select_result_ids,
    tag_source,
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
        "situation": {
            "partner_act": "statement",
            "current_topic": "the current everyday topic",
            "conversational_obligation": "react",
            "explicit_request": "",
            "uncertainty": "low",
        },
        "alignment": {
            "orientation": "balanced",
            "lambda_trace": 0.35,
            "basis": "Keep the person's voice while staying responsive.",
            "affected_dimensions": ["tone"],
        },
        "response_guidance": {
            "must_address": ["Continue the active topic naturally in character."],
            "optional_elements": [],
            "avoid": ["Do not turn this into a generic advice response."],
            "question_permission": "none",
            "self_disclosure_permission": "allowed_if_natural",
            "reflection_permission": "allowed_if_supported",
            "length_preference": "short",
        },
        "evidence_ids": [evidence_id] if evidence_id else [],
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


class InvalidStructuredBackend(FakeBackend):
    def chat(self, *args, **kwargs):
        if kwargs.get("max_tokens") == 8:
            return super().chat(*args, **kwargs)
        return ChatResult(
            content="not valid json", model=self.model, prompt_tokens=10,
            completion_tokens=3, latency_seconds=0.01, attempts=1,
            reasoning_content="", finish_reason="stop", response_id="bad-id",
        )


class EvidenceSchemaTests(unittest.TestCase):
    def test_schemas_are_strict_and_accept_sparse_profiles(self):
        self_value = normalize_self_domain(_self_value("Chat.json::session_1:turn_0"))
        user_value = normalize_user_domain(empty_user_domain())
        decision = normalize_decision(_decision_value())
        self.assertEqual(self_value["self_claims"][0]["confidence"], 0.8)
        self.assertEqual(user_value["core"], [])
        self.assertEqual(decision["alignment"]["lambda_trace"], 0.35)

        invalid = _decision_value()
        invalid["alignment"]["score_hack"] = True
        with self.assertRaisesRegex(ValueError, "extra=.*score_hack"):
            normalize_decision(invalid)

        too_many = _self_value("Chat.json::session_1:turn_0")
        too_many["self_claims"] *= 13
        with self.assertRaisesRegex(ValueError, "self_claims exceeds maxItems"):
            normalize_self_domain(too_many)

        too_much_evidence = _self_value("Chat.json::session_1:turn_0")
        too_much_evidence["voice"][0]["evidence_ids"] = [f"id-{i}" for i in range(5)]
        with self.assertRaisesRegex(ValueError, "evidence_ids exceeds maxItems=4"):
            normalize_self_domain(too_much_evidence)

    def test_evidence_validation_rejects_wrong_speaker_or_future_ids(self):
        with self.assertRaisesRegex(ValueError, "invalid evidence IDs"):
            validate_evidence_ids(
                normalize_self_domain(_self_value("future.json::session_1:turn_0")),
                {"Chat.json::session_1:turn_0"},
            )

    def test_v16_response_guidance_is_structural_and_enum_checked(self):
        value = normalize_decision(_decision_value())
        self.assertEqual(value["response_guidance"]["length_preference"], "short")
        legacy = _decision_value()
        legacy["turn_plan"] = {}
        with self.assertRaisesRegex(ValueError, "extra=.*turn_plan"):
            normalize_decision(legacy)
        invalid_dimension = _decision_value()
        invalid_dimension["alignment"]["affected_dimensions"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "must be one of"):
            normalize_decision(invalid_dimension)


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

    def test_contiguous_window_matches_the_frozen_v9_hard_segment(self):
        formal = build_gate_manifests(self.formal, "cb")
        selected = select_result_ids(formal, 519, "cb", 373, 60)
        self.assertEqual(len(selected), 60)
        self.assertEqual(selected[0], "cb:Vanessa:session_2:turn_15")
        self.assertEqual(selected[5], "cb:Vanessa:session_2:turn_25")
        self.assertEqual(selected[34], "cb:Vanessa:session_3:turn_4")
        self.assertEqual(selected[-1], "cb:Vanessa:session_3:turn_54")
        self.assertTrue(all(result_id.startswith("cb:Vanessa:") for result_id in selected))

        with self.assertRaisesRegex(ValueError, "requires cb mode"):
            select_result_ids(build_gate_manifests(self.dev, "ca-dev"), 68, "ca-dev", 1, 6)

    def test_prompts_do_not_receive_target_or_future_and_dev_text_is_not_duplicated(self):
        item = self.dev[0]
        point = item["points"][0]
        target = dict(point["target"])
        target["content"] = "GROUND_TRUTH_SENTINEL"
        safe_point = {**point, "target": target}
        generation_input = build_generation_input(item, safe_point, empty_user_domain())
        self_domain = _self_value(next(iter(evidence_ids(item["reference_turns"], item["speaker"]))))
        decision_text = decision_prompt(generation_input, self_domain)
        actor_text = actor_prompt(generation_input, self_domain, _decision_value())
        joined = decision_text + actor_text
        self.assertNotIn("GROUND_TRUTH_SENTINEL", joined)
        self.assertNotIn(point["target"]["source_id"], joined)
        first_reference_content = item["reference_turns"][0]["content"]
        self.assertEqual(decision_text.count(first_reference_content), 1)
        self.assertEqual(actor_text.count(first_reference_content), 1)
        self.assertNotIn('"lambda_trace"', actor_text)
        self.assertNotIn('"update_summary"', actor_text)

    def test_new_session_position_is_visible_without_target_text(self):
        item = self.dev[0]
        point = item["points"][0]
        generation_input = build_generation_input(item, point, empty_user_domain())
        self.assertEqual(generation_input["conversation_position"], {
            "target_session": "session_3",
            "observed_turns_in_target_session": 0,
            "starts_new_session": True,
        })
        self_domain = _self_value(next(iter(evidence_ids(
            item["reference_turns"], item["speaker"]
        ))))
        decision_text = decision_prompt(generation_input, self_domain)
        actor_text = actor_prompt(generation_input, self_domain, _decision_value())
        self.assertIn('"starts_new_session": true', decision_text)
        self.assertIn('"starts_new_session": true', actor_text)
        self.assertNotIn(point["target_message"], decision_text + actor_text)

    def test_evidence_ids_are_namespaced_by_file(self):
        raw = [{
            "turn_id": "session_1:turn_0", "session_id": "session_1",
            "speaker": "Emi", "content": "hello", "dia_ids": ["D1:1"],
        }]
        left = tag_source(raw, "Chat_1.json")
        right = tag_source(raw, "Chat_4.json")
        self.assertEqual(evidence_ids(left), {"Chat_1.json::session_1:turn_0"})
        self.assertEqual(evidence_ids(right), {"Chat_4.json::session_1:turn_0"})
        self.assertTrue(evidence_ids(left).isdisjoint(evidence_ids(right)))

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

    def test_exhausted_operation_writes_incomplete_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            with self.assertRaises(ValueError):
                run(EvidenceConditionedConfig(
                    dataset_dir=str(DATASET), output_dir=str(output),
                    mode="ca-dev", gate=6, fresh=True,
                ), InvalidStructuredBackend())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            unresolved = [
                json.loads(line) for line in
                (output / "unresolved_errors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(manifest["unresolved_records"], 1)
            self.assertEqual(unresolved[0]["operation_key"], "self:emi")
            self.assertFalse((output / "GENERATION_COMPLETE").exists())

    def test_profile_source_reuses_validated_domains_without_model_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            run(EvidenceConditionedConfig(
                dataset_dir=str(DATASET), output_dir=str(source),
                mode="ca-dev", gate=6, fresh=True,
            ), FakeBackend())

            backend = FakeBackend()
            output = Path(temporary) / "replay"
            result = run(EvidenceConditionedConfig(
                dataset_dir=str(DATASET), output_dir=str(output),
                mode="ca-dev", gate=6, fresh=True,
                profile_source_dir=str(source),
            ), backend)
            self.assertEqual(result["status"], "generation_complete")
            self.assertTrue(result["manifest"]["self_user_domains_reused"])
            self.assertEqual(
                result["manifest"]["profile_source"]["source_run_signature"],
                json.loads((source / "manifest.json").read_text(encoding="utf-8"))["run_signature"],
            )
            self.assertEqual(sum(
                call["schema"] == SELF_DOMAIN_SCHEMA["name"]
                for call in backend.calls
            ), 0)
            self.assertEqual(sum(
                call["schema"] == USER_DOMAIN_SCHEMA["name"]
                for call in backend.calls
            ), 0)
            self.assertEqual(sum(
                call["schema"] == DECISION_SCHEMA["name"]
                for call in backend.calls
            ), 6)

    def test_actor_prompt_is_identity_driven_and_metric_free(self):
        system = ACTOR_SYSTEM_TEMPLATE.format(speaker="Emi")
        self.assertIn("You are Emi", system)
        self.assertIn("next-utterance prediction task", DECISION_SYSTEM_PROMPT)
        self.assertIn("most likely natural message", system)
        for forbidden in ("Reflectiveness", "Grounding", "Intimacy", "Empathy"):
            self.assertNotIn(forbidden, system)
            self.assertNotIn(forbidden, DECISION_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
