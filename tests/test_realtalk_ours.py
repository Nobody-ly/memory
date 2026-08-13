from __future__ import annotations

import json
import signal
import re
import tempfile
import time
import unittest
from pathlib import Path

from src.experiments.exp1_protocol import select_realtalk_splits
from src.experiments.operation_checkpoint import OperationCheckpoint
from src.experiments.personaemp.client import ChatResult
from src.experiments.realtalk_ours import (
    EXPECTED_FULL_TARGETS,
    EXPECTED_MODEL,
    EXPECTED_SPEAKER_TARGETS,
    ModelCallHardTimeout,
    RealTalkOursConfig,
    _action_contract,
    _behavioral_self_domain,
    _behavior_calibration,
    _call_with_hard_timeout,
    _profile_activation_whitelist,
    _validate_decision_context,
    _prepare_dataset,
    _select_even_points_per_session,
    _structured_call,
    _target_spoke_in_session,
    _turns_with_session_boundaries,
    _validate_decision_profile_activation,
    run_realtalk_ours,
)
from src.experiments.realtalk_ours_schemas import (
    SELF_DOMAIN_SCHEMA,
    normalize_alignment,
    normalize_self_domain,
)


def _self_domain(stats: dict) -> dict:
    return {
        "identity_context": {
            "self_descriptions": [], "life_background": [],
            "relationships": [], "recurring_interests": [],
        },
        "communication_signature": {
            "tone": ["casual"], "vocabulary_and_phrasing": ["informal"],
            "information_density": "medium", "typical_message_scale": "short",
            "expression_patterns": ["direct replies"],
        },
        "interaction_policy_prior": {
            "initiative": "medium", "self_disclosure": "sometimes",
            "question_behavior": "sometimes", "topic_continuation": "usually",
            "topic_shift": "sometimes", "advice_behavior": "rarely",
            "response_to_partner_emotion": "warm but concise",
        },
        "affective_social_signature": {
            "emotion_expression": "moderate", "sentiment_style": "casual",
            "introspection_style": "brief", "follow_up_style": "direct",
            "warmth_style": "friendly", "closeness_style": "informal",
        },
        "boundaries_and_uncertainty": {
            "stable_boundaries": [], "uncertain_attributes": [],
        },
        "observable_statistics": stats,
    }


def _decision() -> dict:
    return {
        "situation": {
            "topic": "casual conversation", "partner_move": "continues the exchange",
            "explicit_affect": "", "affect_intensity": "low",
            "support_request": False, "open_question": "", "uncertainty": "medium",
            "partner_has_open_thread": False, "missing_information": "",
            "continuation_value": "none",
        },
        "relevant_user_domain": [],
        "alignment": {
            "orientation": "self-led", "lambda_trace": 0.2,
            "decision_basis": "The target's usual style is the main guide.",
        },
        "next_action": {
            "communicative_intent": "continue naturally", "primary_move": "answer",
            "content_direction": "respond to the visible exchange",
            "self_expression": "use the target's casual voice",
            "partner_adaptation": "match the current topic", "tone": "casual",
            "message_scale": "typical", "question_mode": "none",
            "continuation_move": "none",
        },
    }


class FakeBackend:
    model = EXPECTED_MODEL
    base_url = "https://example.invalid/v1"
    enable_thinking = False

    def __init__(self, malformed_self_once: bool = False) -> None:
        self.calls = []
        self.malformed_self_once = malformed_self_once
        self.token_usage = {key: 0 for key in (
            "prompt_tokens", "completion_tokens", "calls",
            "network_attempts", "network_retries",
        )}

    def available_models(self):
        return [EXPECTED_MODEL]

    def chat(self, system_prompt, user_prompt, *, temperature, max_tokens,
             top_p=0.9, response_schema=None, enable_thinking=None):
        schema_name = response_schema["name"] if response_schema else None
        self.calls.append({
            "system": system_prompt, "user": user_prompt, "schema": schema_name,
            "max_tokens": max_tokens, "enable_thinking": enable_thinking,
        })
        if schema_name == "realtalk_ours_agentic_self_domain_v4":
            if self.malformed_self_once and sum(c["schema"] == schema_name for c in self.calls) == 1:
                content = "not-json"
            else:
                stats = json.loads(user_prompt.split(
                    "DETERMINISTIC OBSERVABLE STATISTICS (copy exactly):\n", 1
                )[1].split("\n\nBuild", 1)[0])
                content = json.dumps(_self_domain(stats))
        elif schema_name == "realtalk_ours_agentic_user_domain_v2":
            partner = re.search(r"PARTNER \(the modeled user\): (.+)", user_prompt).group(1)
            session_block = user_prompt.split("COMPLETE FINISHED SESSION", 1)[1]
            evidence = re.search(
                rf"\[([^\]]+)\] {re.escape(partner)}:", session_block, re.I
            ).group(1)
            content = json.dumps({
                "core": [], "regulation": [], "cognition": [], "identity": [],
                "behavior": [{"value": "Converses casually", "confidence": "low", "evidence_ids": [evidence]}],
                "update_summary": {"added": ["casual conversation"], "revised": [], "removed": [], "uncertainties": []},
            })
        elif schema_name == "realtalk_ours_agentic_decision_v4":
            decision = _decision()
            history = user_prompt.split(
                "REAL CAUSAL HISTORY BEFORE THE TARGET MESSAGE:\n", 1
            )[1].split("\n\nCURRENT SESSION", 1)[0]
            if not history.strip():
                decision["next_action"]["primary_move"] = "open"
            content = json.dumps(decision)
        else:
            content = "READY" if max_tokens in {8, 32} else "Akib: FAKE_GENERATED"
        self.token_usage["calls"] += 1
        self.token_usage["network_attempts"] += 1
        self.token_usage["prompt_tokens"] += 10
        self.token_usage["completion_tokens"] += 5
        return ChatResult(content, self.model, 10, 5, 0.01, 1, "private" if enable_thinking else "")


class FakeLabels:
    def annotate(self, text):
        return {"emotion": "joy" if "FAKE" in text else "sadness",
                "sentiment": "positive" if "FAKE" in text else "negative",
                "intimacy": 0.8 if "FAKE" in text else 0.3}

    def metadata(self):
        return {"provider": "fake"}


class RealTalkOursTests(unittest.TestCase):
    def test_actor_history_preserves_session_boundaries_without_compression(self):
        turns = [
            {"session_id": "session_1", "speaker": "A", "content": "bye"},
            {"session_id": "session_2", "speaker": "B", "content": "hello"},
        ]
        rendered = _turns_with_session_boundaries(turns)
        self.assertEqual(
            rendered,
            "--- session_1 ---\nA: bye\n--- session_2 ---\nB: hello",
        )
        self.assertFalse(_target_spoke_in_session(turns, "A", "session_2"))
        self.assertTrue(_target_spoke_in_session(turns, "A", "session_1"))

    def test_model_call_hard_timeout_interrupts_stalled_operation(self):
        if not hasattr(signal, "SIGALRM"):
            self.skipTest("POSIX hard-timeout signal is unavailable")
        started = time.monotonic()
        with self.assertRaisesRegex(ModelCallHardTimeout, "stalled"):
            _call_with_hard_timeout(
                lambda: (time.sleep(2), None)[1],
                1,
                "stalled",
            )
        self.assertLess(time.monotonic() - started, 1.8)

    def test_resume_requires_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RealTalkOursConfig(
                output_dir=str(Path(directory) / "missing"),
                resume=True,
                compute_local_metrics=False,
            )
            with self.assertRaisesRegex(ValueError, "existing checkpoint"):
                run_realtalk_ours(config, backend=FakeBackend())

    def test_even_session_sampler_uses_positions_only(self):
        points = [
            {"target_session": session, "sample_id": f"{session}-{index}"}
            for session in ("session_1", "session_2", "session_3")
            for index in range(5)
        ]
        selected = _select_even_points_per_session(points, selected_count=2)
        self.assertEqual(
            [point["sample_id"] for point in selected],
            [
                "session_1-0", "session_1-4",
                "session_2-0", "session_2-4",
                "session_3-0", "session_3-4",
            ],
        )

    def test_interior_session_sampler_is_disjoint_from_full_span(self):
        points = [
            {"target_session": session, "sample_id": f"{session}-{index}"}
            for session in ("session_1", "session_2", "session_3")
            for index in range(5)
        ]
        full_span = _select_even_points_per_session(
            points, selected_count=2, position_mode="full-span"
        )
        interior = _select_even_points_per_session(
            points, selected_count=2, position_mode="interior"
        )
        self.assertEqual(
            [point["sample_id"] for point in interior],
            [
                "session_1-1", "session_1-3",
                "session_2-1", "session_2-3",
                "session_3-1", "session_3-3",
            ],
        )
        self.assertTrue(
            {point["sample_id"] for point in full_span}.isdisjoint(
                point["sample_id"] for point in interior
            )
        )

    def test_action_contracts_isolate_primary_moves(self):
        self.assertIn("natural greeting", _action_contract("open"))
        self.assertIn("Do not add a return question", _action_contract("answer"))
        self.assertIn("Only ask one relevant question", _action_contract("follow-up"))
        self.assertIn(
            "exactly one short question",
            _action_contract("answer", "reciprocal-question"),
        )
        self.assertIn(
            "explain motivations",
            _action_contract("self-disclose", "reciprocal-question"),
        )
        self.assertIn("Do not interpret", _action_contract("self-disclose"))
        with self.assertRaisesRegex(ValueError, "unknown primary move"):
            _action_contract("mixed")

    def test_actor_receives_behavioral_self_domain_without_identity_facts(self):
        domain = _self_domain({
            "target_message_count": 1, "mean_characters": 1.0,
            "median_characters": 1.0, "question_rate": 0.0,
            "first_person_rate": 0.0, "median_merged_bubbles": 1.0,
            "reflective_marker_rate": 0.0, "evaluative_opener_rate": 0.0,
            "reflective_marker_rate": 0.0, "evaluative_opener_rate": 0.0,
        })
        view = _behavioral_self_domain(domain)
        self.assertNotIn("identity_context", view)
        self.assertNotIn("boundaries_and_uncertainty", view)
        self.assertIn("communication_signature", view)
        self.assertEqual(
            view["deterministic_behavior_calibration"]["question_tendency"],
            "rare_without_a_direct_need",
        )

    def test_behavior_calibration_uses_observed_rates_and_scale(self):
        value = _behavior_calibration({
            "question_rate": 0.8,
            "reflective_marker_rate": 0.05,
            "evaluative_opener_rate": 0.0,
            "mean_characters": 200.0,
            "median_characters": 300.0,
        })
        self.assertEqual(
            value["question_tendency"],
            "common_when_latest_content_naturally_invites_it",
        )
        self.assertEqual(value["reflective_explanation"], "rare_unless_explicitly_called_for")
        self.assertEqual(value["generic_positive_opener"], "normally_omit")
        self.assertEqual(value["typical_character_guide"], 250.0)
        self.assertEqual(value["extended_character_guide"], 405.0)
        self.assertTrue(value["statistics_available"])

    def test_behavior_calibration_marks_legacy_statistics_unavailable(self):
        value = _behavior_calibration({})
        self.assertFalse(value["statistics_available"])
        self.assertEqual(value["question_tendency"], "unknown")

    def test_profile_activation_whitelist_is_explicit_when_empty(self):
        domain = {
            "core": [], "regulation": [], "cognition": [], "identity": [],
            "behavior": [], "update_summary": {},
        }
        self.assertIn("must be []", _profile_activation_whitelist(domain))
        domain["identity"] = [{"value": "likes basketball"}]
        whitelist = _profile_activation_whitelist(domain)
        self.assertIn('"layer": "identity"', whitelist)
        self.assertIn('"value": "likes basketball"', whitelist)

    def test_empty_history_requires_open_move(self):
        decision = normalize_alignment(_decision())
        with self.assertRaisesRegex(ValueError, "requires open"):
            _validate_decision_context(decision, has_history=False)
        decision["next_action"]["primary_move"] = "open"
        self.assertIs(_validate_decision_context(decision, has_history=False), decision)

    def test_reciprocal_question_requires_supported_open_thread(self):
        decision = normalize_alignment(_decision())
        decision["next_action"]["continuation_move"] = "reciprocal-question"
        normalized = _validate_decision_context(decision, has_history=True)
        self.assertEqual(normalized["next_action"]["continuation_move"], "none")
        decision["situation"].update({
            "partner_has_open_thread": True,
            "missing_information": "the outcome of the partner's plan",
            "continuation_value": "high",
        })
        decision["next_action"]["continuation_move"] = "reciprocal-question"
        self.assertIs(_validate_decision_context(decision, has_history=True), decision)

    def test_closed_thread_derivatives_are_deterministically_normalized(self):
        decision = normalize_alignment(_decision())
        decision["situation"].update({
            "partner_has_open_thread": False,
            "missing_information": "stale detail",
            "continuation_value": "high",
        })
        decision["next_action"]["continuation_move"] = "reciprocal-question"
        normalized = _validate_decision_context(decision, has_history=True)
        self.assertEqual(normalized["situation"]["missing_information"], "")
        self.assertEqual(normalized["situation"]["continuation_value"], "none")
        self.assertEqual(normalized["next_action"]["continuation_move"], "none")

    def test_public_table8_reconstruction_has_expected_519_merged_targets(self):
        config = RealTalkOursConfig(compute_local_metrics=False)
        manifest, prepared = _prepare_dataset(config, select_realtalk_splits(config.dataset_dir))
        self.assertEqual(manifest["total_targets"], EXPECTED_FULL_TARGETS)
        self.assertEqual(manifest["targets_by_speaker"], EXPECTED_SPEAKER_TARGETS)
        self.assertEqual(len(prepared), 10)
        self.assertFalse(manifest["history_compression_enabled"])
        self.assertFalse(manifest["history_truncation_enabled"])
        self.assertTrue(all(not point["context_truncated"] for item in prepared for point in item["points"]))

    def test_strict_schema_and_profile_activation(self):
        value = _self_domain({
            "target_message_count": 1, "mean_characters": 1.0,
            "median_characters": 1.0, "question_rate": 0.0,
            "first_person_rate": 0.0, "median_merged_bubbles": 1.0,
        })
        value["extra"] = True
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            normalize_self_domain(value)
        decision = normalize_alignment(_decision())
        decision["relevant_user_domain"] = [{"layer": "identity", "value": "unknown"}]
        with self.assertRaisesRegex(ValueError, "unknown User Domain"):
            _validate_decision_profile_activation(
                decision,
                {"core": [], "regulation": [], "cognition": [], "identity": [], "behavior": [], "update_summary": {}},
            )

    def test_structured_call_repairs_and_records_thinking_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeBackend(malformed_self_once=True)
            stats = {"target_message_count": 1, "mean_characters": 1.0,
                     "median_characters": 1.0, "question_rate": 0.0,
                     "first_person_rate": 0.0, "median_merged_bubbles": 1.0,
                     "reflective_marker_rate": 0.0,
                     "evaluative_opener_rate": 0.0}
            result = _structured_call(
                checkpoint=OperationCheckpoint(root / "checkpoint.json", "sig"),
                backend=backend, operation_key="self:test", system_prompt="system",
                user_prompt="DETERMINISTIC OBSERVABLE STATISTICS (copy exactly):\n" + json.dumps(stats) + "\n\nBuild",
                schema=SELF_DOMAIN_SCHEMA, normalizer=normalize_self_domain,
                max_tokens=1800, max_attempts=3, raw_audit=root / "raw.jsonl",
                enable_thinking=False,
            )
            self.assertEqual(result["audit"]["logical_attempts"], 2)
            self.assertFalse(result["audit"]["thinking_enabled"])
            self.assertIn("FORMAT REPAIR REQUIRED", backend.calls[1]["user"])

    def test_akib_21_run_updates_user_domain_only_at_session_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config = RealTalkOursConfig(
                output_dir=str(output), speaker_filter=("Akib",),
                max_eval_points_per_speaker=21, compute_local_metrics=False,
            )
            backend = FakeBackend()
            summary = run_realtalk_ours(config, backend=backend)
            self.assertTrue(summary["generation_complete"])
            self.assertEqual(summary["records"], 21)
            user_calls = [c for c in backend.calls if c["schema"] == "realtalk_ours_agentic_user_domain_v2"]
            self.assertEqual(len(user_calls), 1)
            decision_calls = [c for c in backend.calls if c["schema"] == "realtalk_ours_agentic_decision_v4"]
            self.assertEqual(len(decision_calls), 21)
            self.assertTrue(all(c["enable_thinking"] is True for c in decision_calls))
            generation_calls = [c for c in backend.calls if c["max_tokens"] == 300]
            self.assertEqual(len(generation_calls), 21)
            for call in generation_calls:
                lower = call["user"].casefold()
                self.assertNotIn("user domain", lower)
                self.assertNotIn("lambda_trace", lower)
                self.assertNotIn("reflectiveness", lower)
                self.assertNotIn("identity_context", lower)
            predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(not item["user_domain_completed_session_updates"] for item in predictions), 16)
            self.assertEqual(predictions[16]["user_domain_completed_session_updates"], ["session_1"])

    def test_local_metrics_complete_but_pipeline_stays_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RealTalkOursConfig(
                output_dir=str(Path(directory) / "run"), speaker_filter=("Emi",),
                max_eval_points_per_speaker=2,
            )
            summary = run_realtalk_ours(
                config, backend=FakeBackend(), label_evaluator=FakeLabels(),
                bertscore_fn=lambda references, candidates: [0.75] * len(references),
            )
            output = Path(config.output_dir)
            self.assertEqual(summary["local_metrics"]["message_count"], 2)
            self.assertTrue((output / "LOCAL_METRICS_COMPLETE").exists())
            self.assertFalse((output / "PIPELINE_COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
