from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.exp1_protocol import select_realtalk_splits
from src.experiments.personaemp.client import ChatResult
from src.experiments.realtalk_ours import RealTalkOursConfig, _prepare_dataset
from src.experiments.realtalk_ours_schemas import empty_user_domain
from src.experiments.realtalk_v14 import (
    ACTOR_USER_TEMPLATE,
    DECISION_USER_TEMPLATE,
    actor_structure_audit,
    _actor_plan_view,
    build_ca_behavior_bank,
    build_progressive_gate_manifest,
    classify_interaction_trigger,
    decision_plan_audit,
    _information_question_count,
    retrieve_ca_behavior_examples,
    run_v14,
    summarize_behavior_bank,
    V14Config,
    _validate_v14_context,
)
from src.experiments.exp1_protocol import stable_hash
from src.experiments.realtalk_v14_schemas import normalize_v14_decision


def _decision() -> dict:
    return {
        "situation": {
            "topic": "weekend plans",
            "latest_partner_act": "question",
            "current_obligation": "answer the partner's question",
            "open_question": True,
            "explicit_affect": False,
            "support_request": False,
            "uncertainty": "low",
        },
        "relevant_user_domain": [],
        "alignment": {
            "orientation": "balanced",
            "lambda_trace": 0.55,
            "decision_basis": "The direct question shapes the response while voice stays stable.",
        },
        "message_plan": {
            "primary_move": "answer",
            "supporting_moves": ["self-disclose"],
            "bubble_count": 2,
            "question_plan": "reciprocal",
            "reflection_depth": "surface",
            "relationship_register": "casual",
            "length_band": "typical",
            "content_direction": "answer and return the same conversational slot",
            "tone": "informal",
        },
    }


def _self_domain() -> dict:
    return {
        "identity_context": {
            "self_descriptions": [],
            "life_background": [],
            "relationships": [],
            "recurring_interests": [],
        },
        "communication_signature": {
            "tone": ["casual"],
            "vocabulary_and_phrasing": ["informal"],
            "information_density": "medium",
            "typical_message_scale": "typical",
            "expression_patterns": ["natural replies"],
        },
        "interaction_policy_prior": {
            "initiative": "moderate",
            "self_disclosure": "moderate",
            "question_behavior": "moderate",
            "topic_continuation": "moderate",
            "topic_shift": "occasional",
            "advice_behavior": "occasional",
            "response_to_partner_emotion": "contextual",
        },
        "affective_social_signature": {
            "emotion_expression": "moderate",
            "sentiment_style": "casual",
            "introspection_style": "brief",
            "follow_up_style": "natural",
            "warmth_style": "friendly",
            "closeness_style": "casual",
        },
        "boundaries_and_uncertainty": {
            "stable_boundaries": [],
            "uncertain_attributes": [],
        },
        "observable_statistics": {
            "target_message_count": 10,
            "mean_characters": 80.0,
            "median_characters": 70.0,
            "question_rate": 0.4,
            "first_person_rate": 0.5,
            "reflective_marker_rate": 0.1,
            "evaluative_opener_rate": 0.1,
            "median_merged_bubbles": 2.0,
        },
    }


class FakeV14Backend:
    model = "deepseek-v4-flash"
    base_url = "https://example.invalid/v1"

    def __init__(self):
        self.calls = []
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
            "network_attempts": 0,
            "network_retries": 0,
        }

    def available_models(self):
        return [self.model]

    def chat(
        self,
        system_prompt,
        user_prompt,
        *,
        temperature,
        max_tokens,
        top_p=0.9,
        response_schema=None,
        enable_thinking=None,
    ):
        self.calls.append({
            "system": system_prompt,
            "user": user_prompt,
            "schema": response_schema["name"] if response_schema else None,
        })
        if response_schema:
            value = _decision()
            trigger = user_prompt.split("CURRENT INTERACTION TRIGGER (deterministic hint): ", 1)[1].splitlines()[0]
            if trigger == "session-opening":
                value["situation"].update({
                    "latest_partner_act": "none",
                    "current_obligation": "open naturally",
                    "open_question": False,
                })
                value["alignment"].update({
                    "orientation": "self-led", "lambda_trace": 0.2,
                })
                value["message_plan"].update({
                    "primary_move": "open",
                    "supporting_moves": [],
                    "question_plan": "none",
                })
            content = json.dumps(value)
        else:
            content = "READY" if max_tokens == 8 else "One natural bubble\nA second bubble?"
        self.token_usage["calls"] += 1
        self.token_usage["network_attempts"] += 1
        return ChatResult(content, self.model, 10, 5, 0.01, 1, "")


class RealTalkV14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = RealTalkOursConfig(compute_local_metrics=False)
        _, cls.prepared = _prepare_dataset(
            config, select_realtalk_splits(config.dataset_dir)
        )

    def test_gate_lists_are_nested_and_complete(self):
        manifest = build_progressive_gate_manifest(self.prepared)
        previous = set()
        for gate in (6, 18, 30, 60, 120, 519):
            current = set(manifest[str(gate)])
            self.assertEqual(len(current), gate)
            self.assertTrue(previous.issubset(current))
            previous = current

    def test_gate30_covers_each_speaker_session_cell_once(self):
        manifest = build_progressive_gate_manifest(self.prepared)
        selected = set(manifest["30"])
        cells = []
        for item in self.prepared:
            for session in item["points"][0]["test_sessions"]:
                count = sum(
                    f"{item['speaker'].casefold().replace(' ', '_')}:{point['sample_id']}" in selected
                    for point in item["points"]
                    if point["target_session"] == session
                )
                cells.append(count)
        self.assertEqual(cells, [1] * 30)

    def test_ca_bank_preserves_multibubble_text_and_source_ids(self):
        item = self.prepared[0]
        bank = build_ca_behavior_bank(item["profile"]["turns"], item["speaker"])
        multibubble = next(example for example in bank if example["bubble_count"] > 1)
        self.assertIn("\n", multibubble["target_turn"])
        self.assertTrue(multibubble["source_turn_id"].startswith("session_"))
        summary = summarize_behavior_bank(bank)
        self.assertGreater(summary["overall"]["multibubble_rate"], 0)

    def test_retrieval_is_deterministic_and_does_not_accept_target_text(self):
        item = self.prepared[0]
        point = item["points"][5]
        bank = build_ca_behavior_bank(item["profile"]["turns"], item["speaker"])
        first = retrieve_ca_behavior_examples(
            bank,
            point["context_turns"],
            item["speaker"],
            current_session=point["target_session"],
        )
        second = retrieve_ca_behavior_examples(
            bank,
            point["context_turns"],
            item["speaker"],
            current_session=point["target_session"],
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertNotIn("ground_truth", DECISION_USER_TEMPLATE.casefold())

    def test_new_session_trigger_does_not_use_prior_session_tail(self):
        turns = [
            {
                "session_id": "session_1",
                "speaker": "Partner",
                "content": "What do you think?",
            }
        ]
        self.assertEqual(
            classify_interaction_trigger(turns, "Target", "session_2"),
            "session-opening",
        )

    def test_decision_schema_accepts_bundle_and_rejects_questionless_followup(self):
        normalized = normalize_v14_decision(_decision())
        self.assertEqual(normalized["message_plan"]["bubble_count"], 2)
        invalid = _decision()
        invalid["message_plan"]["primary_move"] = "follow-up"
        invalid["message_plan"]["question_plan"] = "none"
        with self.assertRaisesRegex(ValueError, "follow-up or clarify"):
            normalize_v14_decision(invalid)

    def test_followup_question_can_support_an_acknowledgement(self):
        value = _decision()
        value["message_plan"].update({
            "primary_move": "acknowledge",
            "question_plan": "follow-up",
        })
        normalized = normalize_v14_decision(value)
        self.assertEqual(normalized["message_plan"]["question_plan"], "follow-up")

    def test_actor_plan_has_one_question_control_and_no_free_direction(self):
        view = _actor_plan_view(_decision()["message_plan"])
        self.assertEqual(view["question_plan"], "reciprocal")
        self.assertNotIn("content_direction", view)
        self.assertNotIn("reciprocal-question", view["supporting_moves"])

    def test_structured_question_plan_remains_authoritative_over_free_text(self):
        invalid = _decision()
        invalid["message_plan"].update({
            "supporting_moves": ["acknowledge"],
            "question_plan": "none",
            "content_direction": "Acknowledge the point and end with a question.",
        })
        normalized = _validate_v14_context(
            normalize_v14_decision(invalid),
            trigger="after-partner-statement",
            has_history=True,
        )
        audit = decision_plan_audit(normalized["message_plan"])
        self.assertTrue(audit["direction_question_conflict"])
        self.assertTrue(audit["question_plan_is_authoritative"])

    def test_actor_prompt_has_no_metrics_lambda_or_full_user_domain(self):
        lower = ACTOR_USER_TEMPLATE.casefold()
        self.assertNotIn("lambda", lower)
        self.assertNotIn("reflectiveness", lower)
        self.assertNotIn("grounding", lower)
        self.assertNotIn("five-layer user domain", lower)
        self.assertIn("relevant partner facts", lower)

    def test_actor_structure_audit_reports_without_rewriting(self):
        audit = actor_structure_audit("First bubble\nSecond bubble?", _decision()["message_plan"])
        self.assertTrue(audit["bubble_count_match"])
        self.assertTrue(audit["question_permission_match"])

    def test_question_audit_ignores_rhetorical_tag_and_rejects_two_questions(self):
        self.assertEqual(_information_question_count("That's fair, you know?"), 0)
        self.assertEqual(_information_question_count("Where? Why?"), 2)
        audit = actor_structure_audit("Where? Why?", _decision()["message_plan"])
        self.assertFalse(audit["question_permission_match"])

    def test_gate6_replay_freezes_v9_upstream_and_completes(self):
        self_domains = {item["speaker"]: _self_domain() for item in self.prepared}
        rows = []
        for item in self.prepared:
            speaker_id = item["speaker"].casefold().replace(" ", "_")
            for point in item["points"]:
                rows.append({
                    "result_id": f"{speaker_id}:{point['sample_id']}",
                    "speaker": item["speaker"],
                    "partner": item["partner"],
                    "train_chat": item["split"]["train_chat"],
                    "test_chat": item["split"]["test_chat"],
                    "profile_sessions": list(item["profile"]["sessions"]),
                    "test_sessions": list(point["test_sessions"]),
                    "target_session": point["target_session"],
                    "message_level_index": point["message_level_index"],
                    "target_turn_id": point["target"]["turn_id"],
                    "context_turn_ids": [turn["turn_id"] for turn in point["context_turns"]],
                    "context_hash": point["history_hash"],
                    "context_truncated": False,
                    "ground_truth": point["target_message"],
                    "generated_message": "V9 frozen output",
                    "self_domain_hash": stable_hash(self_domains[item["speaker"]]),
                    "user_domain": empty_user_domain(),
                    "user_domain_completed_session_updates": [],
                })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "v9_predictions.jsonl"
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            domains = root / "self_domains.json"
            domains.write_text(json.dumps(self_domains), encoding="utf-8")
            output = root / "v14"
            backend = FakeV14Backend()
            summary = run_v14(
                V14Config(
                    dataset_dir="dataset",
                    v9_predictions=str(predictions),
                    v9_self_domains=str(domains),
                    output_dir=str(output),
                    gate=6,
                    fresh=True,
                    enforce_canonical_v9=False,
                ),
                backend=backend,
            )
            self.assertTrue(summary["generation_complete"])
            self.assertEqual(summary["records"], 6)
            replayed = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(row["v9_generated_message"] == "V9 frozen output" for row in replayed))
            actor_calls = [call for call in backend.calls if call["schema"] is None and "PRIVATE TURN PLAN" in call["user"]]
            self.assertEqual(len(actor_calls), 6)
            self.assertTrue(all("V9 frozen output" not in call["user"] for call in actor_calls))
            self.assertTrue(all("target_turn" not in call["user"] for call in actor_calls))
            self.assertTrue(all("identity_context" not in call["user"] for call in actor_calls))


if __name__ == "__main__":
    unittest.main()
