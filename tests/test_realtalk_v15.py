from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.exp1_protocol import select_realtalk_splits, stable_hash
from src.experiments.personaemp.client import ChatResult
from src.experiments.realtalk_ours import RealTalkOursConfig, _prepare_dataset
from src.experiments.realtalk_ours_schemas import empty_user_domain
from src.experiments.realtalk_v14 import build_progressive_gate_manifest
from src.experiments.realtalk_v15 import (
    ACTOR_REPAIR_TEMPLATE,
    ACTOR_USER_TEMPLATE,
    CONTROLLER_SYSTEM_PROMPT,
    CONTROLLER_USER_TEMPLATE,
    V15Config,
    actor_structure_audit,
    build_v15_activation_whitelist,
    _fact_ownership_audit,
    _target_owned_history_text,
    _v15_actor_self_domain,
    resolve_v15_profile_activation,
    run_v15,
)
from src.experiments.realtalk_v15_ca_dev import (
    CA_DEV_USER_DOMAIN_SCHEMA,
    V15CaDevConfig,
    _ca_dev_gate_manifest,
    _normalize_ca_dev_evidence_id,
    _prepare_ca_dev,
    run_v15_ca_dev,
)
from src.experiments.realtalk_v15_schemas import (
    DECISION_SCHEMA,
    PARTNER_ACTS,
    normalize_v15_decision,
)


def _decision() -> dict:
    return {
        "situation": {
            "partner_act": "question",
            "conversational_obligation": "respond",
            "topic": "current plans",
            "uncertainty": "low",
        },
        "relevant_user_domain": [],
        "alignment": {
            "orientation": "balanced",
            "lambda_trace": 0.35,
            "adaptation_source": "current-partner-turn",
            "affected_dimensions": ["turn-composition"],
            "decision_basis": "The direct question shapes the turn composition.",
        },
        "turn_plan": {
            "turn_units": [
                {
                    "act": "answer",
                    "content_slot": "answer the current question",
                    "question_target": "",
                },
                {
                    "act": "reciprocal-question",
                    "content_slot": "ask about the partner's current plan",
                    "question_target": "the partner's current plan",
                },
            ],
            "disclosure_depth": "surface",
            "relationship_register": "casual-close",
            "length_band": "typical",
            "tone": "casual",
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


class FakeV15Backend:
    model = "deepseek-v4-flash"
    base_url = "https://example.invalid/v1"

    def __init__(self, fail_first_actor: bool = False):
        self.calls = []
        self.fail_first_actor = fail_first_actor
        self.actor_calls = 0
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
            content = json.dumps(_decision())
        elif max_tokens == 8:
            content = "READY"
        else:
            self.actor_calls += 1
            if self.fail_first_actor and self.actor_calls == 1:
                content = "One combined bubble with a question?"
            else:
                content = "I am keeping it simple.\nHow about your plan?"
        self.token_usage["calls"] += 1
        self.token_usage["network_attempts"] += 1
        return ChatResult(content, self.model, 10, 5, 0.01, 1, "")


class FakeV15CaDevBackend(FakeV15Backend):
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
        if response_schema and "self_domain" in response_schema["name"]:
            marker = "DETERMINISTIC OBSERVABLE STATISTICS (copy exactly):\n"
            stats_text = user_prompt.split(marker, 1)[1].split(
                "\n\nBuild the fixed", 1
            )[0]
            value = _self_domain()
            value["observable_statistics"] = json.loads(stats_text)
            self.calls.append({"system": system_prompt, "user": user_prompt, "schema": response_schema["name"]})
            return ChatResult(json.dumps(value), self.model, 10, 5, 0.01, 1, "")
        if response_schema and "user_domain" in response_schema["name"]:
            self.calls.append({"system": system_prompt, "user": user_prompt, "schema": response_schema["name"]})
            return ChatResult(json.dumps(empty_user_domain()), self.model, 10, 5, 0.01, 1, "")
        return super().chat(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_schema=response_schema,
            enable_thinking=enable_thinking,
        )


class RealTalkV15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = RealTalkOursConfig(compute_local_metrics=False)
        _, cls.prepared = _prepare_dataset(
            config, select_realtalk_splits(config.dataset_dir)
        )

    def test_schema_accepts_turn_units_and_question_permissions(self):
        normalized = normalize_v15_decision(_decision())
        self.assertEqual(len(normalized["turn_plan"]["turn_units"]), 2)

        invalid = _decision()
        invalid["turn_plan"]["turn_units"][0]["question_target"] = "the plan"
        with self.assertRaisesRegex(ValueError, "must have empty question_target"):
            normalize_v15_decision(invalid)

        invalid = _decision()
        invalid["turn_plan"]["turn_units"][1]["question_target"] = ""
        with self.assertRaisesRegex(ValueError, "requires question_target"):
            normalize_v15_decision(invalid)

        invalid = _decision()
        invalid["turn_plan"]["turn_units"][1]["content_slot"] = (
            "That sounds like an interesting plan."
        )
        with self.assertRaisesRegex(ValueError, "must describe exactly one"):
            normalize_v15_decision(invalid)

        invalid = _decision()
        invalid["turn_plan"]["turn_units"][0]["content_slot"] = (
            "What movie were you watching earlier?"
        )
        with self.assertRaisesRegex(ValueError, "must not describe"):
            normalize_v15_decision(invalid)

        rhetorical = _decision()
        rhetorical["turn_plan"]["turn_units"][0]["content_slot"] = (
            "Interactive stories sound interesting, you know?"
        )
        normalize_v15_decision(rhetorical)

        compound = _decision()
        compound["turn_plan"]["turn_units"][1]["content_slot"] = (
            "ask what her name is and if she is from Miami"
        )
        with self.assertRaisesRegex(ValueError, "combines multiple"):
            normalize_v15_decision(compound)

    def test_ca_dev_user_domain_schema_is_compact(self):
        properties = CA_DEV_USER_DOMAIN_SCHEMA["schema"]["properties"]
        for layer in ("core", "regulation", "cognition", "identity", "behavior"):
            self.assertEqual(properties[layer]["maxItems"], 3)
            evidence = properties[layer]["items"]["properties"]["evidence_ids"]
            self.assertEqual(evidence["maxItems"], 4)

        self.assertEqual(
            _normalize_ca_dev_evidence_id("s1:t6"), "session_1:turn_6"
        )
        self.assertEqual(
            _normalize_ca_dev_evidence_id("session_2:turn_17"),
            "session_2:turn_17",
        )
        self.assertEqual(
            _normalize_ca_dev_evidence_id(
                "turn_33", {"session_1:turn_33", "session_1:turn_31"}
            ),
            "session_1:turn_33",
        )
        self.assertEqual(
            _normalize_ca_dev_evidence_id(
                "turn_3", {"session_1:turn_3", "session_2:turn_3"}
            ),
            "turn_3",
        )

    def test_ca_dev_user_domain_prompt_has_exact_evidence_whitelist(self):
        from src.experiments.realtalk_v15_ca_dev import (
            CA_DEV_USER_DOMAIN_SYSTEM_PROMPT,
            CA_DEV_USER_DOMAIN_USER_TEMPLATE,
        )

        self.assertIn("target speaker's turns are context only", CA_DEV_USER_DOMAIN_SYSTEM_PROMPT)
        self.assertIn("ONLY ALLOWED PARTNER EVIDENCE IDS", CA_DEV_USER_DOMAIN_USER_TEMPLATE)
        rendered = CA_DEV_USER_DOMAIN_USER_TEMPLATE.format(
            speaker="Target",
            partner="Partner",
            previous_domain="{}",
            completed_session="session_1:turn_1 | Target: hi",
            allowed_partner_evidence_ids='["session_1:turn_2"]',
        )
        self.assertIn('session_1:turn_2', rendered)

    def test_lambda_trace_names_only_dimensions_it_changes(self):
        invalid = _decision()
        invalid["alignment"]["affected_dimensions"] = []
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            normalize_v15_decision(invalid)

        invalid = _decision()
        invalid["alignment"]["lambda_trace"] = 0
        with self.assertRaisesRegex(ValueError, "requires no affected"):
            normalize_v15_decision(invalid)

        aligned = _decision()
        aligned["alignment"]["lambda_trace"] = 0
        aligned["alignment"]["affected_dimensions"] = []
        normalize_v15_decision(aligned)

    def test_fact_ownership_audit_checks_first_person_statements_not_questions(self):
        latest = {"content": "It is cold here and I am heading out to work."}
        context = [{"speaker": "Target", "content": "I usually stay inside."}]
        transferred = _fact_ownership_audit(
            "It is cold here too, so I'm heading out anyway.", latest, context, "Target"
        )
        self.assertTrue(transferred["warning"])

        partner_question = _fact_ownership_audit(
            "I'm having a quiet day. How is your cold trip to work?",
            latest,
            context,
            "Target",
        )
        self.assertFalse(partner_question["warning"])

        general_comment = _fact_ownership_audit(
            "Data leaks are difficult for companies. I'm hoping it is a small glitch.",
            {"content": "The company is worried about data leaks."},
            context,
            "Target",
        )
        self.assertFalse(general_comment["warning"])

        self_supported = _fact_ownership_audit(
            "I live in New York.",
            {"content": "How is New York?"},
            context,
            "Target",
            {"identity_context": {"life_background": ["Target lives in New York."]}},
        )
        self.assertFalse(self_supported["warning"])

    def test_fact_ownership_audit_normalizes_sensitive_concepts_and_pairs(self):
        context = [{"speaker": "Target", "content": "I live in Los Angeles."}]
        new_york = _fact_ownership_audit(
            "I'm really enjoying New York so far.",
            {"content": "How are you liking NYC?"},
            context,
            "Target",
        )
        self.assertTrue(new_york["warning"])
        self.assertIn("location:new-york", new_york["unsupported_concepts"])

        supported_cold = _fact_ownership_audit(
            "I'm trying to get through the day without freezing.",
            {"content": "You have to layer up when it is cold."},
            [{"speaker": "Target", "content": "It is really chilly out today."}],
            "Target",
        )
        self.assertFalse(supported_cold["warning"])

        mirrored_office_weather = _fact_ownership_audit(
            "My office is always freezing too.",
            {"content": "My office was chilly until it got a new heater."},
            [
                {"speaker": "Target", "content": "I have a home office."},
                {"speaker": "Target", "content": "It is cold outside today."},
            ],
            "Target",
        )
        self.assertTrue(mirrored_office_weather["warning"])
        self.assertIn(
            "place:office+weather:cold",
            mirrored_office_weather["mirrored_concept_pairs"],
        )

        external_opinion = _fact_ownership_audit(
            "I think the original Japanese Godzilla movies feel more serious.",
            {"content": "Japanese Godzilla movies are much better."},
            [],
            "Target",
        )
        self.assertFalse(external_opinion["warning"])

        tentative_future = _fact_ownership_audit(
            "A YouTube channel could be interesting too, I'll think about it.",
            {"content": "You should start a YouTube channel."},
            [],
            "Target",
        )
        self.assertFalse(tentative_future["warning"])

        backdated_suggestion = _fact_ownership_audit(
            "I've already been planning a YouTube channel for a while.",
            {"content": "You should start a YouTube channel."},
            [],
            "Target",
        )
        self.assertTrue(backdated_suggestion["warning"])

        different_personal_habit = _fact_ownership_audit(
            "I don't watch a ton of foreign films but I get your point about Hollywood movies.",
            {"content": "I love foreign films because Hollywood movies feel shallow."},
            [],
            "Target",
        )
        self.assertFalse(different_personal_habit["warning"])

        negative_experience = _fact_ownership_audit(
            "I haven't watched it, but I've heard good things about Godzilla Minus One.",
            {"content": "Godzilla Minus One is a really good movie."},
            [],
            "Target",
        )
        self.assertFalse(negative_experience["warning"])

    def test_controller_receives_explicit_target_owned_cb_evidence(self):
        turns = [
            {"turn_id": "session_1:turn_0", "session_id": "session_1", "speaker": "Target", "content": "I live in LA."},
            {"turn_id": "session_1:turn_1", "session_id": "session_1", "speaker": "Partner", "content": "I live in NYC."},
        ]
        rendered = _target_owned_history_text(turns, "Target")
        self.assertIn("I live in LA.", rendered)
        self.assertNotIn("I live in NYC.", rendered)
        self.assertIn("target_owned_history", CONTROLLER_USER_TEMPLATE)

    def test_prompts_do_not_backdate_new_partner_suggestions(self):
        controller = CONTROLLER_SYSTEM_PROMPT.casefold()
        actor = ACTOR_USER_TEMPLATE.casefold()
        self.assertIn("partner proposes a new activity", controller)
        self.assertIn("suggestion", controller)
        self.assertIn("previously", controller)
        self.assertIn("tentative future choice", actor)
        self.assertIn("already considered", actor)

    def test_controller_can_represent_closing_praise(self):
        self.assertIn("praise-or-encouragement", PARTNER_ACTS)
        self.assertIn("praise-or-encouragement", CONTROLLER_SYSTEM_PROMPT)
        decision = _decision()
        decision["situation"]["partner_act"] = "praise-or-encouragement"
        decision["situation"]["conversational_obligation"] = "acknowledge"
        normalize_v15_decision(decision)

    def test_v15_actor_self_domain_includes_identity_and_boundaries(self):
        projected = _v15_actor_self_domain(_self_domain())
        self.assertIn("identity_context", projected)
        self.assertIn("boundaries_and_uncertainty", projected)
        self.assertIn("communication_signature", projected)
        self.assertNotIn("interaction_policy_prior", projected)
        self.assertNotIn("affective_social_signature", projected)
        self.assertNotIn("user_domain", projected)

    def test_user_domain_activation_uses_stable_fact_ids(self):
        domain = empty_user_domain()
        domain["behavior"].append({
            "value": "shares detailed daily updates",
            "confidence": "high",
            "evidence_ids": ["session_1:turn_2"],
        })
        facts = build_v15_activation_whitelist(domain)
        decision = _decision()
        decision["relevant_user_domain"] = [{"fact_id": "behavior:0"}]
        resolved = resolve_v15_profile_activation(
            normalize_v15_decision(decision), facts
        )
        self.assertEqual(resolved["relevant_user_domain"], [{
            "fact_id": "behavior:0",
            "layer": "behavior",
            "value": "shares detailed daily updates",
        }])

        decision = _decision()
        decision["relevant_user_domain"] = [{"fact_id": "behavior:99"}]
        with self.assertRaisesRegex(ValueError, "unknown User Domain fact IDs"):
            resolve_v15_profile_activation(normalize_v15_decision(decision), facts)

    def test_provider_schema_avoids_unsupported_array_keywords(self):
        unsupported = {"uniqueItems", "contains", "minContains", "maxContains"}

        def visit(value):
            if isinstance(value, dict):
                self.assertTrue(unsupported.isdisjoint(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(DECISION_SCHEMA)

        duplicate = _decision()
        duplicate["alignment"]["affected_dimensions"] = [
            "turn-composition",
            "turn-composition",
        ]
        with self.assertRaisesRegex(ValueError, "unique array"):
            normalize_v15_decision(duplicate)

    def test_actor_contract_has_one_bubble_per_unit(self):
        audit = actor_structure_audit(
            "First answer.\nWhat about your plan?",
            _decision()["turn_plan"],
            "Emi",
        )
        self.assertTrue(audit["blocking_contract_passed"])

        compound = actor_structure_audit(
            "First answer.\nWhat is your name and are you from Miami?",
            _decision()["turn_plan"],
            "Target",
        )
        self.assertFalse(compound["blocking_contract_passed"])
        self.assertEqual(compound["compound_question_bubbles"], 1)
        failed = actor_structure_audit(
            "First answer. What about your plan?",
            _decision()["turn_plan"],
            "Emi",
        )
        self.assertFalse(failed["bubble_count_match"])

    def test_prompts_exclude_v9_decision_raw_ca_and_metrics(self):
        controller = CONTROLLER_USER_TEMPLATE.casefold()
        actor = ACTOR_USER_TEMPLATE.casefold()
        self.assertNotIn("v9 decision", controller)
        self.assertNotIn("behavior analog", controller)
        self.assertNotIn("ground truth", controller)
        self.assertNotIn("user domain", actor)
        self.assertNotIn("lambda", actor)
        self.assertNotIn("reflectiveness", actor)
        self.assertNotIn("grounding", actor)
        self.assertNotIn("intimacy", actor)
        self.assertIn("preliminary question", actor)
        self.assertIn("one question that directly", ACTOR_REPAIR_TEMPLATE.casefold())

    def test_gate_manifest_is_nested_and_gate30_covers_all_cells(self):
        manifest = build_progressive_gate_manifest(self.prepared)
        previous = set()
        for gate in (6, 18, 30, 60, 120, 519):
            current = set(manifest[str(gate)])
            self.assertEqual(len(current), gate)
            self.assertTrue(previous.issubset(current))
            previous = current

    def test_ca_development_uses_sessions_1_2_and_targets_only_session_3(self):
        prepared, manifest = _prepare_ca_dev("dataset")
        gates = _ca_dev_gate_manifest(prepared)
        self.assertEqual(len(gates["6"]), 6)
        self.assertEqual(len(gates["30"]), 30)
        self.assertTrue(set(gates["6"]).issubset(gates["30"]))
        self.assertEqual(manifest["profile_sessions"], [1, 2])
        for item in prepared:
            self.assertEqual(len(item["profile"]["sessions"]), 2)
            third_session = item["points"][0]["test_sessions"][2]
            self.assertTrue(all(
                point["target_session"] == third_session for point in item["points"]
            ))
            self.assertTrue(all(point["context_truncated"] is False for point in item["points"]))
            self.assertTrue(all(
                point["target_message"] not in [
                    turn["content"] for turn in point["context_turns"][-1:]
                ]
                for point in item["points"]
            ))

    def test_gate6_replays_v9_upstream_without_v9_decision(self):
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
                    "generated_message": "V9 output must remain private",
                    "self_domain_hash": stable_hash(self_domains[item["speaker"]]),
                    "user_domain": empty_user_domain(),
                    "user_domain_completed_session_updates": [],
                    "situation": {"private": "v9 situation"},
                    "alignment": {"private": "v9 alignment"},
                    "next_action": {"private": "v9 action"},
                })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "v9_predictions.jsonl"
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            domains = root / "self_domains.json"
            domains.write_text(json.dumps(self_domains), encoding="utf-8")
            output = root / "v15"
            backend = FakeV15Backend(fail_first_actor=True)
            summary = run_v15(
                V15Config(
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
            controller_calls = [call for call in backend.calls if call["schema"]]
            actor_calls = [
                call for call in backend.calls
                if not call["schema"] and "PRIVATE TURN PLAN" in call["user"]
            ]
            self.assertEqual(len(controller_calls), 6)
            self.assertEqual(len(actor_calls), 7)
            all_prompts = "\n".join(call["user"] for call in backend.calls)
            self.assertNotIn("V9 output must remain private", all_prompts)
            self.assertNotIn('"private": "v9 action"', all_prompts)
            results = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(results[0]["actor_structure_audit"]["actor_attempts"], 2)
            self.assertTrue(all(row["context_truncated"] is False for row in results))

    def test_ca_dev_gate6_runs_only_on_session3_and_is_not_table2(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ca_dev"
            result = run_v15_ca_dev(
                V15CaDevConfig(
                    dataset_dir="dataset",
                    output_dir=str(output),
                    gate=6,
                    fresh=True,
                ),
                backend=FakeV15CaDevBackend(),
            )
            self.assertTrue(result["complete"])
            self.assertEqual(result["records"], 6)
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("excluded from Table 2", manifest["purpose"])
            rows = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(row["ca_internal_development_only"] for row in rows))
            self.assertTrue(all(row["target_session"] not in row["profile_sessions"] for row in rows))
            self.assertTrue(all("fact_ownership_audit" in row for row in rows))
            self.assertTrue(all(not row["fact_ownership_audit"]["warning"] for row in rows))


if __name__ == "__main__":
    unittest.main()
