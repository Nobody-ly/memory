import pytest

from src.experiments.realtalk_behavior_calibrated_v3 import (
    DECISION_SCHEMA,
    SCENES,
    SELF_DOMAIN_SCHEMA,
    _normalize_decision,
    _scene_for,
    behavior_calibrator,
    _actor_prompt,
    _user_prompt,
)


def test_scene_priority_distinguishes_question_opinion_affect_and_opening():
    assert _scene_for("", opening=True) == "session_opening"
    assert _scene_for("What do you think about it?") == "opinion_or_advice"
    assert _scene_for("Are you feeling sad today?") == "direct_question"
    assert _scene_for("I am worried and exhausted.") == "partner_affect"
    assert _scene_for("I went to work and then cooked dinner.") == "partner_disclosure"


def test_calibrator_does_not_read_previous_session_as_current_partner():
    turns = [{"session_id": "session_1", "speaker": "B", "content": "How are you?"}]
    gate = behavior_calibrator(turns, "A", "session_2", turns)
    assert gate["scene"] == "session_opening"
    assert gate["question_slot_count"] == 0


def valid_decision():
    return {
        "situation": {"scene": "direct_question", "partner_act": "question", "topic": "plans", "uncertainty": "low"},
        "user_state": {"interaction_need": "information_exchange", "affect": "neutral", "affect_confidence": 0.8, "topic_continuity": "continue", "response_pressure": "high"},
        "relevant_user_domain": [],
        "alignment": {"orientation": "self_led", "lambda_trace": 0.2, "basis": "Answer the current question in the target's normal style.", "affected_dimensions": ["content_focus"]},
        "behavior_policy": {"primary_action": "answer", "selected_question_slots": ["q1"], "reflection_mode": "none", "self_disclosure_mode": "brief", "grounding_mode": "none", "empathy_mode": "none", "intimacy_mode": "match", "message_shape": "single_typical", "required_content_slots": ["answer q1"], "forbidden_additions": ["unsolicited_exploration"]},
        "evidence_ids": [],
    }


def test_decision_contract_rejects_multi_content_without_two_slots():
    value = valid_decision()
    value["behavior_policy"]["message_shape"] = "multi_content"
    with pytest.raises(ValueError, match="multi_content"):
        _normalize_decision(value, set(), {})


def test_decision_contract_rejects_moderate_empathy_without_affect():
    value = valid_decision()
    value["behavior_policy"]["empathy_mode"] = "moderate"
    with pytest.raises(ValueError, match="moderate empathy"):
        _normalize_decision(value, set(), {})


def test_schemas_have_fixed_scene_keys_and_strict_roots():
    assert set(SELF_DOMAIN_SCHEMA["schema"]["properties"]["behavior_by_scene"]["properties"]) == set(SCENES)
    assert SELF_DOMAIN_SCHEMA["schema"]["additionalProperties"] is False
    assert DECISION_SCHEMA["schema"]["additionalProperties"] is False


def test_actor_prompt_makes_zero_question_contract_explicit():
    decision = valid_decision()
    decision["behavior_policy"]["selected_question_slots"] = []
    prompt = _actor_prompt(
        {"speaker": "Emi"},
        {"context_turns": []},
        {},
        {"user_state": {}, "behavior_policy": decision["behavior_policy"]},
        {},
    )
    assert "MUST contain no question mark" in prompt
    assert "Do not add a greeting question" in prompt


def test_user_prompt_separates_partner_and_target_evidence():
    turns = [
        {"session_id": "session_1", "speaker": "Partner", "content": "Hi", "source_id": "p1", "dia_ids": []},
        {"session_id": "session_1", "speaker": "Target", "content": "Hello", "source_id": "t1", "dia_ids": []},
    ]
    prompt = _user_prompt("Target", "Partner", {}, turns, {"p1"})
    assert "p1" in prompt and "t1" in prompt
    assert "FORBIDDEN TARGET-SPEAKER EVIDENCE IDS" in prompt
