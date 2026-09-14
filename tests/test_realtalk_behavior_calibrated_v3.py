import pytest

from src.experiments.realtalk_behavior_calibrated_v3 import (
    DECISION_SCHEMA,
    SCENES,
    SELF_DOMAIN_SCHEMA,
    _normalize_decision,
    _normalize_actor,
    _normalize_self,
    _normalize_user_v3,
    _actor_retry_instruction,
    _actor_retry_suffix,
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
        "behavior_policy": {"primary_action": "answer", "selected_question_slots": ["q1"], "outbound_question_mode": "none", "outbound_question_focus": "", "reflection_mode": "none", "self_disclosure_mode": "brief", "grounding_mode": "none", "empathy_mode": "none", "intimacy_mode": "match", "message_shape": "single_typical", "required_content_slots": ["answer q1"], "forbidden_additions": ["unsolicited_exploration"]},
        "evidence_ids": [],
    }


def test_decision_contract_rejects_multi_content_without_two_slots():
    value = valid_decision()
    value["behavior_policy"]["message_shape"] = "multi_content"
    with pytest.raises(ValueError, match="multi_content"):
        _normalize_decision(value, set(), {"question_slots": [{"slot_id": "q1"}]})


def test_decision_contract_rejects_moderate_empathy_without_affect():
    value = valid_decision()
    value["behavior_policy"]["empathy_mode"] = "moderate"
    with pytest.raises(ValueError, match="moderate empathy"):
        _normalize_decision(value, set(), {"question_slots": [{"slot_id": "q1"}]})


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
        {"identity_facts": [], "voice_profile": [], "social_profile": [], "behavior_by_scene": {"session_opening": {"usual_action": "ask_follow_up"}}, "observable_statistics": {"question_rate": 1.0}},
        {"user_state": {}, "behavior_policy": decision["behavior_policy"]},
        {},
    )
    assert "MUST contain no question mark" in prompt
    assert "Do not add a greeting question" in prompt
    assert "behavior_by_scene" not in prompt
    assert "observable_statistics" not in prompt
    assert "social_profile" not in prompt


def test_partner_question_slots_do_not_authorize_outbound_question():
    decision = valid_decision()
    with pytest.raises(ValueError, match="outbound question"):
        _normalize_actor("Sure, I can do that. Anything else?", "Emi", decision)


def test_clarifying_question_requires_an_outbound_question():
    decision = valid_decision()
    decision["behavior_policy"]["grounding_mode"] = "clarifying_question"
    decision["behavior_policy"]["outbound_question_mode"] = "clarifying"
    decision["behavior_policy"]["outbound_question_focus"] = "which day"
    assert _normalize_actor("Which day do you mean?", "Emi", decision) == "Which day do you mean?"
    assert _normalize_actor("Which day, and what time?", "Emi", decision) == "Which day, and what time?"
    with pytest.raises(ValueError, match="at least one"):
        _normalize_actor("I am not sure.", "Emi", decision)


def test_decision_rejects_nonexistent_partner_question_slot():
    with pytest.raises(ValueError, match="nonexistent partner question"):
        _normalize_decision(valid_decision(), set(), {"question_slots": []})


def test_opening_outbound_question_is_independent_of_partner_slots():
    value = valid_decision()
    value["situation"]["scene"] = "session_opening"
    value["behavior_policy"]["selected_question_slots"] = []
    value["behavior_policy"]["outbound_question_mode"] = "opening"
    value["behavior_policy"]["outbound_question_focus"] = "how the partner is doing"
    normalized = _normalize_decision(value, set(), {"question_slots": []})
    assert normalized["behavior_policy"]["outbound_question_mode"] == "opening"


def test_session_opening_rejects_reciprocal_question_mode():
    value = valid_decision()
    value["situation"]["scene"] = "session_opening"
    value["behavior_policy"]["selected_question_slots"] = []
    value["behavior_policy"]["outbound_question_mode"] = "reciprocal"
    value["behavior_policy"]["outbound_question_focus"] = "partner wellbeing"
    with pytest.raises(ValueError, match="session opening"):
        _normalize_decision(value, set(), {"question_slots": []})


def test_user_prompt_separates_partner_and_target_evidence():
    turns = [
        {"session_id": "session_1", "speaker": "Partner", "content": "Hi", "source_id": "p1", "dia_ids": []},
        {"session_id": "session_1", "speaker": "Target", "content": "Hello", "source_id": "t1", "dia_ids": []},
    ]
    prompt = _user_prompt("Target", "Partner", {}, turns, {"p1"})
    assert "p1" in prompt and "t1" in prompt
    assert "FORBIDDEN TARGET-SPEAKER EVIDENCE IDS" in prompt


def test_self_domain_compact_budget_is_enforced():
    value = {
        "identity_facts": [],
        "voice_profile": [],
        "social_profile": [],
        "behavior_by_scene": {scene: {} for scene in SCENES},
        "uncertainties": [],
        "observable_statistics": {},
    }
    value["identity_facts"] = [{"value": "x", "evidence_ids": ["e1"], "confidence": 0.5}] * 7
    with pytest.raises(ValueError, match="compact list"):
        _normalize_self(value, {"e1"})


def test_actor_view_filters_turn_behavior_from_voice_profile():
    value = {
        "identity_facts": [],
        "voice_profile": [{"value": "asks questions", "evidence_ids": ["e1"], "confidence": 0.8}],
        "social_profile": [],
        "behavior_by_scene": {scene: {} for scene in SCENES},
        "uncertainties": [],
        "observable_statistics": {},
    }
    normalized = _normalize_self(value, {"e1"})
    decision = valid_decision()
    decision["behavior_policy"]["selected_question_slots"] = []
    prompt = _actor_prompt(
        {"speaker": "Target"},
        {"context_turns": []},
        normalized,
        {"user_state": {}, "behavior_policy": decision["behavior_policy"]},
        {},
    )
    assert "asks questions" not in prompt


def test_actor_view_filters_greeting_question_example_when_question_is_forbidden():
    value = {
        "identity_facts": [],
        "voice_profile": [{"value": "uses casual greetings like 'Hey!' and 'How are you?'", "evidence_ids": ["e1"], "confidence": 0.9}],
        "social_profile": [],
        "behavior_by_scene": {scene: {} for scene in SCENES},
        "uncertainties": [],
        "observable_statistics": {},
    }
    normalized = _normalize_self(value, {"e1"})
    decision = valid_decision()
    decision["behavior_policy"]["selected_question_slots"] = []
    prompt = _actor_prompt(
        {"speaker": "Target"},
        {"context_turns": []},
        normalized,
        {"user_state": {}, "behavior_policy": decision["behavior_policy"]},
        {},
    )
    assert "How are you?" not in prompt


def test_actor_retry_instruction_removes_all_unselected_questions():
    instruction = _actor_retry_instruction("actor added an unselected outbound question")
    assert "Remove every question" in instruction
    assert "history or Self Domain" in instruction


def test_actor_retry_suffix_includes_rejected_draft_and_exact_repair():
    suffix = _actor_retry_suffix(
        "actor added an unselected outbound question",
        "Hey man, how's it going?",
    )
    assert "REJECTED DRAFT" in suffix
    assert "Hey man, how's it going?" in suffix
    assert "Remove every question" in suffix


def test_actor_allows_ordinary_plan_phrase_when_reflection_is_none():
    decision = valid_decision()
    message = "I think I'll take my dog for a walk later and then read for a while."
    assert _normalize_actor(message, "Emi", decision) == message


def test_actor_rejects_explicit_reflection_when_reflection_is_none():
    decision = valid_decision()
    with pytest.raises(ValueError, match="unsupported reflection"):
        _normalize_actor("I think about why I keep avoiding difficult conversations because I feel uncertain.", "Emi", decision)


def test_user_domain_merges_only_exact_duplicate_facts():
    value = {
        layer: [] for layer in ("core", "regulation", "cognition", "identity", "behavior")
    }
    value["regulation"] = [
        {"value": "Politely ends conversation with good night", "evidence_ids": ["e1"], "confidence": 0.7},
        {"value": "  politely  ends conversation with GOOD NIGHT ", "evidence_ids": ["e2", "e1"], "confidence": 0.9},
        {"value": "Politely ends the conversation with good night", "evidence_ids": ["e3"], "confidence": 0.8},
    ]
    value["update_summary"] = {"added": [], "revised": [], "retained": [], "withdrawn": []}

    normalized = _normalize_user_v3(value, {"e1", "e2", "e3"})

    assert len(normalized["regulation"]) == 2
    assert normalized["regulation"][0]["evidence_ids"] == ["e1", "e2"]
    assert normalized["regulation"][0]["confidence"] == 0.9
    assert normalized["regulation"][1]["value"] == "Politely ends the conversation with good night"


def test_user_domain_duplicate_merge_does_not_hide_invalid_evidence():
    value = {
        layer: [] for layer in ("core", "regulation", "cognition", "identity", "behavior")
    }
    value["regulation"] = [
        {"value": "Politely closes", "evidence_ids": ["e1"], "confidence": 0.7},
        {"value": "politely closes", "evidence_ids": ["invisible"], "confidence": 0.8},
    ]
    value["update_summary"] = {"added": [], "revised": [], "retained": [], "withdrawn": []}

    with pytest.raises(ValueError, match="invalid evidence IDs"):
        _normalize_user_v3(value, {"e1"})
