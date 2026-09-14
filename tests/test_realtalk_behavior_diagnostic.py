import pytest

from tools.analyze_realtalk_behavior import causal_proxy_forecast, confusion, context_features, features, indexed, strip_values


def test_new_session_does_not_inherit_old_request_as_current_partner_turn():
    history = [{"speaker": "B", "session_id": "session_1", "content": "Why are you sad?"}]
    scene, last, flags, count = context_features(history, "A", "session_2")
    assert (scene, last, count) == ("session_opening", "", 0)
    assert not flags["question"]


def test_multiline_question_and_reflection_are_independent_proxies():
    observed = features("I think that's okay.\nHow are you\uff1f")
    assert observed["bubbles"] == 2
    assert observed["question"] and observed["reflection_marker"]
    assert not features("I bought a pen.")["reflection_marker"]


def test_confusion_orientation_and_duplicate_validation():
    values = confusion([False, False, True, True], [False, True, False, True])
    assert all(values[k] == 1 for k in ["tp", "fp", "fn", "tn"])
    assert values["accuracy"] == 0.5
    with pytest.raises(ValueError, match="Duplicate"):
        indexed([{"result_id": "a"}, {"result_id": "a"}])


def test_scene_only_uses_last_current_partner_message():
    history = [
        {"speaker": "B", "session_id": "session_1", "content": "I am sad."},
        {"speaker": "A", "session_id": "session_1", "content": "Why?"},
        {"speaker": "B", "session_id": "session_1", "content": "What did you buy?"},
    ]
    assert context_features(history, "A", "session_1")[0] == "explicit_question"


def test_causal_forecast_updates_only_after_scoring_current_point():
    ca = [{"speaker": "A", "content": "Hello"}]
    points = [{"target_message": "Why?"}, {"target_message": "Okay"}]
    scores = causal_proxy_forecast(ca, points, "A")["question"]
    prior = 1/3
    expected = ((prior-1)**2 + ((8*prior+1)/9)**2)/2
    assert scores["ca_cb_brier"] == pytest.approx(expected)


def test_normalization_audit_strips_only_whitespace():
    assert strip_values({"p": [" none "]}) == {"p": ["none"]}
    assert strip_values({"p": "none"}) != strip_values({"p": "allowed"})
