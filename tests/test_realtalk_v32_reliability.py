import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from src.experiments.realtalk_ours import _parse_structured_json, _structured_call
from src.experiments.realtalk_behavior_calibrated_v3 import DECISION_SCHEMA, _actor_call
from src.experiments.operation_checkpoint import OperationCheckpoint
from src.experiments.personaemp.client import ChatResult


def result(content, finish="stop"):
    return ChatResult(content, "deepseek-v4-flash", 10, 20, 0.1, 1, finish_reason=finish)


class Backend:
    model = "deepseek-v4-flash"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def chat(self, system, prompt, **kwargs):
        self.calls.append((system, prompt, kwargs))
        return next(self.outputs)


@pytest.mark.parametrize("text", ['{"behavior": [1, 2], "core":', '{"x": {"safe": 1},', '```json\n{"x": 1}', '{"x": 1} {"x": 2}'])
def test_truncated_outer_json_never_becomes_nested_array_or_object(text):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _parse_structured_json(text)


def test_parser_accepts_only_complete_outer_json_and_whole_fences():
    assert _parse_structured_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert _parse_structured_json('[{"a": 1}]') == [{"a": 1}]


def test_length_response_rejected_before_schema_and_retry_is_bounded(tmp_path):
    schema = {"name": "test", "strict": True, "schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": False}}
    backend = Backend([result('{"x":1}', "length"), result('{"x":2}')])
    checkpoint = OperationCheckpoint(tmp_path / "checkpoint.json", "test")
    kwargs = dict(checkpoint=checkpoint, backend=backend, operation_key="test", system_prompt="JSON", user_prompt="evidence", schema=schema, normalizer=lambda x:x, max_tokens=8192, max_attempts=3, raw_audit=tmp_path / "raw.jsonl", enable_thinking=False, validate_schema=True)
    assert _structured_call(**kwargs)["data"] == {"x": 2}
    assert "truncated" in backend.calls[1][1]
    _structured_call(**kwargs)
    assert len(backend.calls) == 2
    assert not checkpoint.data["failures"]


def test_schema_rejects_extra_fields_and_singleton_arrays(tmp_path):
    schema = {"name": "test", "strict": True, "schema": {"type": "object", "properties": {}, "additionalProperties": False}}
    backend = Backend([result('[{}]'), result('{"extra": 1}'), result('[{}]')])
    checkpoint = OperationCheckpoint(tmp_path / "c.json", "test")
    with pytest.raises(ValidationError):
        _structured_call(checkpoint=checkpoint, backend=backend, operation_key="test", system_prompt="JSON", user_prompt="evidence", schema=schema, normalizer=lambda x:x, max_tokens=100, max_attempts=3, raw_audit=tmp_path / "raw.jsonl", enable_thinking=False, validate_schema=True)
    assert len(backend.calls) == 3
    assert checkpoint.data["failures"]["test"]["attempt"] == 3


def test_actor_repair_keeps_policy_and_history_without_unbounded_retries(tmp_path):
    decision = {"behavior_policy": {"outbound_question_mode": "none", "reflection_mode": "none"}}
    backend = Backend([result("Yes, and you?"), result("Yes, I have plans.")])
    checkpoint = OperationCheckpoint(tmp_path / "c.json", "test")
    outcome = _actor_call(checkpoint, backend, "actor", "Target", "FULL HISTORY: H", decision, tmp_path / "raw.jsonl", 3, 0)
    assert outcome["data"] == "Yes, I have plans."
    assert [call[2]["temperature"] for call in backend.calls] == [0.6, 0.0]
    assert all("FULL HISTORY: H" in call[1] for call in backend.calls)
    assert "Remove every question" in backend.calls[1][0]


def test_schema_uses_variable_slots_not_redundant_composition():
    policy = DECISION_SCHEMA["schema"]["properties"]["behavior_policy"]
    assert "message_shape" not in policy["properties"]
    slots = policy["properties"]["required_content_slots"]
    for n in (1, 2, 3, 4):
        Draft202012Validator(slots).validate(["content"] * n)
    for n in (0, 5):
        with pytest.raises(ValidationError):
            Draft202012Validator(slots).validate(["content"] * n)
