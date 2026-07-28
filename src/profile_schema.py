from __future__ import annotations

from typing import Any, Dict, List, Mapping

PROFILE_LAYERS = ("core", "regulation", "cognition", "identity", "behavior")
PROFILE_FIELDS: Dict[str, tuple[str, ...]] = {
    "core": ("values", "motivations", "long_term_goals"),
    "regulation": ("stress_response", "emotion_regulation", "conflict_style"),
    "cognition": ("thinking_style", "decision_style", "beliefs"),
    "identity": ("self_identity", "social_identity", "life_context"),
    "behavior": ("interaction_style", "habits", "preferences"),
}

# Deterministic legacy-to-fixed mappings. Only semantically clear mappings are
# retained; ambiguous legacy fields are deliberately not forced into a slot.
LEGACY_FIELD_SOURCES: Dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("core", "values"): (("core", "values"),),
    ("core", "motivations"): (("core", "motivations"), ("core", "sources of meaning")),
    ("core", "long_term_goals"): (("core", "long_term_goals"), ("core", "desires")),
    ("regulation", "stress_response"): (("regulation", "stress_response"), ("regulation", "avoidance"), ("regulation", "control"), ("regulation", "obsession")),
    ("regulation", "emotion_regulation"): (("regulation", "emotion_regulation"), ("regulation", "humor"), ("regulation", "rationalization")),
    ("regulation", "conflict_style"): (("regulation", "conflict_style"), ("regulation", "people-pleasing"), ("regulation", "aggression")),
    ("cognition", "thinking_style"): (("cognition", "thinking_style"), ("cognition", "expression style"), ("cognition", "information density")),
    ("cognition", "decision_style"): (("cognition", "decision_style"), ("cognition", "decision style")),
    ("cognition", "beliefs"): (("cognition", "beliefs"), ("cognition", "technology_view")),
    ("identity", "self_identity"): (("identity", "self_identity"),),
    ("identity", "social_identity"): (("identity", "social_identity"), ("identity", "professional_identity"), ("identity", "occupation"), ("identity", "social relationships"), ("identity", "family")),
    ("identity", "life_context"): (("identity", "life_context"), ("identity", "current_stage"), ("identity", "age"), ("identity", "physical environment")),
    ("behavior", "interaction_style"): (("behavior", "interaction_style"), ("cognition", "emotional visibility"), ("cognition", "social distance")),
    ("behavior", "habits"): (("behavior", "habits"), ("behavior", "long-term behavior patterns"), ("behavior", "learning"), ("behavior", "tool_usage")),
    ("behavior", "preferences"): (("behavior", "preferences"), ("behavior", "content preferences"), ("behavior", "consumption preferences"), ("behavior", "entertainment preferences"), ("behavior", "interests")),
}


def create_empty_static_profile() -> Dict[str, Dict[str, Any]]:
    """Return the canonical, persisted bare five-layer profile."""
    return {
        layer: {"summary": "", **{field: [] for field in PROFILE_FIELDS[layer]}}
        for layer in PROFILE_LAYERS
    }


def is_bare_profile(profile: Any) -> bool:
    return (
        isinstance(profile, Mapping)
        and set(profile) == set(PROFILE_LAYERS)
        and all(isinstance(profile.get(layer), Mapping) for layer in PROFILE_LAYERS)
    )


def _string_list(value: Any) -> List[str]:
    if isinstance(value, Mapping) and "value" in value:
        value = value["value"]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def normalize_bare_profile(profile: Any) -> Dict[str, Dict[str, Any]]:
    """Convert legacy wrappers/leaf objects to the fixed persisted contract.

    Unknown fields and runtime context are deliberately not persisted in the new
    profile contract. This function is also safe for already-bare profiles.
    """
    source: Any = profile
    if isinstance(source, Mapping) and isinstance(source.get("state_axis"), Mapping):
        source = source["state_axis"].get("static_profile", {})
    elif isinstance(source, Mapping) and "static_profile" in source:
        source = source.get("static_profile", {})

    result = create_empty_static_profile()
    if not isinstance(source, Mapping):
        return result

    for layer in PROFILE_LAYERS:
        section = source.get(layer)
        if isinstance(section, Mapping):
            raw_summary = section.get("summary", "")
            if isinstance(raw_summary, Mapping):
                raw_summary = raw_summary.get("value", "")
            result[layer]["summary"] = raw_summary.strip() if isinstance(raw_summary, str) else ""

        for field in PROFILE_FIELDS[layer]:
            values: List[str] = []
            for source_layer, source_field in LEGACY_FIELD_SOURCES[(layer, field)]:
                source_section = source.get(source_layer)
                if not isinstance(source_section, Mapping):
                    continue
                for item in _string_list(source_section.get(source_field, [])):
                    if item not in values:
                        values.append(item)
            result[layer][field] = values
    return result


def validate_bare_profile(profile: Any) -> Dict[str, Dict[str, Any]]:
    """Validate an external/API profile without silently dropping bad fields."""
    if not isinstance(profile, Mapping) or set(profile) != set(PROFILE_LAYERS):
        raise ValueError("profile must contain exactly the fixed five layers")
    result = create_empty_static_profile()
    for layer in PROFILE_LAYERS:
        section = profile[layer]
        expected = {"summary", *PROFILE_FIELDS[layer]}
        if not isinstance(section, Mapping) or set(section) != expected:
            raise ValueError(f"{layer} must contain summary and its three fixed fields")
        summary = section["summary"]
        if not isinstance(summary, str):
            raise ValueError(f"{layer}.summary must be a string")
        result[layer]["summary"] = summary.strip()
        for field in PROFILE_FIELDS[layer]:
            raw = section[field]
            if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
                raise ValueError(f"{layer}.{field} must be a string array")
            result[layer][field] = _string_list(raw)
    return result


def profile_has_content(profile: Mapping[str, Any]) -> bool:
    return any(
        bool(profile.get(layer, {}).get("summary"))
        or any(profile.get(layer, {}).get(field) for field in PROFILE_FIELDS[layer])
        for layer in PROFILE_LAYERS
    )
