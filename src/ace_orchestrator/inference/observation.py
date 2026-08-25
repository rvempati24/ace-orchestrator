"""Compact, grounded observation serialization for multimodal CUA prompts."""

from __future__ import annotations

import json
from typing import Any

from ace_orchestrator.execution.environment import EnvironmentObservation


def _attribute(node: dict[str, Any], key: str) -> Any:
    value = node.get(key)
    return value.get("value") if isinstance(value, dict) else value


def _compact_elements(semantic_state: Any, *, max_elements: int) -> list[dict[str, Any]]:
    if not isinstance(semantic_state, dict) or not isinstance(semantic_state.get("nodes"), list):
        return []
    elements: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for node in semantic_state["nodes"]:
        if not isinstance(node, dict) or node.get("browsergym_id") is None:
            continue
        row = {
            "ref": str(node["browsergym_id"]),
            "role": str(_attribute(node, "role") or ""),
            "name": str(_attribute(node, "name") or ""),
        }
        value = _attribute(node, "value")
        if value is not None and value != "":
            row["value"] = value
        states = {
            str(prop.get("name")): _attribute(prop, "value")
            for prop in node.get("properties", [])
            if isinstance(prop, dict)
            and prop.get("name") in {"checked", "disabled", "expanded", "focused", "selected"}
        }
        if states:
            row["states"] = states
        identity = (row["ref"], row["role"], row["name"], str(row.get("value", "")))
        if identity in seen:
            continue
        seen.add(identity)
        elements.append(row)
        if len(elements) >= max_elements:
            break
    return elements


def compact_observation(
    observation: EnvironmentObservation,
    *,
    max_elements: int = 300,
    max_json_chars: int = 80_000,
) -> dict[str, Any]:
    """Drop raw DOM/image duplication while retaining all grounded interaction data."""

    values = observation.values
    excluded = {"semantic_state", "dom_state"}
    page = {key: value for key, value in values.items() if key not in excluded}
    result: dict[str, Any] = {
        "observation_id": observation.observation_id,
        "page": page,
        "available_action_refs": list(observation.available_action_refs),
        "elements": _compact_elements(values.get("semantic_state", {}), max_elements=max_elements),
    }
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > max_json_chars:
        result["elements"] = []
        result["serialization_warning"] = (
            "grounded element list exceeded the prompt-size guard; use screenshot and page fields"
        )
        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > max_json_chars:
        raise ValueError("observation exceeds the prompt-size guard after compaction")
    return result
