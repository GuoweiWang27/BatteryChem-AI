from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_missing_fields(record: dict[str, Any], missing_policy: dict[str, Any]) -> dict[str, Any]:
    """Make missing core values explicit without guessing or promoting inference."""

    normalized = deepcopy(record)
    core_fields = list(missing_policy["rpred_core_fields"]) + list(missing_policy["cpred_core_fields"])
    missing = set(normalized.get("missing_fields", []))
    for field in core_fields:
        if field not in normalized or normalized[field] is None:
            normalized[field] = None
            missing.add(field)
    normalized["missing_fields"] = sorted(missing)
    normalized["inference_fields"] = sorted(set(normalized.get("inference_fields", [])))
    return normalized
