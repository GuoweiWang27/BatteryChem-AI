from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when an input violates the frozen V3 contract."""


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def _feature_rows(feature_registry: str | Path) -> list[dict[str, str]]:
    with Path(feature_registry).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_records(records: Sequence[dict[str, Any]], feature_registry: str | Path) -> list[dict[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ContractError("records must be a sequence")
    rows = _feature_rows(feature_registry)
    forbidden = {row["field_name"].lower() for row in rows if row["axis"] == "FORBIDDEN"}
    identity = [row["field_name"] for row in rows if row["axis"] == "IDENTITY"]
    units = {
        row["field_name"]: row["unit"]
        for row in rows
        if row["unit"] and row["allowed_for_scoring"].lower() == "true"
    }
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ContractError(f"record {index} must be an object")
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ContractError("candidate_id is required")
        if candidate_id in seen:
            raise ContractError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        for field in identity:
            if record.get(field) in (None, ""):
                raise ContractError(f"{candidate_id}: required identity field {field}")
        key_hits = sorted({key for key in _walk_keys(record) if key.lower() in forbidden})
        if key_hits:
            raise ContractError(f"{candidate_id}: forbidden field(s): {', '.join(key_hits)}")
        record_units = record.get("units")
        if not isinstance(record_units, dict):
            raise ContractError(f"{candidate_id}: units object is required")
        for field, expected_unit in units.items():
            if record.get(field) is not None and record_units.get(field) != expected_unit:
                raise ContractError(
                    f"{candidate_id}: unit for {field} must be {expected_unit}, got {record_units.get(field)!r}"
                )
        relevance = record.get("evidence_relevance")
        if relevance is not None and relevance not in {"A", "B", "C", "D"}:
            raise ContractError(f"{candidate_id}: evidence_relevance must be A/B/C/D")
        if record.get("wet_lab_panel") is not True or record.get("selected_by_v3") is not False:
            raise ContractError(f"{candidate_id}: panel flags must remain true/false")
        validated.append(record)
    return validated


def validate_mode(mode: str, external_dataset_audit: dict[str, Any] | None) -> str:
    if mode == "RULE":
        return mode
    if mode != "LOGISTIC_AUGMENTED":
        raise ContractError(f"unsupported mode: {mode}")
    if not external_dataset_audit:
        raise ContractError("LOGISTIC_AUGMENTED requires an external dataset audit")
    required = (
        "labels_compatible",
        "traceable_sources",
        "sample_distribution_audited",
        "grouped_split",
    )
    for field in required:
        if external_dataset_audit.get(field) is not True:
            raise ContractError(f"external dataset audit requires {field}=true")
    return mode
