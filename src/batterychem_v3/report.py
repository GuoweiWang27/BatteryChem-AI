from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .compatibility_head import evaluate_compatibility
from .descriptors import normalize_missing_fields
from .prioritize import prioritize
from .reactivity_head import evaluate_reactivity
from .schema import validate_mode, validate_records
from .uncertainty import calculate_uncertainty


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(
    records: list[dict[str, Any]],
    config_dir: str | Path,
    *,
    mode: str,
    student_signed: bool,
    second_signed: bool,
    external_dataset_audit: dict[str, Any] | None = None,
    synthetic_window: dict[str, float | str] | None = None,
) -> dict[str, Any]:
    config = Path(config_dir)
    validated_mode = validate_mode(mode, external_dataset_audit)
    validated = validate_records(records, config / "feature_registry.csv")
    missing_policy = _load_json(config / "missing_value_policy.json")
    priority_policy = _load_json(config / "priority_policy.json")
    evaluations: list[dict[str, Any]] = []
    for record in validated:
        normalized = normalize_missing_fields(record, missing_policy)
        rpred = evaluate_reactivity(normalized, synthetic_window=synthetic_window)
        cpred = evaluate_compatibility(normalized)
        uncertainty = calculate_uncertainty(normalized, missing_policy, hsp_estimated=cpred["estimated"])
        evaluations.append({"record": normalized, "Rpred": rpred, "Cpred": cpred, "U": uncertainty})
    priority = prioritize(evaluations, priority_policy, student_signed, second_signed)
    dpred_by_id = {row["candidate_id"]: row for row in priority["Dpred"]}
    candidate_rows = []
    for item in evaluations:
        record = item["record"]
        candidate_id = record["candidate_id"]
        selected = dpred_by_id.get(candidate_id)
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "preferred_name": record["preferred_name"],
                "wet_lab_panel": record["wet_lab_panel"],
                "selected_by_v3": False,
                "Rpred": item["Rpred"],
                "Cpred": item["Cpred"],
                "U": item["U"],
                "Dpred": selected is not None,
                "joint_score": selected["joint_score"] if selected else None,
                "low_confidence_prediction": selected["low_confidence_prediction"] if selected else False,
            }
        )
    candidate_rows.sort(key=lambda row: row["candidate_id"])
    config_names = (
        "feature_registry.csv",
        "threshold_registry.csv",
        "missing_value_policy.json",
        "priority_policy.json",
    )
    return {
        "schema_version": "batterychem_v3_report_v1",
        "mode": validated_mode,
        "algorithm": "RULE_TRANSPARENT_V3",
        "config_sha256": {name: _sha256(config / name) for name in config_names},
        "boundary_contract": {
            "student_signed": bool(student_signed),
            "second_signed": bool(second_signed),
            "active": priority["boundary_contract_active"],
            "fallback": "pass_only" if not priority["boundary_contract_active"] else None,
        },
        "candidates": candidate_rows,
        "Dpred": [row["candidate_id"] for row in priority["Dpred"]],
        "P": priority["P"],
        "budget_k": priority["budget_k"],
    }
