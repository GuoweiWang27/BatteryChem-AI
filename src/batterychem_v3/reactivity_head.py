from __future__ import annotations

from typing import Any

from .schema import ContractError


def _verified(record: dict[str, Any], source_id: str, status: str) -> bool:
    return record.get("source_verification", {}).get(source_id) == status


def _result(class_name: str, rules: list[str], evidence: list[str], flags: list[str]) -> dict[str, Any]:
    return {
        "score": 1 if class_name == "pass" else 0,
        "class": class_name,
        "rules": rules,
        "evidence_path": sorted(evidence),
        "transfer_flags": sorted(flags),
        "uncertainty_flags": sorted(flags),
    }


def evaluate_reactivity(
    record: dict[str, Any], *, synthetic_window: dict[str, float | str] | None = None
) -> dict[str, Any]:
    candidate_id = record["candidate_id"]
    sources = set(record.get("source_ids", []))
    if record.get("unresolved_source_conflict"):
        return _result("indeterminate", ["R-003"], list(sources), ["source_conflict"])

    if candidate_id == "P01" and "S21" in sources and _verified(record, "S21", "fulltext_verified"):
        return _result("pass", ["R-004", "R-006"], ["S21"], ["non_target_transfer"])
    if (
        candidate_id == "P02"
        and bool(sources & {"S10", "S11"})
        and _verified(record, "S11", "fulltext_verified")
    ):
        return _result("pass", ["R-005", "R-006"], sorted(sources & {"S10", "S11"}), ["non_target_transfer"])
    if (
        candidate_id == "P03"
        and record.get("verified_nonaqueous_two_step_oxidation") is True
        and {"FIXR2-IJES-2015", "S12"}.issubset(sources)
    ):
        return _result(
            "pass",
            ["R-008", "R-006"],
            ["FIXR2-IJES-2015", "S12"],
            ["family_precedent", "non_target_transfer"],
        )

    if synthetic_window is not None:
        if record.get("synthetic") is not True:
            raise ContractError("synthetic reactivity windows are allowed only for synthetic records")
        value = record.get("redox_value")
        if value is None:
            return _result("indeterminate", ["R-001", "R-003"], [], ["missing_redox_value"])
        if record.get("redox_unit") != synthetic_window["unit"]:
            raise ContractError("synthetic window unit does not match redox_unit")
        lower = float(synthetic_window["lower"])
        upper = float(synthetic_window["upper"])
        if lower > upper:
            raise ContractError("synthetic window lower must not exceed upper")
        if lower <= float(value) <= upper:
            return _result("pass", ["SYNTHETIC-WINDOW"], ["synthetic_fixture"], [])
        direction = "too_weak" if float(value) < lower else "too_strong"
        return _result("fail-leaning", ["SYNTHETIC-WINDOW"], ["synthetic_fixture"], [direction])

    return _result("indeterminate", ["R-001", "R-003"], [], ["no_verified_sufficiency_path"])
