from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

from .schema import ContractError


def _registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "v3_0" / "threshold_registry.csv"


def _red_bands() -> tuple[float, float]:
    with _registry_path().open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["rule_id"]: row for row in csv.DictReader(handle)}
    low_match = re.search(r"<=([0-9.]+)", rows["C-009"]["value"])
    high_matches = re.findall(r"<=([0-9.]+)", rows["C-010"]["value"])
    if low_match is None or not high_matches:
        raise ContractError("registered HSP RED bands are not parseable")
    return float(low_match.group(1)), float(high_matches[-1])


def _output(
    class_name: str,
    support: str,
    rules: list[str],
    evidence: list[str],
    *,
    ra: float | None = None,
    red: float | None = None,
    estimated: bool = False,
    flags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "score": 1 if class_name == "pass" else 0,
        "class": class_name,
        "support": support,
        "rules": rules,
        "evidence_path": sorted(evidence),
        "ra": ra,
        "red": red,
        "estimated": estimated,
        "transfer_flags": sorted(flags or []),
        "uncertainty_flags": sorted(flags or []),
    }


def evaluate_compatibility(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("unresolved_source_conflict"):
        return _output("indeterminate", "indeterminate", ["C-001"], [], flags=["source_conflict"])
    verified = any(
        status in {"fulltext_verified", "synthetic_verified"}
        for status in record.get("source_verification", {}).values()
    )
    if (
        record.get("direct_target_ether_positive_record") is True
        and record.get("precipitation_warning_present") is False
        and verified
    ):
        return _output(
            "pass",
            "pass-leaning",
            ["C-006", "C-007"],
            list(record.get("source_ids", [])),
        )

    deltas = [record.get("hsp_delta_d"), record.get("hsp_delta_p"), record.get("hsp_delta_h")]
    if any(value is None for value in deltas):
        return _output("indeterminate", "indeterminate", ["C-001"], [], flags=["incomplete_hsp_path"])
    delta_d, delta_p, delta_h = (float(value) for value in deltas)
    ra = math.sqrt(4 * delta_d**2 + delta_p**2 + delta_h**2)
    r0 = record.get("hsp_r0")
    estimated = record.get("hsp_estimated") is True
    flags = ["estimated_endpoint"] if estimated else []
    if r0 is None:
        return _output(
            "indeterminate",
            "indeterminate",
            ["HSP_RED_ETHER_COMPAT_v1", "C-012"],
            list(record.get("source_ids", [])),
            ra=ra,
            red=None,
            estimated=estimated,
            flags=[*flags, "missing_hsp_r0"],
        )
    if float(r0) <= 0:
        raise ContractError("hsp_r0 must be positive")
    red = ra / float(r0)
    low, high = _red_bands()
    if red <= low:
        return _output(
            "pass",
            "pass-leaning",
            ["HSP_RED_ETHER_COMPAT_v1", "C-009", "C-007"],
            list(record.get("source_ids", [])),
            ra=ra,
            red=red,
            estimated=estimated,
            flags=flags,
        )
    if red <= high:
        return _output(
            "boundary-leaning",
            "boundary-leaning",
            ["HSP_RED_ETHER_COMPAT_v1", "C-010"],
            list(record.get("source_ids", [])),
            ra=ra,
            red=red,
            estimated=estimated,
            flags=[*flags, "low_confidence_prediction"],
        )
    return _output(
        "fail-leaning",
        "fail-leaning",
        ["HSP_RED_ETHER_COMPAT_v1", "C-011"],
        list(record.get("source_ids", [])),
        ra=ra,
        red=red,
        estimated=estimated,
        flags=flags,
    )
