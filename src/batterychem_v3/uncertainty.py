from __future__ import annotations

from typing import Any

from .schema import ContractError


def calculate_uncertainty(
    record: dict[str, Any],
    missing_policy: dict[str, Any],
    *,
    hsp_estimated: bool,
) -> dict[str, Any]:
    core_fields = list(missing_policy["rpred_core_fields"]) + list(missing_policy["cpred_core_fields"])
    missing = sorted(field for field in core_fields if record.get(field) is None)
    u_missing = len(missing) / len(core_fields)
    context = record.get("transfer_context", "unknown")
    transfer_map = {"target": 0.0, "non_target_nonaqueous": 0.5, "water": 1.0, "unknown": 1.0}
    if context not in transfer_map:
        raise ContractError(f"unsupported transfer_context: {context}")
    u_transfer = transfer_map[context]
    estimation_penalty = False
    if hsp_estimated and u_transfer < transfer_map["non_target_nonaqueous"]:
        u_transfer = transfer_map["non_target_nonaqueous"]
        estimation_penalty = True
    u_ood = 0.0 if record.get("in_reference_domain") is True else 1.0
    u_conflict = 1.0 if record.get("unresolved_source_conflict") is True else 0.0
    value = max(u_missing, u_transfer, u_ood, u_conflict)
    class_name = "low" if value == 0 else "high" if value == 1 else "medium"
    return {
        "u_missing": u_missing,
        "u_transfer": u_transfer,
        "u_ood": u_ood,
        "u_conflict": u_conflict,
        "U": value,
        "class": class_name,
        "missing_fields": missing,
        "estimated_endpoint_penalty": estimation_penalty,
        "formula": "max(u_missing,u_transfer,u_ood,u_conflict)",
    }
