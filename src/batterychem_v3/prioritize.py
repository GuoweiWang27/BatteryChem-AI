from __future__ import annotations

from typing import Any


def prioritize(
    evaluations: list[dict[str, Any]],
    priority_policy: dict[str, Any],
    student_signed: bool,
    second_signed: bool,
) -> dict[str, Any]:
    boundary_active = bool(student_signed and second_signed)
    order = priority_policy["ranking"][2]["mapping"]
    qualitative_rank = {name: index for index, name in enumerate(order, 1)}
    eligible: list[dict[str, Any]] = []
    for item in evaluations:
        rpred = item["Rpred"]
        cpred = item["Cpred"]
        uncertainty = item["U"]
        c_allowed = cpred["class"] == "pass" or (
            boundary_active and cpred["class"] == "boundary-leaning"
        )
        if rpred["class"] != "pass" or not c_allowed or uncertainty["class"] == "high":
            continue
        joint_score = min(float(rpred["score"]), float(cpred["score"])) * (1 - float(uncertainty["U"]))
        eligible.append(
            {
                "candidate_id": item["record"]["candidate_id"],
                "preferred_name": item["record"]["preferred_name"],
                "joint_score": joint_score,
                "U": float(uncertainty["U"]),
                "low_confidence_prediction": cpred["class"] == "boundary-leaning",
            }
        )
    eligible.sort(
        key=lambda row: (
            -row["joint_score"],
            row["U"],
            qualitative_rank.get(row["preferred_name"], len(qualitative_rank) + 1),
            row["candidate_id"],
        )
    )
    budget = int(priority_policy["selection"]["max_candidates"])
    return {
        "boundary_contract_active": boundary_active,
        "Dpred": eligible,
        "P": [row["candidate_id"] for row in eligible[:budget]],
        "budget_k": budget,
    }
