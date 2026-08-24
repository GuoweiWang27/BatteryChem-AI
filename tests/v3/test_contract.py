from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from batterychem_v3.compatibility_head import evaluate_compatibility
from batterychem_v3.descriptors import normalize_missing_fields
from batterychem_v3.prioritize import prioritize
from batterychem_v3.reactivity_head import evaluate_reactivity
from batterychem_v3.report import build_report
from batterychem_v3.schema import ContractError, validate_mode, validate_records
from batterychem_v3.uncertainty import calculate_uncertainty


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "v3_0"
SYNTHETIC_WINDOW = {"lower": 1.0, "upper": 2.0, "unit": "synthetic_unit"}


def load_json(name: str) -> dict:
    return json.loads((CONFIG / name).read_text(encoding="utf-8"))


def base_record(candidate_id: str = "SYN-001", name: str = "Synthetic Alpha") -> dict:
    return {
        "candidate_id": candidate_id,
        "preferred_name": name,
        "canonical_smiles": "SYNTHETIC-CANONICAL",
        "isomeric_smiles": "SYNTHETIC-ISOMERIC",
        "inchikey": f"SYNTHETIC-{candidate_id}",
        "source_ids": ["SYN-SOURCE"],
        "source_verification": {"SYN-SOURCE": "synthetic_verified"},
        "verification_status": "synthetic_verified",
        "missing_fields": [],
        "inference_fields": [],
        "synthetic": True,
        "wet_lab_panel": True,
        "selected_by_v3": False,
        "phenolic_oh_count": 2,
        "aromatic_ring_count": 1,
        "redox_value": 1.5,
        "redox_unit": "synthetic_unit",
        "reference_electrode": "synthetic_reference",
        "medium": "synthetic_target",
        "evidence_relevance": "A",
        "direct_battery_additive_precedent": "synthetic_verified",
        "r_support": None,
        "self_consumption_risk": False,
        "mw": 200.0,
        "tpsa": 50.0,
        "xlogp": 1.0,
        "ether_compatibility_evidence": "synthetic_direct_target",
        "dol_dme_solubility_evidence": "synthetic_positive",
        "direct_target_ether_positive_record": True,
        "precipitation_warning_present": False,
        "c_support": None,
        "hsp_delta_d": None,
        "hsp_delta_p": None,
        "hsp_delta_h": None,
        "hsp_r0": None,
        "hsp_estimated": False,
        "hsp_ra": None,
        "hsp_red": None,
        "hsp_symmetry_sensitivity_red": None,
        "transfer_context": "target",
        "in_reference_domain": True,
        "unresolved_source_conflict": False,
        "units": {
            "phenolic_oh_count": "count",
            "aromatic_ring_count": "count",
            "redox_value": "source_unit",
            "mw": "g/mol",
            "tpsa": "A2",
        },
    }


def evaluate(record: dict, *, boundary: bool = True) -> dict:
    missing = load_json("missing_value_policy.json")
    normalized = normalize_missing_fields(record, missing)
    rpred = evaluate_reactivity(normalized, synthetic_window=SYNTHETIC_WINDOW)
    cpred = evaluate_compatibility(normalized)
    uncertainty = calculate_uncertainty(normalized, missing, hsp_estimated=cpred["estimated"])
    return {
        "record": normalized,
        "Rpred": rpred,
        "Cpred": cpred,
        "U": uncertainty,
        "boundary_contract": {"student_signed": boundary, "second_signed": boundary},
    }


def test_01_schema_units_enums_and_unique_ids() -> None:
    one = base_record()
    assert validate_records([one], CONFIG / "feature_registry.csv") == [one]

    duplicate = deepcopy(one)
    with pytest.raises(ContractError, match="duplicate candidate_id"):
        validate_records([one, duplicate], CONFIG / "feature_registry.csv")

    bad_unit = deepcopy(one)
    bad_unit["units"]["mw"] = "kg"
    with pytest.raises(ContractError, match="unit"):
        validate_records([bad_unit], CONFIG / "feature_registry.csv")

    bad_enum = deepcopy(one)
    bad_enum["evidence_relevance"] = "Z"
    with pytest.raises(ContractError, match="evidence_relevance"):
        validate_records([bad_enum], CONFIG / "feature_registry.csv")


def test_02_forbidden_experimental_fields_are_rejected_recursively() -> None:
    record = base_record()
    record["nested"] = {"cycling_result": 123}
    with pytest.raises(ContractError, match="forbidden"):
        validate_records([record], CONFIG / "feature_registry.csv")


def test_03_core_missing_values_remain_null_and_are_not_imputed() -> None:
    record = base_record()
    del record["redox_value"]
    normalized = normalize_missing_fields(record, load_json("missing_value_policy.json"))
    assert normalized["redox_value"] is None
    assert "redox_value" in normalized["missing_fields"]
    assert normalized["inference_fields"] == []


def test_04_four_quadrants_cover_r004_r005_r006_c006_c007() -> None:
    rpass_cpass = base_record("P01", "Synthetic R4 C6")
    rpass_cpass["source_ids"] = ["S21", "SYN-TARGET"]
    rpass_cpass["source_verification"] = {
        "S21": "fulltext_verified",
        "SYN-TARGET": "synthetic_verified",
    }

    rpass_cfail = base_record("P02", "Synthetic R5 Cfail")
    rpass_cfail["source_ids"] = ["S10", "S11"]
    rpass_cfail["source_verification"] = {
        "S10": "abstract_verified",
        "S11": "fulltext_verified",
    }
    rpass_cfail["direct_target_ether_positive_record"] = False
    rpass_cfail.update(hsp_delta_d=1.0, hsp_delta_p=0.0, hsp_delta_h=0.0, hsp_r0=1.0)

    rfail_cpass = base_record("SYN-003", "Synthetic Rweak Cpass")
    rfail_cpass["redox_value"] = 0.5

    rfail_cfail = base_record("SYN-004", "Synthetic Rstrong Cfail")
    rfail_cfail["redox_value"] = 2.5
    rfail_cfail["direct_target_ether_positive_record"] = False
    rfail_cfail.update(hsp_delta_d=1.0, hsp_delta_p=0.0, hsp_delta_h=0.0, hsp_r0=1.0)

    got = [evaluate(x) for x in (rpass_cpass, rpass_cfail, rfail_cpass, rfail_cfail)]
    quadrants = [(x["Rpred"]["class"], x["Cpred"]["class"]) for x in got]
    assert quadrants == [
        ("pass", "pass"),
        ("pass", "fail-leaning"),
        ("fail-leaning", "pass"),
        ("fail-leaning", "fail-leaning"),
    ]
    assert got[0]["Rpred"]["rules"] == ["R-004", "R-006"]
    assert got[1]["Rpred"]["rules"] == ["R-005", "R-006"]
    assert got[0]["Cpred"]["rules"] == ["C-006", "C-007"]


def test_05_synthetic_weak_moderate_strong_reactivity_window() -> None:
    classes = []
    for value in (0.5, 1.5, 2.5):
        record = base_record(f"SYN-W-{value}")
        record["redox_value"] = value
        classes.append(evaluate_reactivity(record, synthetic_window=SYNTHETIC_WINDOW)["class"])
    assert classes == ["fail-leaning", "pass", "fail-leaning"]

    nonsynthetic = base_record("SYN-NOT-ALLOWED")
    nonsynthetic["synthetic"] = False
    with pytest.raises(ContractError, match="synthetic"):
        evaluate_reactivity(nonsynthetic, synthetic_window=SYNTHETIC_WINDOW)


def test_06_same_cli_input_is_byte_identical() -> None:
    payload = {"synthetic": True, "candidates": [base_record()]}
    runtime = ROOT / "tests" / "v3" / ".runtime"
    runtime.mkdir(exist_ok=True)
    input_path = runtime / "input.json"
    out_a = runtime / "a.json"
    out_b = runtime / "b.json"
    try:
        input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        command = [
            sys.executable,
            "-m",
            "batterychem_v3.cli",
            "--input",
            str(input_path),
            "--config-dir",
            str(CONFIG),
            "--mode",
            "RULE",
            "--boundary-student-signed",
            "--boundary-second-signed",
        ]
        subprocess.run([*command, "--output", str(out_a)], cwd=ROOT, env=env, check=True)
        subprocess.run([*command, "--output", str(out_b)], cwd=ROOT, env=env, check=True)
        assert out_a.read_bytes() == out_b.read_bytes()
        assert hashlib.sha256(out_a.read_bytes()).hexdigest() == hashlib.sha256(out_b.read_bytes()).hexdigest()
    finally:
        for path in (input_path, out_a, out_b):
            path.unlink(missing_ok=True)
        runtime.rmdir()


def test_07_priority_set_is_derived_only_from_rpred_cpred_and_u() -> None:
    eligible = evaluate(base_record("SYN-ELIGIBLE"))
    ineligible = evaluate(base_record("SYN-INELIGIBLE"))
    ineligible["Rpred"] = {**ineligible["Rpred"], "class": "indeterminate", "score": 0}
    result = prioritize([eligible, ineligible], load_json("priority_policy.json"), True, True)
    assert result["P"] == ["SYN-ELIGIBLE"]

    changed = deepcopy(eligible)
    changed["record"]["irrelevant_note"] = "must not affect P"
    assert prioritize([changed], load_json("priority_policy.json"), True, True)["P"] == ["SYN-ELIGIBLE"]


def test_08_priority_budget_ties_and_empty_set() -> None:
    names = ["Resveratrol", "Propyl gallate", "Pyrogallol", "Catechin", "Quercetin"]
    evaluations = []
    for index, name in enumerate(names, 1):
        item = evaluate(base_record(f"SYN-TIE-{index}", name))
        item["Rpred"] = {**item["Rpred"], "class": "pass", "score": 1}
        item["Cpred"] = {**item["Cpred"], "class": "pass", "score": 1}
        item["U"] = {**item["U"], "U": 0.0, "class": "low"}
        evaluations.append(item)
    result = prioritize(evaluations, load_json("priority_policy.json"), True, True)
    assert result["P"] == ["SYN-TIE-5", "SYN-TIE-4", "SYN-TIE-3"]
    assert len(result["P"]) <= 3

    for item in evaluations:
        item["Rpred"] = {**item["Rpred"], "class": "indeterminate", "score": 0}
    assert prioritize(evaluations, load_json("priority_policy.json"), True, True)["P"] == []


def test_09_five_panel_candidates_never_claim_v3_selection() -> None:
    records = [base_record(f"P0{i}", f"Synthetic Panel {i}") for i in range(1, 6)]
    report = build_report(records, CONFIG, mode="RULE", student_signed=True, second_signed=True)
    assert len(report["candidates"]) == 5
    assert all(row["wet_lab_panel"] is True for row in report["candidates"])
    assert all(row["selected_by_v3"] is False for row in report["candidates"])


def test_10_registry_hsp_estimation_and_boundary_signature_contract() -> None:
    with (CONFIG / "threshold_registry.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    allowed = {"literature", "physical_derivation", "preregistered_convention", "lab_detection_limit"}
    assert len(rows) == len({row["rule_id"] for row in rows}) == 52
    assert all(row["source_type"] in allowed for row in rows)
    assert all(row["source_id"] and row["source_location"] for row in rows)
    assert all(row["student_approved_by"] and row["second_approved_by"] for row in rows)

    missing_r0 = base_record("SYN-HSP-NO-R0")
    missing_r0["direct_target_ether_positive_record"] = False
    missing_r0.update(hsp_delta_d=0.1, hsp_delta_p=0.1, hsp_delta_h=0.1, hsp_r0=None)
    assert evaluate_compatibility(missing_r0)["red"] is None
    assert evaluate_compatibility(missing_r0)["class"] == "indeterminate"

    exact = base_record("SYN-HSP-EXACT")
    exact["direct_target_ether_positive_record"] = False
    exact.update(hsp_delta_d=0.1, hsp_delta_p=0.1, hsp_delta_h=0.1, hsp_r0=1.0)
    estimated = deepcopy(exact)
    estimated["candidate_id"] = "SYN-HSP-EST"
    estimated["hsp_estimated"] = True
    exact_eval = evaluate(exact)
    estimated_eval = evaluate(estimated)
    assert estimated_eval["U"]["U"] > exact_eval["U"]["U"]

    boundary = base_record("SYN-BOUNDARY")
    boundary["direct_target_ether_positive_record"] = False
    boundary.update(hsp_delta_d=0.45, hsp_delta_p=0.0, hsp_delta_h=0.0, hsp_r0=1.0)
    boundary_eval = evaluate(boundary)
    assert boundary_eval["Cpred"]["class"] == "boundary-leaning"
    assert prioritize([boundary_eval], load_json("priority_policy.json"), True, True)["P"] == ["SYN-BOUNDARY"]
    assert prioritize([boundary_eval], load_json("priority_policy.json"), True, False)["P"] == []


def test_11_rule_and_logistic_mode_gate() -> None:
    assert validate_mode("RULE", None) == "RULE"
    with pytest.raises(ContractError, match="external dataset audit"):
        validate_mode("LOGISTIC_AUGMENTED", None)
    with pytest.raises(ContractError, match="grouped_split"):
        validate_mode(
            "LOGISTIC_AUGMENTED",
            {
                "labels_compatible": True,
                "traceable_sources": True,
                "sample_distribution_audited": True,
                "grouped_split": False,
            },
        )
    assert validate_mode(
        "LOGISTIC_AUGMENTED",
        {
            "labels_compatible": True,
            "traceable_sources": True,
            "sample_distribution_audited": True,
            "grouped_split": True,
        },
    ) == "LOGISTIC_AUGMENTED"


def test_12_report_never_emits_qd100_prediction_or_result_fields() -> None:
    report = build_report([base_record()], CONFIG, mode="RULE", student_signed=True, second_signed=True)
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True).lower()
    assert "qd100" not in rendered
    assert "experimental_result" not in rendered
    assert "cycling_result" not in rendered
