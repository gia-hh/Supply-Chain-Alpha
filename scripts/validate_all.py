from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from supply_chain_alpha.data.disclosures import sha256_file, verify_raw_manifest
from supply_chain_alpha.data.io import read_table
from supply_chain_alpha.data.schemas import SCHEMAS, validate_table
from supply_chain_alpha.pipeline import missing_mandatory_test_artifacts
from supply_chain_alpha.utils.config import load_config
from supply_chain_alpha.utils.hashing import (
    config_sha256,
    source_tree_manifest,
    source_tree_manifest_sha256,
)
from supply_chain_alpha.utils.status import phase_status_path, read_phase_status

ALLOWED_CONCLUSIONS = {"POSITIVE", "NULL", "INCONCLUSIVE", "INFEASIBLE_DATA"}
PATH_A_CONCLUSIONS = {"POSITIVE", "NULL", "INCONCLUSIVE"}
ALLOWED_STRATEGY_EVIDENCE = {
    "ECONOMICALLY_VIABLE",
    "NOT_ECONOMICALLY_VIABLE",
    "NOT_EVALUATED",
}
MANDATORY_PHASE3_REPORTS = (
    "resolution_audit.csv",
    "supply_chain_edge.csv",
    "yearly_disclosures.csv",
    "yearly_coverage.csv",
    "degree_diagnostics.csv",
    "node_roles.csv",
    "source_document_diagnostics.csv",
    "industry_diagnostics.csv",
    "edge_age_distribution.csv",
    "summary.json",
)
PATH_B_FINAL_ARTIFACTS = (
    "reports/final/FEASIBILITY_REPORT.md",
    "reports/final/results_summary.json",
    "reports/phase3_coverage/summary.json",
)
PATH_A_REQUIRED_FINAL_ARTIFACTS = (
    "reports/final/FINAL_REPORT.md",
    "reports/final/EXECUTIVE_SUMMARY.md",
    "reports/final/results_summary.json",
    "reports/final/limitations.md",
    "reports/final/reproducibility.md",
)
INTEGRITY_CHECK_KEYS = (
    "timing_violations",
    "edge_max_age_violations",
    "invalid_counterparty_mentions",
    "invalid_resolution_audit_names",
    "resolution_status_classification_mismatches",
    "anonymous_resolved_nodes",
    "duplicate_pair_date_exposures",
    "missing_source_provenance_on_active_edges",
    "missing_security_master_endpoints",
    "missing_listing_dates_on_edge_endpoints",
    "pit_industry_unclassified_active_endpoints",
    "pit_industry_ambiguous_active_nodes",
)
COVERAGE_CHECK_KEYS = (
    "consecutive_usable_years",
    "usable_years_median_connected_stocks",
    "pit_industry_count",
    "pit_largest_industry_share",
)
DEFERRED_PIT_CHECK_KEYS = {
    "pit_industry_count",
    "pit_largest_industry_share",
    "pit_industry_unclassified_active_endpoints",
    "pit_industry_ambiguous_active_nodes",
}
REQUIRED_PHASE3_TESTS = (
    "tests/unit/test_edges.py",
    "tests/unit/test_phase3_gate.py",
    "tests/unit/test_run_phase3.py",
    "tests/integration/test_phase1_3_pipeline.py",
    "tests/regression/test_future_invariance_phase3.py",
)
PHASE2_REQUIRED_CRITERIA = (
    "production_qa_eligible",
    "security_master_checks_pass",
    "cninfo_stock_map_checks_pass",
    "disclosure_acquisition_checks_pass",
    "document_extraction_errors_zero",
    "relationship_and_resolution_qa_pass",
    "publication_timing_qa_pass",
    "alias_validity_and_future_invariance_tests_pass",
    "production_raw_manifest_verified",
    "phase2_pass_criteria_met",
)


class ProjectValidationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectValidationError(
            f"Cannot read valid JSON artifact: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ProjectValidationError(f"JSON artifact must be an object: {path}")
    return value


def _validate_canonical_tables(
    root: Path, config: dict[str, Any], names: list[str]
) -> None:
    for name in names:
        path_value = config["paths"].get(name)
        if path_value is None:
            raise ProjectValidationError(f"Config has no canonical path for {name}")
        path = root / path_value
        if not path.is_file():
            raise ProjectValidationError(f"Missing canonical table: {path}")
        validate_table(read_table(path), SCHEMAS[name])


def _validate_artifact_hashes(root: Path, hashes: dict[str, Any], label: str) -> None:
    if not hashes:
        raise ProjectValidationError(f"{label} artifact hashes are missing")
    root_resolved = root.resolve()
    for relative, expected in hashes.items():
        if not isinstance(relative, str):
            raise ProjectValidationError(f"{label} artifact-hash paths must be strings")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ProjectValidationError(f"{label} artifact path is unsafe: {relative}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ProjectValidationError(
                f"{label} artifact hash is invalid: {relative}"
            )
        try:
            path = (root / relative_path).resolve()
            path.relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise ProjectValidationError(
                f"{label} artifact path is unsafe: {relative}"
            ) from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise ProjectValidationError(f"{label} artifact hash mismatch: {relative}")


def _read_statuses(root: Path) -> dict[int, dict[str, Any]]:
    statuses: dict[int, dict[str, Any]] = {}
    for phase in range(1, 12):
        path = phase_status_path(root, phase)
        if path.is_file():
            statuses[phase] = read_phase_status(path)
    return statuses


def _normalised_status_paths(status: dict[str, Any], field: str) -> set[str]:
    values = status.get(field, [])
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ProjectValidationError(f"Phase status {field} must be a list of paths")
    return {Path(value).as_posix() for value in values}


def _check_group_passed(
    checks: dict[str, Any],
    *,
    required_keys: tuple[str, ...],
    label: str,
) -> bool:
    missing = sorted(set(required_keys) - set(checks))
    if missing:
        raise ProjectValidationError(f"{label} omitted mandatory checks: {missing}")
    passed = True
    for key in required_keys:
        check = checks[key]
        if not isinstance(check, dict):
            raise ProjectValidationError(f"{label} check must be a mapping: {key}")
        result = check.get("passed")
        if result is True:
            continue
        if result is False:
            passed = False
            continue
        if (
            key in DEFERRED_PIT_CHECK_KEYS
            and result is None
            and check.get("deferred") is True
            and check.get("reason") == "PIT_INDUSTRY_UNAVAILABLE"
            and check.get("actual") is None
        ):
            continue
        raise ProjectValidationError(f"{label} check has an invalid result: {key}")
    return passed


def _expected_phase3_thresholds(config: dict[str, Any]) -> dict[str, Any]:
    thresholds = dict(config["graph_coverage_gate"])
    for key in ("evaluation_start", "evaluation_end"):
        thresholds[key] = pd.Timestamp(thresholds[key]).date().isoformat()
    thresholds["edge_max_age_days"] = int(
        config["information_timing"]["edge_max_age_days"]
    )
    return thresholds


def _validate_phase3_thresholds(
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    gate = summary.get("gate")
    if not isinstance(gate, dict):
        raise ProjectValidationError("Phase-3 summary gate is missing")
    if gate.get("frozen_thresholds") != _expected_phase3_thresholds(config):
        raise ProjectValidationError(
            "Phase-3 report does not use the current frozen thresholds"
        )


def _validate_phase2_evidence(
    root: Path,
    status: dict[str, Any],
    config: dict[str, Any],
) -> None:
    criteria = status.get("criteria")
    if not isinstance(criteria, dict):
        raise ProjectValidationError("Phase 2 criteria are missing")
    failed = [key for key in PHASE2_REQUIRED_CRITERIA if criteria.get(key) is not True]
    if failed:
        raise ProjectValidationError(
            f"Phase 2 production QA criteria are not all true: {failed}"
        )
    tests_run = status.get("tests_run")
    if not isinstance(tests_run, list) or not tests_run:
        raise ProjectValidationError("Phase 2 does not record its mandatory tests")
    required_outputs = {
        Path(config["paths"][key]).as_posix()
        for key in (
            "security_master",
            "company_alias",
            "disclosure_raw",
            "resolution_audit",
            "raw_manifest",
        )
    }
    if not required_outputs.issubset(_normalised_status_paths(status, "outputs")):
        raise ProjectValidationError(
            "Phase 2 omits canonical outputs or the raw manifest"
        )
    raw_manifest = Path(config["paths"]["raw_manifest"]).as_posix()
    if raw_manifest not in _normalised_status_paths(status, "inputs"):
        raise ProjectValidationError("Phase 2 inputs omit the verified raw manifest")
    metrics = status.get("metrics")
    if not isinstance(metrics, dict):
        raise ProjectValidationError("Phase 2 metrics are missing")
    artifact_hashes = metrics.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ProjectValidationError("Phase 2 artifact hashes are missing")
    hashed_paths = {
        Path(relative).as_posix()
        for relative in artifact_hashes
        if isinstance(relative, str)
    }
    missing_hashes = sorted(required_outputs - hashed_paths)
    if missing_hashes:
        raise ProjectValidationError(
            "Phase 2 artifact hashes omit canonical outputs or raw manifest: "
            f"{missing_hashes}"
        )
    _validate_artifact_hashes(root, artifact_hashes, "Phase 2")


def _validate_phase3_artifact_evidence(
    root: Path,
    status: dict[str, Any],
    config: dict[str, Any],
) -> None:
    metrics = status.get("metrics")
    if not isinstance(metrics, dict):
        raise ProjectValidationError("Phase 3 metrics are missing")
    artifact_hashes = metrics.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ProjectValidationError("Phase 3 artifact hashes are missing")
    required = {
        config["paths"]["raw_manifest"],
        config["paths"]["disclosure_raw"],
        config["paths"]["resolution_audit"],
        config["paths"]["security_master"],
        config["paths"]["supply_chain_edge"],
        *(f"reports/phase3_coverage/{name}" for name in MANDATORY_PHASE3_REPORTS),
    }
    criteria = status.get("criteria")
    if isinstance(criteria, dict) and criteria.get("research_gate_failed") is True:
        required.update(
            {
                "reports/final/FEASIBILITY_REPORT.md",
                "reports/final/results_summary.json",
            }
        )
    hashed_paths = {
        Path(relative).as_posix()
        for relative in artifact_hashes
        if isinstance(relative, str)
    }
    missing = sorted(required - hashed_paths)
    if missing:
        raise ProjectValidationError(
            f"Phase 3 artifact hashes omit required evidence: {missing}"
        )
    _validate_artifact_hashes(root, artifact_hashes, "Phase 3")


def _validate_readme(root: Path) -> None:
    readme = (root / "README.md").read_text(encoding="utf-8")
    for command in (
        "python -m pytest -q tests/integration/test_tiny_e2e_pipeline.py",
        "python scripts/run_pipeline.py",
        "python scripts/run_pipeline.py --through-phase 3",
        "python scripts/validate_all.py",
    ):
        if command not in readme:
            raise ProjectValidationError(f"README is missing exact command: {command}")


def _run_mandatory_tests(root: Path) -> None:
    missing = missing_mandatory_test_artifacts(root)
    if missing:
        raise ProjectValidationError(
            f"Mandatory test registry or tiny fixture is incomplete: {list(missing)}"
        )
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/unit",
            "tests/integration",
            "tests/regression",
        ],
        cwd=root,
        check=False,
    )
    if tests.returncode != 0:
        raise ProjectValidationError("Mandatory test registry failed")


def _validate_run_manifest(
    *,
    root: Path,
    config: dict[str, Any],
    raw_manifest: Path,
    statuses: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    manifest = _read_json(root / "reports" / "run_manifest.json")
    if manifest.get("config_sha256") != config_sha256(config):
        raise ProjectValidationError("Run manifest config hash mismatch")
    if manifest.get("raw_data_manifest_sha256") != sha256_file(raw_manifest):
        raise ProjectValidationError("Run manifest raw-data hash mismatch")
    current_source_hash = source_tree_manifest_sha256(source_tree_manifest(root))
    if manifest.get("source_code_sha256") != current_source_hash:
        raise ProjectValidationError("Run manifest source-code hash mismatch")
    if manifest.get("random_seed") != config["project"]["random_state"]:
        raise ProjectValidationError("Run manifest random seed mismatch")
    try:
        timestamp = datetime.fromisoformat(str(manifest["run_timestamp_utc"]))
    except (KeyError, ValueError) as exc:
        raise ProjectValidationError("Run manifest has no valid timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ProjectValidationError("Run manifest timestamp must include a UTC offset")
    versions = manifest.get("package_versions")
    if not isinstance(versions, dict) or not versions:
        raise ProjectValidationError("Run manifest package versions are missing")
    expected_states = {
        str(phase): status["state"] for phase, status in statuses.items()
    }
    if manifest.get("phase_states") != expected_states:
        raise ProjectValidationError("Run manifest phase states disagree with statuses")
    final_paths = manifest.get("final_artifact_paths")
    if not isinstance(final_paths, list) or not all(
        isinstance(path, str) for path in final_paths
    ):
        raise ProjectValidationError("Run manifest final artifact paths are invalid")
    return manifest


def _validate_phase3_reports(root: Path) -> dict[str, Any]:
    report_dir = root / "reports" / "phase3_coverage"
    missing = [
        name
        for name in MANDATORY_PHASE3_REPORTS
        if not (report_dir / name).is_file() or (report_dir / name).stat().st_size == 0
    ]
    if missing:
        raise ProjectValidationError(f"Missing mandatory Phase-3 reports: {missing}")
    payload = _read_json(report_dir / "summary.json")
    if not isinstance(payload.get("coverage"), dict) or not isinstance(
        payload.get("gate"), dict
    ):
        raise ProjectValidationError("Phase-3 summary lacks coverage or gate")
    return payload


def _validate_path_a_phase3(
    *,
    root: Path,
    config: dict[str, Any],
    status: dict[str, Any],
    phase3_summary: dict[str, Any],
    results: dict[str, Any],
    run_manifest: dict[str, Any],
) -> None:
    criteria = status.get("criteria")
    if not isinstance(criteria, dict):
        raise ProjectValidationError("Phase-3 PASS criteria are missing")
    expected_criteria = {
        "real_data_gate_evaluated": True,
        "phase3_tests_pass": True,
        "graph_integrity_passed": True,
        "frozen_coverage_passed": True,
        "research_gate_failed": False,
        "holdout_integrity_violation": False,
    }
    inconsistent = [
        key
        for key, expected in expected_criteria.items()
        if criteria.get(key) is not expected
    ]
    if inconsistent:
        raise ProjectValidationError(
            f"Phase-3 PASS criteria are inconsistent: {inconsistent}"
        )

    metrics = status.get("metrics")
    if (
        not isinstance(metrics, dict)
        or type(metrics.get("phase3_test_return_code")) is not int
    ):
        raise ProjectValidationError("Phase-3 mandatory pytest evidence is invalid")
    if metrics["phase3_test_return_code"] != 0:
        raise ProjectValidationError("Phase-3 mandatory pytest return code is not zero")
    tests_run = status.get("tests_run")
    if not isinstance(tests_run, list) or not all(
        isinstance(command, str) for command in tests_run
    ):
        raise ProjectValidationError("Phase-3 test-command evidence is invalid")
    test_evidence = "\n".join(tests_run)
    if "-m pytest" not in test_evidence or any(
        test not in test_evidence for test in REQUIRED_PHASE3_TESTS
    ):
        raise ProjectValidationError(
            "Phase-3 status does not record the executed mandatory pytest command"
        )

    gate = phase3_summary.get("gate")
    if not isinstance(gate, dict):
        raise ProjectValidationError("Phase-3 summary gate is missing")
    integrity_checks = gate.get("integrity_checks")
    coverage_checks = gate.get("coverage_checks")
    checks = gate.get("checks")
    if not all(
        isinstance(value, dict) for value in (integrity_checks, coverage_checks, checks)
    ):
        raise ProjectValidationError("Phase-3 gate checks are missing")
    if not _check_group_passed(
        integrity_checks,
        required_keys=INTEGRITY_CHECK_KEYS,
        label="Phase-3 integrity gate",
    ) or not _check_group_passed(
        coverage_checks,
        required_keys=COVERAGE_CHECK_KEYS,
        label="Phase-3 coverage gate",
    ):
        raise ProjectValidationError("Phase-3 PASS contains a failed gate check")
    if (
        gate.get("status") != "PASS"
        or gate.get("outcome") != "PROCEED"
        or gate.get("passed") is not True
        or gate.get("integrity_passed") is not True
        or gate.get("coverage_passed") is not True
    ):
        raise ProjectValidationError("Phase-3 PASS gate summary is inconsistent")
    if criteria.get("gate_checks") != checks:
        raise ProjectValidationError("Phase-3 status and summary gate checks disagree")
    if metrics.get("coverage") != phase3_summary.get("coverage"):
        raise ProjectValidationError("Phase-3 status and summary coverage disagree")

    required_inputs = {
        config["paths"]["disclosure_raw"],
        config["paths"]["resolution_audit"],
        config["paths"]["security_master"],
        config["paths"]["raw_manifest"],
    }
    if not required_inputs.issubset(_normalised_status_paths(status, "inputs")):
        raise ProjectValidationError("Phase 3 did not record Phase-2 canonical inputs")
    required_outputs = {
        config["paths"]["supply_chain_edge"],
        *(f"reports/phase3_coverage/{name}" for name in MANDATORY_PHASE3_REPORTS),
    }
    recorded_outputs = _normalised_status_paths(status, "outputs")
    if not required_outputs.issubset(recorded_outputs):
        raise ProjectValidationError("Phase-3 PASS omits mandatory graph outputs")
    stale_path_b_outputs = {
        "reports/final/FEASIBILITY_REPORT.md",
        "reports/final/results_summary.json",
    } & recorded_outputs
    if stale_path_b_outputs:
        raise ProjectValidationError(
            f"Phase-3 PASS retains Path-B outputs: {sorted(stale_path_b_outputs)}"
        )

    if results.get("research_conclusion") not in PATH_A_CONCLUSIONS:
        raise ProjectValidationError(
            "Path A conclusion must be POSITIVE, NULL, or INCONCLUSIVE"
        )
    if results.get("terminal_path") == "PATH_B_DATA_INFEASIBILITY":
        raise ProjectValidationError("Path A retains a Path-B terminal label")
    feasibility = root / "reports" / "final" / "FEASIBILITY_REPORT.md"
    if feasibility.exists():
        raise ProjectValidationError("Path A retains a stale feasibility report")

    missing_final = [
        relative
        for relative in PATH_A_REQUIRED_FINAL_ARTIFACTS
        if not (root / relative).is_file() or (root / relative).stat().st_size == 0
    ]
    if missing_final:
        raise ProjectValidationError(
            f"Path A is missing required final artifacts: {missing_final}"
        )
    final_artifact_paths = set(run_manifest["final_artifact_paths"])
    if not set(PATH_A_REQUIRED_FINAL_ARTIFACTS).issubset(final_artifact_paths):
        raise ProjectValidationError("Run manifest omits Path-A final artifacts")
    if "reports/final/FEASIBILITY_REPORT.md" in final_artifact_paths:
        raise ProjectValidationError("Run manifest retains a stale Path-B artifact")


def _validate_path_b(
    *,
    root: Path,
    config: dict[str, Any],
    statuses: dict[int, dict[str, Any]],
    phase3_summary: dict[str, Any],
    results: dict[str, Any],
    run_manifest: dict[str, Any],
) -> tuple[str, str]:
    status = statuses[3]
    criteria = status.get("criteria", {})
    if criteria.get("real_data_gate_evaluated") is not True:
        raise ProjectValidationError("Phase 3 did not evaluate the real-data gate")
    if criteria.get("holdout_integrity_violation") is not False:
        raise ProjectValidationError("Phase 3 stopped on a holdout integrity violation")
    if criteria.get("phase3_tests_pass") is not True:
        raise ProjectValidationError("Phase-3 mandatory tests did not pass")
    test_return_code = status.get("metrics", {}).get("phase3_test_return_code")
    if type(test_return_code) is not int or test_return_code != 0:
        raise ProjectValidationError("Phase-3 mandatory pytest return code is not zero")
    tests_run = status.get("tests_run")
    if not isinstance(tests_run, list) or not all(
        isinstance(command, str) for command in tests_run
    ):
        raise ProjectValidationError("Phase-3 test-command evidence is invalid")
    test_evidence = "\n".join(tests_run)
    if "-m pytest" not in test_evidence or any(
        test not in test_evidence for test in REQUIRED_PHASE3_TESTS
    ):
        raise ProjectValidationError(
            "Phase-3 status does not record the executed mandatory pytest command"
        )
    if criteria.get("graph_integrity_passed") is not True:
        raise ProjectValidationError("Phase-3 graph integrity did not pass")
    if criteria.get("frozen_coverage_passed") is not False:
        raise ProjectValidationError("Path B requires a frozen coverage failure")
    if criteria.get("research_gate_failed") is not True:
        raise ProjectValidationError("Phase-3 FAIL is not a research-gate failure")

    gate = phase3_summary["gate"]
    checks = gate.get("checks")
    integrity_checks = gate.get("integrity_checks")
    coverage_checks = gate.get("coverage_checks")
    if (
        not isinstance(checks, dict)
        or not isinstance(integrity_checks, dict)
        or not isinstance(coverage_checks, dict)
    ):
        raise ProjectValidationError("Phase-3 gate checks are missing")
    integrity_clean = _check_group_passed(
        integrity_checks,
        required_keys=INTEGRITY_CHECK_KEYS,
        label="Phase-3 integrity gate",
    )
    coverage_clean = _check_group_passed(
        coverage_checks,
        required_keys=COVERAGE_CHECK_KEYS,
        label="Phase-3 coverage gate",
    )
    coverage_failed = not coverage_clean
    if not integrity_clean or gate.get("integrity_passed") is not True:
        raise ProjectValidationError("INFEASIBLE_DATA requires clean graph integrity")
    if not coverage_failed or gate.get("coverage_passed") is not False:
        raise ProjectValidationError("INFEASIBLE_DATA requires failed frozen coverage")
    if gate.get("passed") is not False or gate.get("outcome") != "INFEASIBLE_DATA":
        raise ProjectValidationError("Phase-3 gate outcome is not INFEASIBLE_DATA")
    if criteria.get("gate_checks") != checks:
        raise ProjectValidationError("Phase-3 status and summary gate checks disagree")
    if status.get("metrics", {}).get("coverage") != phase3_summary["coverage"]:
        raise ProjectValidationError("Phase-3 status and summary coverage disagree")
    required_inputs = {
        config["paths"]["disclosure_raw"],
        config["paths"]["resolution_audit"],
        config["paths"]["security_master"],
        config["paths"]["raw_manifest"],
    }
    if not required_inputs.issubset(set(status.get("inputs", []))):
        raise ProjectValidationError("Phase 3 did not record Phase-2 canonical inputs")

    if results.get("research_conclusion") != "INFEASIBLE_DATA":
        raise ProjectValidationError(
            "Phase-3 coverage FAIL must trace to INFEASIBLE_DATA"
        )
    if results.get("strategy_evidence") != "NOT_EVALUATED":
        raise ProjectValidationError("Path B strategy evidence must be NOT_EVALUATED")
    if results.get("terminal_path") != "PATH_B_DATA_INFEASIBILITY":
        raise ProjectValidationError("Path B terminal-path label is missing")
    if results.get("last_completed_empirical_gate") != 3:
        raise ProjectValidationError(
            "Path B must identify Phase 3 as the terminal gate"
        )
    if results.get("coverage") != phase3_summary["coverage"]:
        raise ProjectValidationError("Results and Phase-3 coverage disagree")
    if results.get("phase3_gate") != gate:
        raise ProjectValidationError("Results and Phase-3 gate disagree")

    required_hashes = {
        config["paths"]["raw_manifest"],
        config["paths"]["disclosure_raw"],
        config["paths"]["resolution_audit"],
        config["paths"]["security_master"],
        config["paths"]["supply_chain_edge"],
        "reports/final/FEASIBILITY_REPORT.md",
        "reports/phase3_coverage/summary.json",
        "reports/phase3_coverage/yearly_coverage.csv",
    }
    claim_hashes = results.get("artifact_hashes")
    if not isinstance(claim_hashes, dict) or not required_hashes.issubset(claim_hashes):
        raise ProjectValidationError("Path B supporting artifact hashes are incomplete")

    feasibility_path = root / "reports" / "final" / "FEASIBILITY_REPORT.md"
    if (
        not feasibility_path.is_file()
        or "INFEASIBLE_DATA" not in feasibility_path.read_text(encoding="utf-8")
    ):
        raise ProjectValidationError(
            "INFEASIBLE_DATA feasibility report is missing or invalid"
        )
    required_final = set(PATH_B_FINAL_ARTIFACTS)
    final_artifact_paths = set(run_manifest["final_artifact_paths"])
    if not required_final.issubset(final_artifact_paths):
        raise ProjectValidationError("Run manifest omits Path-B final artifacts")
    unexpected_final_reports = sorted(
        path
        for path in final_artifact_paths
        if path.startswith("reports/final/") and path not in required_final
    )
    if unexpected_final_reports:
        raise ProjectValidationError(
            f"Run manifest includes non-Path-B final reports: {unexpected_final_reports}"
        )
    required_outputs = {
        config["paths"]["supply_chain_edge"],
        *PATH_B_FINAL_ARTIFACTS,
        *(f"reports/phase3_coverage/{name}" for name in MANDATORY_PHASE3_REPORTS),
    }
    if not required_outputs.issubset(set(status.get("outputs", []))):
        raise ProjectValidationError("Phase-3 status omits mandatory Path-B outputs")

    if any(phase in statuses for phase in range(4, 12)):
        raise ProjectValidationError("Path B must stop before Phase 4")
    forbidden = [
        root / config["paths"]["experiment_registry"],
        root / config["paths"]["research_freeze"],
        root / "reports" / "holdout" / "holdout_run_manifest.json",
        root / "BLOCKER_REPORT.md",
        root / "reports" / "final" / "FINAL_REPORT.md",
        root / "reports" / "final" / "EXECUTIVE_SUMMARY.md",
        *[
            root / config["paths"][name]
            for name in ("equity_daily", "returns_daily", "signal_daily")
        ],
    ]
    present = [path.relative_to(root).as_posix() for path in forbidden if path.exists()]
    if present:
        raise ProjectValidationError(
            f"Forbidden post-Phase-3 Path-B artifacts exist: {present}"
        )
    return "INFEASIBLE_DATA", "NOT_EVALUATED"


def validate_project(root: Path, config_path: Path) -> tuple[str, str]:
    config = load_config(config_path)
    current_config_hash = config_sha256(config)
    statuses = _read_statuses(root)
    if 1 not in statuses or statuses[1]["state"] != "PASS":
        raise ProjectValidationError("Phase 1 is not PASS")
    if 2 not in statuses:
        raise ProjectValidationError("Phase 2 status is missing")
    if statuses[2]["state"] == "BLOCKED":
        raise ProjectValidationError(
            "Phase 2 is externally blocked; empirical project is incomplete"
        )
    if statuses[2]["state"] != "PASS":
        raise ProjectValidationError(f"Phase 2 is {statuses[2]['state']}")
    if 3 not in statuses:
        raise ProjectValidationError("Phase 3 status is missing")
    current_source_hash = source_tree_manifest_sha256(source_tree_manifest(root))
    for phase in (1, 2, 3):
        if statuses[phase].get("config_sha256") != current_config_hash:
            raise ProjectValidationError(f"Phase {phase} used a different config")
        if statuses[phase].get("source_tree_sha256") != current_source_hash:
            raise ProjectValidationError(f"Phase {phase} used different source code")
    _validate_phase2_evidence(root, statuses[2], config)
    _validate_phase3_artifact_evidence(root, statuses[3], config)

    raw_manifest = root / config["paths"]["raw_manifest"]
    raw_dir = config["paths"].get("raw_dir")
    if raw_dir is None:  # compatibility for isolated validator unit fixtures
        verify_raw_manifest(raw_manifest)
    else:
        verify_raw_manifest(
            raw_manifest,
            inventory_root=root / raw_dir / "official_sources",
        )
    _validate_canonical_tables(
        root,
        config,
        [
            "security_master",
            "company_alias",
            "disclosure_raw",
            "resolution_audit",
            "supply_chain_edge",
        ],
    )
    phase3_summary = _validate_phase3_reports(root)
    _validate_phase3_thresholds(config, phase3_summary)

    results_path = root / "reports" / "final" / "results_summary.json"
    results = _read_json(results_path)
    conclusion = results.get("research_conclusion")
    strategy = results.get("strategy_evidence")
    if (
        conclusion not in ALLOWED_CONCLUSIONS
        or strategy not in ALLOWED_STRATEGY_EVIDENCE
    ):
        raise ProjectValidationError("Final evidence labels are invalid")
    claim_hashes = results.get("artifact_hashes")
    if not isinstance(claim_hashes, dict) or not claim_hashes:
        raise ProjectValidationError(
            "Final numerical claims do not provide artifact hashes"
        )
    _validate_artifact_hashes(root, claim_hashes, "final claim")
    _validate_readme(root)
    run_manifest = _validate_run_manifest(
        root=root,
        config=config,
        raw_manifest=raw_manifest,
        statuses=statuses,
    )
    _run_mandatory_tests(root)

    if statuses[3]["state"] == "FAIL":
        return _validate_path_b(
            root=root,
            config=config,
            statuses=statuses,
            phase3_summary=phase3_summary,
            results=results,
            run_manifest=run_manifest,
        )
    if statuses[3]["state"] != "PASS":
        raise ProjectValidationError(f"Phase 3 is {statuses[3]['state']}")

    _validate_path_a_phase3(
        root=root,
        config=config,
        status=statuses[3],
        phase3_summary=phase3_summary,
        results=results,
        run_manifest=run_manifest,
    )

    for phase in range(4, 12):
        if phase not in statuses:
            raise ProjectValidationError(f"Phase {phase} status is missing")
        allowed = {"PASS"}
        if phase == 8:
            allowed.add("SKIPPED_BY_DESIGN")
        if statuses[phase]["state"] not in allowed:
            raise ProjectValidationError(f"Phase {phase} is {statuses[phase]['state']}")
        if statuses[phase].get("config_sha256") != current_config_hash:
            raise ProjectValidationError(f"Phase {phase} used a different config")
        if statuses[phase].get("source_tree_sha256") != current_source_hash:
            raise ProjectValidationError(f"Phase {phase} used different source code")

    _validate_canonical_tables(
        root,
        config,
        ["equity_daily", "returns_daily", "signal_daily"],
    )
    registry = root / config["paths"]["experiment_registry"]
    if not registry.is_file():
        raise ProjectValidationError("Experiment registry is missing")

    freeze_path = root / config["paths"]["research_freeze"]
    freeze = _read_json(freeze_path)
    if freeze.get("config_sha256") != config_sha256(config):
        raise ProjectValidationError("Research freeze config hash mismatch")
    if freeze.get("raw_data_manifest_sha256") != sha256_file(raw_manifest):
        raise ProjectValidationError("Research freeze raw-manifest hash mismatch")
    validation_hashes = freeze.get("validation_artifact_hashes")
    if not isinstance(validation_hashes, dict):
        raise ProjectValidationError("Research freeze validation hashes are missing")
    _validate_artifact_hashes(root, validation_hashes, "freeze")

    holdout_manifest_path = root / "reports" / "holdout" / "holdout_run_manifest.json"
    holdout = _read_json(holdout_manifest_path)
    if holdout.get("freeze_sha256") != sha256_file(freeze_path):
        raise ProjectValidationError("Holdout freeze hash mismatch")
    if holdout.get("config_sha256") != config_sha256(config):
        raise ProjectValidationError("Holdout config hash mismatch")

    return str(conclusion), str(strategy)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a completed V2 research package."
    )
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        conclusion, strategy = validate_project(root, config_path)
    except (OSError, TypeError, ValueError, ProjectValidationError) as exc:
        print("PROJECT VALIDATION: INCOMPLETE")
        print(f"REASON: {exc}")
        return 1
    print("PROJECT VALIDATION: PASS")
    print(f"RESEARCH CONCLUSION: {conclusion}")
    print(f"STRATEGY EVIDENCE: {strategy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
