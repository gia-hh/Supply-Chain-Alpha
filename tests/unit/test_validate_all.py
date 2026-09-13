from __future__ import annotations

from pathlib import Path

import pytest

from scripts import validate_all


def _path_b_payloads() -> tuple[
    dict[int, dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    checks = {
        **{key: {"passed": True} for key in validate_all.INTEGRITY_CHECK_KEYS},
        "consecutive_usable_years": {"passed": False},
        "usable_years_median_connected_stocks": {"passed": False},
        "pit_industry_count": {
            "actual": None,
            "passed": None,
            "deferred": True,
            "reason": "PIT_INDUSTRY_UNAVAILABLE",
        },
        "pit_largest_industry_share": {
            "actual": None,
            "passed": None,
            "deferred": True,
            "reason": "PIT_INDUSTRY_UNAVAILABLE",
        },
    }
    coverage = {"mentions": 10, "listed_to_listed_edges": 2}
    gate = {
        "passed": False,
        "integrity_passed": True,
        "coverage_passed": False,
        "outcome": "INFEASIBLE_DATA",
        "checks": checks,
        "integrity_checks": {
            key: checks[key] for key in validate_all.INTEGRITY_CHECK_KEYS
        },
        "coverage_checks": {
            key: checks[key] for key in validate_all.COVERAGE_CHECK_KEYS
        },
    }
    statuses: dict[int, dict[str, object]] = {
        1: {"state": "PASS", "source_tree_sha256": "source-hash"},
        2: {
            "state": "PASS",
            "source_tree_sha256": "source-hash",
            "criteria": {key: True for key in validate_all.PHASE2_REQUIRED_CRITERIA},
            "tests_run": ["-m pytest -q tests/unit/test_phase2_quality.py"],
            "inputs": ["data/raw/MANIFEST.json"],
            "outputs": [
                "data/processed/security_master.parquet",
                "data/processed/company_alias.parquet",
                "data/processed/disclosure_raw.parquet",
                "data/processed/resolution_audit.parquet",
                "data/raw/MANIFEST.json",
            ],
            "metrics": {
                "artifact_hashes": {
                    "data/processed/security_master.parquet": "0" * 64,
                    "data/processed/company_alias.parquet": "0" * 64,
                    "data/processed/disclosure_raw.parquet": "0" * 64,
                    "data/processed/resolution_audit.parquet": "0" * 64,
                    "data/raw/MANIFEST.json": "0" * 64,
                }
            },
        },
        3: {
            "state": "FAIL",
            "source_tree_sha256": "source-hash",
            "metrics": {
                "coverage": coverage,
                "phase3_test_return_code": 0,
                "artifact_hashes": {
                    "data/raw/MANIFEST.json": "0" * 64,
                    "data/processed/disclosure_raw.parquet": "0" * 64,
                    "data/processed/resolution_audit.parquet": "0" * 64,
                    "data/processed/security_master.parquet": "0" * 64,
                    "data/processed/supply_chain_edge.parquet": "0" * 64,
                    "reports/final/FEASIBILITY_REPORT.md": "0" * 64,
                    "reports/final/results_summary.json": "0" * 64,
                    **{
                        f"reports/phase3_coverage/{name}": "0" * 64
                        for name in validate_all.MANDATORY_PHASE3_REPORTS
                    },
                },
            },
            "criteria": {
                "real_data_gate_evaluated": True,
                "phase3_tests_pass": True,
                "graph_integrity_passed": True,
                "frozen_coverage_passed": False,
                "research_gate_failed": True,
                "holdout_integrity_violation": False,
                "gate_checks": checks,
            },
            "outputs": [
                "data/processed/supply_chain_edge.parquet",
                "reports/final/FEASIBILITY_REPORT.md",
                "reports/final/results_summary.json",
                *[
                    f"reports/phase3_coverage/{name}"
                    for name in validate_all.MANDATORY_PHASE3_REPORTS
                ],
            ],
            "tests_run": [
                "-m pytest -q " + " ".join(validate_all.REQUIRED_PHASE3_TESTS)
            ],
            "inputs": [
                "data/processed/disclosure_raw.parquet",
                "data/processed/resolution_audit.parquet",
                "data/processed/security_master.parquet",
                "data/raw/MANIFEST.json",
            ],
        },
    }
    summary = {"coverage": coverage, "gate": gate}
    results = {
        "research_conclusion": "INFEASIBLE_DATA",
        "strategy_evidence": "NOT_EVALUATED",
        "terminal_path": "PATH_B_DATA_INFEASIBILITY",
        "last_completed_empirical_gate": 3,
        "coverage": coverage,
        "phase3_gate": gate,
        "artifact_hashes": {
            "data/raw/MANIFEST.json": "0" * 64,
            "data/processed/disclosure_raw.parquet": "0" * 64,
            "data/processed/resolution_audit.parquet": "0" * 64,
            "data/processed/security_master.parquet": "0" * 64,
            "data/processed/supply_chain_edge.parquet": "0" * 64,
            "reports/final/FEASIBILITY_REPORT.md": "0" * 64,
            "reports/phase3_coverage/summary.json": "0" * 64,
            "reports/phase3_coverage/yearly_coverage.csv": "0" * 64,
        },
    }
    run_manifest = {"final_artifact_paths": list(validate_all.PATH_B_FINAL_ARTIFACTS)}
    return statuses, summary, results, run_manifest


def _path_b_config() -> dict[str, object]:
    return {
        "paths": {
            "raw_manifest": "data/raw/MANIFEST.json",
            "disclosure_raw": "data/processed/disclosure_raw.parquet",
            "resolution_audit": "data/processed/resolution_audit.parquet",
            "security_master": "data/processed/security_master.parquet",
            "company_alias": "data/processed/company_alias.parquet",
            "supply_chain_edge": "data/processed/supply_chain_edge.parquet",
            "experiment_registry": "reports/experiment_registry.csv",
            "research_freeze": "reports/research_freeze.json",
            "equity_daily": "data/processed/equity_daily.parquet",
            "returns_daily": "data/processed/returns_daily.parquet",
            "signal_daily": "data/processed/signal_daily.parquet",
        }
    }


def _path_a_payloads() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    statuses, summary, _, _ = _path_b_payloads()
    status = statuses[3]
    status["state"] = "PASS"
    status["criteria"].update(  # type: ignore[union-attr]
        {
            "frozen_coverage_passed": True,
            "research_gate_failed": False,
        }
    )
    status["outputs"] = [  # type: ignore[assignment]
        "data/processed/supply_chain_edge.parquet",
        *[
            f"reports/phase3_coverage/{name}"
            for name in validate_all.MANDATORY_PHASE3_REPORTS
        ],
    ]
    gate = summary["gate"]  # type: ignore[assignment]
    gate.update(  # type: ignore[union-attr]
        {
            "status": "PASS",
            "outcome": "PROCEED",
            "passed": True,
            "integrity_passed": True,
            "coverage_passed": True,
        }
    )
    for check in gate["coverage_checks"].values():  # type: ignore[index,union-attr]
        check.clear()
        check["passed"] = True
    results = {
        "research_conclusion": "NULL",
        "strategy_evidence": "NOT_EVALUATED",
        "terminal_path": "PATH_A_FULL_EMPIRICAL_COMPLETION",
    }
    run_manifest = {
        "final_artifact_paths": list(validate_all.PATH_A_REQUIRED_FINAL_ARTIFACTS)
    }
    return status, summary, results, run_manifest


def _write_path_a_final_artifacts(root: Path) -> None:
    for relative in validate_all.PATH_A_REQUIRED_FINAL_ARTIFACTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("final", encoding="utf-8")


def test_path_a_requires_clean_phase3_pass_and_final_artifacts(tmp_path: Path) -> None:
    status, summary, results, run_manifest = _path_a_payloads()
    _write_path_a_final_artifacts(tmp_path)

    validate_all._validate_path_a_phase3(
        root=tmp_path,
        config=_path_b_config(),
        status=status,
        phase3_summary=summary,
        results=results,
        run_manifest=run_manifest,
    )


@pytest.mark.parametrize(
    ("conclusion", "terminal_path", "message"),
    [
        ("INFEASIBLE_DATA", "PATH_A_FULL_EMPIRICAL_COMPLETION", "conclusion"),
        ("NULL", "PATH_B_DATA_INFEASIBILITY", "terminal label"),
    ],
)
def test_path_a_rejects_stale_path_b_results(
    tmp_path: Path,
    conclusion: str,
    terminal_path: str,
    message: str,
) -> None:
    status, summary, results, run_manifest = _path_a_payloads()
    results["research_conclusion"] = conclusion
    results["terminal_path"] = terminal_path
    _write_path_a_final_artifacts(tmp_path)

    with pytest.raises(validate_all.ProjectValidationError, match=message):
        validate_all._validate_path_a_phase3(
            root=tmp_path,
            config=_path_b_config(),
            status=status,
            phase3_summary=summary,
            results=results,
            run_manifest=run_manifest,
        )


def test_path_a_rejects_stale_feasibility_report(tmp_path: Path) -> None:
    status, summary, results, run_manifest = _path_a_payloads()
    _write_path_a_final_artifacts(tmp_path)
    feasibility = tmp_path / "reports" / "final" / "FEASIBILITY_REPORT.md"
    feasibility.write_text("stale", encoding="utf-8")

    with pytest.raises(validate_all.ProjectValidationError, match="stale feasibility"):
        validate_all._validate_path_a_phase3(
            root=tmp_path,
            config=_path_b_config(),
            status=status,
            phase3_summary=summary,
            results=results,
            run_manifest=run_manifest,
        )


def test_path_b_accepts_only_integrity_clean_coverage_failure(tmp_path: Path) -> None:
    statuses, summary, results, run_manifest = _path_b_payloads()
    final_dir = tmp_path / "reports" / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "FEASIBILITY_REPORT.md").write_text(
        "Research conclusion: INFEASIBLE_DATA", encoding="utf-8"
    )

    assert validate_all._validate_path_b(
        root=tmp_path,
        config=_path_b_config(),
        statuses=statuses,
        phase3_summary=summary,
        results=results,
        run_manifest=run_manifest,
    ) == ("INFEASIBLE_DATA", "NOT_EVALUATED")


def test_path_b_rejects_engineering_failure_mislabeled_as_infeasible(
    tmp_path: Path,
) -> None:
    statuses, summary, results, run_manifest = _path_b_payloads()
    statuses[3]["criteria"]["graph_integrity_passed"] = False  # type: ignore[index]
    final_dir = tmp_path / "reports" / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "FEASIBILITY_REPORT.md").write_text(
        "INFEASIBLE_DATA", encoding="utf-8"
    )

    with pytest.raises(validate_all.ProjectValidationError, match="graph integrity"):
        validate_all._validate_path_b(
            root=tmp_path,
            config=_path_b_config(),
            statuses=statuses,
            phase3_summary=summary,
            results=results,
            run_manifest=run_manifest,
        )


@pytest.mark.parametrize(
    ("criterion", "value", "message"),
    [
        ("real_data_gate_evaluated", None, "real-data gate"),
        ("holdout_integrity_violation", None, "holdout integrity"),
    ],
)
def test_path_b_requires_explicit_real_data_and_holdout_evidence(
    tmp_path: Path,
    criterion: str,
    value: object,
    message: str,
) -> None:
    statuses, summary, results, run_manifest = _path_b_payloads()
    statuses[3]["criteria"][criterion] = value  # type: ignore[index]
    final_dir = tmp_path / "reports" / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "FEASIBILITY_REPORT.md").write_text(
        "INFEASIBLE_DATA", encoding="utf-8"
    )

    with pytest.raises(validate_all.ProjectValidationError, match=message):
        validate_all._validate_path_b(
            root=tmp_path,
            config=_path_b_config(),
            statuses=statuses,
            phase3_summary=summary,
            results=results,
            run_manifest=run_manifest,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "data/processed/disclosure_raw.parquet",
        "data/processed/security_master.parquet",
    ],
)
def test_path_b_hashes_every_graph_input(tmp_path: Path, relative: str) -> None:
    statuses, summary, results, run_manifest = _path_b_payloads()
    del results["artifact_hashes"][relative]  # type: ignore[index]
    final_dir = tmp_path / "reports" / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "FEASIBILITY_REPORT.md").write_text(
        "INFEASIBLE_DATA", encoding="utf-8"
    )

    with pytest.raises(
        validate_all.ProjectValidationError, match="hashes are incomplete"
    ):
        validate_all._validate_path_b(
            root=tmp_path,
            config=_path_b_config(),
            statuses=statuses,
            phase3_summary=summary,
            results=results,
            run_manifest=run_manifest,
        )


def test_path_b_rejects_any_downstream_phase_status(tmp_path: Path) -> None:
    statuses, summary, results, run_manifest = _path_b_payloads()
    statuses[4] = {"state": "PASS"}
    final_dir = tmp_path / "reports" / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "FEASIBILITY_REPORT.md").write_text(
        "INFEASIBLE_DATA", encoding="utf-8"
    )

    with pytest.raises(validate_all.ProjectValidationError, match="before Phase 4"):
        validate_all._validate_path_b(
            root=tmp_path,
            config=_path_b_config(),
            statuses=statuses,
            phase3_summary=summary,
            results=results,
            run_manifest=run_manifest,
        )


def test_common_validation_runs_before_path_b_branch(
    monkeypatch, tmp_path: Path
) -> None:
    order: list[str] = []
    statuses, summary, _, run_manifest = _path_b_payloads()
    for status in statuses.values():
        status["config_sha256"] = "config-hash"
    config = {
        "paths": {
            "raw_manifest": "data/raw/MANIFEST.json",
            "security_master": "data/processed/security_master.parquet",
            "company_alias": "data/processed/company_alias.parquet",
            "disclosure_raw": "data/processed/disclosure_raw.parquet",
            "resolution_audit": "data/processed/resolution_audit.parquet",
            "supply_chain_edge": "data/processed/supply_chain_edge.parquet",
        },
        "project": {"random_state": 42},
    }
    results = {
        "research_conclusion": "INFEASIBLE_DATA",
        "strategy_evidence": "NOT_EVALUATED",
        "artifact_hashes": {"reports/phase3_coverage/summary.json": "0" * 64},
    }

    monkeypatch.setattr(validate_all, "load_config", lambda _: config)
    monkeypatch.setattr(validate_all, "config_sha256", lambda _: "config-hash")
    monkeypatch.setattr(validate_all, "source_tree_manifest", lambda _: {})
    monkeypatch.setattr(
        validate_all, "source_tree_manifest_sha256", lambda _: "source-hash"
    )
    monkeypatch.setattr(validate_all, "_read_statuses", lambda _: statuses)
    monkeypatch.setattr(
        validate_all,
        "verify_raw_manifest",
        lambda _: order.append("raw_manifest"),
    )
    monkeypatch.setattr(
        validate_all,
        "_validate_canonical_tables",
        lambda *args, **kwargs: order.append("canonical_tables"),
    )
    monkeypatch.setattr(
        validate_all,
        "_validate_phase3_reports",
        lambda _: order.append("phase3_reports") or summary,
    )
    monkeypatch.setattr(
        validate_all,
        "_validate_phase3_thresholds",
        lambda *args: order.append("phase3_thresholds"),
    )
    monkeypatch.setattr(validate_all, "_read_json", lambda _: results)
    monkeypatch.setattr(
        validate_all,
        "_validate_artifact_hashes",
        lambda *args, **kwargs: order.append("artifact_hashes"),
    )
    monkeypatch.setattr(
        validate_all, "_validate_readme", lambda _: order.append("readme")
    )
    monkeypatch.setattr(
        validate_all,
        "_validate_run_manifest",
        lambda **kwargs: order.append("run_manifest") or run_manifest,
    )
    monkeypatch.setattr(
        validate_all,
        "_run_mandatory_tests",
        lambda _: order.append("mandatory_tests"),
    )
    monkeypatch.setattr(
        validate_all,
        "_validate_path_b",
        lambda **kwargs: order.append("path_b") or ("INFEASIBLE_DATA", "NOT_EVALUATED"),
    )

    assert validate_all.validate_project(tmp_path, tmp_path / "config.yaml") == (
        "INFEASIBLE_DATA",
        "NOT_EVALUATED",
    )
    assert order == [
        "artifact_hashes",
        "artifact_hashes",
        "raw_manifest",
        "canonical_tables",
        "phase3_reports",
        "phase3_thresholds",
        "artifact_hashes",
        "readme",
        "run_manifest",
        "mandatory_tests",
        "path_b",
    ]


def test_artifact_hashes_reject_path_escape(tmp_path: Path) -> None:
    with pytest.raises(validate_all.ProjectValidationError, match="unsafe"):
        validate_all._validate_artifact_hashes(
            tmp_path, {"../outside.json": "0" * 64}, "final claim"
        )


def test_phase3_thresholds_must_match_current_frozen_config() -> None:
    config = {
        "graph_coverage_gate": {
            "evaluation_start": "2016-01-01",
            "evaluation_end": "2022-12-31",
            "min_consecutive_calendar_years": 3,
            "min_year_end_active_directed_edges": 200,
        },
        "information_timing": {"edge_max_age_days": 550},
    }
    expected = validate_all._expected_phase3_thresholds(config)
    summary = {"gate": {"frozen_thresholds": expected}}
    validate_all._validate_phase3_thresholds(config, summary)

    summary["gate"]["frozen_thresholds"][  # type: ignore[index]
        "min_year_end_active_directed_edges"
    ] = 199
    with pytest.raises(validate_all.ProjectValidationError, match="frozen thresholds"):
        validate_all._validate_phase3_thresholds(config, summary)


def test_path_b_rejects_none_for_non_deferred_integrity_check(
    tmp_path: Path,
) -> None:
    statuses, summary, results, run_manifest = _path_b_payloads()
    summary["gate"]["integrity_checks"]["timing_violations"] = {  # type: ignore[index]
        "passed": None
    }
    final_dir = tmp_path / "reports" / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "FEASIBILITY_REPORT.md").write_text(
        "INFEASIBLE_DATA", encoding="utf-8"
    )

    with pytest.raises(validate_all.ProjectValidationError, match="invalid result"):
        validate_all._validate_path_b(
            root=tmp_path,
            config=_path_b_config(),
            statuses=statuses,
            phase3_summary=summary,
            results=results,
            run_manifest=run_manifest,
        )


@pytest.mark.parametrize(
    "check_name",
    (
        "invalid_counterparty_mentions",
        "invalid_resolution_audit_names",
        "resolution_status_classification_mismatches",
    ),
)
def test_name_boundary_integrity_checks_are_mandatory(check_name: str) -> None:
    assert check_name in validate_all.INTEGRITY_CHECK_KEYS
    checks = {key: {"passed": True} for key in validate_all.INTEGRITY_CHECK_KEYS}
    checks[check_name] = {"passed": False}

    assert not validate_all._check_group_passed(
        checks,
        required_keys=validate_all.INTEGRITY_CHECK_KEYS,
        label="Phase-3 integrity gate",
    )


@pytest.mark.parametrize("criterion", validate_all.PHASE2_REQUIRED_CRITERIA)
def test_phase2_evidence_requires_every_production_qa_criterion(
    criterion: str,
) -> None:
    statuses, _, _, _ = _path_b_payloads()
    statuses[2]["criteria"][criterion] = False  # type: ignore[index]

    with pytest.raises(validate_all.ProjectValidationError, match=criterion):
        validate_all._validate_phase2_evidence(Path("."), statuses[2], _path_b_config())


def test_phase2_evidence_recomputes_all_required_artifact_hashes(
    tmp_path: Path,
) -> None:
    statuses, _, _, _ = _path_b_payloads()
    status = statuses[2]
    hashes: dict[str, str] = {}
    for relative in status["outputs"]:  # type: ignore[index]
        path = tmp_path / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact:{relative}", encoding="utf-8")
        hashes[Path(str(relative)).as_posix()] = validate_all.sha256_file(path)
    status["metrics"] = {"artifact_hashes": hashes}

    validate_all._validate_phase2_evidence(
        tmp_path,
        status,
        _path_b_config(),
    )

    disclosure = tmp_path / "data" / "processed" / "disclosure_raw.parquet"
    disclosure.write_text("tampered", encoding="utf-8")
    with pytest.raises(validate_all.ProjectValidationError, match="hash mismatch"):
        validate_all._validate_phase2_evidence(
            tmp_path,
            status,
            _path_b_config(),
        )


def test_phase3_evidence_requires_hash_for_every_input_and_report(
    tmp_path: Path,
) -> None:
    statuses, _, _, _ = _path_b_payloads()
    del statuses[3]["metrics"]["artifact_hashes"][  # type: ignore[index]
        "data/processed/security_master.parquet"
    ]

    with pytest.raises(validate_all.ProjectValidationError, match="required evidence"):
        validate_all._validate_phase3_artifact_evidence(
            tmp_path,
            statuses[3],
            _path_b_config(),
        )


def test_path_b_rejects_old_final_reports_and_manifest_claims(tmp_path: Path) -> None:
    statuses, summary, results, run_manifest = _path_b_payloads()
    final_dir = tmp_path / "reports" / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "FEASIBILITY_REPORT.md").write_text(
        "INFEASIBLE_DATA", encoding="utf-8"
    )
    (final_dir / "FINAL_REPORT.md").write_text("stale", encoding="utf-8")

    with pytest.raises(validate_all.ProjectValidationError, match="Forbidden"):
        validate_all._validate_path_b(
            root=tmp_path,
            config=_path_b_config(),
            statuses=statuses,
            phase3_summary=summary,
            results=results,
            run_manifest=run_manifest,
        )

    (final_dir / "FINAL_REPORT.md").unlink()
    run_manifest["final_artifact_paths"].append("reports/final/FINAL_REPORT.md")  # type: ignore[union-attr]
    with pytest.raises(validate_all.ProjectValidationError, match="non-Path-B"):
        validate_all._validate_path_b(
            root=tmp_path,
            config=_path_b_config(),
            statuses=statuses,
            phase3_summary=summary,
            results=results,
            run_manifest=run_manifest,
        )
