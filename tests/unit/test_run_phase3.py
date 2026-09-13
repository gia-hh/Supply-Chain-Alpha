from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_phase3_coverage as phase3
from supply_chain_alpha.data.disclosures import sha256_file
from supply_chain_alpha.data.schemas import SECURITY_MASTER
from supply_chain_alpha.pipeline import ExitCode
from supply_chain_alpha.utils.status import PhaseState


def test_compact_gate_replaces_duplicated_document_rows_with_csv_reference() -> None:
    original = {
        "status": "PASS",
        "diagnostics": {
            "yearly_coverage": [{"year": 2020}],
            "source_documents": [{"document_id": "D1"}, {"document_id": "D2"}],
        },
    }

    compact = phase3._compact_gate_for_artifacts(original)

    assert compact["diagnostics"]["source_documents"] == {
        "artifact": "source_document_diagnostics.csv",
        "row_count": 2,
    }
    assert compact["diagnostics"]["yearly_coverage"] == [{"year": 2020}]
    assert isinstance(original["diagnostics"]["source_documents"], list)


def _gate(
    *, integrity_passed: bool, coverage_passed: bool, outcome: str
) -> dict[str, object]:
    integrity_checks = {
        key: {"passed": integrity_passed} for key in phase3.INTEGRITY_CHECK_KEYS
    }
    coverage_checks = {
        key: {"passed": coverage_passed} for key in phase3.COVERAGE_CHECK_KEYS
    }
    checks = {**coverage_checks, **integrity_checks}
    return {
        "passed": integrity_passed and coverage_passed,
        "integrity_passed": integrity_passed,
        "coverage_passed": coverage_passed,
        "outcome": outcome,
        "checks": checks,
        "integrity_checks": integrity_checks,
        "coverage_checks": coverage_checks,
    }


def _path_b_archive_config() -> dict[str, object]:
    return {
        "paths": {
            "experiment_registry": "reports/experiment_registry.csv",
            "research_freeze": "reports/research_freeze.json",
            "equity_daily": "data/processed/equity_daily.parquet",
            "returns_daily": "data/processed/returns_daily.parquet",
            "signal_daily": "data/processed/signal_daily.parquet",
        }
    }


def _prerequisite_payloads() -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    config: dict[str, object] = {
        "paths": {
            "security_master": "data/processed/security_master.parquet",
            "company_alias": "data/processed/company_alias.parquet",
            "disclosure_raw": "data/processed/disclosure_raw.parquet",
            "resolution_audit": "data/processed/resolution_audit.parquet",
            "raw_manifest": "data/raw/MANIFEST.json",
        }
    }
    phase2_outputs = list(config["paths"].values())  # type: ignore[union-attr]
    statuses = {
        1: {
            "state": "PASS",
            "config_sha256": "config-hash",
            "source_tree_sha256": "source-hash",
        },
        2: {
            "state": "PASS",
            "config_sha256": "config-hash",
            "source_tree_sha256": "source-hash",
            "criteria": {key: True for key in phase3.PHASE2_REQUIRED_CRITERIA},
            "tests_run": ["-m pytest -q tests/unit/test_phase2_quality.py"],
            "inputs": ["data/raw/MANIFEST.json"],
            "outputs": phase2_outputs,
        },
    }
    return config, statuses


def _mock_prerequisite_statuses(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    statuses: dict[int, dict[str, object]],
) -> None:
    artifact_hashes: dict[str, str] = {}
    for relative in statuses[2]["outputs"]:  # type: ignore[index]
        path = root / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact:{relative}", encoding="utf-8")
        artifact_hashes[Path(str(relative)).as_posix()] = sha256_file(path)
    statuses[2]["metrics"] = {"artifact_hashes": artifact_hashes}
    for phase in statuses:
        (root / f"phase_{phase}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        phase3,
        "phase_status_path",
        lambda project_root, phase: root / f"phase_{phase}.json",
    )
    monkeypatch.setattr(
        phase3,
        "read_phase_status",
        lambda path: statuses[int(path.stem.rsplit("_", 1)[1])],
    )
    monkeypatch.setattr(phase3, "config_sha256", lambda _: "config-hash")
    monkeypatch.setattr(phase3, "source_tree_manifest", lambda _: {})
    monkeypatch.setattr(phase3, "source_tree_manifest_sha256", lambda _: "source-hash")


@pytest.mark.parametrize(
    ("gate", "expected"),
    [
        (
            _gate(
                integrity_passed=True,
                coverage_passed=True,
                outcome="PROCEED",
            ),
            (PhaseState.PASS, ExitCode.SUCCESS, False),
        ),
        (
            _gate(
                integrity_passed=False,
                coverage_passed=True,
                outcome="ENGINEERING_FAILURE",
            ),
            (PhaseState.FAIL, ExitCode.ENGINEERING_FAILURE, False),
        ),
        (
            _gate(
                integrity_passed=True,
                coverage_passed=False,
                outcome="INFEASIBLE_DATA",
            ),
            (PhaseState.FAIL, ExitCode.RESEARCH_GATE_FAILED, True),
        ),
    ],
)
def test_gate_classification_has_exact_state_exit_and_path_b_flag(
    gate: dict[str, object], expected: tuple[PhaseState, ExitCode, bool]
) -> None:
    assert phase3._classify_gate(gate) == expected


def test_integrity_failure_cannot_be_mislabeled_as_infeasible_data() -> None:
    gate = _gate(
        integrity_passed=False,
        coverage_passed=False,
        outcome="INFEASIBLE_DATA",
    )
    with pytest.raises(ValueError, match="ENGINEERING_FAILURE"):
        phase3._classify_gate(gate)


def test_gate_rejects_none_for_non_deferred_integrity_check() -> None:
    gate = _gate(integrity_passed=True, coverage_passed=True, outcome="PROCEED")
    gate["integrity_checks"]["timing_violations"] = {"passed": None}  # type: ignore[index]

    with pytest.raises(ValueError, match="invalid result"):
        phase3._classify_gate(gate)


@pytest.mark.parametrize(
    "check_name",
    (
        "invalid_counterparty_mentions",
        "invalid_resolution_audit_names",
        "resolution_status_classification_mismatches",
    ),
)
def test_name_boundary_integrity_checks_are_mandatory(check_name: str) -> None:
    assert check_name in phase3.INTEGRITY_CHECK_KEYS
    gate = _gate(integrity_passed=True, coverage_passed=True, outcome="PROCEED")
    gate["integrity_checks"][check_name] = {"passed": False}  # type: ignore[index]

    with pytest.raises(ValueError, match="integrity summary disagrees"):
        phase3._classify_gate(gate)


def test_phase2_audit_must_cover_exact_disclosure_keys() -> None:
    mentions = pd.DataFrame(
        {
            "source_document_id": ["D1", "D2"],
            "source_company_id": ["S1", "S2"],
            "counterparty_raw_name": ["A", "B"],
        }
    )
    audit = mentions.iloc[[0]].copy()

    with pytest.raises(ValueError, match="missing=1"):
        phase3._validate_audit_alignment(mentions, audit)

    phase3._validate_audit_alignment(mentions, mentions.copy())


def test_holdout_guard_uses_publication_time_and_local_boundary() -> None:
    mentions = pd.DataFrame(
        {
            "publication_datetime": [
                "2022-12-31T15:59:59Z",
                "2022-12-31T16:00:00Z",
            ]
        }
    )
    assert (
        phase3._holdout_row_count(
            mentions,
            holdout_start="2023-01-01",
            timezone_name="Asia/Shanghai",
        )
        == 1
    )


def test_production_publication_datetimes_require_explicit_timezones() -> None:
    mentions = pd.DataFrame(
        {
            "publication_datetime": [
                "2022-12-31T16:00:00Z",
                "2023-01-01T00:00:00+08:00",
                "2023-01-01T00:00:00",
            ]
        }
    )

    assert phase3._publication_timezone_violation_count(mentions) == 1


@pytest.mark.parametrize("criterion", phase3.PHASE2_REQUIRED_CRITERIA)
def test_phase2_prerequisite_requires_every_production_qa_criterion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, criterion: str
) -> None:
    config, statuses = _prerequisite_payloads()
    statuses[2]["criteria"][criterion] = False  # type: ignore[index]
    _mock_prerequisite_statuses(monkeypatch, tmp_path, statuses)

    with pytest.raises(ValueError, match=criterion):
        phase3._validate_phase2_prerequisite(tmp_path, config)


def test_phase2_prerequisite_requires_current_source_and_canonical_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, statuses = _prerequisite_payloads()
    _mock_prerequisite_statuses(monkeypatch, tmp_path, statuses)
    phase3._validate_phase2_prerequisite(tmp_path, config)

    statuses[1]["source_tree_sha256"] = "stale"
    with pytest.raises(ValueError, match="source hash is stale"):
        phase3._validate_phase2_prerequisite(tmp_path, config)

    statuses[1]["source_tree_sha256"] = "source-hash"
    statuses[2]["outputs"] = ["data/raw/MANIFEST.json"]
    with pytest.raises(ValueError, match="canonical outputs"):
        phase3._validate_phase2_prerequisite(tmp_path, config)


def test_phase2_prerequisite_recomputes_required_artifact_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, statuses = _prerequisite_payloads()
    _mock_prerequisite_statuses(monkeypatch, tmp_path, statuses)
    phase3._validate_phase2_prerequisite(tmp_path, config)

    disclosure = tmp_path / config["paths"]["disclosure_raw"]  # type: ignore[index]
    disclosure.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        phase3._validate_phase2_prerequisite(tmp_path, config)


def test_phase2_prerequisite_requires_hash_for_every_required_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, statuses = _prerequisite_payloads()
    _mock_prerequisite_statuses(monkeypatch, tmp_path, statuses)
    alias_path = config["paths"]["company_alias"]  # type: ignore[index]
    del statuses[2]["metrics"]["artifact_hashes"][alias_path]  # type: ignore[index]

    with pytest.raises(ValueError, match="artifact hashes omit"):
        phase3._validate_phase2_prerequisite(tmp_path, config)


def test_synthetic_mode_rejects_production_inputs_and_outputs(tmp_path: Path) -> None:
    root = tmp_path / "project"
    config = {
        "paths": {
            "disclosure_raw": "data/processed/disclosure_raw.parquet",
            "company_alias": "data/processed/company_alias.parquet",
            "security_master": "data/processed/security_master.parquet",
            "processed_dir": "data/processed",
            "interim_dir": "data/interim",
            "raw_dir": "data/raw",
            "status_dir": "reports/status",
        }
    }
    fixture_dir = root / "fixtures"
    phase3._validate_synthetic_isolation(
        root=root,
        config=config,
        mentions=fixture_dir / "mentions.csv",
        aliases=fixture_dir / "aliases.csv",
        master=fixture_dir / "master.csv",
        output_dir=root / "reports" / "trials" / "phase3_synthetic",
    )

    with pytest.raises(ValueError, match="production canonical"):
        phase3._validate_synthetic_isolation(
            root=root,
            config=config,
            mentions=root / config["paths"]["disclosure_raw"],
            aliases=fixture_dir / "aliases.csv",
            master=fixture_dir / "master.csv",
            output_dir=root / "reports" / "trials" / "phase3_synthetic",
        )
    with pytest.raises(ValueError, match="reports/trials"):
        phase3._validate_synthetic_isolation(
            root=root,
            config=config,
            mentions=fixture_dir / "mentions.csv",
            aliases=fixture_dir / "aliases.csv",
            master=fixture_dir / "master.csv",
            output_dir=root / "reports" / "phase3_coverage",
        )

    with pytest.raises(ValueError, match="production data directories"):
        phase3._validate_synthetic_isolation(
            root=root,
            config=config,
            mentions=root / "data" / "interim" / "mentions.csv",
            aliases=fixture_dir / "aliases.csv",
            master=fixture_dir / "master.csv",
            output_dir=root / "reports" / "trials" / "phase3_synthetic",
        )


def test_csv_schema_reader_preserves_string_identifiers(tmp_path: Path) -> None:
    master_path = tmp_path / "security_master.csv"
    master_path.write_text(
        "security_id,ticker,exchange,company_name,board,listing_date,delisting_date\n"
        "S1,000001,SZSE,Example,Main,2020-01-01,\n",
        encoding="utf-8",
    )

    master = phase3._read_schema_table(master_path, SECURITY_MASTER)

    assert master.loc[0, "ticker"] == "000001"


def test_required_tests_are_actually_executed(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="5 passed", stderr="")

    monkeypatch.setattr(phase3.subprocess, "run", fake_run)
    passed, details, tests_run = phase3._run_required_tests(tmp_path)

    assert passed is True
    assert details == "5 passed"
    assert len(tests_run) == 1
    assert "-m pytest -q" in tests_run[0]
    assert all(test in tests_run[0] for test in phase3.PHASE3_TEST_FILES)
    assert observed["command"][-len(phase3.PHASE3_TEST_FILES) :] == list(
        phase3.PHASE3_TEST_FILES
    )
    assert observed["kwargs"]["cwd"] == tmp_path
    assert observed["kwargs"]["check"] is False


def test_path_b_results_hash_the_feasibility_report(tmp_path: Path) -> None:
    gate = {
        "diagnostics": {
            "yearly_coverage": [
                {
                    "year": 2022,
                    "year_end_active_edges": 1,
                    "year_end_connected_stocks": 2,
                }
            ]
        },
        "checks": {"consecutive_usable_years": {"passed": False}},
    }
    report, results_path = phase3._write_feasibility_report(
        tmp_path,
        coverage={"mentions": 3},
        gate=gate,
        artifact_hashes={"reports/phase3_coverage/summary.json": "0" * 64},
    )
    results = json.loads(results_path.read_text(encoding="utf-8"))

    assert results["artifact_hashes"]["reports/final/FEASIBILITY_REPORT.md"] == (
        sha256_file(report)
    )


def test_path_b_archives_exact_incompatible_allowlist_recoverably(
    tmp_path: Path,
) -> None:
    config = _path_b_archive_config()
    candidates = phase3._path_b_archive_candidates(tmp_path, config)
    for relative in candidates:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(relative, encoding="utf-8")
    preserved = tmp_path / "data" / "processed" / "supply_chain_edge.parquet"
    preserved.write_text("current phase 3", encoding="utf-8")
    unrelated = tmp_path / "reports" / "final" / "unrelated.txt"
    unrelated.write_text("unrelated", encoding="utf-8")

    archived = phase3._archive_path_b_incompatible_artifacts(
        tmp_path,
        config,
        timestamp=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
    )

    history_dirs = list((tmp_path / "reports" / "history").iterdir())
    assert len(history_dirs) == 1
    archive_dir = history_dirs[0]
    assert archive_dir.is_dir()
    assert {path.relative_to(archive_dir).as_posix() for path in archived} == set(
        candidates
    )
    for relative in candidates:
        assert not (tmp_path / relative).exists()
        assert (archive_dir / relative).read_text(encoding="utf-8") == relative
    assert preserved.read_text(encoding="utf-8") == "current phase 3"
    assert unrelated.read_text(encoding="utf-8") == "unrelated"


def test_path_b_archive_directory_is_unique_for_same_timestamp(tmp_path: Path) -> None:
    config = _path_b_archive_config()
    feasibility = tmp_path / "reports" / "final" / "FEASIBILITY_REPORT.md"
    feasibility.parent.mkdir(parents=True)
    timestamp = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    feasibility.write_text("first", encoding="utf-8")
    first = phase3._archive_path_b_incompatible_artifacts(
        tmp_path, config, timestamp=timestamp
    )
    feasibility.write_text("second", encoding="utf-8")
    second = phase3._archive_path_b_incompatible_artifacts(
        tmp_path, config, timestamp=timestamp
    )

    history = tmp_path / "reports" / "history"
    first_dir = first[0].relative_to(history).parts[0]
    second_dir = second[0].relative_to(history).parts[0]
    assert first_dir != second_dir
    assert first[0].read_text(encoding="utf-8") == "first"
    assert second[0].read_text(encoding="utf-8") == "second"


def test_path_b_archive_preflight_rejects_directory_without_partial_move(
    tmp_path: Path,
) -> None:
    config = _path_b_archive_config()
    results = tmp_path / "reports" / "final" / "results_summary.json"
    results.parent.mkdir(parents=True)
    results.write_text("old results", encoding="utf-8")
    (results.parent / "FINAL_REPORT.md").mkdir()

    with pytest.raises(ValueError, match="not a file"):
        phase3._archive_path_b_incompatible_artifacts(
            tmp_path,
            config,
            timestamp=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
        )

    assert results.read_text(encoding="utf-8") == "old results"
    assert not (tmp_path / "reports" / "history").exists()


def test_path_b_archive_preflight_rejects_out_of_root_config_path(
    tmp_path: Path,
) -> None:
    config = _path_b_archive_config()
    config["paths"]["equity_daily"] = "../outside.parquet"  # type: ignore[index]
    results = tmp_path / "reports" / "final" / "results_summary.json"
    results.parent.mkdir(parents=True)
    results.write_text("old results", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe"):
        phase3._archive_path_b_incompatible_artifacts(
            tmp_path,
            config,
            timestamp=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
        )

    assert results.read_text(encoding="utf-8") == "old results"
    assert not (tmp_path / "reports" / "history").exists()


def test_path_b_archive_preflight_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _path_b_archive_config()
    link = tmp_path / "reports" / "final" / "FINAL_REPORT.md"
    link.parent.mkdir(parents=True)
    link.write_text("old report", encoding="utf-8")
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == link or original_is_symlink(path),
    )

    with pytest.raises(ValueError, match="cannot be a symlink"):
        phase3._archive_path_b_incompatible_artifacts(
            tmp_path,
            config,
            timestamp=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
        )

    assert link.read_text(encoding="utf-8") == "old report"


def test_path_b_archive_failure_prevents_new_conclusion(tmp_path: Path) -> None:
    config = _path_b_archive_config()
    final_dir = tmp_path / "reports" / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "FINAL_REPORT.md").mkdir()
    support = tmp_path / "reports" / "phase3_coverage" / "summary.json"
    support.parent.mkdir(parents=True)
    support.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="not a file"):
        phase3._write_path_b_conclusion(
            tmp_path,
            config=config,
            coverage={},
            gate={},
            support_paths=[support],
            timestamp=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
        )

    assert not (final_dir / "FEASIBILITY_REPORT.md").exists()
    assert not (final_dir / "results_summary.json").exists()


def test_path_b_rerun_archives_old_conclusion_before_writing_new_one(
    tmp_path: Path,
) -> None:
    config = _path_b_archive_config()
    final_dir = tmp_path / "reports" / "final"
    final_dir.mkdir(parents=True)
    old_feasibility = final_dir / "FEASIBILITY_REPORT.md"
    old_results = final_dir / "results_summary.json"
    old_feasibility.write_text("old feasibility", encoding="utf-8")
    old_results.write_text("old results", encoding="utf-8")
    support = tmp_path / "reports" / "phase3_coverage" / "summary.json"
    support.parent.mkdir(parents=True)
    support.write_text("{}", encoding="utf-8")
    gate = {
        "diagnostics": {"yearly_coverage": []},
        "checks": {"consecutive_usable_years": {"passed": False}},
    }

    report, results_path, archived = phase3._write_path_b_conclusion(
        tmp_path,
        config=config,
        coverage={},
        gate=gate,
        support_paths=[support],
        timestamp=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
    )

    assert report.read_text(encoding="utf-8").startswith(
        "# Supply-chain graph feasibility result"
    )
    assert (
        json.loads(results_path.read_text(encoding="utf-8"))["research_conclusion"]
        == "INFEASIBLE_DATA"
    )
    assert {path.read_text(encoding="utf-8") for path in archived} == {
        "old feasibility",
        "old results",
    }


def test_phase3_pass_archives_superseded_final_artifacts(tmp_path: Path) -> None:
    final_dir = tmp_path / "reports" / "final"
    final_dir.mkdir(parents=True)
    feasibility = final_dir / "FEASIBILITY_REPORT.md"
    results = final_dir / "results_summary.json"
    unrelated = final_dir / "FINAL_REPORT.md"
    feasibility.write_text("old feasibility", encoding="utf-8")
    results.write_text(
        '{"terminal_path":"PATH_B_DATA_INFEASIBILITY"}', encoding="utf-8"
    )
    unrelated.write_text("unrelated", encoding="utf-8")

    archived = phase3._archive_superseded_final_artifacts(
        tmp_path,
        timestamp=datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
    )

    assert not feasibility.exists()
    assert not results.exists()
    assert unrelated.read_text(encoding="utf-8") == "unrelated"
    assert {path.read_text(encoding="utf-8") for path in archived} == {
        "old feasibility",
        '{"terminal_path":"PATH_B_DATA_INFEASIBILITY"}',
    }
    assert all(path.parent == tmp_path / "reports" / "history" for path in archived)


def test_phase3_pass_preserves_non_path_b_results_summary(tmp_path: Path) -> None:
    final_dir = tmp_path / "reports" / "final"
    final_dir.mkdir(parents=True)
    results = final_dir / "results_summary.json"
    results.write_text(
        '{"research_conclusion":"NULL","terminal_path":"PATH_A_FULL_EMPIRICAL"}',
        encoding="utf-8",
    )

    archived = phase3._archive_superseded_final_artifacts(
        tmp_path,
        timestamp=datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
    )

    assert archived == []
    assert results.is_file()


def test_json_safe_replaces_nan_in_nested_gate_diagnostics() -> None:
    assert phase3._json_safe({"diagnostics": [{"value": float("nan")}]}) == {
        "diagnostics": [{"value": None}]
    }
