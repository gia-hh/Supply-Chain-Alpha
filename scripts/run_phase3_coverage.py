from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from supply_chain_alpha.data.disclosures import (
    load_structured_mentions,
    sha256_file,
    verify_raw_manifest,
)
from supply_chain_alpha.data.io import read_table, write_canonical_parquet
from supply_chain_alpha.data.schemas import (
    COMPANY_ALIAS,
    RESOLUTION_AUDIT,
    SECURITY_MASTER,
    SUPPLY_CHAIN_EDGE,
    validate_table,
)
from supply_chain_alpha.entities.resolve import ResolutionConfig, resolve_mentions
from supply_chain_alpha.graph.coverage import (
    degree_diagnostics,
    edge_age_distribution,
    evaluate_phase3_gate,
    listed_edge_intervals,
    node_role_diagnostics,
    source_document_diagnostics,
    summarize_coverage,
    summary_to_dict,
    yearly_diagnostics,
    yearly_disclosure_diagnostics,
)
from supply_chain_alpha.graph.edges import build_point_in_time_edges
from supply_chain_alpha.pipeline import ExitCode
from supply_chain_alpha.utils.config import load_config
from supply_chain_alpha.utils.hashing import (
    atomic_write_json,
    config_sha256,
    source_tree_manifest,
    source_tree_manifest_sha256,
)
from supply_chain_alpha.utils.status import (
    PhaseState,
    build_phase_status,
    phase_status_path,
    read_phase_status,
    write_phase_status,
)

LOGGER = logging.getLogger("run_phase3_coverage")

PHASE3_TEST_FILES = (
    "tests/unit/test_edges.py",
    "tests/unit/test_phase3_gate.py",
    "tests/unit/test_run_phase3.py",
    "tests/integration/test_phase1_3_pipeline.py",
    "tests/regression/test_future_invariance_phase3.py",
)
MANDATORY_REPORT_FILES = (
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
PATH_B_FEASIBILITY_REPORT = "FEASIBILITY_REPORT.md"
FINAL_RESULTS_SUMMARY = "results_summary.json"
PATH_A_FINAL_ARTIFACTS = (
    "reports/final/FINAL_REPORT.md",
    "reports/final/EXECUTIVE_SUMMARY.md",
    "reports/final/limitations.md",
    "reports/final/reproducibility.md",
)
PATH_B_FIXED_ARCHIVE_PATHS = (
    "reports/run_manifest.json",
    "reports/holdout/holdout_run_manifest.json",
    "reports/final/FEASIBILITY_REPORT.md",
    "reports/final/results_summary.json",
    "BLOCKER_REPORT.md",
    *PATH_A_FINAL_ARTIFACTS,
)
PATH_B_CONFIG_ARCHIVE_KEYS = (
    "experiment_registry",
    "research_freeze",
    "equity_daily",
    "returns_daily",
    "signal_daily",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def _read_schema_table(path: Path, schema: Any) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        string_columns = {
            column: str
            for column in (*schema.identifier_columns, *schema.string_columns)
        }
        return validate_table(pd.read_csv(path, dtype=string_columns), schema)
    return validate_table(read_table(path), schema)


def _normalised_status_paths(status: dict[str, Any], field: str) -> set[str]:
    values = status.get(field, [])
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError(f"Phase status {field} must be a list of paths")
    return {Path(value).as_posix() for value in values}


def _validate_recorded_artifact_hashes(
    root: Path,
    hashes: dict[str, Any],
    *,
    label: str,
) -> set[str]:
    if not hashes:
        raise ValueError(f"{label} artifact hashes are missing")
    root_resolved = root.resolve()
    verified: set[str] = set()
    for relative, expected in hashes.items():
        if not isinstance(relative, str):
            raise TypeError(f"{label} artifact-hash paths must be strings")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{label} artifact path is unsafe: {relative}")
        try:
            path = (root / relative_path).resolve()
            path.relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{label} artifact path is unsafe: {relative}") from exc
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{label} artifact hash is invalid: {relative}")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} artifact hash mismatch: {relative}")
        verified.add(relative_path.as_posix())
    return verified


def _validate_phase2_prerequisite(root: Path, config: dict[str, Any]) -> None:
    current_config_hash = config_sha256(config)
    current_source_hash = source_tree_manifest_sha256(source_tree_manifest(root))
    statuses: dict[int, dict[str, Any]] = {}
    for phase in (1, 2):
        path = phase_status_path(root, phase)
        if not path.is_file():
            raise ValueError(f"Phase {phase} status is missing")
        status = read_phase_status(path)
        if status.get("state") != "PASS":
            raise ValueError(f"Phase {phase} prerequisite is not PASS")
        if status.get("config_sha256") != current_config_hash:
            raise ValueError(f"Phase {phase} prerequisite config hash is stale")
        if status.get("source_tree_sha256") != current_source_hash:
            raise ValueError(f"Phase {phase} prerequisite source hash is stale")
        statuses[phase] = status

    phase2 = statuses[2]
    criteria = phase2.get("criteria")
    if not isinstance(criteria, dict):
        raise TypeError("Phase 2 prerequisite criteria are missing")
    failed = [key for key in PHASE2_REQUIRED_CRITERIA if criteria.get(key) is not True]
    if failed:
        raise ValueError(f"Phase 2 production QA criteria are not all true: {failed}")
    tests_run = phase2.get("tests_run")
    if not isinstance(tests_run, list) or not tests_run:
        raise ValueError("Phase 2 prerequisite does not record its tests")

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
    if not required_outputs.issubset(_normalised_status_paths(phase2, "outputs")):
        raise ValueError("Phase 2 prerequisite omits canonical outputs or raw manifest")
    raw_manifest = Path(config["paths"]["raw_manifest"]).as_posix()
    if raw_manifest not in _normalised_status_paths(phase2, "inputs"):
        raise ValueError("Phase 2 prerequisite inputs omit the verified raw manifest")
    metrics = phase2.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("Phase 2 prerequisite metrics are missing")
    artifact_hashes = metrics.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise TypeError("Phase 2 prerequisite artifact hashes are missing")
    verified = _validate_recorded_artifact_hashes(
        root,
        artifact_hashes,
        label="Phase 2 prerequisite",
    )
    missing_hashes = sorted(required_outputs - verified)
    if missing_hashes:
        raise ValueError(
            "Phase 2 prerequisite artifact hashes omit canonical outputs or raw "
            f"manifest: {missing_hashes}"
        )


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_synthetic_isolation(
    *,
    root: Path,
    config: dict[str, Any],
    mentions: Path,
    aliases: Path,
    master: Path,
    output_dir: Path,
) -> None:
    production_inputs = {
        (root / config["paths"][key]).resolve()
        for key in ("disclosure_raw", "company_alias", "security_master")
    }
    supplied_inputs = {path.resolve() for path in (mentions, aliases, master)}
    if supplied_inputs & production_inputs:
        raise ValueError("Synthetic mode may not read production canonical inputs")

    production_data_dirs = tuple(
        (root / config["paths"][key]).resolve()
        for key in ("raw_dir", "interim_dir", "processed_dir")
    )
    if any(
        _is_within(supplied, data_dir)
        for supplied in supplied_inputs
        for data_dir in production_data_dirs
    ):
        raise ValueError(
            "Synthetic inputs may not come from production data directories"
        )

    trials_dir = root / "reports" / "trials"
    if not _is_within(output_dir, trials_dir):
        raise ValueError("Synthetic output must remain under reports/trials")


def _validate_audit_alignment(mentions: pd.DataFrame, audit: pd.DataFrame) -> None:
    key = ["source_document_id", "source_company_id", "counterparty_raw_name"]
    mention_keys = (
        mentions[key].astype("string").drop_duplicates().reset_index(drop=True)
    )
    audit_keys = audit[key].astype("string").drop_duplicates().reset_index(drop=True)
    left = mention_keys.merge(audit_keys, on=key, how="left", indicator=True)
    right = audit_keys.merge(mention_keys, on=key, how="left", indicator=True)
    missing = int(left["_merge"].ne("both").sum())
    extra = int(right["_merge"].ne("both").sum())
    if missing or extra:
        raise ValueError(
            "Phase-2 canonical resolution audit does not align with disclosure_raw "
            f"(missing={missing}, extra={extra})"
        )


def _holdout_row_count(
    mentions: pd.DataFrame, *, holdout_start: str, timezone_name: str
) -> int:
    publication = pd.to_datetime(
        mentions["publication_datetime"], utc=True, errors="raise"
    )
    boundary = pd.Timestamp(holdout_start, tz=timezone_name).tz_convert("UTC")
    return int(publication.ge(boundary).sum())


def _publication_timezone_violation_count(mentions: pd.DataFrame) -> int:
    violations = 0
    for value in mentions["publication_datetime"]:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            violations += 1
            continue
        if (
            pd.isna(timestamp)
            or timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            violations += 1
    return violations


def _run_required_tests(root: Path) -> tuple[bool, str, list[str]]:
    command = [sys.executable, "-m", "pytest", "-q", *PHASE3_TEST_FILES]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    details = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return (
        completed.returncode == 0,
        details,
        [subprocess.list2cmdline(command)],
    )


def _checks_pass_strictly(
    checks: dict[str, Any],
    *,
    required_keys: tuple[str, ...],
) -> bool:
    missing = sorted(set(required_keys) - set(checks))
    if missing:
        raise ValueError(f"Phase-3 gate omitted mandatory checks: {missing}")
    passed = True
    for key in required_keys:
        check = checks[key]
        if not isinstance(check, dict):
            raise TypeError(f"Phase-3 gate check must be a mapping: {key}")
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
        raise ValueError(f"Phase-3 gate check has an invalid result: {key}")
    return passed


def _classify_gate(gate: dict[str, Any]) -> tuple[PhaseState, ExitCode, bool]:
    integrity_passed = gate.get("integrity_passed")
    coverage_passed = gate.get("coverage_passed")
    outcome = gate.get("outcome")
    passed = gate.get("passed")
    if not isinstance(integrity_passed, bool) or not isinstance(coverage_passed, bool):
        raise TypeError("Phase-3 gate omitted integrity_passed/coverage_passed")
    if not isinstance(passed, bool):
        raise TypeError("Phase-3 gate omitted boolean passed")
    checks = gate.get("checks")
    integrity_checks = gate.get("integrity_checks")
    if not isinstance(checks, dict) or not isinstance(integrity_checks, dict):
        raise TypeError("Phase-3 gate omitted checks/integrity_checks")
    coverage_checks = gate.get("coverage_checks")
    if not isinstance(coverage_checks, dict):
        raise TypeError("Phase-3 gate omitted coverage_checks")
    observed_integrity = _checks_pass_strictly(
        integrity_checks,
        required_keys=INTEGRITY_CHECK_KEYS,
    )
    observed_coverage = _checks_pass_strictly(
        coverage_checks,
        required_keys=COVERAGE_CHECK_KEYS,
    )
    if integrity_passed is not observed_integrity:
        raise ValueError("Phase-3 integrity summary disagrees with gate checks")
    if coverage_passed is not observed_coverage:
        raise ValueError("Phase-3 coverage summary disagrees with gate checks")

    if passed:
        if not integrity_passed or not coverage_passed or outcome != "PROCEED":
            raise ValueError("Inconsistent Phase-3 PASS classification")
        return PhaseState.PASS, ExitCode.SUCCESS, False
    if not integrity_passed:
        if outcome != "ENGINEERING_FAILURE":
            raise ValueError(
                "Integrity failure must be classified as ENGINEERING_FAILURE"
            )
        return PhaseState.FAIL, ExitCode.ENGINEERING_FAILURE, False
    if not coverage_passed:
        if outcome != "INFEASIBLE_DATA":
            raise ValueError("Coverage failure must be classified as INFEASIBLE_DATA")
        return PhaseState.FAIL, ExitCode.RESEARCH_GATE_FAILED, True
    raise ValueError("Inconsistent failed Phase-3 gate classification")


def _write_feasibility_report(
    root: Path,
    *,
    coverage: dict[str, Any],
    gate: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> tuple[Path, Path]:
    yearly = gate["diagnostics"]["yearly_coverage"]  # type: ignore[index]
    rows = "\n".join(
        f"| {row['year']} | {row['year_end_active_edges']} | {row['year_end_connected_stocks']} |"
        for row in yearly
    )
    text = f"""# Supply-chain graph feasibility result

Research conclusion: `INFEASIBLE_DATA`

The point-in-time integrity checks and frozen real-data coverage gate were run without using returns.
The disclosed listed-to-listed graph did not meet every pre-specified V2 coverage requirement, so no
return, signal, IC, portfolio, or holdout research was performed.

All mandatory graph-integrity checks passed. This conclusion is caused only by frozen coverage
thresholds and is not an engineering-failure classification.

## Coverage summary

```json
{json.dumps(coverage, ensure_ascii=False, indent=2)}
```

## Frozen annual coverage

| Year | Active directed edges at year-end | Connected listed stocks at year-end |
|---:|---:|---:|
{rows}

## Gate checks

```json
{json.dumps(gate["checks"], ensure_ascii=False, indent=2)}
```

This is a data-feasibility conclusion, not evidence for or against the economic hypothesis.
"""
    final_dir = root / "reports" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    report_path = final_dir / "FEASIBILITY_REPORT.md"
    results_path = final_dir / "results_summary.json"
    report_path.write_text(text, encoding="utf-8")
    final_hashes = dict(artifact_hashes)
    final_hashes[_relative(root, report_path)] = sha256_file(report_path)
    atomic_write_json(
        results_path,
        {
            "research_conclusion": "INFEASIBLE_DATA",
            "strategy_evidence": "NOT_EVALUATED",
            "terminal_path": "PATH_B_DATA_INFEASIBILITY",
            "last_completed_empirical_gate": 3,
            "coverage": coverage,
            "phase3_gate": gate,
            "artifact_hashes": dict(sorted(final_hashes.items())),
        },
    )
    return report_path, results_path


def _write_phase3_status(
    *,
    root: Path,
    config_path: Path,
    started_at: datetime,
    state: PhaseState,
    inputs: list[str],
    outputs: list[str],
    tests_run: list[str],
    metrics: dict[str, Any],
    criteria: dict[str, Any],
    notes: list[str],
) -> None:
    status_path = phase_status_path(root, 3)
    status = build_phase_status(
        phase=3,
        name="point_in_time_graph_coverage",
        state=state,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        project_root=root,
        config=config_path,
        inputs=inputs,
        outputs=sorted({*outputs, _relative(root, status_path)}),
        tests_run=tests_run,
        metrics=metrics,
        criteria=criteria,
        blockers=[],
        notes=notes,
    )
    write_phase_status(
        status_path,
        status,
        validation_rerun=status_path.exists(),
        defect_corrected=status_path.exists() and state is PhaseState.PASS,
    )


def _write_diagnostics(
    *,
    output_dir: Path,
    mentions: pd.DataFrame,
    audit: pd.DataFrame,
    edges: pd.DataFrame,
    master: pd.DataFrame,
    coverage: dict[str, Any],
    gate: dict[str, Any],
    start_year: int,
    end_year: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    listed_edges = (
        listed_edge_intervals(edges, master) if not edges.empty else edges.copy()
    )
    audit.to_csv(output_dir / "resolution_audit.csv", index=False)
    edges.to_csv(output_dir / "supply_chain_edge.csv", index=False)
    yearly_disclosure_diagnostics(mentions, audit).to_csv(
        output_dir / "yearly_disclosures.csv", index=False
    )
    yearly_diagnostics(edges, master, start_year=start_year, end_year=end_year).to_csv(
        output_dir / "yearly_coverage.csv", index=False
    )
    degree_diagnostics(listed_edges).to_csv(
        output_dir / "degree_diagnostics.csv", index=False
    )
    node_role_diagnostics(listed_edges).to_csv(
        output_dir / "node_roles.csv", index=False
    )
    source_document_diagnostics(listed_edges, mentions).to_csv(
        output_dir / "source_document_diagnostics.csv", index=False
    )

    industry_columns = [
        "year",
        "active_edges",
        "industry_count",
        "largest_industry_share",
        "classified_edge_endpoints",
        "total_edge_endpoints",
        "industry_classification_rate",
        "ambiguous_pit_industry_nodes",
    ]
    industry_records = gate["diagnostics"].get("pit_industry_by_usable_year", [])
    pd.DataFrame(industry_records, columns=industry_columns).to_csv(
        output_dir / "industry_diagnostics.csv", index=False
    )

    as_of = (
        pd.to_datetime(listed_edges["effective_start"], errors="raise").max()
        if not listed_edges.empty
        else pd.Timestamp(year=end_year, month=12, day=31)
    )
    edge_age_distribution(listed_edges, as_of).to_csv(
        output_dir / "edge_age_distribution.csv", index=False
    )
    atomic_write_json(output_dir / "summary.json", {"coverage": coverage, "gate": gate})

    artifacts = [output_dir / name for name in MANDATORY_REPORT_FILES]
    missing = [path.name for path in artifacts if not path.is_file()]
    if missing:
        raise RuntimeError(f"Phase-3 mandatory reports were not written: {missing}")
    return artifacts


def _compact_gate_for_artifacts(gate: dict[str, Any]) -> dict[str, Any]:
    """Replace duplicated per-document JSON rows with their canonical CSV index."""

    diagnostics = gate.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return gate
    source_documents = diagnostics.get("source_documents")
    if not isinstance(source_documents, list):
        return gate
    compact_diagnostics = dict(diagnostics)
    compact_diagnostics["source_documents"] = {
        "artifact": "source_document_diagnostics.csv",
        "row_count": len(source_documents),
    }
    return {**gate, "diagnostics": compact_diagnostics}


def _artifact_hashes(root: Path, paths: list[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        relative = _relative(root, path)
        hashes[relative] = sha256_file(path)
    return dict(sorted(hashes.items()))


def _archive_superseded_final_artifacts(
    root: Path,
    *,
    timestamp: datetime,
) -> list[Path]:
    """Move an earlier Path-B conclusion into recoverable history."""

    final_dir = root / "reports" / "final"
    feasibility = final_dir / PATH_B_FEASIBILITY_REPORT
    results = final_dir / FINAL_RESULTS_SUMMARY
    sources: list[Path] = []
    if feasibility.exists() and not feasibility.is_file():
        raise ValueError(f"Superseded final artifact is not a file: {feasibility}")
    if feasibility.is_file():
        sources.append(feasibility)

    if results.exists() and not results.is_file():
        raise ValueError(f"Superseded final artifact is not a file: {results}")
    if results.is_file():
        try:
            payload = json.loads(results.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot classify existing final results summary: {results}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Existing final results summary is not an object: {results}"
            )
        if (
            payload.get("terminal_path") == "PATH_B_DATA_INFEASIBILITY"
            or payload.get("research_conclusion") == "INFEASIBLE_DATA"
        ):
            sources.append(results)

    present: list[Path] = []
    for source in sources:
        if source.is_file():
            present.append(source)
    if not present:
        return []

    history_dir = root / "reports" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived: list[Path] = []
    for source in present:
        target = history_dir / f"PHASE_3_SUPERSEDED_{stamp}_{source.name}"
        suffix = 1
        while target.exists():
            target = history_dir / (
                f"PHASE_3_SUPERSEDED_{stamp}_{suffix}_{source.name}"
            )
            suffix += 1
        source.replace(target)
        archived.append(target)
    return archived


def _path_b_archive_candidates(
    root: Path,
    config: dict[str, Any],
) -> list[str]:
    """Return the exact, non-recursive allowlist superseded by a Path-B result."""

    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise TypeError("Project config paths must be a mapping")
    candidates = [
        *(
            phase_status_path(root, phase).relative_to(root).as_posix()
            for phase in range(3, 12)
        ),
        *PATH_B_FIXED_ARCHIVE_PATHS,
    ]
    for key in PATH_B_CONFIG_ARCHIVE_KEYS:
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Project config has no safe Path-B archive path: {key}")
        candidates.append(value)

    unique: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        normalized = Path(value).as_posix()
        marker = normalized.casefold()
        if marker not in seen:
            seen.add(marker)
            unique.append(normalized)
    return unique


def _archive_path_b_incompatible_artifacts(
    root: Path,
    config: dict[str, Any],
    *,
    timestamp: datetime,
) -> list[Path]:
    """Recoverably archive only files that are incompatible with Path B.

    Every source is preflighted before the first move.  The allowlist is exact;
    no directory tree is discovered or moved recursively.
    """

    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError(f"Project root cannot be a symlink: {root_path}")
    root_resolved = root_path.resolve(strict=True)
    if not root_resolved.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {root_resolved}")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Path-B archive timestamp must include a UTC offset")

    history_relative = Path("reports/history")
    history_root = root_resolved / history_relative
    present: list[tuple[Path, Path]] = []
    seen_sources: set[str] = set()
    for value in _path_b_archive_candidates(root_resolved, config):
        relative = Path(value)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError(f"Path-B archive path is unsafe: {value}")

        source = root_resolved / relative
        cursor = root_resolved
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f"Path-B archive path cannot be a symlink: {value}")
        try:
            resolved = source.resolve(strict=False)
            resolved.relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Path-B archive path is unsafe: {value}") from exc
        if resolved == history_root or history_root in resolved.parents:
            raise ValueError(f"Path-B archive source cannot be in history: {value}")
        if not source.exists():
            continue
        if not source.is_file():
            raise ValueError(f"Path-B archive source is not a file: {value}")
        marker = str(resolved).casefold()
        if marker not in seen_sources:
            seen_sources.add(marker)
            present.append((source, relative))

    if not present:
        return []

    cursor = root_resolved
    for part in history_relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(
                f"Path-B archive destination cannot be a symlink: {cursor}"
            )
    if history_root.exists() and not history_root.is_dir():
        raise ValueError(
            f"Path-B archive destination is not a directory: {history_root}"
        )
    history_root.mkdir(parents=True, exist_ok=True)

    stamp = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base_name = f"PHASE_3_PATH_B_SUPERSEDED_{stamp}"
    archive_dir = history_root / base_name
    suffix = 1
    while archive_dir.exists() or archive_dir.is_symlink():
        if archive_dir.is_symlink() or not archive_dir.is_dir():
            raise ValueError(f"Path-B archive destination is unsafe: {archive_dir}")
        archive_dir = history_root / f"{base_name}_{suffix}"
        suffix += 1
    archive_dir.mkdir()

    archived: list[Path] = []
    for source, relative in present:
        target = archive_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            source.replace(target)
        except OSError as exc:
            raise RuntimeError(
                f"Could not archive Path-B-incompatible artifact: {relative.as_posix()}"
            ) from exc
        archived.append(target)
    return archived


def _write_path_b_conclusion(
    root: Path,
    *,
    config: dict[str, Any],
    coverage: dict[str, Any],
    gate: dict[str, Any],
    support_paths: list[Path],
    timestamp: datetime,
) -> tuple[Path, Path, list[Path]]:
    """Archive incompatible state before emitting a new Path-B conclusion."""

    archived = _archive_path_b_incompatible_artifacts(
        root,
        config,
        timestamp=timestamp,
    )
    hashes = _artifact_hashes(root, support_paths)
    report, results = _write_feasibility_report(
        root,
        coverage=coverage,
        gate=gate,
        artifact_hashes=hashes,
    )
    return report, results, archived


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen real-data Phase-3 graph gate."
    )
    parser.add_argument("mentions", nargs="?", type=Path, help="Synthetic mode only")
    parser.add_argument("aliases", nargs="?", type=Path, help="Synthetic mode only")
    parser.add_argument(
        "security_master", nargs="?", type=Path, help="Synthetic mode only"
    )
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    parser.add_argument("--output-dir", type=Path, help="Synthetic mode only")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate diagnostics only; never writes a production phase status.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    started = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    paths = config["paths"]
    raw_manifest_path = root / paths["raw_manifest"]
    production_inputs = [
        _relative(root, root / paths["disclosure_raw"]),
        _relative(root, root / paths["resolution_audit"]),
        _relative(root, root / paths["security_master"]),
        _relative(root, raw_manifest_path),
    ]

    if args.synthetic:
        if (
            args.mentions is None
            or args.aliases is None
            or args.security_master is None
        ):
            raise ValueError(
                "Synthetic mode requires explicit mentions, aliases, and security-master inputs"
            )
        mention_path = args.mentions
        alias_path = args.aliases
        master_path = args.security_master
        output_dir = args.output_dir or root / "reports" / "trials" / "phase3_synthetic"
        _validate_synthetic_isolation(
            root=root,
            config=config,
            mentions=mention_path,
            aliases=alias_path,
            master=master_path,
            output_dir=output_dir,
        )
        mentions = load_structured_mentions(mention_path)
        aliases = _read_schema_table(alias_path, COMPANY_ALIAS)
        master = _read_schema_table(master_path, SECURITY_MASTER)
        resolution_config = ResolutionConfig(
            fuzzy_enabled=bool(config["entity_resolution"]["fuzzy_matching_enabled"]),
            fuzzy_min_score=float(config["entity_resolution"]["fuzzy_min_score"]),
            fuzzy_min_margin=float(config["entity_resolution"]["fuzzy_min_margin"]),
        )
        audit = resolve_mentions(
            mentions,
            aliases,
            config=resolution_config,
            override_path=root / config["entity_resolution"]["manual_override_file"],
        )
        audit = validate_table(audit, RESOLUTION_AUDIT)
    else:
        if any(
            value is not None
            for value in (
                args.mentions,
                args.aliases,
                args.security_master,
                args.output_dir,
            )
        ):
            raise ValueError(
                "Production Phase 3 uses only the Phase-2 canonical inputs and output path"
            )
        _validate_phase2_prerequisite(root, config)
        mention_path = root / paths["disclosure_raw"]
        audit_path = root / paths["resolution_audit"]
        master_path = root / paths["security_master"]
        output_dir = root / "reports" / "phase3_coverage"
        verify_raw_manifest(
            raw_manifest_path,
            inventory_root=root / paths["raw_dir"] / "official_sources",
        )
        mentions = load_structured_mentions(mention_path)
        timezone_violations = _publication_timezone_violation_count(mentions)
        if timezone_violations:
            _write_phase3_status(
                root=root,
                config_path=config_path,
                started_at=started,
                state=PhaseState.FAIL,
                inputs=production_inputs,
                outputs=[],
                tests_run=[],
                metrics={"publication_timezone_violations": timezone_violations},
                criteria={
                    "real_data_gate_evaluated": False,
                    "phase3_tests_pass": False,
                    "graph_integrity_passed": False,
                    "frozen_coverage_passed": False,
                    "research_gate_failed": False,
                    "holdout_integrity_violation": False,
                },
                notes=[
                    (
                        "Phase 3 requires every production publication_datetime to "
                        "carry an explicit UTC offset."
                    )
                ],
            )
            return int(ExitCode.ENGINEERING_FAILURE)
        holdout_rows = _holdout_row_count(
            mentions,
            holdout_start=str(config["periods"]["holdout_start"]),
            timezone_name=str(config["information_timing"]["timezone"]),
        )
        if holdout_rows:
            _write_phase3_status(
                root=root,
                config_path=config_path,
                started_at=started,
                state=PhaseState.FAIL,
                inputs=production_inputs,
                outputs=[],
                tests_run=[],
                metrics={"holdout_rows_detected": holdout_rows},
                criteria={
                    "real_data_gate_evaluated": False,
                    "phase3_tests_pass": False,
                    "graph_integrity_passed": False,
                    "frozen_coverage_passed": False,
                    "research_gate_failed": False,
                    "holdout_integrity_violation": True,
                },
                notes=[
                    "Phase 3 refused canonical disclosures at or after holdout_start."
                ],
            )
            return int(ExitCode.HOLDOUT_INTEGRITY_VIOLATION)
        audit = _read_schema_table(audit_path, RESOLUTION_AUDIT)
        master = _read_schema_table(master_path, SECURITY_MASTER)

    _validate_audit_alignment(mentions, audit)

    tests_passed = True
    test_details = ""
    tests_run: list[str] = []
    if not args.synthetic:
        tests_passed, test_details, tests_run = _run_required_tests(root)
        if not tests_passed:
            _write_phase3_status(
                root=root,
                config_path=config_path,
                started_at=started,
                state=PhaseState.FAIL,
                inputs=production_inputs,
                outputs=[],
                tests_run=tests_run,
                metrics={"phase3_test_return_code": 1},
                criteria={
                    "real_data_gate_evaluated": False,
                    "phase3_tests_pass": False,
                    "graph_integrity_passed": False,
                    "frozen_coverage_passed": False,
                    "research_gate_failed": False,
                    "holdout_integrity_violation": False,
                },
                notes=[test_details or "Mandatory Phase-3 tests failed."],
            )
            return int(ExitCode.ENGINEERING_FAILURE)

    edges = build_point_in_time_edges(
        mentions,
        audit,
        max_age_days=int(config["information_timing"]["edge_max_age_days"]),
    )
    audit = validate_table(audit, RESOLUTION_AUDIT)
    edges = validate_table(edges, SUPPLY_CHAIN_EDGE)
    summary = summarize_coverage(mentions, audit, edges, master)
    gate = _json_safe(
        evaluate_phase3_gate(
            edges,
            master,
            mentions=mentions,
            audit=audit,
            real_data=not args.synthetic,
            development_start_year=pd.Timestamp(
                config["periods"]["development_start"]
            ).year,
            validation_end_year=pd.Timestamp(config["periods"]["validation_end"]).year,
            frozen_thresholds=config["graph_coverage_gate"],
            max_edge_age_days=int(config["information_timing"]["edge_max_age_days"]),
        )
    )
    gate = _compact_gate_for_artifacts(gate)
    coverage = _json_safe(summary_to_dict(summary))
    report_artifacts = _write_diagnostics(
        output_dir=output_dir,
        mentions=mentions,
        audit=audit,
        edges=edges,
        master=master,
        coverage=coverage,
        gate=gate,
        start_year=pd.Timestamp(config["periods"]["development_start"]).year,
        end_year=pd.Timestamp(config["periods"]["validation_end"]).year,
    )
    LOGGER.info(
        "source=%s mentions=%d audit_rows=%d",
        "synthetic_fixture" if args.synthetic else "phase2_canonical_real_disclosures",
        len(mentions),
        len(audit),
    )
    LOGGER.info("audit_rows=%d edge_evidence_rows=%d", len(audit), len(edges))
    LOGGER.info("coverage=%s gate=%s", coverage, gate["status"])

    if args.synthetic:
        print(
            json.dumps(
                {"coverage": coverage, "gate": gate}, ensure_ascii=False, indent=2
            )
        )
        return int(ExitCode.SUCCESS)

    canonical_edge_path = root / paths["supply_chain_edge"]
    write_canonical_parquet(edges, canonical_edge_path)
    state, exit_code, research_gate_failed = _classify_gate(gate)
    outputs = [
        *[_relative(root, path) for path in report_artifacts],
        _relative(root, canonical_edge_path),
    ]
    archived_final_artifacts: list[Path] = []
    archived_path_b_artifacts: list[Path] = []
    if state is PhaseState.PASS:
        archived_final_artifacts = _archive_superseded_final_artifacts(
            root,
            timestamp=datetime.now(timezone.utc),
        )
        outputs.extend(_relative(root, path) for path in archived_final_artifacts)
    if research_gate_failed:
        support_paths = [
            raw_manifest_path,
            mention_path,
            root / paths["resolution_audit"],
            master_path,
            canonical_edge_path,
            *report_artifacts,
        ]
        (
            feasibility_path,
            results_path,
            archived_path_b_artifacts,
        ) = _write_path_b_conclusion(
            root,
            config=config,
            coverage=coverage,
            gate=gate,
            support_paths=support_paths,
            timestamp=datetime.now(timezone.utc),
        )
        outputs.extend(
            [
                _relative(root, feasibility_path),
                _relative(root, results_path),
                *(_relative(root, path) for path in archived_path_b_artifacts),
            ]
        )

    status_hash_paths = [
        raw_manifest_path,
        mention_path,
        root / paths["resolution_audit"],
        master_path,
        canonical_edge_path,
        *report_artifacts,
        *archived_final_artifacts,
        *archived_path_b_artifacts,
    ]
    if research_gate_failed:
        status_hash_paths.extend([feasibility_path, results_path])
    status_artifact_hashes = _artifact_hashes(root, status_hash_paths)

    notes = [
        "Frozen V2 real-data graph gate evaluated before any return data or holdout access.",
        f"gate_outcome={gate['outcome']}",
    ]
    if test_details:
        notes.append(test_details)
    if archived_final_artifacts:
        notes.append(
            "Archived superseded final artifacts: "
            + ", ".join(_relative(root, path) for path in archived_final_artifacts)
        )
    if archived_path_b_artifacts:
        notes.append(
            "Archived Path-B-incompatible prior artifacts: "
            + ", ".join(_relative(root, path) for path in archived_path_b_artifacts)
        )
    _write_phase3_status(
        root=root,
        config_path=config_path,
        started_at=started,
        state=state,
        inputs=production_inputs,
        outputs=outputs,
        tests_run=tests_run,
        metrics={
            "coverage": coverage,
            "phase3_test_return_code": 0,
            "artifact_hashes": status_artifact_hashes,
        },
        criteria={
            "real_data_gate_evaluated": True,
            "phase3_tests_pass": tests_passed,
            "graph_integrity_passed": bool(gate["integrity_passed"]),
            "frozen_coverage_passed": bool(gate["coverage_passed"]),
            "research_gate_failed": research_gate_failed,
            "holdout_integrity_violation": False,
            "gate_checks": gate["checks"],
        },
        notes=notes,
    )
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
