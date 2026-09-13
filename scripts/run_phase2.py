"""Run Phase 2 without permitting trial data to masquerade as production.

The production path acquires official security/disclosure data, extracts
source-grounded relationship mentions, resolves entities point in time, runs
the frozen Phase-2 QA, and verifies the immutable raw-data manifest.  A limited
trial uses separate interim/report artifacts and never writes the Phase-2
status file.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from supply_chain_alpha.data.acquisition import (
    AcquisitionMode,
    RateLimitedSession,
    RawAssetStore,
    RequestPolicy,
    assert_security_master_qa,
)
from supply_chain_alpha.data.cninfo import (
    acquire_cninfo_disclosures,
    acquire_cninfo_stock_map,
    disclosure_acquisition_diagnostics,
    enrich_company_aliases,
    reconcile_security_ids,
    stock_map_diagnostics,
)
from supply_chain_alpha.data.disclosures import sha256_file, verify_raw_manifest
from supply_chain_alpha.data.exchange_acquisition import (
    acquire_official_security_master,
    company_alias_diagnostics,
)
from supply_chain_alpha.data.io import write_canonical_parquet
from supply_chain_alpha.data.quality import evaluate_phase2_qa
from supply_chain_alpha.entities.resolve import ResolutionConfig, resolve_mentions
from supply_chain_alpha.pipeline import PHASE_NAMES, ExitCode
from supply_chain_alpha.utils.config import load_config
from supply_chain_alpha.utils.hashing import atomic_write_json
from supply_chain_alpha.utils.status import (
    PhaseState,
    build_phase_status,
    phase_status_path,
    read_phase_status,
    write_phase_status,
)

LOGGER = logging.getLogger("run_phase2")
PHASE = 2
OFFICIAL_CACHE_SUBDIR = "official_sources"
TRIAL_REPORT = Path("reports/trials/phase_2_trial.json")
BLOCKER_REPORT = Path("BLOCKER_REPORT.md")
RESOLVED_BLOCKER_HISTORY = Path("reports/history")


class OfficialSourceUnavailable(RuntimeError):
    """A required official source remained unavailable after configured retries."""

    def __init__(
        self,
        dependency: str,
        attempted_sources: list[str],
        reason: str,
        *,
        outcome: Phase2Outcome | None = None,
    ) -> None:
        super().__init__(reason)
        self.dependency = dependency
        self.attempted_sources = attempted_sources
        self.reason = reason
        self.outcome = outcome


@dataclass(frozen=True)
class Phase2Outcome:
    """JSON-safe evidence returned by the Phase-2 data flow."""

    passed: bool
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    criteria: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _prefix_cache_paths(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Make cache-local provenance paths relative to configured ``raw_dir``."""

    result = frame.copy()
    prefix = f"{OFFICIAL_CACHE_SUBDIR}/"
    for column in columns:
        if column not in result:
            continue
        present = result[column].notna() & result[column].astype(
            "string"
        ).str.strip().ne("")
        values = (
            result.loc[present, column].astype(str).str.replace("\\", "/", regex=False)
        )
        result.loc[present, column] = values.map(
            lambda value: value if value.startswith(prefix) else f"{prefix}{value}"
        )
    return result


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalars and timestamps to strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _load_pdf_extractor() -> Any:
    """Delay the optional integration import until extraction is required."""

    try:
        from supply_chain_alpha.data.pdf_extraction import extract_cninfo_documents
    except ImportError as exc:  # pragma: no cover - exercised before module integration
        raise RuntimeError(
            "Phase-2 PDF extraction module is unavailable; "
            "expected supply_chain_alpha.data.pdf_extraction.extract_cninfo_documents"
        ) from exc
    return extract_cninfo_documents


def _load_manifest_writer() -> Any:
    """Delay import so the orchestrator remains compatible during integration."""

    try:
        from supply_chain_alpha.data.acquisition import write_raw_asset_manifest
    except ImportError as exc:  # pragma: no cover - exercised before module integration
        raise RuntimeError(
            "Raw-asset manifest aggregator is unavailable; "
            "expected supply_chain_alpha.data.acquisition.write_raw_asset_manifest"
        ) from exc
    return write_raw_asset_manifest


def _refresh_raw_manifest(
    *,
    root: Path,
    config: dict[str, Any],
    official_cache: Path,
) -> Path:
    """Write and exact-inventory verify raw provenance before derived work."""

    official_cache.mkdir(parents=True, exist_ok=True)
    manifest_writer = _load_manifest_writer()
    manifest_path = _path(root, config["paths"]["raw_manifest"])
    # The verifier below is the trust boundary for the freshly written
    # manifest.  Let it perform the single full asset-byte pass instead of
    # hashing every (potentially large) cached document once in the writer and
    # then immediately hashing it again in the verifier.
    written_manifest = Path(
        manifest_writer(
            official_cache,
            manifest_path,
            verify_asset_bytes=False,
        )
    )
    if written_manifest.resolve() != manifest_path.resolve():
        raise ValueError("Raw-manifest writer returned an unexpected output path")
    verify_raw_manifest(manifest_path, inventory_root=official_cache)
    return manifest_path


def _record_progress(
    progress: dict[str, Phase2Outcome] | None,
    outcome: Phase2Outcome,
) -> None:
    if progress is not None:
        progress["outcome"] = outcome


def _run_required_tests(root: Path) -> tuple[bool, str, list[str]]:
    candidates = [
        "tests/unit/test_acquisition.py",
        "tests/unit/test_entity_resolution.py",
        "tests/unit/test_phase2_quality.py",
        "tests/unit/test_pdf_extraction.py",
        "tests/unit/test_raw_manifest.py",
    ]
    missing = [item for item in candidates if not (root / item).is_file()]
    if missing:
        return False, f"Missing required Phase-2 tests: {missing}", []
    test_files = candidates
    command = [sys.executable, "-m", "pytest", "-q", *test_files]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    details = completed.stdout.strip()
    if completed.stderr.strip():
        details = f"{details}\n{completed.stderr.strip()}".strip()
    return completed.returncode == 0, details, [" ".join(command[1:])]


def _acquire_security_master(
    *,
    transport: RateLimitedSession,
    raw_store: RawAssetStore,
    source_config: dict[str, Any],
) -> Any:
    try:
        return acquire_official_security_master(
            transport,
            raw_store,
            sse_url=str(source_config["sse_url"]),
            szse_url=str(source_config["szse_url"]),
        )
    except requests.RequestException as exc:
        raise OfficialSourceUnavailable(
            "historical SSE/SZSE common-equity security master",
            [str(source_config["sse_url"]), str(source_config["szse_url"])],
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _acquire_disclosures(
    *,
    transport: RateLimitedSession,
    raw_store: RawAssetStore,
    source_config: dict[str, Any],
    start_date: str,
    end_date: str,
    mode: AcquisitionMode,
    page_size: int,
) -> Any:
    try:
        return acquire_cninfo_disclosures(
            transport,
            raw_store,
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            query_url=str(source_config["query_url"]),
            download_base_url=str(source_config["download_base_url"]),
            category=str(source_config["category"]),
            page_size=page_size,
            download_workers=int(source_config.get("download_workers", 1)),
        )
    except requests.RequestException as exc:
        raise OfficialSourceUnavailable(
            "CNINFO annual-report metadata/documents",
            [str(source_config["query_url"]), str(source_config["download_base_url"])],
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _acquire_stock_map(
    *,
    transport: RateLimitedSession,
    raw_store: RawAssetStore,
    source_config: dict[str, Any],
) -> Any:
    try:
        return acquire_cninfo_stock_map(
            transport,
            raw_store,
            url=str(source_config["stock_map_url"]),
        )
    except requests.RequestException as exc:
        raise OfficialSourceUnavailable(
            "CNINFO official current stock identity map",
            [str(source_config["stock_map_url"])],
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _trial_output_paths(root: Path) -> dict[str, Path]:
    base = root / "data" / "interim" / "phase_2_trial"
    return {
        "security_master": base / "security_master.parquet",
        "company_alias": base / "company_alias.parquet",
        "documents": base / "cninfo_documents.parquet",
        "document_audit": base / "document_audit.parquet",
        "disclosure_raw": base / "disclosure_raw.parquet",
        "resolution_audit": base / "resolution_audit.parquet",
    }


def _production_output_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    paths = config["paths"]
    return {
        "security_master": _path(root, paths["security_master"]),
        "company_alias": _path(root, paths["company_alias"]),
        "documents": root
        / "data"
        / "interim"
        / "cninfo_annual_report_documents.parquet",
        "document_audit": root / "data" / "interim" / "cninfo_document_audit.parquet",
        "disclosure_raw": _path(root, paths["disclosure_raw"]),
        "resolution_audit": _path(root, paths["resolution_audit"]),
    }


def _execute_flow(
    *,
    root: Path,
    config: dict[str, Any],
    mode: AcquisitionMode,
    start_date: str,
    end_date: str,
    page_size: int,
    progress: dict[str, Phase2Outcome] | None = None,
) -> Phase2Outcome:
    """Execute acquisition/extraction/resolution in the required order."""

    limited = not mode.production_qa_eligible
    outputs = (
        _trial_output_paths(root) if limited else _production_output_paths(root, config)
    )
    raw_root = _path(root, config["paths"]["raw_dir"])
    official_cache = raw_root / OFFICIAL_CACHE_SUBDIR
    official_cache.mkdir(parents=True, exist_ok=True)
    raw_store = RawAssetStore(official_cache)
    sources = config["data_sources"]
    transport = RateLimitedSession(
        RequestPolicy.from_mapping(sources.get("request_policy"))
    )

    try:
        security_result = _acquire_security_master(
            transport=transport,
            raw_store=raw_store,
            source_config=sources["security_master"],
        )
        stock_map_result = _acquire_stock_map(
            transport=transport,
            raw_store=raw_store,
            source_config=sources["disclosures"],
        )
        disclosure_result = _acquire_disclosures(
            transport=transport,
            raw_store=raw_store,
            source_config=sources["disclosures"],
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            page_size=page_size,
        )
    except Exception as exc:
        try:
            manifest_path = _refresh_raw_manifest(
                root=root,
                config=config,
                official_cache=official_cache,
            )
        except Exception as manifest_exc:
            failed_progress = Phase2Outcome(
                passed=False,
                metrics={
                    "acquisition_exception": f"{type(exc).__name__}: {exc}",
                    "raw_manifest_exception": (
                        f"{type(manifest_exc).__name__}: {manifest_exc}"
                    ),
                },
                criteria={
                    "raw_acquisition_completed": False,
                    "production_raw_manifest_verified": False,
                    "phase2_pass_criteria_met": False,
                },
            )
            _record_progress(progress, failed_progress)
            raise RuntimeError(
                "Raw-manifest refresh failed while preserving acquisition failure "
                f"evidence: {type(manifest_exc).__name__}: {manifest_exc}"
            ) from manifest_exc

        failed_progress = Phase2Outcome(
            passed=False,
            inputs=[_relative(root, manifest_path)],
            outputs=[_relative(root, manifest_path)],
            metrics={
                "mode": mode.label,
                "query_start_date": start_date,
                "query_end_date": end_date,
                "acquisition_exception": f"{type(exc).__name__}: {exc}",
                "raw_manifest_sha256": sha256_file(manifest_path),
            },
            criteria={
                "raw_acquisition_completed": False,
                "production_raw_manifest_verified": True,
                "phase2_pass_criteria_met": False,
            },
        )
        _record_progress(progress, failed_progress)
        if isinstance(exc, OfficialSourceUnavailable):
            exc.outcome = failed_progress
        raise

    try:
        manifest_path = _refresh_raw_manifest(
            root=root,
            config=config,
            official_cache=official_cache,
        )
    except Exception as exc:
        raw_inputs_before_manifest = sorted(
            {
                _relative(root, official_cache / asset.local_path)
                for asset in (
                    *security_result.assets,
                    stock_map_result.asset,
                    *disclosure_result.assets,
                )
            }
        )
        _record_progress(
            progress,
            Phase2Outcome(
                passed=False,
                inputs=raw_inputs_before_manifest,
                metrics={
                    "mode": mode.label,
                    "query_start_date": start_date,
                    "query_end_date": end_date,
                    "security_asset_count": len(security_result.assets),
                    "stock_map_asset_count": 1,
                    "disclosure_asset_count": len(disclosure_result.assets),
                    "download_failure_count": len(disclosure_result.failures),
                    "raw_manifest_exception": f"{type(exc).__name__}: {exc}",
                },
                criteria={
                    "raw_acquisition_completed": True,
                    "production_raw_manifest_verified": False,
                    "phase2_pass_criteria_met": False,
                },
                notes=list(security_result.limitations),
            ),
        )
        raise
    security_master = _prefix_cache_paths(
        security_result.securities,
        "raw_local_path",
    )
    cninfo_stock_map = _prefix_cache_paths(
        stock_map_result.stock_map,
        "raw_local_path",
    )
    master_diagnostics = assert_security_master_qa(security_master)
    cninfo_stock_map_diagnostics = stock_map_diagnostics(
        replace(stock_map_result, stock_map=cninfo_stock_map)
    )
    disclosure_diagnostics = disclosure_acquisition_diagnostics(disclosure_result)
    documents = _prefix_cache_paths(
        disclosure_result.documents,
        "local_raw_path",
        "metadata_raw_local_path",
    )
    documents, reconciliation_diagnostics = reconcile_security_ids(
        documents,
        security_master,
        cninfo_stock_map,
    )
    company_aliases, alias_enrichment_diagnostics = enrich_company_aliases(
        security_result.aliases,
        documents,
        security_master,
        cninfo_stock_map,
    )
    alias_diagnostics = company_alias_diagnostics(company_aliases)
    write_canonical_parquet(security_master, outputs["security_master"])
    write_canonical_parquet(company_aliases, outputs["company_alias"])
    write_canonical_parquet(documents, outputs["documents"])

    common_metrics: dict[str, Any] = {
        "mode": mode.label,
        "query_start_date": start_date,
        "query_end_date": end_date,
        "security_master": master_diagnostics,
        "cninfo_stock_map": cninfo_stock_map_diagnostics,
        "company_alias": alias_diagnostics,
        "cninfo_security_id_reconciliation": reconciliation_diagnostics,
        "cninfo_alias_enrichment": alias_enrichment_diagnostics,
        "disclosure_acquisition": disclosure_diagnostics,
    }
    output_list = [
        *[_relative(root, path) for path in outputs.values() if path.exists()],
        _relative(root, manifest_path),
    ]
    raw_asset_paths = {
        _relative(root, official_cache / asset.local_path)
        for asset in (
            *security_result.assets,
            stock_map_result.asset,
            *disclosure_result.assets,
        )
    }
    # The verified exact-inventory manifest is the canonical aggregate input.
    # Repeating tens of thousands of member paths in every status made the
    # status itself multi-megabyte without adding provenance information.
    raw_inputs = [_relative(root, manifest_path)]
    common_metrics["raw_asset_inventory"] = {
        "manifest_path": raw_inputs[0],
        "unique_asset_count": len(raw_asset_paths),
    }
    notes = list(security_result.limitations)
    acquisition_progress = Phase2Outcome(
        passed=False,
        inputs=raw_inputs,
        outputs=sorted(set(output_list)),
        metrics=_json_safe(
            {
                **common_metrics,
                "raw_manifest_sha256": sha256_file(manifest_path),
            }
        ),
        criteria={
            "raw_acquisition_completed": True,
            "production_raw_manifest_verified": True,
            "phase2_pass_criteria_met": False,
        },
        notes=notes,
    )
    _record_progress(progress, acquisition_progress)

    category_counts = disclosure_diagnostics.get("failure_category_counts", {})
    failed_downloads = int(disclosure_diagnostics.get("failed_download_count", 0))
    downloaded_documents = int(disclosure_diagnostics.get("downloaded_count", 0))
    all_document_requests_unavailable = bool(
        not limited
        and len(documents) > 0
        and downloaded_documents == 0
        and failed_downloads == len(documents)
        and int(category_counts.get("SOURCE_UNAVAILABLE", 0)) == failed_downloads
    )
    if all_document_requests_unavailable:
        blocked_outcome = replace(
            acquisition_progress,
            criteria={
                **acquisition_progress.criteria,
                "official_document_endpoint_reachable": False,
                "all_selected_document_downloads_failed_due_request_errors": True,
            },
            notes=[
                *acquisition_progress.notes,
                (
                    "Every selected CNINFO document request exhausted the configured "
                    "request policy; detailed reasons are retained in the raw failure "
                    "log."
                ),
            ],
        )
        _record_progress(progress, blocked_outcome)
        raise OfficialSourceUnavailable(
            "CNINFO annual-report document endpoint",
            [str(sources["disclosures"]["download_base_url"])],
            f"All {failed_downloads} selected document requests failed with "
            "SOURCE_UNAVAILABLE after configured retries; see the manifested raw "
            "failure log for per-document evidence.",
            outcome=blocked_outcome,
        )

    if mode.metadata_only:
        return Phase2Outcome(
            passed=not bool(disclosure_result.failures),
            inputs=raw_inputs,
            outputs=output_list,
            metrics=_json_safe(common_metrics),
            criteria={
                "limited_trial": True,
                "production_qa_eligible": False,
                "metadata_only": True,
                "production_phase_status_written": False,
            },
            notes=notes,
        )

    extractor = _load_pdf_extractor()
    extraction_documents = documents
    if limited:
        extraction_documents = extraction_documents.loc[
            extraction_documents["download_status"].eq("DOWNLOADED")
        ].copy()
    extraction = extractor(extraction_documents, raw_root=raw_root)
    write_canonical_parquet(extraction.document_audit, outputs["document_audit"])
    write_canonical_parquet(extraction.mentions, outputs["disclosure_raw"])
    extraction_status_counts = (
        extraction.document_audit["extraction_status"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
        if "extraction_status" in extraction.document_audit
        else {}
    )
    extraction_progress = replace(
        acquisition_progress,
        outputs=sorted(
            {
                *[_relative(root, path) for path in outputs.values() if path.exists()],
                _relative(root, manifest_path),
            }
        ),
        metrics=_json_safe(
            {
                **common_metrics,
                "raw_manifest_sha256": sha256_file(manifest_path),
                "extraction": {
                    "document_audit_rows": len(extraction.document_audit),
                    "mention_rows": len(extraction.mentions),
                    "document_extraction_status_counts": extraction_status_counts,
                },
            }
        ),
    )
    _record_progress(progress, extraction_progress)

    if extraction.mentions.empty:
        if not limited:
            raise ValueError(
                "Relationship extraction produced no source-grounded mentions"
            )
        extraction_errors = 0
        if "extraction_status" in extraction.document_audit:
            extraction_errors = int(
                extraction.document_audit["extraction_status"].eq("ERROR").sum()
            )
        common_metrics["extraction_and_resolution"] = {
            "document_audit_rows": len(extraction.document_audit),
            "mention_rows": 0,
            "named_mentions": 0,
            "anonymous_mentions": 0,
            "resolved_mentions": 0,
        }
        return Phase2Outcome(
            passed=not disclosure_result.failures and extraction_errors == 0,
            inputs=raw_inputs,
            outputs=extraction_progress.outputs,
            metrics=_json_safe(common_metrics),
            criteria={
                "limited_trial": True,
                "production_qa_eligible": False,
                "metadata_only": False,
                "download_failures": len(disclosure_result.failures),
                "extraction_errors": extraction_errors,
                "mentions_in_trial_sample": 0,
                "production_phase_status_written": False,
            },
            notes=[*notes, "The limited sample contained no relationship mentions."],
        )

    resolution_config = ResolutionConfig(
        fuzzy_enabled=bool(config["entity_resolution"]["fuzzy_matching_enabled"]),
        fuzzy_min_score=float(config["entity_resolution"]["fuzzy_min_score"]),
        fuzzy_min_margin=float(config["entity_resolution"]["fuzzy_min_margin"]),
    )
    override_path = _path(
        root,
        config["entity_resolution"]["manual_override_file"],
    )
    resolution = resolve_mentions(
        extraction.mentions,
        company_aliases,
        config=resolution_config,
        override_path=override_path,
    )
    write_canonical_parquet(resolution, outputs["resolution_audit"])
    output_list = sorted(
        {
            *[_relative(root, path) for path in outputs.values() if path.exists()],
            _relative(root, manifest_path),
        }
    )
    extraction_metrics = {
        "document_audit_rows": len(extraction.document_audit),
        "mention_rows": len(extraction.mentions),
        "named_mentions": int(resolution["resolution_status"].ne("anonymous").sum()),
        "anonymous_mentions": int(
            resolution["resolution_status"].eq("anonymous").sum()
        ),
        "resolved_mentions": int(resolution["resolution_status"].eq("resolved").sum()),
        "document_extraction_status_counts": (
            extraction.document_audit["extraction_status"]
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
            if "extraction_status" in extraction.document_audit
            else {}
        ),
    }
    common_metrics["extraction_and_resolution"] = extraction_metrics
    _record_progress(
        progress,
        replace(
            extraction_progress,
            outputs=output_list,
            metrics=_json_safe(
                {
                    **common_metrics,
                    "raw_manifest_sha256": sha256_file(manifest_path),
                }
            ),
        ),
    )
    extraction_errors = 0
    if "extraction_status" in extraction.document_audit:
        extraction_errors = int(
            extraction.document_audit["extraction_status"].eq("ERROR").sum()
        )

    if limited:
        trial_ok = not disclosure_result.failures and extraction_errors == 0
        return Phase2Outcome(
            passed=trial_ok,
            inputs=raw_inputs,
            outputs=output_list,
            metrics=_json_safe(common_metrics),
            criteria={
                "limited_trial": True,
                "production_qa_eligible": False,
                "metadata_only": False,
                "download_failures": len(disclosure_result.failures),
                "extraction_errors": extraction_errors,
                "production_phase_status_written": False,
            },
            notes=notes,
        )

    qa = evaluate_phase2_qa(
        security_master=security_master,
        company_aliases=company_aliases,
        documents=extraction.document_audit,
        mentions=extraction.mentions,
        resolution_audit=resolution,
        manifest_verified=True,
        seed=int(config["project"]["random_state"]),
        resolution_config=resolution_config,
        manual_override_path=override_path,
        signal_cutoff_local=str(config["information_timing"]["signal_cutoff_local"]),
        market_timezone=str(config["information_timing"]["timezone"]),
    )
    tests_passed, test_details, tests_run = _run_required_tests(root)
    passed = bool(
        master_diagnostics["qa_state"] == "PASS"
        and cninfo_stock_map_diagnostics["qa_state"] == "PASS"
        and disclosure_diagnostics["qa_state"] == "PASS"
        and extraction_errors == 0
        and qa["passed"]
        and tests_passed
    )
    common_metrics["phase2_qa"] = qa
    common_metrics["required_test_details"] = test_details
    canonical_artifacts = [*outputs.values(), manifest_path]
    common_metrics["artifact_hashes"] = {
        _relative(root, path): sha256_file(path)
        for path in sorted(canonical_artifacts, key=lambda item: _relative(root, item))
    }
    criteria = {
        "limited_trial": False,
        "production_qa_eligible": True,
        "security_master_checks_pass": master_diagnostics["qa_state"] == "PASS",
        "cninfo_stock_map_checks_pass": (
            cninfo_stock_map_diagnostics["qa_state"] == "PASS"
        ),
        "disclosure_acquisition_checks_pass": (
            disclosure_diagnostics["qa_state"] == "PASS"
        ),
        "document_extraction_errors_zero": extraction_errors == 0,
        "relationship_and_resolution_qa_pass": qa["passed"],
        "publication_timing_qa_pass": qa["publication_timing"]["passed"],
        "alias_validity_and_future_invariance_tests_pass": tests_passed,
        "production_raw_manifest_verified": True,
        "phase2_pass_criteria_met": passed,
    }
    return Phase2Outcome(
        passed=passed,
        inputs=raw_inputs,
        outputs=sorted(set(output_list)),
        tests_run=tests_run,
        metrics=_json_safe(common_metrics),
        criteria=_json_safe(criteria),
        notes=notes,
    )


def _write_blocker_report(
    root: Path,
    blocker: OfficialSourceUnavailable,
    *,
    timestamp: datetime,
) -> Path:
    output = root / BLOCKER_REPORT
    attempted = "\n".join(f"- `{source}`" for source in blocker.attempted_sources)
    body = f"""# Phase 2 Blocker Report

Generated: {timestamp.isoformat()}

## Exact missing dependency

{blocker.dependency}

Configured respectful retries were exhausted. The terminal error was:

```text
{blocker.reason}
```

## Attempted legitimate sources

{attempted}

## Why no substitution was made

No legitimate configured or supplied fallback with equivalent provenance was available.
Bypassing access controls, scraping an unstable third-party source, or fabricating data
would violate the project specification.

## Last valid completed phase

Phase 1 (repository, schemas, and validation framework).
"""
    output.write_text(body, encoding="utf-8")
    return output


def _archive_resolved_blocker_report(root: Path, *, timestamp: datetime) -> Path | None:
    """Move a resolved root blocker report into auditable report history."""

    blocker = root / BLOCKER_REPORT
    if not blocker.is_file():
        return None
    history = root / RESOLVED_BLOCKER_HISTORY
    history.mkdir(parents=True, exist_ok=True)
    stamp = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = history / f"PHASE_2_BLOCKER_RESOLVED_{stamp}.md"
    blocker.replace(archived)
    return archived


def _archive_status_for_revalidation(
    root: Path,
    *,
    timestamp: datetime,
) -> Path | None:
    """Retire a prior conclusion before a production rerun mutates outputs.

    A killed or failing rerun therefore leaves Phase 2 as ``not_run`` or with
    the newly written failure status; it can never leave a stale production
    ``PASS`` at the canonical status path.  The prior artifact remains
    recoverable and auditable under ``reports/history``.
    """

    status_path = phase_status_path(root, PHASE)
    if not status_path.is_file():
        return None
    previous = read_phase_status(status_path)
    history = root / RESOLVED_BLOCKER_HISTORY
    history.mkdir(parents=True, exist_ok=True)
    stamp = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    context = f"{previous['config_sha256'][:12]}_{previous['source_tree_sha256'][:12]}"
    archived = history / (
        f"PHASE_2_STATUS_SUPERSEDED_{stamp}_{previous['state']}_{context}.json"
    )
    status_path.replace(archived)
    return archived


def _attach_revalidation_history(
    outcome: Phase2Outcome,
    *,
    root: Path,
    archived_status: Path | None,
) -> Phase2Outcome:
    if archived_status is None:
        return outcome
    relative = _relative(root, archived_status)
    return replace(
        outcome,
        outputs=sorted({*outcome.outputs, relative}),
        notes=[
            *outcome.notes,
            f"The superseded Phase-2 status was archived before revalidation: {relative}",
        ],
    )


def _write_production_status(
    *,
    root: Path,
    config_path: Path,
    started_at: datetime,
    state: PhaseState,
    outcome: Phase2Outcome | None = None,
    blockers: list[str] | None = None,
    notes: list[str] | None = None,
) -> None:
    outcome = outcome or Phase2Outcome(passed=False)
    status_path = phase_status_path(root, PHASE)
    status = build_phase_status(
        phase=PHASE,
        name=PHASE_NAMES[PHASE],
        state=state,
        started_at=started_at,
        finished_at=_utc_now(),
        project_root=root,
        config=config_path,
        inputs=outcome.inputs,
        outputs=[*outcome.outputs, _relative(root, status_path)],
        tests_run=outcome.tests_run,
        metrics=_json_safe(outcome.metrics),
        criteria=_json_safe(outcome.criteria),
        blockers=blockers or [],
        notes=[*outcome.notes, *(notes or [])],
    )
    existing = read_phase_status(status_path) if status_path.is_file() else None
    write_phase_status(
        status_path,
        status,
        validation_rerun=existing is not None,
        blocker_resolved=bool(
            existing
            and existing["state"] == PhaseState.BLOCKED.value
            and state is PhaseState.PASS
        ),
        defect_corrected=bool(
            existing
            and existing["state"] == PhaseState.FAIL.value
            and state is PhaseState.PASS
        ),
    )


def _write_trial_report(
    *,
    root: Path,
    started_at: datetime,
    state: str,
    mode: AcquisitionMode,
    outcome: Phase2Outcome | None = None,
    error: str | None = None,
) -> Path:
    output = root / TRIAL_REPORT
    payload = {
        "artifact_type": "PHASE_2_LIMITED_TRIAL",
        "state": state,
        "started_at": started_at.isoformat(),
        "finished_at": _utc_now().isoformat(),
        "mode": mode.label,
        "metadata_only": mode.metadata_only,
        "max_documents": mode.max_documents,
        "production_qa_eligible": False,
        "production_phase_status_written": False,
        "outcome": _json_safe(outcome.__dict__) if outcome else None,
        "error": error,
    }
    return atomic_write_json(output, payload)


def run_phase2(
    project_root: str | Path,
    config_path: str | Path,
    *,
    limited_trial: bool = False,
    metadata_only: bool = False,
    max_documents: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page_size: int = 30,
) -> int:
    """Run Phase 2 and return the frozen project exit code."""

    root = Path(project_root).resolve()
    config_file = _path(root, config_path).resolve()
    started_at = _utc_now()
    config = load_config(config_file)
    mode = AcquisitionMode(
        metadata_only=metadata_only,
        max_documents=max_documents,
    )
    if metadata_only and max_documents is not None:
        raise ValueError("Choose either --metadata-only or --max-documents, not both")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if limited_trial != (not mode.production_qa_eligible):
        raise ValueError(
            "--metadata-only/--max-documents require --limited-trial, and a "
            "limited trial must specify one of those controls"
        )

    validation_end = str(config["periods"]["validation_end"])
    if limited_trial:
        if not start_date or not end_date:
            raise ValueError(
                "A limited trial requires explicit --start-date and --end-date"
            )
        query_start_timestamp = pd.Timestamp(start_date)
        query_end_timestamp = pd.Timestamp(end_date)
        if query_start_timestamp > query_end_timestamp:
            raise ValueError("--start-date must not be after --end-date")
        if query_start_timestamp < pd.Timestamp(config["periods"]["raw_start"]):
            raise ValueError(
                "A Phase-2 trial may not precede the frozen raw-data period"
            )
        if query_end_timestamp > pd.Timestamp(validation_end):
            raise ValueError("Phase 2 may not access the 2023-2025 holdout period")
        query_start, query_end = start_date, end_date
    else:
        if start_date is not None or end_date is not None:
            raise ValueError("Production Phase 2 uses the frozen config period")
        query_start = str(config["periods"]["raw_start"])
        query_end = validation_end

    archived_status = (
        None
        if limited_trial
        else _archive_status_for_revalidation(root, timestamp=started_at)
    )
    progress: dict[str, Phase2Outcome] = {}
    try:
        outcome = _execute_flow(
            root=root,
            config=config,
            mode=mode,
            start_date=query_start,
            end_date=query_end,
            page_size=page_size,
            progress=progress,
        )
    except OfficialSourceUnavailable as exc:
        if limited_trial:
            _write_trial_report(
                root=root,
                started_at=started_at,
                state="TRIAL_BLOCKED",
                mode=mode,
                error=exc.reason,
            )
        else:
            report = _write_blocker_report(root, exc, timestamp=_utc_now())
            evidence = (
                exc.outcome or progress.get("outcome") or Phase2Outcome(passed=False)
            )
            blocker_outcome = replace(
                evidence,
                passed=False,
                outputs=sorted({*evidence.outputs, _relative(root, report)}),
                criteria={
                    **evidence.criteria,
                    "official_sources_available_after_retries": False,
                    "legitimate_fallback_available": False,
                    "phase2_pass_criteria_met": False,
                },
            )
            blocker_outcome = _attach_revalidation_history(
                blocker_outcome,
                root=root,
                archived_status=archived_status,
            )
            _write_production_status(
                root=root,
                config_path=config_file,
                started_at=started_at,
                state=PhaseState.BLOCKED,
                outcome=blocker_outcome,
                blockers=[f"{exc.dependency}: {exc.reason}"],
            )
        LOGGER.error("Official source unavailable after retries: %s", exc.reason)
        return int(ExitCode.EXTERNAL_DATA_BLOCKER)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if limited_trial:
            _write_trial_report(
                root=root,
                started_at=started_at,
                state="TRIAL_FAILED",
                mode=mode,
                error=error,
            )
        else:
            evidence = progress.get("outcome") or Phase2Outcome(passed=False)
            failure_outcome = replace(
                evidence,
                passed=False,
                criteria={
                    **evidence.criteria,
                    "phase2_pass_criteria_met": False,
                },
            )
            failure_outcome = _attach_revalidation_history(
                failure_outcome,
                root=root,
                archived_status=archived_status,
            )
            _write_production_status(
                root=root,
                config_path=config_file,
                started_at=started_at,
                state=PhaseState.FAIL,
                outcome=failure_outcome,
                notes=[error],
            )
        LOGGER.exception("Phase 2 failed")
        return int(ExitCode.ENGINEERING_FAILURE)

    if limited_trial:
        state = "TRIAL_SUCCEEDED" if outcome.passed else "TRIAL_FAILED"
        _write_trial_report(
            root=root,
            started_at=started_at,
            state=state,
            mode=mode,
            outcome=outcome,
        )
        return int(ExitCode.SUCCESS if outcome.passed else ExitCode.ENGINEERING_FAILURE)

    outcome = _attach_revalidation_history(
        outcome,
        root=root,
        archived_status=archived_status,
    )
    state = PhaseState.PASS if outcome.passed else PhaseState.FAIL
    if outcome.passed:
        archived_blocker = _archive_resolved_blocker_report(
            root,
            timestamp=_utc_now(),
        )
        if archived_blocker is not None:
            outcome = replace(
                outcome,
                outputs=sorted({*outcome.outputs, _relative(root, archived_blocker)}),
                notes=[
                    *outcome.notes,
                    "The resolved Phase-2 blocker report was archived under reports/history.",
                ],
            )
    _write_production_status(
        root=root,
        config_path=config_file,
        started_at=started_at,
        state=state,
        outcome=outcome,
    )
    return int(ExitCode.SUCCESS if outcome.passed else ExitCode.ENGINEERING_FAILURE)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire, extract, resolve, and validate Phase-2 real data."
    )
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    parser.add_argument("--limited-trial", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    root = Path(__file__).resolve().parents[1]
    try:
        return run_phase2(
            root,
            args.config,
            limited_trial=args.limited_trial,
            metadata_only=args.metadata_only,
            max_documents=args.max_documents,
            start_date=args.start_date,
            end_date=args.end_date,
            page_size=args.page_size,
        )
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.error("Invalid Phase-2 invocation: %s", exc)
        return int(ExitCode.ENGINEERING_FAILURE)


if __name__ == "__main__":
    raise SystemExit(main())
