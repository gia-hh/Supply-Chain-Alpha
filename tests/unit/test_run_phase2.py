from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_phase2 as phase2
from supply_chain_alpha.data.acquisition import RawAsset
from supply_chain_alpha.data.cninfo import (
    CninfoStockMapAcquisition,
    DownloadFailureCategory,
)
from supply_chain_alpha.entities.resolve import ResolutionConfig
from supply_chain_alpha.utils.status import read_phase_status

ROOT = Path(__file__).resolve().parents[2]


def _project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    config = root / "config" / "project.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        (ROOT / "config" / "project.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return root, config


def _outcome(*, passed: bool) -> phase2.Phase2Outcome:
    return phase2.Phase2Outcome(
        passed=passed,
        inputs=["data/raw/MANIFEST.json"],
        outputs=["data/processed/security_master.parquet"],
        tests_run=["pytest -q tests/unit/test_entity_resolution.py"],
        metrics={"rows": 10},
        criteria={"phase2_pass_criteria_met": passed},
        notes=["test outcome"],
    )


def _stock_map_result() -> CninfoStockMapAcquisition:
    asset = RawAsset(
        source="CNINFO SZSE stock map",
        source_url_or_identifier="https://www.cninfo.com.cn/new/data/szse_stock.json",
        retrieval_datetime="2026-08-30T04:20:00Z",
        file_size=1,
        sha256="b" * 64,
        local_path="cninfo/security_master/szse_stock.json",
    )
    stock_map = pd.DataFrame(
        {
            "cninfo_org_id": ["gssh0600000"],
            "security_id": ["SSE:600000"],
            "ticker": ["600000"],
            "category": ["A股"],
            "source_url_or_identifier": [asset.source_url_or_identifier],
            "retrieval_datetime": [asset.retrieval_datetime],
            "file_size": [asset.file_size],
            "sha256": [asset.sha256],
            "raw_local_path": [asset.local_path],
        }
    )
    return CninfoStockMapAcquisition(stock_map=stock_map, asset=asset)


def test_production_success_writes_exact_pass_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _project(tmp_path)
    monkeypatch.setattr(
        phase2,
        "_execute_flow",
        lambda **_: _outcome(passed=True),
    )

    code = phase2.run_phase2(root, config)

    assert code == 0
    status = read_phase_status(root / "reports/status/phase_2.json")
    assert status["phase"] == 2
    assert status["name"] == "real_data_and_entity_resolution"
    assert status["state"] == "PASS"
    assert status["criteria"]["phase2_pass_criteria_met"] is True


def test_limited_trial_never_writes_production_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _project(tmp_path)
    monkeypatch.setattr(
        phase2,
        "_execute_flow",
        lambda **_: _outcome(passed=True),
    )

    code = phase2.run_phase2(
        root,
        config,
        limited_trial=True,
        metadata_only=True,
        start_date="2022-01-01",
        end_date="2022-01-07",
    )

    assert code == 0
    assert not (root / "reports/status/phase_2.json").exists()
    report = json.loads(
        (root / "reports/trials/phase_2_trial.json").read_text(encoding="utf-8")
    )
    assert report["state"] == "TRIAL_SUCCEEDED"
    assert report["production_qa_eligible"] is False
    assert report["production_phase_status_written"] is False


def test_exhausted_official_source_is_blocked_with_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _project(tmp_path)

    def unavailable(**_: object) -> phase2.Phase2Outcome:
        raise phase2.OfficialSourceUnavailable(
            "CNINFO annual reports",
            ["https://www.cninfo.com.cn/official"],
            "ConnectionError after four attempts",
        )

    monkeypatch.setattr(phase2, "_execute_flow", unavailable)

    code = phase2.run_phase2(root, config)

    assert code == 2
    status = read_phase_status(root / "reports/status/phase_2.json")
    assert status["state"] == "BLOCKED"
    report = (root / "BLOCKER_REPORT.md").read_text(encoding="utf-8")
    assert "CNINFO annual reports" in report
    assert "Attempted legitimate sources" in report
    assert "Last valid completed phase" in report


def test_resolved_blocker_report_is_archived_on_production_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _project(tmp_path)

    def unavailable(**_: object) -> phase2.Phase2Outcome:
        raise phase2.OfficialSourceUnavailable(
            "CNINFO annual reports",
            ["https://www.cninfo.com.cn/official"],
            "temporary outage",
        )

    monkeypatch.setattr(phase2, "_execute_flow", unavailable)
    assert phase2.run_phase2(root, config) == 2
    assert (root / "BLOCKER_REPORT.md").is_file()

    monkeypatch.setattr(
        phase2,
        "_execute_flow",
        lambda **_: _outcome(passed=True),
    )
    assert phase2.run_phase2(root, config) == 0

    assert not (root / "BLOCKER_REPORT.md").exists()
    archived = list((root / "reports/history").glob("PHASE_2_BLOCKER_RESOLVED_*.md"))
    assert len(archived) == 1
    assert "CNINFO annual reports" in archived[0].read_text(encoding="utf-8")
    status = read_phase_status(root / "reports/status/phase_2.json")
    archived_relative = archived[0].relative_to(root).as_posix()
    assert archived_relative in status["outputs"]
    assert any("archived" in note for note in status["notes"])


def test_failed_qa_is_engineering_failure_not_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _project(tmp_path)
    monkeypatch.setattr(
        phase2,
        "_execute_flow",
        lambda **_: _outcome(passed=False),
    )

    code = phase2.run_phase2(root, config)

    assert code == 1
    status = read_phase_status(root / "reports/status/phase_2.json")
    assert status["state"] == "FAIL"
    assert status["blockers"] == []


def test_failed_revalidation_cannot_leave_stale_pass_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _project(tmp_path)
    monkeypatch.setattr(phase2, "_execute_flow", lambda **_: _outcome(passed=True))
    assert phase2.run_phase2(root, config) == 0

    def fail_with_progress(**kwargs: object) -> phase2.Phase2Outcome:
        progress = kwargs["progress"]
        assert isinstance(progress, dict)
        progress["outcome"] = phase2.Phase2Outcome(
            passed=False,
            metrics={"acquisition": {"downloaded_count": 12}},
            criteria={"production_raw_manifest_verified": True},
        )
        raise RuntimeError("extractor defect")

    monkeypatch.setattr(phase2, "_execute_flow", fail_with_progress)
    assert phase2.run_phase2(root, config) == 1

    status = read_phase_status(root / "reports/status/phase_2.json")
    assert status["state"] == "FAIL"
    assert status["metrics"]["acquisition"]["downloaded_count"] == 12
    archived = list(
        (root / "reports/history").glob("PHASE_2_STATUS_SUPERSEDED_*_PASS_*.json")
    )
    assert len(archived) == 1
    assert archived[0].relative_to(root).as_posix() in status["outputs"]


def test_phase2_refuses_holdout_dates_before_freeze(tmp_path: Path) -> None:
    root, config = _project(tmp_path)

    with pytest.raises(ValueError, match="holdout"):
        phase2.run_phase2(
            root,
            config,
            limited_trial=True,
            max_documents=1,
            start_date="2022-12-01",
            end_date="2023-01-01",
        )


def test_subset_controls_require_explicit_limited_trial(tmp_path: Path) -> None:
    root, config = _project(tmp_path)

    with pytest.raises(ValueError, match="limited-trial"):
        phase2.run_phase2(root, config, max_documents=1)


def test_refresh_raw_manifest_hashes_assets_only_in_immediate_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = _project(tmp_path)
    config = phase2.load_config(config_path)
    calls: dict[str, object] = {}

    def fake_writer(
        raw_root: Path,
        output: Path,
        *,
        verify_asset_bytes: bool = True,
    ) -> Path:
        calls["writer"] = (raw_root, output, verify_asset_bytes)
        return output

    def fake_verifier(manifest: Path, *, inventory_root: Path) -> None:
        calls["verifier"] = (manifest, inventory_root)

    monkeypatch.setattr(phase2, "_load_manifest_writer", lambda: fake_writer)
    monkeypatch.setattr(phase2, "verify_raw_manifest", fake_verifier)

    official_cache = root / "data" / "raw" / phase2.OFFICIAL_CACHE_SUBDIR
    manifest = phase2._refresh_raw_manifest(
        root=root,
        config=config,
        official_cache=official_cache,
    )

    expected = root / "data" / "raw" / "MANIFEST.json"
    assert manifest == expected
    assert calls["writer"] == (official_cache, expected, False)
    assert calls["verifier"] == (expected, official_cache)


def test_production_flow_wires_extraction_resolution_qa_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = _project(tmp_path)
    config = phase2.load_config(config_path)
    config["entity_resolution"]["fuzzy_min_score"] = 87.0
    securities = pd.DataFrame(
        {
            "security_id": ["SSE:600000"],
            "company_name": ["浦发银行股份有限公司"],
        }
    )
    aliases = pd.DataFrame(
        {
            "entity_id": ["SSE:600000"],
            "canonical_name": ["浦发银行股份有限公司"],
            "alias": ["600000"],
            "alias_type": ["ticker"],
            "valid_from": ["1999-11-10"],
            "valid_to": [None],
            "source": ["SSE official listing interval"],
        }
    )
    documents = pd.DataFrame(
        {
            "document_id": ["D1"],
            "cninfo_org_id": ["gssh0600000"],
            "security_id": ["SSE:600000"],
            "security_name": ["浦发银行"],
            "publication_datetime": ["2021-04-01T00:00:00+08:00"],
            "download_status": ["DOWNLOADED"],
        }
    )
    mentions = pd.DataFrame({"counterparty_raw_name": ["某公司"]})
    document_audit = pd.DataFrame({"extraction_status": ["SUCCESS"]})
    resolution = pd.DataFrame({"resolution_status": ["resolved"]})

    monkeypatch.setattr(
        phase2,
        "_acquire_security_master",
        lambda **_: SimpleNamespace(
            securities=securities,
            aliases=aliases,
            assets=(),
            limitations=("PIT-safe aliases",),
        ),
    )
    monkeypatch.setattr(
        phase2,
        "_acquire_disclosures",
        lambda **_: SimpleNamespace(documents=documents, assets=(), failures=()),
    )
    monkeypatch.setattr(phase2, "_acquire_stock_map", lambda **_: _stock_map_result())
    monkeypatch.setattr(
        phase2,
        "assert_security_master_qa",
        lambda _: {"qa_state": "PASS"},
    )
    monkeypatch.setattr(phase2, "company_alias_diagnostics", lambda _: {"rows": 1})
    monkeypatch.setattr(
        phase2,
        "disclosure_acquisition_diagnostics",
        lambda _: {"qa_state": "PASS"},
    )
    monkeypatch.setattr(phase2, "write_canonical_parquet", lambda *_: None)
    monkeypatch.setattr(
        phase2,
        "_load_pdf_extractor",
        lambda: (
            lambda *_args, **_kwargs: SimpleNamespace(
                mentions=mentions,
                document_audit=document_audit,
            )
        ),
    )
    monkeypatch.setattr(
        phase2, "resolve_mentions", lambda *_args, **_kwargs: resolution
    )
    monkeypatch.setattr(
        phase2,
        "_load_manifest_writer",
        lambda: lambda _raw_root, output, **_kwargs: output,
    )
    monkeypatch.setattr(phase2, "verify_raw_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(phase2, "sha256_file", lambda _: "a" * 64)
    qa_arguments: dict[str, object] = {}

    def fake_evaluate_phase2_qa(**kwargs: object) -> dict[str, object]:
        qa_arguments.update(kwargs)
        return {
            "passed": True,
            "publication_timing": {"passed": True},
            "resolution_support": {"status": "PASS", "sample_details": []},
        }

    monkeypatch.setattr(phase2, "evaluate_phase2_qa", fake_evaluate_phase2_qa)
    monkeypatch.setattr(
        phase2,
        "_run_required_tests",
        lambda _: (True, "7 passed", ["pytest -q phase2"]),
    )

    outcome = phase2._execute_flow(
        root=root,
        config=config,
        mode=phase2.AcquisitionMode(),
        start_date="2015-01-01",
        end_date="2022-12-31",
        page_size=30,
    )

    assert outcome.passed is True
    assert outcome.criteria["document_extraction_errors_zero"] is True
    assert outcome.criteria["production_raw_manifest_verified"] is True
    assert outcome.metrics["artifact_hashes"]
    assert outcome.inputs == ["data/raw/MANIFEST.json"]
    assert outcome.metrics["raw_asset_inventory"]["unique_asset_count"] == 1
    resolution_config = qa_arguments["resolution_config"]
    assert isinstance(resolution_config, ResolutionConfig)
    assert resolution_config.fuzzy_min_score == pytest.approx(87.0)
    assert resolution_config.fuzzy_min_margin == pytest.approx(4.0)
    assert qa_arguments["company_aliases"] is not None
    assert qa_arguments["mentions"] is mentions
    assert qa_arguments["manual_override_path"] == root / "config/entity_overrides.csv"
    assert qa_arguments["signal_cutoff_local"] == "15:00:00"
    assert qa_arguments["market_timezone"] == "Asia/Shanghai"


def test_production_flow_classifies_total_document_request_outage_as_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = _project(tmp_path)
    config = phase2.load_config(config_path)
    securities = pd.DataFrame(
        {
            "security_id": ["SSE:600000"],
            "company_name": ["浦发银行股份有限公司"],
        }
    )
    aliases = pd.DataFrame(
        {
            "entity_id": ["SSE:600000"],
            "canonical_name": ["浦发银行股份有限公司"],
            "alias": ["600000"],
            "alias_type": ["ticker"],
            "valid_from": ["1999-11-10"],
            "valid_to": [None],
            "source": ["SSE official listing interval"],
        }
    )
    documents = pd.DataFrame(
        {
            "document_id": ["D1", "D2"],
            "cninfo_org_id": ["gssh0600000", "gssh0600000"],
            "security_id": ["SSE:600000", "SSE:600000"],
            "security_name": ["浦发银行", "浦发银行"],
            "publication_datetime": [
                "2021-04-01T00:00:00+08:00",
                "2022-04-01T00:00:00+08:00",
            ],
            "download_status": ["FAILED", "FAILED"],
        }
    )
    failures = tuple(
        SimpleNamespace(failure_category=DownloadFailureCategory.SOURCE_UNAVAILABLE)
        for _ in range(2)
    )

    monkeypatch.setattr(
        phase2,
        "_acquire_security_master",
        lambda **_: SimpleNamespace(
            securities=securities,
            aliases=aliases,
            assets=(),
            limitations=(),
        ),
    )
    monkeypatch.setattr(
        phase2,
        "_acquire_disclosures",
        lambda **_: SimpleNamespace(
            documents=documents,
            assets=(),
            failures=failures,
        ),
    )
    monkeypatch.setattr(phase2, "_acquire_stock_map", lambda **_: _stock_map_result())
    monkeypatch.setattr(
        phase2, "assert_security_master_qa", lambda _: {"qa_state": "PASS"}
    )
    monkeypatch.setattr(phase2, "company_alias_diagnostics", lambda _: {"rows": 1})
    monkeypatch.setattr(
        phase2,
        "disclosure_acquisition_diagnostics",
        lambda _: {
            "qa_state": "FAIL",
            "downloaded_count": 0,
            "failed_download_count": 2,
            "failure_category_counts": {
                "SOURCE_UNAVAILABLE": 2,
                "CONTENT_OR_INTEGRITY_ERROR": 0,
            },
        },
    )
    monkeypatch.setattr(phase2, "write_canonical_parquet", lambda *_: None)
    monkeypatch.setattr(
        phase2,
        "_load_manifest_writer",
        lambda: lambda _raw_root, output, **_kwargs: output,
    )
    monkeypatch.setattr(phase2, "verify_raw_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(phase2, "sha256_file", lambda _: "a" * 64)

    progress: dict[str, phase2.Phase2Outcome] = {}
    with pytest.raises(phase2.OfficialSourceUnavailable) as raised:
        phase2._execute_flow(
            root=root,
            config=config,
            mode=phase2.AcquisitionMode(),
            start_date="2015-01-01",
            end_date="2022-12-31",
            page_size=30,
            progress=progress,
        )

    assert "All 2 selected document requests failed" in raised.value.reason
    assert raised.value.outcome is progress["outcome"]
    assert (
        raised.value.outcome.criteria[
            "all_selected_document_downloads_failed_due_request_errors"
        ]
        is True
    )
    assert "data/raw/MANIFEST.json" in raised.value.outcome.outputs
