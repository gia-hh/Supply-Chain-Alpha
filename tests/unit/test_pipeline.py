from __future__ import annotations

import os
from pathlib import Path

import pytest

from supply_chain_alpha import pipeline
from supply_chain_alpha.pipeline import (
    PHASE_NAMES,
    ExitCode,
    _clean_install_smoke,
    _status_exit_code,
)


def test_pipeline_contract_has_exact_exit_codes_and_phases() -> None:
    assert [int(code) for code in ExitCode] == [0, 1, 2, 3, 4]
    assert list(PHASE_NAMES) == list(range(1, 12))


@pytest.mark.skipif(os.name != "nt", reason="Windows anchored-relative path semantics")
@pytest.mark.parametrize("unsafe_path", ["C:", "C:Windows", "\\"])
def test_recorded_outputs_reject_windows_anchored_relative_paths(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    assert not pipeline._recorded_outputs_exist(
        tmp_path,
        {"outputs": [unsafe_path]},
    )

    safe = tmp_path / "safe.txt"
    safe.write_text("safe", encoding="utf-8")
    assert not pipeline._recorded_outputs_exist(
        tmp_path,
        {
            "outputs": ["safe.txt"],
            "metrics": {"artifact_hashes": {unsafe_path: "0" * 64}},
        },
    )


def test_required_orchestrator_entrypoint_exists() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / "scripts" / "run_pipeline.py").is_file()
    assert (root / "scripts" / "validate_all.py").is_file()


def test_mandatory_test_registry_covers_every_spec_family_and_fixture() -> None:
    root = Path(__file__).resolve().parents[2]
    required_families = {
        "data_and_schema",
        "entity_resolution",
        "graph",
        "market",
        "residuals",
        "signals",
        "statistics",
        "portfolio",
        "holdout",
        "tiny_end_to_end",
    }

    assert required_families.issubset(pipeline.MANDATORY_TEST_REGISTRY)
    assert not pipeline.missing_mandatory_test_artifacts(root)
    assert (
        "tests/integration/test_tiny_e2e_pipeline.py" in pipeline.MANDATORY_TEST_FILES
    )


def test_package_installs_and_imports_from_isolated_target() -> None:
    root = Path(__file__).resolve().parents[2]

    passed, details = _clean_install_smoke(root)

    assert passed, details


def test_holdout_integrity_exit_code_takes_priority_in_phase3() -> None:
    status = {
        "phase": 3,
        "state": "FAIL",
        "criteria": {
            "holdout_integrity_violation": True,
            "research_gate_failed": False,
        },
    }
    assert _status_exit_code(status) is ExitCode.HOLDOUT_INTEGRITY_VIOLATION


@pytest.mark.parametrize(
    ("status_source_hash", "output_exists", "expected_rerun"),
    [
        ("current-source", True, False),
        ("stale-source", True, True),
        ("current-source", False, True),
    ],
)
def test_pipeline_reuses_status_only_for_same_source_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_source_hash: str,
    output_exists: bool,
    expected_rerun: bool,
) -> None:
    status_path = tmp_path / "phase_1.json"
    status_path.write_text("{}", encoding="utf-8")
    artifact_path = tmp_path / "artifact.txt"
    if output_exists:
        artifact_path.write_text("present", encoding="utf-8")
    existing = {
        "phase": 1,
        "state": "PASS",
        "config_sha256": "current-config",
        "source_tree_sha256": status_source_hash,
        "outputs": ["artifact.txt"],
    }
    reruns: list[bool] = []

    def rerun(*_: object) -> dict[str, object]:
        reruns.append(True)
        artifact_path.write_text("regenerated", encoding="utf-8")
        return {**existing, "source_tree_sha256": "current-source"}

    monkeypatch.setattr(pipeline, "load_config", lambda _: {})
    monkeypatch.setattr(pipeline, "config_sha256", lambda _: "current-config")
    monkeypatch.setattr(pipeline, "source_tree_manifest", lambda _: {})
    monkeypatch.setattr(
        pipeline, "source_tree_manifest_sha256", lambda _: "current-source"
    )
    monkeypatch.setattr(
        pipeline, "phase_status_path", lambda _root, _phase: status_path
    )
    monkeypatch.setattr(pipeline, "read_phase_status", lambda _: existing)
    monkeypatch.setattr(
        pipeline,
        "run_phase1",
        rerun,
    )
    monkeypatch.setattr(
        pipeline, "write_run_manifest", lambda *_: tmp_path / "run.json"
    )

    result = pipeline.run_pipeline(tmp_path, through_phase=1)

    assert bool(reruns) is expected_rerun
    assert result.exit_code is ExitCode.SUCCESS


def test_pipeline_retries_same_context_blocked_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "phase_1.json"
    status_path.write_text("{}", encoding="utf-8")
    (tmp_path / "blocker.txt").write_text("temporary", encoding="utf-8")
    existing = {
        "phase": 1,
        "state": "BLOCKED",
        "config_sha256": "current-config",
        "source_tree_sha256": "current-source",
        "outputs": ["blocker.txt"],
        "criteria": {},
    }
    reruns: list[bool] = []

    monkeypatch.setattr(pipeline, "load_config", lambda _: {})
    monkeypatch.setattr(pipeline, "config_sha256", lambda _: "current-config")
    monkeypatch.setattr(pipeline, "source_tree_manifest", lambda _: {})
    monkeypatch.setattr(
        pipeline, "source_tree_manifest_sha256", lambda _: "current-source"
    )
    monkeypatch.setattr(
        pipeline, "phase_status_path", lambda _root, _phase: status_path
    )
    monkeypatch.setattr(pipeline, "read_phase_status", lambda _: existing)
    monkeypatch.setattr(
        pipeline,
        "run_phase1",
        lambda *_: (
            reruns.append(True)
            or {
                **existing,
                "state": "PASS",
                "outputs": ["blocker.txt"],
            }
        ),
    )
    monkeypatch.setattr(
        pipeline, "write_run_manifest", lambda *_: tmp_path / "run.json"
    )

    result = pipeline.run_pipeline(tmp_path, through_phase=1)

    assert reruns == [True]
    assert result.exit_code is ExitCode.SUCCESS


def test_pipeline_reruns_when_a_hashed_output_was_mutated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "phase_1.json"
    status_path.write_text("{}", encoding="utf-8")
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("original", encoding="utf-8")
    original_hash = pipeline.sha256_file(artifact)
    existing = {
        "phase": 1,
        "state": "PASS",
        "config_sha256": "current-config",
        "source_tree_sha256": "current-source",
        "outputs": ["artifact.txt"],
        "metrics": {"artifact_hashes": {"artifact.txt": original_hash}},
    }
    artifact.write_text("tampered", encoding="utf-8")
    reruns: list[bool] = []

    def rerun(*_: object) -> dict[str, object]:
        reruns.append(True)
        return {
            **existing,
            "metrics": {
                "artifact_hashes": {"artifact.txt": pipeline.sha256_file(artifact)}
            },
        }

    monkeypatch.setattr(pipeline, "load_config", lambda _: {})
    monkeypatch.setattr(pipeline, "config_sha256", lambda _: "current-config")
    monkeypatch.setattr(pipeline, "source_tree_manifest", lambda _: {})
    monkeypatch.setattr(
        pipeline, "source_tree_manifest_sha256", lambda _: "current-source"
    )
    monkeypatch.setattr(
        pipeline, "phase_status_path", lambda _root, _phase: status_path
    )
    monkeypatch.setattr(pipeline, "read_phase_status", lambda _: existing)
    monkeypatch.setattr(
        pipeline,
        "run_phase1",
        rerun,
    )
    monkeypatch.setattr(
        pipeline, "write_run_manifest", lambda *_: tmp_path / "run.json"
    )

    result = pipeline.run_pipeline(tmp_path, through_phase=1)

    assert reruns == [True]
    assert result.exit_code is ExitCode.SUCCESS


def test_pipeline_rejects_status_from_mid_run_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "phase_1.json"
    returned = {
        "phase": 1,
        "state": "PASS",
        "config_sha256": "current-config",
        "source_tree_sha256": "changed-source",
        "outputs": ["artifact.txt"],
    }
    monkeypatch.setattr(pipeline, "load_config", lambda _: {})
    monkeypatch.setattr(pipeline, "config_sha256", lambda _: "current-config")
    monkeypatch.setattr(pipeline, "source_tree_manifest", lambda _: {})
    monkeypatch.setattr(
        pipeline, "source_tree_manifest_sha256", lambda _: "current-source"
    )
    monkeypatch.setattr(
        pipeline, "phase_status_path", lambda _root, _phase: status_path
    )
    monkeypatch.setattr(pipeline, "run_phase1", lambda *_: returned)

    with pytest.raises(RuntimeError, match="different source tree"):
        pipeline.run_pipeline(tmp_path, through_phase=1)
