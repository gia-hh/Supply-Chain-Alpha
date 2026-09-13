from __future__ import annotations

import json
from pathlib import Path

import pytest

from supply_chain_alpha.utils.hashing import (
    source_tree_manifest,
    source_tree_manifest_sha256,
)
from supply_chain_alpha.utils.status import (
    PHASE_STATES,
    InvalidStateTransition,
    PhaseState,
    StatusValidationError,
    build_phase_status,
    read_phase_status,
    validate_phase_status,
    validate_state_transition,
    write_phase_status,
)


def _status(state: str = "PASS") -> dict[str, object]:
    source_manifest = {"version": 1, "algorithm": "sha256", "files": []}
    return {
        "phase": 1,
        "name": "repository_and_schemas",
        "state": state,
        "started_at": "2026-08-29T20:00:00+00:00",
        "finished_at": "2026-08-29T20:01:00+00:00",
        "git_commit": "a" * 40,
        "source_tree_manifest": source_manifest,
        "source_tree_sha256": source_tree_manifest_sha256(source_manifest),
        "config_sha256": "b" * 64,
        "inputs": [],
        "outputs": [],
        "tests_run": ["pytest tests/unit"],
        "metrics": {"tests_passed": 1},
        "criteria": {"phase_1_tests": True},
        "blockers": [],
        "notes": [],
    }


@pytest.mark.parametrize("state", sorted(PHASE_STATES))
def test_exactly_four_phase_states_validate(state: str) -> None:
    assert validate_phase_status(_status(state))["state"] == state


@pytest.mark.parametrize("state", ["pass", "NOT_RUN", "POSITIVE", "INCONCLUSIVE", ""])
def test_non_engineering_states_are_rejected(state: str) -> None:
    with pytest.raises(StatusValidationError, match="state must be exactly"):
        validate_phase_status(_status(state))


def test_status_schema_rejects_missing_and_unknown_fields() -> None:
    missing = _status()
    missing.pop("criteria")
    with pytest.raises(
        StatusValidationError, match="Missing phase-status fields: criteria"
    ):
        validate_phase_status(missing)

    unknown = _status()
    unknown["research_conclusion"] = "POSITIVE"
    with pytest.raises(StatusValidationError, match="Unknown phase-status fields"):
        validate_phase_status(unknown)


def test_every_status_requires_matching_source_tree_manifest(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("pass\n", encoding="utf-8")
    status = _status()
    status["git_commit"] = None
    status.pop("source_tree_manifest")
    status.pop("source_tree_sha256")
    with pytest.raises(StatusValidationError, match="Every phase status requires"):
        validate_phase_status(status)

    manifest = source_tree_manifest(tmp_path)
    status["source_tree_manifest"] = manifest
    status["source_tree_sha256"] = source_tree_manifest_sha256(manifest)
    validate_phase_status(status)

    status["source_tree_sha256"] = "0" * 64
    with pytest.raises(StatusValidationError, match="does not match"):
        validate_phase_status(status)


def test_git_commit_still_requires_source_tree_manifest() -> None:
    status = _status()
    status.pop("source_tree_manifest")
    status.pop("source_tree_sha256")
    with pytest.raises(StatusValidationError, match="Every phase status requires"):
        validate_phase_status(status)


def test_status_timestamps_are_ordered_and_timezone_aware() -> None:
    naive = _status()
    naive["started_at"] = "2026-08-29T20:00:00"
    with pytest.raises(StatusValidationError, match="UTC offset"):
        validate_phase_status(naive)

    reversed_times = _status()
    reversed_times["finished_at"] = "2026-08-29T19:59:00+00:00"
    with pytest.raises(StatusValidationError, match="must not precede"):
        validate_phase_status(reversed_times)


def test_allowed_state_transitions_require_evidence_flags() -> None:
    for state in PHASE_STATES:
        validate_state_transition(None, state)

    with pytest.raises(InvalidStateTransition, match="blocker_resolved"):
        validate_state_transition("BLOCKED", "PASS", validation_rerun=True)
    validate_state_transition(
        "BLOCKED", "PASS", validation_rerun=True, blocker_resolved=True
    )

    with pytest.raises(InvalidStateTransition, match="defect_corrected"):
        validate_state_transition("FAIL", "PASS", validation_rerun=True)
    validate_state_transition(
        "FAIL", "PASS", validation_rerun=True, defect_corrected=True
    )

    with pytest.raises(InvalidStateTransition, match="not permitted"):
        validate_state_transition("PASS", "FAIL", validation_rerun=True)


def test_status_write_is_atomic_and_validates_before_replacing(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "status" / "phase_1.json"
    original = _status("FAIL")
    write_phase_status(output, original)
    assert read_phase_status(output) == original

    invalid = dict(original)
    invalid["config_sha256"] = "bad"
    with pytest.raises(StatusValidationError, match="config_sha256"):
        write_phase_status(output, invalid, validation_rerun=True)
    assert read_phase_status(output) == original
    assert not list(output.parent.glob("*.tmp"))

    corrected = dict(original)
    corrected["state"] = "PASS"
    write_phase_status(
        output,
        corrected,
        validation_rerun=True,
        defect_corrected=True,
    )
    assert read_phase_status(output)["state"] == "PASS"


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "phase_1.json"
    text = json.dumps(_status())
    text = text.replace('"phase": 1', '"phase": 1, "phase": 2', 1)
    output.write_text(text, encoding="utf-8")
    with pytest.raises(StatusValidationError, match="Duplicate JSON key"):
        read_phase_status(output)


def test_build_phase_status_uses_no_git_fallback_and_config_hash(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = project_root / "project.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")
    (project_root / "module.py").write_text("pass\n", encoding="utf-8")

    status = build_phase_status(
        phase=1,
        name="repository_and_schemas",
        state=PhaseState.PASS,
        started_at="2026-08-29T20:00:00Z",
        finished_at="2026-08-29T20:01:00Z",
        project_root=project_root,
        config=config,
        tests_run=["pytest tests/unit"],
    ).to_dict()

    assert status["git_commit"] is None
    assert status["source_tree_sha256"] == source_tree_manifest_sha256(
        status["source_tree_manifest"]
    )
