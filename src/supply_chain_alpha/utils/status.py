"""Strict, atomic phase-status artifacts for the project orchestrator."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .hashing import (
    SHA256_HEX_LENGTH,
    atomic_write_json,
    canonical_json_bytes,
    config_sha256,
    source_provenance,
    source_tree_manifest_sha256,
    validate_source_tree_manifest,
)


class PhaseState(str, Enum):
    """The four engineering states permitted by the global contract."""

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    SKIPPED_BY_DESIGN = "SKIPPED_BY_DESIGN"


PHASE_STATES = frozenset(state.value for state in PhaseState)
REQUIRED_STATUS_FIELDS = frozenset(
    {
        "phase",
        "name",
        "state",
        "started_at",
        "finished_at",
        "git_commit",
        "config_sha256",
        "inputs",
        "outputs",
        "tests_run",
        "metrics",
        "criteria",
        "blockers",
        "notes",
    }
)
SOURCE_FALLBACK_FIELDS = frozenset({"source_tree_manifest", "source_tree_sha256"})
ALLOWED_STATUS_FIELDS = REQUIRED_STATUS_FIELDS | SOURCE_FALLBACK_FIELDS


class StatusValidationError(ValueError):
    """Raised when a phase-status artifact violates its schema."""


class InvalidStateTransition(StatusValidationError):
    """Raised when an existing status is relabeled impermissibly."""


@dataclass(frozen=True)
class PhaseStatus:
    """Typed representation of the required phase-status schema."""

    phase: int
    name: str
    state: PhaseState | str
    started_at: str
    finished_at: str
    git_commit: str | None
    config_sha256: str
    inputs: list[Any] = field(default_factory=list)
    outputs: list[Any] = field(default_factory=list)
    tests_run: list[Any] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    criteria: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_tree_manifest: dict[str, Any] | None = None
    source_tree_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, schema-valid mapping."""

        data = asdict(self)
        data["state"] = (
            self.state.value if isinstance(self.state, PhaseState) else self.state
        )
        if data["source_tree_manifest"] is None:
            data.pop("source_tree_manifest")
        if data["source_tree_sha256"] is None:
            data.pop("source_tree_sha256")
        return validate_phase_status(data)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PhaseStatus:
        """Validate a mapping and construct a detached status object."""

        data = validate_phase_status(value)
        data["state"] = PhaseState(data["state"])
        return cls(**data)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _parse_timestamp(field_name: str, value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise StatusValidationError(f"{field_name} must be a non-empty ISO-8601 string")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise StatusValidationError(
            f"{field_name} must be a valid ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StatusValidationError(f"{field_name} must include a UTC offset")
    return parsed


def _require_json_value(field_name: str, value: Any) -> None:
    """Reject values that the on-disk JSON representation cannot preserve."""

    try:
        # canonical_json_bytes rejects NaN, sets, and non-string object keys.
        canonical_json_bytes(value)
        # The status schema itself is JSON-only (unlike parsed YAML config
        # values), so reject convenience conversions such as datetime/Path.
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StatusValidationError(
            f"{field_name} must contain only JSON values"
        ) from exc


def validate_phase_status(status: Mapping[str, Any] | PhaseStatus) -> dict[str, Any]:
    """Validate the exact phase-status schema and return a detached mapping."""

    if isinstance(status, PhaseStatus):
        status = status.to_dict()
    if not isinstance(status, Mapping):
        raise StatusValidationError("Phase status must be a mapping")

    fields = set(status)
    missing = REQUIRED_STATUS_FIELDS - fields
    unknown = fields - ALLOWED_STATUS_FIELDS
    if missing:
        raise StatusValidationError(
            f"Missing phase-status fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise StatusValidationError(
            f"Unknown phase-status fields: {', '.join(sorted(unknown))}"
        )

    data = copy.deepcopy(dict(status))
    phase = data["phase"]
    if isinstance(phase, bool) or not isinstance(phase, int) or phase < 1:
        raise StatusValidationError("phase must be a positive integer")
    if not isinstance(data["name"], str) or not data["name"].strip():
        raise StatusValidationError("name must be a non-empty string")
    if not isinstance(data["state"], str) or data["state"] not in PHASE_STATES:
        raise StatusValidationError(
            "state must be exactly one of: " + ", ".join(sorted(PHASE_STATES))
        )

    started_at = _parse_timestamp("started_at", data["started_at"])
    finished_at = _parse_timestamp("finished_at", data["finished_at"])
    if finished_at < started_at:
        raise StatusValidationError("finished_at must not precede started_at")

    git_commit = data["git_commit"]
    if git_commit is not None and not _is_git_commit(git_commit):
        raise StatusValidationError(
            "git_commit must be null or a full 40/64-character Git hash"
        )
    if not _is_sha256(data["config_sha256"]):
        raise StatusValidationError(
            "config_sha256 must be a 64-character hexadecimal SHA-256"
        )

    for field_name in ("inputs", "outputs", "tests_run", "blockers", "notes"):
        if not isinstance(data[field_name], list):
            raise StatusValidationError(f"{field_name} must be a list")
        _require_json_value(field_name, data[field_name])
    for field_name in ("blockers", "notes"):
        if any(not isinstance(item, str) for item in data[field_name]):
            raise StatusValidationError(f"Every {field_name} item must be a string")
    for field_name in ("metrics", "criteria"):
        if not isinstance(data[field_name], dict):
            raise StatusValidationError(f"{field_name} must be an object")
        _require_json_value(field_name, data[field_name])

    provenance_fields = SOURCE_FALLBACK_FIELDS & fields
    if provenance_fields != SOURCE_FALLBACK_FIELDS:
        raise StatusValidationError(
            "Every phase status requires source_tree_manifest and source_tree_sha256"
        )

    try:
        manifest = validate_source_tree_manifest(data["source_tree_manifest"])
    except (TypeError, ValueError) as exc:
        raise StatusValidationError(f"Invalid source_tree_manifest: {exc}") from exc
    manifest_digest = data["source_tree_sha256"]
    if not _is_sha256(manifest_digest):
        raise StatusValidationError(
            "source_tree_sha256 must be a 64-character hexadecimal SHA-256"
        )
    expected_digest = source_tree_manifest_sha256(manifest)
    if manifest_digest != expected_digest:
        raise StatusValidationError(
            "source_tree_sha256 does not match source_tree_manifest"
        )
    data["source_tree_manifest"] = manifest

    _require_json_value("phase status", data)
    return data


def _coerce_state(state: PhaseState | str | None, *, previous: bool = False) -> str:
    if isinstance(state, PhaseState):
        return state.value
    if previous and state in {None, "not_run"}:
        return "not_run"
    if not isinstance(state, str) or state not in PHASE_STATES:
        label = "previous_state" if previous else "new_state"
        allowed = sorted(PHASE_STATES | ({"not_run"} if previous else set()))
        raise InvalidStateTransition(f"{label} must be one of: {', '.join(allowed)}")
    return state


def validate_state_transition(
    previous_state: PhaseState | str | None,
    new_state: PhaseState | str,
    *,
    validation_rerun: bool = False,
    blocker_resolved: bool = False,
    defect_corrected: bool = False,
) -> None:
    """Enforce the state-transition table and its revalidation conditions."""

    previous_value = _coerce_state(previous_state, previous=True)
    new_value = _coerce_state(new_state)
    if previous_value == "not_run":
        return

    if previous_value == new_value:
        if not validation_rerun:
            raise InvalidStateTransition(
                "Overwriting an existing status requires validation_rerun=True"
            )
        return

    if (
        previous_value == PhaseState.BLOCKED.value
        and new_value == PhaseState.PASS.value
    ):
        if not blocker_resolved or not validation_rerun:
            raise InvalidStateTransition(
                "BLOCKED -> PASS requires blocker_resolved=True and validation_rerun=True"
            )
        return

    if previous_value == PhaseState.FAIL.value and new_value == PhaseState.PASS.value:
        if not defect_corrected or not validation_rerun:
            raise InvalidStateTransition(
                "FAIL -> PASS requires defect_corrected=True and validation_rerun=True"
            )
        return

    raise InvalidStateTransition(
        f"State transition is not permitted: {previous_value} -> {new_value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StatusValidationError(f"Duplicate JSON key in phase status: {key}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    raise StatusValidationError(
        f"Non-standard JSON numeric value in phase status: {value}"
    )


def read_phase_status(path: str | Path) -> dict[str, Any]:
    """Read and validate a phase-status JSON artifact."""

    status_path = Path(path)
    try:
        with status_path.open("r", encoding="utf-8") as file_handle:
            status = json.load(
                file_handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonstandard_number,
            )
    except json.JSONDecodeError as exc:
        raise StatusValidationError(
            f"Invalid phase-status JSON: {status_path}"
        ) from exc
    return validate_phase_status(status)


def write_phase_status(
    path: str | Path,
    status: Mapping[str, Any] | PhaseStatus,
    *,
    previous_status: Mapping[str, Any] | PhaseStatus | None = None,
    validation_rerun: bool = False,
    blocker_resolved: bool = False,
    defect_corrected: bool = False,
) -> Path:
    """Validate and atomically write a phase-status artifact.

    When *path* already exists its status is used as the transition origin.
    A supplied *previous_status* is useful when migrating an artifact to a new
    path, and must agree with an existing artifact if both are present.
    """

    output_path = Path(path)
    data = validate_phase_status(status)
    prior: dict[str, Any] | None = None
    if previous_status is not None:
        prior = validate_phase_status(previous_status)
    if output_path.exists():
        on_disk = read_phase_status(output_path)
        if prior is not None and prior != on_disk:
            raise InvalidStateTransition(
                "previous_status does not match the existing artifact"
            )
        prior = on_disk

    if prior is not None:
        if data["phase"] != prior["phase"] or data["name"] != prior["name"]:
            raise InvalidStateTransition(
                "An existing phase artifact cannot change phase or name"
            )
        validate_state_transition(
            prior["state"],
            data["state"],
            validation_rerun=validation_rerun,
            blocker_resolved=blocker_resolved,
            defect_corrected=defect_corrected,
        )
    else:
        validate_state_transition(None, data["state"])

    return atomic_write_json(output_path, data)


def build_phase_status(
    *,
    phase: int,
    name: str,
    state: PhaseState | str,
    started_at: str | datetime,
    finished_at: str | datetime,
    project_root: str | Path,
    config: Mapping[str, Any] | str | Path,
    inputs: Iterable[Any] = (),
    outputs: Iterable[Any] = (),
    tests_run: Iterable[Any] = (),
    metrics: Mapping[str, Any] | None = None,
    criteria: Mapping[str, Any] | None = None,
    blockers: Iterable[str] = (),
    notes: Iterable[str] = (),
) -> PhaseStatus:
    """Build a status with canonical config and Git/source-tree provenance."""

    def format_timestamp(value: str | datetime) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise StatusValidationError(
                    "Phase timestamps must include a UTC offset"
                )
            return value.isoformat()
        return value

    data: dict[str, Any] = {
        "phase": phase,
        "name": name,
        "state": state.value if isinstance(state, PhaseState) else state,
        "started_at": format_timestamp(started_at),
        "finished_at": format_timestamp(finished_at),
        "config_sha256": config_sha256(config),
        "inputs": list(inputs),
        "outputs": list(outputs),
        "tests_run": list(tests_run),
        "metrics": dict(metrics or {}),
        "criteria": dict(criteria or {}),
        "blockers": list(blockers),
        "notes": list(notes),
    }
    data.update(source_provenance(project_root))
    return PhaseStatus.from_mapping(data)


def phase_status_path(project_root: str | Path, phase: int) -> Path:
    """Return ``reports/status/phase_<N>.json`` after validating *phase*."""

    if isinstance(phase, bool) or not isinstance(phase, int) or phase < 1:
        raise ValueError("phase must be a positive integer")
    return Path(project_root) / "reports" / "status" / f"phase_{phase}.json"


__all__ = [
    "ALLOWED_STATUS_FIELDS",
    "PHASE_STATES",
    "REQUIRED_STATUS_FIELDS",
    "SOURCE_FALLBACK_FIELDS",
    "InvalidStateTransition",
    "PhaseState",
    "PhaseStatus",
    "StatusValidationError",
    "build_phase_status",
    "phase_status_path",
    "read_phase_status",
    "validate_phase_status",
    "validate_state_transition",
    "write_phase_status",
]
