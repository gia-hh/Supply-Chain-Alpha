"""Append-only experiment registry with deterministic integrity checks.

The CSV contains exactly the fields mandated by project-spec section 22.  A
companion ``<registry>.integrity.json`` file records the canonical CSV digest
and row count.  It is deliberately part of the public contract: appends refuse
to proceed if either file has been edited independently.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from supply_chain_alpha.utils.hashing import atomic_write_json

EXPERIMENT_REGISTRY_FIELDS = (
    "experiment_id",
    "timestamp",
    "phase",
    "sample_period",
    "signal_name",
    "direction",
    "horizon",
    "edge_weighting",
    "residual_model",
    "controls",
    "purpose",
    "pre_specified",
    "result_artifact",
)

_INTEGRITY_FIELDS = {"format_version", "registry_sha256", "row_count"}
_INTEGRITY_FORMAT_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PHASE = re.compile(r"^(?:phase[ _-]?)?(\d+)$", flags=re.IGNORECASE)


class ExperimentRegistryError(RuntimeError):
    """Base class for registry validation and integrity failures."""


class ExperimentRegistryIntegrityError(ExperimentRegistryError):
    """Raised when existing registry state is missing, malformed, or changed."""


class DuplicateExperimentError(ExperimentRegistryError):
    """Raised when an experiment ID is already present in the registry."""


class ExperimentRegistryBusyError(ExperimentRegistryError):
    """Raised instead of risking a lost update during a concurrent append."""


@dataclass(frozen=True)
class RegistryVerification:
    """Deterministic verification result for an experiment registry."""

    path: Path
    sha256: str
    row_count: int
    experiment_ids: tuple[str, ...]


def experiment_registry_integrity_path(path: str | Path) -> Path:
    """Return the documented companion integrity path for *path*."""

    registry_path = Path(path)
    return registry_path.with_name(f"{registry_path.name}.integrity.json")


def _lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalised = value.strip()
    if not normalised:
        raise ValueError(f"{field} must not be empty")
    if any(character in normalised for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{field} must not contain control line breaks")
    return normalised


def _normalise_timestamp(value: Any) -> str:
    timestamp_text = _require_text(value, "timestamp")
    try:
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be a valid ISO-8601 datetime") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    canonical = timestamp.astimezone(timezone.utc).isoformat()
    return canonical.replace("+00:00", "Z")


def _normalise_phase(value: Any) -> str:
    if isinstance(value, bool):
        raise TypeError("phase must identify a numbered project phase")
    text = _require_text(str(value), "phase")
    match = _PHASE.fullmatch(text)
    if match is None or int(match.group(1)) < 7:
        raise ValueError("phase must be phase 7 or later")
    return str(int(match.group(1)))


def _normalise_result_artifact(value: Any) -> str:
    text = _require_text(value, "result_artifact").replace("\\", "/")
    native = Path(text)
    portable = PurePosixPath(text)
    if (
        native.is_absolute()
        or native.anchor
        or portable.is_absolute()
        or ".." in portable.parts
    ):
        raise ValueError("result_artifact must be a project-relative path")
    normalised = portable.as_posix()
    if normalised in {"", "."}:
        raise ValueError("result_artifact must identify a file")
    return normalised


def _normalise_record(record: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    keys = set(record)
    expected = set(EXPERIMENT_REGISTRY_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(
            f"experiment record fields must match exactly; missing={missing}, extra={extra}"
        )

    experiment_id = _require_text(record["experiment_id"], "experiment_id")
    if _EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ValueError(
            "experiment_id must contain only letters, digits, dot, underscore, or hyphen"
        )
    pre_specified = record["pre_specified"]
    if not isinstance(pre_specified, bool):
        raise TypeError("pre_specified must be boolean")

    result: dict[str, str] = {}
    for field in EXPERIMENT_REGISTRY_FIELDS:
        if field == "experiment_id":
            result[field] = experiment_id
        elif field == "timestamp":
            result[field] = _normalise_timestamp(record[field])
        elif field == "phase":
            result[field] = _normalise_phase(record[field])
        elif field == "pre_specified":
            result[field] = "true" if pre_specified else "false"
        elif field == "result_artifact":
            result[field] = _normalise_result_artifact(record[field])
        else:
            result[field] = _require_text(record[field], field)
    return result


def _normalise_persisted_row(row: Mapping[str, Any]) -> dict[str, str]:
    persisted = dict(row)
    token = persisted.get("pre_specified")
    if token not in {"true", "false"}:
        raise ExperimentRegistryIntegrityError(
            "registry pre_specified values must be canonical true/false tokens"
        )
    persisted["pre_specified"] = token == "true"
    try:
        return _normalise_record(persisted)
    except (TypeError, ValueError) as exc:
        raise ExperimentRegistryIntegrityError(
            f"registry contains a non-canonical row: {exc}"
        ) from exc


def _csv_bytes(rows: list[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=EXPERIMENT_REGISTRY_FIELDS,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _parse_canonical_csv(data: bytes) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExperimentRegistryIntegrityError("registry is not valid UTF-8") from exc
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != EXPERIMENT_REGISTRY_FIELDS:
            raise ExperimentRegistryIntegrityError(
                "registry header does not exactly match required field order"
            )
        rows = [dict(row) for row in reader]
    except csv.Error as exc:
        raise ExperimentRegistryIntegrityError("registry CSV is malformed") from exc

    canonical_rows = [_normalise_persisted_row(row) for row in rows]
    if canonical_rows != rows or _csv_bytes(canonical_rows) != data:
        raise ExperimentRegistryIntegrityError("registry CSV is not canonical")
    identifiers = [row["experiment_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ExperimentRegistryIntegrityError(
            "registry contains duplicate experiment_id values"
        )
    return rows


def _load_integrity(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ExperimentRegistryIntegrityError("registry integrity file is a symlink")
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentRegistryIntegrityError(
            "registry integrity file is missing or invalid"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != _INTEGRITY_FIELDS:
        raise ExperimentRegistryIntegrityError("registry integrity schema is invalid")
    if payload["format_version"] != _INTEGRITY_FORMAT_VERSION:
        raise ExperimentRegistryIntegrityError(
            "registry integrity version is unsupported"
        )
    if not isinstance(payload["row_count"], int) or isinstance(
        payload["row_count"], bool
    ):
        raise ExperimentRegistryIntegrityError(
            "registry integrity row_count is invalid"
        )
    if (
        not isinstance(payload["registry_sha256"], str)
        or _SHA256.fullmatch(payload["registry_sha256"]) is None
    ):
        raise ExperimentRegistryIntegrityError("registry integrity digest is invalid")
    return payload


def _read_and_verify(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[RegistryVerification, list[dict[str, str]], bytes]:
    integrity_path = experiment_registry_integrity_path(path)
    if path.is_symlink():
        raise ExperimentRegistryIntegrityError("experiment registry is a symlink")
    if not path.is_file():
        raise ExperimentRegistryIntegrityError("experiment registry is missing")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExperimentRegistryIntegrityError(
            "experiment registry is unreadable"
        ) from exc
    rows = _parse_canonical_csv(data)
    digest = hashlib.sha256(data).hexdigest()
    integrity = _load_integrity(integrity_path)
    if integrity["registry_sha256"] != digest or integrity["row_count"] != len(rows):
        raise ExperimentRegistryIntegrityError(
            "experiment registry differs from its append-only integrity record"
        )
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or _SHA256.fullmatch(expected_sha256) is None
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        if digest != expected_sha256:
            raise ExperimentRegistryIntegrityError(
                "experiment registry does not match expected_sha256"
            )
    verification = RegistryVerification(
        path=path,
        sha256=digest,
        row_count=len(rows),
        experiment_ids=tuple(row["experiment_id"] for row in rows),
    )
    return verification, rows, data


def verify_experiment_registry(
    path: str | Path, *, expected_sha256: str | None = None
) -> RegistryVerification:
    """Validate schema, canonical bytes, unique IDs, and recorded integrity.

    ``expected_sha256`` can additionally bind validation to a run manifest or
    research-freeze digest.
    """

    verification, _, _ = _read_and_verify(Path(path), expected_sha256=expected_sha256)
    return verification


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(data)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


@contextmanager
def _exclusive_append_lock(path: Path) -> Iterator[None]:
    lock_path = _lock_path(path)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ExperimentRegistryBusyError(
            f"another registry append holds {lock_path.name}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(b"experiment-registry append lock\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def append_experiment(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> RegistryVerification:
    """Atomically append one canonical row and update its integrity companion.

    There is no update or delete operation.  If the registry already exists,
    its companion digest must verify before an append.  ``expected_sha256`` is
    an optional stronger compare-and-append guard for callers retaining the
    digest from the preceding run.
    """

    row = _normalise_record(record)
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    integrity_path = experiment_registry_integrity_path(registry_path)

    with _exclusive_append_lock(registry_path):
        if registry_path.exists() or integrity_path.exists():
            if not registry_path.exists() or not integrity_path.exists():
                raise ExperimentRegistryIntegrityError(
                    "registry and integrity companion must either both exist or both be absent"
                )
            _, existing_rows, existing_bytes = _read_and_verify(
                registry_path, expected_sha256=expected_sha256
            )
        else:
            if expected_sha256 is not None:
                raise ExperimentRegistryIntegrityError(
                    "expected_sha256 was supplied for a registry that does not exist"
                )
            existing_rows = []
            existing_bytes = b""

        if row["experiment_id"] in {
            existing["experiment_id"] for existing in existing_rows
        }:
            raise DuplicateExperimentError(
                f"experiment_id already exists: {row['experiment_id']}"
            )

        new_bytes = _csv_bytes([*existing_rows, row])
        if existing_bytes and not new_bytes.startswith(existing_bytes):
            raise ExperimentRegistryIntegrityError(
                "append would alter existing registry bytes"
            )
        _atomic_write_bytes(registry_path, new_bytes)
        digest = hashlib.sha256(new_bytes).hexdigest()
        atomic_write_json(
            integrity_path,
            {
                "format_version": _INTEGRITY_FORMAT_VERSION,
                "registry_sha256": digest,
                "row_count": len(existing_rows) + 1,
            },
        )
        return verify_experiment_registry(registry_path, expected_sha256=digest)


__all__ = [
    "EXPERIMENT_REGISTRY_FIELDS",
    "DuplicateExperimentError",
    "ExperimentRegistryBusyError",
    "ExperimentRegistryError",
    "ExperimentRegistryIntegrityError",
    "RegistryVerification",
    "append_experiment",
    "experiment_registry_integrity_path",
    "verify_experiment_registry",
]
