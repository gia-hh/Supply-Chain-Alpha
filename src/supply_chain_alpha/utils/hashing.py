"""Deterministic hashing helpers used by reproducibility artifacts.

The project can be distributed without ``.git`` metadata.  In that case the
source-tree manifest produced here is the auditable replacement for a commit
identifier.  Manifest paths are project-relative and use POSIX separators so
that the same tree hashes identically on Windows and POSIX systems.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

SHA256_HEX_LENGTH = 64
SOURCE_TREE_MANIFEST_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024

# Run outputs and environment-specific files must not alter the source hash.
# Raw and processed data have their own manifest under the global engineering
# contract, so they are intentionally excluded here as well.
DEFAULT_SOURCE_TREE_EXCLUDES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "BLOCKER_REPORT.md",
        "dist",
        "PHASE_1_3_STATUS.md",
        "reports",
        "data/raw",
        "data/interim",
        "data/processed",
        "venv",
    }
)


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of *data*."""

    if not isinstance(data, bytes):
        raise TypeError("sha256_bytes expects bytes")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Hash a file without loading it entirely into memory."""

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_for_json(value: Any) -> Any:
    """Convert supported Python values to a deterministic JSON value.

    YAML parses ISO dates as :class:`datetime.date`; normalising them to their
    ISO representation makes hashing a parsed YAML config practical while
    retaining the exact date semantics.  Sets and non-string mapping keys are
    rejected because neither has an unambiguous JSON representation.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical JSON does not permit NaN or infinity")
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return _normalise_for_json(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _normalise_for_json(asdict(value))
    if isinstance(value, Mapping):
        normalised: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Canonical JSON mapping keys must be strings")
            normalised[key] = _normalise_for_json(item)
        return normalised
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise_for_json(item) for item in value]
    raise TypeError(f"Unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize *value* to the project's canonical UTF-8 JSON form."""

    normalised = _normalise_for_json(value)
    return json.dumps(
        normalised,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Hash a value after canonical JSON serialization."""

    return sha256_bytes(canonical_json_bytes(value))


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_config(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as file_handle:
        if suffix == ".json":
            config = json.load(
                file_handle, object_pairs_hook=_reject_duplicate_json_keys
            )
        elif suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(file_handle)
        else:
            raise ValueError(f"Unsupported config format for hashing: {path.suffix}")
    if not isinstance(config, Mapping):
        raise TypeError(f"Config must be a mapping: {path}")
    return config


def config_sha256(config_or_path: Mapping[str, Any] | str | Path) -> str:
    """Hash a parsed config canonically, independent of key order/whitespace."""

    if isinstance(config_or_path, (str, Path)):
        config: Mapping[str, Any] = _load_config(Path(config_or_path))
    elif isinstance(config_or_path, Mapping):
        config = config_or_path
    else:
        raise TypeError("config_sha256 expects a mapping or a YAML/JSON path")
    return canonical_json_sha256(config)


def _normalise_relative_path(path: str | Path) -> str:
    raw = Path(path)
    if raw.is_absolute() or raw.anchor:
        raise ValueError(f"Excluded paths must be project-relative: {path}")
    value = PurePosixPath(raw.as_posix()).as_posix().strip("/")
    if value in {"", "."}:
        raise ValueError("Excluded paths may not name the source-tree root")
    if ".." in PurePosixPath(value).parts:
        raise ValueError(f"Excluded path escapes source-tree root: {path}")
    return value


def _path_is_excluded(relative_path: str, excluded: frozenset[str]) -> bool:
    parts = PurePosixPath(relative_path).parts
    if any(
        part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        for part in parts
    ):
        return True
    if any(part.endswith(".egg-info") for part in parts):
        return True
    if relative_path.endswith((".pyc", ".pyo")):
        return True
    return any(
        relative_path == item or relative_path.startswith(f"{item}/")
        for item in excluded
    )


def _path_is_link_like(path: Path) -> bool:
    """Return whether *path* is a symlink or Windows reparse-point link.

    ``Path.is_symlink`` covers ordinary links on every supported platform.
    The explicit reparse-point check additionally fails closed for Windows
    junctions, which otherwise may be traversed like directories on older
    Python versions.
    """

    if path.is_symlink():
        return True
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    return bool(attributes & reparse_flag)


def _raise_walk_error(error: OSError) -> None:
    """Make ``os.walk`` traversal failures explicit rather than skipping data."""

    raise error


def source_tree_manifest(
    root: str | Path,
    *,
    exclude_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Build a deterministic SHA-256 manifest for a source tree.

    Generated artifacts, caches, virtual environments, and separately
    manifested data directories are excluded by default.  Additional excludes
    are relative to *root*.  Symlinks are rejected so a manifest can never
    silently depend on a machine-specific path outside the project.
    """

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Source-tree root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Source-tree root is not a directory: {root_path}")
    if _path_is_link_like(root_path):
        raise ValueError(f"Source-tree root must not be a symlink: {root_path}")

    excluded = frozenset(
        set(DEFAULT_SOURCE_TREE_EXCLUDES)
        | {_normalise_relative_path(path) for path in exclude_paths}
    )
    files: list[dict[str, Any]] = []
    for current_root, directory_names, file_names in os.walk(
        root_path,
        topdown=True,
        onerror=_raise_walk_error,
        followlinks=False,
    ):
        current = Path(current_root)
        if _path_is_link_like(current):
            relative = current.relative_to(root_path).as_posix()
            raise ValueError(
                f"Source-tree manifest does not permit symlinks: {relative}"
            )

        retained_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current / name
            relative_path = candidate.relative_to(root_path).as_posix()
            # Inspect the exclusion boundary itself before pruning.  Otherwise
            # an excluded directory could be replaced by a link and conceal a
            # path outside the intended tree.
            if _path_is_link_like(candidate):
                raise ValueError(
                    f"Source-tree manifest does not permit symlinks: {relative_path}"
                )
            if not _path_is_excluded(relative_path, excluded):
                retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in sorted(file_names):
            candidate = current / name
            relative_path = candidate.relative_to(root_path).as_posix()
            if _path_is_link_like(candidate):
                raise ValueError(
                    f"Source-tree manifest does not permit symlinks: {relative_path}"
                )
            if _path_is_excluded(relative_path, excluded):
                continue
            stat_before = candidate.stat()
            if not stat.S_ISREG(stat_before.st_mode):
                continue
            digest = sha256_file(candidate)
            if _path_is_link_like(candidate):
                raise ValueError(
                    f"Source-tree manifest does not permit symlinks: {relative_path}"
                )
            stat_after = candidate.stat()
            if (stat_before.st_size, stat_before.st_mtime_ns) != (
                stat_after.st_size,
                stat_after.st_mtime_ns,
            ):
                raise RuntimeError(
                    f"Source file changed while hashing: {relative_path}"
                )
            files.append(
                {
                    "path": relative_path,
                    "bytes": stat_after.st_size,
                    "sha256": digest,
                }
            )

    # Directory-first walking is not globally path-sorted when a root-level
    # file sorts after a child directory.  Preserve the original canonical
    # ordering, and therefore the exact manifest hash, explicitly.
    files.sort(key=lambda record: record["path"])

    return {
        "version": SOURCE_TREE_MANIFEST_VERSION,
        "algorithm": "sha256",
        "files": files,
    }


def validate_source_tree_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached source-tree manifest mapping."""

    if not isinstance(manifest, Mapping):
        raise TypeError("Source-tree manifest must be a mapping")
    required_keys = {"version", "algorithm", "files"}
    if set(manifest) != required_keys:
        raise ValueError(
            "Source-tree manifest fields must be exactly: "
            + ", ".join(sorted(required_keys))
        )
    if manifest["version"] != SOURCE_TREE_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported source-tree manifest version: {manifest['version']!r}"
        )
    if manifest["algorithm"] != "sha256":
        raise ValueError("Source-tree manifest algorithm must be 'sha256'")
    records = manifest["files"]
    if not isinstance(records, list):
        raise TypeError("Source-tree manifest files must be a list")

    previous_path: str | None = None
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValueError(
                "Each source-tree record requires exactly path, bytes, and sha256"
            )
        path = record["path"]
        byte_count = record["bytes"]
        digest = record["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or Path(path).anchor
        ):
            raise ValueError(
                "Manifest paths must be non-empty project-relative strings"
            )
        pure_path = PurePosixPath(path)
        if pure_path.as_posix() != path or ".." in pure_path.parts:
            raise ValueError(f"Manifest path is not canonical: {path!r}")
        if path in seen or (previous_path is not None and path <= previous_path):
            raise ValueError("Manifest file paths must be unique and strictly sorted")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError(f"Manifest byte count is invalid for {path!r}")
        if not _is_sha256(digest):
            raise ValueError(f"Manifest SHA-256 is invalid for {path!r}")
        seen.add(path)
        previous_path = path

    # Round-tripping through canonical JSON also ensures nested values are safe.
    return json.loads(canonical_json_bytes(manifest).decode("utf-8"))


def source_tree_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the canonical digest of a validated source-tree manifest."""

    return canonical_json_sha256(validate_source_tree_manifest(manifest))


def read_source_tree_manifest(path: str | Path) -> dict[str, Any]:
    """Read a source-tree manifest while rejecting duplicate JSON keys."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file_handle:
        try:
            manifest = json.load(
                file_handle, object_pairs_hook=_reject_duplicate_json_keys
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid source-tree manifest JSON: {manifest_path}"
            ) from exc
    return validate_source_tree_manifest(manifest)


def source_tree_sha256(
    root: str | Path,
    *,
    exclude_paths: Iterable[str | Path] = (),
) -> str:
    """Build and hash a source-tree manifest."""

    return source_tree_manifest_sha256(
        source_tree_manifest(root, exclude_paths=exclude_paths)
    )


def verify_source_tree_manifest(
    root: str | Path,
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    exclude_paths: Iterable[str | Path] = (),
) -> None:
    """Raise when the current source tree differs from a stored manifest."""

    root_path = Path(root).resolve()
    additional_excludes = list(exclude_paths)
    if isinstance(manifest_or_path, Mapping):
        expected = validate_source_tree_manifest(manifest_or_path)
    else:
        manifest_path = Path(manifest_or_path).resolve()
        expected = read_source_tree_manifest(manifest_path)
        try:
            additional_excludes.append(manifest_path.relative_to(root_path))
        except ValueError:
            pass

    actual = source_tree_manifest(root_path, exclude_paths=additional_excludes)
    if actual == expected:
        return

    expected_records = {record["path"]: record for record in expected["files"]}
    actual_records = {record["path"]: record for record in actual["files"]}
    missing = sorted(expected_records.keys() - actual_records.keys())
    unexpected = sorted(actual_records.keys() - expected_records.keys())
    changed = sorted(
        path
        for path in expected_records.keys() & actual_records.keys()
        if expected_records[path] != actual_records[path]
    )
    details = []
    if changed:
        details.append(f"changed={changed}")
    if missing:
        details.append(f"missing={missing}")
    if unexpected:
        details.append(f"unexpected={unexpected}")
    raise ValueError("Source-tree manifest mismatch: " + "; ".join(details))


def get_git_commit(root: str | Path) -> str | None:
    """Return a simple repository's HEAD without executing a PATH binary.

    Source-tree hashing is the authoritative fallback, so linked worktrees and
    packed-only refs may safely return ``None``.  Reading a small local HEAD is
    preferable to launching an untrusted executable named ``git`` from PATH.
    """

    git_dir = Path(root).resolve() / ".git"
    if not git_dir.is_dir() or git_dir.is_symlink():
        return None

    def read_small(path: Path) -> str | None:
        try:
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 4096:
                return None
            return path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            return None

    head = read_small(git_dir / "HEAD")
    if head is None:
        return None
    if head.startswith("ref: "):
        ref = PurePosixPath(head[5:].strip())
        if ref.is_absolute() or ".." in ref.parts or ref.parts[:1] != ("refs",):
            return None
        commit = read_small(git_dir.joinpath(*ref.parts))
        if commit is None:
            return None
    else:
        commit = head
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdefABCDEF" for character in commit
    ):
        return None
    return commit.lower()


def source_provenance(root: str | Path) -> dict[str, Any]:
    """Return Git provenance plus an independently verifiable source manifest."""

    commit = get_git_commit(root)
    manifest = source_tree_manifest(root)
    return {
        "git_commit": commit,
        "source_tree_manifest": manifest,
        "source_tree_sha256": source_tree_manifest_sha256(manifest),
    }


def atomic_write_json(path: str | Path, value: Any, *, indent: int = 2) -> Path:
    """Atomically replace *path* with UTF-8 JSON serialized in its directory."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalised = _normalise_for_json(value)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8", newline="\n"
        ) as file_handle:
            json.dump(
                normalised,
                file_handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=indent,
                allow_nan=False,
            )
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return output_path


def write_source_tree_manifest(
    root: str | Path,
    output_path: str | Path,
    *,
    exclude_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Build and atomically write a source-tree manifest."""

    root_path = Path(root).resolve()
    output = Path(output_path).resolve()
    additional_excludes = list(exclude_paths)
    try:
        additional_excludes.append(output.relative_to(root_path))
    except ValueError:
        pass
    manifest = source_tree_manifest(root_path, exclude_paths=additional_excludes)
    atomic_write_json(output, manifest)
    return manifest


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


# Explicit aliases keep call sites readable while preserving one implementation.
hash_json = canonical_json_sha256
hash_config = config_sha256


__all__ = [
    "DEFAULT_SOURCE_TREE_EXCLUDES",
    "SHA256_HEX_LENGTH",
    "SOURCE_TREE_MANIFEST_VERSION",
    "atomic_write_json",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "config_sha256",
    "get_git_commit",
    "hash_config",
    "hash_json",
    "read_source_tree_manifest",
    "sha256_bytes",
    "sha256_file",
    "source_provenance",
    "source_tree_manifest",
    "source_tree_manifest_sha256",
    "source_tree_sha256",
    "validate_source_tree_manifest",
    "verify_source_tree_manifest",
    "write_source_tree_manifest",
]
