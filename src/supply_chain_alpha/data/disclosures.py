from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import DISCLOSURE_RAW, coerce_disclosure_types, validate_table

_NAME_PATTERNS = {
    "customer": re.compile(
        r"(?:客户名称|客户名|主要客户|客户)\s*[：:]\s*([^\n\r,，;；]{2,80})"
    ),
    "supplier": re.compile(
        r"(?:供应商名称|供应商名|主要供应商|供应商)\s*[：:]\s*([^\n\r,，;；]{2,80})"
    ),
}

_RAW_MANIFEST_FIELDS = frozenset(
    {
        "path",
        "source",
        "source_url_or_identifier",
        "retrieval_datetime",
        "file_size",
        "sha256",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RAW_MANIFEST_HASH_WORKERS = 4
_RAW_MANIFEST_HASH_BATCH_SIZE = 256
LOGGER = logging.getLogger(__name__)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _retrieval_timestamp(value: str | datetime | None) -> str:
    timestamp = datetime.now(timezone.utc) if value is None else value
    if isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("retrieval_datetime must be an ISO-8601 datetime") from exc
    elif isinstance(timestamp, datetime):
        parsed = timestamp
    else:
        raise TypeError("retrieval_datetime must be a string, datetime, or None")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("retrieval_datetime must include a timezone offset")
    return parsed.isoformat()


def _manifested_path(path: Path, manifest_parent: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(manifest_parent.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def write_raw_manifest(
    paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    source: str = "local_file",
    source_url_or_identifier: str | None = None,
    retrieval_datetime: str | datetime | None = None,
) -> None:
    """Write a deterministic raw-file inventory with complete provenance.

    ``source_url_or_identifier`` defaults to each stored path, which keeps the
    original two-argument API useful for local files while callers acquiring
    remote data can provide the authoritative source URL explicitly.
    """

    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")
    if source_url_or_identifier is not None and (
        not isinstance(source_url_or_identifier, str)
        or not source_url_or_identifier.strip()
    ):
        raise ValueError("source_url_or_identifier must be a non-empty string")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_resolved = output.resolve()
    timestamp = _retrieval_timestamp(retrieval_datetime)
    records: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for item in paths:
        path = Path(item)
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"Raw manifest input is not a regular file: {path}")
        if resolved == output_resolved:
            raise ValueError("Raw manifest cannot include itself")
        if resolved in seen_paths:
            raise ValueError(f"Raw manifest contains a duplicate path: {path}")
        seen_paths.add(resolved)
        stored_path = _manifested_path(path, output.parent)
        records.append(
            {
                "path": stored_path,
                "source": source.strip(),
                "source_url_or_identifier": source_url_or_identifier or stored_path,
                "retrieval_datetime": timestamp,
                "file_size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    records.sort(key=lambda record: record["path"])
    output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _validate_manifest_record(record: Any, *, index: int) -> None:
    if not isinstance(record, dict):
        raise TypeError(f"Raw manifest record {index} must be an object")
    missing = sorted(_RAW_MANIFEST_FIELDS - record.keys())
    if missing:
        raise ValueError(f"Raw manifest record {index} is missing fields: {missing}")
    for field in (
        "path",
        "source",
        "source_url_or_identifier",
        "retrieval_datetime",
        "sha256",
    ):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"Raw manifest record {index} has invalid {field}")
    if isinstance(record["file_size"], bool) or not isinstance(
        record["file_size"], int
    ):
        raise TypeError(f"Raw manifest record {index} has invalid file_size")
    if record["file_size"] < 0:
        raise ValueError(f"Raw manifest record {index} has negative file_size")
    if not _SHA256_RE.fullmatch(record["sha256"]):
        raise ValueError(f"Raw manifest record {index} has invalid sha256")
    _retrieval_timestamp(record["retrieval_datetime"])


def verify_raw_manifest(
    manifest_path: str | Path,
    *,
    inventory_root: str | Path | None = None,
) -> None:
    """Verify hashes and exact file inventory for an immutable raw-data store.

    ``inventory_root`` defines the directory that the manifest must cover
    exactly.  For backwards-compatible callers, a sibling ``official_sources``
    directory is selected when every record is inside it; otherwise the
    manifest parent is the inventory root.  The manifest itself is excluded
    when it lives inside that root.
    """

    manifest = Path(manifest_path)
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("Raw manifest root must be a list")
    resolved_records: list[tuple[dict[str, Any], Path]] = []
    for index, record in enumerate(records):
        _validate_manifest_record(record, index=index)
        stored_path = Path(record["path"])
        path = (
            stored_path if stored_path.is_absolute() else manifest.parent / stored_path
        )
        resolved_records.append((record, path.resolve()))

    if inventory_root is None:
        official_sources_path = manifest.parent / "official_sources"
        if official_sources_path.is_symlink():
            raise ValueError(
                f"Raw inventory root cannot be a symlink: {official_sources_path}"
            )
        official_sources = official_sources_path.resolve()
        if official_sources.is_dir() and all(
            path == official_sources or official_sources in path.parents
            for _, path in resolved_records
        ):
            root = official_sources
        else:
            root = manifest.parent.resolve()
    else:
        inventory_root_path = Path(inventory_root)
        if inventory_root_path.is_symlink():
            raise ValueError(
                f"Raw inventory root cannot be a symlink: {inventory_root_path}"
            )
        root = inventory_root_path.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Raw inventory root is not a directory: {root}")
    if root.is_symlink():
        raise ValueError(f"Raw inventory root cannot be a symlink: {root}")

    seen_paths: set[Path] = set()
    manifest_resolved = manifest.resolve()
    for record, resolved in resolved_records:
        if resolved != root and root not in resolved.parents:
            raise ValueError(
                f"Raw manifest record is outside the inventory root: {record['path']}"
            )
        if resolved == manifest_resolved:
            raise ValueError("Raw manifest cannot include itself")
        if resolved in seen_paths:
            raise ValueError(
                f"Raw manifest contains a duplicate path: {record['path']}"
            )
        seen_paths.add(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(f"Manifested raw file is missing: {resolved}")

    def inspect_file(item: tuple[dict[str, Any], Path]) -> tuple[int, str]:
        _, resolved = item
        return resolved.stat().st_size, sha256_file(resolved)

    # Hashing releases the GIL and the raw corpus consists of many independent
    # PDFs.  A small bounded pool materially reduces wall time on SSD storage
    # while executor.map preserves manifest order and deterministic failures.
    with ThreadPoolExecutor(
        max_workers=min(_RAW_MANIFEST_HASH_WORKERS, max(1, len(resolved_records))),
        thread_name_prefix="raw-manifest-sha256",
    ) as executor:
        for offset in range(0, len(resolved_records), _RAW_MANIFEST_HASH_BATCH_SIZE):
            batch = resolved_records[offset : offset + _RAW_MANIFEST_HASH_BATCH_SIZE]
            inspections = executor.map(inspect_file, batch)
            for batch_index, (
                (record, resolved),
                (actual_size, actual_hash),
            ) in enumerate(zip(batch, inspections, strict=True), start=1):
                if (
                    actual_size != record["file_size"]
                    or actual_hash != record["sha256"]
                ):
                    raise ValueError(f"Raw data immutability violation: {resolved}")
                completed = offset + batch_index
                if completed % 1000 == 0 or completed == len(resolved_records):
                    LOGGER.info(
                        "Raw manifest verification progress completed=%d total=%d",
                        completed,
                        len(resolved_records),
                    )

    actual_paths: set[Path] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"Raw inventory does not permit symlinks: {candidate}")
        if candidate.is_file():
            resolved = candidate.resolve(strict=True)
            if resolved != manifest_resolved:
                actual_paths.add(resolved)

    unmanifested = sorted(
        path.relative_to(root).as_posix() for path in actual_paths - seen_paths
    )
    if unmanifested:
        raise ValueError(
            "Raw inventory contains unmanifested files: " + ", ".join(unmanifested)
        )
    out_of_inventory = sorted(
        str(path) for path in seen_paths - actual_paths if path != manifest_resolved
    )
    if out_of_inventory:
        raise ValueError(
            "Raw manifest contains files outside the exact inventory: "
            + ", ".join(out_of_inventory)
        )


def load_structured_mentions(path: str | Path) -> pd.DataFrame:
    """Load already-extracted relationship mentions and enforce the canonical raw schema."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(
            p, dtype={"source_company_id": "string", "source_document_id": "string"}
        )
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(p, lines=True)
    elif suffix == ".json":
        df = pd.read_json(p)
    elif suffix == ".parquet":
        try:
            df = pd.read_parquet(p)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet support requires pyarrow or fastparquet"
            ) from exc
    else:
        raise ValueError(f"Unsupported disclosure format: {suffix}")
    df = coerce_disclosure_types(df)
    return validate_table(df, DISCLOSURE_RAW)


def _candidate_evidence(text: str, relationship_type: str) -> list[tuple[str, str]]:
    relation = relationship_type.lower().strip()
    if relation not in _NAME_PATTERNS:
        raise ValueError("relationship_type must be customer or supplier")
    values: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _NAME_PATTERNS[relation].finditer(text):
        name = match.group(1).strip().strip("。；;：:")
        if name and name not in seen:
            values.append((name, match.group(0).strip()))
            seen.add(name)
    return values


def extract_counterparty_candidates(text: str, relationship_type: str) -> list[str]:
    """Conservative text candidate extractor; candidates still require strict entity resolution."""

    return [name for name, _ in _candidate_evidence(text, relationship_type)]


def mentions_from_text(
    text: str,
    *,
    source_company_id: str,
    relationship_type: str,
    source_period_end: str | pd.Timestamp,
    publication_datetime: str | pd.Timestamp,
    source_document_id: str,
    source_document_url_or_path: str,
) -> pd.DataFrame:
    """Create raw candidate mentions from text without inventing exposure values/shares."""
    candidates = _candidate_evidence(text, relationship_type)
    rows: list[dict[str, Any]] = []
    for name, evidence_text in candidates:
        rows.append(
            {
                "source_company_id": source_company_id,
                "counterparty_raw_name": name,
                "relationship_type": relationship_type,
                "source_period_end": source_period_end,
                "publication_datetime": publication_datetime,
                "source_document_id": source_document_id,
                "source_document_url_or_path": source_document_url_or_path,
                "evidence_text": evidence_text,
                "exposure_value": None,
                "exposure_share": None,
            }
        )
    if not rows:
        return pd.DataFrame(columns=DISCLOSURE_RAW.required_columns)
    df = coerce_disclosure_types(pd.DataFrame(rows))
    return validate_table(df, DISCLOSURE_RAW)
