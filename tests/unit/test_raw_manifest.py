import json
from pathlib import Path

import pytest

from supply_chain_alpha.data.disclosures import verify_raw_manifest, write_raw_manifest


def test_raw_manifest_records_v2_provenance_contract(tmp_path: Path):
    raw = tmp_path / "raw.txt"
    raw.write_text("original", encoding="utf-8")
    manifest = tmp_path / "MANIFEST.json"

    write_raw_manifest(
        [raw],
        manifest,
        source="official-test-source",
        source_url_or_identifier="https://example.test/raw.txt",
        retrieval_datetime="2026-08-29T12:00:00+00:00",
    )

    records = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["source"] == "official-test-source"
    assert records[0]["source_url_or_identifier"] == "https://example.test/raw.txt"
    assert records[0]["retrieval_datetime"] == "2026-08-29T12:00:00+00:00"
    assert records[0]["file_size"] == len(b"original")
    assert len(records[0]["sha256"]) == 64
    assert "bytes" not in records[0]
    verify_raw_manifest(manifest)


def test_raw_manifest_detects_mutation(tmp_path: Path):
    raw = tmp_path / "raw.txt"
    raw.write_text("original", encoding="utf-8")
    manifest = tmp_path / "MANIFEST.json"
    write_raw_manifest([raw], manifest)
    verify_raw_manifest(manifest)
    # Same-length mutation proves verification is not relying on file size.
    raw.write_text("changed!", encoding="utf-8")
    with pytest.raises(ValueError, match="immutability violation"):
        verify_raw_manifest(manifest)


def test_raw_manifest_detects_unlisted_inventory_file(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("original", encoding="utf-8")
    manifest = tmp_path / "MANIFEST.json"
    write_raw_manifest([raw], manifest)
    (tmp_path / "new.txt").write_text("new", encoding="utf-8")

    with pytest.raises(ValueError, match="unmanifested files: new.txt"):
        verify_raw_manifest(manifest, inventory_root=tmp_path)


def test_raw_manifest_rejects_missing_provenance(tmp_path: Path):
    raw = tmp_path / "raw.txt"
    raw.write_text("original", encoding="utf-8")
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "path": "raw.txt",
                    "file_size": raw.stat().st_size,
                    "sha256": "0" * 64,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing fields"):
        verify_raw_manifest(manifest)


def test_raw_manifest_rejects_naive_retrieval_datetime(tmp_path: Path):
    raw = tmp_path / "raw.txt"
    raw.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="timezone offset"):
        write_raw_manifest(
            [raw],
            tmp_path / "MANIFEST.json",
            retrieval_datetime="2026-08-29T12:00:00",
        )
