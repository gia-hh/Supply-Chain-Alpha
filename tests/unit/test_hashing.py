from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from supply_chain_alpha.utils import hashing as hashing_module
from supply_chain_alpha.utils.hashing import (
    canonical_json_sha256,
    config_sha256,
    get_git_commit,
    source_provenance,
    source_tree_manifest,
    source_tree_manifest_sha256,
    source_tree_sha256,
    validate_source_tree_manifest,
    verify_source_tree_manifest,
    write_source_tree_manifest,
)


def _make_directory_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")


def test_canonical_json_hash_ignores_mapping_order() -> None:
    left = {"z": [3, 2, 1], "a": {"unicode": "供应链", "enabled": True}}
    right = {"a": {"enabled": True, "unicode": "供应链"}, "z": [3, 2, 1]}
    assert canonical_json_sha256(left) == canonical_json_sha256(right)


def test_canonical_json_hash_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        canonical_json_sha256({"bad": float("nan")})


def test_config_hash_is_semantic_and_normalises_yaml_dates(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("period: 2021-01-01\nseed: 42\n", encoding="utf-8")
    second.write_text("seed: 42\nperiod: 2021-01-01\n", encoding="utf-8")
    assert config_sha256(first) == config_sha256(second)
    assert config_sha256(first) == canonical_json_sha256(
        {"period": "2021-01-01", "seed": 42}
    )


def test_source_tree_hash_changes_on_source_mutation_and_ignores_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    report = tmp_path / "reports" / "status.json"
    report.parent.mkdir()
    report.write_text("generated", encoding="utf-8")

    first_manifest = source_tree_manifest(tmp_path)
    first_hash = source_tree_manifest_sha256(first_manifest)
    assert [record["path"] for record in first_manifest["files"]] == ["src/module.py"]

    report.write_text("changed output", encoding="utf-8")
    assert source_tree_sha256(tmp_path) == first_hash
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path) != first_hash


def test_source_tree_manifest_prunes_excluded_deep_trees_before_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    excluded_roots = [tmp_path / "reports", tmp_path / "vendor"]
    for excluded_root in excluded_roots:
        deep = excluded_root / "one" / "two" / "three"
        deep.mkdir(parents=True)
        (deep / "large.bin").write_bytes(b"x" * 1024)

    original_scandir = os.scandir
    visited: list[Path] = []

    def guarded_scandir(path: str | bytes | os.PathLike[str]) -> os.ScandirIterator:
        candidate = Path(path).resolve()
        visited.append(candidate)
        if any(
            candidate == excluded or excluded in candidate.parents
            for excluded in excluded_roots
        ):
            raise AssertionError(f"excluded tree was accessed: {candidate}")
        return original_scandir(path)

    monkeypatch.setattr(hashing_module.os, "scandir", guarded_scandir)
    manifest = source_tree_manifest(tmp_path, exclude_paths=["vendor"])

    assert [record["path"] for record in manifest["files"]] == ["src/module.py"]
    assert not any(
        candidate == excluded or excluded in candidate.parents
        for candidate in visited
        for excluded in excluded_roots
    )


def test_source_tree_manifest_preserves_canonical_order_and_hash_semantics(
    tmp_path: Path,
) -> None:
    content_by_path = {
        "a/deep.py": b"deep\n",
        "m.py": b"middle\n",
        "z/source.py": b"last\n",
    }
    for relative, content in content_by_path.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    ignored = tmp_path / "src" / "__pycache__" / "module.pyc"
    ignored.parent.mkdir(parents=True)
    ignored.write_bytes(b"generated")

    expected = {
        "version": 1,
        "algorithm": "sha256",
        "files": [
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for relative, content in sorted(content_by_path.items())
        ],
    }
    manifest = source_tree_manifest(tmp_path)

    assert manifest == expected
    assert source_tree_manifest_sha256(manifest) == canonical_json_sha256(expected)


def test_excluded_directory_symlink_is_rejected_before_pruning(tmp_path: Path) -> None:
    target = tmp_path / "concealed_source"
    target.mkdir()
    (target / "payload.py").write_text("VALUE = 1\n", encoding="utf-8")
    _make_directory_symlink(tmp_path / "reports", target)

    with pytest.raises(ValueError, match="does not permit symlinks.*reports"):
        source_tree_manifest(tmp_path)


def test_excluded_link_boundary_cannot_bypass_check_when_links_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "concealed.py").write_text("VALUE = 1\n", encoding="utf-8")
    original = hashing_module._path_is_link_like
    inspected: list[Path] = []

    def simulated_link(path: Path) -> bool:
        inspected.append(path)
        return path == reports or original(path)

    monkeypatch.setattr(hashing_module, "_path_is_link_like", simulated_link)
    with pytest.raises(ValueError, match="does not permit symlinks.*reports"):
        source_tree_manifest(tmp_path)

    assert reports in inspected


def test_source_tree_root_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "real_root"
    target.mkdir()
    (target / "source.py").write_text("pass\n", encoding="utf-8")
    link = tmp_path / "linked_root"
    _make_directory_symlink(link, target)

    with pytest.raises(ValueError, match="root must not be a symlink"):
        source_tree_manifest(link)


def test_source_tree_manifest_detects_file_change_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    original_hash = hashing_module.sha256_file

    def mutating_hash(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
        digest = original_hash(path, chunk_size=chunk_size)
        Path(path).write_text("VALUE = 2\n", encoding="utf-8")
        return digest

    monkeypatch.setattr(hashing_module, "sha256_file", mutating_hash)
    with pytest.raises(RuntimeError, match="changed while hashing: source.py"):
        source_tree_manifest(tmp_path)


def test_generated_phase_status_document_is_not_source_provenance(
    tmp_path: Path,
) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    status_document = tmp_path / "PHASE_1_3_STATUS.md"
    status_document.write_text("pending\n", encoding="utf-8")
    before = source_tree_sha256(tmp_path)

    status_document.write_text("final metrics\n", encoding="utf-8")

    assert source_tree_sha256(tmp_path) == before


def test_generated_blocker_report_is_not_source_provenance(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = source_tree_sha256(tmp_path)

    (tmp_path / "BLOCKER_REPORT.md").write_text("blocked\n", encoding="utf-8")

    assert source_tree_sha256(tmp_path) == before


def test_source_tree_manifest_write_does_not_hash_itself(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text("stable", encoding="utf-8")
    output = tmp_path / "SOURCE_MANIFEST.json"
    first = write_source_tree_manifest(tmp_path, output)
    second = write_source_tree_manifest(tmp_path, output)
    assert first == second
    assert [record["path"] for record in second["files"]] == ["source.txt"]
    verify_source_tree_manifest(tmp_path, output)

    (tmp_path / "source.txt").write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="changed=.*source.txt"):
        verify_source_tree_manifest(tmp_path, output)


def test_manifest_validation_rejects_unsorted_or_tampered_records(
    tmp_path: Path,
) -> None:
    (tmp_path / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    manifest = source_tree_manifest(tmp_path)
    validate_source_tree_manifest(manifest)

    reversed_manifest = {**manifest, "files": list(reversed(manifest["files"]))}
    with pytest.raises(ValueError, match="strictly sorted"):
        validate_source_tree_manifest(reversed_manifest)

    tampered = {**manifest, "files": [dict(record) for record in manifest["files"]]}
    tampered["files"][0]["sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_source_tree_manifest(tampered)


@pytest.mark.parametrize("unsafe", [r"C:escape", r"C:\\escape", r"\\escape"])
def test_source_tree_excludes_reject_windows_anchored_paths(
    tmp_path: Path, unsafe: str
) -> None:
    (tmp_path / "source.py").write_text("pass\n", encoding="utf-8")

    with pytest.raises(ValueError, match="project-relative"):
        source_tree_manifest(tmp_path, exclude_paths=[unsafe])


@pytest.mark.parametrize("unsafe", [r"C:escape", r"C:\\escape", r"\\escape"])
def test_manifest_validation_rejects_windows_anchored_paths(unsafe: str) -> None:
    manifest = {
        "version": 1,
        "algorithm": "sha256",
        "files": [{"path": unsafe, "bytes": 0, "sha256": "0" * 64}],
    }

    with pytest.raises(ValueError, match="project-relative"):
        validate_source_tree_manifest(manifest)


def test_no_git_repository_uses_source_tree_provenance(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("pass\n", encoding="utf-8")
    assert get_git_commit(tmp_path) is None
    provenance = source_provenance(tmp_path)
    assert provenance["git_commit"] is None
    assert provenance["source_tree_sha256"] == source_tree_manifest_sha256(
        provenance["source_tree_manifest"]
    )


def test_git_commit_is_read_without_launching_a_path_executable(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    ref = git_dir / "refs" / "heads" / "main"
    ref.parent.mkdir(parents=True)
    git_dir.joinpath("HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
    ref.write_text("A" * 40 + "\n", encoding="ascii")

    assert get_git_commit(tmp_path) == "a" * 40


def test_git_commit_rejects_escaping_or_oversized_head(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    git_dir.joinpath("HEAD").write_text("ref: ../secret\n", encoding="ascii")
    assert get_git_commit(tmp_path) is None

    git_dir.joinpath("HEAD").write_text("a" * 4097, encoding="ascii")
    assert get_git_commit(tmp_path) is None


def test_git_repository_still_uses_source_tree_provenance(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("pass\n", encoding="utf-8")
    with patch(
        "supply_chain_alpha.utils.hashing.get_git_commit",
        return_value="a" * 40,
    ):
        provenance = source_provenance(tmp_path)

    assert provenance["git_commit"] == "a" * 40
    assert provenance["source_tree_sha256"] == source_tree_manifest_sha256(
        provenance["source_tree_manifest"]
    )
