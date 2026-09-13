from __future__ import annotations

import csv
from pathlib import Path

import pytest

from supply_chain_alpha.evaluation.registry import (
    EXPERIMENT_REGISTRY_FIELDS,
    DuplicateExperimentError,
    ExperimentRegistryIntegrityError,
    append_experiment,
    experiment_registry_integrity_path,
    verify_experiment_registry,
)


def _record(experiment_id: str, *, succeeded: bool = True) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "timestamp": "2022-06-30T16:00:00+08:00",
        "phase": "phase-7",
        "sample_period": "validation",
        "signal_name": "customer_shock",
        "direction": "customer_to_supplier",
        "horizon": "1d",
        "edge_weighting": "equal",
        "residual_model": "market_only",
        "controls": "target_own_return",
        "purpose": "PRIMARY" if succeeded else "EXPLORATORY_FAILED",
        "pre_specified": succeeded,
        "result_artifact": (
            f"reports/phase7_validation/{experiment_id}.json"
            if succeeded
            else f"reports/phase7_validation/failed/{experiment_id}.json"
        ),
    }


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def test_each_run_appends_and_never_deletes_failed_experiment(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "experiment_registry.csv"

    first = append_experiment(path, _record("failed-run", succeeded=False))
    second = append_experiment(path, _record("successful-run"))

    assert first.row_count == 1
    assert second.row_count == 2
    assert second.experiment_ids == ("failed-run", "successful-run")
    rows = _rows(path)
    assert rows[0]["experiment_id"] == "failed-run"
    assert rows[0]["purpose"] == "EXPLORATORY_FAILED"
    assert rows[0]["result_artifact"].endswith("failed/failed-run.json")
    assert tuple(rows[0]) == EXPERIMENT_REGISTRY_FIELDS


def test_duplicate_id_is_rejected_without_changing_files(tmp_path: Path) -> None:
    path = tmp_path / "experiment_registry.csv"
    append_experiment(path, _record("same-id"))
    integrity_path = experiment_registry_integrity_path(path)
    before = (path.read_bytes(), integrity_path.read_bytes())

    with pytest.raises(DuplicateExperimentError, match="same-id"):
        append_experiment(path, _record("same-id", succeeded=False))

    assert (path.read_bytes(), integrity_path.read_bytes()) == before


def test_existing_row_tampering_blocks_verification_and_append(tmp_path: Path) -> None:
    path = tmp_path / "experiment_registry.csv"
    append_experiment(path, _record("original"))
    tampered = path.read_text(encoding="utf-8").replace(
        "customer_shock", "best_after_search"
    )
    path.write_text(tampered, encoding="utf-8", newline="")

    with pytest.raises(ExperimentRegistryIntegrityError, match="integrity record"):
        verify_experiment_registry(path)
    before = path.read_bytes()
    with pytest.raises(ExperimentRegistryIntegrityError, match="integrity record"):
        append_experiment(path, _record("next"))
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", EXPERIMENT_REGISTRY_FIELDS)
def test_record_requires_every_exact_field(tmp_path: Path, field: str) -> None:
    path = tmp_path / "experiment_registry.csv"
    record = _record("missing-field")
    del record[field]

    with pytest.raises(ValueError, match="fields must match exactly"):
        append_experiment(path, record)
    assert not path.exists()


def test_record_rejects_extra_field_and_non_boolean_pre_specified(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment_registry.csv"
    extra = _record("extra")
    extra["outcome"] = "PASS"
    with pytest.raises(ValueError, match="fields must match exactly"):
        append_experiment(path, extra)

    invalid_bool = _record("invalid-bool")
    invalid_bool["pre_specified"] = "false"
    with pytest.raises(TypeError, match="must be boolean"):
        append_experiment(path, invalid_bool)


def test_verification_is_deterministic_and_can_bind_expected_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment_registry.csv"
    created = append_experiment(path, _record("deterministic"))

    first = verify_experiment_registry(path, expected_sha256=created.sha256)
    second = verify_experiment_registry(path, expected_sha256=created.sha256)

    assert first == second == created
    with pytest.raises(ExperimentRegistryIntegrityError, match="expected_sha256"):
        verify_experiment_registry(path, expected_sha256="0" * 64)


def test_noncanonical_header_is_rejected_even_with_updated_integrity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment_registry.csv"
    append_experiment(path, _record("header"))
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("experiment_id,timestamp", "timestamp,experiment_id")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")

    with pytest.raises(ExperimentRegistryIntegrityError, match="header"):
        verify_experiment_registry(path)
