from __future__ import annotations

from pathlib import Path

import pytest

from supply_chain_alpha.evaluation.holdout import (
    HoldoutIntegrityError,
    authorize_holdout_access,
    write_research_freeze,
)

SOURCE_HASH = "1" * 64
CONFIG_HASH = "2" * 64
REGISTRY_HASH = "3" * 64


def _freeze(path: Path) -> None:
    write_research_freeze(
        path,
        frozen_at="2022-12-31T16:00:00Z",
        source_tree_sha256=SOURCE_HASH,
        config_sha256=CONFIG_HASH,
        experiment_registry_sha256=REGISTRY_HASH,
        portfolio_candidate=False,
    )


def test_holdout_access_is_denied_before_freeze(tmp_path: Path) -> None:
    with pytest.raises(HoldoutIntegrityError, match="denied before research freeze"):
        authorize_holdout_access(
            tmp_path / "missing.json",
            source_tree_sha256=SOURCE_HASH,
            config_sha256=CONFIG_HASH,
            experiment_registry_sha256=REGISTRY_HASH,
        )


def test_holdout_same_hash_rerun_is_authorized(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    _freeze(path)

    authorized = authorize_holdout_access(
        path,
        source_tree_sha256=SOURCE_HASH,
        config_sha256=CONFIG_HASH,
        experiment_registry_sha256=REGISTRY_HASH,
    )

    assert authorized["portfolio_candidate"] is False


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("source_tree_sha256", "4" * 64),
        ("config_sha256", "5" * 64),
        ("experiment_registry_sha256", "6" * 64),
    ],
)
def test_holdout_rejects_post_freeze_drift(
    tmp_path: Path,
    field: str,
    changed: str,
) -> None:
    path = tmp_path / "freeze.json"
    _freeze(path)
    current = {
        "source_tree_sha256": SOURCE_HASH,
        "config_sha256": CONFIG_HASH,
        "experiment_registry_sha256": REGISTRY_HASH,
    }
    current[field] = changed

    with pytest.raises(HoldoutIntegrityError, match=field):
        authorize_holdout_access(path, **current)
