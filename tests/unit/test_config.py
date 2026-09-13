from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from supply_chain_alpha.utils.config import (
    ConfigValidationError,
    load_config,
    validate_project_config,
)

ROOT = Path(__file__).resolve().parents[2]


def test_v2_project_config_validates() -> None:
    config = load_config(ROOT / "config" / "project.yaml")
    assert config["edge_weighting"]["primary"] == "equal"
    assert config["graph_coverage_gate"]["min_consecutive_calendar_years"] == 3
    assert config["data_sources"]["disclosures"]["download_workers"] == 4
    assert (
        config["data_sources"]["disclosures"]["stock_map_url"]
        == "https://www.cninfo.com.cn/new/data/szse_stock.json"
    )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "C:/Users/example/data",
        "C:drive-relative-escape",
        r"\rooted-relative-escape",
    ],
)
def test_config_rejects_anchored_machine_path(unsafe_path: str) -> None:
    config = load_config(ROOT / "config" / "project.yaml")
    bad = deepcopy(config)
    bad["paths"]["raw_dir"] = unsafe_path
    with pytest.raises(ConfigValidationError, match="project-relative"):
        validate_project_config(bad)


def test_config_rejects_holdout_or_primary_signal_drift() -> None:
    config = load_config(ROOT / "config" / "project.yaml")
    moved = deepcopy(config)
    moved["periods"]["holdout_start"] = "2024-01-01"
    with pytest.raises(ConfigValidationError, match="frozen V2 holdout"):
        validate_project_config(moved)

    weighted = deepcopy(config)
    weighted["edge_weighting"]["primary"] = "exposure_share"
    with pytest.raises(ConfigValidationError, match="primary edge weighting"):
        validate_project_config(weighted)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("signal_time", "open"),
        ("signal_cutoff_local", "14:59:59"),
        ("timezone", "UTC"),
    ],
)
def test_config_rejects_frozen_information_timing_drift(field: str, value: str) -> None:
    config = load_config(ROOT / "config" / "project.yaml")
    bad = deepcopy(config)
    bad["information_timing"][field] = value
    with pytest.raises(ConfigValidationError, match="15:00:00 Asia/Shanghai"):
        validate_project_config(bad)


@pytest.mark.parametrize("workers", [True, 0, -1, 9])
def test_config_rejects_invalid_disclosure_download_workers(workers: object) -> None:
    config = load_config(ROOT / "config" / "project.yaml")
    bad = deepcopy(config)
    bad["data_sources"]["disclosures"]["download_workers"] = workers
    with pytest.raises(ConfigValidationError, match="download_workers"):
        validate_project_config(bad)


def test_config_requires_explicit_https_cninfo_stock_map_url() -> None:
    config = load_config(ROOT / "config" / "project.yaml")
    missing = deepcopy(config)
    del missing["data_sources"]["disclosures"]["stock_map_url"]
    with pytest.raises(ConfigValidationError, match="stock_map_url"):
        validate_project_config(missing)

    insecure = deepcopy(config)
    insecure["data_sources"]["disclosures"]["stock_map_url"] = (
        "http://www.cninfo.com.cn/new/data/szse_stock.json"
    )
    with pytest.raises(ConfigValidationError, match="approved official host"):
        validate_project_config(insecure)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("security_master", "sse_url"),
        ("security_master", "szse_url"),
        ("disclosures", "query_url"),
        ("disclosures", "download_base_url"),
        ("disclosures", "stock_map_url"),
    ],
)
def test_config_rejects_nonofficial_or_credentialed_source_urls(
    section: str,
    field: str,
) -> None:
    config = load_config(ROOT / "config" / "project.yaml")
    for unsafe in (
        "https://attacker.example/source",
        "https://www.cninfo.com.cn@attacker.example/source",
    ):
        bad = deepcopy(config)
        bad["data_sources"][section][field] = unsafe
        with pytest.raises(ConfigValidationError, match="approved official host"):
            validate_project_config(bad)


def test_config_rejects_official_host_with_unfrozen_endpoint_path() -> None:
    config = load_config(ROOT / "config" / "project.yaml")
    bad = deepcopy(config)
    bad["data_sources"]["disclosures"]["query_url"] = (
        "https://www.cninfo.com.cn/redirect"
    )
    with pytest.raises(ConfigValidationError, match="frozen official endpoint path"):
        validate_project_config(bad)
