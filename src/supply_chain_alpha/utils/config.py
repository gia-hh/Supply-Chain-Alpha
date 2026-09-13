from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from .network import require_official_https_url


class ConfigValidationError(ValueError):
    """Raised when the frozen research configuration is incomplete or unsafe."""


_REQUIRED_SECTIONS = {
    "project",
    "periods",
    "paths",
    "data_sources",
    "information_timing",
    "entity_resolution",
    "edge_weighting",
    "graph_coverage_gate",
    "market_panel",
    "residual_returns",
    "signals",
    "evaluation",
    "portfolio",
    "holdout",
    "engineering",
}


def _require_keys(mapping: dict[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(mapping)
    if missing:
        raise ConfigValidationError(f"{label} missing required keys: {sorted(missing)}")


def _validate_relative_path(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{label} must be a non-empty relative path")
    path = Path(value)
    # On Windows, ``is_absolute`` is false for both drive-relative paths such
    # as ``C:escape`` and rooted-relative paths such as ``\\escape``.  Either
    # form discards part of the project root when joined, so reject every
    # non-empty anchor in addition to ordinary absolute paths.
    if path.anchor or path.is_absolute() or ".." in path.parts:
        raise ConfigValidationError(
            f"{label} must be project-relative and may not escape the project"
        )


def _validate_official_source_url(
    value: Any,
    *,
    label: str,
    host: str,
    path: str,
) -> None:
    try:
        require_official_https_url(
            value,
            label=label,
            allowed_hosts={host},
            expected_path=path,
        )
    except ValueError as exc:
        raise ConfigValidationError(str(exc)) from exc


def validate_project_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the V2 contract without silently supplying research defaults."""

    if not isinstance(config, dict):
        raise ConfigValidationError("Project config must be a mapping")
    _require_keys(config, _REQUIRED_SECTIONS, "project config")

    project = config["project"]
    _require_keys(
        project,
        {"name", "version", "spec_version", "random_state", "market"},
        "project",
    )
    if float(project["spec_version"]) != 2.0:
        raise ConfigValidationError("project.spec_version must be 2.0")
    if isinstance(project["random_state"], bool) or not isinstance(
        project["random_state"], int
    ):
        raise ConfigValidationError("project.random_state must be an integer")

    periods = config["periods"]
    period_keys = {
        "raw_start",
        "warmup_end",
        "development_start",
        "development_end",
        "validation_start",
        "validation_end",
        "holdout_start",
        "holdout_end",
    }
    _require_keys(periods, period_keys, "periods")
    ordered = [
        periods["raw_start"],
        periods["warmup_end"],
        periods["development_start"],
        periods["development_end"],
        periods["validation_start"],
        periods["validation_end"],
        periods["holdout_start"],
        periods["holdout_end"],
    ]
    parsed = [yaml.safe_load(str(value)) for value in ordered]
    if any(left > right for left, right in pairwise(parsed)):
        raise ConfigValidationError("Research periods must be chronologically ordered")
    if (
        str(periods["holdout_start"]) != "2023-01-01"
        or str(periods["holdout_end"]) != "2025-12-31"
    ):
        raise ConfigValidationError(
            "The frozen V2 holdout must remain 2023-01-01 through 2025-12-31"
        )

    paths = config["paths"]
    required_paths = {
        "raw_dir",
        "interim_dir",
        "processed_dir",
        "reports_dir",
        "security_master",
        "company_alias",
        "disclosure_raw",
        "resolution_audit",
        "supply_chain_edge",
        "equity_daily",
        "returns_daily",
        "signal_daily",
        "raw_manifest",
        "source_tree_manifest",
        "status_dir",
        "experiment_registry",
        "research_freeze",
    }
    _require_keys(paths, required_paths, "paths")
    for key, value in paths.items():
        _validate_relative_path(value, f"paths.{key}")
    _validate_relative_path(
        config["entity_resolution"].get("manual_override_file"),
        "entity_resolution.manual_override_file",
    )
    _validate_relative_path(
        config["market_panel"].get("market_rules_file"),
        "market_panel.market_rules_file",
    )

    sources = config["data_sources"]
    security_master_source = sources.get("security_master", {})
    _require_keys(
        security_master_source,
        {"sse_url", "szse_url"},
        "data_sources.security_master",
    )
    _validate_official_source_url(
        security_master_source["sse_url"],
        label="data_sources.security_master.sse_url",
        host="query.sse.com.cn",
        path="/sseQuery/commonQuery.do",
    )
    _validate_official_source_url(
        security_master_source["szse_url"],
        label="data_sources.security_master.szse_url",
        host="www.szse.cn",
        path="/api/report/ShowReport",
    )

    disclosures = sources.get("disclosures", {})
    _require_keys(
        disclosures,
        {"query_url", "download_base_url", "stock_map_url", "download_workers"},
        "data_sources.disclosures",
    )
    _validate_official_source_url(
        disclosures["query_url"],
        label="data_sources.disclosures.query_url",
        host="www.cninfo.com.cn",
        path="/new/hisAnnouncement/query",
    )
    _validate_official_source_url(
        disclosures["download_base_url"],
        label="data_sources.disclosures.download_base_url",
        host="static.cninfo.com.cn",
        path="/",
    )
    _validate_official_source_url(
        disclosures["stock_map_url"],
        label="data_sources.disclosures.stock_map_url",
        host="www.cninfo.com.cn",
        path="/new/data/szse_stock.json",
    )
    download_workers = disclosures.get("download_workers")
    if (
        isinstance(download_workers, bool)
        or not isinstance(download_workers, int)
        or not 1 <= download_workers <= 8
    ):
        raise ConfigValidationError(
            "data_sources.disclosures.download_workers must be an integer from 1 to 8"
        )

    timing = config["information_timing"]
    _require_keys(
        timing,
        {
            "signal_time",
            "signal_cutoff_local",
            "timezone",
            "publication_timestamp_required",
            "backdate_to_fiscal_period_allowed",
            "edge_max_age_days",
        },
        "information_timing",
    )
    if (
        timing["publication_timestamp_required"] is not True
        or timing["backdate_to_fiscal_period_allowed"] is not False
    ):
        raise ConfigValidationError(
            "Disclosure publication timing protections may not be disabled"
        )
    if (
        timing["signal_time"] != "close"
        or timing["signal_cutoff_local"] != "15:00:00"
        or timing["timezone"] != "Asia/Shanghai"
    ):
        raise ConfigValidationError(
            "V2 information timing must remain close at 15:00:00 Asia/Shanghai"
        )
    if timing["edge_max_age_days"] != 550:
        raise ConfigValidationError(
            "V2 baseline edge_max_age_days must be 550 before validation"
        )

    if config["edge_weighting"].get("primary") != "equal":
        raise ConfigValidationError("The V2 primary edge weighting must be equal")
    signals = config["signals"]
    if signals.get("primary_direction") != "customer_to_supplier":
        raise ConfigValidationError(
            "The frozen primary direction is customer_to_supplier"
        )
    if signals.get("forward_horizons_trading_days") != [1, 3, 5, 10]:
        raise ConfigValidationError("The frozen signal horizons must be [1, 3, 5, 10]")

    gate = config["graph_coverage_gate"]
    frozen_gate = {
        "min_consecutive_calendar_years": 3,
        "min_year_end_active_directed_edges": 200,
        "min_year_end_connected_stocks": 150,
        "min_median_connected_stocks_across_usable_years": 200,
        "max_timing_violations": 0,
        "max_anonymous_resolved_nodes": 0,
        "max_duplicate_pair_date_exposures": 0,
        "max_missing_source_provenance": 0,
    }
    for key, expected in frozen_gate.items():
        if gate.get(key) != expected:
            raise ConfigValidationError(
                f"graph_coverage_gate.{key} must equal frozen value {expected}"
            )

    states = config["engineering"].get("allowed_phase_states")
    if set(states or []) != {"PASS", "FAIL", "BLOCKED", "SKIPPED_BY_DESIGN"}:
        raise ConfigValidationError("engineering.allowed_phase_states is invalid")
    return config


def load_config(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping: {config_path}")
    return validate_project_config(config) if validate else config
