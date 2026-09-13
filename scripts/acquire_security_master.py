from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import requests

from supply_chain_alpha.data.acquisition import (
    RateLimitedSession,
    RawAssetStore,
    RequestPolicy,
    assert_security_master_qa,
)
from supply_chain_alpha.data.exchange_acquisition import (
    acquire_official_security_master,
    company_alias_diagnostics,
)
from supply_chain_alpha.data.io import write_canonical_parquet
from supply_chain_alpha.utils.config import load_config

LOGGER = logging.getLogger("acquire_security_master")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire official SSE/SZSE security master and PIT-safe aliases."
    )
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alias-output", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_config(args.config)
    paths = config["paths"]
    sources = config["data_sources"]
    source_config = sources["security_master"]
    raw_dir = args.raw_dir or Path(paths["raw_dir"])
    output = args.output or Path(paths["security_master"])
    alias_output = args.alias_output or Path(paths["company_alias"])

    transport = RateLimitedSession(
        RequestPolicy.from_mapping(sources.get("request_policy"))
    )
    try:
        result = acquire_official_security_master(
            transport,
            RawAssetStore(raw_dir),
            sse_url=source_config["sse_url"],
            szse_url=source_config["szse_url"],
        )
    except requests.RequestException as exc:
        LOGGER.error("Official security-master source unavailable: %s", exc)
        return 2

    diagnostics = assert_security_master_qa(result.securities)
    alias_diagnostics = company_alias_diagnostics(result.aliases)
    write_canonical_parquet(result.securities, output)
    write_canonical_parquet(result.aliases, alias_output)

    LOGGER.info("security_master rows=%d", len(result.securities))
    LOGGER.info("security_master missing=%s", result.securities.isna().sum().to_dict())
    LOGGER.info(
        "security_master listing_date_range=%s..%s sources=%s",
        diagnostics["listing_date_min"],
        diagnostics["listing_date_max"],
        diagnostics["sources"],
    )
    LOGGER.info("security_master QA=%s", json.dumps(diagnostics, ensure_ascii=False))
    LOGGER.info(
        "company_alias QA=%s", json.dumps(alias_diagnostics, ensure_ascii=False)
    )
    LOGGER.info(
        "raw_assets=%s",
        json.dumps([asset.to_dict() for asset in result.assets], ensure_ascii=False),
    )
    for limitation in result.limitations:
        LOGGER.warning("PIT limitation: %s", limitation)
    LOGGER.info("wrote security_master=%s company_alias=%s", output, alias_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
