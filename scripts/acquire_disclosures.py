from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import requests

from supply_chain_alpha.data.acquisition import (
    AcquisitionMode,
    RateLimitedSession,
    RawAssetStore,
    RequestPolicy,
)
from supply_chain_alpha.data.cninfo import (
    acquire_cninfo_disclosures,
    disclosure_acquisition_diagnostics,
)
from supply_chain_alpha.data.io import write_canonical_parquet
from supply_chain_alpha.utils.config import load_config

LOGGER = logging.getLogger("acquire_disclosures")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire CNINFO annual-report metadata and immutable documents."
    )
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/cninfo_annual_report_documents.parquet"),
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_config(args.config)
    sources = config["data_sources"]
    disclosure_config = sources["disclosures"]
    raw_dir = args.raw_dir or Path(config["paths"]["raw_dir"])
    start_date = args.start_date or str(config["periods"]["raw_start"])
    end_date = args.end_date or str(config["periods"]["validation_end"])
    if end_date > str(config["periods"]["validation_end"]):
        LOGGER.error(
            "Phase-2 disclosure acquisition may not access the frozen holdout: %s",
            end_date,
        )
        return 4
    mode = AcquisitionMode(
        metadata_only=args.metadata_only,
        max_documents=args.max_documents,
    )
    if not mode.production_qa_eligible:
        LOGGER.warning(
            "LIMITED_TRIAL mode=%s max_documents=%s; this run is not production-QA eligible",
            mode.label,
            mode.max_documents,
        )

    transport = RateLimitedSession(
        RequestPolicy.from_mapping(sources.get("request_policy"))
    )
    try:
        result = acquire_cninfo_disclosures(
            transport,
            RawAssetStore(raw_dir),
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            query_url=disclosure_config["query_url"],
            download_base_url=disclosure_config["download_base_url"],
            category=disclosure_config["category"],
            page_size=args.page_size,
            download_workers=int(disclosure_config.get("download_workers", 1)),
        )
    except requests.RequestException as exc:
        LOGGER.error("CNINFO source unavailable: %s", exc)
        return 2

    diagnostics = disclosure_acquisition_diagnostics(result)
    write_canonical_parquet(result.documents, args.output)
    LOGGER.info(
        "documents rows=%d unique_ids=%d",
        len(result.documents),
        diagnostics["unique_document_ids"],
    )
    LOGGER.info("documents missing=%s", result.documents.isna().sum().to_dict())
    LOGGER.info(
        "publication_datetime_range=%s..%s sources=%s",
        diagnostics["publication_datetime_min"],
        diagnostics["publication_datetime_max"],
        diagnostics["sources"],
    )
    LOGGER.info("acquisition QA=%s", json.dumps(diagnostics, ensure_ascii=False))
    LOGGER.info(
        "raw_assets=%s",
        json.dumps([asset.to_dict() for asset in result.assets], ensure_ascii=False),
    )
    if result.failure_log_asset is not None:
        LOGGER.error("failed_download_log=%s", result.failure_log_asset.local_path)
    LOGGER.info("wrote metadata=%s", args.output)
    if diagnostics["qa_state"] == "FAIL":
        return 2 if result.failures else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
