from __future__ import annotations

import argparse
import logging
from pathlib import Path

from supply_chain_alpha.data.disclosures import (
    load_structured_mentions,
    write_raw_manifest,
)
from supply_chain_alpha.data.io import write_canonical_parquet
from supply_chain_alpha.utils.config import load_config

LOGGER = logging.getLogger("ingest_disclosures")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a source-grounded structured mention file."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-identifier", required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    config = load_config(args.config)
    output = args.output or Path(config["paths"]["disclosure_raw"])
    manifest = args.manifest or Path(config["paths"]["raw_manifest"])
    mentions = load_structured_mentions(args.input)
    LOGGER.info("source=%s identifier=%s", args.source, args.source_identifier)
    LOGGER.info("input_rows=%d output_rows=%d", len(mentions), len(mentions))
    LOGGER.info(
        "duplicate_primary_keys=%d",
        mentions.duplicated(
            [
                "source_document_id",
                "source_company_id",
                "relationship_type",
                "counterparty_raw_name",
            ]
        ).sum(),
    )
    LOGGER.info("required_missing=%s", mentions.isna().sum().to_dict())
    LOGGER.info(
        "publication_range=%s..%s",
        mentions["publication_datetime"].min(),
        mentions["publication_datetime"].max(),
    )
    write_canonical_parquet(mentions, output)
    write_raw_manifest(
        [args.input],
        manifest,
        source=args.source,
        source_url_or_identifier=args.source_identifier,
    )
    LOGGER.info("wrote=%s manifest=%s", output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
