from __future__ import annotations

import argparse
import logging
from pathlib import Path

from supply_chain_alpha.data.disclosures import load_structured_mentions
from supply_chain_alpha.data.io import read_table, write_canonical_parquet
from supply_chain_alpha.data.schemas import COMPANY_ALIAS, validate_table
from supply_chain_alpha.entities.resolve import ResolutionConfig, resolve_mentions
from supply_chain_alpha.graph.edges import build_point_in_time_edges
from supply_chain_alpha.utils.config import load_config

LOGGER = logging.getLogger("build_supply_chain_graph")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve disclosure counterparties and build PIT edges."
    )
    parser.add_argument("mentions", nargs="?", type=Path)
    parser.add_argument("aliases", nargs="?", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--edge-output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    config = load_config(args.config)
    paths = config["paths"]
    mention_path = args.mentions or Path(paths["disclosure_raw"])
    alias_path = args.aliases or Path(paths["company_alias"])
    audit_output = args.audit_output or Path(paths["resolution_audit"])
    edge_output = args.edge_output or Path(paths["supply_chain_edge"])
    override_path = args.overrides or Path(
        config["entity_resolution"]["manual_override_file"]
    )

    mentions = load_structured_mentions(mention_path)
    aliases = validate_table(read_table(alias_path), COMPANY_ALIAS)
    resolution_config = ResolutionConfig(
        fuzzy_enabled=bool(config["entity_resolution"]["fuzzy_matching_enabled"]),
        fuzzy_min_score=float(config["entity_resolution"]["fuzzy_min_score"]),
        fuzzy_min_margin=float(config["entity_resolution"]["fuzzy_min_margin"]),
    )
    audit = resolve_mentions(
        mentions,
        aliases,
        config=resolution_config,
        override_path=override_path,
    )
    edges = build_point_in_time_edges(
        mentions,
        audit,
        max_age_days=int(config["information_timing"]["edge_max_age_days"]),
    )
    write_canonical_parquet(audit, audit_output)
    write_canonical_parquet(edges, edge_output)

    LOGGER.info("source=canonical_disclosure_mentions input_rows=%d", len(mentions))
    LOGGER.info(
        "alias_rows=%d audit_rows=%d edge_evidence_rows=%d",
        len(aliases),
        len(audit),
        len(edges),
    )
    LOGGER.info(
        "resolved_rows=%d anonymous_rows=%d",
        audit["resolution_status"].eq("resolved").sum(),
        audit["resolution_status"].eq("anonymous").sum(),
    )
    LOGGER.info(
        "duplicate_edge_evidence_pk=%d",
        edges.duplicated(["supplier_id", "customer_id", "source_document_id"]).sum(),
    )
    LOGGER.info("audit_missing=%s", audit.isna().sum().to_dict())
    if not edges.empty:
        LOGGER.info(
            "edge_effective_range=%s..%s",
            edges["effective_start"].min(),
            edges["effective_end"].max(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
