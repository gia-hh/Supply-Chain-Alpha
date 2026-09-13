from pathlib import Path

import pandas as pd

from supply_chain_alpha.data.disclosures import load_structured_mentions
from supply_chain_alpha.entities.resolve import resolve_mentions
from supply_chain_alpha.graph.edges import build_point_in_time_edges
from supply_chain_alpha.graph.snapshots import graph_snapshot
from supply_chain_alpha.graph.time import market_instants_utc, market_timestamp_utc

FIXTURES = Path(__file__).parents[2] / "fixtures"


def test_future_disclosure_does_not_change_old_snapshot():
    base = load_structured_mentions(FIXTURES / "disclosure_mentions.csv")
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    cutoff = market_timestamp_utc("2020-04-20 23:59:59+08:00")
    old = base.loc[market_instants_utc(base["publication_datetime"]) <= cutoff].copy()
    old_edges = build_point_in_time_edges(old, resolve_mentions(old, aliases))
    old_snapshot = (
        graph_snapshot(old_edges, cutoff)[["supplier_id", "customer_id"]]
        .sort_values(["supplier_id", "customer_id"])
        .reset_index(drop=True)
    )

    all_edges = build_point_in_time_edges(base, resolve_mentions(base, aliases))
    all_snapshot = (
        graph_snapshot(all_edges, cutoff)[["supplier_id", "customer_id"]]
        .sort_values(["supplier_id", "customer_id"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(old_snapshot, all_snapshot)
