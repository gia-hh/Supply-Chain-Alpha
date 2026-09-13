from pathlib import Path

import pandas as pd

from supply_chain_alpha.data.disclosures import load_structured_mentions
from supply_chain_alpha.data.schemas import SECURITY_MASTER, validate_table
from supply_chain_alpha.entities.resolve import resolve_mentions
from supply_chain_alpha.graph.coverage import degree_diagnostics, summarize_coverage
from supply_chain_alpha.graph.edges import build_point_in_time_edges

FIXTURES = Path(__file__).parents[2] / "fixtures"


def test_phase1_3_end_to_end():
    mentions = load_structured_mentions(FIXTURES / "disclosure_mentions.csv")
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    master = pd.read_csv(
        FIXTURES / "security_master.csv", dtype={"security_id": str, "ticker": str}
    )
    validate_table(master, SECURITY_MASTER)
    audit = resolve_mentions(mentions, aliases)
    edges = build_point_in_time_edges(mentions, audit)
    summary = summarize_coverage(mentions, audit, edges, master)
    degree = degree_diagnostics(edges)

    assert summary.mentions == 5
    assert summary.explicit_name_mentions == 4
    assert summary.resolved_mentions == 3
    assert summary.listed_to_listed_edges == 3
    assert summary.unique_connected_stocks == 4
    assert not degree.empty
