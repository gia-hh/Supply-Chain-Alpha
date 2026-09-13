from pathlib import Path

import pandas as pd
import pytest

from supply_chain_alpha.data.disclosures import load_structured_mentions
from supply_chain_alpha.entities.resolve import (
    attach_resolution_audit,
    resolve_mentions,
)
from supply_chain_alpha.graph.coverage import listed_edge_intervals, summarize_coverage
from supply_chain_alpha.graph.edges import build_point_in_time_edges
from supply_chain_alpha.graph.snapshots import graph_snapshot

FIXTURES = Path(__file__).parents[2] / "fixtures"


def _edges():
    mentions = load_structured_mentions(FIXTURES / "disclosure_mentions.csv")
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    audit = resolve_mentions(mentions, aliases)
    return build_point_in_time_edges(mentions, audit, max_age_days=550)


def test_edge_direction_customer_disclosure():
    edges = _edges()
    row = edges.loc[edges["source_document_id"].eq("D1")].iloc[0]
    assert row["supplier_id"] == "S1"
    assert row["customer_id"] == "S2"


def test_edge_direction_supplier_disclosure():
    edges = _edges()
    row = edges.loc[
        (edges["source_document_id"].eq("D2")) & (edges["supplier_id"].eq("S4"))
    ].iloc[0]
    assert row["supplier_id"] == "S4"
    assert row["customer_id"] == "S2"


def test_graph_never_backdates_publication():
    edges = _edges()
    d1 = edges.loc[edges["source_document_id"].eq("D1")]
    assert graph_snapshot(d1, "2020-04-15 17:59:59").empty
    assert len(graph_snapshot(d1, "2020-04-15 18:00:00")) == 1
    assert (
        pd.to_datetime(edges["effective_start"])
        >= pd.to_datetime(edges["publication_datetime"])
    ).all()


def test_duplicate_evidence_collapses_to_one_economic_edge():
    edges = _edges()
    d1 = edges.loc[edges["source_document_id"].eq("D1")].copy()
    duplicate = d1.copy()
    duplicate["source_document_id"] = "D1B"
    duplicate["effective_start"] = pd.Timestamp("2020-05-01")
    duplicate["publication_datetime"] = pd.Timestamp("2020-05-01")
    combined = pd.concat([d1, duplicate], ignore_index=True)
    snap = graph_snapshot(combined, "2020-05-02")
    assert len(snap) == 1
    assert snap.iloc[0]["source_document_id"] == "D1B"


def test_listing_interval_intersection_uses_both_endpoints() -> None:
    d1 = _edges().loc[lambda frame: frame["source_document_id"].eq("D1")].copy()
    master = pd.read_csv(
        FIXTURES / "security_master.csv", dtype={"security_id": str, "ticker": str}
    )
    master[["listing_date", "delisting_date"]] = master[
        ["listing_date", "delisting_date"]
    ].astype("string")
    master.loc[master["security_id"].eq("S1"), ["listing_date", "delisting_date"]] = [
        "2020-04-16",
        "2020-06-15",
    ]
    master.loc[master["security_id"].eq("S2"), ["listing_date", "delisting_date"]] = [
        "2020-04-20",
        "2020-05-31",
    ]

    listed = listed_edge_intervals(d1, master)

    assert len(listed) == 1
    assert listed.iloc[0]["usable_start"] == pd.Timestamp(
        "2020-04-20", tz="Asia/Shanghai"
    ).tz_convert("UTC")
    assert listed.iloc[0]["usable_end"] == pd.Timestamp(
        "2020-05-31", tz="Asia/Shanghai"
    ).tz_convert("UTC")


def test_self_edge_is_filtered_from_point_in_time_graph() -> None:
    mentions = (
        load_structured_mentions(FIXTURES / "disclosure_mentions.csv").iloc[[0]].copy()
    )
    mentions.loc[:, "source_company_id"] = "S2"
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    audit = resolve_mentions(mentions, aliases)

    edges = build_point_in_time_edges(mentions, audit)

    assert audit.iloc[0]["resolved_entity_id"] == "S2"
    assert edges.empty


def test_edge_expiry_is_a_half_open_effective_end_boundary() -> None:
    d1 = _edges().loc[lambda frame: frame["source_document_id"].eq("D1")].copy()
    effective_start = d1.iloc[0]["effective_start"]
    effective_end = d1.iloc[0]["effective_end"]

    assert effective_end - effective_start == pd.Timedelta(days=550)
    assert len(graph_snapshot(d1, effective_end - pd.Timedelta(nanoseconds=1))) == 1
    assert graph_snapshot(d1, effective_end).empty


def test_cross_relation_mentions_attach_to_one_document_name_audit_row():
    mentions = load_structured_mentions(FIXTURES / "disclosure_mentions.csv")
    repeated = mentions.loc[mentions["source_document_id"].eq("D1")].copy()
    repeated["relationship_type"] = "supplier"
    mentions = pd.concat([mentions, repeated], ignore_index=True)
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    audit = resolve_mentions(mentions, aliases)

    attached = attach_resolution_audit(mentions, audit)
    edges = build_point_in_time_edges(mentions, audit)

    assert len(attached.loc[attached["source_document_id"].eq("D1")]) == 2
    assert len(audit.loc[audit["source_document_id"].eq("D1")]) == 1
    assert set(
        edges.loc[
            edges["source_document_id"].eq("D1"), ["supplier_id", "customer_id"]
        ].itertuples(index=False, name=None)
    ) == {("S1", "S2"), ("S2", "S1")}


def test_every_mention_requires_a_matching_resolution_audit_row():
    mentions = load_structured_mentions(FIXTURES / "disclosure_mentions.csv")
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    audit = resolve_mentions(mentions, aliases).iloc[:-1].copy()

    with pytest.raises(ValueError, match="Every disclosure mention must match"):
        build_point_in_time_edges(mentions, audit)


def test_same_pair_and_document_aliases_collapse_deterministically():
    mentions = load_structured_mentions(FIXTURES / "disclosure_mentions.csv")
    alias_mention = mentions.loc[mentions["source_document_id"].eq("D1")].copy()
    alias_mention["counterparty_raw_name"] = "华东电池"
    mentions = pd.concat([mentions, alias_mention], ignore_index=True)
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    audit = resolve_mentions(mentions, aliases)

    forward = build_point_in_time_edges(mentions, audit)
    reversed_rows = build_point_in_time_edges(
        mentions.iloc[::-1].reset_index(drop=True),
        audit.iloc[::-1].reset_index(drop=True),
    )

    assert len(forward.loc[forward["source_document_id"].eq("D1")]) == 1
    assert forward.loc[
        forward["source_document_id"].eq("D1"), "weight_source"
    ].item() == ("equal")
    pd.testing.assert_frame_equal(forward, reversed_rows)


def test_resolution_coverage_is_counted_at_mention_level():
    mentions = load_structured_mentions(FIXTURES / "disclosure_mentions.csv")
    repeated = mentions.loc[mentions["source_document_id"].eq("D1")].copy()
    repeated["relationship_type"] = "supplier"
    mentions = pd.concat([mentions, repeated], ignore_index=True)
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    master = pd.read_csv(
        FIXTURES / "security_master.csv", dtype={"security_id": str, "ticker": str}
    )
    audit = resolve_mentions(mentions, aliases)
    edges = build_point_in_time_edges(mentions, audit)

    summary = summarize_coverage(mentions, audit, edges, master)

    assert summary.explicit_name_mentions == 5
    assert summary.resolved_mentions == 4
    assert summary.resolution_rate_given_explicit == 0.8


def test_only_named_counterparties_count_as_explicit() -> None:
    mentions = load_structured_mentions(FIXTURES / "disclosure_mentions.csv")
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    master = pd.read_csv(
        FIXTURES / "security_master.csv", dtype={"security_id": str, "ticker": str}
    )
    audit = resolve_mentions(mentions, aliases)
    edges = build_point_in_time_edges(mentions, audit)
    mentions.loc[mentions["source_document_id"].eq("D4"), "counterparty_raw_name"] = (
        "不适用"
    )
    audit.loc[audit["source_document_id"].eq("D4"), "counterparty_raw_name"] = "不适用"

    summary = summarize_coverage(mentions, audit, edges, master)

    assert summary.mentions == 5
    assert summary.explicit_name_mentions == 3
    assert summary.explicit_name_rate == 0.6


def test_naive_snapshot_cutoff_is_interpreted_as_shanghai_market_time():
    edges = _edges()
    d1 = edges.loc[edges["source_document_id"].eq("D1")]

    assert d1["effective_start"].dt.tz is not None
    assert graph_snapshot(d1, "2020-04-15 17:59:59").empty
    assert len(graph_snapshot(d1, "2020-04-15 18:00:00")) == 1
