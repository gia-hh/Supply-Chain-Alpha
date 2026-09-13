from __future__ import annotations

import pandas as pd

from supply_chain_alpha.data.schemas import (
    DISCLOSURE_RAW,
    SUPPLY_CHAIN_EDGE,
    validate_table,
)
from supply_chain_alpha.entities.resolve import attach_resolution_audit

from .time import market_instants_utc

_PAIR_DOCUMENT_KEY = ("supplier_id", "customer_id", "source_document_id")
_PAIR_DOCUMENT_INVARIANTS = (
    "effective_start",
    "effective_end",
    "publication_datetime",
    "source_period_end",
    "source_company_id",
)


def _collapse_pair_document_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Collapse aliases in one document to one deterministic evidence edge."""
    if edges.empty:
        return edges.copy()
    grouped = edges.groupby(list(_PAIR_DOCUMENT_KEY), sort=False, dropna=False)
    for column in _PAIR_DOCUMENT_INVARIANTS:
        conflicts = grouped[column].nunique(dropna=False).gt(1)
        if conflicts.any():
            sample = conflicts.loc[conflicts].index.tolist()[:5]
            raise ValueError(
                f"Conflicting {column} for repeated pair/document edges: {sample}"
            )

    ordered = edges.sort_values(
        [*_PAIR_DOCUMENT_KEY, "relationship_confidence"],
        kind="stable",
        na_position="first",
    )
    collapsed = (
        ordered.groupby(list(_PAIR_DOCUMENT_KEY), as_index=False, sort=True)
        .agg(
            effective_start=("effective_start", "first"),
            effective_end=("effective_end", "first"),
            source_period_end=("source_period_end", "first"),
            source_company_id=("source_company_id", "first"),
            relationship_confidence=("relationship_confidence", "max"),
            publication_datetime=("publication_datetime", "first"),
        )
        .reset_index(drop=True)
    )
    # V2 freezes equal edge weighting as the primary graph.  Raw exposure
    # values/shares remain in disclosure_raw for explicitly labelled robustness.
    collapsed["economic_weight"] = 1.0
    collapsed["weight_source"] = "equal"
    return collapsed[list(SUPPLY_CHAIN_EDGE.required_columns)]


def build_point_in_time_edges(
    mentions: pd.DataFrame,
    resolution_audit: pd.DataFrame,
    *,
    max_age_days: int = 550,
) -> pd.DataFrame:
    validate_table(mentions, DISCLOSURE_RAW)
    if isinstance(max_age_days, bool):
        raise TypeError("max_age_days must be a positive integer")
    try:
        parsed_max_age_days = int(max_age_days)
    except (TypeError, ValueError) as exc:
        raise TypeError("max_age_days must be a positive integer") from exc
    if parsed_max_age_days != max_age_days or parsed_max_age_days <= 0:
        raise ValueError("max_age_days must be a positive integer")
    max_age_days = parsed_max_age_days

    merged = attach_resolution_audit(mentions, resolution_audit)
    merged = merged.loc[
        merged["resolution_status"].eq("resolved")
        & merged["resolved_entity_id"].notna()
        & merged["resolved_entity_id"].astype("string").str.strip().ne("")
    ].copy()
    if merged.empty:
        return pd.DataFrame(columns=SUPPLY_CHAIN_EDGE.required_columns)

    is_customer = merged["relationship_type"].eq("customer")
    merged["supplier_id"] = merged["source_company_id"].where(
        is_customer, merged["resolved_entity_id"]
    )
    merged["customer_id"] = merged["resolved_entity_id"].where(
        is_customer, merged["source_company_id"]
    )
    merged["supplier_id"] = merged["supplier_id"].astype(str)
    merged["customer_id"] = merged["customer_id"].astype(str)
    merged["publication_datetime"] = market_instants_utc(merged["publication_datetime"])
    merged["effective_start"] = merged["publication_datetime"]
    merged["effective_end"] = merged["effective_start"] + pd.to_timedelta(
        max_age_days, unit="D"
    )
    merged["relationship_confidence"] = merged["resolution_confidence"].astype(float)
    merged["economic_weight"] = 1.0
    merged["weight_source"] = "equal"

    edge_cols = list(SUPPLY_CHAIN_EDGE.required_columns)
    edges = merged[edge_cols].copy()
    edges = edges.loc[
        edges["supplier_id"].astype(str) != edges["customer_id"].astype(str)
    ].copy()
    edges = _collapse_pair_document_edges(edges)
    if (edges["effective_start"] < edges["publication_datetime"]).any():
        raise AssertionError(
            "PIT violation: effective_start precedes publication_datetime"
        )
    return validate_table(edges, SUPPLY_CHAIN_EDGE)
