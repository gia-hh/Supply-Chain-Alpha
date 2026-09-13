from __future__ import annotations

import pandas as pd

from supply_chain_alpha.data.schemas import SUPPLY_CHAIN_EDGE, validate_table

from .time import market_instants_utc, market_timestamp_utc


def graph_snapshot(
    edges: pd.DataFrame,
    as_of: str | pd.Timestamp,
    *,
    collapse_economic_edges: bool = True,
) -> pd.DataFrame:
    """Return edges public and active by the requested timestamp.

    Multiple source documents can support the same economic edge. By default the
    snapshot keeps only the most recent active evidence row per directed pair,
    preventing duplicated evidence from becoming duplicated graph exposure.
    """
    if edges.empty:
        return edges.copy()
    validate_table(edges, SUPPLY_CHAIN_EDGE)
    ts = market_timestamp_utc(as_of)
    start = market_instants_utc(edges["effective_start"])
    end = market_instants_utc(edges["effective_end"])
    mask = (start <= ts) & (ts < end)
    out = edges.loc[mask].copy()
    out["effective_start"] = start.loc[mask]
    out["effective_end"] = end.loc[mask]
    out["publication_datetime"] = market_instants_utc(out["publication_datetime"])
    if out.empty or not collapse_economic_edges:
        return out.reset_index(drop=True)
    sort_columns = [
        "supplier_id",
        "customer_id",
        "effective_start",
        "relationship_confidence",
        "source_document_id",
    ]
    out = out.sort_values(
        sort_columns,
        ascending=[True] * len(sort_columns),
        kind="stable",
        na_position="first",
    )
    out = out.drop_duplicates(["supplier_id", "customer_id"], keep="last")
    return out.reset_index(drop=True)
