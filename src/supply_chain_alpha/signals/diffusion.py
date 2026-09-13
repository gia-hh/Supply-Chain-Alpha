"""Leakage-safe one-hop supply-chain diffusion signals."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from supply_chain_alpha.data.schemas import SIGNAL_DAILY, validate_table

_EDGE_COLUMNS = {"supplier_id", "customer_id", "effective_start", "effective_end"}
_RETURN_COLUMNS = {"date", "security_id", "residual_return"}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def build_diffusion_signals(
    edges: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build equal-weight customer and supplier shocks at each local close.

    Edge convention is ``supplier_id -> customer_id``.  Each neighbor appears
    at most once in a snapshot even when several source documents support the
    same relationship. Missing neighbor residuals are excluded, never filled.
    ``effective_end`` is treated as an exclusive bound.
    """

    if not isinstance(edges, pd.DataFrame) or not isinstance(residuals, pd.DataFrame):
        raise TypeError("edges and residuals must be pandas DataFrames")
    _require_columns(edges, _EDGE_COLUMNS, "edges")
    _require_columns(residuals, _RETURN_COLUMNS, "residuals")
    if edges[["supplier_id", "customer_id"]].isna().any().any():
        raise ValueError("edge endpoints must be non-null")
    if edges["supplier_id"].astype(str).eq(edges["customer_id"].astype(str)).any():
        raise ValueError("self-neighbor edges are not permitted")

    returns = residuals.loc[:, ["date", "security_id", "residual_return"]].copy()
    parsed_dates = pd.to_datetime(returns["date"], errors="raise").dt.date
    returns["date"] = parsed_dates.astype(str)
    if returns.duplicated(["date", "security_id"]).any():
        raise ValueError("residuals contain duplicate date/security_id rows")
    returns["residual_return"] = pd.to_numeric(
        returns["residual_return"], errors="coerce"
    )

    if dates is None:
        signal_dates = sorted(returns["date"].unique().tolist())
    else:
        signal_dates = sorted(
            {pd.Timestamp(value).date().isoformat() for value in dates}
        )

    edge_frame = edges.loc[:, sorted(_EDGE_COLUMNS)].copy()
    edge_frame["effective_start"] = pd.to_datetime(
        edge_frame["effective_start"], errors="raise", utc=True
    )
    edge_frame["effective_end"] = pd.to_datetime(
        edge_frame["effective_end"], errors="raise", utc=True
    )

    output_rows: list[dict[str, object]] = []
    for day in signal_dates:
        close = pd.Timestamp(f"{day}T15:00:00", tz="Asia/Shanghai").tz_convert("UTC")
        active = edge_frame.loc[
            edge_frame["effective_start"].le(close)
            & (
                edge_frame["effective_end"].isna()
                | edge_frame["effective_end"].gt(close)
            ),
            ["supplier_id", "customer_id"],
        ].drop_duplicates()
        if active.empty:
            continue

        day_returns = returns.loc[
            returns["date"].eq(day), ["security_id", "residual_return"]
        ]
        return_by_security = day_returns.set_index("security_id")["residual_return"]
        targets = sorted(
            set(active["supplier_id"].astype(str))
            | set(active["customer_id"].astype(str))
        )
        for target in targets:
            customer_ids = active.loc[
                active["supplier_id"].astype(str).eq(target), "customer_id"
            ].astype(str)
            supplier_ids = active.loc[
                active["customer_id"].astype(str).eq(target), "supplier_id"
            ].astype(str)
            customer_values = pd.to_numeric(
                return_by_security.reindex(customer_ids).dropna(), errors="raise"
            )
            supplier_values = pd.to_numeric(
                return_by_security.reindex(supplier_ids).dropna(), errors="raise"
            )
            output_rows.append(
                {
                    "date": day,
                    "security_id": target,
                    "customer_shock": (
                        float(customer_values.mean())
                        if not customer_values.empty
                        else None
                    ),
                    "supplier_shock": (
                        float(supplier_values.mean())
                        if not supplier_values.empty
                        else None
                    ),
                    "customer_neighbor_count": len(customer_values),
                    "supplier_neighbor_count": len(supplier_values),
                    "graph_snapshot_date": day,
                }
            )

    result = pd.DataFrame(output_rows, columns=SIGNAL_DAILY.required_columns)
    if not result.empty:
        result = result.sort_values(
            ["date", "security_id"], kind="mergesort"
        ).reset_index(drop=True)
        validate_table(result, SIGNAL_DAILY)
    return result


__all__ = ["build_diffusion_signals"]
