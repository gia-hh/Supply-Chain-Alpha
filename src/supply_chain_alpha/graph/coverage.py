from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from supply_chain_alpha.data.schemas import (
    DISCLOSURE_RAW,
    RESOLUTION_AUDIT,
    SECURITY_MASTER,
    SUPPLY_CHAIN_EDGE,
    validate_table,
)
from supply_chain_alpha.entities.normalize import (
    CounterpartyNameClass,
    classify_counterparty_name,
    normalize_company_name,
)
from supply_chain_alpha.entities.resolve import attach_resolution_audit

from .snapshots import graph_snapshot
from .time import (
    MARKET_TIMEZONE,
    market_calendar_year,
    market_dates_utc,
    market_instants_utc,
    market_timestamp_utc,
    market_year_end_utc,
    market_year_start_utc,
)

PHASE3_DEVELOPMENT_START_YEAR = 2016
PHASE3_VALIDATION_END_YEAR = 2022
PHASE3_MIN_CONSECUTIVE_YEARS = 3
PHASE3_MIN_YEAR_END_ACTIVE_EDGES = 200
PHASE3_MIN_YEAR_END_CONNECTED_STOCKS = 150
PHASE3_MIN_MEDIAN_CONNECTED_STOCKS = 200
PHASE3_MIN_INDUSTRIES = 5
PHASE3_MAX_INDUSTRY_SHARE = 0.60

_PHASE3_GATE_THRESHOLD_KEYS = (
    "evaluation_start",
    "evaluation_end",
    "min_consecutive_calendar_years",
    "min_year_end_active_directed_edges",
    "min_year_end_connected_stocks",
    "min_median_connected_stocks_across_usable_years",
    "min_industries_if_pit_available",
    "max_largest_industry_edge_share_if_pit_available",
    "max_timing_violations",
    "max_anonymous_resolved_nodes",
    "max_duplicate_pair_date_exposures",
    "max_missing_source_provenance",
)

_DEFAULT_PHASE3_FROZEN_THRESHOLDS: dict[str, Any] = {
    "evaluation_start": f"{PHASE3_DEVELOPMENT_START_YEAR}-01-01",
    "evaluation_end": f"{PHASE3_VALIDATION_END_YEAR}-12-31",
    "min_consecutive_calendar_years": PHASE3_MIN_CONSECUTIVE_YEARS,
    "min_year_end_active_directed_edges": PHASE3_MIN_YEAR_END_ACTIVE_EDGES,
    "min_year_end_connected_stocks": PHASE3_MIN_YEAR_END_CONNECTED_STOCKS,
    "min_median_connected_stocks_across_usable_years": (
        PHASE3_MIN_MEDIAN_CONNECTED_STOCKS
    ),
    "min_industries_if_pit_available": PHASE3_MIN_INDUSTRIES,
    "max_largest_industry_edge_share_if_pit_available": (PHASE3_MAX_INDUSTRY_SHARE),
    "max_timing_violations": 0,
    "max_anonymous_resolved_nodes": 0,
    "max_duplicate_pair_date_exposures": 0,
    "max_missing_source_provenance": 0,
}


def _integer_threshold(
    thresholds: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
) -> int:
    value = thresholds[key]
    if isinstance(value, bool):
        raise TypeError(f"graph_coverage_gate.{key} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"graph_coverage_gate.{key} must be an integer") from exc
    if parsed != value or parsed < minimum:
        raise ValueError(f"graph_coverage_gate.{key} must be an integer >= {minimum}")
    return parsed


def _resolved_phase3_thresholds(
    frozen_thresholds: Mapping[str, Any] | None,
    *,
    development_start_year: int | None,
    validation_end_year: int | None,
) -> tuple[dict[str, Any], int, int]:
    explicit = frozen_thresholds is not None
    raw = dict(
        _DEFAULT_PHASE3_FROZEN_THRESHOLDS
        if frozen_thresholds is None
        else frozen_thresholds
    )
    missing = sorted(set(_PHASE3_GATE_THRESHOLD_KEYS) - set(raw))
    if missing:
        raise ValueError(f"graph_coverage_gate missing frozen thresholds: {missing}")

    try:
        evaluation_start = pd.Timestamp(raw["evaluation_start"])
        evaluation_end = pd.Timestamp(raw["evaluation_end"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "graph_coverage_gate evaluation_start/evaluation_end must be dates"
        ) from exc
    if evaluation_start.tzinfo is not None:
        evaluation_start = evaluation_start.tz_convert(MARKET_TIMEZONE).tz_localize(
            None
        )
    if evaluation_end.tzinfo is not None:
        evaluation_end = evaluation_end.tz_convert(MARKET_TIMEZONE).tz_localize(None)
    evaluation_start = evaluation_start.normalize()
    evaluation_end = evaluation_end.normalize()
    if (evaluation_start.month, evaluation_start.day) != (1, 1):
        raise ValueError("graph_coverage_gate.evaluation_start must be January 1")
    if (evaluation_end.month, evaluation_end.day) != (12, 31):
        raise ValueError("graph_coverage_gate.evaluation_end must be December 31")
    if evaluation_start > evaluation_end:
        raise ValueError(
            "graph_coverage_gate evaluation_start must precede evaluation_end"
        )

    start_year = int(evaluation_start.year)
    end_year = int(evaluation_end.year)
    if explicit:
        if development_start_year is not None and development_start_year != start_year:
            raise ValueError(
                "development_start_year conflicts with frozen evaluation_start"
            )
        if validation_end_year is not None and validation_end_year != end_year:
            raise ValueError("validation_end_year conflicts with frozen evaluation_end")
    else:
        if development_start_year is not None:
            start_year = int(development_start_year)
            evaluation_start = pd.Timestamp(year=start_year, month=1, day=1)
        if validation_end_year is not None:
            end_year = int(validation_end_year)
            evaluation_end = pd.Timestamp(year=end_year, month=12, day=31)
    if start_year > end_year:
        raise ValueError("development_start_year must be <= validation_end_year")

    resolved = {
        "evaluation_start": evaluation_start.date().isoformat(),
        "evaluation_end": evaluation_end.date().isoformat(),
        "min_consecutive_calendar_years": _integer_threshold(
            raw, "min_consecutive_calendar_years", minimum=1
        ),
        "min_year_end_active_directed_edges": _integer_threshold(
            raw, "min_year_end_active_directed_edges", minimum=1
        ),
        "min_year_end_connected_stocks": _integer_threshold(
            raw, "min_year_end_connected_stocks", minimum=1
        ),
        "min_median_connected_stocks_across_usable_years": _integer_threshold(
            raw,
            "min_median_connected_stocks_across_usable_years",
            minimum=1,
        ),
        "min_industries_if_pit_available": _integer_threshold(
            raw, "min_industries_if_pit_available", minimum=1
        ),
        "max_timing_violations": _integer_threshold(
            raw, "max_timing_violations", minimum=0
        ),
        "max_anonymous_resolved_nodes": _integer_threshold(
            raw, "max_anonymous_resolved_nodes", minimum=0
        ),
        "max_duplicate_pair_date_exposures": _integer_threshold(
            raw, "max_duplicate_pair_date_exposures", minimum=0
        ),
        "max_missing_source_provenance": _integer_threshold(
            raw, "max_missing_source_provenance", minimum=0
        ),
    }
    try:
        max_share = float(raw["max_largest_industry_edge_share_if_pit_available"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "graph_coverage_gate.max_largest_industry_edge_share_if_pit_available "
            "must be numeric"
        ) from exc
    if not 0.0 <= max_share <= 1.0:
        raise ValueError(
            "graph_coverage_gate.max_largest_industry_edge_share_if_pit_available "
            "must be in [0, 1]"
        )
    resolved["max_largest_industry_edge_share_if_pit_available"] = max_share
    return resolved, start_year, end_year


@dataclass(frozen=True)
class CoverageSummary:
    mentions: int
    explicit_name_mentions: int
    explicit_name_rate: float
    resolved_mentions: int
    resolution_rate_given_explicit: float
    listed_to_listed_edges: int
    unique_connected_stocks: int
    first_publication: str | None
    last_publication: str | None
    supplier_nodes: int = 0
    customer_nodes: int = 0
    source_documents: int = 0


def _listing_intervals(security_master: pd.DataFrame) -> pd.DataFrame:
    validate_table(security_master, SECURITY_MASTER)
    out = security_master[["security_id", "listing_date", "delisting_date"]].copy()
    out["security_id"] = out["security_id"].astype(str)
    out["listing_date"] = market_dates_utc(out["listing_date"], errors="coerce")
    out["delisting_date"] = market_dates_utc(out["delisting_date"], errors="coerce")
    return out


def listed_edge_intervals(
    edges: pd.DataFrame, security_master: pd.DataFrame
) -> pd.DataFrame:
    """Intersect relationship-active intervals with both companies' listing intervals."""
    if edges.empty:
        return edges.copy()
    validate_table(edges, SUPPLY_CHAIN_EDGE)
    return _intersect_listing_intervals(edges, security_master)


def _intersect_listing_intervals(
    edges: pd.DataFrame,
    security_master: pd.DataFrame,
) -> pd.DataFrame:
    """Internal tolerant form used by the gate to report, rather than raise on, bad rows."""
    required = {"supplier_id", "customer_id", "effective_start", "effective_end"}
    missing = sorted(required - set(edges.columns))
    if missing:
        raise ValueError(f"supply_chain_edge: missing coverage columns: {missing}")
    listing = _listing_intervals(security_master)
    supplier = listing.rename(
        columns={
            "security_id": "supplier_id",
            "listing_date": "supplier_listing_date",
            "delisting_date": "supplier_delisting_date",
        }
    )
    customer = listing.rename(
        columns={
            "security_id": "customer_id",
            "listing_date": "customer_listing_date",
            "delisting_date": "customer_delisting_date",
        }
    )
    out = edges.copy()
    out["supplier_id"] = out["supplier_id"].astype(str)
    out["customer_id"] = out["customer_id"].astype(str)
    out = out.merge(supplier, on="supplier_id", how="inner", validate="many_to_one")
    out = out.merge(customer, on="customer_id", how="inner", validate="many_to_one")

    starts = pd.concat(
        [
            market_instants_utc(out["effective_start"], errors="coerce"),
            market_instants_utc(out["supplier_listing_date"], errors="coerce"),
            market_instants_utc(out["customer_listing_date"], errors="coerce"),
        ],
        axis=1,
    )
    out["usable_start"] = starts.max(axis=1)

    far_future = pd.Timestamp("2262-01-01", tz="UTC")
    ends = pd.concat(
        [
            market_instants_utc(out["effective_end"], errors="coerce"),
            market_instants_utc(out["supplier_delisting_date"], errors="coerce").fillna(
                far_future
            ),
            market_instants_utc(out["customer_delisting_date"], errors="coerce").fillna(
                far_future
            ),
        ],
        axis=1,
    )
    out["usable_end"] = ends.min(axis=1)
    return out.loc[out["usable_start"] < out["usable_end"]].copy()


def summarize_coverage(
    mentions: pd.DataFrame,
    audit: pd.DataFrame,
    edges: pd.DataFrame,
    security_master: pd.DataFrame,
) -> CoverageSummary:
    validate_table(mentions, DISCLOSURE_RAW)
    validate_table(audit, RESOLUTION_AUDIT)
    validate_table(security_master, SECURITY_MASTER)
    if not edges.empty:
        validate_table(edges, SUPPLY_CHAIN_EDGE)

    attached = attach_resolution_audit(mentions, audit)
    explicit_mask = _classify_names(attached["counterparty_raw_name"]).eq(
        CounterpartyNameClass.NAMED
    )
    explicit = int(explicit_mask.sum())
    resolved_mask = (
        attached["resolution_status"].eq("resolved")
        & _nonblank(attached["resolved_entity_id"])
        & explicit_mask
    )
    resolved = int(resolved_mask.sum())
    listed_edges = (
        listed_edge_intervals(edges, security_master)
        if not edges.empty
        else edges.copy()
    )
    economic_pairs = (
        listed_edges[["supplier_id", "customer_id"]].drop_duplicates()
        if not listed_edges.empty
        else pd.DataFrame(columns=["supplier_id", "customer_id"])
    )
    suppliers = set(economic_pairs["supplier_id"].astype(str))
    customers = set(economic_pairs["customer_id"].astype(str))
    connected = suppliers | customers
    pubs = market_instants_utc(mentions["publication_datetime"], errors="coerce")
    return CoverageSummary(
        mentions=len(mentions),
        explicit_name_mentions=explicit,
        explicit_name_rate=(explicit / len(mentions)) if len(mentions) else 0.0,
        resolved_mentions=resolved,
        resolution_rate_given_explicit=(resolved / explicit) if explicit else 0.0,
        listed_to_listed_edges=len(economic_pairs),
        unique_connected_stocks=len(connected),
        first_publication=pubs.min().isoformat() if pubs.notna().any() else None,
        last_publication=pubs.max().isoformat() if pubs.notna().any() else None,
        supplier_nodes=len(suppliers),
        customer_nodes=len(customers),
        source_documents=(
            int(listed_edges["source_document_id"].dropna().astype(str).nunique())
            if not listed_edges.empty
            else 0
        ),
    )


def yearly_disclosure_diagnostics(
    mentions: pd.DataFrame, audit: pd.DataFrame
) -> pd.DataFrame:
    merged = attach_resolution_audit(mentions, audit)
    merged["year"] = market_calendar_year(merged["publication_datetime"])
    merged["explicit"] = _classify_names(merged["counterparty_raw_name"]).eq(
        CounterpartyNameClass.NAMED
    )
    merged["resolved"] = (
        merged["resolution_status"].eq("resolved")
        & _nonblank(merged["resolved_entity_id"])
        & merged["explicit"]
    )
    out = (
        merged.groupby("year")
        .agg(
            mentions=("counterparty_raw_name", "size"),
            explicit_name_mentions=("explicit", "sum"),
            resolved_mentions=("resolved", "sum"),
        )
        .reset_index()
    )
    out["explicit_name_rate"] = out["explicit_name_mentions"] / out["mentions"]
    out["resolution_rate_given_explicit"] = out["resolved_mentions"] / out[
        "explicit_name_mentions"
    ].replace(0, pd.NA)
    return out


def _year_end_timestamp(year: int) -> pd.Timestamp:
    return market_year_end_utc(year)


def _active_listed_edges(listed: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return one directed economic exposure per pair from pre-intersected intervals."""
    if listed.empty:
        return listed.copy()
    usable_start = market_instants_utc(listed["usable_start"], errors="coerce")
    usable_end = market_instants_utc(listed["usable_end"], errors="coerce")
    as_of_utc = market_timestamp_utc(as_of)
    active = listed.loc[(usable_start <= as_of_utc) & (as_of_utc < usable_end)].copy()
    if active.empty:
        return active
    sort_columns = ["supplier_id", "customer_id", "effective_start"]
    if "relationship_confidence" in active.columns:
        sort_columns.append("relationship_confidence")
    if "source_document_id" in active.columns:
        sort_columns.append("source_document_id")
    return (
        active.sort_values(sort_columns, kind="stable", na_position="first")
        .drop_duplicates(["supplier_id", "customer_id"], keep="last")
        .reset_index(drop=True)
    )


def _yearly_diagnostics_from_listed(
    listed: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        as_of = _year_end_timestamp(year)
        active = _active_listed_edges(listed, as_of)
        suppliers = set(active.get("supplier_id", pd.Series(dtype=str)).astype(str))
        customers = set(active.get("customer_id", pd.Series(dtype=str)).astype(str))
        if active.empty:
            median_age: float | None = None
        else:
            ages = (
                as_of - market_instants_utc(active["effective_start"], errors="coerce")
            ).dt.total_seconds() / 86400.0
            median_age = float(ages.median()) if ages.notna().any() else None
        rows.append(
            {
                "year": year,
                "year_end_active_edges": len(active),
                "year_end_connected_stocks": len(suppliers | customers),
                "supplier_node_count": len(suppliers),
                "customer_node_count": len(customers),
                "median_edge_age_days": median_age,
            }
        )
    return pd.DataFrame(rows)


def yearly_diagnostics(
    edges: pd.DataFrame,
    security_master: pd.DataFrame,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Report year-end listed graph coverage, including zero-coverage requested years.

    ``start_year`` and ``end_year`` are optional for backwards compatibility.  The
    Phase-3 gate always supplies its frozen development/validation bounds so that
    gaps before the first edge and after the last edge cannot disappear.
    """
    listed = (
        listed_edge_intervals(edges, security_master)
        if not edges.empty
        else edges.copy()
    )
    if start_year is None or end_year is None:
        if listed.empty:
            return pd.DataFrame(
                columns=[
                    "year",
                    "year_end_active_edges",
                    "year_end_connected_stocks",
                    "supplier_node_count",
                    "customer_node_count",
                    "median_edge_age_days",
                ]
            )
        inferred_start = int(market_calendar_year(listed["usable_start"]).min())
        inferred_end = int(market_calendar_year(listed["usable_end"]).max())
        start_year = inferred_start if start_year is None else start_year
        end_year = inferred_end if end_year is None else end_year
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    return _yearly_diagnostics_from_listed(listed, int(start_year), int(end_year))


def degree_diagnostics(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(
            columns=["security_id", "out_degree", "in_degree", "total_degree"]
        )
    pairs = edges[["supplier_id", "customer_id"]].drop_duplicates()
    out_degree = (
        pairs.groupby("supplier_id")["customer_id"].nunique().rename("out_degree")
    )
    in_degree = (
        pairs.groupby("customer_id")["supplier_id"].nunique().rename("in_degree")
    )
    degree = pd.concat([out_degree, in_degree], axis=1).fillna(0).astype(int)
    degree["total_degree"] = degree["out_degree"] + degree["in_degree"]
    return degree.reset_index(names="security_id")


def node_role_diagnostics(edges: pd.DataFrame) -> pd.DataFrame:
    """Return one row per graph node with supplier/customer role indicators."""
    columns = [
        "security_id",
        "is_supplier_node",
        "is_customer_node",
        "node_role",
        "supplier_out_degree",
        "customer_in_degree",
    ]
    if edges.empty:
        return pd.DataFrame(columns=columns)
    pairs = edges[["supplier_id", "customer_id"]].astype(str).drop_duplicates()
    out_degree = pairs.groupby("supplier_id")["customer_id"].nunique()
    in_degree = pairs.groupby("customer_id")["supplier_id"].nunique()
    node_ids = sorted(set(out_degree.index) | set(in_degree.index))
    rows = []
    for security_id in node_ids:
        is_supplier = security_id in out_degree.index
        is_customer = security_id in in_degree.index
        role = (
            "both"
            if is_supplier and is_customer
            else ("supplier" if is_supplier else "customer")
        )
        rows.append(
            {
                "security_id": security_id,
                "is_supplier_node": bool(is_supplier),
                "is_customer_node": bool(is_customer),
                "node_role": role,
                "supplier_out_degree": int(out_degree.get(security_id, 0)),
                "customer_in_degree": int(in_degree.get(security_id, 0)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _node_role_counts(edges: pd.DataFrame) -> dict[str, int]:
    roles = node_role_diagnostics(edges)
    if roles.empty:
        return {
            "supplier_node_count": 0,
            "customer_node_count": 0,
            "both_role_node_count": 0,
            "connected_node_count": 0,
        }
    return {
        "supplier_node_count": int(roles["is_supplier_node"].sum()),
        "customer_node_count": int(roles["is_customer_node"].sum()),
        "both_role_node_count": int(roles["node_role"].eq("both").sum()),
        "connected_node_count": len(roles),
    }


def _nonblank(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("")


def _classify_names(series: pd.Series) -> pd.Series:
    """Classify raw labels while preserving null as the invalid third state."""

    return series.map(
        lambda value: (
            CounterpartyNameClass.INVALID
            if pd.isna(value)
            else classify_counterparty_name(value)
        )
    )


def _resolution_name_diagnostics(audit: pd.DataFrame) -> dict[str, Any]:
    classes = _classify_names(audit["counterparty_raw_name"])
    statuses = audit["resolution_status"].astype("string")
    resolved_ids = _nonblank(audit["resolved_entity_id"])
    anonymous = classes.eq(CounterpartyNameClass.ANONYMOUS)
    named = classes.eq(CounterpartyNameClass.NAMED)
    allowed_named_statuses = {"resolved", "ambiguous", "unresolved"}
    resolved_methods = {
        "exact_alias",
        "deterministic_identifier",
        "high_confidence_fuzzy",
        "manual_override",
    }
    mismatch = (
        (anonymous & (statuses.ne("anonymous").fillna(True) | resolved_ids))
        | (named & ~statuses.isin(allowed_named_statuses))
        | (named & statuses.eq("resolved") & ~resolved_ids)
        | (named & statuses.ne("resolved") & resolved_ids)
    )
    if "resolution_method" in audit.columns:
        methods = audit["resolution_method"].astype("string")
        mismatch |= anonymous & methods.ne("blocked_anonymous").fillna(True)
        mismatch |= named & statuses.eq("resolved") & ~methods.isin(resolved_methods)
        mismatch |= (
            named
            & statuses.eq("ambiguous")
            & methods.ne("exact_alias_collision").fillna(True)
        )
        mismatch |= named & statuses.eq("unresolved") & methods.ne("none").fillna(True)
    if "normalized_name" in audit.columns:
        normalized = audit["normalized_name"].astype("string")
        expected = audit["counterparty_raw_name"].map(
            lambda value: "" if pd.isna(value) else normalize_company_name(str(value))
        )
        mismatch |= anonymous & _nonblank(normalized)
        mismatch |= named & normalized.ne(expected).fillna(True)
    return {
        "classes": classes,
        "invalid_resolution_audit_names": int(
            classes.eq(CounterpartyNameClass.INVALID).sum()
        ),
        "resolution_status_classification_mismatches": int(mismatch.sum()),
        "anonymous_with_resolved_id": anonymous & resolved_ids,
    }


def source_document_diagnostics(
    edges: pd.DataFrame,
    mentions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Describe edge evidence and raw-mention provenance by source document."""
    columns = [
        "source_document_id",
        "edge_evidence_rows",
        "unique_directed_edges",
        "supplier_nodes",
        "customer_nodes",
        "first_publication",
        "last_publication",
        "relationship_mentions",
        "explicit_name_mentions",
        "source_provenance_complete",
    ]
    edge_rows: list[dict[str, Any]] = []
    if not edges.empty and "source_document_id" in edges.columns:
        for document_id, group in edges.groupby("source_document_id", dropna=False):
            pubs = market_instants_utc(
                group.get("publication_datetime"), errors="coerce"
            )
            edge_rows.append(
                {
                    "source_document_id": document_id,
                    "edge_evidence_rows": len(group),
                    "unique_directed_edges": len(
                        group[["supplier_id", "customer_id"]].drop_duplicates()
                    ),
                    "supplier_nodes": int(group["supplier_id"].astype(str).nunique()),
                    "customer_nodes": int(group["customer_id"].astype(str).nunique()),
                    "first_publication": pubs.min() if pubs.notna().any() else pd.NaT,
                    "last_publication": pubs.max() if pubs.notna().any() else pd.NaT,
                }
            )
    edge_diag = pd.DataFrame(edge_rows)

    mention_rows: list[dict[str, Any]] = []
    if mentions is not None and not mentions.empty:
        required = {
            "source_document_id",
            "counterparty_raw_name",
            "source_document_url_or_path",
        }
        missing = sorted(required - set(mentions.columns))
        if missing:
            raise ValueError(f"disclosure_raw missing diagnostic columns: {missing}")
        for document_id, group in mentions.groupby("source_document_id", dropna=False):
            explicit = _classify_names(group["counterparty_raw_name"]).eq(
                CounterpartyNameClass.NAMED
            )
            provenance = _nonblank(group["source_document_url_or_path"])
            mention_rows.append(
                {
                    "source_document_id": document_id,
                    "relationship_mentions": len(group),
                    "explicit_name_mentions": int(explicit.sum()),
                    "source_provenance_complete": bool(provenance.all()),
                }
            )
    mention_diag = pd.DataFrame(mention_rows)

    if edge_diag.empty and mention_diag.empty:
        return pd.DataFrame(columns=columns)
    if edge_diag.empty:
        out = mention_diag
    elif mention_diag.empty:
        out = edge_diag
    else:
        out = edge_diag.merge(mention_diag, on="source_document_id", how="outer")
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return (
        out[columns]
        .sort_values("source_document_id", na_position="last")
        .reset_index(drop=True)
    )


def edge_age_distribution(
    edges: pd.DataFrame, as_of: str | pd.Timestamp
) -> pd.DataFrame:
    snap = graph_snapshot(edges, as_of)
    if snap.empty:
        return pd.DataFrame(columns=["age_bucket", "edges"])
    age_days = (
        market_timestamp_utc(as_of) - market_instants_utc(snap["effective_start"])
    ).dt.total_seconds() / 86400.0
    buckets = pd.cut(
        age_days,
        bins=[-0.001, 90, 180, 365, float("inf")],
        labels=["0-90d", "91-180d", "181-365d", "366d+"],
    )
    return (
        buckets.value_counts(sort=False)
        .rename_axis("age_bucket")
        .reset_index(name="edges")
    )


def industry_diagnostics(
    edges: pd.DataFrame, security_master: pd.DataFrame
) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(
            columns=["industry_code", "industry_name", "connected_stocks"]
        )
    nodes = pd.DataFrame(
        {
            "security_id": sorted(
                set(edges["supplier_id"].astype(str))
                | set(edges["customer_id"].astype(str))
            )
        }
    )
    master = security_master.copy()
    master["security_id"] = master["security_id"].astype(str)
    merged = nodes.merge(
        master[["security_id", "industry_code", "industry_name"]],
        on="security_id",
        how="left",
    )
    return (
        merged.groupby(["industry_code", "industry_name"], dropna=False)["security_id"]
        .nunique()
        .rename("connected_stocks")
        .reset_index()
        .sort_values("connected_stocks", ascending=False)
    )


def _pit_industry_history(
    security_master: pd.DataFrame,
    pit_industry: pd.DataFrame | None,
) -> pd.DataFrame | None:
    history = pit_industry
    if history is None and {
        "industry_code",
        "industry_valid_from",
        "industry_valid_to",
    }.issubset(security_master.columns):
        has_codes = _nonblank(security_master["industry_code"]).any()
        has_validity = (
            market_dates_utc(security_master["industry_valid_from"], errors="coerce")
            .notna()
            .any()
            or market_dates_utc(security_master["industry_valid_to"], errors="coerce")
            .notna()
            .any()
        )
        if has_codes and has_validity:
            history = security_master
    if history is None:
        return None
    required = {"security_id", "industry_code"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"PIT industry data missing columns: {missing}")
    if {"industry_valid_from", "industry_valid_to"}.issubset(history.columns):
        valid_from_column = "industry_valid_from"
        valid_to_column = "industry_valid_to"
    elif {"valid_from", "valid_to"}.issubset(history.columns):
        valid_from_column = "valid_from"
        valid_to_column = "valid_to"
    else:
        raise ValueError(
            "PIT industry data require industry_valid_from/industry_valid_to "
            "(or valid_from/valid_to); static current industry is not PIT-safe"
        )
    out = history.copy()
    out["security_id"] = out["security_id"].astype(str)
    out["industry_code"] = out["industry_code"].astype("string")
    out["_industry_valid_from"] = market_dates_utc(
        out[valid_from_column], errors="raise"
    )
    out["_industry_valid_to"] = market_dates_utc(out[valid_to_column], errors="raise")
    bounded = out["_industry_valid_from"].notna() & out["_industry_valid_to"].notna()
    invalid_interval = bounded & (
        out["_industry_valid_from"] >= out["_industry_valid_to"]
    )
    if invalid_interval.any():
        raise ValueError("PIT industry valid_from must precede valid_to")
    if "industry_name" not in out.columns:
        out["industry_name"] = pd.NA
    return out


def _industry_at(
    history: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, set[str]]:
    as_of = market_timestamp_utc(as_of)
    active = history.loc[
        (
            history["_industry_valid_from"].isna()
            | (history["_industry_valid_from"] <= as_of)
        )
        & (
            history["_industry_valid_to"].isna()
            | (as_of < history["_industry_valid_to"])
        )
        & _nonblank(history["industry_code"])
    ].copy()
    if active.empty:
        return active[["security_id", "industry_code", "industry_name"]], set()
    distinct = active[["security_id", "industry_code"]].drop_duplicates()
    ambiguous_ids = set(
        distinct.groupby("security_id")["industry_code"]
        .nunique()
        .loc[lambda counts: counts.gt(1)]
        .index.astype(str)
    )
    active = active.loc[~active["security_id"].isin(ambiguous_ids)].copy()
    active = active.sort_values(
        ["security_id", "_industry_valid_from"],
        kind="stable",
        na_position="first",
    ).drop_duplicates("security_id", keep="last")
    return active[["security_id", "industry_code", "industry_name"]], ambiguous_ids


def pit_industry_distribution(
    edges: pd.DataFrame,
    security_master: pd.DataFrame,
    pit_industry: pd.DataFrame,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return PIT industry shares of active directed-edge endpoint incidences.

    Every active directed edge contributes one supplier endpoint and one customer
    endpoint.  This avoids assigning a cross-industry relationship arbitrarily to
    one side while still making the concentration denominator explicit.
    """
    columns = ["industry_code", "industry_name", "edge_endpoint_incidences", "share"]
    history = _pit_industry_history(security_master, pit_industry)
    if history is None:
        return pd.DataFrame(columns=columns)
    listed = (
        listed_edge_intervals(edges, security_master)
        if not edges.empty
        else edges.copy()
    )
    as_of_utc = market_timestamp_utc(as_of)
    active = _active_listed_edges(listed, as_of_utc)
    if active.empty:
        return pd.DataFrame(columns=columns)
    at_date, _ = _industry_at(history, as_of_utc)
    supplier = active[["supplier_id", "customer_id"]].copy()
    supplier["security_id"] = supplier["supplier_id"].astype(str)
    customer = active[["supplier_id", "customer_id"]].copy()
    customer["security_id"] = customer["customer_id"].astype(str)
    endpoints = pd.concat([supplier, customer], ignore_index=True)
    mapped = endpoints.merge(
        at_date, on="security_id", how="left", validate="many_to_one"
    )
    mapped = mapped.loc[_nonblank(mapped["industry_code"])].copy()
    if mapped.empty:
        return pd.DataFrame(columns=columns)
    counts = (
        mapped.groupby(["industry_code", "industry_name"], dropna=False)
        .size()
        .rename("edge_endpoint_incidences")
        .reset_index()
    )
    counts["share"] = (
        counts["edge_endpoint_incidences"] / counts["edge_endpoint_incidences"].sum()
    )
    return (
        counts[columns]
        .sort_values(
            ["edge_endpoint_incidences", "industry_code"], ascending=[False, True]
        )
        .reset_index(drop=True)
    )


def pit_industry_yearly_diagnostics(
    edges: pd.DataFrame,
    security_master: pd.DataFrame,
    pit_industry: pd.DataFrame,
    years: Iterable[int],
) -> pd.DataFrame:
    """Summarize PIT industry breadth/concentration for requested year ends."""
    history = _pit_industry_history(security_master, pit_industry)
    if history is None:
        raise ValueError("pit_industry is required")
    listed = (
        _intersect_listing_intervals(edges, security_master)
        if not edges.empty
        else edges.copy()
    )
    rows: list[dict[str, Any]] = []
    for raw_year in years:
        year = int(raw_year)
        as_of = _year_end_timestamp(year)
        active = _active_listed_edges(listed, as_of)
        at_date, ambiguous_ids = _industry_at(history, as_of)
        if active.empty:
            classified = 0
            total = 0
            industry_count = 0
            largest_share: float | None = None
        else:
            endpoint_ids = (
                pd.concat(
                    [active["supplier_id"], active["customer_id"]], ignore_index=True
                )
                .astype(str)
                .rename("security_id")
                .to_frame()
            )
            active_endpoint_ids = set(endpoint_ids["security_id"].astype(str))
            mapped = endpoint_ids.merge(
                at_date, on="security_id", how="left", validate="many_to_one"
            )
            valid = _nonblank(mapped["industry_code"])
            classified = int(valid.sum())
            total = len(mapped)
            counts = mapped.loc[valid, "industry_code"].value_counts()
            industry_count = len(counts)
            largest_share = float(counts.max() / classified) if classified else None
        rows.append(
            {
                "year": year,
                "active_edges": len(active),
                "industry_count": industry_count,
                "largest_industry_share": largest_share,
                "classified_edge_endpoints": classified,
                "total_edge_endpoints": total,
                "industry_classification_rate": (classified / total) if total else None,
                "unclassified_edge_endpoints": total - classified,
                "ambiguous_pit_industry_nodes": len(active_endpoint_ids & ambiguous_ids)
                if not active.empty
                else 0,
            }
        )
    return pd.DataFrame(rows)


def _minimum_check(actual: float | None, threshold: float) -> dict[str, Any]:
    return {
        "actual": actual,
        "threshold": threshold,
        "operator": ">=",
        "passed": bool(actual is not None and actual >= threshold),
    }


def _maximum_check(actual: float | None, threshold: float) -> dict[str, Any]:
    return {
        "actual": actual,
        "threshold": threshold,
        "operator": "<=",
        "passed": bool(actual is not None and actual <= threshold),
    }


def _zero_check(actual: int) -> dict[str, Any]:
    return {
        "actual": int(actual),
        "threshold": 0,
        "operator": "==",
        "passed": actual == 0,
    }


def _consecutive_runs(years: Iterable[int]) -> list[list[int]]:
    ordered = sorted({int(year) for year in years})
    if not ordered:
        return []
    runs: list[list[int]] = [[ordered[0]]]
    for year in ordered[1:]:
        if year == runs[-1][-1] + 1:
            runs[-1].append(year)
        else:
            runs.append([year])
    return runs


def _active_during_window(
    edges: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.Series:
    starts = market_instants_utc(edges["effective_start"], errors="coerce")
    ends = market_instants_utc(edges["effective_end"], errors="coerce")
    window_start = market_year_start_utc(start_year)
    window_end = market_year_start_utc(end_year + 1)
    return starts.notna() & ends.notna() & (starts < window_end) & (window_start < ends)


def _endpoint_master_diagnostics(
    edges: pd.DataFrame,
    security_master: pd.DataFrame,
) -> tuple[int, int]:
    endpoints = pd.concat(
        [edges["supplier_id"], edges["customer_id"]], ignore_index=True
    ).astype(str)
    endpoint_ids = pd.DataFrame({"security_id": sorted(set(endpoints))})
    if endpoint_ids.empty:
        return 0, 0
    master = security_master[["security_id", "listing_date"]].copy()
    master["security_id"] = master["security_id"].astype(str)
    joined = endpoint_ids.merge(
        master,
        on="security_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_master = joined["_merge"].ne("both")
    listing_dates = market_dates_utc(joined["listing_date"], errors="coerce")
    missing_listing = ~missing_master & listing_dates.isna()
    return int(missing_master.sum()), int(missing_listing.sum())


def _phase3_integrity_diagnostics(
    edges: pd.DataFrame,
    *,
    security_master: pd.DataFrame,
    mentions: pd.DataFrame | None,
    audit: pd.DataFrame | None,
    listed: pd.DataFrame,
    start_year: int,
    end_year: int,
    max_edge_age_days: int,
    graph_exposures: Mapping[Any, pd.DataFrame] | None,
) -> dict[str, Any]:
    starts = market_instants_utc(edges["effective_start"], errors="coerce")
    ends = market_instants_utc(edges["effective_end"], errors="coerce")
    publications = market_instants_utc(edges["publication_datetime"], errors="coerce")
    timing_invalid = (
        starts.isna()
        | ends.isna()
        | publications.isna()
        | starts.ne(publications)
        | ends.le(starts)
    )
    edge_ages = ends - starts
    max_age_invalid = edge_ages.gt(pd.Timedelta(days=max_edge_age_days))

    invalid_mentions = 0
    if mentions is not None and not mentions.empty:
        if "counterparty_raw_name" not in mentions.columns:
            raise ValueError(
                "disclosure_raw missing integrity column: counterparty_raw_name"
            )
        mention_classes = _classify_names(mentions["counterparty_raw_name"])
        invalid_mentions = int(mention_classes.eq(CounterpartyNameClass.INVALID).sum())

    anonymous_node_ids: set[str] = set()
    invalid_audit_names = 0
    status_classification_mismatches = 0
    if audit is not None and not audit.empty:
        required = {"counterparty_raw_name", "resolution_status", "resolved_entity_id"}
        missing = sorted(required - set(audit.columns))
        if missing:
            raise ValueError(f"resolution_audit missing integrity columns: {missing}")
        name_diagnostics = _resolution_name_diagnostics(audit)
        invalid_audit_names = int(name_diagnostics["invalid_resolution_audit_names"])
        status_classification_mismatches = int(
            name_diagnostics["resolution_status_classification_mismatches"]
        )
        anonymous_node_ids.update(
            audit.loc[
                name_diagnostics["anonymous_with_resolved_id"],
                "resolved_entity_id",
            ]
            .astype(str)
            .tolist()
        )

    active_window = _active_during_window(edges, start_year, end_year)
    document_ids = edges["source_document_id"].astype("string")
    missing_provenance = ~_nonblank(edges["source_document_id"])
    if mentions is not None:
        required = {"source_document_id", "source_document_url_or_path"}
        missing = sorted(required - set(mentions.columns))
        if missing:
            raise ValueError(f"disclosure_raw missing provenance columns: {missing}")
        raw = mentions[["source_document_id", "source_document_url_or_path"]].copy()
        raw["source_document_id"] = raw["source_document_id"].astype("string")
        raw["_complete"] = _nonblank(raw["source_document_url_or_path"])
        complete_documents = set(
            raw.groupby("source_document_id", dropna=False)["_complete"]
            .all()
            .loc[lambda values: values]
            .index.astype(str)
        )
        missing_provenance = missing_provenance | ~document_ids.astype(str).isin(
            complete_documents
        )
    missing_provenance_count = int((active_window & missing_provenance).sum())

    raw_pair_dates = pd.DataFrame(
        {
            "supplier_id": edges["supplier_id"].astype(str),
            "customer_id": edges["customer_id"].astype(str),
            "date": starts.dt.tz_convert(MARKET_TIMEZONE).dt.normalize(),
        }
    )
    raw_duplicate_evidence = int(
        raw_pair_dates.loc[raw_pair_dates["date"].notna()]
        .duplicated(["supplier_id", "customer_id", "date"])
        .sum()
    )

    exposure_frames: Iterable[pd.DataFrame]
    if graph_exposures is None:
        exposure_frames = (
            _active_listed_edges(listed, _year_end_timestamp(year))
            for year in range(start_year, end_year + 1)
        )
    else:
        exposure_frames = graph_exposures.values()
    duplicate_exposures = 0
    for exposure in exposure_frames:
        if exposure.empty:
            continue
        required = {"supplier_id", "customer_id"}
        if not required.issubset(exposure.columns):
            raise ValueError(
                "graph exposure snapshots require supplier_id and customer_id"
            )
        duplicate_exposures += int(
            exposure.duplicated(["supplier_id", "customer_id"]).sum()
        )

    missing_master_endpoints, missing_listing_dates = _endpoint_master_diagnostics(
        edges, security_master
    )

    return {
        "timing_violations": int(timing_invalid.sum()),
        "edge_max_age_violations": int(max_age_invalid.sum()),
        "invalid_counterparty_mentions": invalid_mentions,
        "invalid_resolution_audit_names": invalid_audit_names,
        "resolution_status_classification_mismatches": (
            status_classification_mismatches
        ),
        "anonymous_resolved_nodes": len(anonymous_node_ids),
        "duplicate_pair_date_exposures": int(duplicate_exposures),
        "missing_source_provenance_on_active_edges": missing_provenance_count,
        "missing_security_master_endpoints": missing_master_endpoints,
        "missing_listing_dates_on_edge_endpoints": missing_listing_dates,
        "raw_duplicate_pair_date_evidence_rows": raw_duplicate_evidence,
    }


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    normalized = frame.astype(object).where(pd.notna(frame), None)
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: (
                value.isoformat() if isinstance(value, pd.Timestamp) else value
            )
        )
    return normalized.to_dict("records")


def evaluate_phase3_gate(
    edges: pd.DataFrame,
    security_master: pd.DataFrame,
    *,
    mentions: pd.DataFrame | None = None,
    audit: pd.DataFrame | None = None,
    pit_industry: pd.DataFrame | None = None,
    real_data: bool = False,
    frozen_thresholds: Mapping[str, Any] | None = None,
    development_start_year: int | None = None,
    validation_end_year: int | None = None,
    max_edge_age_days: int = 550,
    graph_exposures: Mapping[Any, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen V2 Phase-3 real-coverage and graph-integrity gate.

    Production callers must pass ``config['graph_coverage_gate']`` as
    ``frozen_thresholds``.  The default exists only for API compatibility; every
    reported threshold and every check is driven by the resolved mapping.
    """
    if real_data and frozen_thresholds is None:
        raise ValueError(
            "Real-data Phase-3 evaluation requires explicit frozen_thresholds"
        )
    thresholds, start_year, end_year = _resolved_phase3_thresholds(
        frozen_thresholds,
        development_start_year=development_start_year,
        validation_end_year=validation_end_year,
    )
    if isinstance(max_edge_age_days, bool):
        raise TypeError("max_edge_age_days must be a positive integer")
    try:
        parsed_max_edge_age_days = int(max_edge_age_days)
    except (TypeError, ValueError) as exc:
        raise TypeError("max_edge_age_days must be a positive integer") from exc
    if parsed_max_edge_age_days != max_edge_age_days or parsed_max_edge_age_days <= 0:
        raise ValueError("max_edge_age_days must be a positive integer")
    max_edge_age_days = parsed_max_edge_age_days
    validate_table(security_master, SECURITY_MASTER)
    required_edge_columns = {
        "supplier_id",
        "customer_id",
        "effective_start",
        "effective_end",
        "publication_datetime",
        "source_document_id",
    }
    missing = sorted(required_edge_columns - set(edges.columns))
    if missing:
        raise ValueError(f"supply_chain_edge missing Phase-3 columns: {missing}")

    listed = (
        _intersect_listing_intervals(edges, security_master)
        if not edges.empty
        else edges.copy()
    )
    yearly = _yearly_diagnostics_from_listed(listed, start_year, end_year)
    qualifies = (
        yearly.loc[
            (
                yearly["year_end_active_edges"]
                >= thresholds["min_year_end_active_directed_edges"]
            )
            & (
                yearly["year_end_connected_stocks"]
                >= thresholds["min_year_end_connected_stocks"]
            ),
            "year",
        ]
        .astype(int)
        .tolist()
    )
    runs = _consecutive_runs(qualifies)
    qualifying_runs = [
        run for run in runs if len(run) >= thresholds["min_consecutive_calendar_years"]
    ]
    usable_years = sorted({year for run in qualifying_runs for year in run})
    max_consecutive = max((len(run) for run in runs), default=0)
    usable_connected = yearly.loc[
        yearly["year"].isin(usable_years), "year_end_connected_stocks"
    ]
    median_connected = (
        float(usable_connected.median()) if not usable_connected.empty else None
    )

    integrity = _phase3_integrity_diagnostics(
        edges,
        security_master=security_master,
        mentions=mentions,
        audit=audit,
        listed=listed,
        start_year=start_year,
        end_year=end_year,
        max_edge_age_days=max_edge_age_days,
        graph_exposures=graph_exposures,
    )
    coverage_checks: dict[str, dict[str, Any]] = {
        "consecutive_usable_years": _minimum_check(
            max_consecutive, thresholds["min_consecutive_calendar_years"]
        ),
        "usable_years_median_connected_stocks": _minimum_check(
            median_connected,
            thresholds["min_median_connected_stocks_across_usable_years"],
        ),
    }
    integrity_checks: dict[str, dict[str, Any]] = {
        "timing_violations": _maximum_check(
            integrity["timing_violations"], thresholds["max_timing_violations"]
        ),
        "edge_max_age_violations": _zero_check(integrity["edge_max_age_violations"]),
        "invalid_counterparty_mentions": _zero_check(
            integrity["invalid_counterparty_mentions"]
        ),
        "invalid_resolution_audit_names": _zero_check(
            integrity["invalid_resolution_audit_names"]
        ),
        "resolution_status_classification_mismatches": _zero_check(
            integrity["resolution_status_classification_mismatches"]
        ),
        "anonymous_resolved_nodes": _maximum_check(
            integrity["anonymous_resolved_nodes"],
            thresholds["max_anonymous_resolved_nodes"],
        ),
        "duplicate_pair_date_exposures": _maximum_check(
            integrity["duplicate_pair_date_exposures"],
            thresholds["max_duplicate_pair_date_exposures"],
        ),
        "missing_source_provenance_on_active_edges": _maximum_check(
            integrity["missing_source_provenance_on_active_edges"],
            thresholds["max_missing_source_provenance"],
        ),
        "missing_security_master_endpoints": _zero_check(
            integrity["missing_security_master_endpoints"]
        ),
        "missing_listing_dates_on_edge_endpoints": _zero_check(
            integrity["missing_listing_dates_on_edge_endpoints"]
        ),
    }

    history = _pit_industry_history(security_master, pit_industry)
    industry_yearly = pd.DataFrame()
    industry_integrity_yearly = pd.DataFrame()
    if history is None:
        coverage_checks["pit_industry_count"] = {
            "actual": None,
            "threshold": thresholds["min_industries_if_pit_available"],
            "operator": ">=",
            "passed": None,
            "deferred": True,
            "reason": "PIT_INDUSTRY_UNAVAILABLE",
        }
        coverage_checks["pit_largest_industry_share"] = {
            "actual": None,
            "threshold": thresholds["max_largest_industry_edge_share_if_pit_available"],
            "operator": "<=",
            "passed": None,
            "deferred": True,
            "reason": "PIT_INDUSTRY_UNAVAILABLE",
        }
        for key in (
            "pit_industry_unclassified_active_endpoints",
            "pit_industry_ambiguous_active_nodes",
        ):
            integrity_checks[key] = {
                "actual": None,
                "threshold": 0,
                "operator": "==",
                "passed": None,
                "deferred": True,
                "reason": "PIT_INDUSTRY_UNAVAILABLE",
            }
        integrity["pit_industry_unclassified_active_endpoints"] = None
        integrity["pit_industry_ambiguous_active_nodes"] = None
    else:
        active_years = yearly.loc[
            yearly["year_end_active_edges"].gt(0), "year"
        ].tolist()
        industry_integrity_yearly = pit_industry_yearly_diagnostics(
            edges, security_master, history, active_years
        )
        industry_yearly = industry_integrity_yearly.loc[
            industry_integrity_yearly["year"].isin(usable_years)
        ].copy()
        minimum_industries = (
            int(industry_yearly["industry_count"].min())
            if not industry_yearly.empty
            else None
        )
        largest_share = (
            float(industry_yearly["largest_industry_share"].max())
            if not industry_yearly.empty
            and industry_yearly["largest_industry_share"].notna().any()
            else None
        )
        coverage_checks["pit_industry_count"] = _minimum_check(
            minimum_industries, thresholds["min_industries_if_pit_available"]
        )
        coverage_checks["pit_largest_industry_share"] = _maximum_check(
            largest_share,
            thresholds["max_largest_industry_edge_share_if_pit_available"],
        )
        unclassified_endpoints = (
            int(industry_integrity_yearly["unclassified_edge_endpoints"].sum())
            if not industry_integrity_yearly.empty
            else 0
        )
        ambiguous_nodes = (
            int(industry_integrity_yearly["ambiguous_pit_industry_nodes"].sum())
            if not industry_integrity_yearly.empty
            else 0
        )
        integrity["pit_industry_unclassified_active_endpoints"] = unclassified_endpoints
        integrity["pit_industry_ambiguous_active_nodes"] = ambiguous_nodes
        integrity_checks["pit_industry_unclassified_active_endpoints"] = _zero_check(
            unclassified_endpoints
        )
        integrity_checks["pit_industry_ambiguous_active_nodes"] = _zero_check(
            ambiguous_nodes
        )

    period_active = listed.copy()
    if not listed.empty:
        usable_start = market_instants_utc(listed["usable_start"], errors="coerce")
        usable_end = market_instants_utc(listed["usable_end"], errors="coerce")
        window_start = market_year_start_utc(start_year)
        window_end = market_year_start_utc(end_year + 1)
        period_active = listed.loc[
            (usable_start < window_end) & (window_start < usable_end)
        ].drop_duplicates(["supplier_id", "customer_id"])
    period_role_counts = _node_role_counts(period_active)
    latest_active = pd.DataFrame(columns=edges.columns)
    if usable_years:
        latest_active = _active_listed_edges(
            listed, _year_end_timestamp(max(usable_years))
        )
    role_counts = _node_role_counts(latest_active)
    documents = source_document_diagnostics(edges, mentions)
    integrity_passed = all(
        check["passed"] is not False for check in integrity_checks.values()
    )
    coverage_passed = all(
        check["passed"] is not False for check in coverage_checks.values()
    )
    passed = bool(real_data and integrity_passed and coverage_passed)
    if not real_data:
        status = "BLOCKED_NO_REAL_DISCLOSURE_DATA"
        outcome = "BLOCKED"
        failure_class = "NO_REAL_DATA"
    elif passed:
        status = "PASS"
        outcome = "PROCEED"
        failure_class = None
    elif not integrity_passed:
        status = "FAIL"
        outcome = "ENGINEERING_FAILURE"
        failure_class = "ENGINEERING_INTEGRITY"
    else:
        status = "FAIL"
        outcome = "INFEASIBLE_DATA"
        failure_class = "COVERAGE_INFEASIBLE"
    checks = {**coverage_checks, **integrity_checks}
    reported_thresholds = dict(thresholds)
    reported_thresholds["edge_max_age_days"] = max_edge_age_days
    return {
        "status": status,
        "outcome": outcome,
        "failure_class": failure_class,
        "passed": passed,
        "integrity_passed": bool(integrity_passed),
        "coverage_passed": bool(coverage_passed),
        "frozen_thresholds": reported_thresholds,
        "checks": checks,
        "integrity_checks": integrity_checks,
        "coverage_checks": coverage_checks,
        "usable_years": usable_years,
        "qualifying_year_runs": qualifying_runs,
        "integrity": integrity,
        "diagnostics": {
            "yearly_coverage": _json_records(yearly),
            "development_validation_node_roles": period_role_counts,
            "latest_usable_year_node_roles": role_counts,
            "source_documents": _json_records(documents),
            "pit_industry_by_usable_year": _json_records(industry_yearly),
            "pit_industry_integrity_by_active_year": _json_records(
                industry_integrity_yearly
            ),
        },
    }


def gate_status(
    summary: CoverageSummary,
    thresholds: dict[str, Any],
    *,
    real_data: bool,
    edges: pd.DataFrame | None = None,
    security_master: pd.DataFrame | None = None,
    mentions: pd.DataFrame | None = None,
    audit: pd.DataFrame | None = None,
    pit_industry: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return gate status while preserving the original summary-only API.

    New callers should pass ``edges`` and ``security_master`` (or call
    :func:`evaluate_phase3_gate` directly) to apply the frozen V2 gate.  Calls
    using only a summary retain the pre-V2 behavior for existing report scripts.
    """
    if edges is not None or security_master is not None:
        if edges is None or security_master is None:
            raise ValueError("edges and security_master must be supplied together")
        return evaluate_phase3_gate(
            edges,
            security_master,
            mentions=mentions,
            audit=audit,
            pit_industry=pit_industry,
            real_data=real_data,
            frozen_thresholds=thresholds,
        )
    if not real_data:
        return {
            "status": "BLOCKED_NO_REAL_DISCLOSURE_DATA",
            "passed": False,
            "checks": {},
        }
    mapping = {
        "min_explicit_name_rate": summary.explicit_name_rate,
        "min_resolution_rate_given_explicit": summary.resolution_rate_given_explicit,
        "min_listed_to_listed_edges": summary.listed_to_listed_edges,
        "min_connected_stocks": summary.unique_connected_stocks,
    }
    checks = {}
    frozen = True
    passed = True
    for key, actual in mapping.items():
        threshold = thresholds.get(key)
        if threshold is None:
            frozen = False
            checks[key] = {"actual": actual, "threshold": None, "passed": None}
        else:
            ok = actual >= threshold
            checks[key] = {"actual": actual, "threshold": threshold, "passed": ok}
            passed = passed and ok
    if not frozen:
        return {
            "status": "BLOCKED_THRESHOLDS_NOT_FROZEN",
            "passed": False,
            "checks": checks,
        }
    return {"status": "PASS" if passed else "FAIL", "passed": passed, "checks": checks}


def summary_to_dict(summary: CoverageSummary) -> dict[str, Any]:
    return asdict(summary)
