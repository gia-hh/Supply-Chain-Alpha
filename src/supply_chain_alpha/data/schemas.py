from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Number

import pandas as pd

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VALID_EXCHANGES = frozenset({"SSE", "SZSE"})


@dataclass(frozen=True)
class TableSchema:
    """An explicit, dependency-free contract for a canonical table."""

    name: str
    required_columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    identifier_columns: tuple[str, ...] = ()
    string_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    integer_columns: tuple[str, ...] = ()
    boolean_columns: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        """All declared columns, preserving contract order."""

        return self.required_columns + self.optional_columns


SECURITY_MASTER = TableSchema(
    "security_master",
    (
        "security_id",
        "ticker",
        "exchange",
        "company_name",
        "listing_date",
        "delisting_date",
        "board",
    ),
    ("security_id",),
    optional_columns=(
        "industry_code",
        "industry_name",
        "industry_valid_from",
        "industry_valid_to",
    ),
    identifier_columns=("security_id", "ticker"),
    string_columns=(
        "exchange",
        "company_name",
        "board",
        "industry_code",
        "industry_name",
    ),
    date_columns=(
        "listing_date",
        "delisting_date",
        "industry_valid_from",
        "industry_valid_to",
    ),
)

COMPANY_ALIAS = TableSchema(
    "company_alias",
    (
        "entity_id",
        "canonical_name",
        "alias",
        "alias_type",
        "valid_from",
        "valid_to",
        "source",
    ),
    ("entity_id", "alias", "valid_from"),
    identifier_columns=("entity_id",),
    string_columns=("canonical_name", "alias", "alias_type", "source"),
    date_columns=("valid_from", "valid_to"),
)

CNINFO_DOCUMENT_AUDIT = TableSchema(
    "cninfo_document_audit",
    (
        "source_document_id",
        "source_company_id",
        "source_company_reported_id",
        "source_company_org_id",
        "source_company_id_reconciliation_method",
        "source_company_id_reconciliation_candidate_ids",
        "source_company_id_reconciliation_candidate_count",
        "source_period_end",
        "source_announcement_time_ms",
        "source_publication_datetime",
        "publication_datetime",
        "publication_time_precision",
        "publication_timing_rule",
        "source_url",
        "local_raw_path",
        "retrieval_datetime",
        "sha256",
        "contains_relationship_section",
        "captured_relationship_section",
        "section_types",
        "mention_count",
        "named_mention_count",
        "anonymous_mention_count",
        "page_count",
        "pages_with_text",
        "extraction_status",
        "extraction_error",
    ),
    ("source_document_id",),
    identifier_columns=(
        "source_document_id",
        "source_company_id",
        "source_company_reported_id",
        "source_company_org_id",
    ),
    string_columns=(
        "source_company_id_reconciliation_method",
        "source_company_id_reconciliation_candidate_ids",
        "publication_time_precision",
        "publication_timing_rule",
        "source_url",
        "local_raw_path",
        "sha256",
        "section_types",
        "extraction_status",
        "extraction_error",
    ),
    date_columns=("source_period_end",),
    datetime_columns=(
        "source_publication_datetime",
        "publication_datetime",
        "retrieval_datetime",
    ),
    integer_columns=(
        "source_announcement_time_ms",
        "source_company_id_reconciliation_candidate_count",
        "mention_count",
        "named_mention_count",
        "anonymous_mention_count",
        "page_count",
        "pages_with_text",
    ),
    boolean_columns=(
        "contains_relationship_section",
        "captured_relationship_section",
    ),
)

DISCLOSURE_RAW = TableSchema(
    "disclosure_raw",
    (
        "source_company_id",
        "counterparty_raw_name",
        "relationship_type",
        "source_period_end",
        "publication_datetime",
        "source_document_id",
        "source_document_url_or_path",
        "evidence_text",
        "exposure_value",
        "exposure_share",
    ),
    (
        "source_document_id",
        "source_company_id",
        "relationship_type",
        "counterparty_raw_name",
    ),
    optional_columns=(
        "source_announcement_time_ms",
        "source_publication_datetime",
        "publication_time_precision",
        "publication_timing_rule",
    ),
    identifier_columns=("source_company_id", "source_document_id"),
    string_columns=(
        "counterparty_raw_name",
        "relationship_type",
        "source_document_url_or_path",
        "evidence_text",
        "publication_time_precision",
        "publication_timing_rule",
    ),
    date_columns=("source_period_end",),
    datetime_columns=("publication_datetime", "source_publication_datetime"),
    numeric_columns=("exposure_value", "exposure_share"),
    integer_columns=("source_announcement_time_ms",),
)

RESOLUTION_AUDIT = TableSchema(
    "resolution_audit",
    (
        "source_document_id",
        "source_company_id",
        "counterparty_raw_name",
        "resolved_entity_id",
        "resolution_method",
        "resolution_confidence",
        "resolution_status",
        "candidate_count",
        "notes",
    ),
    ("source_document_id", "source_company_id", "counterparty_raw_name"),
    optional_columns=("normalized_name",),
    identifier_columns=(
        "source_document_id",
        "source_company_id",
        "resolved_entity_id",
    ),
    string_columns=(
        "counterparty_raw_name",
        "resolution_method",
        "resolution_status",
        "notes",
        "normalized_name",
    ),
    numeric_columns=("resolution_confidence",),
    integer_columns=("candidate_count",),
)

SUPPLY_CHAIN_EDGE = TableSchema(
    "supply_chain_edge",
    (
        "supplier_id",
        "customer_id",
        "effective_start",
        "effective_end",
        "source_document_id",
        "source_period_end",
        "source_company_id",
        "relationship_confidence",
        "economic_weight",
        "weight_source",
        # Kept from V1: point-in-time graph checks still require publication
        # provenance even though V2 adds source_company_id to this contract.
        "publication_datetime",
    ),
    ("supplier_id", "customer_id", "source_document_id"),
    identifier_columns=(
        "supplier_id",
        "customer_id",
        "source_document_id",
        "source_company_id",
    ),
    string_columns=("weight_source",),
    date_columns=("source_period_end",),
    datetime_columns=("effective_start", "effective_end", "publication_datetime"),
    numeric_columns=("relationship_confidence", "economic_weight"),
)

EQUITY_DAILY = TableSchema(
    "equity_daily",
    ("date", "security_id", "open", "close", "volume", "amount", "tradable"),
    ("date", "security_id"),
    optional_columns=(
        "high",
        "low",
        "market_cap",
        "free_float_market_cap",
        "suspended",
        "limit_up",
        "limit_down",
        "st_status",
    ),
    identifier_columns=("security_id",),
    string_columns=("st_status",),
    date_columns=("date",),
    numeric_columns=(
        "open",
        "close",
        "volume",
        "amount",
        "high",
        "low",
        "market_cap",
        "free_float_market_cap",
    ),
    boolean_columns=("tradable", "suspended", "limit_up", "limit_down"),
)

RETURNS_DAILY = TableSchema(
    "returns_daily",
    (
        "date",
        "security_id",
        "raw_return",
        "market_return",
        "industry_return",
        "residual_return",
        "beta_market",
        "beta_industry",
        "residual_model",
    ),
    ("date", "security_id"),
    identifier_columns=("security_id",),
    string_columns=("residual_model",),
    date_columns=("date",),
    numeric_columns=(
        "raw_return",
        "market_return",
        "industry_return",
        "residual_return",
        "beta_market",
        "beta_industry",
    ),
)

SIGNAL_DAILY = TableSchema(
    "signal_daily",
    (
        "date",
        "security_id",
        "customer_shock",
        "supplier_shock",
        "customer_neighbor_count",
        "supplier_neighbor_count",
        "graph_snapshot_date",
    ),
    ("date", "security_id"),
    optional_columns=(
        "customer_shock_weighted",
        "supplier_shock_weighted",
        "edge_age_mean",
        "edge_age_max",
    ),
    identifier_columns=("security_id",),
    date_columns=("date", "graph_snapshot_date"),
    numeric_columns=(
        "customer_shock",
        "supplier_shock",
        "customer_shock_weighted",
        "supplier_shock_weighted",
        "edge_age_mean",
        "edge_age_max",
    ),
    integer_columns=("customer_neighbor_count", "supplier_neighbor_count"),
)

SCHEMAS = {
    schema.name: schema
    for schema in (
        SECURITY_MASTER,
        COMPANY_ALIAS,
        CNINFO_DOCUMENT_AUDIT,
        DISCLOSURE_RAW,
        RESOLUTION_AUDIT,
        SUPPLY_CHAIN_EDGE,
        EQUITY_DAILY,
        RETURNS_DAILY,
        SIGNAL_DAILY,
    )
}


def _missing_columns(columns: Iterable[str], required: Iterable[str]) -> list[str]:
    present = set(columns)
    return [column for column in required if column not in present]


def _present_columns(df: pd.DataFrame, declared: Iterable[str]) -> list[str]:
    return [column for column in declared if column in df.columns]


def _nonnull_values(series: pd.Series) -> list[object]:
    return series.loc[series.notna()].tolist()


def _validate_string_columns(df: pd.DataFrame, schema: TableSchema) -> None:
    columns = _present_columns(df, (*schema.identifier_columns, *schema.string_columns))
    for column in dict.fromkeys(columns):
        values = _nonnull_values(df[column])
        invalid = [value for value in values if not isinstance(value, str)]
        if invalid:
            raise ValueError(
                f"{schema.name}: {column} must contain strings; sample={invalid[:5]}"
            )
        if any(not value.strip() for value in values):
            raise ValueError(f"{schema.name}: {column} contains empty strings")


def _validate_temporal_column(
    df: pd.DataFrame,
    schema: TableSchema,
    column: str,
    *,
    date_only: bool,
) -> None:
    values = _nonnull_values(df[column])
    for value in values:
        if isinstance(value, (bool, Number)):
            raise TypeError(
                f"{schema.name}: {column} contains a non-date value: {value!r}"
            )
        if isinstance(value, str):
            if date_only and not _ISO_DATE_RE.fullmatch(value):
                raise ValueError(
                    f"{schema.name}: {column} must use ISO YYYY-MM-DD strings; got {value!r}"
                )
            try:
                parsed = pd.Timestamp(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{schema.name}: {column} contains an invalid date: {value!r}"
                ) from exc
            if pd.isna(parsed):
                raise ValueError(
                    f"{schema.name}: {column} contains an invalid date: {value!r}"
                )
            continue
        if isinstance(value, (date, datetime, pd.Timestamp)):
            continue
        # NumPy datetime scalars are accepted, but objects that merely happen
        # to stringify like a date are not.
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{schema.name}: {column} contains an invalid date: {value!r}"
            ) from exc
        if pd.isna(parsed) or type(value).__module__ != "numpy":
            raise ValueError(
                f"{schema.name}: {column} contains a non-date value: {value!r}"
            )


def _validate_numeric_columns(df: pd.DataFrame, schema: TableSchema) -> None:
    columns = _present_columns(df, (*schema.numeric_columns, *schema.integer_columns))
    integer_columns = set(schema.integer_columns)
    for column in dict.fromkeys(columns):
        invalid: list[object] = []
        for value in _nonnull_values(df[column]):
            if isinstance(value, bool) or not isinstance(value, Number):
                invalid.append(value)
                continue
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                finite = False
            if not finite:
                invalid.append(value)
                continue
            if (
                column in integer_columns
                and not isinstance(value, Integral)
                and not float(value).is_integer()
            ):
                invalid.append(value)
        if invalid:
            raise ValueError(
                f"{schema.name}: {column} must contain finite numeric values; sample={invalid[:5]}"
            )


def _validate_boolean_columns(df: pd.DataFrame, schema: TableSchema) -> None:
    for column in _present_columns(df, schema.boolean_columns):
        invalid = [
            value
            for value in _nonnull_values(df[column])
            if not isinstance(value, bool)
        ]
        if invalid:
            raise ValueError(
                f"{schema.name}: {column} must contain booleans; sample={invalid[:5]}"
            )


def _validate_ordered_dates(
    df: pd.DataFrame,
    schema: TableSchema,
    earlier: str,
    later: str,
    *,
    strict: bool,
) -> None:
    if earlier not in df.columns or later not in df.columns:
        return
    # Canonical date fields are sometimes compared with timezone-aware
    # publication timestamps (CNINFO uses Asia/Shanghai offsets).  Normalise
    # both operands to a single timeline before ordering them; pandas rejects
    # comparisons between tz-naive and tz-aware values.
    left = pd.to_datetime(df[earlier], errors="raise", utc=True)
    right = pd.to_datetime(df[later], errors="raise", utc=True)
    both = left.notna() & right.notna()
    invalid = both & ((left >= right) if strict else (left > right))
    if invalid.any():
        sample = df.loc[invalid, [earlier, later]].head(5).to_dict("records")
        operator = "<" if strict else "<="
        raise ValueError(
            f"{schema.name}: {earlier} must be {operator} {later}; sample={sample}"
        )


def _validate_domain_rules(df: pd.DataFrame, schema: TableSchema) -> None:
    if schema is SECURITY_MASTER:
        if df["exchange"].isna().any():
            raise ValueError("security_master: exchange must be non-null")
        invalid_exchange = ~df["exchange"].isin(_VALID_EXCHANGES)
        if invalid_exchange.any():
            bad = sorted(
                df.loc[invalid_exchange, "exchange"].astype(str).unique().tolist()
            )
            raise ValueError(f"security_master: invalid exchange; got {bad}")
        _validate_ordered_dates(
            df, schema, "listing_date", "delisting_date", strict=True
        )
        _validate_ordered_dates(
            df, schema, "industry_valid_from", "industry_valid_to", strict=True
        )

    elif schema is COMPANY_ALIAS:
        _validate_ordered_dates(df, schema, "valid_from", "valid_to", strict=True)

    elif schema is CNINFO_DOCUMENT_AUDIT:
        from .cninfo import SECURITY_ID_RECONCILIATION_METHODS

        methods = df["source_company_id_reconciliation_method"].dropna()
        invalid_methods = ~methods.isin(SECURITY_ID_RECONCILIATION_METHODS)
        if invalid_methods.any():
            bad = sorted(methods.loc[invalid_methods].astype(str).unique().tolist())
            raise ValueError(
                "cninfo_document_audit: invalid security-ID reconciliation "
                f"method; got {bad}"
            )
        counts = df["source_company_id_reconciliation_candidate_count"]
        if (counts.notna() & counts.lt(0)).any():
            raise ValueError(
                "cninfo_document_audit: reconciliation candidate count must be "
                "non-negative"
            )

    elif schema is DISCLOSURE_RAW:
        invalid_relation = ~df["relationship_type"].isin({"customer", "supplier"})
        if invalid_relation.any():
            bad = sorted(
                df.loc[invalid_relation, "relationship_type"]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                f"disclosure_raw: relationship_type must be customer/supplier; got {bad}"
            )
        invalid_share = df["exposure_share"].notna() & ~df["exposure_share"].between(
            0.0, 1.0, inclusive="both"
        )
        if invalid_share.any():
            sample = df.loc[invalid_share, "exposure_share"].head(5).tolist()
            raise ValueError(
                f"disclosure_raw: exposure_share must be decimal in [0, 1]; sample={sample}"
            )
        invalid_value = df["exposure_value"].notna() & (df["exposure_value"] < 0)
        if invalid_value.any():
            sample = df.loc[invalid_value, "exposure_value"].head(5).tolist()
            raise ValueError(
                f"disclosure_raw: exposure_value must be non-negative; sample={sample}"
            )
        _validate_ordered_dates(
            df, schema, "source_period_end", "publication_datetime", strict=False
        )

    elif schema is RESOLUTION_AUDIT:
        invalid_confidence = df["resolution_confidence"].notna() & ~df[
            "resolution_confidence"
        ].between(0.0, 1.0, inclusive="both")
        if invalid_confidence.any():
            raise ValueError(
                "resolution_audit: resolution_confidence must be in [0, 1]"
            )
        if (df["candidate_count"].notna() & (df["candidate_count"] < 0)).any():
            raise ValueError("resolution_audit: candidate_count must be non-negative")

    elif schema is SUPPLY_CHAIN_EDGE:
        invalid_confidence = df["relationship_confidence"].notna() & ~df[
            "relationship_confidence"
        ].between(0.0, 1.0, inclusive="both")
        if invalid_confidence.any():
            raise ValueError(
                "supply_chain_edge: relationship_confidence must be in [0, 1]"
            )
        if (df["economic_weight"].notna() & (df["economic_weight"] < 0)).any():
            raise ValueError("supply_chain_edge: economic_weight must be non-negative")
        _validate_ordered_dates(
            df, schema, "effective_start", "effective_end", strict=True
        )
        _validate_ordered_dates(
            df, schema, "publication_datetime", "effective_start", strict=False
        )

    elif schema is EQUITY_DAILY:
        for column in ("volume", "amount", "market_cap", "free_float_market_cap"):
            if column in df.columns and (df[column].notna() & (df[column] < 0)).any():
                raise ValueError(f"equity_daily: {column} must be non-negative")

    elif schema is RETURNS_DAILY:
        missing_industry = df["industry_return"].isna() | df["beta_industry"].isna()
        undocumented_fallback = missing_industry & ~df["residual_model"].eq(
            "market_only"
        )
        if undocumented_fallback.any():
            raise ValueError(
                "returns_daily: null industry fields require residual_model=market_only"
            )

    elif schema is SIGNAL_DAILY:
        for column in ("customer_neighbor_count", "supplier_neighbor_count"):
            if (df[column].notna() & (df[column] < 0)).any():
                raise ValueError(f"signal_daily: {column} must be non-negative")
        _validate_ordered_dates(df, schema, "graph_snapshot_date", "date", strict=False)


def validate_table(
    df: pd.DataFrame, schema: TableSchema, *, allow_extra: bool = True
) -> pd.DataFrame:
    """Validate a canonical table without silently changing values or dtypes."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{schema.name}: expected a pandas DataFrame")
    if df.columns.has_duplicates:
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{schema.name}: duplicate column names: {duplicated}")

    missing = _missing_columns(df.columns, schema.required_columns)
    if missing:
        raise ValueError(f"{schema.name}: missing required columns: {missing}")
    if not allow_extra:
        declared = set(schema.columns)
        extra = [column for column in df.columns if column not in declared]
        if extra:
            raise ValueError(f"{schema.name}: unexpected columns: {extra}")

    if schema.primary_key:
        key = list(schema.primary_key)
        if df[key].isna().any(axis=None):
            raise ValueError(f"{schema.name}: primary key contains nulls")
        duplicated = df.duplicated(key, keep=False)
        if duplicated.any():
            sample = df.loc[duplicated, key].head(5).to_dict("records")
            raise ValueError(f"{schema.name}: duplicate primary key; sample={sample}")

    _validate_string_columns(df, schema)
    for column in _present_columns(df, schema.date_columns):
        _validate_temporal_column(df, schema, column, date_only=True)
    for column in _present_columns(df, schema.datetime_columns):
        _validate_temporal_column(df, schema, column, date_only=False)
    _validate_numeric_columns(df, schema)
    _validate_boolean_columns(df, schema)
    _validate_domain_rules(df, schema)
    return df


def _coerce_datetime(series: pd.Series, *, column: str) -> pd.Series:
    invalid_type = series.notna() & series.map(
        lambda value: isinstance(value, (bool, Number))
    )
    if invalid_type.any():
        sample = series.loc[invalid_type].head(5).tolist()
        raise ValueError(
            f"disclosure_raw: {column} contains non-date values; sample={sample}"
        )
    try:
        return pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"disclosure_raw: {column} contains invalid dates") from exc


def _coerce_numeric(series: pd.Series, *, column: str) -> pd.Series:
    blank = series.notna() & series.map(
        lambda value: isinstance(value, str) and not value.strip()
    )
    if blank.any():
        raise ValueError(f"disclosure_raw: {column} contains blank numeric values")
    try:
        return pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        sample = series.loc[series.notna()].head(5).tolist()
        raise ValueError(
            f"disclosure_raw: {column} must be numeric; sample={sample}"
        ) from exc


def coerce_disclosure_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce disclosure value fields while rejecting malformed source data.

    Identifiers are intentionally not coerced: a numeric identifier has already
    lost lexical information such as leading zeroes and must fail validation.
    """

    out = df.copy()
    needed = (
        "source_period_end",
        "publication_datetime",
        "exposure_value",
        "exposure_share",
    )
    missing = _missing_columns(out.columns, needed)
    if missing:
        raise ValueError(f"disclosure_raw: missing type-coercion columns: {missing}")

    out["source_period_end"] = _coerce_datetime(
        out["source_period_end"], column="source_period_end"
    )
    out["publication_datetime"] = _coerce_datetime(
        out["publication_datetime"], column="publication_datetime"
    )
    if "source_publication_datetime" in out.columns:
        out["source_publication_datetime"] = _coerce_datetime(
            out["source_publication_datetime"], column="source_publication_datetime"
        )
    if "source_announcement_time_ms" in out.columns:
        out["source_announcement_time_ms"] = _coerce_numeric(
            out["source_announcement_time_ms"], column="source_announcement_time_ms"
        )
    out["exposure_value"] = _coerce_numeric(
        out["exposure_value"], column="exposure_value"
    )
    out["exposure_share"] = _coerce_numeric(
        out["exposure_share"], column="exposure_share"
    )

    invalid_share = out["exposure_share"].notna() & ~out["exposure_share"].between(
        0.0, 1.0, inclusive="both"
    )
    if invalid_share.any():
        sample = out.loc[invalid_share, "exposure_share"].head(5).tolist()
        raise ValueError(
            f"disclosure_raw: exposure_share must be decimal in [0, 1]; sample={sample}"
        )
    invalid_value = out["exposure_value"].notna() & (out["exposure_value"] < 0)
    if invalid_value.any():
        sample = out.loc[invalid_value, "exposure_value"].head(5).tolist()
        raise ValueError(
            f"disclosure_raw: exposure_value must be non-negative; sample={sample}"
        )

    if "relationship_type" in out.columns:
        relation = out["relationship_type"]
        non_string = relation.notna() & ~relation.map(
            lambda value: isinstance(value, str)
        )
        if non_string.any():
            raise ValueError("disclosure_raw: relationship_type must contain strings")
        out["relationship_type"] = relation.str.lower().str.strip()
        invalid = ~out["relationship_type"].isin({"customer", "supplier"})
        if invalid.any():
            bad = sorted(
                out.loc[invalid, "relationship_type"].astype(str).unique().tolist()
            )
            raise ValueError(
                f"disclosure_raw: relationship_type must be customer/supplier; got {bad}"
            )
    return out
