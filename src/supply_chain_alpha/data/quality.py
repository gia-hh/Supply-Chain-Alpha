"""Deterministic Phase-2 quality audits and promotion gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import process
from rapidfuzz.fuzz import ratio

from supply_chain_alpha.data.cninfo import (
    CNINFO_PUBLICATION_TIMEZONE,
    CNINFO_SIGNAL_CUTOFF_LOCAL,
    RECONCILIATION_ORG_ID_UNIQUE_MASTER_MATCH,
    RECONCILIATION_REPORTED_ID_IN_MASTER,
    RECONCILIATION_REPORTED_ID_IN_MASTER_MISSING_ORG_ID,
    RECONCILIATION_UNRESOLVED_AMBIGUOUS_MASTER_MATCH,
    RECONCILIATION_UNRESOLVED_MISSING_ORG_ID,
    RECONCILIATION_UNRESOLVED_NO_MASTER_MATCH,
    SECURITY_ID_RECONCILIATION_METHODS,
    publication_timing_diagnostics,
)
from supply_chain_alpha.data.schemas import (
    COMPANY_ALIAS,
    SECURITY_MASTER,
    validate_table,
)
from supply_chain_alpha.entities.normalize import (
    CounterpartyNameClass,
    classify_counterparty_name,
    normalize_company_name,
)
from supply_chain_alpha.entities.resolve import ResolutionConfig


@dataclass(frozen=True)
class AuditMetric:
    numerator: int
    denominator: int
    rate: float | None
    status: str


@dataclass(frozen=True)
class ResolutionSupportAudit:
    """Resolution metric plus the independently reconstructed sample evidence."""

    numerator: int
    denominator: int
    rate: float | None
    status: str
    sample_seed: int
    sample_limit: int
    stratification: tuple[str, ...]
    replay_parameters: dict[str, Any]
    sample_details: tuple[dict[str, Any], ...]


_NAMED_RESOLUTION_STATUSES = frozenset({"resolved", "ambiguous", "unresolved"})
_RESOLVED_METHODS = frozenset(
    {
        "exact_alias",
        "deterministic_identifier",
        "high_confidence_fuzzy",
        "manual_override",
    }
)


def _classify_names(series: pd.Series) -> pd.Series:
    """Classify persisted values without turning nulls into the string ``nan``."""

    return series.map(
        lambda value: (
            CounterpartyNameClass.INVALID
            if pd.isna(value)
            else classify_counterparty_name(value)
        )
    )


def _nonblank(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return text.notna() & text.str.strip().ne("")


def _stable_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "<NA>"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _digest(seed: int, *values: object) -> str:
    payload = "\x1f".join([str(seed), *(_stable_value(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deterministic_stratified_sample(
    frame: pd.DataFrame,
    *,
    strata: pd.DataFrame,
    key_columns: tuple[str, ...],
    seed: int,
    sample_size: int,
) -> pd.DataFrame:
    """Take a stable, order-independent sample with representation per stratum.

    Each non-empty stratum receives one slot when the sample can accommodate all
    strata. Remaining slots use deterministic proportional (D'Hondt) allocation.
    Rows inside a stratum are ordered by a seeded digest of caller-supplied,
    frozen identity fields; audited outcomes and future extra columns are excluded.
    """

    if len(frame) <= sample_size:
        return frame.copy()
    if len(strata) != len(frame):
        raise ValueError("strata must align one-to-one with the sampled frame")

    work = frame.reset_index(drop=True).copy()
    aligned_strata = strata.reset_index(drop=True)
    stratum_keys = [
        tuple(_stable_value(value) for value in row)
        for row in aligned_strata.itertuples(index=False, name=None)
    ]
    row_columns = [column for column in key_columns if column in work.columns]
    if not row_columns:
        raise ValueError("No stable sampling-key columns are present")
    row_keys = [
        _digest(seed, *row)
        for row in work[row_columns].itertuples(index=False, name=None)
    ]
    groups: dict[tuple[str, ...], list[int]] = {}
    for position, key in enumerate(stratum_keys):
        groups.setdefault(key, []).append(position)

    ordered_groups = sorted(groups)
    allocations = {key: 0 for key in ordered_groups}
    if len(ordered_groups) <= sample_size:
        for key in ordered_groups:
            allocations[key] = 1
    else:
        chosen = sorted(
            ordered_groups,
            key=lambda key: (_digest(seed, *key), key),
        )[:sample_size]
        for key in chosen:
            allocations[key] = 1

    while sum(allocations.values()) < sample_size:
        eligible = [
            key for key in ordered_groups if allocations[key] < len(groups[key])
        ]
        if not eligible:
            break
        selected = min(
            eligible,
            key=lambda key: (
                -(len(groups[key]) / (allocations[key] + 1)),
                key,
            ),
        )
        allocations[selected] += 1

    selected_positions: list[int] = []
    for key in ordered_groups:
        positions = sorted(groups[key], key=lambda position: row_keys[position])
        selected_positions.extend(positions[: allocations[key]])
    selected_positions.sort(
        key=lambda position: (stratum_keys[position], row_keys[position])
    )
    return work.iloc[selected_positions].reset_index(drop=True)


def _evidence_strata(named: pd.DataFrame) -> pd.DataFrame:
    relationship = (
        named["relationship_type"].astype("string").fillna("<NA>")
        if "relationship_type" in named.columns
        else pd.Series("all", index=named.index, dtype="string")
    )
    year = pd.Series(pd.NA, index=named.index, dtype="Int64")
    for column in ("source_period_end", "publication_datetime"):
        if column not in named.columns:
            continue
        parsed = pd.to_datetime(named[column], errors="coerce")
        candidate = pd.Series(parsed.dt.year, index=named.index, dtype="Int64")
        year = year.fillna(candidate)
    return pd.DataFrame(
        {
            "relationship_type": relationship,
            "source_year": year.astype("string").fillna("<NA>"),
        },
        index=named.index,
    )


def _resolution_boundary_diagnostics(
    mentions: pd.DataFrame,
    resolution_audit: pd.DataFrame,
    *,
    known_entity_ids: set[str],
) -> dict[str, int | bool]:
    mention_classes = _classify_names(mentions["counterparty_raw_name"])
    audit_classes = _classify_names(resolution_audit["counterparty_raw_name"])
    invalid_mentions = int(mention_classes.eq(CounterpartyNameClass.INVALID).sum())
    invalid_audit_names = int(audit_classes.eq(CounterpartyNameClass.INVALID).sum())

    statuses = resolution_audit["resolution_status"].astype("string")
    resolved_ids = _nonblank(resolution_audit["resolved_entity_id"])
    anonymous = audit_classes.eq(CounterpartyNameClass.ANONYMOUS)
    named = audit_classes.eq(CounterpartyNameClass.NAMED)
    mismatch = (
        (anonymous & (statuses.ne("anonymous").fillna(True) | resolved_ids))
        | (named & ~statuses.isin(_NAMED_RESOLUTION_STATUSES))
        | (named & statuses.eq("resolved") & ~resolved_ids)
        | (named & statuses.ne("resolved") & resolved_ids)
    )
    if "resolution_method" in resolution_audit.columns:
        methods = resolution_audit["resolution_method"].astype("string")
        mismatch |= anonymous & methods.ne("blocked_anonymous").fillna(True)
        mismatch |= named & statuses.eq("resolved") & ~methods.isin(_RESOLVED_METHODS)
        mismatch |= (
            named
            & statuses.eq("ambiguous")
            & methods.ne("exact_alias_collision").fillna(True)
        )
        mismatch |= named & statuses.eq("unresolved") & methods.ne("none").fillna(True)
    if "normalized_name" in resolution_audit.columns:
        normalized = resolution_audit["normalized_name"].astype("string")
        expected = resolution_audit["counterparty_raw_name"].map(
            lambda value: "" if pd.isna(value) else normalize_company_name(str(value))
        )
        mismatch |= anonymous & _nonblank(normalized)
        mismatch |= named & normalized.ne(expected).fillna(True)

    key_columns = ["source_document_id", "counterparty_raw_name"]
    if (
        "source_company_id" in mentions.columns
        or "source_company_id" in resolution_audit.columns
    ):
        key_columns.insert(1, "source_company_id")
    missing_mention_keys = sorted(set(key_columns) - set(mentions.columns))
    missing_audit_keys = sorted(set(key_columns) - set(resolution_audit.columns))
    if missing_mention_keys or missing_audit_keys:
        raise ValueError(
            "Phase-2 name boundary requires aligned mention/audit keys; "
            f"mention_missing={missing_mention_keys}; audit_missing={missing_audit_keys}"
        )
    mention_keys = {
        tuple(_stable_value(value) for value in row)
        for row in mentions[key_columns]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    audit_keys = {
        tuple(_stable_value(value) for value in row)
        for row in resolution_audit[key_columns]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    key_mismatches = len(mention_keys.symmetric_difference(audit_keys))
    membership_mismatches = int(
        (
            resolved_ids
            & ~resolution_audit["resolved_entity_id"]
            .astype("string")
            .isin(known_entity_ids)
        ).sum()
    )

    return {
        "invalid_counterparty_mentions": invalid_mentions,
        "invalid_resolution_audit_names": invalid_audit_names,
        "resolution_status_classification_mismatches": int(mismatch.sum()),
        "mention_resolution_key_mismatches": key_mismatches,
        "resolved_entity_membership_mismatches": membership_mismatches,
        "passed": bool(
            invalid_mentions == 0
            and invalid_audit_names == 0
            and not mismatch.any()
            and key_mismatches == 0
            and membership_mismatches == 0
        ),
    }


def _rate(
    numerator: int, denominator: int, *, limited_below: int | None = None
) -> AuditMetric:
    value = numerator / denominator if denominator else None
    status = "NOT_AUDITABLE" if denominator == 0 else "PASS"
    if limited_below is not None and 0 < denominator < limited_below:
        status = "LIMITED_SAMPLE"
    return AuditMetric(numerator, denominator, value, status)


def security_master_qa(
    security_master: pd.DataFrame,
    *,
    expected_security_count: int | None = None,
) -> dict[str, Any]:
    validate_table(security_master, SECURITY_MASTER)
    if expected_security_count is not None and expected_security_count <= 0:
        raise ValueError("expected_security_count must be positive")
    listing_nonnull = int(security_master["listing_date"].notna().sum())
    expected = expected_security_count or len(security_master)
    listing_rate = listing_nonnull / expected if expected else 0.0
    return {
        "row_count": len(security_master),
        "duplicate_security_ids": int(security_master.duplicated("security_id").sum()),
        "listing_date_nonnull_rate_vs_expected": listing_rate,
        "exchange_nonnull_rate": float(security_master["exchange"].notna().mean())
        if len(security_master)
        else 0.0,
        "ticker_all_strings": bool(
            security_master["ticker"].map(lambda value: isinstance(value, str)).all()
        ),
        "passed": bool(
            len(security_master) > 0
            and listing_rate >= 0.95
            and security_master["exchange"].notna().all()
            and security_master["ticker"]
            .map(lambda value: isinstance(value, str))
            .all()
        ),
    }


def disclosure_provenance_qa(documents: pd.DataFrame) -> dict[str, Any]:
    required = {
        "source_document_id",
        "publication_datetime",
        "source_url",
        "local_raw_path",
        "retrieval_datetime",
        "sha256",
    }
    missing = sorted(required - set(documents.columns))
    if missing:
        raise ValueError(f"Disclosure metadata missing columns: {missing}")
    completeness = documents[list(required)].notna().all(axis=1)
    nonblank = (
        documents[list(required)]
        .astype("string")
        .apply(lambda column: column.str.strip().ne(""))
    )
    complete = completeness & nonblank.all(axis=1)
    duplicate_ids = int(documents.duplicated("source_document_id").sum())
    rate = float(complete.mean()) if len(documents) else 0.0
    return {
        "document_count": len(documents),
        "provenance_complete_count": int(complete.sum()),
        "provenance_completeness": rate,
        "duplicate_document_ids": duplicate_ids,
        "passed": bool(len(documents) > 0 and rate == 1.0 and duplicate_ids == 0),
    }


def publication_timing_qa(
    documents: pd.DataFrame,
    mentions: pd.DataFrame,
    *,
    signal_cutoff_local: str = CNINFO_SIGNAL_CUTOFF_LOCAL,
    market_timezone: str = CNINFO_PUBLICATION_TIMEZONE,
) -> dict[str, Any]:
    """Verify conservative CNINFO timing and exact document-to-mention propagation."""

    document_columns = {
        "source_document_id",
        "source_announcement_time_ms",
        "source_publication_datetime",
        "publication_datetime",
        "publication_time_precision",
        "publication_timing_rule",
    }
    missing_documents = sorted(document_columns - set(documents.columns))
    if missing_documents:
        raise ValueError(
            f"CNINFO document timing audit missing columns: {missing_documents}"
        )
    if documents["source_document_id"].duplicated().any():
        raise ValueError("CNINFO document timing audit has duplicate document IDs")
    timing = publication_timing_diagnostics(
        documents,
        epoch_column="source_announcement_time_ms",
        signal_cutoff_local=signal_cutoff_local,
        market_timezone=market_timezone,
    )

    propagation_columns = [
        "source_announcement_time_ms",
        "source_publication_datetime",
        "publication_datetime",
        "publication_time_precision",
        "publication_timing_rule",
    ]
    mention_columns = {"source_document_id", *propagation_columns}
    missing_mentions = sorted(mention_columns - set(mentions.columns))
    if missing_mentions:
        raise ValueError(
            f"Disclosure mention timing audit missing columns: {missing_mentions}"
        )
    expected = documents[["source_document_id", *propagation_columns]].copy()
    observed = mentions[["source_document_id", *propagation_columns]].copy()
    compared = observed.merge(
        expected,
        on="source_document_id",
        how="left",
        suffixes=("_mention", "_document"),
        validate="many_to_one",
        indicator=True,
    )
    mismatch = compared["_merge"].ne("both")
    for column in ("source_publication_datetime", "publication_datetime"):
        mention_values = pd.to_datetime(
            compared[f"{column}_mention"], errors="coerce", utc=True, format="mixed"
        )
        document_values = pd.to_datetime(
            compared[f"{column}_document"], errors="coerce", utc=True, format="mixed"
        )
        mismatch |= mention_values.isna() | document_values.isna()
        mismatch |= mention_values.ne(document_values)
    mention_epoch = pd.to_numeric(
        compared["source_announcement_time_ms_mention"], errors="coerce"
    )
    document_epoch = pd.to_numeric(
        compared["source_announcement_time_ms_document"], errors="coerce"
    )
    mismatch |= mention_epoch.isna() | document_epoch.isna()
    mismatch |= mention_epoch.ne(document_epoch)
    for column in ("publication_time_precision", "publication_timing_rule"):
        mention_values = compared[f"{column}_mention"].astype("string")
        document_values = compared[f"{column}_document"].astype("string")
        mismatch |= mention_values.isna() | document_values.isna()
        mismatch |= mention_values.ne(document_values).fillna(True)
    mismatch_count = int(mismatch.sum())
    return {
        **timing,
        "mention_count": len(mentions),
        "mention_timing_mismatch_count": mismatch_count,
        "passed": bool(timing["passed"] and mismatch_count == 0),
    }


def source_company_identity_qa(
    documents: pd.DataFrame,
    mentions: pd.DataFrame,
    security_master: pd.DataFrame,
) -> dict[str, Any]:
    """Validate CNINFO issuer-ID reconciliation and mention propagation.

    Unresolved identities remain visible rather than being guessed.  Mention
    membership is reported here, while the Phase-3 endpoint gate remains the
    authoritative fail-closed boundary for securities that actually form an
    edge.
    """

    validate_table(security_master, SECURITY_MASTER)
    required = {
        "source_document_id",
        "source_company_id",
        "source_company_reported_id",
        "source_company_org_id",
        "source_company_id_reconciliation_method",
        "source_company_id_reconciliation_candidate_ids",
        "source_company_id_reconciliation_candidate_count",
    }
    missing = sorted(required - set(documents.columns))
    if missing:
        raise ValueError(f"CNINFO document audit missing identity columns: {missing}")
    mention_required = {"source_document_id", "source_company_id"}
    missing_mentions = sorted(mention_required - set(mentions.columns))
    if missing_mentions:
        raise ValueError(
            f"Accepted mentions missing source identity columns: {missing_mentions}"
        )

    master_ids = set(security_master["security_id"].astype(str))
    work = documents.loc[:, sorted(required)].copy()
    methods = work["source_company_id_reconciliation_method"].astype("string")
    canonical = work["source_company_id"].astype("string")
    reported = work["source_company_reported_id"].astype("string")
    org_ids = work["source_company_org_id"].astype("string")
    required_nonblank = (
        _nonblank(work["source_document_id"])
        & _nonblank(canonical)
        & _nonblank(reported)
        & _nonblank(methods)
    )
    missing_org_method = methods.isin(
        {
            RECONCILIATION_REPORTED_ID_IN_MASTER_MISSING_ORG_ID,
            RECONCILIATION_UNRESOLVED_MISSING_ORG_ID,
        }
    )
    org_id_valid = _nonblank(org_ids) | (missing_org_method & org_ids.isna())
    invalid_method = ~methods.isin(SECURITY_ID_RECONCILIATION_METHODS)
    numeric_counts = pd.to_numeric(
        work["source_company_id_reconciliation_candidate_count"], errors="coerce"
    )

    decoded_candidates: list[tuple[str, ...] | None] = []
    malformed_candidates = 0
    for value in work["source_company_id_reconciliation_candidate_ids"]:
        if value is None or pd.isna(value):
            decoded_candidates.append(())
            continue
        try:
            decoded = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded_candidates.append(None)
            malformed_candidates += 1
            continue
        if (
            not isinstance(decoded, list)
            or any(not isinstance(item, str) or not item.strip() for item in decoded)
            or decoded != sorted(set(decoded))
        ):
            decoded_candidates.append(None)
            malformed_candidates += 1
            continue
        decoded_candidates.append(tuple(decoded))

    row_invalid = pd.Series(False, index=work.index)
    row_invalid |= ~required_nonblank | ~org_id_valid | invalid_method.fillna(True)
    for position, index in enumerate(work.index):
        candidates = decoded_candidates[position]
        count = numeric_counts.loc[index]
        if (
            candidates is None
            or pd.isna(count)
            or float(count) < 0
            or not float(count).is_integer()
            or int(count) != len(candidates)
            or any(candidate not in master_ids for candidate in candidates)
        ):
            row_invalid.loc[index] = True
            continue
        method = str(methods.loc[index])
        source_id = str(canonical.loc[index])
        reported_id = str(reported.loc[index])
        if method == RECONCILIATION_REPORTED_ID_IN_MASTER:
            valid = (
                reported_id in master_ids
                and source_id == reported_id
                and bool(str(org_ids.loc[index]).strip())
            )
        elif method == RECONCILIATION_REPORTED_ID_IN_MASTER_MISSING_ORG_ID:
            valid = (
                reported_id in master_ids
                and source_id == reported_id
                and pd.isna(org_ids.loc[index])
                and not candidates
            )
        elif method == RECONCILIATION_ORG_ID_UNIQUE_MASTER_MATCH:
            valid = (
                reported_id not in master_ids
                and source_id in master_ids
                and candidates == (source_id,)
            )
        elif method == RECONCILIATION_UNRESOLVED_NO_MASTER_MATCH:
            valid = (
                reported_id not in master_ids
                and source_id == reported_id
                and not candidates
            )
        elif method == RECONCILIATION_UNRESOLVED_MISSING_ORG_ID:
            valid = (
                reported_id not in master_ids
                and source_id == reported_id
                and pd.isna(org_ids.loc[index])
                and not candidates
            )
        elif method == RECONCILIATION_UNRESOLVED_AMBIGUOUS_MASTER_MATCH:
            valid = (
                reported_id not in master_ids
                and source_id == reported_id
                and len(candidates) > 1
            )
        else:
            valid = False
        row_invalid.loc[index] |= not valid

    consistency = (
        work.assign(
            _candidate_ids=work[
                "source_company_id_reconciliation_candidate_ids"
            ].astype("string")
        )
        .groupby(
            ["source_company_org_id", "source_company_reported_id"],
            dropna=False,
        )[
            [
                "source_company_id",
                "source_company_id_reconciliation_method",
                "source_company_id_reconciliation_candidate_count",
                "_candidate_ids",
            ]
        ]
        .nunique(dropna=False)
    )
    inconsistent_mappings = int(consistency.gt(1).any(axis=1).sum())

    document_identity = work[
        ["source_document_id", "source_company_id"]
    ].drop_duplicates()
    attached = mentions[["source_document_id", "source_company_id"]].merge(
        document_identity,
        on="source_document_id",
        how="left",
        suffixes=("_mention", "_document"),
        validate="many_to_one",
        indicator=True,
    )
    mention_document_mismatches = int(
        (
            attached["_merge"].ne("both")
            | attached["source_company_id_mention"]
            .astype("string")
            .ne(attached["source_company_id_document"].astype("string"))
            .fillna(True)
        ).sum()
    )
    mention_master_mismatches = int(
        (~mentions["source_company_id"].astype("string").isin(master_ids)).sum()
    )
    unresolved_documents = methods.isin(
        {
            RECONCILIATION_UNRESOLVED_NO_MASTER_MATCH,
            RECONCILIATION_UNRESOLVED_MISSING_ORG_ID,
            RECONCILIATION_UNRESOLVED_AMBIGUOUS_MASTER_MATCH,
        }
    )
    passed = bool(
        not row_invalid.any()
        and inconsistent_mappings == 0
        and mention_document_mismatches == 0
    )
    return {
        "document_count": len(documents),
        "invalid_reconciliation_rows": int(row_invalid.sum()),
        "malformed_candidate_sets": malformed_candidates,
        "inconsistent_org_reported_id_mappings": inconsistent_mappings,
        "unresolved_document_count": int(unresolved_documents.sum()),
        "mention_document_identity_mismatches": mention_document_mismatches,
        "mention_security_master_membership_mismatches": mention_master_mismatches,
        "phase3_source_membership_ready": mention_master_mismatches == 0,
        "passed": passed,
    }


def section_coverage_audit(
    documents: pd.DataFrame, mentions: pd.DataFrame
) -> AuditMetric:
    required = {"source_document_id", "contains_relationship_section"}
    if not required.issubset(documents.columns):
        raise ValueError(
            f"Section inventory missing columns: {sorted(required - set(documents.columns))}"
        )
    section_docs = set(
        documents.loc[
            documents["contains_relationship_section"].eq(True), "source_document_id"
        ].astype(str)
    )
    mention_required = {"source_document_id", "counterparty_raw_name"}
    missing_mentions = sorted(mention_required - set(mentions.columns))
    if missing_mentions:
        raise ValueError(f"Accepted mentions missing columns: {missing_mentions}")
    classes = _classify_names(mentions["counterparty_raw_name"])
    accepted = classes.isin(
        {CounterpartyNameClass.NAMED, CounterpartyNameClass.ANONYMOUS}
    )
    captured_docs = section_docs & set(
        mentions.loc[accepted, "source_document_id"].astype(str)
    )
    return _rate(len(captured_docs), len(section_docs))


def evidence_support_audit(
    mentions: pd.DataFrame,
    *,
    seed: int,
    sample_size: int = 200,
) -> AuditMetric:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    classes = _classify_names(mentions["counterparty_raw_name"])
    named = mentions.loc[classes.eq(CounterpartyNameClass.NAMED)].copy()
    if len(named) > sample_size:
        named = _deterministic_stratified_sample(
            named,
            strata=_evidence_strata(named),
            key_columns=(
                "source_document_id",
                "source_company_id",
                "relationship_type",
                "counterparty_raw_name",
                "source_period_end",
                "publication_datetime",
            ),
            seed=seed,
            sample_size=sample_size,
        )
    supported = 0
    for row in named.itertuples(index=False):
        raw = normalize_company_name(str(row.counterparty_raw_name))
        evidence = normalize_company_name(str(row.evidence_text))
        supported += int(bool(raw) and raw in evidence)
    metric = _rate(supported, len(named), limited_below=50)
    if metric.denominator and metric.rate is not None and metric.rate < 0.98:
        return AuditMetric(metric.numerator, metric.denominator, metric.rate, "FAIL")
    return metric


_RESOLUTION_SAMPLE_LIMIT = 200
_RESOLUTION_KEY = (
    "source_document_id",
    "source_company_id",
    "counterparty_raw_name",
)


def _shanghai_publication_day(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("publication_datetime is missing")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Shanghai")
    else:
        timestamp = timestamp.tz_convert("Asia/Shanghai")
    return timestamp.date().isoformat()


def _canonical_alias_bound(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("company_alias contains an invalid validity bound")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai")
    return timestamp.date().isoformat()


def _prepared_alias_evidence(aliases: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize independent point-in-time alias evidence."""

    validate_table(aliases, COMPANY_ALIAS)
    evidence = aliases.copy()
    evidence["normalized_alias_evidence"] = evidence["alias"].map(
        normalize_company_name
    )
    if evidence["normalized_alias_evidence"].eq("").any():
        raise ValueError("company_alias contains an alias with no normalized value")
    evidence["valid_from_evidence"] = evidence["valid_from"].map(_canonical_alias_bound)
    evidence["valid_to_evidence"] = evidence["valid_to"].map(_canonical_alias_bound)
    return evidence


def _active_alias_evidence(
    aliases: pd.DataFrame,
    publication_day: str,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    active_mask = aliases["valid_from_evidence"].fillna("0000-00-00").le(
        publication_day
    ) & aliases["valid_to_evidence"].fillna("9999-99-99").gt(publication_day)
    active = aliases.loc[active_mask]
    entity_index: dict[str, set[str]] = {}
    ordered = active.sort_values(
        ["normalized_alias_evidence", "entity_id", "alias", "valid_from"],
        kind="mergesort",
    )
    for row in ordered.itertuples(index=False):
        normalized = str(row.normalized_alias_evidence)
        entity_index.setdefault(normalized, set()).add(str(row.entity_id))
    return entity_index, ordered


def _alias_source_rows(
    active_aliases: pd.DataFrame,
    normalized_alias: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matching = active_aliases.loc[
        active_aliases["normalized_alias_evidence"].eq(normalized_alias)
    ]
    for row in matching.itertuples(index=False):
        rows.append(
            {
                "entity_id": str(row.entity_id),
                "alias": str(row.alias),
                "alias_type": str(row.alias_type),
                "valid_from": (
                    None
                    if row.valid_from is None or pd.isna(row.valid_from)
                    else str(row.valid_from)
                ),
                "valid_to": (
                    None
                    if row.valid_to is None or pd.isna(row.valid_to)
                    else str(row.valid_to)
                ),
                "source": str(row.source),
            }
        )
    return rows


def _publication_day_index(mentions: pd.DataFrame) -> dict[tuple[str, ...], str | None]:
    required = {*_RESOLUTION_KEY, "publication_datetime"}
    missing = sorted(required - set(mentions.columns))
    if missing:
        raise ValueError(f"mentions missing resolution evidence columns: {missing}")
    days_by_key: dict[tuple[str, ...], set[str]] = {}
    invalid_keys: set[tuple[str, ...]] = set()
    for row in mentions.loc[:, [*_RESOLUTION_KEY, "publication_datetime"]].itertuples(
        index=False, name=None
    ):
        key = tuple(_stable_value(value) for value in row[:-1])
        try:
            day = _shanghai_publication_day(row[-1])
        except (TypeError, ValueError, OverflowError):
            invalid_keys.add(key)
            continue
        days_by_key.setdefault(key, set()).add(day)
    keys = set(days_by_key) | invalid_keys
    return {
        key: (
            next(iter(days_by_key[key]))
            if key not in invalid_keys and len(days_by_key.get(key, set())) == 1
            else None
        )
        for key in keys
    }


def _load_manual_override_evidence(
    override_path: str | Path | None,
) -> dict[str, dict[str, str]]:
    if override_path is None or not Path(override_path).is_file():
        return {}
    overrides = pd.read_csv(override_path, dtype=str, keep_default_na=False)
    if overrides.empty:
        return {}
    required = {"counterparty_raw_name", "entity_id"}
    missing = sorted(required - set(overrides.columns))
    if missing:
        raise ValueError(f"Override file missing columns: {missing}")
    normalized = overrides["counterparty_raw_name"].map(normalize_company_name)
    if normalized.eq("").any() or overrides["entity_id"].str.strip().eq("").any():
        raise ValueError(
            "Manual overrides require non-empty counterparty_raw_name and entity_id"
        )
    if normalized.duplicated().any():
        duplicates = sorted(normalized[normalized.duplicated(keep=False)].unique())
        raise ValueError(f"Duplicate normalized manual overrides: {duplicates}")
    method_column = next(
        (
            column
            for column in ("resolution_method", "method")
            if column in overrides.columns
        ),
        None,
    )
    evidence: dict[str, dict[str, str]] = {}
    for position, row in overrides.iterrows():
        evidence[str(normalized.loc[position])] = {
            "counterparty_raw_name": str(row["counterparty_raw_name"]),
            "entity_id": str(row["entity_id"]).strip(),
            "resolution_method": (
                str(row[method_column]).strip()
                if method_column is not None
                else "manual_override"
            ),
            "reason": str(row["reason"]).strip() if "reason" in row else "",
        }
    return evidence


def _numeric_claim(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if pd.notna(numeric) else None


def resolution_support_audit(
    resolution_audit: pd.DataFrame,
    *,
    aliases: pd.DataFrame,
    mentions: pd.DataFrame,
    seed: int,
    sample_size: int = _RESOLUTION_SAMPLE_LIMIT,
    config: ResolutionConfig | None = None,
    override_path: str | Path | None = None,
    known_entity_ids: set[str] | None = None,
) -> ResolutionSupportAudit:
    """Independently replay sampled mappings from source-supported evidence.

    Persisted resolver claims are compared with, but never substituted for,
    the point-in-time alias rows, recomputed fuzzy ranking, and versioned
    manual-override row used as evidence here.
    """

    if sample_size <= 0 or sample_size > _RESOLUTION_SAMPLE_LIMIT:
        raise ValueError("sample_size must be in [1, 200]")
    resolution_config = config or ResolutionConfig()
    support_columns = {
        *_RESOLUTION_KEY,
        "candidate_count",
        "notes",
        "resolution_confidence",
        "resolution_method",
        "resolution_status",
        "resolved_entity_id",
    }
    missing = sorted(support_columns - set(resolution_audit.columns))
    if missing:
        raise ValueError(f"resolution_audit missing support columns: {missing}")
    alias_evidence = _prepared_alias_evidence(aliases)
    publication_days = _publication_day_index(mentions)
    override_evidence = _load_manual_override_evidence(override_path)
    override_file = Path(override_path) if override_path is not None else None
    override_sha256 = (
        hashlib.sha256(override_file.read_bytes()).hexdigest()
        if override_file is not None and override_file.is_file()
        else None
    )

    accepted = resolution_audit.loc[
        resolution_audit["resolution_status"].eq("resolved")
    ].copy()
    if len(accepted) > sample_size:
        strata = pd.DataFrame(
            {
                "resolution_method": accepted["resolution_method"]
                .astype("string")
                .fillna("<NA>")
            },
            index=accepted.index,
        )
        accepted = _deterministic_stratified_sample(
            accepted,
            strata=strata,
            key_columns=_RESOLUTION_KEY,
            seed=seed,
            sample_size=sample_size,
        )

    # Bound memory even when a stratified sample spans many publication days.
    # Eight entries retain common-day reuse without keeping a full alias index
    # for every one of the at-most-200 sampled mappings.
    active_cache: dict[str, tuple[dict[str, set[str]], pd.DataFrame]] = {}
    sample_details: list[dict[str, Any]] = []
    supported_count = 0
    expected_notes = {
        "exact_alias": "exact_alias_valid_at_publication",
        "deterministic_identifier": "unique_identifier_valid_at_publication",
        "high_confidence_fuzzy": "score_and_uniqueness_margin_passed",
        "manual_override": "version_controlled_manual_override",
    }

    for row in accepted.itertuples(index=False):
        key = tuple(_stable_value(getattr(row, column)) for column in _RESOLUTION_KEY)
        raw_name = (
            "" if pd.isna(row.counterparty_raw_name) else str(row.counterparty_raw_name)
        )
        normalized = normalize_company_name(raw_name)
        method = str(row.resolution_method)
        resolved_entity_id = (
            "" if pd.isna(row.resolved_entity_id) else str(row.resolved_entity_id)
        )
        publication_day = publication_days.get(key)
        failures: list[str] = []
        if classify_counterparty_name(raw_name) is not CounterpartyNameClass.NAMED:
            failures.append("accepted_mapping_is_not_a_named_counterparty")
        if not normalized:
            failures.append("counterparty_name_has_no_normalized_value")
        if publication_day is None:
            failures.append("missing_or_conflicting_publication_datetime")
        if not resolved_entity_id.strip():
            failures.append("resolved_entity_id_is_blank")
        if known_entity_ids is not None and resolved_entity_id not in known_entity_ids:
            failures.append("resolved_entity_id_not_in_security_master")
        if "normalized_name" in accepted.columns:
            claimed_normalized = row.normalized_name
            if pd.isna(claimed_normalized) or str(claimed_normalized) != normalized:
                failures.append("persisted_normalized_name_mismatch")

        entity_index: dict[str, set[str]] = {}
        active_aliases = alias_evidence.iloc[0:0]
        if publication_day is not None:
            if publication_day not in active_cache:
                if len(active_cache) >= 8:
                    active_cache.pop(next(iter(active_cache)))
                active_cache[publication_day] = _active_alias_evidence(
                    alias_evidence, publication_day
                )
            entity_index, active_aliases = active_cache[publication_day]
        exact_entities = sorted(entity_index.get(normalized, set()))
        exact_sources = _alias_source_rows(active_aliases, normalized)

        best_alias: str | None = None
        best_entities: list[str] = []
        best_score: float | None = None
        second_score: float | None = None
        fuzzy_margin: float | None = None
        candidate_names = tuple(sorted(entity_index, reverse=True))
        if normalized and candidate_names:
            fuzzy_matches = process.extract(
                normalized,
                candidate_names,
                scorer=ratio,
                limit=2,
            )
            if fuzzy_matches:
                best_alias = str(fuzzy_matches[0][0])
                best_score = float(fuzzy_matches[0][1])
                second_score = (
                    float(fuzzy_matches[1][1]) if len(fuzzy_matches) > 1 else 0.0
                )
                fuzzy_margin = best_score - second_score
                best_entities = sorted(entity_index[best_alias])
        fuzzy_rule_passes = bool(
            resolution_config.fuzzy_enabled
            and len(best_entities) == 1
            and best_score is not None
            and best_score >= resolution_config.fuzzy_min_score
            and fuzzy_margin is not None
            and fuzzy_margin >= resolution_config.fuzzy_min_margin
        )

        independent_evidence: dict[str, Any] | None = None
        expected_confidence: float | None = None
        expected_candidate_count: int | None = None
        if method == "exact_alias":
            expected_confidence = 1.0
            expected_candidate_count = len(exact_entities)
            independent_evidence = {
                "evidence_type": "active_exact_alias",
                "active_alias_rows": exact_sources,
            }
            if exact_entities != [resolved_entity_id]:
                failures.append("exact_alias_not_unique_for_resolved_entity")
        elif method == "deterministic_identifier":
            expected_confidence = 1.0
            expected_candidate_count = len(exact_entities)
            compact_identifier = raw_name.strip().upper()
            identifier_shape = bool(
                compact_identifier
                and (
                    compact_identifier.isdigit()
                    or compact_identifier.startswith(("SH.", "SZ."))
                )
            )
            independent_evidence = {
                "evidence_type": "active_identifier_alias",
                "identifier_shape_valid": identifier_shape,
                "active_alias_rows": exact_sources,
            }
            if not identifier_shape:
                failures.append("counterparty_is_not_a_supported_identifier_shape")
            if exact_entities != [resolved_entity_id]:
                failures.append("identifier_not_unique_for_resolved_entity")
        elif method == "high_confidence_fuzzy":
            expected_confidence = best_score / 100.0 if best_score is not None else None
            expected_candidate_count = len(best_entities)
            independent_evidence = {
                "evidence_type": "recomputed_point_in_time_fuzzy_ranking",
                "best_alias": best_alias,
                "best_alias_entity_ids": best_entities,
                "best_alias_rows": (
                    _alias_source_rows(active_aliases, best_alias)
                    if best_alias is not None
                    else []
                ),
                "best_score": best_score,
                "second_score": second_score,
                "uniqueness_margin": fuzzy_margin,
                "required_min_score": resolution_config.fuzzy_min_score,
                "required_min_margin": resolution_config.fuzzy_min_margin,
            }
            if not resolution_config.fuzzy_enabled:
                failures.append("fuzzy_matching_disabled_by_frozen_config")
            if len(exact_entities) == 1:
                failures.append("earlier_exact_alias_rule_would_have_resolved")
            if best_entities != [resolved_entity_id]:
                failures.append("fuzzy_top_alias_not_unique_for_resolved_entity")
            if best_score is None or best_score < resolution_config.fuzzy_min_score:
                failures.append("recomputed_fuzzy_score_below_threshold")
            if (
                fuzzy_margin is None
                or fuzzy_margin < resolution_config.fuzzy_min_margin
            ):
                failures.append("recomputed_fuzzy_margin_below_threshold")
        elif method == "manual_override":
            expected_confidence = 1.0
            override = override_evidence.get(normalized)
            expected_candidate_count = 1 if override is not None else 0
            independent_evidence = {
                "evidence_type": "version_controlled_manual_override",
                "override_row": override,
            }
            if override is None:
                failures.append("manual_override_row_missing")
            else:
                if (
                    normalize_company_name(override["counterparty_raw_name"])
                    != normalized
                ):
                    failures.append("manual_override_raw_name_mismatch")
                if override["entity_id"] != resolved_entity_id:
                    failures.append("manual_override_entity_mismatch")
                if override["resolution_method"] != "manual_override":
                    failures.append("manual_override_method_mismatch")
                if not override["reason"]:
                    failures.append("manual_override_reason_missing")
            if len(exact_entities) == 1:
                failures.append("earlier_exact_alias_rule_would_have_resolved")
            if fuzzy_rule_passes:
                failures.append("earlier_fuzzy_rule_would_have_resolved")
        else:
            failures.append("unsupported_resolution_method")

        claimed_confidence = _numeric_claim(row.resolution_confidence)
        claimed_candidate_count = _numeric_claim(row.candidate_count)
        if (
            expected_confidence is None
            or claimed_confidence is None
            or abs(claimed_confidence - expected_confidence) > 1e-12
        ):
            failures.append("persisted_confidence_mismatch")
        if (
            expected_candidate_count is None
            or claimed_candidate_count is None
            or not claimed_candidate_count.is_integer()
            or int(claimed_candidate_count) != expected_candidate_count
        ):
            failures.append("persisted_candidate_count_mismatch")
        if str(row.notes) != expected_notes.get(method):
            failures.append("persisted_audit_note_mismatch")

        supported = not failures
        supported_count += int(supported)
        sample_details.append(
            {
                "source_document_id": key[0],
                "source_company_id": key[1],
                "counterparty_raw_name": raw_name,
                "normalized_name": normalized,
                "publication_day_asia_shanghai": publication_day,
                "resolved_entity_id": resolved_entity_id,
                "resolution_method": method,
                "source_supported": supported,
                "failure_reasons": failures,
                "active_exact_candidate_entity_ids": exact_entities,
                "independent_evidence": independent_evidence,
            }
        )

    metric = _rate(supported_count, len(accepted), limited_below=50)
    status = (
        "FAIL"
        if metric.denominator and metric.rate is not None and metric.rate < 0.98
        else metric.status
    )
    return ResolutionSupportAudit(
        numerator=metric.numerator,
        denominator=metric.denominator,
        rate=metric.rate,
        status=status,
        sample_seed=seed,
        sample_limit=sample_size,
        stratification=("resolution_method",),
        replay_parameters={
            "publication_timezone": "Asia/Shanghai",
            "alias_valid_to_semantics": "exclusive",
            "fuzzy_enabled": resolution_config.fuzzy_enabled,
            "fuzzy_min_score": resolution_config.fuzzy_min_score,
            "fuzzy_min_margin": resolution_config.fuzzy_min_margin,
            "manual_override_file_name": (
                override_file.name if override_file is not None else None
            ),
            "manual_override_file_sha256": override_sha256,
            "manual_override_row_count": len(override_evidence),
        },
        sample_details=tuple(sample_details),
    )


def evaluate_phase2_qa(
    *,
    security_master: pd.DataFrame,
    company_aliases: pd.DataFrame,
    documents: pd.DataFrame,
    mentions: pd.DataFrame,
    resolution_audit: pd.DataFrame,
    manifest_verified: bool,
    seed: int,
    resolution_config: ResolutionConfig | None = None,
    manual_override_path: str | Path | None = None,
    signal_cutoff_local: str = CNINFO_SIGNAL_CUTOFF_LOCAL,
    market_timezone: str = CNINFO_PUBLICATION_TIMEZONE,
) -> dict[str, Any]:
    master = security_master_qa(security_master)
    provenance = disclosure_provenance_qa(documents)
    publication_timing = publication_timing_qa(
        documents,
        mentions,
        signal_cutoff_local=signal_cutoff_local,
        market_timezone=market_timezone,
    )
    source_identity = source_company_identity_qa(documents, mentions, security_master)
    sections = section_coverage_audit(documents, mentions)
    evidence = evidence_support_audit(mentions, seed=seed)
    known_entity_ids = set(security_master["security_id"].astype(str))
    resolution = resolution_support_audit(
        resolution_audit,
        aliases=company_aliases,
        mentions=mentions,
        seed=seed,
        config=resolution_config,
        override_path=manual_override_path,
        known_entity_ids=known_entity_ids,
    )
    name_boundary = _resolution_boundary_diagnostics(
        mentions,
        resolution_audit,
        known_entity_ids=known_entity_ids,
    )
    audit_classes = _classify_names(resolution_audit["counterparty_raw_name"])
    anonymous_resolved = int(
        (
            audit_classes.eq(CounterpartyNameClass.ANONYMOUS)
            & _nonblank(resolution_audit["resolved_entity_id"])
        ).sum()
    )
    resolved_notes = resolution_audit.loc[
        resolution_audit["resolution_status"].eq("resolved"), "notes"
    ].astype("string")
    accepted_without_audit = int(
        (resolved_notes.isna() | resolved_notes.str.strip().eq("")).sum()
    )
    evidence_ok = evidence.status in {"PASS", "LIMITED_SAMPLE"} and (
        evidence.rate is None or evidence.rate >= 0.98
    )
    resolution_ok = resolution.status in {"PASS", "LIMITED_SAMPLE"} and (
        resolution.rate is None or resolution.rate >= 0.98
    )
    passed = all(
        (
            master["passed"],
            provenance["passed"],
            publication_timing["passed"],
            source_identity["passed"],
            sections.rate is not None and sections.rate >= 0.80,
            evidence_ok,
            resolution_ok,
            name_boundary["passed"],
            anonymous_resolved == 0,
            accepted_without_audit == 0,
            manifest_verified,
        )
    )
    return {
        "passed": bool(passed),
        "security_master": master,
        "disclosure_provenance": provenance,
        "publication_timing": publication_timing,
        "source_company_identity": source_identity,
        "section_coverage": asdict(sections),
        "evidence_support": asdict(evidence),
        "resolution_support": asdict(resolution),
        "counterparty_name_boundary": name_boundary,
        "anonymous_to_listed_mappings": anonymous_resolved,
        "accepted_mappings_without_audit_trail": accepted_without_audit,
        "production_raw_manifest_verified": bool(manifest_verified),
    }


__all__ = [
    "AuditMetric",
    "ResolutionSupportAudit",
    "disclosure_provenance_qa",
    "evaluate_phase2_qa",
    "evidence_support_audit",
    "publication_timing_qa",
    "resolution_support_audit",
    "section_coverage_audit",
    "security_master_qa",
    "source_company_identity_qa",
]
