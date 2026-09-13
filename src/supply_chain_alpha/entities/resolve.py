from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rapidfuzz import process
from rapidfuzz.fuzz import ratio

from supply_chain_alpha.data.schemas import (
    COMPANY_ALIAS,
    DISCLOSURE_RAW,
    RESOLUTION_AUDIT,
    validate_table,
)

from .normalize import (
    CounterpartyNameClass,
    classify_counterparty_name,
    normalize_company_name,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolutionConfig:
    fuzzy_enabled: bool = True
    fuzzy_min_score: float = 94.0
    fuzzy_min_margin: float = 4.0


_DEFAULT_RESOLUTION_CONFIG = ResolutionConfig()

RESOLUTION_KEY = (
    "source_document_id",
    "source_company_id",
    "counterparty_raw_name",
)


def attach_resolution_audit(
    mentions: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Attach document/name-level resolution decisions to mention-level rows.

    ``RESOLUTION_AUDIT`` intentionally has no relationship-type key: the same
    name may appear in both the customer and supplier tables of one disclosure.
    Every mention must nevertheless have exactly one matching audit decision.
    """
    validate_table(mentions, DISCLOSURE_RAW)
    validate_table(audit, RESOLUTION_AUDIT)
    attached = mentions.merge(
        audit,
        on=list(RESOLUTION_KEY),
        how="left",
        validate="many_to_one",
        indicator="_resolution_audit_merge",
    )
    unmatched = attached["_resolution_audit_merge"].ne("both")
    if unmatched.any():
        sample = (
            attached.loc[unmatched, list(RESOLUTION_KEY)].head(5).to_dict("records")
        )
        raise ValueError(
            "Every disclosure mention must match one resolution_audit row; "
            f"unmatched={int(unmatched.sum())}; sample={sample}"
        )
    return attached.drop(columns="_resolution_audit_merge")


def _alias_index(alias_df: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, set[str]]:
    """Build an alias index using only aliases valid at the disclosure timestamp."""

    def strict_nullable_dates(series: pd.Series, field: str) -> pd.Series:
        text = series.astype("string")
        present = text.notna() & text.str.strip().ne("")
        try:
            parsed = pd.to_datetime(text.where(present), errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid company_alias.{field} date") from exc
        # Alias bounds are canonical date-only values in the China market
        # calendar.  Localise them explicitly so they can be compared with
        # CNINFO's timezone-aware publication timestamps.
        return parsed.dt.tz_localize("Asia/Shanghai")

    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("Asia/Shanghai")
    else:
        as_of = as_of.tz_convert("Asia/Shanghai")

    valid_from = strict_nullable_dates(alias_df["valid_from"], "valid_from")
    valid_to = strict_nullable_dates(alias_df["valid_to"], "valid_to")
    active = (valid_from.isna() | (valid_from <= as_of)) & (
        valid_to.isna() | (as_of < valid_to)
    )
    active_aliases = alias_df.loc[active]
    index: dict[str, set[str]] = {}
    for row in active_aliases.itertuples(index=False):
        key = normalize_company_name(row.alias)
        if key:
            index.setdefault(key, set()).add(str(row.entity_id))
    return index


def _shanghai_publication_day(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Shanghai")
    else:
        timestamp = timestamp.tz_convert("Asia/Shanghai")
    return timestamp.normalize()


def _load_overrides(path: str | Path | None) -> dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if df.empty:
        return {}
    required = {"counterparty_raw_name", "entity_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Override file missing {required - set(df.columns)}")
    normalized = df["counterparty_raw_name"].map(normalize_company_name)
    if normalized.eq("").any() or df["entity_id"].astype(str).str.strip().eq("").any():
        raise ValueError(
            "Manual overrides require non-empty counterparty_raw_name and entity_id"
        )
    if normalized.duplicated().any():
        duplicates = sorted(normalized[normalized.duplicated(keep=False)].unique())
        raise ValueError(f"Duplicate normalized manual overrides: {duplicates}")
    return dict(zip(normalized, df["entity_id"].astype(str), strict=True))


def resolve_mentions(
    mentions: pd.DataFrame,
    aliases: pd.DataFrame,
    *,
    config: ResolutionConfig = _DEFAULT_RESOLUTION_CONFIG,
    override_path: str | Path | None = None,
) -> pd.DataFrame:
    validate_table(mentions, DISCLOSURE_RAW)
    validate_table(aliases, COMPANY_ALIAS)
    overrides = _load_overrides(override_path)
    known_entities = set(aliases["entity_id"].astype(str))
    unknown_override_entities = sorted(set(overrides.values()) - known_entities)
    if unknown_override_entities:
        raise ValueError(
            f"Manual overrides reference unknown entities: {unknown_override_entities}"
        )
    rows = []
    alias_cache: dict[
        pd.Timestamp,
        tuple[dict[str, set[str]], tuple[str, ...]],
    ] = {}
    fuzzy_cache: dict[
        tuple[pd.Timestamp, str],
        tuple[float, str, float] | None,
    ] = {}

    # Resolution is a document/name-level decision.  The same company can
    # legitimately appear in both the customer and supplier tables of one
    # report, while RESOLUTION_AUDIT intentionally has no relationship-type
    # key.  Collapse only those cross-relation repetitions, after proving the
    # document timing/provenance fields agree.
    resolution_key = list(RESOLUTION_KEY)
    consistency_columns = [
        "source_period_end",
        "publication_datetime",
        "source_document_url_or_path",
    ]
    grouped = mentions.groupby(resolution_key, sort=False, dropna=False)
    for column in consistency_columns:
        conflicts = grouped[column].nunique(dropna=False) > 1
        if conflicts.any():
            sample = conflicts.loc[conflicts].index.tolist()[:5]
            raise ValueError(
                f"Conflicting {column} for repeated document/name mentions: {sample}"
            )
    resolution_mentions = (
        mentions.sort_values([*resolution_key, "relationship_type"], kind="mergesort")
        .drop_duplicates(resolution_key, keep="first")
        .reset_index(drop=True)
    )
    normalized_names = (
        resolution_mentions["counterparty_raw_name"]
        .astype(str)
        .map(normalize_company_name)
    )
    name_classes = (
        resolution_mentions["counterparty_raw_name"]
        .astype(str)
        .map(classify_counterparty_name)
    )
    invalid_names = name_classes.eq(CounterpartyNameClass.INVALID)
    if invalid_names.any():
        sample = resolution_mentions.loc[
            invalid_names,
            [*resolution_key, "relationship_type", "evidence_text"],
        ].head(5)
        raise ValueError(
            "counterparty_raw_name is neither a named nor explicit anonymous "
            f"mention; rows={int(invalid_names.sum())}; "
            f"sample={sample.to_dict('records')}"
        )

    total_mentions = len(resolution_mentions)

    def record_resolution(record: dict[str, object], completed: int) -> None:
        rows.append(record)
        if completed % 1000 == 0 or completed == total_mentions:
            LOGGER.info(
                "Entity resolution progress completed=%d total=%d",
                completed,
                total_mentions,
            )

    for completed, (row, normalized, name_class) in enumerate(
        zip(
            resolution_mentions.itertuples(index=False),
            normalized_names,
            name_classes,
            strict=True,
        ),
        start=1,
    ):
        publication_day = _shanghai_publication_day(row.publication_datetime)
        cached_aliases = alias_cache.get(publication_day)
        if cached_aliases is None:
            idx = _alias_index(aliases, publication_day)
            # RapidFuzz preserves input order for equal scores.  Reverse
            # lexical order exactly reproduces the old reverse sort of
            # ``(score, candidate_name)`` tuples.
            candidate_names = tuple(sorted(idx, reverse=True))
            alias_cache[publication_day] = (idx, candidate_names)
        else:
            idx, candidate_names = cached_aliases
        raw = str(row.counterparty_raw_name)
        base = {
            "source_document_id": str(row.source_document_id),
            "source_company_id": str(row.source_company_id),
            "counterparty_raw_name": raw,
            "normalized_name": (
                normalized if name_class is CounterpartyNameClass.NAMED else None
            ),
        }

        if name_class is CounterpartyNameClass.ANONYMOUS:
            record_resolution(
                {
                    **base,
                    "resolution_status": "anonymous",
                    "resolved_entity_id": None,
                    "resolution_method": "blocked_anonymous",
                    "resolution_confidence": 0.0,
                    "candidate_count": 0,
                    "notes": "anonymous_label_blocked_by_rule",
                },
                completed,
            )
            continue

        exact = idx.get(normalized, set())
        if len(exact) == 1:
            entity_id = next(iter(exact))
            record_resolution(
                {
                    **base,
                    "resolution_status": "resolved",
                    "resolved_entity_id": entity_id,
                    "resolution_method": "exact_alias",
                    "resolution_confidence": 1.0,
                    "candidate_count": 1,
                    "notes": "exact_alias_valid_at_publication",
                },
                completed,
            )
            continue

        # A code-like alias is a deterministic identifier only when the
        # point-in-time alias table contains exactly one matching entity.
        compact_identifier = raw.strip().upper()
        if compact_identifier and (
            compact_identifier.isdigit()
            or compact_identifier.startswith(("SH.", "SZ."))
        ):
            identifier_match = idx.get(
                normalize_company_name(compact_identifier), set()
            )
            if len(identifier_match) == 1:
                record_resolution(
                    {
                        **base,
                        "resolution_status": "resolved",
                        "resolved_entity_id": next(iter(identifier_match)),
                        "resolution_method": "deterministic_identifier",
                        "resolution_confidence": 1.0,
                        "candidate_count": 1,
                        "notes": "unique_identifier_valid_at_publication",
                    },
                    completed,
                )
                continue

        if config.fuzzy_enabled and candidate_names and normalized:
            fuzzy_key = (publication_day, normalized)
            if fuzzy_key not in fuzzy_cache:
                matches = process.extract(
                    normalized,
                    candidate_names,
                    scorer=ratio,
                    limit=2,
                    score_cutoff=max(
                        0.0,
                        config.fuzzy_min_score - config.fuzzy_min_margin,
                    ),
                )
                fuzzy_cache[fuzzy_key] = (
                    (
                        float(matches[0][1]),
                        str(matches[0][0]),
                        float(matches[1][1]) if len(matches) > 1 else 0.0,
                    )
                    if matches
                    else None
                )
            fuzzy_result = fuzzy_cache[fuzzy_key]
            if fuzzy_result is None:
                best_score, best_name, second_score = 0.0, "", 0.0
                entity_ids: set[str] = set()
            else:
                best_score, best_name, second_score = fuzzy_result
                entity_ids = idx[best_name]
            if (
                best_score >= config.fuzzy_min_score
                and (best_score - second_score) >= config.fuzzy_min_margin
                and len(entity_ids) == 1
            ):
                record_resolution(
                    {
                        **base,
                        "resolution_status": "resolved",
                        "resolved_entity_id": next(iter(entity_ids)),
                        "resolution_method": "high_confidence_fuzzy",
                        "resolution_confidence": best_score / 100.0,
                        "candidate_count": 1,
                        "notes": "score_and_uniqueness_margin_passed",
                    },
                    completed,
                )
                continue

        if normalized in overrides:
            record_resolution(
                {
                    **base,
                    "resolution_status": "resolved",
                    "resolved_entity_id": overrides[normalized],
                    "resolution_method": "manual_override",
                    "resolution_confidence": 1.0,
                    "candidate_count": 1,
                    "notes": "version_controlled_manual_override",
                },
                completed,
            )
            continue

        if len(exact) > 1:
            record_resolution(
                {
                    **base,
                    "resolution_status": "ambiguous",
                    "resolved_entity_id": None,
                    "resolution_method": "exact_alias_collision",
                    "resolution_confidence": 0.0,
                    "candidate_count": len(exact),
                    "notes": "ambiguous_exact_alias_not_auto_resolved",
                },
                completed,
            )
        else:
            record_resolution(
                {
                    **base,
                    "resolution_status": "unresolved",
                    "resolved_entity_id": None,
                    "resolution_method": "none",
                    "resolution_confidence": 0.0,
                    "candidate_count": 0,
                    "notes": "no_reproducible_candidate_passed_thresholds",
                },
                completed,
            )

    audit = pd.DataFrame(rows)
    return validate_table(audit, RESOLUTION_AUDIT)
