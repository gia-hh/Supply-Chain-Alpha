from __future__ import annotations

from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio

from supply_chain_alpha.data.quality import (
    _deterministic_stratified_sample,
    evaluate_phase2_qa,
    evidence_support_audit,
    publication_timing_qa,
    resolution_support_audit,
    section_coverage_audit,
    source_company_identity_qa,
)
from supply_chain_alpha.entities.resolve import ResolutionConfig


def _aliases(rows: list[tuple[str, str, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entity_id": entity_id,
                "canonical_name": alias,
                "alias": alias,
                "alias_type": "historical_name",
                "valid_from": valid_from,
                "valid_to": valid_to,
                "source": "official point-in-time test evidence",
            }
            for entity_id, alias, valid_from, valid_to in rows
        ]
    )


def _resolution_mentions(audit: pd.DataFrame) -> pd.DataFrame:
    return audit.loc[
        :, ["source_document_id", "source_company_id", "counterparty_raw_name"]
    ].assign(publication_datetime="2021-04-01T18:00:00+08:00")


def test_section_coverage_denominator_is_source_sections() -> None:
    documents = pd.DataFrame(
        {
            "source_document_id": ["D1", "D2", "D3"],
            "contains_relationship_section": [True, True, False],
        }
    )
    mentions = pd.DataFrame(
        {"source_document_id": ["D1"], "counterparty_raw_name": ["华东企业"]}
    )
    result = section_coverage_audit(documents, mentions)
    assert result.numerator == 1
    assert result.denominator == 2
    assert result.rate == 0.5


def test_invalid_name_does_not_count_as_a_captured_section() -> None:
    documents = pd.DataFrame(
        {
            "source_document_id": ["D1", "D2"],
            "contains_relationship_section": [True, True],
        }
    )
    mentions = pd.DataFrame(
        {
            "source_document_id": ["D1", "D2"],
            "counterparty_raw_name": ["供应商A", "不适用"],
        }
    )

    result = section_coverage_audit(documents, mentions)

    assert result.numerator == 1
    assert result.denominator == 2


def test_evidence_support_is_deterministic_and_source_grounded() -> None:
    mentions = pd.DataFrame(
        {
            "counterparty_raw_name": ["华东电池股份有限公司", "华南汽车", "供应商A"],
            "evidence_text": [
                "客户名称：华东电池股份有限公司",
                "无关内容",
                "供应商名称：供应商A",
            ],
        }
    )
    first = evidence_support_audit(mentions, seed=42)
    second = evidence_support_audit(mentions, seed=42)
    assert first == second
    assert first.denominator == 2
    assert first.numerator == 1
    assert first.status == "FAIL"


def test_evidence_sample_is_order_independent_and_stratified_by_relation_year() -> None:
    rows = []
    for index in range(20):
        supported = index < 10
        raw_name = f"华东企业{index}"
        rows.append(
            {
                "counterparty_raw_name": raw_name,
                "evidence_text": raw_name if supported else "无关证据",
                "relationship_type": "customer" if supported else "supplier",
                "source_period_end": "2020-12-31" if supported else "2021-12-31",
            }
        )
    mentions = pd.DataFrame(rows)

    forward = evidence_support_audit(mentions, seed=17, sample_size=2)
    reversed_rows = evidence_support_audit(
        mentions.iloc[::-1].reset_index(drop=True), seed=17, sample_size=2
    )

    assert forward == reversed_rows
    assert forward.denominator == 2
    assert forward.numerator == 1


def test_sampling_identity_ignores_row_order_outcomes_and_extra_columns() -> None:
    frame = pd.DataFrame(
        {
            "source_document_id": [f"D{index:02d}" for index in range(20)],
            "counterparty_raw_name": [f"华东企业{index}" for index in range(20)],
            "evidence_text": [
                "supported" if index % 2 else "unsupported" for index in range(20)
            ],
        }
    )
    strata = pd.DataFrame({"relationship_type": ["customer"] * len(frame)})
    selected = _deterministic_stratified_sample(
        frame,
        strata=strata,
        key_columns=("source_document_id", "counterparty_raw_name"),
        seed=17,
        sample_size=5,
    )

    reordered = frame.iloc[::-1].reset_index(drop=True).copy()
    reordered["unrelated_future_column"] = range(len(reordered))
    selected_after_schema_change = _deterministic_stratified_sample(
        reordered,
        strata=strata,
        key_columns=("source_document_id", "counterparty_raw_name"),
        seed=17,
        sample_size=5,
    )

    assert set(selected["source_document_id"]) == set(
        selected_after_schema_change["source_document_id"]
    )


def test_resolution_sample_is_order_independent_and_stratified_by_method() -> None:
    exact_names = [f"华东企业{index}" for index in range(10)]
    fuzzy_pairs = [
        (f"{chr(97 + index)}" * 30 + "x", f"{chr(97 + index)}" * 30 + "y")
        for index in range(10)
    ]
    raw_names = [*exact_names, *(raw for raw, _alias in fuzzy_pairs)]
    alias_names = [*exact_names, *(alias for _raw, alias in fuzzy_pairs)]
    accepted = pd.DataFrame(
        {
            "source_document_id": [f"D{index}" for index in range(20)],
            "source_company_id": ["S1"] * 20,
            "counterparty_raw_name": raw_names,
            "resolution_status": ["resolved"] * 20,
            "resolution_method": ["exact_alias"] * 10 + ["high_confidence_fuzzy"] * 10,
            "resolution_confidence": [1.0] * 10
            + [ratio(raw, alias) / 100.0 for raw, alias in fuzzy_pairs],
            "candidate_count": [1] * 20,
            "resolved_entity_id": [f"S{index}" for index in range(20)],
            "notes": ["exact_alias_valid_at_publication"] * 10
            + ["score_and_uniqueness_margin_passed"] * 10,
        }
    )
    aliases = _aliases(
        [
            (f"S{index}", alias, "2010-01-01", None)
            for index, alias in enumerate(alias_names)
        ]
    )
    mentions = _resolution_mentions(accepted)

    forward = resolution_support_audit(
        accepted,
        aliases=aliases,
        mentions=mentions,
        seed=23,
        sample_size=2,
    )
    shuffled = resolution_support_audit(
        accepted.sample(frac=1, random_state=99).reset_index(drop=True),
        aliases=aliases.sample(frac=1, random_state=7).reset_index(drop=True),
        mentions=mentions.sample(frac=1, random_state=8).reset_index(drop=True),
        seed=23,
        sample_size=2,
    )

    assert forward == shuffled
    assert forward.denominator == 2
    assert forward.numerator == 2


def test_resolution_support_requires_deterministic_stored_rule_evidence() -> None:
    accepted = pd.DataFrame(
        {
            "source_document_id": ["D1", "D2"],
            "source_company_id": ["S0", "S0"],
            "counterparty_raw_name": ["华东科技", "华南科技"],
            "resolution_status": ["resolved", "resolved"],
            "resolution_method": ["exact_alias", "exact_alias"],
            "resolution_confidence": [1.0, 1.0],
            "candidate_count": [1, 1],
            "resolved_entity_id": ["S1", "S2"],
            "notes": ["exact_alias_valid_at_publication", "unsupported claim"],
        }
    )
    aliases = _aliases(
        [
            ("S1", "华东科技", "2010-01-01", None),
            ("S2", "华南科技", "2010-01-01", None),
        ]
    )

    result = resolution_support_audit(
        accepted,
        aliases=aliases,
        mentions=_resolution_mentions(accepted),
        seed=23,
    )

    assert result.numerator == 1
    assert result.denominator == 2
    assert result.status == "FAIL"


def test_resolution_support_enforces_fuzzy_floor_and_master_membership() -> None:
    fuzzy_raw = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaax"
    fuzzy_alias = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaay"
    accepted = pd.DataFrame(
        {
            "source_document_id": ["D1", "D2"],
            "source_company_id": ["S0", "S0"],
            "counterparty_raw_name": [fuzzy_raw, "华南科技"],
            "resolution_status": ["resolved", "resolved"],
            "resolution_method": ["high_confidence_fuzzy", "exact_alias"],
            "resolution_confidence": [0.01, 1.0],
            "candidate_count": [1, 1],
            "resolved_entity_id": ["S1", "UNKNOWN"],
            "notes": [
                "score_and_uniqueness_margin_passed",
                "exact_alias_valid_at_publication",
            ],
        }
    )

    result = resolution_support_audit(
        accepted,
        aliases=_aliases(
            [
                ("S1", fuzzy_alias, "2010-01-01", None),
                ("S2", "华南科技", "2010-01-01", None),
            ]
        ),
        mentions=_resolution_mentions(accepted),
        seed=23,
        known_entity_ids={"S1", "S2"},
    )

    assert result.numerator == 0
    assert result.denominator == 2
    assert result.status == "FAIL"


def test_resolution_support_rejects_exact_alias_ambiguity_despite_claims() -> None:
    accepted = pd.DataFrame(
        {
            "source_document_id": ["D1"],
            "source_company_id": ["S0"],
            "counterparty_raw_name": ["同名科技有限公司"],
            "resolution_status": ["resolved"],
            "resolution_method": ["exact_alias"],
            "resolution_confidence": [1.0],
            "candidate_count": [1],
            "resolved_entity_id": ["S1"],
            "notes": ["exact_alias_valid_at_publication"],
        }
    )
    result = resolution_support_audit(
        accepted,
        aliases=_aliases(
            [
                ("S1", "同名科技有限公司", "2010-01-01", None),
                ("S2", "同名科技有限公司", "2010-01-01", None),
            ]
        ),
        mentions=_resolution_mentions(accepted),
        seed=42,
    )

    assert result.numerator == 0
    assert result.sample_details[0]["source_supported"] is False
    assert (
        "exact_alias_not_unique_for_resolved_entity"
        in result.sample_details[0]["failure_reasons"]
    )


def test_resolution_support_recomputes_fuzzy_uniqueness_not_claimed_margin() -> None:
    raw = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaax"
    accepted = pd.DataFrame(
        {
            "source_document_id": ["D1"],
            "source_company_id": ["S0"],
            "counterparty_raw_name": [raw],
            "resolution_status": ["resolved"],
            "resolution_method": ["high_confidence_fuzzy"],
            "resolution_confidence": [ratio(raw, "a" * 30 + "y") / 100.0],
            "candidate_count": [1],
            "resolved_entity_id": ["S1"],
            "notes": ["score_and_uniqueness_margin_passed"],
        }
    )
    result = resolution_support_audit(
        accepted,
        aliases=_aliases(
            [
                ("S1", "a" * 30 + "y", "2010-01-01", None),
                ("S2", "a" * 30 + "z", "2010-01-01", None),
            ]
        ),
        mentions=_resolution_mentions(accepted),
        config=ResolutionConfig(fuzzy_min_score=94.0, fuzzy_min_margin=4.0),
        seed=42,
    )

    assert result.numerator == 0
    assert (
        "recomputed_fuzzy_margin_below_threshold"
        in result.sample_details[0]["failure_reasons"]
    )


def test_resolution_support_validates_active_identifier_evidence() -> None:
    accepted = pd.DataFrame(
        {
            "source_document_id": ["D1"],
            "source_company_id": ["S0"],
            "counterparty_raw_name": ["SH.600001"],
            "resolution_status": ["resolved"],
            "resolution_method": ["deterministic_identifier"],
            "resolution_confidence": [1.0],
            "candidate_count": [1],
            "resolved_entity_id": ["S1"],
            "notes": ["unique_identifier_valid_at_publication"],
        }
    )
    result = resolution_support_audit(
        accepted,
        aliases=_aliases([("S1", "SH.600001", "2010-01-01", None)]),
        mentions=_resolution_mentions(accepted),
        seed=42,
    )

    assert result.numerator == 1
    assert result.sample_details[0]["source_supported"] is True


def test_resolution_support_requires_versioned_override_entity_method_and_reason(
    tmp_path: Path,
) -> None:
    accepted = pd.DataFrame(
        {
            "source_document_id": ["D1"],
            "source_company_id": ["S0"],
            "counterparty_raw_name": ["华东特殊科技有限公司"],
            "resolution_status": ["resolved"],
            "resolution_method": ["manual_override"],
            "resolution_confidence": [1.0],
            "candidate_count": [1],
            "resolved_entity_id": ["S1"],
            "notes": ["version_controlled_manual_override"],
        }
    )
    override_path = tmp_path / "entity_overrides.csv"
    pd.DataFrame(
        {
            "counterparty_raw_name": ["华东特殊科技"],
            "entity_id": ["S1"],
            "method": ["manual_override"],
            "reason": ["official merger filing reviewed"],
        }
    ).to_csv(override_path, index=False)
    aliases = _aliases([("S1", "无关历史名称", "2010-01-01", None)])
    supported = resolution_support_audit(
        accepted,
        aliases=aliases,
        mentions=_resolution_mentions(accepted),
        override_path=override_path,
        seed=42,
    )
    assert supported.numerator == 1
    assert supported.replay_parameters["manual_override_row_count"] == 1
    assert len(supported.replay_parameters["manual_override_file_sha256"]) == 64
    assert (
        supported.sample_details[0]["independent_evidence"]["override_row"]["reason"]
        == "official merger filing reviewed"
    )

    pd.DataFrame(
        {
            "counterparty_raw_name": ["华东特殊科技"],
            "entity_id": ["S1"],
            "method": ["manual_override"],
            "reason": [""],
        }
    ).to_csv(override_path, index=False)
    unsupported = resolution_support_audit(
        accepted,
        aliases=aliases,
        mentions=_resolution_mentions(accepted),
        override_path=override_path,
        seed=42,
    )
    assert unsupported.numerator == 0
    assert (
        "manual_override_reason_missing"
        in unsupported.sample_details[0]["failure_reasons"]
    )


def test_resolution_support_ignores_aliases_not_yet_valid_at_publication() -> None:
    accepted = pd.DataFrame(
        {
            "source_document_id": ["D1"],
            "source_company_id": ["S0"],
            "counterparty_raw_name": ["华东科技有限公司"],
            "resolution_status": ["resolved"],
            "resolution_method": ["exact_alias"],
            "resolution_confidence": [1.0],
            "candidate_count": [1],
            "resolved_entity_id": ["S1"],
            "notes": ["exact_alias_valid_at_publication"],
        }
    )
    baseline_aliases = _aliases([("S1", "华东科技有限公司", "2010-01-01", None)])
    future_appended = pd.concat(
        [
            baseline_aliases,
            _aliases([("S2", "华东科技有限公司", "2022-01-01", None)]),
        ],
        ignore_index=True,
    )

    baseline = resolution_support_audit(
        accepted,
        aliases=baseline_aliases,
        mentions=_resolution_mentions(accepted),
        seed=42,
    )
    after_append = resolution_support_audit(
        accepted,
        aliases=future_appended,
        mentions=_resolution_mentions(accepted),
        seed=42,
    )

    assert baseline.numerator == after_append.numerator == 1
    assert baseline.sample_details == after_append.sample_details


def _phase2_boundary_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    security_master = pd.DataFrame(
        {
            "security_id": ["S1", "S2"],
            "ticker": ["600001", "600002"],
            "exchange": ["SSE", "SSE"],
            "company_name": ["甲公司", "乙公司"],
            "listing_date": ["2010-01-01", "2010-01-01"],
            "delisting_date": [None, None],
            "board": ["MAIN", "MAIN"],
        }
    )
    documents = pd.DataFrame(
        {
            "source_document_id": ["D1"],
            "source_company_id": ["S1"],
            "source_company_reported_id": ["S1"],
            "source_company_org_id": ["ORG-S1"],
            "source_company_id_reconciliation_method": ["reported_id_in_master"],
            "source_company_id_reconciliation_candidate_ids": ['["S1"]'],
            "source_company_id_reconciliation_candidate_count": [1],
            "source_announcement_time_ms": [1617271200000],
            "source_publication_datetime": ["2021-04-01 18:00:00+08:00"],
            "publication_datetime": ["2021-04-01 18:00:00+08:00"],
            "publication_time_precision": ["exact"],
            "publication_timing_rule": ["cninfo_source_timestamp"],
            "source_url": ["https://example.test/D1.pdf"],
            "local_raw_path": ["raw/D1.pdf"],
            "retrieval_datetime": ["2021-04-02 00:00:00+00:00"],
            "sha256": ["a" * 64],
            "contains_relationship_section": [True],
        }
    )
    mentions = pd.DataFrame(
        {
            "source_document_id": ["D1", "D1"],
            "source_company_id": ["S1", "S1"],
            "counterparty_raw_name": ["华东科技有限公司", "供应商A"],
            "evidence_text": [
                "客户名称：华东科技有限公司",
                "供应商名称：供应商A",
            ],
            "relationship_type": ["customer", "supplier"],
            "source_period_end": ["2020-12-31", "2020-12-31"],
            "source_announcement_time_ms": [1617271200000, 1617271200000],
            "source_publication_datetime": [
                "2021-04-01 18:00:00+08:00",
                "2021-04-01 18:00:00+08:00",
            ],
            "publication_datetime": [
                "2021-04-01 18:00:00+08:00",
                "2021-04-01 18:00:00+08:00",
            ],
            "publication_time_precision": ["exact", "exact"],
            "publication_timing_rule": [
                "cninfo_source_timestamp",
                "cninfo_source_timestamp",
            ],
        }
    )
    audit = pd.DataFrame(
        {
            "source_document_id": ["D1", "D1"],
            "source_company_id": ["S1", "S1"],
            "counterparty_raw_name": ["华东科技有限公司", "供应商A"],
            "normalized_name": ["华东科技", None],
            "resolution_status": ["resolved", "anonymous"],
            "resolved_entity_id": ["S2", None],
            "resolution_method": ["exact_alias", "blocked_anonymous"],
            "resolution_confidence": [1.0, 0.0],
            "candidate_count": [1, 0],
            "notes": [
                "exact_alias_valid_at_publication",
                "anonymous_label_blocked_by_rule",
            ],
        }
    )
    return security_master, documents, mentions, audit


def _phase2_aliases() -> pd.DataFrame:
    return _aliases(
        [
            ("S1", "甲公司", "2010-01-01", None),
            ("S2", "华东科技有限公司", "2010-01-01", None),
        ]
    )


def test_publication_timing_qa_accepts_exact_and_date_only_conservative_rules() -> None:
    _, documents, mentions, _ = _phase2_boundary_inputs()
    exact = publication_timing_qa(documents, mentions)
    assert exact["passed"] is True
    assert exact["exact_source_timestamp_after_signal_cutoff_count"] == 1

    midnight_ms = int(pd.Timestamp("2021-04-01T00:00:00+08:00").timestamp() * 1000)
    documents.loc[0, "source_announcement_time_ms"] = midnight_ms
    documents.loc[0, "source_publication_datetime"] = "2021-04-01T00:00:00+08:00"
    documents.loc[0, "publication_datetime"] = "2021-04-02T00:00:00+08:00"
    documents.loc[0, "publication_time_precision"] = "date_only"
    documents.loc[0, "publication_timing_rule"] = "next_calendar_day_midnight"
    for column in (
        "source_announcement_time_ms",
        "source_publication_datetime",
        "publication_datetime",
        "publication_time_precision",
        "publication_timing_rule",
    ):
        mentions.loc[:, column] = documents.loc[0, column]

    date_only = publication_timing_qa(documents, mentions)
    assert date_only["passed"] is True
    assert date_only["date_only_source_timestamp_count"] == 1
    assert date_only["conservatively_shifted_count"] == 1


def test_publication_timing_qa_rejects_mention_drift_and_same_day_date_only() -> None:
    _, documents, mentions, _ = _phase2_boundary_inputs()
    mentions.loc[0, "publication_datetime"] = "2021-04-02T00:00:00+08:00"
    drift = publication_timing_qa(documents, mentions)
    assert drift["mention_timing_mismatch_count"] == 1
    assert drift["passed"] is False

    midnight_ms = int(pd.Timestamp("2021-04-01T00:00:00+08:00").timestamp() * 1000)
    documents.loc[0, "source_announcement_time_ms"] = midnight_ms
    documents.loc[0, "source_publication_datetime"] = "2021-04-01T00:00:00+08:00"
    documents.loc[0, "publication_datetime"] = "2021-04-01T15:00:00+08:00"
    documents.loc[0, "publication_time_precision"] = "date_only"
    documents.loc[0, "publication_timing_rule"] = "next_calendar_day_midnight"
    for column in (
        "source_announcement_time_ms",
        "source_publication_datetime",
        "publication_datetime",
        "publication_time_precision",
        "publication_timing_rule",
    ):
        mentions.loc[:, column] = documents.loc[0, column]
    early = publication_timing_qa(documents, mentions)
    assert early["publication_timing_violation_count"] == 1
    assert early["passed"] is False


def test_phase2_boundary_fails_invalid_names_and_persisted_status_mismatch() -> None:
    security_master, documents, mentions, audit = _phase2_boundary_inputs()
    mentions.loc[1, "counterparty_raw_name"] = "不适用"
    audit.loc[1, "counterparty_raw_name"] = "不适用"
    audit.loc[0, "resolution_status"] = "anonymous"
    audit.loc[0, "resolved_entity_id"] = None
    audit.loc[0, "resolution_method"] = "blocked_anonymous"

    result = evaluate_phase2_qa(
        security_master=security_master,
        company_aliases=_phase2_aliases(),
        documents=documents,
        mentions=mentions,
        resolution_audit=audit,
        manifest_verified=True,
        seed=42,
    )

    boundary = result["counterparty_name_boundary"]
    assert boundary["invalid_counterparty_mentions"] == 1
    assert boundary["invalid_resolution_audit_names"] == 1
    assert boundary["resolution_status_classification_mismatches"] == 1
    assert boundary["mention_resolution_key_mismatches"] == 0
    assert boundary["resolved_entity_membership_mismatches"] == 0
    assert boundary["passed"] is False
    assert result["passed"] is False


def test_phase2_boundary_requires_exact_mention_resolution_key_alignment() -> None:
    security_master, documents, mentions, audit = _phase2_boundary_inputs()
    audit.loc[0, "source_document_id"] = "D-UNRELATED"

    result = evaluate_phase2_qa(
        security_master=security_master,
        company_aliases=_phase2_aliases(),
        documents=documents,
        mentions=mentions,
        resolution_audit=audit,
        manifest_verified=True,
        seed=42,
    )

    assert (
        result["counterparty_name_boundary"]["mention_resolution_key_mismatches"] == 2
    )
    assert result["passed"] is False


def test_source_identity_reports_unknown_mentions_for_phase3_membership_gate() -> None:
    security_master, documents, mentions, _ = _phase2_boundary_inputs()
    missing_org = documents.copy()
    missing_org.loc[0, "source_company_org_id"] = None
    missing_org.loc[0, "source_company_id_reconciliation_method"] = (
        "reported_id_in_master_missing_org_id"
    )
    missing_org.loc[0, "source_company_id_reconciliation_candidate_ids"] = None
    missing_org.loc[0, "source_company_id_reconciliation_candidate_count"] = 0
    missing_org_result = source_company_identity_qa(
        missing_org, mentions, security_master
    )
    assert missing_org_result["passed"] is True

    documents.loc[0, "source_company_id"] = "UNKNOWN"
    documents.loc[0, "source_company_reported_id"] = "UNKNOWN"
    documents.loc[0, "source_company_id_reconciliation_method"] = (
        "unresolved_no_master_match"
    )
    documents.loc[0, "source_company_id_reconciliation_candidate_ids"] = None
    documents.loc[0, "source_company_id_reconciliation_candidate_count"] = 0

    no_mentions = source_company_identity_qa(
        documents, mentions.iloc[0:0], security_master
    )
    assert no_mentions["unresolved_document_count"] == 1
    assert no_mentions["passed"] is True

    mentions.loc[:, "source_company_id"] = "UNKNOWN"
    with_mentions = source_company_identity_qa(documents, mentions, security_master)
    assert with_mentions["mention_security_master_membership_mismatches"] == 2
    assert with_mentions["phase3_source_membership_ready"] is False
    assert with_mentions["passed"] is True


def test_phase2_boundary_rejects_unknown_resolved_entity() -> None:
    security_master, documents, mentions, audit = _phase2_boundary_inputs()
    audit.loc[0, "resolved_entity_id"] = "UNKNOWN"

    result = evaluate_phase2_qa(
        security_master=security_master,
        company_aliases=_phase2_aliases(),
        documents=documents,
        mentions=mentions,
        resolution_audit=audit,
        manifest_verified=True,
        seed=42,
    )

    assert (
        result["counterparty_name_boundary"]["resolved_entity_membership_mismatches"]
        == 1
    )
    assert result["passed"] is False


def test_phase2_boundary_requires_canonical_status_method_and_normalized_name() -> None:
    security_master, documents, mentions, audit = _phase2_boundary_inputs()
    baseline = evaluate_phase2_qa(
        security_master=security_master,
        company_aliases=_phase2_aliases(),
        documents=documents,
        mentions=mentions,
        resolution_audit=audit,
        manifest_verified=True,
        seed=42,
    )
    assert baseline["counterparty_name_boundary"]["passed"] is True
    assert baseline["passed"] is True

    for column, value in (
        ("resolution_status", "resolved "),
        ("resolution_method", "exact_alias "),
        ("normalized_name", "wrong-name"),
    ):
        mutated = audit.copy()
        mutated.loc[0, column] = value
        result = evaluate_phase2_qa(
            security_master=security_master,
            company_aliases=_phase2_aliases(),
            documents=documents,
            mentions=mentions,
            resolution_audit=mutated,
            manifest_verified=True,
            seed=42,
        )
        assert (
            result["counterparty_name_boundary"][
                "resolution_status_classification_mismatches"
            ]
            == 1
        )
        assert result["passed"] is False


def test_phase2_anonymous_binding_fails_even_when_status_says_anonymous() -> None:
    security_master, documents, mentions, audit = _phase2_boundary_inputs()
    audit.loc[1, "resolved_entity_id"] = "S2"

    result = evaluate_phase2_qa(
        security_master=security_master,
        company_aliases=_phase2_aliases(),
        documents=documents,
        mentions=mentions,
        resolution_audit=audit,
        manifest_verified=True,
        seed=42,
    )

    assert result["anonymous_to_listed_mappings"] == 1
    assert (
        result["counterparty_name_boundary"][
            "resolution_status_classification_mismatches"
        ]
        == 1
    )
    assert result["passed"] is False
