import logging
from pathlib import Path

import pandas as pd
import pytest
from rapidfuzz.fuzz import ratio

from supply_chain_alpha.data.disclosures import load_structured_mentions
from supply_chain_alpha.entities import resolve as resolve_module
from supply_chain_alpha.entities.normalize import (
    CounterpartyNameClass,
    classify_counterparty_name,
    is_anonymous_counterparty,
    normalize_company_name,
)
from supply_chain_alpha.entities.resolve import ResolutionConfig, resolve_mentions

FIXTURES = Path(__file__).parents[2] / "fixtures"


def test_anonymous_patterns_are_blocked():
    for name in [
        "Customer A",
        "Customer 1",
        "Supplier A",
        "客户A",
        "供应商A",
        "某客户",
        "第一名客户",
        "第一大供应商",
        "第1名客户",
        "A供应商",
        "第一名",
        "第三位",
        "A",
        "J",
        "A 公司",
        "公司五",
        "单位-4",
        "法人二",
        "自然人三",
        "客户第一名",
        "供应商(二)",
        "销售客户一",
        "经销商客户 A",
        "收入第五名",
        "保密单位一",
        "同一实际控制人控制的供应商",
        "客戶一",
        "上海**有限公司",
        "上海某公司",
        "上海某有限公司",
        "-",
        "--",
        "*****",
    ]:
        assert is_anonymous_counterparty(name)


@pytest.mark.parametrize(
    "name",
    [
        "/",
        "”。",
        "无",
        "不适用",
        "未披露",
        "不适用/",
        "无/不适用",
        "未披露、未知",
        "N/A / none",
    ],
)
def test_invalid_counterparty_names_are_not_anonymous(name: str) -> None:
    assert classify_counterparty_name(name) is CounterpartyNameClass.INVALID
    assert not is_anonymous_counterparty(name)


def test_missing_token_rule_does_not_reject_a_real_name_prefix() -> None:
    assert (
        classify_counterparty_name("无锡产业发展集团有限公司")
        is CounterpartyNameClass.NAMED
    )


@pytest.mark.parametrize(
    "name",
    [
        "*ST华源",
        "*ST 华源",
        "S*ST佳通",
        "1524*1067mm",
        "12 × 24 型材",
    ],
)
def test_market_prefixes_and_numeric_multiplication_are_named(name: str) -> None:
    assert classify_counterparty_name(name) is CounterpartyNameClass.NAMED


@pytest.mark.parametrize(
    "name",
    [
        "宁波某某科技有限公司",
        "Customer Service Holdings Ltd",
        "Supplier Dynamics Ltd",
        "Company Zeta Inc",
        "Customer K",
    ],
)
def test_realistic_some_names_and_english_names_are_not_placeholder_codes(
    name: str,
) -> None:
    assert classify_counterparty_name(name) is CounterpartyNameClass.NAMED


def test_name_normalization_removes_common_suffix():
    assert normalize_company_name("华东材料股份有限公司") == normalize_company_name(
        "华东材料"
    )


@pytest.mark.parametrize("invalid_name", ["”。", "/", "无", "不适用"])
def test_resolution_rejects_invalid_names_before_matching(invalid_name: str):
    mentions = (
        load_structured_mentions(FIXTURES / "disclosure_mentions.csv").iloc[[0]].copy()
    )
    mentions.loc[:, "counterparty_raw_name"] = invalid_name
    mentions.loc[:, "evidence_text"] = f"2) {invalid_name}"
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)

    with pytest.raises(ValueError, match="neither a named nor explicit anonymous"):
        resolve_mentions(mentions, aliases)


@pytest.mark.parametrize("redacted_name", ["-", "*****"])
def test_redacted_counterparty_is_retained_as_blocked_anonymous(
    redacted_name: str,
):
    mentions = (
        load_structured_mentions(FIXTURES / "disclosure_mentions.csv").iloc[[0]].copy()
    )
    mentions.loc[:, "counterparty_raw_name"] = redacted_name
    mentions.loc[:, "evidence_text"] = f"1 {redacted_name} 4,375.11 36.52"
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)

    audit = resolve_mentions(mentions, aliases)

    row = audit.iloc[0]
    assert row["resolution_status"] == "anonymous"
    assert row["resolution_method"] == "blocked_anonymous"
    assert pd.isna(row["normalized_name"])
    assert pd.isna(row["resolved_entity_id"])


def test_resolution_is_conservative():
    mentions = load_structured_mentions(FIXTURES / "disclosure_mentions.csv")
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    audit = resolve_mentions(mentions, aliases)
    status = dict(zip(audit["counterparty_raw_name"], audit["resolution_status"]))
    assert status["华东电池股份有限公司"] == "resolved"
    assert status["华南汽车"] == "resolved"
    assert status["供应商A"] == "anonymous"
    assert status["不存在公司"] == "unresolved"


def test_future_alias_is_not_used_for_past_disclosure():
    mentions = (
        load_structured_mentions(FIXTURES / "disclosure_mentions.csv").iloc[[4]].copy()
    )
    mentions.loc[:, "counterparty_raw_name"] = "未来名称"
    mentions.loc[:, "publication_datetime"] = pd.Timestamp("2020-04-28 18:00:00")
    aliases = pd.DataFrame(
        [
            {
                "entity_id": "S5",
                "canonical_name": "新科电子股份有限公司",
                "alias": "未来名称",
                "alias_type": "former_or_future_name",
                "valid_from": "2021-01-01",
                "valid_to": None,
                "source": "fixture",
            }
        ]
    )
    audit = resolve_mentions(mentions, aliases)
    assert audit.iloc[0]["resolution_status"] == "unresolved"


def test_canonical_display_name_does_not_bypass_alias_validity():
    mentions = (
        load_structured_mentions(FIXTURES / "disclosure_mentions.csv").iloc[[4]].copy()
    )
    mentions.loc[:, "counterparty_raw_name"] = "未来法定名称"
    mentions.loc[:, "publication_datetime"] = pd.Timestamp("2020-04-28 18:00:00")
    aliases = pd.DataFrame(
        [
            {
                "entity_id": "S5",
                "canonical_name": "未来法定名称",
                "alias": "000005",
                "alias_type": "ticker",
                "valid_from": "2010-01-01",
                "valid_to": None,
                "source": "fixture",
            },
            {
                "entity_id": "S5",
                "canonical_name": "未来法定名称",
                "alias": "未来法定名称",
                "alias_type": "legal_name_current_snapshot",
                "valid_from": "2021-01-01",
                "valid_to": None,
                "source": "fixture",
            },
        ]
    )

    audit = resolve_mentions(mentions, aliases)

    assert audit.iloc[0]["resolution_status"] == "unresolved"


def test_alias_dates_compare_with_cninfo_timezone_aware_publication():
    mentions = (
        load_structured_mentions(FIXTURES / "disclosure_mentions.csv").iloc[[0]].copy()
    )
    mentions.loc[:, "counterparty_raw_name"] = "时区样本"
    mentions["publication_datetime"] = mentions["publication_datetime"].astype("object")
    mentions.loc[:, "publication_datetime"] = pd.Timestamp("2021-04-29T00:00:00+08:00")
    aliases = pd.DataFrame(
        [
            {
                "entity_id": "S_TIME",
                "canonical_name": "时区样本股份有限公司",
                "alias": "时区样本",
                "alias_type": "legal_name_history",
                "valid_from": "2021-04-29",
                "valid_to": None,
                "source": "fixture",
            }
        ]
    )

    audit = resolve_mentions(mentions, aliases)

    assert audit.iloc[0]["resolution_status"] == "resolved"
    assert audit.iloc[0]["resolved_entity_id"] == "S_TIME"


def test_same_name_in_customer_and_supplier_tables_has_one_resolution_audit_row():
    mentions = (
        load_structured_mentions(FIXTURES / "disclosure_mentions.csv").iloc[[0]].copy()
    )
    repeated = mentions.copy()
    repeated.loc[:, "relationship_type"] = "supplier"
    mentions = pd.concat([mentions, repeated], ignore_index=True)
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)

    audit = resolve_mentions(mentions, aliases)

    assert len(audit) == 1
    assert audit.iloc[0]["resolution_status"] == "resolved"


def _mention_rows(
    values: list[tuple[str, str, str]],
) -> pd.DataFrame:
    template = load_structured_mentions(FIXTURES / "disclosure_mentions.csv").iloc[0]
    rows = []
    for document_id, raw_name, publication_datetime in values:
        row = template.to_dict()
        row.update(
            {
                "source_document_id": document_id,
                "counterparty_raw_name": raw_name,
                "publication_datetime": pd.Timestamp(publication_datetime),
                "evidence_text": f"1 {raw_name} 100.00 10.00",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _fuzzy_aliases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entity_id": "S_X",
                "canonical_name": "abcx",
                "alias": "abcx",
                "alias_type": "legal_name_history",
                "valid_from": "2020-01-01",
                "valid_to": None,
                "source": "fixture",
            },
            {
                "entity_id": "S_Y",
                "canonical_name": "abcy",
                "alias": "abcy",
                "alias_type": "legal_name_history",
                "valid_from": "2020-01-01",
                "valid_to": None,
                "source": "fixture",
            },
        ]
    )


def test_ambiguous_exact_alias_remains_unresolved() -> None:
    mentions = _mention_rows(
        [("D-AMBIGUOUS", "同名材料有限公司", "2021-04-29T12:00:00+08:00")]
    )
    aliases = pd.DataFrame(
        [
            {
                "entity_id": "S_X",
                "canonical_name": "同名材料甲股份有限公司",
                "alias": "同名材料有限公司",
                "alias_type": "legal_name_history",
                "valid_from": "2020-01-01",
                "valid_to": None,
                "source": "fixture",
            },
            {
                "entity_id": "S_Y",
                "canonical_name": "同名材料乙股份有限公司",
                "alias": "同名材料有限公司",
                "alias_type": "legal_name_history",
                "valid_from": "2020-01-01",
                "valid_to": None,
                "source": "fixture",
            },
        ]
    )

    row = resolve_mentions(
        mentions,
        aliases,
        config=ResolutionConfig(fuzzy_enabled=False),
    ).iloc[0]

    assert row["resolution_status"] == "ambiguous"
    assert row["resolution_method"] == "exact_alias_collision"
    assert pd.isna(row["resolved_entity_id"])
    assert row["candidate_count"] == 2


def test_manual_override_is_reproducible_across_input_order(
    tmp_path: Path,
) -> None:
    mentions = _mention_rows(
        [
            ("D-OVERRIDE-2", "未收录乙材料有限公司", "2021-04-30T12:00:00+08:00"),
            ("D-OVERRIDE-1", "未收录甲科技有限公司", "2021-04-29T12:00:00+08:00"),
        ]
    )
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    override_path = tmp_path / "entity_overrides.csv"
    pd.DataFrame(
        [
            {"counterparty_raw_name": "未收录甲科技有限公司", "entity_id": "S1"},
            {"counterparty_raw_name": "未收录乙材料有限公司", "entity_id": "S2"},
        ]
    ).to_csv(override_path, index=False)
    config = ResolutionConfig(fuzzy_enabled=False)

    forward = resolve_mentions(
        mentions,
        aliases,
        config=config,
        override_path=override_path,
    )
    reversed_inputs = resolve_mentions(
        mentions.iloc[::-1].reset_index(drop=True),
        aliases.iloc[::-1].reset_index(drop=True),
        config=config,
        override_path=override_path,
    )

    pd.testing.assert_frame_equal(forward, reversed_inputs)
    assert set(forward["resolution_method"]) == {"manual_override"}
    assert set(forward["notes"]) == {"version_controlled_manual_override"}
    assert dict(
        zip(
            forward["counterparty_raw_name"],
            forward["resolved_entity_id"],
            strict=True,
        )
    ) == {
        "未收录甲科技有限公司": "S1",
        "未收录乙材料有限公司": "S2",
    }


def test_appending_future_alias_does_not_change_past_resolution() -> None:
    mentions = _mention_rows(
        [("D-PAST", "华东电池股份有限公司", "2020-04-15T18:00:00+08:00")]
    )
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    future_collision = pd.DataFrame(
        [
            {
                "entity_id": "S5",
                "canonical_name": "未来电池股份有限公司",
                "alias": "华东电池股份有限公司",
                "alias_type": "future_legal_name",
                "valid_from": "2030-01-01",
                "valid_to": None,
                "source": "future_fixture",
            }
        ]
    )
    config = ResolutionConfig(fuzzy_enabled=False)

    baseline = resolve_mentions(mentions, aliases, config=config)
    after_append = resolve_mentions(
        mentions,
        pd.concat([aliases, future_collision], ignore_index=True),
        config=config,
    )

    pd.testing.assert_frame_equal(baseline, after_append)
    assert baseline.iloc[0]["resolved_entity_id"] == "S2"


def test_alias_index_is_built_once_per_shanghai_publication_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mentions = _mention_rows(
        [
            ("D2", "unknown-two", "2021-04-29T23:30:00+08:00"),
            ("D1", "unknown-one", "2021-04-29T01:00:00+08:00"),
            ("D3", "unknown-three", "2021-04-29T16:30:00Z"),
        ]
    )
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    calls: list[pd.Timestamp] = []
    original = resolve_module._alias_index

    def spy(alias_df: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, set[str]]:
        calls.append(as_of)
        return original(alias_df, as_of)

    monkeypatch.setattr(resolve_module, "_alias_index", spy)
    audit = resolve_mentions(
        mentions,
        aliases,
        config=ResolutionConfig(fuzzy_enabled=False),
    )

    assert [value.strftime("%Y-%m-%d") for value in calls] == [
        "2021-04-29",
        "2021-04-30",
    ]
    assert audit["source_document_id"].tolist() == ["D1", "D2", "D3"]


def test_fuzzy_top_two_preserves_reverse_lexical_tie_and_margin_semantics() -> None:
    mentions = _mention_rows([("D1", "abcz", "2021-04-29T12:00:00+08:00")])
    aliases = _fuzzy_aliases()
    normalized = normalize_company_name("abcz")
    old_scoring = sorted(
        (ratio(normalized, candidate), candidate) for candidate in ("abcx", "abcy")
    )[::-1]
    assert old_scoring[0][1] == "abcy"
    assert old_scoring[0][0] == old_scoring[1][0]

    tie_allowed = resolve_mentions(
        mentions,
        aliases,
        config=ResolutionConfig(
            fuzzy_enabled=True,
            fuzzy_min_score=70.0,
            fuzzy_min_margin=0.0,
        ),
    )
    margin_required = resolve_mentions(
        mentions,
        aliases,
        config=ResolutionConfig(
            fuzzy_enabled=True,
            fuzzy_min_score=70.0,
            fuzzy_min_margin=1.0,
        ),
    )

    assert tie_allowed.iloc[0]["resolution_method"] == "high_confidence_fuzzy"
    assert tie_allowed.iloc[0]["resolved_entity_id"] == "S_Y"
    assert margin_required.iloc[0]["resolution_status"] == "unresolved"


def test_fuzzy_top_two_is_cached_by_market_day_and_normalized_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mentions = _mention_rows(
        [
            ("D1", "abcz", "2021-04-29T01:00:00+08:00"),
            ("D2", "a b c z", "2021-04-29T20:00:00+08:00"),
        ]
    )
    calls = 0
    original = resolve_module.process.extract

    def spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(resolve_module.process, "extract", spy)
    resolve_mentions(
        mentions,
        _fuzzy_aliases(),
        config=ResolutionConfig(
            fuzzy_enabled=True,
            fuzzy_min_score=70.0,
            fuzzy_min_margin=1.0,
        ),
    )

    assert calls == 1


def test_resolution_logs_every_thousand_and_final_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mentions = _mention_rows(
        [
            (
                f"D{index:04d}",
                "华东电池股份有限公司",
                "2021-04-29T12:00:00+08:00",
            )
            for index in range(1001)
        ]
    )
    aliases = pd.read_csv(FIXTURES / "company_alias.csv", dtype=str)
    caplog.set_level(logging.INFO, logger=resolve_module.__name__)

    audit = resolve_mentions(
        mentions,
        aliases,
        config=ResolutionConfig(fuzzy_enabled=False),
    )

    assert len(audit) == 1001
    messages = [record.getMessage() for record in caplog.records]
    assert any("completed=1000 total=1001" in message for message in messages)
    assert any("completed=1001 total=1001" in message for message in messages)
