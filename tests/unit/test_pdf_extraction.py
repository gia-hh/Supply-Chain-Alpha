from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pypdf import PdfWriter

from supply_chain_alpha.data import pdf_extraction
from supply_chain_alpha.data.pdf_extraction import (
    extract_cninfo_documents,
    extract_mentions_from_text,
    infer_source_period_end,
)
from supply_chain_alpha.data.schemas import DISCLOSURE_RAW, validate_table


def _extract(text: str):
    return extract_mentions_from_text(
        text,
        source_company_id="SZSE:000001",
        source_period_end="2020-12-31",
        publication_datetime="2021-04-30T18:00:00+08:00",
        source_document_id="DOC-2020",
        source_document_url_or_path="https://static.cninfo.com.cn/report.pdf",
    )


def test_source_period_requires_an_explicit_annual_report_year() -> None:
    assert (
        infer_source_period_end("平安银行：2020 年年度报告（修订版）") == "2020-12-31"
    )
    assert infer_source_period_end("招商银行2020年度报告") == "2020-12-31"
    assert infer_source_period_end("2019年H股年度报告") == "2019-12-31"
    assert infer_source_period_end("601616_2021年_年度报告") == "2021-12-31"
    assert infer_source_period_end("威龙股份2021年年度年度报告") == "2021-12-31"
    assert infer_source_period_end("2020年度年报") == "2020-12-31"
    with pytest.raises(ValueError, match="Cannot infer"):
        infer_source_period_end("年度报告（修订版）")


def test_standardized_sections_extract_named_and_anonymous_rows_with_evidence() -> None:
    result = _extract(
        """
        1 章节之外公司 99.00 1.00
        公司主要客户情况
        序号 客户名称 销售额（万元） 占年度销售总额比例（%） 是否存在关联关系
        1 华东电池股份有限公司 1,234.50 12.50 否
        2 第一名客户 200.00 2.00 否
        公司主要供应商情况
        序号 供应商名称 采购额（元） 占年度采购总额比例（%） 是否存在关联关系
        1 华南材料有限责任公司 500,000.00 10.00% 否
        2 供应商A 未披露 未披露
        第六节 财务报告
        1 章节之后公司 88.00 1.00
        """
    )

    assert result.section_types == ("customer", "supplier")
    assert result.contains_relationship_section
    assert set(result.mentions["counterparty_raw_name"]) == {
        "华东电池股份有限公司",
        "第一名客户",
        "华南材料有限责任公司",
        "供应商A",
    }
    assert not result.mentions["counterparty_raw_name"].str.contains("章节").any()
    named = result.mentions.set_index("counterparty_raw_name")
    assert named.loc["华东电池股份有限公司", "exposure_value"] == 12_345_000.0
    assert named.loc["华东电池股份有限公司", "exposure_share"] == 0.125
    assert named.loc["华南材料有限责任公司", "exposure_value"] == 500_000.0
    assert named.loc["华南材料有限责任公司", "exposure_share"] == 0.10
    assert pd.isna(named.loc["供应商A", "exposure_value"])
    assert pd.isna(named.loc["供应商A", "exposure_share"])
    for row in result.mentions.itertuples(index=False):
        assert row.counterparty_raw_name in row.evidence_text
    validate_table(result.mentions, DISCLOSURE_RAW, allow_extra=False)


def test_one_bare_measure_in_two_column_table_is_left_null() -> None:
    result = _extract(
        """
        公司主要客户情况
        序号 客户名称 销售额（万元） 占年度销售总额比例（%）
        1 上海模糊数据有限公司 7.00
        """
    )
    row = result.mentions.iloc[0]
    assert pd.isna(row["exposure_value"])
    assert pd.isna(row["exposure_share"])


def test_redaction_markers_require_positive_disclosed_exposure() -> None:
    result = _extract(
        """
        公司主要供应商情况
        序号 供应商名称 采购额（万元） 占年度采购总额比例（%）
        1 -- 0.00 0.00%
        2 - 100.00 1.00%
        3 ”。 100.00 1.00%
        4 - - -
        5 ***** 200.00 2.00%
        """
    )

    assert result.section_types == ("supplier",)
    assert set(result.mentions["counterparty_raw_name"]) == {"-", "*****"}
    row = result.mentions.set_index("counterparty_raw_name").loc["-"]
    assert row["exposure_value"] == 1_000_000.0
    assert row["exposure_share"] == 0.01
    assert pdf_extraction._is_anonymous_label("-", "supplier") is True
    assert pdf_extraction._is_anonymous_label("*****", "supplier") is True


def test_invalid_counterparty_placeholders_are_rejected() -> None:
    result = _extract(
        """
        公司主要客户情况
        序号 客户名称 销售额（万元） 占年度销售总额比例（%）
        1 / 224.00 1.00%
        2 无 100.00 1.00%
        3 不适用 100.00 1.00%
        4 未披露 100.00 1.00%
        5 。 100.00 1.00%
        """
    )

    assert result.mentions.empty
    assert result.section_types == ("customer",)


def test_zero_or_unsubstantiated_named_rows_are_not_relationship_mentions() -> None:
    result = _extract(
        """
        公司主要客户情况
        序号 客户名称 销售额（万元） 占年度销售总额比例（%）
        1 空表公司有限公司 0.00 0.00%
        2 费用
        3 研发投入
        4 有证据公司有限公司 100.00 1.00%
        """
    )

    assert result.mentions["counterparty_raw_name"].tolist() == ["有证据公司有限公司"]


def test_completed_top_five_table_closes_before_numbered_financial_sections() -> None:
    result = _extract(
        """
        公司主要客户情况
        序号 客户名称 销售额（万元） 占年度销售总额比例（%）
        1 客户一 100.00 1.00%
        2 客户二 90.00 0.90%
        3 客户三 80.00 0.80%
        4 客户四 70.00 0.70%
        5 客户五 60.00 0.60%
        1 管理费用 500.00 5.00%
        2 销售费用 400.00 4.00%
        """
    )

    assert set(result.mentions["counterparty_raw_name"]) == {
        "客户一",
        "客户二",
        "客户三",
        "客户四",
        "客户五",
    }


def test_backward_rank_terminates_table_before_plausible_numeric_prose() -> None:
    result = _extract(
        """
        公司主要供应商情况
        序号 供应商名称 采购额（万元） 占年度采购总额比例（%）
        1 上海真实供应商有限公司 100.00 1.00%
        2 北京真实供应商有限公司 90.00 0.90%
        1 管理费用 500.00 5.00%
        3 不应恢复有限公司 80.00 0.80%
        """
    )

    assert set(result.mentions["counterparty_raw_name"]) == {
        "上海真实供应商有限公司",
        "北京真实供应商有限公司",
    }


def test_table_scan_window_expires_before_distant_numbered_content() -> None:
    filler = "\n".join(f"普通说明行 {index}" for index in range(30))
    result = _extract(
        f"""
        公司主要客户情况
        序号 客户名称 销售额（万元） 占年度销售总额比例（%）
        {filler}
        1 远端假阳性有限公司 100.00 1.00%
        """
    )

    assert result.mentions.empty
    assert result.section_types == ("customer",)


def test_split_header_and_rank_only_row_are_assembled() -> None:
    result = _extract(
        """
        公司主要客户情况
        序号
        客户名称 销售额（万元）
        占年度销售总额比例（%）
        1
        上海跨行客户有限公司
        123.00 4.50%
        2 北京同行客户有限公司
        100.00
        3.00%
        """
    )

    mentions = result.mentions.set_index("counterparty_raw_name")
    assert set(mentions.index) == {
        "上海跨行客户有限公司",
        "北京同行客户有限公司",
    }
    assert mentions.loc["上海跨行客户有限公司", "exposure_value"] == 1_230_000.0
    assert mentions.loc["上海跨行客户有限公司", "exposure_share"] == 0.045
    assert mentions.loc["北京同行客户有限公司", "exposure_value"] == 1_000_000.0
    assert mentions.loc["北京同行客户有限公司", "exposure_share"] == 0.03
    assert (
        "1 上海跨行客户有限公司 123.00 4.50%"
        == mentions.loc["上海跨行客户有限公司", "evidence_text"]
    )


def test_table_wide_unit_before_header_applies_to_bare_amounts() -> None:
    result = _extract(
        """
        公司主要客户情况
        单位：万元 币种：人民币
        序号 客户名称 销售额 占年度销售总额比例（%）
        1 客户一 156,641.43 31.96 否
        """
    )

    row = result.mentions.iloc[0]
    assert row["exposure_value"] == 1_566_414_300.0
    assert row["exposure_share"] == 0.3196


def test_numeric_anonymous_labels_are_preserved_without_deduplication() -> None:
    result = _extract(
        """
        公司主要客户情况
        序号 客户名称 销售额（万元） 占年度销售总额比例（%）
        1 客户 1 100.00 5.00%
        2 第 2 名 90.00 4.00%
        3 客户（三） 80.00 3.00%
        4 客户 4 70.00 2.00%
        5 第五名 60.00 1.00%
        """
    )

    assert set(result.mentions["counterparty_raw_name"]) == {
        "客户 1",
        "客户 4",
        "客户(三)",
        "第五名",
        "第 2 名",
    }
    assert len(result.mentions) == 5


def test_table_requires_rank_one_and_closes_on_rank_jump() -> None:
    starts_at_two = _extract(
        """
        公司主要供应商情况
        序号 供应商名称 采购额（万元） 占年度采购总额比例（%）
        2 不应抽取有限公司 100.00 1.00%
        """
    )
    assert starts_at_two.mentions.empty

    jumps_rank = _extract(
        """
        公司主要供应商情况
        序号 供应商名称 采购额（万元） 占年度采购总额比例（%）
        1 正确供应商有限公司 100.00 1.00%
        3 跳号供应商有限公司 90.00 0.90%
        2 不应恢复有限公司 80.00 0.80%
        """
    )
    assert jumps_rank.mentions["counterparty_raw_name"].tolist() == [
        "正确供应商有限公司"
    ]


def test_prose_tail_is_rejected_but_association_cell_is_allowed() -> None:
    result = _extract(
        """
        公司主要供应商情况
        序号 供应商名称 采购额（万元） 占年度采购总额比例（%） 是否关联
        1 叙述误报有限公司 100.00 1.00% 该公司主要从事研发活动
        2 合法供应商有限公司 90.00 0.90% 非关联单位。
        """
    )

    assert result.mentions["counterparty_raw_name"].tolist() == ["合法供应商有限公司"]


def test_summary_without_strong_table_header_is_not_parsed() -> None:
    result = _extract(
        """
        公司前五名客户销售额合计为 100 万元
        前五名客户占年度销售总额比例 20%
        1 管理费用 100.00 1.00%
        2 研发投入 90.00 0.90%
        """
    )

    assert result.mentions.empty
    assert result.section_types == ()


def test_unranked_named_table_is_bounded_to_five_qualified_rows() -> None:
    result = _extract(
        """
        公司主要客户情况
        客户名称 金额 占营业收入的比例
        北京甲方有限公司 91,252,733.60 2.73%
        中央电视台 62,013,548.73 1.85%
        深圳乙方有限公司 56,192,211.23 1.68%
        彰武丙方有限公司 40,283,810.58 1.20%
        北京丁方有限公司 37,499,274.95 1.12%
        管理费用 500.00 5.00%
        """
    )

    assert set(result.mentions["counterparty_raw_name"]) == {
        "北京甲方有限公司",
        "中央电视台",
        "深圳乙方有限公司",
        "彰武丙方有限公司",
        "北京丁方有限公司",
    }
    assert "管理费用" not in set(result.mentions["counterparty_raw_name"])


def test_unranked_header_continuation_and_page_furniture_are_not_names() -> None:
    result = _extract(
        """
        公司主要客户情况
        客户名称 营业收入 占营业收入的比例
        (%)
        第一客户有限公司 100.00 5.00%
        第二客户有限公司 90.00 4.00%
        2020 年年度报告
        15 / 170
        第三客户有限公司 80.00 3.00%
        """
    )

    assert set(result.mentions["counterparty_raw_name"]) == {
        "第一客户有限公司",
        "第二客户有限公司",
        "第三客户有限公司",
    }
    assert not result.mentions["counterparty_raw_name"].str.startswith("(%)").any()


def test_unranked_encoded_rank_labels_are_preserved() -> None:
    result = _extract(
        """
        公司主要客户情况
        前五名销售商 销售额 占全公司销售收入的比例
        客户 1 48,083,833.70 4.95%
        客户 2 46,022,267.70 4.74%
        客户 3 41,822,549.26 4.31%
        客户 4 41,477,538.91 4.27%
        客户 5 21,484,895.60 2.22%
        """
    )

    assert set(result.mentions["counterparty_raw_name"]) == {
        "客户 1",
        "客户 2",
        "客户 3",
        "客户 4",
        "客户 5",
    }


def test_ranked_generic_amount_header_accepts_association_none_cell() -> None:
    result = _extract(
        """
        公司主要供应商情况
        单位：元
        排名 供应商排名 金额 关联关系
        1 第一名 469,791,220.48 无
        2 第二名 374,546,120.03 无
        3 第三名 323,688,344.90 无
        4 第四名 200,000,000.00 无
        5 第五名 100,000,000.00 无
        """
    )

    assert len(result.mentions) == 5
    assert result.mentions["exposure_value"].notna().all()


def test_aggregate_summary_is_not_an_unranked_table_header() -> None:
    result = _extract(
        """
        公司主要供应商情况
        报告期内，公司向前五名供应商采购金额 152,818,091.44 元，占年度采购总额的比例为 40.10%。
        1 管理费用 500.00 5.00%
        2 研发投入 400.00 4.00%
        """
    )

    assert result.mentions.empty
    assert result.section_types == ()


def test_duplicate_mentions_are_deterministic_and_conflicting_value_is_not_selected() -> (
    None
):
    result = _extract(
        """
        公司前五名客户资料
        序号 客户名称 销售额（元） 占年度销售总额比例（%）
        1 同一客户有限公司 100.00 10.00
        2 同一客户有限公司 200.00 10.00
        """
    )
    assert len(result.mentions) == 1
    row = result.mentions.iloc[0]
    assert pd.isna(row["exposure_value"])
    assert row["exposure_share"] == 0.10
    assert row["evidence_text"] == "1 同一客户有限公司 100.00 10.00"


class _FakePage:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self.text = text
        self.error = error

    def extract_text(self) -> str | None:
        if self.error is not None:
            raise self.error
        return self.text


class _FakeReader:
    def __init__(self, path: str, *, strict: bool) -> None:
        assert strict is False
        assert path.endswith("d1.pdf")
        self.pages = [
            _FakePage(
                """
                公司主要供应商情况
                序号 供应商名称 采购额（元） 占年度采购总额比例（%）
                1 第一名供应商 100.00 5.00
                2 上海零部件有限公司 200.00 10.00
                3 ***** 300.00 15.00
                """
            ),
            _FakePage(error=RuntimeError("damaged text stream")),
        ]


class _CoverYearReader:
    def __init__(self, path: str, *, strict: bool) -> None:
        assert strict is False
        assert path.endswith("d1.pdf")
        self.pages = [
            _FakePage(
                """
                2020年度报告
                公司主要客户情况
                序号 客户名称 销售额（元） 占年度销售总额比例（%）
                1 封面年份客户有限公司 100.00 5.00%
                """
            )
        ]


def _document(path: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "document_id": "D1",
        "cninfo_org_id": "gssh0600000",
        "security_id": "SSE:600000",
        "reported_security_id": "SSE:600000",
        "security_id_reconciliation_method": "reported_id_in_master",
        "security_id_reconciliation_candidate_ids": '["SSE:600000"]',
        "security_id_reconciliation_candidate_count": 1,
        "announcement_title": "2020年年度报告",
        "announcement_time_ms": int(
            pd.Timestamp("2021-03-31T18:00:00+08:00").timestamp() * 1000
        ),
        "source_publication_datetime": "2021-03-31T18:00:00+08:00",
        "publication_datetime": "2021-03-31T18:00:00+08:00",
        "publication_time_precision": "exact",
        "publication_timing_rule": "cninfo_source_timestamp",
        "adjunct_url": "https://static.cninfo.com.cn/d1.pdf",
        "local_raw_path": path,
        "retrieval_datetime": "2021-04-01T00:00:00Z",
        "sha256": "a" * 64,
        "download_status": "DOWNLOADED",
    }
    row.update(overrides)
    return row


def test_pdf_cover_supplies_year_when_official_title_omits_it(tmp_path: Path) -> None:
    relative_path = "documents/d1.pdf"
    pdf_path = tmp_path / relative_path
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"fake PDF: reader is injected")
    documents = pd.DataFrame([_document(relative_path, announcement_title="年报全文")])

    result = extract_cninfo_documents(
        documents,
        raw_root=tmp_path,
        reader_factory=_CoverYearReader,
        max_workers=1,
    )

    assert result.document_audit.loc[0, "extraction_status"] == "SUCCESS"
    assert result.document_audit.loc[0, "source_period_end"] == "2020-12-31"
    assert result.mentions["counterparty_raw_name"].tolist() == ["封面年份客户有限公司"]


@pytest.mark.parametrize("max_workers", [1, 4])
def test_pdf_batch_retains_provenance_and_document_audit(
    tmp_path: Path,
    max_workers: int,
) -> None:
    relative_path = "documents/d1.pdf"
    pdf_path = tmp_path / relative_path
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"fake PDF: reader is injected")
    documents = pd.DataFrame(
        [
            _document(relative_path),
            _document(
                "documents/missing.pdf",
                document_id="D2",
                announcement_title="2021年年度报告",
                publication_datetime="2022-03-31T18:00:00+08:00",
                adjunct_url="https://static.cninfo.com.cn/d2.pdf",
                sha256="b" * 64,
            ),
        ]
    )

    result = extract_cninfo_documents(
        documents,
        raw_root=tmp_path,
        reader_factory=_FakeReader,
        max_workers=max_workers,
    )

    assert set(result.mentions["counterparty_raw_name"]) == {
        "第一名供应商",
        "上海零部件有限公司",
        "*****",
    }
    assert (
        result.mentions["source_period_end"]
        .dt.strftime("%Y-%m-%d")
        .eq("2020-12-31")
        .all()
    )
    assert result.mentions["publication_datetime"].notna().all()
    assert (
        result.mentions["source_publication_datetime"]
        .eq(pd.Timestamp("2021-03-31T18:00:00+08:00"))
        .all()
    )
    assert result.mentions["publication_time_precision"].eq("exact").all()
    assert (
        result.mentions["publication_timing_rule"].eq("cninfo_source_timestamp").all()
    )
    assert result.mentions["source_document_id"].eq("D1").all()
    assert (
        result.mentions["source_document_url_or_path"]
        .eq("https://static.cninfo.com.cn/d1.pdf")
        .all()
    )

    audits = result.documents.set_index("source_document_id")
    assert audits.loc["D1", "source_period_end"] == "2020-12-31"
    assert audits.loc["D1", "source_company_reported_id"] == "SSE:600000"
    assert audits.loc["D1", "source_company_org_id"] == "gssh0600000"
    assert (
        audits.loc["D1", "source_company_id_reconciliation_method"]
        == "reported_id_in_master"
    )
    assert (
        audits.loc["D1", "source_company_id_reconciliation_candidate_ids"]
        == '["SSE:600000"]'
    )
    assert audits.loc["D1", "source_company_id_reconciliation_candidate_count"] == 1
    assert audits.loc["D1", "source_announcement_time_ms"] == int(
        pd.Timestamp("2021-03-31T18:00:00+08:00").timestamp() * 1000
    )
    assert (
        audits.loc["D1", "source_publication_datetime"] == "2021-03-31T18:00:00+08:00"
    )
    assert bool(audits.loc["D1", "contains_relationship_section"])
    assert bool(audits.loc["D1", "captured_relationship_section"])
    assert audits.loc["D1", "section_types"] == "supplier"
    assert audits.loc["D1", "named_mention_count"] == 1
    assert audits.loc["D1", "anonymous_mention_count"] == 2
    assert audits.loc["D1", "page_count"] == 2
    assert audits.loc["D1", "pages_with_text"] == 1
    assert audits.loc["D1", "extraction_status"] == "PARTIAL"
    assert "damaged text stream" in audits.loc["D1", "extraction_error"]
    assert audits.loc["D1", "source_url"] == "https://static.cninfo.com.cn/d1.pdf"
    assert audits.loc["D1", "sha256"] == "a" * 64
    assert audits.loc["D2", "extraction_status"] == "ERROR"
    assert "annual-report PDF is missing" in audits.loc["D2", "extraction_error"]


def test_duplicate_document_identifiers_fail_loudly() -> None:
    documents = pd.DataFrame([_document("a.pdf"), _document("b.pdf")])
    with pytest.raises(ValueError, match="duplicate document identifiers"):
        extract_cninfo_documents(documents)


@pytest.mark.parametrize("unsafe_path", ["../outside.pdf", "../../outside.pdf"])
def test_pdf_source_path_cannot_escape_raw_root(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    outside = tmp_path.parent / "outside.pdf"
    outside.write_bytes(b"must never be opened")

    result = extract_cninfo_documents(
        pd.DataFrame([_document(unsafe_path)]),
        raw_root=tmp_path,
        reader_factory=_FakeReader,
        max_workers=1,
    )

    audit = result.document_audit.iloc[0]
    assert audit["extraction_status"] == "ERROR"
    assert "escapes raw_root" in str(audit["extraction_error"])


@pytest.mark.parametrize("max_workers", [True, 0, 9, 1.5])
def test_pdf_extraction_worker_count_is_bounded(max_workers: Any) -> None:
    documents = pd.DataFrame([{"document_id": "D1"}])
    with pytest.raises(ValueError, match="integer from 1 to 8"):
        extract_cninfo_documents(documents, max_workers=max_workers)


def test_threaded_custom_reader_merges_out_of_order_work_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(4)
    completion_order: list[str] = []
    worker_threads: set[int] = set()
    state_lock = threading.Lock()
    d4_done = threading.Event()
    d3_done = threading.Event()
    d2_done = threading.Event()

    def fake_extract(
        row: dict[str, Any],
        *,
        raw_root: Path,
        reader_factory: Any,
        cancel_event: threading.Event | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        del raw_root, reader_factory
        assert cancel_event is not None
        with state_lock:
            worker_threads.add(threading.get_ident())
        barrier.wait(timeout=3)
        document_id = str(row["document_id"])
        if document_id == "D4":
            pass
        elif document_id == "D3":
            assert d4_done.wait(timeout=3)
        elif document_id == "D2":
            assert d3_done.wait(timeout=3)
        else:
            assert d2_done.wait(timeout=3)
        with state_lock:
            completion_order.append(document_id)
        if document_id == "D4":
            d4_done.set()
        elif document_id == "D3":
            d3_done.set()
        elif document_id == "D2":
            d2_done.set()
        return pd.DataFrame(columns=DISCLOSURE_RAW.required_columns), {
            "source_document_id": document_id,
            "extraction_status": "SUCCESS",
        }

    monkeypatch.setattr(pdf_extraction, "_extract_pdf_document", fake_extract)
    input_order = ["D2", "D1", "D4", "D3"]
    result = extract_cninfo_documents(
        pd.DataFrame({"document_id": input_order}),
        reader_factory=_FakeReader,
        max_workers=4,
    )

    assert len(worker_threads) == 4
    assert completion_order == ["D4", "D3", "D2", "D1"]
    assert result.document_audit["source_document_id"].tolist() == input_order


def test_parallel_extraction_cancels_workers_and_propagates_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    worker_cancelled = threading.Event()

    def blocking_extract(
        row: dict[str, Any],
        *,
        raw_root: Path,
        reader_factory: Any,
        cancel_event: threading.Event | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        del row, raw_root, reader_factory
        assert cancel_event is not None
        worker_started.set()
        if cancel_event.wait(timeout=3):
            worker_cancelled.set()
            raise pdf_extraction._ExtractionCancelled("test cancellation")
        raise AssertionError("worker did not receive cancellation")

    def interrupting_wait(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        assert worker_started.wait(timeout=3)
        raise KeyboardInterrupt

    monkeypatch.setattr(pdf_extraction, "_extract_pdf_document", blocking_extract)
    monkeypatch.setattr(pdf_extraction, "wait", interrupting_wait)

    documents = pd.DataFrame({"document_id": ["D1", "D2"]})
    with pytest.raises(KeyboardInterrupt):
        extract_cninfo_documents(
            documents,
            reader_factory=_FakeReader,
            max_workers=2,
        )
    assert worker_cancelled.is_set()


def test_pdfium_defaults_to_eight_process_workers_and_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_workers: list[int] = []

    def fake_process_pool(
        jobs: list[pdf_extraction._PdfExtractionJob],
        *,
        raw_root: Path,
        max_workers: int,
        document_timeout_seconds: float,
    ) -> list[pdf_extraction._PdfExtractionOutcome]:
        del raw_root
        observed_workers.append(max_workers)
        assert document_timeout_seconds == 300.0
        return [
            pdf_extraction._PdfExtractionOutcome(
                job=job,
                mentions=pd.DataFrame(columns=DISCLOSURE_RAW.required_columns),
                audit={
                    "source_document_id": str(job.row["document_id"]),
                    "extraction_status": "SUCCESS",
                },
            )
            for job in jobs
        ]

    monkeypatch.setattr(
        pdf_extraction,
        "_parallel_extract_documents_process",
        fake_process_pool,
    )
    input_order = ["D3", "D1", "D2"]

    result = extract_cninfo_documents(pd.DataFrame({"document_id": input_order}))

    assert observed_workers == [8]
    assert result.document_audit["source_document_id"].tolist() == input_order


def test_pdfium_single_worker_still_uses_terminable_process_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[int, float]] = []

    def fake_process_pool(
        jobs: list[pdf_extraction._PdfExtractionJob],
        *,
        raw_root: Path,
        max_workers: int,
        document_timeout_seconds: float,
    ) -> list[pdf_extraction._PdfExtractionOutcome]:
        del raw_root
        observed.append((max_workers, document_timeout_seconds))
        return [
            pdf_extraction._PdfExtractionOutcome(
                job=job,
                mentions=pd.DataFrame(columns=DISCLOSURE_RAW.required_columns),
                audit={"source_document_id": job.row["document_id"]},
            )
            for job in jobs
        ]

    monkeypatch.setattr(
        pdf_extraction,
        "_parallel_extract_documents_process",
        fake_process_pool,
    )

    result = extract_cninfo_documents(
        pd.DataFrame({"document_id": ["D1"]}),
        max_workers=1,
        document_timeout_seconds=12.5,
    )

    assert observed == [(1, 12.5)]
    assert result.document_audit["source_document_id"].tolist() == ["D1"]


@pytest.mark.parametrize(
    "timeout",
    [True, 0, -1, float("inf"), float("nan"), "300"],
)
def test_pdf_extraction_document_timeout_must_be_positive_and_finite(
    timeout: Any,
) -> None:
    with pytest.raises(ValueError, match="positive finite number"):
        extract_cninfo_documents(
            pd.DataFrame({"document_id": ["D1"]}),
            document_timeout_seconds=timeout,
        )


def test_pdfium_process_timeout_terminates_and_joins_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEvent:
        def __init__(self) -> None:
            self.was_set = False

        def set(self) -> None:
            self.was_set = True

    class FakeQueue:
        def __init__(self) -> None:
            self.closed = False
            self.cancelled_join = False

        def get_nowait(self) -> Any:
            raise pdf_extraction.Empty

        def close(self) -> None:
            self.closed = True

        def cancel_join_thread(self) -> None:
            self.cancelled_join = True

    class NeverReadyResult:
        def ready(self) -> bool:
            return False

    class FakePool:
        def __init__(self) -> None:
            self.submitted = 0
            self.terminated = False
            self.joined = False
            self.closed = False

        def apply_async(self, *args: Any, **kwargs: Any) -> NeverReadyResult:
            del args, kwargs
            self.submitted += 1
            return NeverReadyResult()

        def terminate(self) -> None:
            self.terminated = True

        def join(self) -> None:
            self.joined = True

        def close(self) -> None:
            self.closed = True

    event = FakeEvent()
    started_queue = FakeQueue()
    pool = FakePool()

    class FakeContext:
        Event = lambda self: event
        Queue = lambda self: started_queue
        Pool = lambda self, **kwargs: pool

    clock = [0.0]

    def advance(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(
        pdf_extraction.multiprocessing,
        "get_context",
        lambda _: FakeContext(),
    )
    monkeypatch.setattr(pdf_extraction.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pdf_extraction.time, "sleep", advance)
    documents = pd.DataFrame({"document_id": ["D1", "D2", "D3"]})

    with pytest.raises(
        pdf_extraction.PdfExtractionTimeoutError,
        match="document_id=D1.*timeout_seconds=0.1",
    ):
        extract_cninfo_documents(
            documents,
            max_workers=2,
            document_timeout_seconds=0.1,
        )

    assert pool.submitted == 2
    assert event.was_set
    assert pool.terminated and pool.joined
    assert not pool.closed
    assert started_queue.closed and started_queue.cancelled_join


def test_pdfium_watchdog_excludes_host_suspend_but_keeps_active_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEvent:
        def set(self) -> None:
            pass

    class FakeQueue:
        def get_nowait(self) -> Any:
            raise pdf_extraction.Empty

        def close(self) -> None:
            pass

        def join_thread(self) -> None:
            pass

    clock = [0.0]
    sleep_count = [0]

    class ReadyAfterResumeResult:
        def __init__(self, job: pdf_extraction._PdfExtractionJob) -> None:
            self.job = job

        def ready(self) -> bool:
            # Remain pending for one normal poll after the simulated resume.
            # Without suspend compensation the second check times out first.
            return sleep_count[0] >= 2

        def get(self) -> pdf_extraction._PdfExtractionOutcome:
            return pdf_extraction._PdfExtractionOutcome(
                job=self.job,
                mentions=pd.DataFrame(columns=DISCLOSURE_RAW.required_columns),
                audit={"source_document_id": self.job.row["document_id"]},
            )

    class FakePool:
        def __init__(self) -> None:
            self.closed = False
            self.joined = False
            self.terminated = False

        def apply_async(
            self,
            function: Any,
            args: tuple[pdf_extraction._PdfExtractionJob],
            kwds: dict[str, Any],
        ) -> ReadyAfterResumeResult:
            del function, kwds
            return ReadyAfterResumeResult(args[0])

        def close(self) -> None:
            self.closed = True

        def join(self) -> None:
            self.joined = True

        def terminate(self) -> None:
            self.terminated = True

    pool = FakePool()

    class FakeContext:
        Event = lambda self: FakeEvent()
        Queue = lambda self: FakeQueue()
        Pool = lambda self, **kwargs: pool

    def sleep_with_one_host_suspend(seconds: float) -> None:
        sleep_count[0] += 1
        clock[0] += 3 * 60 * 60 if sleep_count[0] == 1 else seconds

    monkeypatch.setattr(
        pdf_extraction.multiprocessing,
        "get_context",
        lambda _: FakeContext(),
    )
    monkeypatch.setattr(pdf_extraction.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pdf_extraction.time, "sleep", sleep_with_one_host_suspend)
    jobs = [pdf_extraction._PdfExtractionJob(ordinal=0, row={"document_id": "D1"})]

    outcomes = pdf_extraction._parallel_extract_documents_process(
        jobs,
        raw_root=Path("."),
        max_workers=1,
        document_timeout_seconds=0.1,
    )

    assert [outcome.job.ordinal for outcome in outcomes] == [0]
    assert sleep_count[0] == 2
    assert pool.closed and pool.joined
    assert not pool.terminated


def test_pdfium_process_keyboard_interrupt_terminates_and_joins_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptingResult:
        def ready(self) -> bool:
            raise KeyboardInterrupt

    class FakeEvent:
        def __init__(self) -> None:
            self.was_set = False

        def set(self) -> None:
            self.was_set = True

    class FakeQueue:
        def get_nowait(self) -> Any:
            raise pdf_extraction.Empty

        def close(self) -> None:
            pass

        def cancel_join_thread(self) -> None:
            pass

    class FakePool:
        def __init__(self) -> None:
            self.terminated = False
            self.joined = False

        def apply_async(self, *args: Any, **kwargs: Any) -> InterruptingResult:
            del args, kwargs
            return InterruptingResult()

        def terminate(self) -> None:
            self.terminated = True

        def join(self) -> None:
            self.joined = True

    event = FakeEvent()
    started_queue = FakeQueue()
    pool = FakePool()

    class FakeContext:
        Event = lambda self: event
        Queue = lambda self: started_queue
        Pool = lambda self, **kwargs: pool

    monkeypatch.setattr(
        pdf_extraction.multiprocessing,
        "get_context",
        lambda _: FakeContext(),
    )

    with pytest.raises(KeyboardInterrupt):
        extract_cninfo_documents(
            pd.DataFrame({"document_id": ["D1"]}),
            max_workers=1 + 1,
        )

    assert event.was_set
    assert pool.terminated and pool.joined


@pytest.mark.parametrize("shutdown_stage", ["close", "join"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, RuntimeError])
def test_pdfium_process_shutdown_failure_terminates_and_joins_pool(
    monkeypatch: pytest.MonkeyPatch,
    shutdown_stage: str,
    error_type: type[BaseException],
) -> None:
    class FakeEvent:
        def __init__(self) -> None:
            self.was_set = False

        def set(self) -> None:
            self.was_set = True

    class FakeQueue:
        def __init__(self) -> None:
            self.closed = False
            self.cancelled_join = False

        def get_nowait(self) -> Any:
            raise pdf_extraction.Empty

        def close(self) -> None:
            self.closed = True

        def cancel_join_thread(self) -> None:
            self.cancelled_join = True

    class ReadyResult:
        def __init__(self, job: pdf_extraction._PdfExtractionJob) -> None:
            self.job = job

        def ready(self) -> bool:
            return True

        def get(self) -> pdf_extraction._PdfExtractionOutcome:
            return pdf_extraction._PdfExtractionOutcome(
                job=self.job,
                mentions=pd.DataFrame(columns=DISCLOSURE_RAW.required_columns),
                audit={"source_document_id": self.job.row["document_id"]},
            )

    class FakePool:
        def __init__(self) -> None:
            self.closed = False
            self.terminated = False
            self.joined = False
            self.join_calls = 0

        def apply_async(
            self,
            function: Any,
            args: tuple[pdf_extraction._PdfExtractionJob],
            kwds: dict[str, Any],
        ) -> ReadyResult:
            del function, kwds
            return ReadyResult(args[0])

        def close(self) -> None:
            self.closed = True
            if shutdown_stage == "close":
                raise error_type("shutdown interrupted")

        def terminate(self) -> None:
            self.terminated = True

        def join(self) -> None:
            self.join_calls += 1
            if shutdown_stage == "join" and not self.terminated:
                raise error_type("shutdown interrupted")
            self.joined = True

    event = FakeEvent()
    started_queue = FakeQueue()
    pool = FakePool()

    class FakeContext:
        Event = lambda self: event
        Queue = lambda self: started_queue
        Pool = lambda self, **kwargs: pool

    monkeypatch.setattr(
        pdf_extraction.multiprocessing,
        "get_context",
        lambda _: FakeContext(),
    )

    with pytest.raises(error_type, match="shutdown interrupted"):
        extract_cninfo_documents(
            pd.DataFrame({"document_id": ["D1"]}),
            max_workers=1,
        )

    assert pool.closed
    assert event.was_set
    assert pool.terminated and pool.joined
    assert pool.join_calls == (1 if shutdown_stage == "close" else 2)
    assert started_queue.closed and started_queue.cancelled_join


def test_real_pdf_process_pool_smoke_preserves_input_order(tmp_path: Path) -> None:
    documents: list[dict[str, Any]] = []
    for document_id in ("D2", "D1"):
        relative_path = f"documents/{document_id}.pdf"
        pdf_path = tmp_path / relative_path
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with pdf_path.open("wb") as output:
            writer.write(output)
        documents.append(
            _document(
                relative_path,
                document_id=document_id,
                adjunct_url=f"https://static.cninfo.com.cn/{document_id}.pdf",
            )
        )

    result = extract_cninfo_documents(
        pd.DataFrame(documents),
        raw_root=tmp_path,
        max_workers=2,
    )

    assert result.document_audit["source_document_id"].tolist() == ["D2", "D1"]
    assert result.document_audit["extraction_status"].eq("NO_TEXT").all()


def test_pdfium_adapter_closes_document_pages_and_text_pages(tmp_path: Path) -> None:
    class FakeTextPage:
        def __init__(self, text: str = "", error: Exception | None = None) -> None:
            self.text = text
            self.error = error
            self.closed = False

        def get_text_range(self) -> str:
            if self.error is not None:
                raise self.error
            return self.text

        def close(self) -> None:
            self.closed = True

    class FakeNativePage:
        def __init__(self, text_page: FakeTextPage) -> None:
            self.text_page = text_page
            self.closed = False

        def get_textpage(self) -> FakeTextPage:
            return self.text_page

        def close(self) -> None:
            self.closed = True

    class FakeDocument:
        def __init__(self, pages: list[FakeNativePage]) -> None:
            self.pages = pages
            self.closed = False

        def __len__(self) -> int:
            return len(self.pages)

        def __getitem__(self, index: int) -> FakeNativePage:
            return self.pages[index]

        def close(self) -> None:
            self.closed = True

    text_pages = [
        FakeTextPage(
            """
            公司主要客户情况
            序号 客户名称 销售额（元） 占年度销售总额比例（%）
            1 上海客户有限公司 100.00 10.00
            """
        ),
        FakeTextPage(error=RuntimeError("pdfium text failure")),
    ]
    native_pages = [FakeNativePage(page) for page in text_pages]
    document = FakeDocument(native_pages)
    relative_path = "documents/pdfium-contract.pdf"
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"factory owns parsing in this contract test")

    def reader_factory(path_value: str, *, strict: bool) -> Any:
        return pdf_extraction._PdfiumReader(
            path_value,
            strict=strict,
            document_factory=lambda _: document,
        )

    result = extract_cninfo_documents(
        pd.DataFrame([_document(relative_path)]),
        raw_root=tmp_path,
        reader_factory=reader_factory,
        max_workers=1,
    )

    assert result.document_audit.loc[0, "extraction_status"] == "PARTIAL"
    assert "pdfium text failure" in result.document_audit.loc[0, "extraction_error"]
    assert result.mentions["counterparty_raw_name"].tolist() == ["上海客户有限公司"]
    assert document.closed
    assert all(page.closed for page in native_pages)
    assert all(page.closed for page in text_pages)


def test_pdfium_runtime_dependency_is_declared() -> None:
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"pypdfium2>=5.13,<6.0"' in pyproject


def test_pdf_extraction_logs_every_thousand_and_final_document(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=pdf_extraction.__name__)
    pdf_extraction._log_extraction_progress(999, 1001)
    pdf_extraction._log_extraction_progress(1000, 1001)
    pdf_extraction._log_extraction_progress(1001, 1001)

    messages = [record.getMessage() for record in caplog.records]
    assert not any("completed=999 total=1001" in message for message in messages)
    assert any("completed=1000 total=1001" in message for message in messages)
    assert any("completed=1001 total=1001" in message for message in messages)
