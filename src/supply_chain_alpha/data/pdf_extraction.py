"""Conservative, source-grounded extraction from Chinese annual-report PDFs.

The parser intentionally targets the standardized top-customer/top-supplier
tables found in annual reports.  It records explicit anonymous labels as raw
mentions, but it does not perform (or imply) entity resolution.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import re
import threading
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import pandas as pd
import pypdfium2 as pdfium

from supply_chain_alpha.entities.normalize import (
    CounterpartyNameClass,
    classify_counterparty_name,
    is_anonymous_counterparty,
    is_redacted_counterparty,
)

from .schemas import (
    CNINFO_DOCUMENT_AUDIT,
    DISCLOSURE_RAW,
    coerce_disclosure_types,
    validate_table,
)

LOGGER = logging.getLogger(__name__)
PDF_MAX_EXTRACTION_WORKERS = 8
_PDF_WATCHDOG_POLL_SECONDS = 0.05
# A normal scheduler may overshoot a short sleep slightly.  Only a gap far
# beyond the poll cadence is treated as a host/process suspension.  The
# requested sleep itself always counts toward the hard document timeout.
_PDF_WATCHDOG_OVERSLEEP_TOLERANCE_SECONDS = 1.0

_RELATION_ORDER = ("customer", "supplier")
_SECTION_PATTERNS = {
    "customer": (
        re.compile(r"(?:公司)?主要客户情况"),
        re.compile(r"(?:公司)?前\s*(?:五|5)\s*(?:名|大)客户(?:情况|资料)?"),
        re.compile(r"客户名称\s*(?:销售额|销售金额|营业收入)"),
        re.compile(r"序号\s*客户(?:名称)?\s*(?:销售额|销售金额|营业收入)"),
    ),
    "supplier": (
        re.compile(r"(?:公司)?主要供应商情况"),
        re.compile(r"(?:公司)?前\s*(?:五|5)\s*(?:名|大)供应商(?:情况|资料)?"),
        re.compile(r"供应商名称\s*(?:采购额|采购金额|采购成本)"),
        re.compile(r"序号\s*供应商(?:名称)?\s*(?:采购额|采购金额|采购成本)"),
    ),
}
_ROW_PREFIX = re.compile(
    r"^\s*(?P<rank>[1-5一二三四五])"
    r"(?:\s*\.(?!\d)\s*|\s*[、)）]\s*|\s+)"
    r"(?P<body>.+?)\s*$"
)
_RANK_ONLY = re.compile(r"^\s*(?P<rank>[1-5一二三四五])\s*[.、)）]?\s*$")
_RANK_VALUE = {
    "1": 1,
    "一": 1,
    "2": 2,
    "二": 2,
    "3": 3,
    "三": 3,
    "4": 4,
    "四": 4,
    "5": 5,
    "五": 5,
}
_NUMBER_TOKEN = re.compile(
    r"(?<!\S)(?P<currency>[¥￥])?\s*"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
    r"(?P<unit>%|％|亿元|万元|亿|万|元)?(?=\s|$)"
)
_EXPLICIT_MISSING_MEASURE = re.compile(
    r"(?:未披露|未提供|不适用|不詳|不详|未知|保密|N\s*/?\s*A)",
    re.IGNORECASE,
)
_MAX_HEADER_SEEK_LINES = 12
_MAX_HEADER_BLOCK_LINES = 3
_MAX_ROW_CONTINUATION_LINES = 2
_ALLOWED_TAIL_WORDS = re.compile(
    r"(?:不存在关联关系|存在关联关系|无关联关系|非关联单位|关联单位|"
    r"非关联方|关联方|未披露|未提供|不适用|不詳|不详|未知|保密|"
    r"人民币|RMB|CNY|USD|N\s*/?\s*A|是|否|无)",
    re.IGNORECASE,
)
_TAIL_SEPARATORS = re.compile(r"[\s\u3000|｜,，;；:：.!?。！？·•()（）\[\]【】]+")
_REPORT_YEAR = re.compile(
    r"(?<!\d)((?:19|20)\d{2})"
    r"(?=[\s_\-—–:：·.()（）\[\]【】AHB股年度]{0,20}(?:年度报告|年报))"
)
_PAGE_FURNITURE = re.compile(
    r"^(?:(?:(?:19|20)\d{2})\s*年{0,2}\s*年度报告(?:全文)?|"
    r"\d+\s*/\s*\d+|第?\s*\d+\s*页\s*(?:共\s*\d+\s*页)?)$"
)
_SECTION_BREAK = re.compile(
    r"^(?:第\s*[一二三四五六七八九十\d]+\s*节|"
    r"[一二三四五六七八九十]+[、.]\s*\S+)"
)
_AMOUNT_WORDS = {
    "customer": (
        "销售额",
        "销售金额",
        "营业收入",
        "金额",
        "銷售額",
        "銷售金額",
        "金額",
    ),
    "supplier": (
        "采购额",
        "采购金额",
        "采购成本",
        "金额",
        "採購額",
        "採購金額",
        "金額",
    ),
}
_UNIT_MULTIPLIERS = {
    "元": 1.0,
    "万": 10_000.0,
    "万元": 10_000.0,
    "亿": 100_000_000.0,
    "亿元": 100_000_000.0,
}
_AUDIT_COLUMNS = (
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
)


@dataclass(frozen=True)
class TextExtractionResult:
    """Canonical mentions plus section-level observations for one text."""

    mentions: pd.DataFrame
    section_types: tuple[str, ...]

    @property
    def contains_relationship_section(self) -> bool:
        return bool(self.section_types)


@dataclass(frozen=True)
class PdfExtractionResult:
    """Batch PDF extraction output."""

    mentions: pd.DataFrame
    document_audit: pd.DataFrame

    @property
    def documents(self) -> pd.DataFrame:
        """Compatibility alias for consumers treating audit rows as documents."""

        return self.document_audit


@dataclass(frozen=True)
class _PdfExtractionJob:
    ordinal: int
    row: dict[str, Any]


@dataclass(frozen=True)
class _PdfExtractionOutcome:
    job: _PdfExtractionJob
    mentions: pd.DataFrame
    audit: dict[str, Any]


class _ExtractionCancelled(BaseException):
    """Stop worker work without converting cancellation into a PDF audit error."""


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise _ExtractionCancelled("PDF extraction cancelled")


class PdfExtractionTimeoutError(TimeoutError):
    """A native PDF worker exceeded the configured per-document wall time."""


class _PdfiumPage:
    """Small adapter that releases both PDFium text-page and page handles."""

    def __init__(self, page: Any) -> None:
        self._page = page

    def extract_text(self) -> str:
        text_page = self._page.get_textpage()
        try:
            return str(text_page.get_text_range())
        finally:
            text_page.close()

    def close(self) -> None:
        self._page.close()


class _PdfiumPages:
    def __init__(self, document: Any) -> None:
        self._document = document

    def __len__(self) -> int:
        return len(self._document)

    def __iter__(self) -> Iterator[_PdfiumPage]:
        for page_number in range(len(self)):
            yield _PdfiumPage(self._document[page_number])


class _PdfiumReader:
    """Reader facade matching the narrow interface used by the extractor."""

    def __init__(
        self,
        path: str,
        *,
        strict: bool,
        document_factory: Callable[[str], Any] = pdfium.PdfDocument,
    ) -> None:
        del strict
        self._document = document_factory(path)
        self.pages = _PdfiumPages(self._document)

    def close(self) -> None:
        self._document.close()


def _close_resource(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


_PROCESS_CANCEL_EVENT: Any = None
_PROCESS_STARTED_QUEUE: Any = None


def _initialize_pdf_process(cancel_event: Any, started_queue: Any) -> None:
    """Install spawn-safe coordination primitives in one PDF worker."""

    global _PROCESS_CANCEL_EVENT, _PROCESS_STARTED_QUEUE
    _PROCESS_CANCEL_EVENT = cancel_event
    _PROCESS_STARTED_QUEUE = started_queue


def _clean_line(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\u00a0", " ")
    return re.sub(r"[ \t\u3000]+", " ", value).strip()


def _is_header_continuation(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    return bool(
        re.fullmatch(
            r"(?:\(?[%％]\)?|(?:占比|比例)(?:\(?[%％]\)?)?)",
            compact,
        )
    )


def _empty_mentions() -> pd.DataFrame:
    return pd.DataFrame(columns=DISCLOSURE_RAW.columns)


def infer_source_period_end(announcement_title: str) -> str:
    """Infer fiscal year-end only when the annual-report year is explicit."""

    if not isinstance(announcement_title, str) or not announcement_title.strip():
        raise ValueError("announcement_title must be a non-empty string")
    match = _REPORT_YEAR.search(unicodedata.normalize("NFKC", announcement_title))
    if match is None:
        raise ValueError(
            f"Cannot infer annual-report year from title: {announcement_title!r}"
        )
    return f"{match.group(1)}-12-31"


def _section_marker(line: str) -> str | None:
    compact = re.sub(r"\s+", "", line)
    matches = [
        relation
        for relation in _RELATION_ORDER
        if any(pattern.search(compact) for pattern in _SECTION_PATTERNS[relation])
    ]
    return matches[0] if len(matches) == 1 else None


def _is_anonymous_label(name: str, relation: str) -> bool:
    del relation
    return is_anonymous_counterparty(name)


def _header_unit_multiplier(context: str, relation: str) -> float | None:
    compact = re.sub(r"\s+", "", context)
    if not any(word in compact for word in _AMOUNT_WORDS[relation]):
        return None
    # Prefer the most specific unit, and require it near a standardized amount
    # column rather than accepting an unrelated unit elsewhere in the section.
    amount_words = "|".join(map(re.escape, _AMOUNT_WORDS[relation]))
    match = re.search(
        rf"(?:{amount_words}).{{0,18}}?(亿元|万元|亿|万|元)(?:\)|）|$)",
        compact,
    )
    if match is not None:
        return _UNIT_MULTIPLIERS.get(match.group(1))
    # Common CNINFO tables declare the unit immediately *before* the column
    # header (``单位：万元 币种：人民币``).  This remains source-grounded because
    # the caller only supplies the bounded strong-header block.
    table_unit = re.search(r"单位[:：]?(亿元|万元|亿|万|元)", compact)
    return _UNIT_MULTIPLIERS.get(table_unit.group(1)) if table_unit else None


def _has_name_header(compact: str, relation: str) -> tuple[bool, bool]:
    sequence_header = "序号" in compact or "序號" in compact or "排名" in compact
    if relation == "customer":
        name_header = any(
            value in compact
            for value in (
                "客户名称",
                "客戶名稱",
                "前五名销售商",
                "前5名销售商",
                "主要销售商",
            )
        )
    else:
        name_header = any(
            value in compact
            for value in (
                "供应商名称",
                "供應商名稱",
                "供货商名称",
                "供貨商名稱",
                "供应商排名",
                "供應商排名",
                "前五名供应商",
                "前5名供应商",
                "前五名供货商",
                "前5名供货商",
                "公司名称",
            )
        )
    return sequence_header, name_header


def _table_header_mode(context: str, relation: str) -> str | None:
    """Return ``ranked`` or ``unranked`` for a bounded strong header block."""

    lines = [line for line in context.splitlines() if line.strip()]
    for start in range(len(lines)):
        block = "\n".join(lines[start:])
        compact = re.sub(r"\s+", "", block)
        sequence_header, name_header = _has_name_header(compact, relation)
        amount_header = any(word in compact for word in _AMOUNT_WORDS[relation])
        if sequence_header and name_header and amount_header:
            return "ranked"
        if (
            not name_header
            or not amount_header
            or not any(word in compact for word in ("比例", "占比"))
        ):
            continue
        # An unranked table header contains column labels, not already-disclosed
        # aggregate values.  Years in column names are harmless; monetary or
        # percentage values distinguish prose summaries from table headers.
        numeric_cells = [
            token.group("number") for token in _NUMBER_TOKEN.finditer(block)
        ]
        if any(not re.fullmatch(r"(?:19|20)\d{2}", value) for value in numeric_cells):
            continue
        return "unranked"
    return None


def _is_relationship_table_header(context: str, relation: str) -> bool:
    """Inventory mention-bearing tables independently of row acceptance."""

    lines = [line for line in context.splitlines() if line.strip()]
    for start in range(len(lines)):
        block = "\n".join(lines[start:])
        compact = re.sub(r"\s+", "", block)
        sequence_header, name_header = _has_name_header(compact, relation)
        if not name_header or not any(
            word in compact for word in _AMOUNT_WORDS[relation]
        ):
            continue
        if sequence_header:
            return True
        numeric_cells = [
            token.group("number") for token in _NUMBER_TOKEN.finditer(block)
        ]
        if not any(
            not re.fullmatch(r"(?:19|20)\d{2}", value) for value in numeric_cells
        ):
            return True
    return False


def _has_table_header(context: str, relation: str) -> bool:
    return _table_header_mode(context, relation) is not None


def _parse_number(value: str) -> float:
    return float(value.replace(",", ""))


def _tail_is_qualified(tail: str) -> bool:
    """Reject prose after a candidate name; allow only table-cell values."""

    residue = _NUMBER_TOKEN.sub(" ", tail)
    residue = _ALLOWED_TAIL_WORDS.sub(" ", residue)
    return not _TAIL_SEPARATORS.sub("", residue)


def _valid_name_candidate(name: str) -> str | None:
    candidate = name.strip(" \t|｜,，;；:：")
    if not candidate or len(candidate) > 120 or candidate in {"合计", "总计"}:
        return None
    if classify_counterparty_name(candidate) is CounterpartyNameClass.INVALID:
        return None
    if any(
        word in candidate
        for word in (
            "客户名称",
            "客戶名稱",
            "供应商名称",
            "供應商名稱",
            "占年度",
            "是否存在",
        )
    ):
        return None
    return candidate


def _name_encodes_rank(name: str, rank: int) -> bool:
    compact = re.sub(r"[\s\u3000_\-()（）]+", "", name)
    arabic = str(rank)
    chinese = "一二三四五"[rank - 1]
    symbol = rf"(?:{arabic}|{chinese})"
    entity = r"(?:客户|客戶|供应商|供應商|公司|单位|單位|法人|自然人|个人|個人)"
    patterns = (
        rf"^(?:第)?{symbol}(?:名|大|位|号)?$",
        rf"^{entity}(?:第)?{symbol}(?:名|大|位|号)?$",
        rf"^(?:第)?{symbol}(?:名|大|位|号)?{entity}$",
    )
    return any(re.fullmatch(pattern, compact) for pattern in patterns)


def _measure_values(
    tail: str,
    *,
    relation: str,
    context: str,
) -> tuple[float | None, float | None]:
    tokens = list(_NUMBER_TOKEN.finditer(tail))
    exposure_value: float | None = None
    exposure_share: float | None = None
    compact_context = re.sub(r"\s+", "", context)
    header_multiplier = _header_unit_multiplier(context, relation)
    amount_semantics = any(word in compact_context for word in _AMOUNT_WORDS[relation])
    header_percent = bool(
        re.search(r"(?:比例|占比).{0,12}(?:%|百分比)", compact_context)
    )

    if len(tokens) == 1 and tokens[0].group("unit") is None:
        number = _parse_number(tokens[0].group("number"))
        if amount_semantics and header_percent:
            # A lone bare value in a two-measure row may be either column.
            return None, None
        if header_percent and not amount_semantics:
            candidate = number / 100.0
            return None, candidate if 0.0 <= candidate <= 1.0 else None

    for index, token in enumerate(tokens):
        number = _parse_number(token.group("number"))
        unit = token.group("unit")
        if unit in {"%", "％"}:
            candidate = number / 100.0
            if 0.0 <= candidate <= 1.0 and exposure_share is None:
                exposure_share = candidate
            continue
        if unit in _UNIT_MULTIPLIERS and exposure_value is None:
            exposure_value = number * _UNIT_MULTIPLIERS[unit]
            continue
        if (
            exposure_value is None
            and amount_semantics
            and header_multiplier is not None
        ):
            exposure_value = number * header_multiplier
            continue
        if index > 0 and exposure_share is None and header_percent:
            candidate = number / 100.0
            if 0.0 <= candidate <= 1.0:
                exposure_share = candidate
    return exposure_value, exposure_share


def _accepted_name_and_measures(
    name: str,
    exposure_value: float | None,
    exposure_share: float | None,
    *,
    has_economic_evidence: bool,
) -> tuple[str, float | None, float | None] | None:
    name_class = classify_counterparty_name(name)
    if name_class is CounterpartyNameClass.INVALID:
        return None
    # Pure dash/star markers are accepted only with positive economic evidence.
    # This separates masked disclosures from blank template rows.  Explicit
    # labels such as "Customer A" remain valid anonymous mentions even when the
    # source explicitly marks the amount/share as unavailable.
    if is_redacted_counterparty(name) and not any(
        value is not None and value > 0 for value in (exposure_value, exposure_share)
    ):
        return None
    if not has_economic_evidence:
        return None
    return name, exposure_value, exposure_share


def _parse_measures(
    body: str,
    *,
    relation: str,
    context: str,
    row_rank: int,
) -> tuple[str, float | None, float | None] | None:
    tokens = list(_NUMBER_TOKEN.finditer(body))
    candidates: list[
        tuple[tuple[int, int, int, int], tuple[str, float | None, float | None]]
    ] = []
    for token_index, token in enumerate(tokens):
        name = _valid_name_candidate(body[: token.start()])
        if name is None:
            continue
        # A prior standalone numeric token normally belongs to the measure
        # cells, not to the entity label.  The deliberately narrow exception
        # preserves disclosure masks such as ``客户 1`` and ``第 1 名``;
        # without this guard an invalid label such as ``/`` can be widened to
        # ``/ 224.00`` and incorrectly accepted at the percentage boundary.
        if _NUMBER_TOKEN.search(name) and not _name_encodes_rank(name, row_rank):
            continue
        tail = body[token.start() :].strip()
        if not _tail_is_qualified(tail):
            continue
        tail_tokens = tokens[token_index:]
        explicit_missing = bool(_EXPLICIT_MISSING_MEASURE.search(tail))
        positive_number = any(
            _parse_number(item.group("number")) > 0 for item in tail_tokens
        )
        exposure_value, exposure_share = _measure_values(
            tail,
            relation=relation,
            context=context,
        )
        accepted = _accepted_name_and_measures(
            name,
            exposure_value,
            exposure_share,
            has_economic_evidence=positive_number or explicit_missing,
        )
        if accepted is None:
            continue
        remaining_tokens = len(tail_tokens)
        score = (
            int(_name_encodes_rank(name, row_rank)),
            int(remaining_tokens == 2),
            int(exposure_value is not None) + int(exposure_share is not None),
            -abs(remaining_tokens - 2),
        )
        candidates.append((score, accepted))

    if not tokens:
        for missing in _EXPLICIT_MISSING_MEASURE.finditer(body):
            name = _valid_name_candidate(body[: missing.start()])
            if name is None:
                continue
            tail = body[missing.start() :].strip()
            if not _tail_is_qualified(tail):
                continue
            accepted = _accepted_name_and_measures(
                name,
                None,
                None,
                has_economic_evidence=True,
            )
            if accepted is not None:
                return accepted

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _row_start(line: str) -> tuple[int, str] | None:
    match = _ROW_PREFIX.match(line)
    if match is not None:
        return _RANK_VALUE[match.group("rank")], match.group("body")
    rank_only = _RANK_ONLY.match(line)
    if rank_only is not None:
        return _RANK_VALUE[rank_only.group("rank")], ""
    return None


def _assemble_candidate_row(
    lines: list[str],
    start_index: int,
    *,
    relation: str,
    header_context: str,
) -> tuple[int, str, float | None, float | None, str, int] | None:
    start = _row_start(lines[start_index])
    if start is None:
        return None
    rank, initial_body = start
    parts = [initial_body] if initial_body else []
    evidence_lines = [lines[start_index]]
    best: (
        tuple[
            tuple[int, int, int],
            tuple[int, str, float | None, float | None, str, int],
        ]
        | None
    ) = None
    cursor = start_index

    for extension_count in range(_MAX_ROW_CONTINUATION_LINES + 1):
        if parts:
            combined = " ".join(parts)
            parsed = _parse_measures(
                combined,
                relation=relation,
                context=header_context,
                row_rank=rank,
            )
            if parsed is not None:
                name, exposure_value, exposure_share = parsed
                evidence = " ".join(evidence_lines)
                score = (
                    int(exposure_value is not None) + int(exposure_share is not None),
                    int(bool(_EXPLICIT_MISSING_MEASURE.search(combined))),
                    -extension_count,
                )
                candidate = (
                    rank,
                    name,
                    exposure_value,
                    exposure_share,
                    evidence,
                    cursor,
                )
                if best is None or score > best[0]:
                    best = score, candidate
                if score[0] == 2 or score[1] == 1:
                    break

        if extension_count == _MAX_ROW_CONTINUATION_LINES:
            break
        next_index = cursor + 1
        while next_index < len(lines) and not lines[next_index]:
            next_index += 1
        if next_index >= len(lines):
            break
        next_line = lines[next_index]
        if (
            _section_marker(next_line) is not None
            or _row_start(next_line) is not None
            or _SECTION_BREAK.match(next_line)
        ):
            break
        parts.append(next_line)
        evidence_lines.append(next_line)
        cursor = next_index

    return best[1] if best is not None else None


def _assemble_unranked_candidate_row(
    lines: list[str],
    start_index: int,
    *,
    relation: str,
    header_context: str,
    row_rank: int,
) -> tuple[int, str, float | None, float | None, str, int] | None:
    """Assemble an unnumbered row under a source-grounded strong header."""

    parts = [lines[start_index]]
    evidence_lines = [lines[start_index]]
    best: (
        tuple[
            tuple[int, int, int],
            tuple[int, str, float | None, float | None, str, int],
        ]
        | None
    ) = None
    cursor = start_index

    for extension_count in range(_MAX_ROW_CONTINUATION_LINES + 1):
        combined = " ".join(parts)
        parsed = _parse_measures(
            combined,
            relation=relation,
            context=header_context,
            row_rank=row_rank,
        )
        if parsed is not None:
            name, exposure_value, exposure_share = parsed
            score = (
                int(exposure_value is not None) + int(exposure_share is not None),
                int(bool(_EXPLICIT_MISSING_MEASURE.search(combined))),
                -extension_count,
            )
            candidate = (
                row_rank,
                name,
                exposure_value,
                exposure_share,
                " ".join(evidence_lines),
                cursor,
            )
            if best is None or score > best[0]:
                best = score, candidate
            if score[0] == 2 or score[1] == 1:
                break

        if extension_count == _MAX_ROW_CONTINUATION_LINES:
            break
        next_index = cursor + 1
        while next_index < len(lines) and not lines[next_index]:
            next_index += 1
        if next_index >= len(lines):
            break
        next_line = lines[next_index]
        if (
            _section_marker(next_line) is not None
            or _row_start(next_line) is not None
            or _SECTION_BREAK.match(next_line)
            or _has_table_header(next_line, relation)
        ):
            break
        parts.append(next_line)
        evidence_lines.append(next_line)
        cursor = next_index

    return best[1] if best is not None else None


def _deduplicate_mentions(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_mentions()
    frame = pd.DataFrame(rows, columns=DISCLOSURE_RAW.columns)
    key = [
        "source_document_id",
        "source_company_id",
        "relationship_type",
        "counterparty_raw_name",
    ]
    reduced: list[dict[str, Any]] = []
    for _, group in frame.groupby(key, sort=True, dropna=False):
        first = group.sort_values("evidence_text", kind="mergesort").iloc[0].to_dict()
        for column in ("exposure_value", "exposure_share"):
            values = sorted({float(value) for value in group[column].dropna()})
            first[column] = values[0] if len(values) == 1 else None
        reduced.append(first)
    result = pd.DataFrame(reduced, columns=DISCLOSURE_RAW.columns)
    result = result.sort_values(key, kind="mergesort").reset_index(drop=True)
    result = coerce_disclosure_types(result)
    return validate_table(result, DISCLOSURE_RAW, allow_extra=False)


def extract_mentions_from_text(
    text: str,
    *,
    source_company_id: str,
    source_period_end: str | pd.Timestamp,
    publication_datetime: str | pd.Timestamp,
    source_document_id: str,
    source_document_url_or_path: str,
    source_announcement_time_ms: int | None = None,
    source_publication_datetime: str | pd.Timestamp | None = None,
    publication_time_precision: str | None = None,
    publication_timing_rule: str | None = None,
) -> TextExtractionResult:
    """Extract standardized customer/supplier rows without resolving names."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    metadata = {
        "source_company_id": source_company_id,
        "source_period_end": source_period_end,
        "publication_datetime": publication_datetime,
        "source_document_id": source_document_id,
        "source_document_url_or_path": source_document_url_or_path,
    }
    for field, value in metadata.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{field} must be non-empty")
    timing_audit = {
        "source_announcement_time_ms": source_announcement_time_ms,
        "source_publication_datetime": source_publication_datetime,
        "publication_time_precision": publication_time_precision,
        "publication_timing_rule": publication_timing_rule,
    }
    timing_fields_present = [value is not None for value in timing_audit.values()]
    if any(timing_fields_present) and not all(timing_fields_present):
        raise ValueError("publication timing audit fields must be supplied together")
    metadata.update(timing_audit)

    lines = [_clean_line(line) for line in text.splitlines()]
    rows: list[dict[str, Any]] = []
    sections: set[str] = set()
    current_relation: str | None = None
    header_seek: deque[str] = deque(maxlen=_MAX_HEADER_BLOCK_LINES)
    header_seek_lines = 0
    header_context: str | None = None
    header_mode: str | None = None
    expected_rank = 1
    lines_before_first_rank = 0
    line_number = 0

    while line_number < len(lines):
        line = lines[line_number]
        if not line:
            line_number += 1
            continue
        marker = _section_marker(line)
        # A split strong header can itself match a section pattern (for
        # example ``客户名称 / 销售额``).  While seeking a header for the same
        # relationship, let that line join the prior 1--2 fragments instead
        # of discarding them.  A new relation, a first marker, or a repeated
        # header after rows deliberately re-arms the parser.
        if marker is not None and (
            current_relation is None
            or marker != current_relation
            or header_context is not None
        ):
            current_relation = marker
            header_seek = deque([line], maxlen=_MAX_HEADER_BLOCK_LINES)
            header_seek_lines = 0
            header_mode = _table_header_mode(line, marker)
            header_context = line if header_mode is not None else None
            if _is_relationship_table_header(line, marker):
                sections.add(marker)
            expected_rank = 1
            lines_before_first_rank = 0
            line_number += 1
            continue
        if current_relation is None:
            line_number += 1
            continue
        if _SECTION_BREAK.match(line) and _row_start(line) is None:
            current_relation = None
            header_seek.clear()
            header_context = None
            header_mode = None
            line_number += 1
            continue

        if header_context is None:
            header_seek.append(line)
            header_seek_lines += 1
            candidate_header = "\n".join(header_seek)
            if _is_relationship_table_header(candidate_header, current_relation):
                sections.add(current_relation)
            header_mode = _table_header_mode(candidate_header, current_relation)
            if header_mode is not None:
                header_context = candidate_header
                expected_rank = 1
                lines_before_first_rank = 0
            elif header_seek_lines >= _MAX_HEADER_SEEK_LINES:
                current_relation = None
                header_mode = None
                header_seek.clear()
            line_number += 1
            continue

        if _PAGE_FURNITURE.fullmatch(line):
            line_number += 1
            continue

        if header_mode == "unranked":
            if expected_rank == 1 and _is_header_continuation(line):
                header_context = f"{header_context}\n{line}"
                line_number += 1
                continue
            candidate = _assemble_unranked_candidate_row(
                lines,
                line_number,
                relation=current_relation,
                header_context=header_context,
                row_rank=expected_rank,
            )
            if candidate is None:
                if expected_rank == 1:
                    lines_before_first_rank += 1
                    header_context = f"{header_context}\n{line}"
                    if lines_before_first_rank > _MAX_HEADER_SEEK_LINES:
                        current_relation = None
                        header_context = None
                        header_mode = None
                        header_seek.clear()
                else:
                    current_relation = None
                    header_context = None
                    header_mode = None
                    header_seek.clear()
                line_number += 1
                continue
        else:
            row_start = _row_start(line)
            if row_start is None:
                if expected_rank == 1:
                    lines_before_first_rank += 1
                    # Preserve split unit/percentage header fragments that follow
                    # the first line satisfying the strong-header predicate.
                    # Measure parsing needs the complete semantics even though the
                    # table was already safely armed by sequence/name/amount.
                    header_context = f"{header_context}\n{line}"
                    if lines_before_first_rank > _MAX_HEADER_SEEK_LINES:
                        current_relation = None
                        header_context = None
                        header_mode = None
                        header_seek.clear()
                else:
                    current_relation = None
                    header_context = None
                    header_mode = None
                    header_seek.clear()
                line_number += 1
                continue

            rank, _ = row_start
            if rank != expected_rank:
                current_relation = None
                header_context = None
                header_mode = None
                header_seek.clear()
                line_number += 1
                continue

            candidate = _assemble_candidate_row(
                lines,
                line_number,
                relation=current_relation,
                header_context=header_context,
            )
            if candidate is None:
                # A structurally valid rank can be an empty/zero template row or
                # an invalid placeholder.  It advances the table sequence without
                # becoming a mention so that a later valid rank remains capturable.
                expected_rank += 1
                line_number += 1
                if expected_rank > 5:
                    current_relation = None
                    header_context = None
                    header_mode = None
                    header_seek.clear()
                continue
        (
            _,
            raw_name,
            exposure_value,
            exposure_share,
            evidence_text,
            consumed_through,
        ) = candidate
        # The evidence is the normalized source line itself.  Keeping the raw
        # name verbatim inside it makes every accepted mention auditable.
        if raw_name not in evidence_text:
            current_relation = None
            header_context = None
            header_mode = None
            header_seek.clear()
            line_number += 1
            continue
        rows.append(
            {
                **metadata,
                "counterparty_raw_name": raw_name,
                "relationship_type": current_relation,
                "evidence_text": evidence_text,
                "exposure_value": exposure_value,
                "exposure_share": exposure_share,
            }
        )
        expected_rank += 1
        lines_before_first_rank = 0
        line_number = consumed_through + 1
        if expected_rank > 5:
            current_relation = None
            header_context = None
            header_mode = None
            header_seek.clear()

    mentions = _deduplicate_mentions(rows)
    ordered_sections = tuple(
        relation for relation in _RELATION_ORDER if relation in sections
    )
    return TextExtractionResult(mentions=mentions, section_types=ordered_sections)


def _metadata_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]):
            value = row[name]
            if not isinstance(value, str) or value.strip():
                return value
    return None


def _source_path(row: Mapping[str, Any], raw_root: Path) -> tuple[str, Path]:
    local = _metadata_value(row, "local_raw_path", "raw_local_path")
    if local is None:
        raise ValueError("document has no local_raw_path")
    local_text = str(local)
    path = Path(local_text)
    if not path.is_absolute():
        path = raw_root / path
    resolved_root = raw_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"document local_raw_path escapes raw_root: {local_text!r}"
        ) from exc
    return local_text, resolved_path


def _audit_row(
    row: Mapping[str, Any],
    *,
    source_period_end: str | None,
    section_types: tuple[str, ...] = (),
    mentions: pd.DataFrame | None = None,
    page_count: int = 0,
    pages_with_text: int = 0,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    mentions = _empty_mentions() if mentions is None else mentions
    anonymous = 0
    if len(mentions):
        anonymous = sum(
            _is_anonymous_label(
                str(item.counterparty_raw_name), str(item.relationship_type)
            )
            for item in mentions.itertuples(index=False)
        )
    return {
        "source_document_id": _metadata_value(row, "document_id", "source_document_id"),
        "source_company_id": _metadata_value(row, "security_id", "source_company_id"),
        "source_company_reported_id": _metadata_value(
            row, "reported_security_id", "source_company_reported_id"
        ),
        "source_company_org_id": _metadata_value(
            row, "cninfo_org_id", "source_company_org_id"
        ),
        "source_company_id_reconciliation_method": _metadata_value(
            row,
            "security_id_reconciliation_method",
            "source_company_id_reconciliation_method",
        ),
        "source_company_id_reconciliation_candidate_ids": _metadata_value(
            row,
            "security_id_reconciliation_candidate_ids",
            "source_company_id_reconciliation_candidate_ids",
        ),
        "source_company_id_reconciliation_candidate_count": _metadata_value(
            row,
            "security_id_reconciliation_candidate_count",
            "source_company_id_reconciliation_candidate_count",
        ),
        "source_period_end": source_period_end,
        "source_announcement_time_ms": _metadata_value(
            row, "announcement_time_ms", "source_announcement_time_ms"
        ),
        "source_publication_datetime": _metadata_value(
            row, "source_publication_datetime"
        ),
        "publication_datetime": _metadata_value(row, "publication_datetime"),
        "publication_time_precision": _metadata_value(
            row, "publication_time_precision"
        ),
        "publication_timing_rule": _metadata_value(row, "publication_timing_rule"),
        "source_url": _metadata_value(
            row,
            "adjunct_url",
            "source_url",
            "source_url_or_identifier",
        ),
        "local_raw_path": _metadata_value(row, "local_raw_path", "raw_local_path"),
        "retrieval_datetime": _metadata_value(row, "retrieval_datetime"),
        "sha256": _metadata_value(row, "sha256"),
        "contains_relationship_section": bool(section_types),
        "captured_relationship_section": bool(section_types and len(mentions)),
        "section_types": ",".join(section_types) or None,
        "mention_count": len(mentions),
        "named_mention_count": len(mentions) - anonymous,
        "anonymous_mention_count": anonymous,
        "page_count": page_count,
        "pages_with_text": pages_with_text,
        "extraction_status": status,
        "extraction_error": error,
    }


def _required_document_metadata(
    row: Mapping[str, Any],
) -> tuple[str, str, str, Any, str]:
    document_id = _metadata_value(row, "document_id", "source_document_id")
    company_id = _metadata_value(row, "security_id", "source_company_id")
    title = _metadata_value(row, "announcement_title")
    publication = _metadata_value(row, "publication_datetime")
    source = _metadata_value(
        row,
        "adjunct_url",
        "source_url",
        "source_url_or_identifier",
    )
    missing = [
        name
        for name, value in (
            ("document_id", document_id),
            ("security_id", company_id),
            ("announcement_title", title),
            ("publication_datetime", publication),
            ("source URL", source),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"document metadata missing: {', '.join(missing)}")
    return str(document_id), str(company_id), str(title), publication, str(source)


def _extract_pdf_document(
    row: Mapping[str, Any],
    *,
    raw_root: Path,
    reader_factory: Callable[..., Any],
    cancel_event: threading.Event | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    period: str | None = None
    try:
        _raise_if_cancelled(cancel_event)
        document_id, company_id, title, publication, source = (
            _required_document_metadata(row)
        )
        try:
            period = infer_source_period_end(title)
        except ValueError:
            # Some official CNINFO titles are only ``年报全文``.  Do not infer
            # from publication date: recover the explicit fiscal year from the
            # PDF cover after extracting source text instead.
            period = None
        _, pdf_path = _source_path(row, raw_root)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"annual-report PDF is missing: {pdf_path}")
        reader = reader_factory(str(pdf_path), strict=False)
        try:
            _raise_if_cancelled(cancel_event)
            page_count = len(reader.pages)
            page_text: list[str] = []
            page_errors: list[str] = []
            pages_with_text = 0
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    _raise_if_cancelled(cancel_event)
                    try:
                        extracted = page.extract_text() or ""
                    # Individual malformed PDF objects can surface several
                    # built-in exception types; retain the remaining pages.
                    except Exception as exc:  # noqa: BLE001
                        page_errors.append(
                            f"page {page_number}: {type(exc).__name__}: {exc}"
                        )
                        continue
                    if extracted.strip():
                        pages_with_text += 1
                        page_text.append(extracted)
                finally:
                    _close_resource(page)
        finally:
            _close_resource(reader)
        _raise_if_cancelled(cancel_event)
        if period is None and pages_with_text == 0:
            status = "PARTIAL" if page_errors else "NO_TEXT"
            error = (
                "; ".join(page_errors)
                if page_errors
                else "PDF text extraction produced no text"
            )
            return _empty_mentions(), _audit_row(
                row,
                source_period_end=None,
                page_count=page_count,
                pages_with_text=0,
                status=status,
                error=error,
            )
        if period is None:
            period = infer_source_period_end("\n".join(page_text[:10]))
        text_result = extract_mentions_from_text(
            "\n".join(page_text),
            source_company_id=company_id,
            source_period_end=period,
            publication_datetime=publication,
            source_document_id=document_id,
            source_document_url_or_path=source,
            source_announcement_time_ms=_metadata_value(
                row, "announcement_time_ms", "source_announcement_time_ms"
            ),
            source_publication_datetime=_metadata_value(
                row, "source_publication_datetime"
            ),
            publication_time_precision=_metadata_value(
                row, "publication_time_precision"
            ),
            publication_timing_rule=_metadata_value(row, "publication_timing_rule"),
        )
        if page_errors:
            status = "PARTIAL"
            error = "; ".join(page_errors)
        elif pages_with_text == 0:
            status = "NO_TEXT"
            error = "PDF text extraction produced no text"
        else:
            status = "SUCCESS"
            error = None
        audit = _audit_row(
            row,
            source_period_end=period,
            section_types=text_result.section_types,
            mentions=text_result.mentions,
            page_count=page_count,
            pages_with_text=pages_with_text,
            status=status,
            error=error,
        )
        return text_result.mentions, audit
    # Retain one audit row even for a malformed third-party document.  This is
    # the batch error boundary, not a silent recovery path.
    except Exception as exc:  # noqa: BLE001
        audit = _audit_row(
            row,
            source_period_end=period,
            status="ERROR",
            error=f"{type(exc).__name__}: {exc}",
        )
        return _empty_mentions(), audit


def _extract_document_job(
    job: _PdfExtractionJob,
    *,
    raw_root: Path,
    reader_factory: Callable[..., Any],
    cancel_event: threading.Event | None,
) -> _PdfExtractionOutcome:
    """Read one PDF without mutating shared batch state."""

    _raise_if_cancelled(cancel_event)
    mentions, audit = _extract_pdf_document(
        job.row,
        raw_root=raw_root,
        reader_factory=reader_factory,
        cancel_event=cancel_event,
    )
    _raise_if_cancelled(cancel_event)
    return _PdfExtractionOutcome(job=job, mentions=mentions, audit=audit)


def _log_extraction_progress(completed: int, total: int) -> None:
    if completed % 1000 == 0 or completed == total:
        LOGGER.info(
            "CNINFO PDF extraction progress completed=%d total=%d",
            completed,
            total,
        )


def _collect_parallel_outcomes(
    jobs: list[_PdfExtractionJob],
    *,
    submit_job: Callable[[_PdfExtractionJob], Future[_PdfExtractionOutcome]],
    cancel: Callable[[], None],
    max_workers: int,
) -> list[_PdfExtractionOutcome]:
    """Collect bounded futures without exposing completion order downstream."""

    outcomes: list[_PdfExtractionOutcome | None] = [None] * len(jobs)
    pending: dict[Future[_PdfExtractionOutcome], int] = {}
    pending_limit = min(len(jobs), max_workers * 2)
    next_job = 0
    completed_count = 0

    def submit_one() -> None:
        nonlocal next_job
        job = jobs[next_job]
        future = submit_job(job)
        pending[future] = job.ordinal
        next_job += 1

    try:
        while next_job < pending_limit:
            submit_one()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: pending[item]):
                ordinal = pending.pop(future)
                outcome = future.result()
                if outcome.job.ordinal != ordinal:
                    raise RuntimeError("PDF extraction outcome order identity changed")
                outcomes[ordinal] = outcome
                completed_count += 1
                _log_extraction_progress(completed_count, len(jobs))
                if next_job < len(jobs):
                    submit_one()
    except BaseException:
        cancel()
        for future in pending:
            future.cancel()
        raise

    if any(outcome is None for outcome in outcomes):
        raise RuntimeError("PDF extraction pool returned an incomplete outcome set")
    return [outcome for outcome in outcomes if outcome is not None]


def _parallel_extract_documents_threaded(
    jobs: list[_PdfExtractionJob],
    *,
    raw_root: Path,
    reader_factory: Callable[..., Any],
    max_workers: int,
) -> list[_PdfExtractionOutcome]:
    """Support injected reader factories without requiring them to be picklable."""

    cancel_event = threading.Event()
    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="cninfo-pdf",
    )
    try:
        outcomes = _collect_parallel_outcomes(
            jobs,
            submit_job=lambda job: executor.submit(
                _extract_document_job,
                job,
                raw_root=raw_root,
                reader_factory=reader_factory,
                cancel_event=cancel_event,
            ),
            cancel=cancel_event.set,
            max_workers=max_workers,
        )
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
        return outcomes


def _extract_document_process_job(
    job: _PdfExtractionJob,
    *,
    raw_root: Path,
) -> _PdfExtractionOutcome:
    """Spawn-safe production worker using one process-local PDFium document."""

    if _PROCESS_STARTED_QUEUE is None:
        raise RuntimeError("PDF process worker has no start-notification queue")
    _PROCESS_STARTED_QUEUE.put(job.ordinal)
    return _extract_document_job(
        job,
        raw_root=raw_root,
        reader_factory=_PdfiumReader,
        cancel_event=_PROCESS_CANCEL_EVENT,
    )


def _parallel_extract_documents_process(
    jobs: list[_PdfExtractionJob],
    *,
    raw_root: Path,
    max_workers: int,
    document_timeout_seconds: float,
) -> list[_PdfExtractionOutcome]:
    """Extract real PDFs with bounded work and a hard native-call timeout."""

    # PDFium is inherently not thread-safe.  Production extraction must keep
    # each document in an isolated process; never route _PdfiumReader through
    # the custom-reader ThreadPoolExecutor path.
    process_context = multiprocessing.get_context("spawn")
    cancel_event = process_context.Event()
    started_queue = process_context.Queue()
    pool = process_context.Pool(
        processes=max_workers,
        initializer=_initialize_pdf_process,
        initargs=(cancel_event, started_queue),
    )
    outcomes: list[_PdfExtractionOutcome | None] = [None] * len(jobs)
    pending: dict[int, tuple[Any, float]] = {}
    started_at: dict[int, float] = {}
    next_job = 0
    completed_count = 0

    def submit_one() -> None:
        nonlocal next_job
        job = jobs[next_job]
        result = pool.apply_async(
            _extract_document_process_job,
            args=(job,),
            kwds={"raw_root": raw_root},
        )
        pending[job.ordinal] = (result, time.monotonic())
        next_job += 1

    def terminate_and_join() -> None:
        cancel_event.set()
        pool.terminate()
        # A second Ctrl+C must not bypass process reaping after terminate().
        while True:
            try:
                pool.join()
                return
            except KeyboardInterrupt:
                continue

    def sleep_until_next_poll(seconds: float) -> None:
        """Sleep once and exclude only an external scheduling/suspend gap.

        Windows ``time.monotonic()`` advances while the host is suspended.  A
        multi-hour suspend during this 50 ms poll used to make every active
        document appear hung immediately after resume.  Moving the active job
        timestamps forward by only the measured oversleep preserves the
        requested polling time in the timeout budget.  A genuinely stuck
        worker therefore still expires after the configured amount of active
        watchdog time.
        """

        before_sleep = time.monotonic()
        time.sleep(seconds)
        after_sleep = time.monotonic()
        oversleep = after_sleep - before_sleep - seconds
        if oversleep <= _PDF_WATCHDOG_OVERSLEEP_TOLERANCE_SECONDS:
            return
        for ordinal, (result, submitted_at) in tuple(pending.items()):
            pending[ordinal] = (result, submitted_at + oversleep)
        for ordinal in tuple(started_at):
            if ordinal in pending:
                started_at[ordinal] += oversleep
        LOGGER.warning(
            "PDF extraction watchdog excluded scheduler/suspend gap seconds=%.3f "
            "active_documents=%d",
            oversleep,
            len(pending),
        )

    try:
        # Keeping at most one active document per worker prevents queued work
        # from consuming another document's timeout budget.  This is stricter
        # than the global <= 2 * workers pending bound.
        while next_job < min(max_workers, len(jobs)):
            submit_one()
        while pending:
            while True:
                try:
                    ordinal = int(started_queue.get_nowait())
                except Empty:
                    break
                if ordinal in pending:
                    started_at.setdefault(ordinal, time.monotonic())

            ready = sorted(
                ordinal for ordinal, (result, _) in pending.items() if result.ready()
            )
            for ordinal in ready:
                result, _ = pending.pop(ordinal)
                outcome = result.get()
                if outcome.job.ordinal != ordinal:
                    raise RuntimeError("PDF extraction outcome order identity changed")
                outcomes[ordinal] = outcome
                started_at.pop(ordinal, None)
                completed_count += 1
                _log_extraction_progress(completed_count, len(jobs))
                if next_job < len(jobs):
                    submit_one()
            if ready:
                continue

            now = time.monotonic()
            expired = []
            remaining = []
            for ordinal, (_, submitted_at) in pending.items():
                elapsed = now - started_at.get(ordinal, submitted_at)
                if elapsed >= document_timeout_seconds:
                    expired.append(ordinal)
                else:
                    remaining.append(document_timeout_seconds - elapsed)
            if expired:
                ordinal = min(expired)
                document_id = jobs[ordinal].row.get(
                    "document_id",
                    jobs[ordinal].row.get("source_document_id", ordinal),
                )
                raise PdfExtractionTimeoutError(
                    "PDF extraction exceeded the per-document timeout: "
                    f"document_id={document_id} ordinal={ordinal} "
                    f"timeout_seconds={document_timeout_seconds:g}"
                )
            sleep_until_next_poll(min(_PDF_WATCHDOG_POLL_SECONDS, min(remaining)))
        # Keep shutdown inside the protected region: Ctrl+C (or any other
        # BaseException) during close/join must still terminate native workers.
        pool.close()
        pool.join()
    except BaseException:
        try:
            terminate_and_join()
        finally:
            started_queue.close()
            started_queue.cancel_join_thread()
        raise
    else:
        started_queue.close()
        started_queue.join_thread()

    if any(outcome is None for outcome in outcomes):
        raise RuntimeError("PDF extraction pool returned an incomplete outcome set")
    return [outcome for outcome in outcomes if outcome is not None]


def extract_cninfo_documents(
    documents: pd.DataFrame,
    *,
    raw_root: str | Path = ".",
    reader_factory: Callable[..., Any] = _PdfiumReader,
    max_workers: int = 8,
    document_timeout_seconds: float = 300.0,
) -> PdfExtractionResult:
    """Extract all CNINFO annual-report PDFs and retain one audit row per document."""

    if not isinstance(documents, pd.DataFrame):
        raise TypeError("documents must be a pandas DataFrame")
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= PDF_MAX_EXTRACTION_WORKERS
    ):
        raise ValueError(
            f"max_workers must be an integer from 1 to {PDF_MAX_EXTRACTION_WORKERS}"
        )
    if (
        isinstance(document_timeout_seconds, bool)
        or not isinstance(document_timeout_seconds, (int, float))
        or not math.isfinite(float(document_timeout_seconds))
        or document_timeout_seconds <= 0
    ):
        raise ValueError("document_timeout_seconds must be a positive finite number")
    identifier_column = (
        "document_id" if "document_id" in documents.columns else "source_document_id"
    )
    if identifier_column not in documents.columns:
        raise ValueError("documents must contain document_id or source_document_id")
    if documents[identifier_column].isna().any():
        raise ValueError("document identifiers must be non-null")
    if documents[identifier_column].duplicated().any():
        raise ValueError("documents contain duplicate document identifiers")

    jobs = [
        _PdfExtractionJob(ordinal=ordinal, row=row)
        for ordinal, row in enumerate(documents.to_dict("records"))
    ]
    resolved_root = Path(raw_root)
    if reader_factory is _PdfiumReader:
        # The native backend always runs out of process, even with one worker,
        # so its per-document timeout remains a hard, terminable boundary.
        outcomes = _parallel_extract_documents_process(
            jobs,
            raw_root=resolved_root,
            max_workers=max_workers,
            document_timeout_seconds=float(document_timeout_seconds),
        )
    elif max_workers == 1:
        outcomes = []
        for job in jobs:
            outcomes.append(
                _extract_document_job(
                    job,
                    raw_root=resolved_root,
                    reader_factory=reader_factory,
                    cancel_event=None,
                )
            )
            _log_extraction_progress(len(outcomes), len(jobs))
    else:
        outcomes = _parallel_extract_documents_threaded(
            jobs,
            raw_root=resolved_root,
            reader_factory=reader_factory,
            max_workers=max_workers,
        )

    # Only the main thread merges pandas objects.  Outcomes are indexed by the
    # input ordinal, so completion timing can never change canonical artifacts.
    all_mentions = [outcome.mentions for outcome in outcomes if len(outcome.mentions)]
    audits = [outcome.audit for outcome in outcomes]

    combined = (
        _deduplicate_mentions(
            pd.concat(all_mentions, ignore_index=True).to_dict("records")
        )
        if all_mentions
        else _empty_mentions()
    )
    document_audit = pd.DataFrame(audits, columns=_AUDIT_COLUMNS)
    validate_table(document_audit, CNINFO_DOCUMENT_AUDIT)
    return PdfExtractionResult(mentions=combined, document_audit=document_audit)


__all__ = [
    "PdfExtractionResult",
    "PdfExtractionTimeoutError",
    "TextExtractionResult",
    "extract_cninfo_documents",
    "extract_mentions_from_text",
    "infer_source_period_end",
]
