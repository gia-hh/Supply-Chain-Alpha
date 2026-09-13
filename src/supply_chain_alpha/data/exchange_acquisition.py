"""Official SSE/SZSE security-master and point-in-time alias acquisition."""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import pandas as pd

from supply_chain_alpha.utils.network import require_official_https_url

from .acquisition import (
    RateLimitedSession,
    RawAsset,
    RawAssetStore,
    fetch_raw_asset,
    provenance_columns,
    request_fingerprint,
)
from .schemas import COMPANY_ALIAS, SECURITY_MASTER, validate_table

SSE_COMMON_QUERY_URL = "https://query.sse.com.cn/sseQuery/commonQuery.do"
SSE_SECURITY_SQL_ID = "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L"
SSE_TERMINATED_SECURITY_SQL_ID = "COMMON_SSE_CP_GPJCTPZ_GPLB_ZZGP_L"
SZSE_SHOW_REPORT_URL = "https://www.szse.cn/api/report/ShowReport"
SZSE_ACTIVE_CATALOG_ID = "1110"
SZSE_DELISTED_CATALOG_ID = "1793_ssgs"
SZSE_NAME_CHANGE_CATALOG_ID = "SSGSGMXX"

_SSE_A_SHARE_PREFIXES = ("600", "601", "603", "605", "688", "689")
_SZSE_A_SHARE_PREFIXES = ("000", "001", "002", "003", "300", "301", "302")
_MISSING_TEXT = frozenset({"", "-", "--", "—", "None", "nan", "NaN", "N/A"})


@dataclass(frozen=True)
class SecurityMasterAcquisition:
    securities: pd.DataFrame
    aliases: pd.DataFrame
    assets: tuple[RawAsset, ...]
    limitations: tuple[str, ...]


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return None if cleaned in _MISSING_TEXT else cleaned


def _ticker(value: Any) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    if re.fullmatch(r"\d+\.0", cleaned):
        cleaned = cleaned[:-2]
    if not cleaned.isdigit():
        raise ValueError(f"Official security code is not numeric: {cleaned!r}")
    if len(cleaned) > 6:
        raise ValueError(
            f"Official security code is longer than six digits: {cleaned!r}"
        )
    return cleaned.zfill(6)


def _iso_date(value: Any) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    digits = cleaned.replace("/", "").replace("-", "").replace(".", "")
    if re.fullmatch(r"\d{8}", digits):
        cleaned = f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    try:
        return pd.Timestamp(cleaned).date().isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Official date is invalid: {value!r}") from exc


def _board(exchange: str, ticker: str, source_value: Any = None) -> str:
    explicit = _clean_text(source_value)
    if exchange == "SSE":
        if explicit in {"8", "科创板", "STAR"}:
            return "STAR"
        if explicit and explicit not in {"1", "A", "A股", "人民币普通股", "主板"}:
            return explicit
        return "STAR" if ticker.startswith(("688", "689")) else "MAIN"
    if explicit and explicit not in {"1", "A", "A股", "人民币普通股", "主板"}:
        if explicit in {"创业板", "CHINEXT"}:
            return "CHINEXT"
        return explicit
    return "CHINEXT" if ticker.startswith(("300", "301", "302")) else "MAIN"


def _validate_json(content: bytes) -> None:
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Official endpoint did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError("Official endpoint JSON root must be an object")


def _validate_xlsx(content: bytes) -> None:
    if not zipfile.is_zipfile(BytesIO(content)):
        raise ValueError("SZSE ShowReport did not return an XLSX workbook")


def _extract_sse_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result", [])
    if isinstance(result, dict):
        for key in ("data", "result", "rows"):
            candidate = result.get(key)
            if isinstance(candidate, list):
                result = candidate
                break
    if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
        raise ValueError("SSE commonQuery result must be a list of objects")
    return result


def _sse_page_count(
    payload: dict[str, Any], *, rows: int, page_size: int, page: int
) -> int:
    page_help = payload.get("pageHelp")
    if isinstance(page_help, dict):
        for key in ("pageCount", "pagecount", "totalPages", "totalpages"):
            value = page_help.get(key)
            if value not in (None, ""):
                try:
                    return max(page, int(value))
                except (TypeError, ValueError):
                    break
        total = page_help.get("total") or page_help.get("totalCount")
        if total not in (None, ""):
            try:
                return max(page, math.ceil(int(total) / page_size))
            except (TypeError, ValueError):
                pass
    return page if rows < page_size else page + 1


def _canonical_security_columns() -> list[str]:
    return [
        *SECURITY_MASTER.required_columns,
        "company_name_kind",
        "source_industry_name_current",
        "security_status",
        "source",
        "source_url_or_identifier",
        "retrieval_datetime",
        "file_size",
        "sha256",
        "raw_local_path",
        "cache_hit",
    ]


def normalize_sse_security_master(
    rows: list[dict[str, Any]] | pd.DataFrame,
    asset: RawAsset,
) -> pd.DataFrame:
    """Normalize official SSE ``result`` fields without losing provenance."""

    raw = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    for record in raw.to_dict("records"):
        ticker = _ticker(record.get("A_STOCK_CODE"))
        if ticker is None:
            continue
        if not ticker.startswith(_SSE_A_SHARE_PREFIXES):
            continue
        company_name = _clean_text(record.get("FULL_NAME"))
        if company_name is None:
            raise ValueError(f"SSE security {ticker} has no FULL_NAME")
        delisting_date = _iso_date(record.get("DELIST_DATE"))
        output.append(
            {
                "security_id": f"SSE:{ticker}",
                "ticker": ticker,
                "exchange": "SSE",
                "company_name": company_name,
                "listing_date": _iso_date(record.get("LIST_DATE")),
                "delisting_date": delisting_date,
                "board": _board("SSE", ticker, record.get("STOCK_TYPE")),
                "company_name_kind": "legal_full_name",
                # Current industry is deliberately segregated: it is not PIT history.
                "source_industry_name_current": None,
                "security_status": "delisted" if delisting_date else "active",
                **provenance_columns(asset),
            }
        )
    return pd.DataFrame(output, columns=_canonical_security_columns())


def _find_column(
    frame: pd.DataFrame, candidates: tuple[str, ...], *, required: bool = True
) -> str | None:
    normalized = {
        re.sub(r"\s+", "", str(column)): str(column) for column in frame.columns
    }
    for candidate in candidates:
        found = normalized.get(re.sub(r"\s+", "", candidate))
        if found is not None:
            return found
    if required:
        raise ValueError(f"Official workbook is missing one of columns: {candidates}")
    return None


def parse_szse_workbook(
    content: bytes, *, code_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Locate the actual header row in an official ShowReport XLSX response."""

    _validate_xlsx(content)
    raw = pd.read_excel(BytesIO(content), header=None, dtype=str)
    header_index: int | None = None
    normalized_codes = {re.sub(r"\s+", "", column) for column in code_columns}
    for index in range(min(20, len(raw))):
        row_values = {
            re.sub(r"\s+", "", str(value))
            for value in raw.iloc[index].dropna().tolist()
        }
        if row_values & normalized_codes:
            header_index = index
            break
    if header_index is None:
        raise ValueError(
            f"Unable to locate ShowReport header containing {code_columns}"
        )
    columns = [
        str(value).strip() if not pd.isna(value) else f"unnamed_{position}"
        for position, value in enumerate(raw.iloc[header_index])
    ]
    if len(set(columns)) != len(columns):
        raise ValueError("SZSE ShowReport contains duplicate column names")
    frame = raw.iloc[header_index + 1 :].copy()
    frame.columns = columns
    return frame.dropna(how="all").reset_index(drop=True)


def normalize_szse_security_master(
    active: pd.DataFrame,
    delisted: pd.DataFrame,
    *,
    active_asset: RawAsset,
    delisted_asset: RawAsset,
) -> pd.DataFrame:
    """Normalize active and terminated SZSE A-share workbooks."""

    output: list[dict[str, Any]] = []
    specifications = (
        (
            active,
            active_asset,
            "active",
            ("A股代码", "证券代码"),
            ("公司全称", "公司名称"),
            ("A股上市日期", "上市日期"),
            None,
        ),
        (
            delisted,
            delisted_asset,
            "delisted",
            ("证券代码", "A股代码"),
            ("公司全称", "公司名称", "证券简称"),
            ("上市日期", "A股上市日期"),
            ("终止上市日期", "摘牌日期"),
        ),
    )
    for (
        frame,
        asset,
        status,
        code_names,
        company_names,
        listing_names,
        delisting_names,
    ) in specifications:
        code_column = _find_column(frame, code_names)
        company_column = _find_column(frame, company_names)
        listing_column = _find_column(frame, listing_names)
        delisting_column = (
            _find_column(frame, delisting_names) if delisting_names else None
        )
        board_column = _find_column(frame, ("板块",), required=False)
        industry_column = _find_column(frame, ("所属行业", "行业"), required=False)
        for record in frame.to_dict("records"):
            ticker = _ticker(record.get(code_column))
            if ticker is None or not ticker.startswith(_SZSE_A_SHARE_PREFIXES):
                continue
            company_name = _clean_text(record.get(company_column))
            if company_name is None:
                raise ValueError(f"SZSE security {ticker} has no company name")
            output.append(
                {
                    "security_id": f"SZSE:{ticker}",
                    "ticker": ticker,
                    "exchange": "SZSE",
                    "company_name": company_name,
                    "listing_date": _iso_date(record.get(listing_column)),
                    "delisting_date": _iso_date(record.get(delisting_column))
                    if delisting_column
                    else None,
                    "board": _board("SZSE", ticker, record.get(board_column))
                    if board_column
                    else _board("SZSE", ticker),
                    "company_name_kind": "legal_full_name"
                    if status == "active" or company_column in {"公司全称", "公司名称"}
                    else "security_short_name",
                    "source_industry_name_current": _clean_text(
                        record.get(industry_column)
                    )
                    if industry_column
                    else None,
                    "security_status": status,
                    **provenance_columns(asset),
                }
            )
    return pd.DataFrame(output, columns=_canonical_security_columns())


def normalize_szse_name_changes(frame: pd.DataFrame, asset: RawAsset) -> pd.DataFrame:
    """Normalize the official SZSE full legal-name change table."""

    code_column = _find_column(frame, ("证券代码", "公司代码", "A股代码"))
    old_column = _find_column(
        frame,
        ("变更前公司全称", "原公司全称", "变更前全称", "原全称", "原公司名称"),
    )
    new_column = _find_column(
        frame,
        ("变更后公司全称", "变更后全称", "现公司全称", "新公司全称", "公司全称"),
    )
    date_column = _find_column(frame, ("变更日期", "更名日期", "公告日期"))
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        ticker = _ticker(record.get(code_column))
        old_name = _clean_text(record.get(old_column))
        new_name = _clean_text(record.get(new_column))
        change_date = _iso_date(record.get(date_column))
        if (
            ticker is None
            or old_name is None
            or new_name is None
            or change_date is None
        ):
            continue
        rows.append(
            {
                "security_id": f"SZSE:{ticker}",
                "ticker": ticker,
                "old_name": old_name,
                "new_name": new_name,
                "change_date": change_date,
                **provenance_columns(asset),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return (
        result.sort_values(
            ["security_id", "change_date", "old_name", "new_name"], kind="mergesort"
        )
        .drop_duplicates(
            ["security_id", "change_date", "old_name", "new_name"], keep="first"
        )
        .reset_index(drop=True)
    )


def combine_security_masters(*frames: pd.DataFrame) -> pd.DataFrame:
    """Deterministically coalesce official records to one security identifier."""

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=_canonical_security_columns())
    records: list[dict[str, Any]] = []
    for security_id, group in combined.groupby("security_id", sort=True):
        for column in ("ticker", "exchange"):
            values = sorted(group[column].dropna().astype(str).unique().tolist())
            if len(values) != 1:
                raise ValueError(f"Conflicting {column} for {security_id}: {values}")
        ranked = group.assign(
            _delisted=group["delisting_date"].notna().astype(int),
            _complete=group.notna().sum(axis=1),
        ).sort_values(
            ["_delisted", "_complete", "source_url_or_identifier"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        chosen = ranked.iloc[0].drop(labels=["_delisted", "_complete"]).to_dict()
        for column in ("listing_date", "delisting_date", "board"):
            values = ranked[column].dropna().astype(str).tolist()
            if values:
                chosen[column] = values[0]
        legal_names = (
            ranked.loc[
                ranked["company_name_kind"].eq("legal_full_name"), "company_name"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )
        if legal_names:
            chosen["company_name"] = legal_names[0]
            chosen["company_name_kind"] = "legal_full_name"
        records.append(chosen)
    result = (
        pd.DataFrame(records, columns=_canonical_security_columns())
        .sort_values(["exchange", "ticker"], kind="mergesort")
        .reset_index(drop=True)
    )
    validate_table(result, SECURITY_MASTER)
    return result


def _retrieval_date(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Raw retrieval timestamp must be timezone-aware")
    return timestamp.tz_convert("Asia/Shanghai").date().isoformat()


def build_company_aliases(
    securities: pd.DataFrame,
    name_changes: pd.DataFrame,
) -> pd.DataFrame:
    """Build only time intervals supported by official source evidence.

    Ticker aliases use official listing/delisting intervals.  Current legal
    names without a change-history proof start at the retrieval date, never at
    listing.  This deliberately prevents present-day names leaking backward.
    """

    validate_table(securities, SECURITY_MASTER)
    security_index = securities.set_index("security_id", drop=False)
    rows: list[dict[str, Any]] = []

    for security in securities.to_dict("records"):
        valid_from = security.get("listing_date")
        valid_to = security.get("delisting_date")
        valid_from = None if pd.isna(valid_from) else valid_from
        valid_to = None if pd.isna(valid_to) else valid_to
        if valid_from is not None and (valid_to is None or valid_from < valid_to):
            rows.append(
                {
                    "entity_id": security["security_id"],
                    "canonical_name": security["company_name"],
                    "alias": security["ticker"],
                    "alias_type": "ticker",
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "source": f"{security['exchange']} official listing interval",
                }
            )

    if not name_changes.empty:
        for security_id, changes in name_changes.groupby("security_id", sort=True):
            if security_id not in security_index.index:
                continue
            security = security_index.loc[security_id]
            changes = changes.sort_values(
                ["change_date", "old_name", "new_name"], kind="mergesort"
            ).drop_duplicates(["change_date", "old_name", "new_name"])
            change_records = changes.to_dict("records")
            for index, change in enumerate(change_records):
                change_date = change["change_date"]
                listing_date = security["listing_date"]
                delisting_date = security["delisting_date"]
                listing_date = None if pd.isna(listing_date) else listing_date
                delisting_date = None if pd.isna(delisting_date) else delisting_date
                prior_start = (
                    listing_date
                    if index == 0
                    else change_records[index - 1]["change_date"]
                )
                if prior_start is not None and prior_start < change_date:
                    rows.append(
                        {
                            "entity_id": security_id,
                            "canonical_name": security["company_name"],
                            "alias": change["old_name"],
                            "alias_type": "legal_name_history",
                            "valid_from": prior_start,
                            "valid_to": change_date,
                            "source": "SZSE SSGSGMXX official legal-name history",
                        }
                    )
                next_date = (
                    change_records[index + 1]["change_date"]
                    if index + 1 < len(change_records)
                    else delisting_date
                )
                if next_date is None or change_date < next_date:
                    rows.append(
                        {
                            "entity_id": security_id,
                            "canonical_name": security["company_name"],
                            "alias": change["new_name"],
                            "alias_type": "legal_name_history",
                            "valid_from": change_date,
                            "valid_to": next_date,
                            "source": "SZSE SSGSGMXX official legal-name history",
                        }
                    )

    # Snapshot names are useful going forward, but cannot claim pre-retrieval validity.
    for security in securities.to_dict("records"):
        if security["security_status"] == "delisted":
            continue
        snapshot_start = _retrieval_date(security["retrieval_datetime"])
        delisting_date = security["delisting_date"]
        delisting_date = None if pd.isna(delisting_date) else delisting_date
        already_supported = any(
            row["entity_id"] == security["security_id"]
            and row["alias"] == security["company_name"]
            and row["valid_from"] <= snapshot_start
            and (row["valid_to"] is None or snapshot_start < row["valid_to"])
            for row in rows
        )
        if not already_supported:
            rows.append(
                {
                    "entity_id": security["security_id"],
                    "canonical_name": security["company_name"],
                    "alias": security["company_name"],
                    "alias_type": "legal_name_current_snapshot",
                    "valid_from": snapshot_start,
                    "valid_to": delisting_date,
                    "source": (
                        f"{security['exchange']} current-name snapshot; "
                        "validity begins at raw retrieval"
                    ),
                }
            )

    aliases = pd.DataFrame(rows, columns=COMPANY_ALIAS.required_columns)
    aliases = (
        aliases.drop_duplicates(["entity_id", "alias", "valid_from"], keep="last")
        .sort_values(
            ["entity_id", "valid_from", "alias_type", "alias"], kind="mergesort"
        )
        .reset_index(drop=True)
    )
    validate_table(aliases, COMPANY_ALIAS)
    return aliases


def _fetch_sse(
    transport: RateLimitedSession,
    store: RawAssetStore,
    *,
    endpoint: str,
    page_size: int,
) -> tuple[pd.DataFrame, list[RawAsset]]:
    frames: list[pd.DataFrame] = []
    assets: list[RawAsset] = []
    # These parameter sets mirror the SSE stock-list page's own official
    # JavaScript.  ``type=inParams`` is material for STAR results, and the
    # terminated catalogue uses a distinct SQL id.  Querying STOCK_TYPE=1,8
    # covers Main Board A shares and STAR while excluding B shares.
    queries = (
        (
            "active",
            SSE_SECURITY_SQL_ID,
            "2,4,5,7,8",
        ),
        (
            "delisted",
            SSE_TERMINATED_SECURITY_SQL_ID,
            "3",
        ),
    )
    for label, sql_id, company_status in queries:
        page = 1
        page_count = 1
        while page <= page_count:
            params = {
                "sqlId": sql_id,
                "type": "inParams",
                "isPagination": "true",
                "STOCK_TYPE": "1,8",
                "COMPANY_STATUS": company_status,
                "REG_PROVINCE": "",
                "CSRC_CODE": "",
                "STOCK_CODE": "",
                "pageHelp.cacheSize": "1",
                "pageHelp.pageSize": str(page_size),
                "pageHelp.pageNo": str(page),
                "pageHelp.beginPage": str(page),
                "pageHelp.endPage": str(page),
            }
            fingerprint = request_fingerprint("GET", endpoint, params=params)
            path = (
                f"sse/security_master/{label}/page_{page:05d}_{fingerprint[:16]}.json"
            )
            asset, content = fetch_raw_asset(
                transport,
                store,
                path,
                source="SSE",
                method="GET",
                url=endpoint,
                params=params,
                headers={
                    "Referer": "https://www.sse.com.cn/assortment/stock/list/share/"
                },
                validate_content=_validate_json,
            )
            payload = json.loads(content.decode("utf-8-sig"))
            rows = _extract_sse_rows(payload)
            frames.append(normalize_sse_security_master(rows, asset))
            assets.append(asset)
            page_count = _sse_page_count(
                payload,
                rows=len(rows),
                page_size=page_size,
                page=page,
            )
            if page_count > 10_000:
                raise ValueError(
                    f"SSE endpoint reported unreasonable page count: {page_count}"
                )
            if not rows:
                break
            page += 1
    return pd.concat(frames, ignore_index=True), assets


def _fetch_szse_workbook(
    transport: RateLimitedSession,
    store: RawAssetStore,
    *,
    endpoint: str,
    catalog_id: str,
    tab_key: str,
    label: str,
    code_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, RawAsset]:
    params = {
        "SHOWTYPE": "xlsx",
        "CATALOGID": catalog_id,
        "ENCODE": "1",
        "TABKEY": tab_key,
    }
    fingerprint = request_fingerprint("GET", endpoint, params=params)
    path = f"szse/security_master/{label}_{fingerprint[:16]}.xlsx"
    asset, content = fetch_raw_asset(
        transport,
        store,
        path,
        source="SZSE",
        method="GET",
        url=endpoint,
        params=params,
        headers={"Referer": "https://www.szse.cn/market/"},
        validate_content=_validate_xlsx,
    )
    return parse_szse_workbook(content, code_columns=code_columns), asset


def acquire_official_security_master(
    transport: RateLimitedSession,
    raw_store: RawAssetStore,
    *,
    sse_url: str = SSE_COMMON_QUERY_URL,
    szse_url: str = SZSE_SHOW_REPORT_URL,
    sse_page_size: int = 1_000,
) -> SecurityMasterAcquisition:
    """Acquire SSE and SZSE active/delisted common equities and aliases."""

    require_official_https_url(
        sse_url,
        label="SSE security-master URL",
        allowed_hosts={"query.sse.com.cn"},
        expected_path="/sseQuery/commonQuery.do",
    )
    require_official_https_url(
        szse_url,
        label="SZSE security-master URL",
        allowed_hosts={"www.szse.cn"},
        expected_path="/api/report/ShowReport",
    )

    sse, sse_assets = _fetch_sse(
        transport,
        raw_store,
        endpoint=sse_url,
        page_size=sse_page_size,
    )
    active, active_asset = _fetch_szse_workbook(
        transport,
        raw_store,
        endpoint=szse_url,
        catalog_id=SZSE_ACTIVE_CATALOG_ID,
        tab_key="tab1",
        label="active",
        code_columns=("A股代码", "证券代码"),
    )
    delisted, delisted_asset = _fetch_szse_workbook(
        transport,
        raw_store,
        endpoint=szse_url,
        catalog_id=SZSE_DELISTED_CATALOG_ID,
        tab_key="tab2",
        label="delisted",
        code_columns=("证券代码", "A股代码"),
    )
    changes, changes_asset = _fetch_szse_workbook(
        transport,
        raw_store,
        endpoint=szse_url,
        catalog_id=SZSE_NAME_CHANGE_CATALOG_ID,
        tab_key="tab1",
        label="legal_name_changes",
        code_columns=("证券代码", "公司代码", "A股代码"),
    )
    szse = normalize_szse_security_master(
        active,
        delisted,
        active_asset=active_asset,
        delisted_asset=delisted_asset,
    )
    securities = combine_security_masters(sse, szse)
    name_changes = normalize_szse_name_changes(changes, changes_asset)
    aliases = build_company_aliases(securities, name_changes)
    limitations = (
        "Current industry labels are preserved only as source_industry_name_current and must not be used historically.",
        "Where a terminated-company workbook exposes only a security short name, company_name_kind records that limitation.",
        "Current legal names lacking official change history begin at retrieval date, not listing date.",
        "SSE legal-name history is not inferred from the current commonQuery snapshot.",
    )
    return SecurityMasterAcquisition(
        securities=securities,
        aliases=aliases,
        assets=(*sse_assets, active_asset, delisted_asset, changes_asset),
        limitations=limitations,
    )


def company_alias_diagnostics(aliases: pd.DataFrame) -> dict[str, Any]:
    validate_table(aliases, COMPANY_ALIAS)
    return {
        "rows": len(aliases),
        "entities": int(aliases["entity_id"].nunique()),
        "alias_types": aliases["alias_type"].value_counts().sort_index().to_dict(),
        "snapshot_only_rows": int(
            aliases["alias_type"].eq("legal_name_current_snapshot").sum()
        ),
        "missing_valid_from": int(aliases["valid_from"].isna().sum()),
    }


__all__ = [
    "SSE_COMMON_QUERY_URL",
    "SSE_SECURITY_SQL_ID",
    "SSE_TERMINATED_SECURITY_SQL_ID",
    "SZSE_ACTIVE_CATALOG_ID",
    "SZSE_DELISTED_CATALOG_ID",
    "SZSE_NAME_CHANGE_CATALOG_ID",
    "SZSE_SHOW_REPORT_URL",
    "SecurityMasterAcquisition",
    "acquire_official_security_master",
    "build_company_aliases",
    "combine_security_masters",
    "company_alias_diagnostics",
    "normalize_sse_security_master",
    "normalize_szse_name_changes",
    "normalize_szse_security_master",
    "parse_szse_workbook",
]
