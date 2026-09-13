"""CNINFO annual-report metadata and immutable document acquisition."""

from __future__ import annotations

import html
import json
import logging
import math
import re
import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from datetime import time as datetime_time
from enum import Enum
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from supply_chain_alpha.utils.network import require_official_https_url

from .acquisition import (
    AcquisitionMode,
    RateLimitedSession,
    RawAsset,
    RawAssetIntegrityError,
    RawAssetStore,
    fetch_raw_asset,
    provenance_columns,
    request_fingerprint,
    utc_now,
)

CNINFO_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_DOWNLOAD_BASE_URL = "https://static.cninfo.com.cn/"
CNINFO_SZSE_STOCK_MAP_URL = "https://www.cninfo.com.cn/new/data/szse_stock.json"
CNINFO_ANNUAL_REPORT_CATEGORY = "category_ndbg_szsh"
CNINFO_MAX_RETRIEVABLE_PAGES = 100
CNINFO_MAX_DOWNLOAD_WORKERS = 8
CNINFO_PUBLICATION_TIMEZONE = "Asia/Shanghai"
CNINFO_SIGNAL_CUTOFF_LOCAL = "15:00:00"
PUBLICATION_TIME_PRECISION_EXACT = "exact"
PUBLICATION_TIME_PRECISION_DATE_ONLY = "date_only"
PUBLICATION_TIMING_RULE_EXACT = "cninfo_source_timestamp"
PUBLICATION_TIMING_RULE_DATE_ONLY = "next_calendar_day_midnight"
RECONCILIATION_REPORTED_ID_IN_MASTER = "reported_id_in_master"
RECONCILIATION_REPORTED_ID_IN_MASTER_MISSING_ORG_ID = (
    "reported_id_in_master_missing_org_id"
)
RECONCILIATION_ORG_ID_UNIQUE_MASTER_MATCH = "org_id_unique_master_match"
RECONCILIATION_UNRESOLVED_MISSING_ORG_ID = "unresolved_missing_org_id"
RECONCILIATION_UNRESOLVED_NO_MASTER_MATCH = "unresolved_no_master_match"
RECONCILIATION_UNRESOLVED_AMBIGUOUS_MASTER_MATCH = "unresolved_ambiguous_master_match"
SECURITY_ID_RECONCILIATION_METHODS = frozenset(
    {
        RECONCILIATION_REPORTED_ID_IN_MASTER,
        RECONCILIATION_REPORTED_ID_IN_MASTER_MISSING_ORG_ID,
        RECONCILIATION_ORG_ID_UNIQUE_MASTER_MATCH,
        RECONCILIATION_UNRESOLVED_MISSING_ORG_ID,
        RECONCILIATION_UNRESOLVED_NO_MASTER_MATCH,
        RECONCILIATION_UNRESOLVED_AMBIGUOUS_MASTER_MATCH,
    }
)
CNINFO_SHORT_NAME_ALIAS_TYPE = "cninfo_official_short_name"
CNINFO_SHORT_NAME_ALIAS_SOURCE = (
    "CNINFO official annual-report metadata; first_observed_document_id="
)
CNINFO_OBSERVED_TICKER_ALIAS_TYPE = "cninfo_official_observed_ticker"
CNINFO_OBSERVED_TICKER_ALIAS_SOURCE = (
    "CNINFO official annual-report metadata; first_observed_document_id="
)
LOGGER = logging.getLogger(__name__)
_ANNUAL_REPORT_PATTERN = re.compile(r"(?:年度报告|年报)")
_EXCLUDED_TITLE_PATTERN = re.compile(r"(?:摘要|英文版|取消)")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_SSE_A_SHARE_PREFIXES = ("600", "601", "603", "605", "688", "689")
_SZSE_A_SHARE_PREFIXES = ("000", "001", "002", "003", "300", "301", "302")


class DownloadFailureCategory(str, Enum):
    """Machine-readable cause class for a failed document acquisition."""

    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    CONTENT_OR_INTEGRITY_ERROR = "CONTENT_OR_INTEGRITY_ERROR"


@dataclass(frozen=True)
class DownloadFailure:
    document_id: str
    source_url_or_identifier: str
    attempted_at: str
    reason: str
    failure_category: DownloadFailureCategory = (
        DownloadFailureCategory.CONTENT_OR_INTEGRITY_ERROR
    )

    def to_dict(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "source_url_or_identifier": self.source_url_or_identifier,
            "attempted_at": self.attempted_at,
            "reason": self.reason,
            "failure_category": self.failure_category.value,
        }


@dataclass(frozen=True)
class _DocumentDownloadJob:
    ordinal: int
    frame_index: int
    document_id: str
    source_url_or_identifier: str
    relative_path: str
    extension: str


@dataclass(frozen=True)
class _DocumentDownloadOutcome:
    job: _DocumentDownloadJob
    asset: RawAsset | None = None
    failure: DownloadFailure | None = None


@dataclass(frozen=True)
class DisclosureAcquisitionResult:
    documents: pd.DataFrame
    assets: tuple[RawAsset, ...]
    failures: tuple[DownloadFailure, ...]
    mode: AcquisitionMode
    failure_log_asset: RawAsset | None = None
    timing_excluded_after_query_end_count: int = 0


@dataclass(frozen=True)
class CninfoStockMapAcquisition:
    stock_map: pd.DataFrame
    asset: RawAsset


@dataclass(frozen=True)
class PublicationTiming:
    """Source timestamp plus the conservative instant permitted in PIT research."""

    announcement_time_ms: int
    source_publication_datetime: str
    publication_datetime: str
    publication_time_precision: str
    publication_timing_rule: str


def _utc_timestamp(clock: Any = utc_now) -> str:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Acquisition clock must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_title(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("CNINFO announcementTitle must be a string")
    return re.sub(r"\s+", " ", html.unescape(_HTML_TAG_PATTERN.sub("", value))).strip()


def is_annual_report_title(title: str, security_name: str | None = None) -> bool:
    cleaned = _clean_title(title)
    report_match = _ANNUAL_REPORT_PATTERN.search(cleaned)
    if report_match is None or _EXCLUDED_TITLE_PATTERN.search(cleaned):
        return False
    if security_name is None or re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", cleaned):
        return True

    # CNINFO's annual-report category occasionally includes an issuer's
    # attachment for an unrelated investee (for example ``NTU年报`` filed by a
    # Chinese A-share issuer).  For titles without an explicit year, require
    # either a generic report title or an issuer-identifying prefix.
    compact_prefix = re.sub(r"\s+", "", cleaned[: report_match.start()])
    if compact_prefix in {"", "A股", "H股", "B股", "A股公告", "H股公告", "B股公告"}:
        return True
    compact_security = re.sub(r"\s+", "", str(security_name))
    security_without_status = re.sub(
        r"^(?:S\*ST|\*ST|SST|ST)", "", compact_security, flags=re.IGNORECASE
    )
    return any(
        candidate and candidate in compact_prefix
        for candidate in (compact_security, security_without_status)
    )


def _announcement_time(milliseconds: Any) -> PublicationTiming:
    if isinstance(milliseconds, bool):
        raise TypeError("CNINFO announcementTime must be epoch milliseconds")
    try:
        value = int(milliseconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CNINFO announcementTime must be epoch milliseconds") from exc
    if value <= 0:
        raise ValueError("CNINFO announcementTime must be positive")
    source_publication = datetime.fromtimestamp(
        value / 1000, tz=timezone.utc
    ).astimezone(ZoneInfo(CNINFO_PUBLICATION_TIMEZONE))
    # CNINFO's official archive encodes most historical announcementTime values
    # as exactly local midnight.  That is a date placeholder, not evidence that
    # the filing was public at 00:00.  Delay those rows until the next calendar
    # day so a 15:00 signal can never consume them on the source date.  This rule
    # is deterministic and deliberately does not consult future trading calendars.
    date_only = source_publication.time() == datetime_time.min
    if date_only:
        publication = source_publication + timedelta(days=1)
        precision = PUBLICATION_TIME_PRECISION_DATE_ONLY
        rule = PUBLICATION_TIMING_RULE_DATE_ONLY
    else:
        publication = source_publication
        precision = PUBLICATION_TIME_PRECISION_EXACT
        rule = PUBLICATION_TIMING_RULE_EXACT
    return PublicationTiming(
        announcement_time_ms=value,
        source_publication_datetime=source_publication.isoformat(),
        publication_datetime=publication.isoformat(),
        publication_time_precision=precision,
        publication_timing_rule=rule,
    )


def publication_timing_diagnostics(
    frame: pd.DataFrame,
    *,
    epoch_column: str = "announcement_time_ms",
    signal_cutoff_local: str = CNINFO_SIGNAL_CUTOFF_LOCAL,
    market_timezone: str = CNINFO_PUBLICATION_TIMEZONE,
) -> dict[str, Any]:
    """Audit source precision and the fail-closed PIT availability transform."""

    required = {
        epoch_column,
        "source_publication_datetime",
        "publication_datetime",
        "publication_time_precision",
        "publication_timing_rule",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CNINFO publication timing missing columns: {missing}")
    try:
        cutoff = datetime_time.fromisoformat(signal_cutoff_local)
    except (TypeError, ValueError) as exc:
        raise ValueError("signal_cutoff_local must be an ISO local time") from exc
    if cutoff.tzinfo is not None:
        raise ValueError("signal_cutoff_local must not include a UTC offset")
    zone = ZoneInfo(market_timezone)

    violation_count = 0
    date_only_count = 0
    exact_count = 0
    shifted_count = 0
    exact_after_cutoff_count = 0
    for row in frame.loc[:, sorted(required)].itertuples(index=False):
        values = row._asdict()
        try:
            raw_epoch = values[epoch_column]
            if isinstance(raw_epoch, bool):
                raise TypeError("boolean epoch")
            epoch = int(raw_epoch)
            if epoch <= 0 or float(raw_epoch) != epoch:
                raise ValueError("invalid epoch")
            source = pd.Timestamp(values["source_publication_datetime"])
            available = pd.Timestamp(values["publication_datetime"])
            if (
                pd.isna(source)
                or source.tzinfo is None
                or source.utcoffset() is None
                or pd.isna(available)
                or available.tzinfo is None
                or available.utcoffset() is None
            ):
                raise ValueError("publication timestamps must be timezone-aware")
            expected_source = pd.Timestamp(epoch, unit="ms", tz="UTC")
            if source.tz_convert("UTC") != expected_source:
                raise ValueError("source timestamp disagrees with raw epoch")
            source_local = source.tz_convert(zone)
            available_local = available.tz_convert(zone)
            is_date_only = source_local.time() == datetime_time.min
            if is_date_only:
                date_only_count += 1
                expected_precision = PUBLICATION_TIME_PRECISION_DATE_ONLY
                expected_rule = PUBLICATION_TIMING_RULE_DATE_ONLY
                expected_available = source_local + pd.Timedelta(days=1)
                source_cutoff = source_local.normalize() + pd.Timedelta(
                    hours=cutoff.hour,
                    minutes=cutoff.minute,
                    seconds=cutoff.second,
                    microseconds=cutoff.microsecond,
                )
                if available_local <= source_cutoff:
                    raise ValueError("date-only publication is usable on source date")
            else:
                exact_count += 1
                expected_precision = PUBLICATION_TIME_PRECISION_EXACT
                expected_rule = PUBLICATION_TIMING_RULE_EXACT
                expected_available = source_local
                if source_local.time() > cutoff:
                    exact_after_cutoff_count += 1
            if values["publication_time_precision"] != expected_precision:
                raise ValueError("publication precision classification disagrees")
            if values["publication_timing_rule"] != expected_rule:
                raise ValueError("publication timing rule disagrees")
            if available_local != expected_available:
                raise ValueError("PIT availability transform disagrees")
            shifted_count += int(available_local > source_local)
        except (TypeError, ValueError, OverflowError, OSError):
            violation_count += 1
    return {
        "document_count": len(frame),
        "date_only_source_timestamp_count": date_only_count,
        "exact_source_timestamp_count": exact_count,
        "conservatively_shifted_count": shifted_count,
        "exact_source_timestamp_after_signal_cutoff_count": (exact_after_cutoff_count),
        "publication_timing_violation_count": violation_count,
        "signal_cutoff_local": signal_cutoff_local,
        "market_timezone": market_timezone,
        "date_only_availability_rule": PUBLICATION_TIMING_RULE_DATE_ONLY,
        "passed": violation_count == 0,
    }


def _security_id(ticker: str) -> str:
    if not re.fullmatch(r"\d{6}", ticker):
        raise ValueError(f"CNINFO secCode must retain six digits: {ticker!r}")
    if ticker.startswith(_SSE_A_SHARE_PREFIXES):
        exchange = "SSE"
    elif ticker.startswith(_SZSE_A_SHARE_PREFIXES):
        exchange = "SZSE"
    else:
        raise ValueError(f"CNINFO secCode is not a target SSE/SZSE A share: {ticker!r}")
    return f"{exchange}:{ticker}"


def _stock_map_rows(content: bytes) -> list[dict[str, str]]:
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CNINFO stock map did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError("CNINFO stock map JSON root must be an object")
    stock_list = payload.get("stockList")
    if not isinstance(stock_list, list):
        raise TypeError("CNINFO stock map stockList must be a list")

    rows: list[dict[str, str]] = []
    for position, record in enumerate(stock_list):
        if not isinstance(record, dict):
            raise TypeError(f"CNINFO stock map row {position} must be an object")
        values: dict[str, str] = {}
        for field in ("code", "orgId", "category"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"CNINFO stock map row {position} has invalid {field}")
            values[field] = value.strip()
        ticker = values["code"]
        if not re.fullmatch(r"\d{6}", ticker):
            raise ValueError(
                f"CNINFO stock map row {position} code must retain six digits"
            )
        if values["category"] != "A股":
            continue
        if not ticker.startswith((*_SSE_A_SHARE_PREFIXES, *_SZSE_A_SHARE_PREFIXES)):
            continue
        rows.append(
            {
                "cninfo_org_id": values["orgId"],
                "security_id": _security_id(ticker),
                "ticker": ticker,
                "category": values["category"],
            }
        )
    if not rows:
        raise ValueError("CNINFO stock map contains no target SSE/SZSE A shares")
    return rows


def _validate_stock_map_json(content: bytes) -> None:
    _stock_map_rows(content)


def normalize_cninfo_stock_map(content: bytes, asset: RawAsset) -> pd.DataFrame:
    """Strictly normalize CNINFO's official current code-to-orgId map."""

    rows = _stock_map_rows(content)
    frame = pd.DataFrame(rows)
    conflicts = frame.groupby("security_id", sort=True)["cninfo_org_id"].nunique().gt(1)
    if conflicts.any():
        sample = conflicts.loc[conflicts].index.astype(str).tolist()[:5]
        raise ValueError(
            "CNINFO stock map assigns one security code to multiple orgIds; "
            f"sample={sample}"
        )
    frame = (
        frame.drop_duplicates(["cninfo_org_id", "security_id", "ticker", "category"])
        .sort_values(["security_id", "cninfo_org_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    for column, value in provenance_columns(asset).items():
        frame[column] = value
    return frame


def acquire_cninfo_stock_map(
    transport: RateLimitedSession,
    raw_store: RawAssetStore,
    *,
    url: str = CNINFO_SZSE_STOCK_MAP_URL,
) -> CninfoStockMapAcquisition:
    """Acquire and cache CNINFO's official current SZSE stock identity map."""

    require_official_https_url(
        url,
        label="CNINFO stock-map URL",
        allowed_hosts={"www.cninfo.com.cn"},
        expected_path="/new/data/szse_stock.json",
    )

    fingerprint = request_fingerprint("GET", url)
    relative_path = f"cninfo/security_master/szse_stock_{fingerprint[:16]}.json"
    asset, content = fetch_raw_asset(
        transport,
        raw_store,
        relative_path,
        source="CNINFO SZSE stock map",
        method="GET",
        url=url,
        headers={"Referer": "https://www.cninfo.com.cn/"},
        validate_content=_validate_stock_map_json,
    )
    return CninfoStockMapAcquisition(
        stock_map=normalize_cninfo_stock_map(content, asset),
        asset=asset,
    )


def stock_map_diagnostics(result: CninfoStockMapAcquisition) -> dict[str, Any]:
    frame = result.stock_map
    provenance = [
        "source_url_or_identifier",
        "retrieval_datetime",
        "file_size",
        "sha256",
        "raw_local_path",
    ]
    complete = frame[provenance].notna().all(axis=1)
    return {
        "rows": len(frame),
        "org_id_groups": int(frame["cninfo_org_id"].nunique()),
        "security_ids": int(frame["security_id"].nunique()),
        "multi_code_org_id_groups": int(
            frame.groupby("cninfo_org_id")["security_id"].nunique().gt(1).sum()
        ),
        "provenance_complete_count": int(complete.sum()),
        "qa_state": "PASS" if len(frame) and bool(complete.all()) else "FAIL",
    }


def _metadata_columns() -> list[str]:
    return [
        "document_id",
        "cninfo_org_id",
        "security_id",
        "ticker",
        "security_name",
        "announcement_title",
        "source_publication_datetime",
        "publication_datetime",
        "publication_time_precision",
        "publication_timing_rule",
        "announcement_time_ms",
        "adjunct_url",
        "adjunct_type",
        "source",
        "source_url_or_identifier",
        "metadata_source",
        "metadata_source_url_or_identifier",
        "metadata_retrieval_datetime",
        "metadata_file_size",
        "metadata_sha256",
        "metadata_raw_local_path",
        "metadata_cache_hit",
        "local_raw_path",
        "retrieval_datetime",
        "file_size",
        "sha256",
        "download_status",
        "cache_hit",
        "failure_reason",
    ]


def normalize_announcements(
    announcements: list[dict[str, Any]],
    metadata_asset: RawAsset,
    *,
    download_base_url: str = CNINFO_DOWNLOAD_BASE_URL,
) -> pd.DataFrame:
    """Normalize a CNINFO page while preserving actual announcementTime."""

    require_official_https_url(
        download_base_url,
        label="CNINFO download base URL",
        allowed_hosts={"static.cninfo.com.cn"},
        expected_path="/",
    )
    rows: list[dict[str, Any]] = []
    metadata_provenance = provenance_columns(metadata_asset, prefix="metadata_")
    for announcement in announcements:
        title = _clean_title(announcement.get("announcementTitle"))
        if not is_annual_report_title(title):
            continue
        document_id = str(announcement.get("announcementId", "")).strip()
        raw_org_id = announcement.get("orgId")
        org_id = str(raw_org_id).strip() if raw_org_id is not None else None
        if org_id == "":
            org_id = None
        ticker = str(announcement.get("secCode", "")).strip().zfill(6)
        if re.fullmatch(r"\d{6}", ticker) and not ticker.startswith(
            (*_SSE_A_SHARE_PREFIXES, *_SZSE_A_SHARE_PREFIXES)
        ):
            continue
        security_name = str(announcement.get("secName", "")).strip()
        if security_name and not is_annual_report_title(title, security_name):
            continue
        adjunct_path = str(announcement.get("adjunctUrl", "")).strip()
        if not document_id or not security_name or not adjunct_path:
            raise ValueError(
                "CNINFO metadata requires announcementId, secName, and adjunctUrl"
            )
        publication_timing = _announcement_time(announcement.get("announcementTime"))
        adjunct_url = urljoin(
            download_base_url.rstrip("/") + "/", adjunct_path.lstrip("/")
        )
        require_official_https_url(
            adjunct_url,
            label=f"CNINFO document URL for {document_id}",
            allowed_hosts={"static.cninfo.com.cn"},
        )
        rows.append(
            {
                "document_id": document_id,
                "cninfo_org_id": org_id,
                "security_id": _security_id(ticker),
                "ticker": ticker,
                "security_name": security_name,
                "announcement_title": title,
                "source_publication_datetime": (
                    publication_timing.source_publication_datetime
                ),
                "publication_datetime": publication_timing.publication_datetime,
                "publication_time_precision": (
                    publication_timing.publication_time_precision
                ),
                "publication_timing_rule": publication_timing.publication_timing_rule,
                "announcement_time_ms": publication_timing.announcement_time_ms,
                "adjunct_url": adjunct_url,
                "adjunct_type": str(announcement.get("adjunctType", ""))
                .strip()
                .upper(),
                "source": "CNINFO",
                "source_url_or_identifier": adjunct_url,
                **metadata_provenance,
                "local_raw_path": None,
                "retrieval_datetime": None,
                "file_size": None,
                "sha256": None,
                "download_status": "PENDING",
                "cache_hit": None,
                "failure_reason": None,
            }
        )
    return pd.DataFrame(rows, columns=_metadata_columns())


def deduplicate_announcements(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate identical document evidence, rejecting identity conflicts."""

    if frame.empty:
        return pd.DataFrame(columns=_metadata_columns())
    identity_columns = [
        "cninfo_org_id",
        "security_id",
        "security_name",
        "announcement_time_ms",
        "adjunct_url",
    ]
    missing_identity = sorted(set(identity_columns) - set(frame.columns))
    if missing_identity:
        raise ValueError(
            f"CNINFO metadata missing issuer identity columns: {missing_identity}"
        )
    conflicts = (
        frame.groupby("document_id", dropna=False)[identity_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        document_ids = conflicts.loc[conflicts].index.astype(str).tolist()[:5]
        raise ValueError(
            "CNINFO duplicate document IDs disagree on issuer/document evidence; "
            f"sample={document_ids}"
        )
    ordered = frame.sort_values(
        [
            "document_id",
            "announcement_time_ms",
            "adjunct_url",
            "announcement_title",
            "metadata_sha256",
        ],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )
    deduplicated = ordered.drop_duplicates("document_id", keep="first")
    return deduplicated.sort_values(
        ["publication_datetime", "document_id"], kind="mergesort"
    ).reset_index(drop=True)


def reconcile_security_ids(
    documents: pd.DataFrame,
    security_master: pd.DataFrame,
    stock_map: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconcile obsolete CNINFO codes without collapsing valid securities.

    ``orgId`` is CNINFO's stable issuer key.  A reported security ID already in
    the official master is authoritative and is never rewritten.  A reported
    ID absent from the master is bridged only when the same ``orgId`` group has
    exactly one ID present in the master across document observations and the
    separately acquired official current stock map.  Zero or multiple
    candidates remain unresolved instead of being guessed; downstream
    mention/edge membership checks then fail closed if such a document
    contributes data.
    """

    if not isinstance(documents, pd.DataFrame):
        raise TypeError("documents must be a pandas DataFrame")
    if not isinstance(security_master, pd.DataFrame):
        raise TypeError("security_master must be a pandas DataFrame")
    required_documents = {"document_id", "cninfo_org_id", "security_id"}
    missing_documents = sorted(required_documents - set(documents.columns))
    if missing_documents:
        raise ValueError(
            f"CNINFO documents missing reconciliation columns: {missing_documents}"
        )
    if "security_id" not in security_master.columns:
        raise ValueError("security_master missing reconciliation column: security_id")
    if security_master["security_id"].isna().any():
        raise ValueError("security_master security_id must be non-null")
    if security_master["security_id"].duplicated().any():
        duplicated = (
            security_master.loc[
                security_master["security_id"].duplicated(keep=False), "security_id"
            ]
            .astype(str)
            .unique()
            .tolist()[:5]
        )
        raise ValueError(
            f"security_master contains duplicate security_id values: {duplicated}"
        )

    if stock_map is None:
        stock_evidence = pd.DataFrame(columns=["cninfo_org_id", "security_id"])
    else:
        if not isinstance(stock_map, pd.DataFrame):
            raise TypeError("stock_map must be a pandas DataFrame")
        required_stock_map = {"cninfo_org_id", "security_id"}
        missing_stock_map = sorted(required_stock_map - set(stock_map.columns))
        if missing_stock_map:
            raise ValueError(
                f"CNINFO stock map missing reconciliation columns: {missing_stock_map}"
            )
        stock_evidence = stock_map.loc[:, ["cninfo_org_id", "security_id"]].copy()
        for column in ("cninfo_org_id", "security_id"):
            invalid = stock_evidence[column].isna() | ~stock_evidence[column].map(
                lambda value: isinstance(value, str) and bool(value.strip())
            )
            if invalid.any():
                raise ValueError(
                    f"CNINFO stock map {column} must contain nonblank strings"
                )
        stock_conflicts = (
            stock_evidence.groupby("security_id", sort=True)["cninfo_org_id"]
            .nunique()
            .gt(1)
        )
        if stock_conflicts.any():
            sample = stock_conflicts.loc[stock_conflicts].index.tolist()[:5]
            raise ValueError(
                "CNINFO stock map assigns security IDs to multiple orgIds; "
                f"sample={sample}"
            )
        stock_evidence = stock_evidence.drop_duplicates().reset_index(drop=True)

    result = documents.copy()
    for column in ("document_id", "security_id"):
        invalid = result[column].isna() | ~result[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if invalid.any():
            sample = result.loc[invalid, "document_id"].head(5).tolist()
            raise ValueError(
                f"CNINFO {column} must contain nonblank strings; documents={sample}"
            )
    if result["document_id"].duplicated().any():
        duplicated = (
            result.loc[result["document_id"].duplicated(keep=False), "document_id"]
            .astype(str)
            .unique()
            .tolist()[:5]
        )
        raise ValueError(f"CNINFO documents contain duplicate IDs: {duplicated}")

    if "reported_security_id" in result.columns:
        reported = result["reported_security_id"]
        invalid = reported.isna() | ~reported.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if invalid.any():
            raise ValueError("reported_security_id must contain nonblank strings")
        result["reported_security_id"] = reported.astype(str)
    else:
        result["reported_security_id"] = result["security_id"].astype(str)

    master_ids = set(security_master["security_id"].astype(str))
    group_candidates: dict[str, tuple[str, ...]] = {}
    has_org_id = result["cninfo_org_id"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    invalid_org_id = result["cninfo_org_id"].notna() & ~has_org_id
    if invalid_org_id.any():
        raise ValueError("CNINFO cninfo_org_id must be null or a nonblank string")
    stock_org_by_security = stock_evidence.set_index("security_id")[
        "cninfo_org_id"
    ].to_dict()
    comparable = result.loc[has_org_id, ["document_id", "cninfo_org_id"]].copy()
    comparable["stock_org_id"] = result.loc[has_org_id, "reported_security_id"].map(
        stock_org_by_security
    )
    conflicting_official_identity = comparable["stock_org_id"].notna() & comparable[
        "stock_org_id"
    ].ne(comparable["cninfo_org_id"])
    if conflicting_official_identity.any():
        sample = comparable.loc[conflicting_official_identity, "document_id"].tolist()[
            :5
        ]
        raise ValueError(
            "CNINFO document and stock-map orgId evidence conflicts; "
            f"documents={sample}"
        )

    observed_by_org: dict[str, set[str]] = {}
    for org_id, group in result.loc[has_org_id].groupby("cninfo_org_id", sort=True):
        observed_by_org[str(org_id)] = set(group["reported_security_id"].astype(str))
    for org_id, group in stock_evidence.groupby("cninfo_org_id", sort=True):
        observed_by_org.setdefault(str(org_id), set()).update(
            group["security_id"].astype(str)
        )
    for org_id in sorted(set(result.loc[has_org_id, "cninfo_org_id"].astype(str))):
        group_candidates[org_id] = tuple(
            sorted(observed_by_org.get(org_id, set()) & master_ids)
        )

    canonical_ids: list[str] = []
    methods: list[str] = []
    candidate_ids: list[str | None] = []
    candidate_counts: list[int] = []
    for row in result.itertuples(index=False):
        reported_id = str(row.reported_security_id)
        org_id = row.cninfo_org_id
        missing_org_id = org_id is None or pd.isna(org_id)
        candidates = () if missing_org_id else group_candidates[str(org_id)]
        if reported_id in master_ids and missing_org_id:
            canonical_id = reported_id
            method = RECONCILIATION_REPORTED_ID_IN_MASTER_MISSING_ORG_ID
        elif reported_id in master_ids:
            canonical_id = reported_id
            method = RECONCILIATION_REPORTED_ID_IN_MASTER
        elif missing_org_id:
            canonical_id = reported_id
            method = RECONCILIATION_UNRESOLVED_MISSING_ORG_ID
        elif len(candidates) == 1:
            canonical_id = candidates[0]
            method = RECONCILIATION_ORG_ID_UNIQUE_MASTER_MATCH
        elif not candidates:
            canonical_id = reported_id
            method = RECONCILIATION_UNRESOLVED_NO_MASTER_MATCH
        else:
            canonical_id = reported_id
            method = RECONCILIATION_UNRESOLVED_AMBIGUOUS_MASTER_MATCH
        canonical_ids.append(canonical_id)
        methods.append(method)
        candidate_ids.append(
            json.dumps(list(candidates), ensure_ascii=False, separators=(",", ":"))
            if candidates
            else None
        )
        candidate_counts.append(len(candidates))

    result["security_id"] = canonical_ids
    result["security_id_reconciliation_method"] = methods
    result["security_id_reconciliation_candidate_ids"] = candidate_ids
    result["security_id_reconciliation_candidate_count"] = candidate_counts

    method_counts = result["security_id_reconciliation_method"].value_counts()
    diagnostics: dict[str, Any] = {
        "document_count": len(result),
        "org_id_group_count": len(group_candidates),
        "stock_map_row_count": len(stock_evidence),
        "stock_map_org_id_group_count": int(stock_evidence["cninfo_org_id"].nunique()),
        "missing_org_id_count": int((~has_org_id).sum()),
        "reported_id_in_master_count": int(
            method_counts.get(RECONCILIATION_REPORTED_ID_IN_MASTER, 0)
            + method_counts.get(RECONCILIATION_REPORTED_ID_IN_MASTER_MISSING_ORG_ID, 0)
        ),
        "mapped_via_org_id_count": int(
            method_counts.get(RECONCILIATION_ORG_ID_UNIQUE_MASTER_MATCH, 0)
        ),
        "unresolved_no_master_match_count": int(
            method_counts.get(RECONCILIATION_UNRESOLVED_NO_MASTER_MATCH, 0)
        ),
        "unresolved_missing_org_id_count": int(
            method_counts.get(RECONCILIATION_UNRESOLVED_MISSING_ORG_ID, 0)
        ),
        "unresolved_ambiguous_master_match_count": int(
            method_counts.get(
                RECONCILIATION_UNRESOLVED_AMBIGUOUS_MASTER_MATCH,
                0,
            )
        ),
        "canonical_id_in_master_count": int(
            result["security_id"].isin(master_ids).sum()
        ),
    }
    return result, diagnostics


def enrich_company_aliases(
    aliases: pd.DataFrame,
    documents: pd.DataFrame,
    security_master: pd.DataFrame,
    stock_map: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add source-dated CNINFO issuer short names to the alias history.

    An alias becomes valid on the first Shanghai-calendar publication date on
    which CNINFO itself associates that name with the reconciled security.  No
    earlier validity or end date is inferred.  Documents whose security ID is
    still absent from the official master cannot create an entity alias.
    """

    alias_columns = (
        "entity_id",
        "canonical_name",
        "alias",
        "alias_type",
        "valid_from",
        "valid_to",
        "source",
    )
    missing_aliases = sorted(set(alias_columns) - set(aliases.columns))
    if missing_aliases:
        raise ValueError(f"company_alias missing columns: {missing_aliases}")
    required_documents = {
        "cninfo_org_id",
        "document_id",
        "publication_datetime",
        "reported_security_id",
        "security_id",
        "security_name",
        "security_id_reconciliation_method",
    }
    missing_documents = sorted(required_documents - set(documents.columns))
    if missing_documents:
        raise ValueError(
            f"CNINFO documents missing alias evidence columns: {missing_documents}"
        )
    required_master = {"security_id", "company_name"}
    missing_master = sorted(required_master - set(security_master.columns))
    if missing_master:
        raise ValueError(f"security_master missing alias columns: {missing_master}")
    if security_master["security_id"].duplicated().any():
        raise ValueError("security_master contains duplicate security_id values")

    if stock_map is None:
        stock_evidence = pd.DataFrame(
            columns=[
                "cninfo_org_id",
                "security_id",
                "retrieval_datetime",
                "raw_local_path",
            ]
        )
    else:
        required_stock = {
            "cninfo_org_id",
            "security_id",
            "retrieval_datetime",
            "raw_local_path",
        }
        missing_stock = sorted(required_stock - set(stock_map.columns))
        if missing_stock:
            raise ValueError(
                f"CNINFO stock map missing alias evidence columns: {missing_stock}"
            )
        stock_evidence = stock_map.loc[:, sorted(required_stock)].copy()

    master_names = security_master.set_index("security_id")["company_name"]
    eligible = documents.loc[
        documents["security_id"].astype(str).isin(master_names.index.astype(str))
    ].copy()
    invalid_name = eligible["security_name"].isna() | ~eligible["security_name"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    if invalid_name.any():
        raise ValueError("CNINFO security_name evidence must contain nonblank strings")
    invalid_document_id = eligible["document_id"].isna() | ~eligible["document_id"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    if invalid_document_id.any():
        raise ValueError("CNINFO alias document_id must contain nonblank strings")
    invalid_method = ~eligible["security_id_reconciliation_method"].isin(
        {
            RECONCILIATION_REPORTED_ID_IN_MASTER,
            RECONCILIATION_REPORTED_ID_IN_MASTER_MISSING_ORG_ID,
            RECONCILIATION_ORG_ID_UNIQUE_MASTER_MATCH,
        }
    )
    if invalid_method.any():
        raise ValueError(
            "CNINFO alias evidence references an unresolved reconciliation method"
        )

    try:
        publication = pd.to_datetime(
            eligible["publication_datetime"], errors="raise", utc=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "CNINFO alias publication_datetime contains invalid values"
        ) from exc
    if publication.isna().any():
        raise ValueError("CNINFO alias publication_datetime must be non-null")
    eligible["_valid_from"] = publication.dt.tz_convert("Asia/Shanghai").dt.date.astype(
        str
    )
    evidence = (
        eligible.sort_values(
            ["security_id", "security_name", "_valid_from", "document_id"],
            kind="mergesort",
        )
        .drop_duplicates(["security_id", "security_name"], keep="first")
        .sort_values(
            ["security_id", "_valid_from", "security_name", "document_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    short_name_rows = pd.DataFrame(
        {
            "entity_id": evidence["security_id"].astype(str),
            "canonical_name": evidence["security_id"].map(master_names),
            "alias": evidence["security_name"].astype(str),
            "alias_type": CNINFO_SHORT_NAME_ALIAS_TYPE,
            "valid_from": evidence["_valid_from"],
            "valid_to": None,
            "source": evidence["document_id"].map(
                lambda value: f"{CNINFO_SHORT_NAME_ALIAS_SOURCE}{value}"
            ),
        },
        columns=alias_columns,
    )

    base_aliases = aliases.loc[:, alias_columns].copy()
    observed_ticker_rows = pd.DataFrame(columns=alias_columns)
    constrained_ticker_aliases = 0
    preserved_closed_ticker_aliases = 0
    code_change_org_ids: set[str] = set()
    if not stock_evidence.empty:
        for column in ("cninfo_org_id", "security_id", "raw_local_path"):
            invalid = stock_evidence[column].isna() | ~stock_evidence[column].map(
                lambda value: isinstance(value, str) and bool(value.strip())
            )
            if invalid.any():
                raise ValueError(
                    f"CNINFO stock map {column} must contain nonblank strings"
                )
        try:
            stock_retrieval = pd.to_datetime(
                stock_evidence["retrieval_datetime"], errors="raise", utc=True
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CNINFO stock map retrieval_datetime contains invalid values"
            ) from exc
        if stock_retrieval.isna().any():
            raise ValueError("CNINFO stock map retrieval_datetime must be non-null")
        stock_evidence["retrieval_day"] = stock_retrieval.dt.tz_convert(
            "Asia/Shanghai"
        ).dt.date.astype(str)
        code_counts = stock_evidence.groupby("cninfo_org_id")["security_id"].nunique()
        code_change_org_ids = set(code_counts.loc[code_counts.gt(1)].index.astype(str))
        change_codes = stock_evidence.loc[
            stock_evidence["cninfo_org_id"].astype(str).isin(code_change_org_ids)
        ].copy()

        observed_codes = documents.loc[
            documents["cninfo_org_id"].notna()
            & documents["cninfo_org_id"].astype(str).isin(code_change_org_ids)
        ].copy()
        observed_codes["_valid_from"] = (
            pd.to_datetime(
                observed_codes["publication_datetime"], errors="raise", utc=True
            )
            .dt.tz_convert("Asia/Shanghai")
            .dt.date.astype(str)
        )
        first_observations = (
            observed_codes.sort_values(
                [
                    "cninfo_org_id",
                    "reported_security_id",
                    "_valid_from",
                    "document_id",
                ],
                kind="mergesort",
            )
            .drop_duplicates(["cninfo_org_id", "reported_security_id"], keep="first")
            .set_index(["cninfo_org_id", "reported_security_id"])
        )

        code_evidence: dict[str, tuple[str, str]] = {}
        for row in change_codes.sort_values(
            ["cninfo_org_id", "security_id", "retrieval_day", "raw_local_path"],
            kind="mergesort",
        ).itertuples(index=False):
            key = (str(row.cninfo_org_id), str(row.security_id))
            if key in first_observations.index:
                observation = first_observations.loc[key]
                if isinstance(observation, pd.DataFrame):
                    raise ValueError(
                        "CNINFO ticker evidence is not unique after deterministic "
                        "selection"
                    )
                evidence_day = str(observation["_valid_from"])
                source = (
                    f"{CNINFO_OBSERVED_TICKER_ALIAS_SOURCE}{observation['document_id']}"
                )
            else:
                evidence_day = str(row.retrieval_day)
                source = (
                    "CNINFO official current stock map; "
                    f"first_observed_raw_asset={row.raw_local_path}"
                )
            prior = code_evidence.get(str(row.security_id))
            if prior is not None and prior != (evidence_day, source):
                raise ValueError(
                    "CNINFO stock-map code evidence conflicts across orgId groups: "
                    f"{row.security_id}"
                )
            code_evidence[str(row.security_id)] = (evidence_day, source)

        for security_id, (evidence_day, evidence_source) in code_evidence.items():
            ticker = security_id.split(":", maxsplit=1)[-1]
            mask = (
                base_aliases["entity_id"].eq(security_id)
                & base_aliases["alias_type"].eq("ticker")
                & base_aliases["alias"].eq(ticker)
            )
            if not mask.any():
                continue
            current_start = pd.to_datetime(
                base_aliases.loc[mask, "valid_from"], errors="raise"
            ).dt.date.astype(str)
            current_end = pd.to_datetime(
                base_aliases.loc[mask, "valid_to"], errors="raise"
            )
            evidence_timestamp = pd.Timestamp(evidence_day)
            closed_before_evidence = current_end.notna() & current_end.le(
                evidence_timestamp
            )
            preserved_closed_ticker_aliases += int(
                (current_start.lt(evidence_day) & closed_before_evidence).sum()
            )
            tighten = current_start.lt(evidence_day) & ~closed_before_evidence
            if tighten.any():
                indices = current_start.loc[tighten].index
                base_aliases.loc[indices, "valid_from"] = evidence_day
                base_aliases.loc[indices, "source"] = evidence_source
                constrained_ticker_aliases += len(indices)

        mapped_observations = observed_codes.loc[
            observed_codes["security_id"]
            .astype(str)
            .isin(master_names.index.astype(str))
            & observed_codes["reported_security_id"]
            .astype(str)
            .ne(observed_codes["security_id"].astype(str))
        ].copy()
        if not mapped_observations.empty:
            mapped_observations = (
                mapped_observations.sort_values(
                    [
                        "security_id",
                        "reported_security_id",
                        "_valid_from",
                        "document_id",
                    ],
                    kind="mergesort",
                )
                .drop_duplicates(["security_id", "reported_security_id"], keep="first")
                .reset_index(drop=True)
            )
            reported_tickers = mapped_observations["reported_security_id"].map(
                lambda value: str(value).split(":", maxsplit=1)[-1]
            )
            if not bool(reported_tickers.str.fullmatch(r"\d{6}").all()):
                raise ValueError(
                    "CNINFO observed ticker aliases require six-digit security IDs"
                )
            observed_ticker_rows = pd.DataFrame(
                {
                    "entity_id": mapped_observations["security_id"].astype(str),
                    "canonical_name": mapped_observations["security_id"].map(
                        master_names
                    ),
                    "alias": reported_tickers,
                    "alias_type": CNINFO_OBSERVED_TICKER_ALIAS_TYPE,
                    "valid_from": mapped_observations["_valid_from"],
                    "valid_to": None,
                    "source": mapped_observations["document_id"].map(
                        lambda value: f"{CNINFO_OBSERVED_TICKER_ALIAS_SOURCE}{value}"
                    ),
                },
                columns=alias_columns,
            )

    combined = pd.concat(
        [base_aliases, short_name_rows, observed_ticker_rows], ignore_index=True
    ).drop_duplicates(list(alias_columns))
    key = ["entity_id", "alias", "valid_from"]
    conflicting_keys = (
        combined.groupby(key, dropna=False)[
            ["canonical_name", "alias_type", "valid_to", "source"]
        ]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicting_keys.any():
        sample = conflicting_keys.loc[conflicting_keys].index.tolist()[:5]
        raise ValueError(
            f"company_alias rows conflict on canonical primary key; sample={sample}"
        )
    combined = (
        combined.drop_duplicates(key, keep="first")
        .sort_values(
            ["entity_id", "valid_from", "alias_type", "alias"], kind="mergesort"
        )
        .reset_index(drop=True)
    )
    cross_entity_aliases = int(
        short_name_rows.groupby("alias", dropna=False)["entity_id"]
        .nunique()
        .gt(1)
        .sum()
    )
    diagnostics: dict[str, Any] = {
        "eligible_document_count": len(eligible),
        "source_name_evidence_rows": len(evidence),
        "code_change_org_id_groups": len(code_change_org_ids),
        "constrained_backdated_ticker_alias_rows": constrained_ticker_aliases,
        "preserved_closed_ticker_alias_rows": preserved_closed_ticker_aliases,
        "observed_historical_ticker_alias_rows": len(observed_ticker_rows),
        "added_alias_rows": len(combined) - len(aliases),
        "cross_entity_exact_alias_count": cross_entity_aliases,
        "validity_rule": "first_cninfo_publication_date_to_open_end",
    }
    return combined, diagnostics


def _validate_query_json(content: bytes) -> None:
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CNINFO query did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError("CNINFO query JSON root must be an object")
    announcements = payload.get("announcements")
    if announcements is not None and (
        not isinstance(announcements, list)
        or any(not isinstance(item, dict) for item in announcements)
    ):
        raise ValueError("CNINFO announcements must be a list of objects")


def _reported_total_count(payload: dict[str, Any]) -> int | None:
    reported: list[int] = []
    for key in ("totalAnnouncement", "totalRecordNum", "total"):
        raw_value = payload.get(key)
        if raw_value in (None, ""):
            continue
        if isinstance(raw_value, bool):
            raise TypeError(f"CNINFO {key} must be a non-negative integer")
        try:
            value = int(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"CNINFO {key} must be a non-negative integer") from exc
        if value < 0:
            raise ValueError(f"CNINFO {key} must be a non-negative integer")
        reported.append(value)
    if not reported:
        return None
    if len(set(reported)) != 1:
        raise ValueError(f"CNINFO reported inconsistent result totals: {reported}")
    return reported[0]


def _reported_page_count(
    payload: dict[str, Any], page_size: int, current_page: int
) -> int:
    candidates = [current_page]
    for key in ("totalpages", "totalPages", "pageCount"):
        raw_value = payload.get(key)
        if raw_value in (None, ""):
            continue
        if isinstance(raw_value, bool):
            raise TypeError(f"CNINFO {key} must be a non-negative integer")
        try:
            value = int(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"CNINFO {key} must be a non-negative integer") from exc
        if value < 0:
            raise ValueError(f"CNINFO {key} must be a non-negative integer")
        candidates.append(value)
    total = _reported_total_count(payload)
    if total is not None:
        candidates.append(math.ceil(total / page_size))
    if len(candidates) > 1:
        return max(candidates)
    announcements = payload.get("announcements") or []
    return current_page if len(announcements) < page_size else current_page + 1


def _calendar_year_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Partition an inclusive range for CNINFO's bounded date query.

    CNINFO returns an empty result for very long multi-year ranges even when
    the constituent years contain data. Calendar-year windows are explicit,
    deterministic, and non-overlapping.
    """

    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        window_end = min(end, date(cursor.year, 12, 31))
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def query_annual_report_metadata(
    transport: RateLimitedSession,
    raw_store: RawAssetStore,
    *,
    start_date: str,
    end_date: str,
    query_url: str = CNINFO_QUERY_URL,
    download_base_url: str = CNINFO_DOWNLOAD_BASE_URL,
    category: str = CNINFO_ANNUAL_REPORT_CATEGORY,
    page_size: int = 30,
) -> tuple[pd.DataFrame, tuple[RawAsset, ...]]:
    """Page through CNINFO annual-report metadata and deduplicate it."""

    require_official_https_url(
        query_url,
        label="CNINFO metadata query URL",
        allowed_hosts={"www.cninfo.com.cn"},
        expected_path="/new/hisAnnouncement/query",
    )
    require_official_https_url(
        download_base_url,
        label="CNINFO download base URL",
        allowed_hosts={"static.cninfo.com.cn"},
        expected_path="/",
    )

    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if start > end:
        raise ValueError("start_date must not follow end_date")
    if isinstance(page_size, bool) or page_size < 1:
        raise ValueError("page_size must be a positive integer")

    frames: list[pd.DataFrame] = []
    assets: list[RawAsset] = []

    def fetch_page(
        window_start: date,
        window_end: date,
        page: int,
    ) -> tuple[dict[str, Any], RawAsset]:
        window_label = f"{window_start:%Y%m%d}_{window_end:%Y%m%d}"
        form = {
            "pageNum": str(page),
            "pageSize": str(page_size),
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": "",
            "searchkey": "",
            "secid": "",
            "category": category,
            "trade": "",
            "seDate": f"{window_start.isoformat()}~{window_end.isoformat()}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        fingerprint = request_fingerprint("POST", query_url, data=form)
        relative_path = (
            "cninfo/annual_reports/metadata/"
            f"{window_label}/page_{page:05d}_{fingerprint[:16]}.json"
        )
        asset, content = fetch_raw_asset(
            transport,
            raw_store,
            relative_path,
            source="CNINFO",
            method="POST",
            url=query_url,
            data=form,
            headers={
                "Origin": "https://www.cninfo.com.cn",
                "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch",
                "X-Requested-With": "XMLHttpRequest",
            },
            validate_content=_validate_query_json,
        )
        return json.loads(content.decode("utf-8-sig")), asset

    pending = deque(_calendar_year_windows(start, end))
    while pending:
        window_start, window_end = pending.popleft()
        first_payload, first_asset = fetch_page(window_start, window_end, 1)
        assets.append(first_asset)
        first_announcements = first_payload.get("announcements") or []
        reported_total = _reported_total_count(first_payload)
        page_count = _reported_page_count(first_payload, page_size, 1)

        # CNINFO silently returns page 1 again for pageNum > 100. Split before
        # consuming any rows from an oversized window so no apparent success
        # can hide omitted documents.
        if page_count > CNINFO_MAX_RETRIEVABLE_PAGES:
            if window_start == window_end:
                raise ValueError(
                    "CNINFO one-day result exceeds the 100-page retrievable limit: "
                    f"{window_start.isoformat()} ({page_count} pages)"
                )
            midpoint = window_start + (window_end - window_start) // 2
            pending.appendleft((midpoint + timedelta(days=1), window_end))
            pending.appendleft((window_start, midpoint))
            continue

        frames.append(
            normalize_announcements(
                first_announcements,
                first_asset,
                download_base_url=download_base_url,
            )
        )
        raw_announcement_count = len(first_announcements)
        if page_count > 1 and not first_announcements:
            raise ValueError(
                "CNINFO returned an empty first page for a non-empty paginated "
                f"window: {window_start.isoformat()}~{window_end.isoformat()}"
            )

        for page in range(2, page_count + 1):
            payload, asset = fetch_page(window_start, window_end, page)
            assets.append(asset)
            announcements = payload.get("announcements") or []
            if not announcements:
                raise ValueError(
                    "CNINFO returned an empty page before the reported result set "
                    f"ended: {window_start.isoformat()}~{window_end.isoformat()} "
                    f"page {page}/{page_count}"
                )
            raw_announcement_count += len(announcements)
            frames.append(
                normalize_announcements(
                    announcements,
                    asset,
                    download_base_url=download_base_url,
                )
            )

        if reported_total is not None and raw_announcement_count != reported_total:
            raise ValueError(
                "CNINFO result count disagrees with totalAnnouncement: "
                f"{window_start.isoformat()}~{window_end.isoformat()} expected "
                f"{reported_total}, received {raw_announcement_count}"
            )

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=_metadata_columns())
    )
    deduplicated = deduplicate_announcements(combined)
    if deduplicated.empty:
        excluded_count = 0
    else:
        source_local = pd.to_datetime(
            deduplicated["source_publication_datetime"], errors="raise", utc=True
        ).dt.tz_convert(CNINFO_PUBLICATION_TIMEZONE)
        available_local = pd.to_datetime(
            deduplicated["publication_datetime"], errors="raise", utc=True
        ).dt.tz_convert(CNINFO_PUBLICATION_TIMEZONE)
        end_day = pd.Timestamp(end, tz=CNINFO_PUBLICATION_TIMEZONE)
        end_exclusive = end_day + pd.Timedelta(days=1)
        # The source endpoint is date-ranged.  A midnight placeholder on the
        # inclusive query end is not safely usable until the following day and
        # therefore belongs outside this research window (notably the frozen
        # validation/holdout boundary).
        excluded = source_local.dt.normalize().eq(end_day) & available_local.ge(
            end_exclusive
        )
        excluded_count = int(excluded.sum())
        deduplicated = deduplicated.loc[~excluded].reset_index(drop=True)
    deduplicated.attrs["timing_excluded_after_query_end_count"] = excluded_count
    return deduplicated, tuple(assets)


def _document_extension(row: pd.Series) -> str:
    adjunct_type = str(row.get("adjunct_type") or "").strip().lower()
    suffix = PurePosixPath(urlparse(str(row["adjunct_url"])).path).suffix.lower()
    extension = suffix or (f".{adjunct_type}" if adjunct_type else ".bin")
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", extension):
        return ".bin"
    return extension


def _document_validator(extension: str) -> Any:
    def validate(content: bytes) -> None:
        if not content:
            raise ValueError("CNINFO document response is empty")
        if extension == ".pdf" and not content.lstrip().startswith(b"%PDF-"):
            raise ValueError("CNINFO PDF URL returned non-PDF content")

    return validate


def _document_cache_path(row: pd.Series) -> str:
    safe_document_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["document_id"]))
    url_fingerprint = request_fingerprint("GET", str(row["adjunct_url"]))[:16]
    return (
        f"cninfo/annual_reports/documents/{safe_document_id}_{url_fingerprint}"
        f"{_document_extension(row)}"
    )


def write_failure_log(
    raw_store: RawAssetStore,
    failures: tuple[DownloadFailure, ...] | list[DownloadFailure],
    *,
    run_timestamp: str,
) -> RawAsset | None:
    """Write an immutable JSONL failure log with an explicit reason per row."""

    if not failures:
        return None
    records = [failure.to_dict() for failure in failures]
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for record in records
    ).encode("utf-8")
    safe_timestamp = re.sub(r"[^0-9TZ]+", "", run_timestamp)
    digest = request_fingerprint(
        "LOG", "cninfo_failed_downloads", data={"rows": records}
    )[:16]
    path = f"cninfo/annual_reports/failures/{safe_timestamp}_{digest}.jsonl"
    return raw_store.store_bytes(
        path,
        content,
        source="CNINFO acquisition failure log",
        source_url_or_identifier=f"cninfo_failed_downloads:{run_timestamp}",
    )


def _download_document_job(
    job: _DocumentDownloadJob,
    *,
    transport: RateLimitedSession,
    raw_store: RawAssetStore,
    clock: Any,
    cancel_event: threading.Event | None = None,
) -> _DocumentDownloadOutcome:
    try:
        asset, _ = fetch_raw_asset(
            transport,
            raw_store,
            job.relative_path,
            source="CNINFO",
            method="GET",
            url=job.source_url_or_identifier,
            headers={"Referer": "https://www.cninfo.com.cn/"},
            validate_content=_document_validator(job.extension),
            cancel_event=cancel_event,
            # CNINFO's CDN can terminate large responses mid-transfer even for
            # byte ranges.  Keep recovery segments comfortably below the
            # observed truncation boundary while preserving global pacing.
            range_fallback_chunk_bytes=256 * 1024,
        )
    except (
        requests.RequestException,
        RawAssetIntegrityError,
        OSError,
        ValueError,
    ) as exc:
        failure_category = (
            DownloadFailureCategory.SOURCE_UNAVAILABLE
            if isinstance(exc, requests.RequestException)
            else DownloadFailureCategory.CONTENT_OR_INTEGRITY_ERROR
        )
        failure = DownloadFailure(
            document_id=job.document_id,
            source_url_or_identifier=job.source_url_or_identifier,
            attempted_at=_utc_timestamp(clock),
            reason=f"{type(exc).__name__}: {exc}",
            failure_category=failure_category,
        )
        return _DocumentDownloadOutcome(job=job, failure=failure)
    return _DocumentDownloadOutcome(job=job, asset=asset)


def _parallel_download_jobs(
    jobs: list[_DocumentDownloadJob],
    *,
    transport: RateLimitedSession,
    raw_store: RawAssetStore,
    clock: Any,
    max_workers: int,
) -> list[_DocumentDownloadOutcome]:
    """Run a bounded pool and return outcomes in immutable job order."""

    cancel_event = threading.Event()
    outcomes: list[_DocumentDownloadOutcome | None] = [None] * len(jobs)
    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="cninfo-download",
    )
    pending: dict[Future[_DocumentDownloadOutcome], int] = {}
    next_job = 0
    completed_count = 0

    def submit_one() -> None:
        nonlocal next_job
        job = jobs[next_job]
        future = executor.submit(
            _download_document_job,
            job,
            transport=transport,
            raw_store=raw_store,
            clock=clock,
            cancel_event=cancel_event,
        )
        pending[future] = job.ordinal
        next_job += 1

    try:
        while next_job < min(max_workers, len(jobs)):
            submit_one()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: pending[item]):
                ordinal = pending.pop(future)
                outcome = future.result()
                if outcome.job.ordinal != ordinal:
                    raise RuntimeError("CNINFO download outcome order identity changed")
                outcomes[ordinal] = outcome
                completed_count += 1
                if completed_count % 1000 == 0 or completed_count == len(jobs):
                    LOGGER.info(
                        "CNINFO document download progress completed=%d total=%d",
                        completed_count,
                        len(jobs),
                    )
                if next_job < len(jobs):
                    submit_one()
    except BaseException:
        cancel_event.set()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        transport.close_thread_sessions()
        raise
    else:
        executor.shutdown(wait=True)
        transport.close_thread_sessions()

    if any(outcome is None for outcome in outcomes):
        raise RuntimeError("CNINFO download pool returned an incomplete outcome set")
    return [outcome for outcome in outcomes if outcome is not None]


def download_documents(
    metadata: pd.DataFrame,
    transport: RateLimitedSession,
    raw_store: RawAssetStore,
    *,
    mode: AcquisitionMode | None = None,
    clock: Any = utc_now,
    max_workers: int = 1,
) -> tuple[
    pd.DataFrame, tuple[RawAsset, ...], tuple[DownloadFailure, ...], RawAsset | None
]:
    """Download a deterministic selection and retain every provenance field."""

    mode = mode or AcquisitionMode()
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= CNINFO_MAX_DOWNLOAD_WORKERS
    ):
        raise ValueError(
            f"max_workers must be an integer from 1 to {CNINFO_MAX_DOWNLOAD_WORKERS}"
        )
    documents = deduplicate_announcements(metadata).copy()
    if documents.empty:
        return documents, (), (), None
    selected_ids: set[str]
    if mode.metadata_only:
        selected_ids = set()
        documents.loc[:, "download_status"] = "METADATA_ONLY"
    else:
        ordered_ids = documents["document_id"].tolist()
        if mode.max_documents is not None:
            ordered_ids = ordered_ids[: mode.max_documents]
        selected_ids = set(ordered_ids)
        documents.loc[
            ~documents["document_id"].isin(selected_ids), "download_status"
        ] = "NOT_REQUESTED_LIMIT"

    jobs: list[_DocumentDownloadJob] = []
    cache_paths: set[str] = set()
    for index, row in documents.iterrows():
        if row["document_id"] not in selected_ids:
            continue
        document_url = str(row["adjunct_url"])
        require_official_https_url(
            document_url,
            label=f"CNINFO document URL for {row['document_id']}",
            allowed_hosts={"static.cninfo.com.cn"},
        )
        extension = _document_extension(row)
        relative_path = _document_cache_path(row)
        if relative_path in cache_paths:
            raise ValueError(f"Duplicate CNINFO document cache path: {relative_path}")
        cache_paths.add(relative_path)
        jobs.append(
            _DocumentDownloadJob(
                ordinal=len(jobs),
                frame_index=int(index),
                document_id=str(row["document_id"]),
                source_url_or_identifier=document_url,
                relative_path=relative_path,
                extension=extension,
            )
        )

    if max_workers == 1:
        outcomes = [
            _download_document_job(
                job,
                transport=transport,
                raw_store=raw_store,
                clock=clock,
            )
            for job in jobs
        ]
    else:
        outcomes = _parallel_download_jobs(
            jobs,
            transport=transport,
            raw_store=raw_store,
            clock=clock,
            max_workers=max_workers,
        )

    assets: list[RawAsset] = []
    failures: list[DownloadFailure] = []
    for outcome in outcomes:
        index = outcome.job.frame_index
        if outcome.failure is not None:
            failures.append(outcome.failure)
            documents.at[index, "download_status"] = "FAILED"
            documents.at[index, "failure_reason"] = outcome.failure.reason
            continue
        if outcome.asset is None:  # pragma: no cover - dataclass invariant guard
            raise RuntimeError("CNINFO successful download outcome has no asset")
        asset = outcome.asset
        documents.at[index, "local_raw_path"] = asset.local_path
        documents.at[index, "retrieval_datetime"] = asset.retrieval_datetime
        documents.at[index, "file_size"] = asset.file_size
        documents.at[index, "sha256"] = asset.sha256
        documents.at[index, "download_status"] = "DOWNLOADED"
        documents.at[index, "cache_hit"] = asset.cache_hit
        documents.at[index, "failure_reason"] = None
        assets.append(asset)

    run_timestamp = _utc_timestamp(clock)
    failure_log = write_failure_log(
        raw_store,
        failures,
        run_timestamp=run_timestamp,
    )
    return documents, tuple(assets), tuple(failures), failure_log


def acquire_cninfo_disclosures(
    transport: RateLimitedSession,
    raw_store: RawAssetStore,
    *,
    start_date: str,
    end_date: str,
    mode: AcquisitionMode | None = None,
    query_url: str = CNINFO_QUERY_URL,
    download_base_url: str = CNINFO_DOWNLOAD_BASE_URL,
    category: str = CNINFO_ANNUAL_REPORT_CATEGORY,
    page_size: int = 30,
    clock: Any = utc_now,
    download_workers: int = 1,
) -> DisclosureAcquisitionResult:
    """Acquire metadata, then cache annual-report documents unless trial says not to."""

    mode = mode or AcquisitionMode()
    metadata, metadata_assets = query_annual_report_metadata(
        transport,
        raw_store,
        start_date=start_date,
        end_date=end_date,
        query_url=query_url,
        download_base_url=download_base_url,
        category=category,
        page_size=page_size,
    )
    timing_excluded = int(
        metadata.attrs.get("timing_excluded_after_query_end_count", 0)
    )
    documents, document_assets, failures, failure_log = download_documents(
        metadata,
        transport,
        raw_store,
        mode=mode,
        clock=clock,
        max_workers=download_workers,
    )
    all_assets = (*metadata_assets, *document_assets)
    if failure_log is not None:
        all_assets = (*all_assets, failure_log)
    return DisclosureAcquisitionResult(
        documents=documents,
        assets=tuple(all_assets),
        failures=failures,
        mode=mode,
        failure_log_asset=failure_log,
        timing_excluded_after_query_end_count=timing_excluded,
    )


def disclosure_acquisition_diagnostics(
    result: DisclosureAcquisitionResult,
) -> dict[str, Any]:
    """Evaluate provenance while guarding limited trials from a false PASS."""

    frame = result.documents
    required_metadata = [
        "document_id",
        "announcement_time_ms",
        "source_publication_datetime",
        "publication_datetime",
        "publication_time_precision",
        "publication_timing_rule",
        "source_url_or_identifier",
        "metadata_retrieval_datetime",
        "metadata_sha256",
    ]
    required_download = [
        "local_raw_path",
        "retrieval_datetime",
        "file_size",
        "sha256",
    ]
    duplicate_ids = (
        int(frame.duplicated("document_id", keep=False).sum()) if len(frame) else 0
    )
    missing_metadata = (
        int(frame[required_metadata].isna().any(axis=1).sum()) if len(frame) else 0
    )
    missing_org_id = (
        int(
            (
                frame["cninfo_org_id"].isna()
                | frame["cninfo_org_id"].astype("string").str.strip().eq("")
            ).sum()
        )
        if len(frame) and "cninfo_org_id" in frame
        else len(frame)
    )
    downloaded = (
        frame["download_status"].eq("DOWNLOADED")
        if len(frame)
        else pd.Series(dtype=bool)
    )
    missing_download_provenance = (
        int(frame.loc[downloaded, required_download].isna().any(axis=1).sum())
        if downloaded.any()
        else 0
    )
    timing = publication_timing_diagnostics(frame)
    if not result.mode.production_qa_eligible:
        qa_state = "NOT_EVALUATED_LIMITED_RUN"
    else:
        passed = bool(
            len(frame)
            and duplicate_ids == 0
            and missing_metadata == 0
            and missing_download_provenance == 0
            and timing["passed"]
            and not result.failures
            and downloaded.all()
        )
        qa_state = "PASS" if passed else "FAIL"
    result.mode.guard_qa_state(qa_state)
    publication = pd.to_datetime(
        frame["publication_datetime"], errors="coerce", utc=True
    )
    failure_category_counts = {
        category.value: sum(
            failure.failure_category == category for failure in result.failures
        )
        for category in DownloadFailureCategory
    }
    return {
        "mode": result.mode.label,
        "production_qa_eligible": result.mode.production_qa_eligible,
        "rows": len(frame),
        "unique_document_ids": int(frame["document_id"].nunique()) if len(frame) else 0,
        "duplicate_document_id_count": duplicate_ids,
        "missing_metadata_provenance_count": missing_metadata,
        "missing_org_id_count": missing_org_id,
        "downloaded_count": int(downloaded.sum()) if len(frame) else 0,
        "cache_hit_count": int(frame["cache_hit"].eq(True).sum()) if len(frame) else 0,
        "failed_download_count": len(result.failures),
        "failure_category_counts": failure_category_counts,
        "missing_download_provenance_count": missing_download_provenance,
        "timing_excluded_after_query_end_count": (
            result.timing_excluded_after_query_end_count
        ),
        "publication_datetime_min": publication.min().isoformat()
        if publication.notna().any()
        else None,
        "publication_datetime_max": publication.max().isoformat()
        if publication.notna().any()
        else None,
        "publication_timing": timing,
        "sources": sorted(frame["source"].dropna().astype(str).unique().tolist())
        if len(frame)
        else [],
        "qa_state": qa_state,
    }


__all__ = [
    "CNINFO_ANNUAL_REPORT_CATEGORY",
    "CNINFO_DOWNLOAD_BASE_URL",
    "CNINFO_MAX_RETRIEVABLE_PAGES",
    "CNINFO_OBSERVED_TICKER_ALIAS_SOURCE",
    "CNINFO_OBSERVED_TICKER_ALIAS_TYPE",
    "CNINFO_PUBLICATION_TIMEZONE",
    "CNINFO_QUERY_URL",
    "CNINFO_SHORT_NAME_ALIAS_SOURCE",
    "CNINFO_SHORT_NAME_ALIAS_TYPE",
    "CNINFO_SIGNAL_CUTOFF_LOCAL",
    "CNINFO_SZSE_STOCK_MAP_URL",
    "PUBLICATION_TIME_PRECISION_DATE_ONLY",
    "PUBLICATION_TIME_PRECISION_EXACT",
    "PUBLICATION_TIMING_RULE_DATE_ONLY",
    "PUBLICATION_TIMING_RULE_EXACT",
    "SECURITY_ID_RECONCILIATION_METHODS",
    "CninfoStockMapAcquisition",
    "DisclosureAcquisitionResult",
    "DownloadFailure",
    "DownloadFailureCategory",
    "acquire_cninfo_disclosures",
    "acquire_cninfo_stock_map",
    "deduplicate_announcements",
    "disclosure_acquisition_diagnostics",
    "download_documents",
    "enrich_company_aliases",
    "is_annual_report_title",
    "normalize_announcements",
    "normalize_cninfo_stock_map",
    "publication_timing_diagnostics",
    "query_annual_report_metadata",
    "reconcile_security_ids",
    "stock_map_diagnostics",
    "write_failure_log",
]
