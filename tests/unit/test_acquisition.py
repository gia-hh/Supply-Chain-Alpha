from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from supply_chain_alpha.data.acquisition import (
    RAW_ASSET_REQUIRED_FIELDS,
    AcquisitionMode,
    RateLimitedSession,
    RawAsset,
    RawAssetIntegrityError,
    RawAssetRecoveryRequired,
    RawAssetStore,
    RequestCancelledError,
    RequestPolicy,
    fetch_raw_asset,
    write_raw_asset_manifest,
)
from supply_chain_alpha.data.cninfo import (
    CNINFO_OBSERVED_TICKER_ALIAS_TYPE,
    CNINFO_SHORT_NAME_ALIAS_SOURCE,
    CNINFO_SHORT_NAME_ALIAS_TYPE,
    CNINFO_SZSE_STOCK_MAP_URL,
    DownloadFailureCategory,
    acquire_cninfo_disclosures,
    acquire_cninfo_stock_map,
    deduplicate_announcements,
    disclosure_acquisition_diagnostics,
    download_documents,
    enrich_company_aliases,
    normalize_announcements,
    normalize_cninfo_stock_map,
    publication_timing_diagnostics,
    query_annual_report_metadata,
    reconcile_security_ids,
)
from supply_chain_alpha.data.disclosures import verify_raw_manifest
from supply_chain_alpha.data.exchange_acquisition import (
    SSE_COMMON_QUERY_URL,
    SZSE_SHOW_REPORT_URL,
    acquire_official_security_master,
    company_alias_diagnostics,
)
from supply_chain_alpha.data.schemas import COMPANY_ALIAS, validate_table
from supply_chain_alpha.entities.resolve import resolve_mentions

FIXED_NOW = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def close(self) -> None:
        self.closed = True


class QueueSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"Unexpected network call: {method} {url}")
        return self.responses.pop(0)


class RouterSession:
    def __init__(
        self,
        *,
        sse_pages: list[bytes] | None = None,
        szse_workbooks: Mapping[str, bytes] | None = None,
        cninfo_pages: list[bytes] | None = None,
        documents: Mapping[str, bytes] | None = None,
    ) -> None:
        self.sse_pages = list(sse_pages or [])
        self.szse_workbooks = dict(szse_workbooks or {})
        self.cninfo_pages = list(cninfo_pages or [])
        self.documents = dict(documents or {})
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if url == SSE_COMMON_QUERY_URL:
            if not self.sse_pages:
                raise AssertionError("Unexpected SSE network call")
            return FakeResponse(self.sse_pages.pop(0))
        if url == SZSE_SHOW_REPORT_URL:
            catalog = kwargs["params"]["CATALOGID"]
            if catalog not in self.szse_workbooks:
                raise AssertionError(f"Unexpected SZSE catalog: {catalog}")
            return FakeResponse(self.szse_workbooks.pop(catalog))
        if url.endswith("/new/hisAnnouncement/query"):
            if not self.cninfo_pages:
                raise AssertionError("Unexpected CNINFO metadata call")
            return FakeResponse(self.cninfo_pages.pop(0))
        if url in self.documents:
            return FakeResponse(self.documents.pop(url))
        raise AssertionError(f"Unexpected network call: {method} {url}")


def _transport(session: Any) -> RateLimitedSession:
    return RateLimitedSession(
        RequestPolicy(
            user_agent="test-agent",
            timeout_seconds=1,
            max_attempts=2,
            backoff_seconds=0,
            min_interval_seconds=0,
        ),
        session=session,
    )


def _store(path: Path) -> RawAssetStore:
    return RawAssetStore(path, clock=lambda: FIXED_NOW)


def _xlsx(frame: pd.DataFrame, *, title_rows: int = 0) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, startrow=title_rows)
    return buffer.getvalue()


def test_rate_limiter_retries_transient_status_and_honors_retry_after() -> None:
    clock_value = [0.0]
    sleeps: list[float] = []

    def monotonic() -> float:
        return clock_value[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock_value[0] += seconds

    first = FakeResponse(b"busy", status_code=429, headers={"Retry-After": "3"})
    second = FakeResponse(b"ok")
    session = QueueSession([first, second])
    transport = RateLimitedSession(
        RequestPolicy(
            user_agent="unit-test",
            timeout_seconds=5,
            max_attempts=2,
            backoff_seconds=1,
            min_interval_seconds=0.5,
        ),
        session=session,
        monotonic=monotonic,
        sleep=sleep,
    )
    response = transport.get("https://official.example/data")
    assert response.content == b"ok"
    assert len(session.calls) == 2
    assert sleeps == [3.0]
    assert first.closed
    assert session.calls[0]["headers"]["User-Agent"] == "unit-test"


def test_rate_limiter_spaces_successive_requests() -> None:
    clock_value = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock_value[0] += seconds

    session = QueueSession([FakeResponse(b"one"), FakeResponse(b"two")])
    transport = RateLimitedSession(
        RequestPolicy(min_interval_seconds=0.75),
        session=session,
        monotonic=lambda: clock_value[0],
        sleep=sleep,
    )
    transport.get("https://official.example/one")
    transport.get("https://official.example/two")
    assert sleeps == [0.75]


def test_rate_limiter_disables_and_rejects_redirects() -> None:
    redirected = FakeResponse(
        b"redirect",
        status_code=302,
        headers={"Location": "https://attacker.example/payload"},
    )
    session = QueueSession([redirected])
    transport = _transport(session)

    with pytest.raises(requests.TooManyRedirects, match="not permitted"):
        transport.get("https://official.example/source")

    assert redirected.closed
    assert session.calls[0]["allow_redirects"] is False

    with pytest.raises(ValueError, match="cannot be enabled"):
        transport.get(
            "https://official.example/source",
            allow_redirects=True,
        )


def test_rate_limiter_shared_pacer_spaces_attempts_across_threads() -> None:
    clock_value = [0.0]

    def sleep(seconds: float) -> None:
        clock_value[0] += seconds

    transport = RateLimitedSession(
        RequestPolicy(min_interval_seconds=0.75),
        session=QueueSession([]),
        monotonic=lambda: clock_value[0],
        sleep=sleep,
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        starts = list(executor.map(lambda _: transport._pace(), range(4)))

    assert sorted(starts) == [0.0, 0.75, 1.5, 2.25]


def test_rate_limiter_retry_cooldown_extends_a_waiting_worker() -> None:
    clock_value = [0.0]
    sleep_entered = threading.Event()
    release_sleep = threading.Event()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        sleep_entered.set()
        assert release_sleep.wait(timeout=5)
        clock_value[0] += seconds

    transport = RateLimitedSession(
        RequestPolicy(min_interval_seconds=0.75),
        session=QueueSession([]),
        monotonic=lambda: clock_value[0],
        sleep=sleep,
    )
    assert transport._pace() == 0.0
    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(transport._pace)
        assert sleep_entered.wait(timeout=5)
        transport._defer_all_requests(3.0)
        release_sleep.set()
        assert waiting.result(timeout=5) == 3.0

    assert sleeps == [0.75, 2.25]


def test_rate_limiter_cancellation_prevents_retries() -> None:
    cancel_event = threading.Event()

    class CancellingSession:
        def __init__(self) -> None:
            self.calls = 0

        def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            del method, url, kwargs
            self.calls += 1
            cancel_event.set()
            raise requests.Timeout("slow source")

    session = CancellingSession()
    transport = RateLimitedSession(
        RequestPolicy(
            max_attempts=4,
            backoff_seconds=2,
            min_interval_seconds=0,
        ),
        session=session,
    )

    with pytest.raises(RequestCancelledError, match="cancelled"):
        transport.get("https://official.example/slow", cancel_event=cancel_event)
    assert session.calls == 1


def test_raw_asset_store_is_append_only_and_returns_required_metadata(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    asset = store.store_bytes(
        "official/response.json",
        b'{"ok":true}',
        source="OFFICIAL",
        source_url_or_identifier="https://official.example/response",
    )
    assert RAW_ASSET_REQUIRED_FIELDS <= asset.to_dict().keys()
    assert asset.file_size == len(b'{"ok":true}')
    assert len(asset.sha256) == 64
    cached = store.load("official/response.json")
    assert cached is not None and cached.cache_hit
    assert cached.retrieval_datetime == "2026-08-29T18:00:00Z"

    with pytest.raises(RawAssetIntegrityError, match="Append-only raw asset collision"):
        store.store_bytes(
            "official/response.json",
            b'{"ok":false}',
            source="OFFICIAL",
            source_url_or_identifier="https://official.example/response",
        )


def test_raw_asset_store_rejects_link_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    store_root = tmp_path / "raw"
    store_root.mkdir()
    link = store_root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable on this platform: {exc}")

    store = _store(store_root)
    with pytest.raises(ValueError, match="cannot traverse a link"):
        store.store_bytes(
            "linked/escape.bin",
            b"must stay inside raw root",
            source="OFFICIAL",
            source_url_or_identifier="https://official.example/escape",
        )
    assert not (outside / "escape.bin").exists()


def test_concurrent_identical_raw_asset_creation_converges_on_one_pair(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    def store_same_asset(_: int) -> RawAsset:
        return store.store_bytes(
            "official/concurrent.bin",
            b"same immutable bytes",
            source="OFFICIAL",
            source_url_or_identifier="https://official.example/concurrent",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        assets = list(executor.map(store_same_asset, range(16)))

    assert {asset.sha256 for asset in assets} == {assets[0].sha256}
    assert {asset.retrieval_datetime for asset in assets} == {"2026-08-29T18:00:00Z"}
    cached = store.load("official/concurrent.bin")
    assert cached is not None
    assert cached.sha256 == assets[0].sha256


def test_fetch_recovers_provenance_reserved_sidecar_without_rewriting_metadata(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    url = "https://official.example/recover"
    first_session = QueueSession([FakeResponse(b"recoverable")])
    original, _ = fetch_raw_asset(
        _transport(first_session),
        store,
        "official/recover.bin",
        source="OFFICIAL",
        method="GET",
        url=url,
    )
    asset_path = tmp_path / original.local_path
    sidecar_path = asset_path.with_name(f"{asset_path.name}.metadata.json")
    sidecar_before = sidecar_path.read_bytes()
    asset_path.unlink()

    with pytest.raises(RawAssetRecoveryRequired, match="must be refetched"):
        store.load(original.local_path)

    recovery_session = QueueSession([FakeResponse(b"recoverable")])
    recovered, content = fetch_raw_asset(
        _transport(recovery_session),
        store,
        original.local_path,
        source="OFFICIAL",
        method="GET",
        url=url,
    )

    assert content == b"recoverable"
    assert len(recovery_session.calls) == 1
    assert recovered.cache_hit is True
    assert recovered.retrieval_datetime == original.retrieval_datetime
    assert sidecar_path.read_bytes() == sidecar_before
    assert asset_path.read_bytes() == b"recoverable"


def test_fetch_recovers_matching_orphan_asset_with_new_provenance(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    url = "https://official.example/orphan"
    original, _ = fetch_raw_asset(
        _transport(QueueSession([FakeResponse(b"same-source-bytes")])),
        store,
        "official/orphan.bin",
        source="OFFICIAL",
        method="GET",
        url=url,
    )
    asset_path = tmp_path / original.local_path
    sidecar_path = asset_path.with_name(f"{asset_path.name}.metadata.json")
    sidecar_path.unlink()

    with pytest.raises(RawAssetRecoveryRequired, match="without provenance"):
        store.load(original.local_path)

    recovery_session = QueueSession([FakeResponse(b"same-source-bytes")])
    recovered, _ = fetch_raw_asset(
        _transport(recovery_session),
        store,
        original.local_path,
        source="OFFICIAL",
        method="GET",
        url=url,
    )

    assert len(recovery_session.calls) == 1
    assert recovered.cache_hit is True
    assert asset_path.read_bytes() == b"same-source-bytes"
    assert sidecar_path.is_file()


def test_raw_asset_manifest_covers_assets_and_sidecars_deterministically(
    tmp_path: Path,
) -> None:
    raw_parent = tmp_path / "raw"
    store_root = raw_parent / "official_sources"
    store = _store(store_root)
    second = store.store_bytes(
        "z-source/second.bin",
        b"second",
        source="SECOND_OFFICIAL_SOURCE",
        source_url_or_identifier="https://official.example/second",
    )
    first = store.store_bytes(
        "a-source/first.json",
        b'{"first":true}',
        source="FIRST_OFFICIAL_SOURCE",
        source_url_or_identifier="https://official.example/first",
    )

    manifest = write_raw_asset_manifest(store_root, raw_parent / "MANIFEST.json")
    first_bytes = manifest.read_bytes()
    records = json.loads(first_bytes)
    expected_paths = [
        "official_sources/a-source/first.json",
        "official_sources/a-source/first.json.metadata.json",
        "official_sources/z-source/second.bin",
        "official_sources/z-source/second.bin.metadata.json",
    ]
    assert [record["path"] for record in records] == expected_paths
    assert all(
        set(record)
        == {
            "path",
            "source",
            "source_url_or_identifier",
            "retrieval_datetime",
            "file_size",
            "sha256",
        }
        for record in records
    )
    assert records[0]["source"] == first.source
    assert records[0]["source_url_or_identifier"] == first.source_url_or_identifier
    assert records[0]["retrieval_datetime"] == first.retrieval_datetime
    assert records[0]["file_size"] == first.file_size
    assert records[0]["sha256"] == first.sha256
    assert records[2]["source"] == second.source
    assert records[2]["sha256"] == second.sha256
    verify_raw_manifest(manifest)

    assert write_raw_asset_manifest(store_root, manifest) == manifest
    assert manifest.read_bytes() == first_bytes
    assert not list(raw_parent.glob(".MANIFEST.json.*.tmp"))


def test_raw_asset_manifest_rejects_untracked_files_without_replacing_output(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "official_sources"
    store = _store(store_root)
    store.store_bytes(
        "official/response.json",
        b'{"ok":true}',
        source="OFFICIAL",
        source_url_or_identifier="https://official.example/response",
    )
    manifest = write_raw_asset_manifest(store_root)
    original = manifest.read_bytes()
    (store_root / "untracked.txt").write_text("no provenance", encoding="utf-8")

    with pytest.raises(RawAssetIntegrityError, match="lack valid provenance sidecars"):
        write_raw_asset_manifest(store_root)
    assert manifest.read_bytes() == original


def test_raw_asset_manifest_includes_sidecar_integrity_in_verification(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.store_bytes(
        "official/response.json",
        b'{"ok":true}',
        source="OFFICIAL",
        source_url_or_identifier="https://official.example/response",
    )
    manifest = write_raw_asset_manifest(tmp_path)
    verify_raw_manifest(manifest)

    sidecar = tmp_path / "official" / "response.json.metadata.json"
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="immutability violation"):
        verify_raw_manifest(manifest)


def test_raw_asset_manifest_default_still_hashes_asset_bytes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.store_bytes(
        "official/response.json",
        b'{"ok":true}',
        source="OFFICIAL",
        source_url_or_identifier="https://official.example/response",
    )
    manifest = write_raw_asset_manifest(tmp_path)
    original = manifest.read_bytes()
    (tmp_path / "official" / "response.json").write_bytes(b'{"no":true}')

    with pytest.raises(RawAssetIntegrityError, match="SHA-256 mismatch"):
        write_raw_asset_manifest(tmp_path)

    assert manifest.read_bytes() == original


def test_raw_asset_manifest_deferred_hash_requires_immediate_verification(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.store_bytes(
        "official/response.json",
        b'{"ok":true}',
        source="OFFICIAL",
        source_url_or_identifier="https://official.example/response",
    )
    (tmp_path / "official" / "response.json").write_bytes(b'{"no":true}')

    manifest = write_raw_asset_manifest(tmp_path, verify_asset_bytes=False)

    with pytest.raises(ValueError, match="immutability violation"):
        verify_raw_manifest(manifest, inventory_root=tmp_path)


def test_raw_asset_manifest_verification_rejects_new_unlisted_pair(
    tmp_path: Path,
) -> None:
    raw_parent = tmp_path / "raw"
    store_root = raw_parent / "official_sources"
    store = _store(store_root)
    store.store_bytes(
        "official/first.json",
        b'{"first":true}',
        source="OFFICIAL",
        source_url_or_identifier="https://official.example/first",
    )
    manifest = write_raw_asset_manifest(store_root, raw_parent / "MANIFEST.json")
    verify_raw_manifest(manifest, inventory_root=store_root)

    store.store_bytes(
        "official/second.json",
        b'{"second":true}',
        source="OFFICIAL",
        source_url_or_identifier="https://official.example/second",
    )

    with pytest.raises(ValueError, match="unmanifested files"):
        verify_raw_manifest(manifest, inventory_root=store_root)


def test_fetch_raw_asset_uses_verified_cache_without_network(tmp_path: Path) -> None:
    store = _store(tmp_path)
    session = QueueSession([FakeResponse(b"payload")])
    transport = _transport(session)
    first, content = fetch_raw_asset(
        transport,
        store,
        "source/data.bin",
        source="OFFICIAL",
        method="GET",
        url="https://official.example/data",
    )
    assert content == b"payload" and not first.cache_hit

    no_network = QueueSession([])
    second, content = fetch_raw_asset(
        _transport(no_network),
        store,
        "source/data.bin",
        source="OFFICIAL",
        method="GET",
        url="https://official.example/data",
    )
    assert content == b"payload" and second.cache_hit
    assert not no_network.calls


def test_fetch_raw_asset_cache_hit_uses_single_pass_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    url = "https://official.example/one-pass.bin"
    first, _ = fetch_raw_asset(
        _transport(QueueSession([FakeResponse(b"one-pass-payload")])),
        store,
        "official/one-pass.bin",
        source="OFFICIAL",
        method="GET",
        url=url,
    )

    def duplicate_streaming_load(_: object) -> None:
        raise AssertionError(
            "fetch_raw_asset must not re-read through RawAssetStore.load"
        )

    monkeypatch.setattr(store, "load", duplicate_streaming_load)
    cached, content = fetch_raw_asset(
        _transport(QueueSession([])),
        store,
        first.local_path,
        source="OFFICIAL",
        method="GET",
        url=url,
    )

    assert cached.cache_hit is True
    assert content == b"one-pass-payload"


def test_fetch_raw_asset_recovers_truncated_get_with_strict_ranges(
    tmp_path: Path,
) -> None:
    payload = b"%PDF-1.7\ncomplete-payload"

    class TruncatedThenRangedSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.full_attempts = 0

        def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            self.calls.append({"method": method, "url": url, **kwargs})
            raw_range = kwargs["headers"].get("Range")
            if raw_range is None:
                self.full_attempts += 1
                raise requests.exceptions.ChunkedEncodingError("response ended early")
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", raw_range)
            assert match is not None
            start, requested_end = map(int, match.groups())
            end = min(requested_end, len(payload) - 1)
            return FakeResponse(
                payload[start : end + 1],
                status_code=206,
                headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
            )

    session = TruncatedThenRangedSession()
    asset, content = fetch_raw_asset(
        _transport(session),
        _store(tmp_path),
        "source/report.pdf",
        source="OFFICIAL",
        method="GET",
        url="https://official.example/report.pdf",
        range_fallback_chunk_bytes=7,
    )

    assert session.full_attempts == 2
    assert [call["headers"].get("Range") for call in session.calls[2:]] == [
        "bytes=0-6",
        "bytes=7-13",
        "bytes=14-20",
        "bytes=21-27",
    ]
    assert all(
        call["headers"].get("Cache-Control") == "no-cache"
        and call["headers"].get("Pragma") == "no-cache"
        for call in session.calls[2:]
    )
    assert content == payload
    assert asset.file_size == len(payload)
    assert (tmp_path / asset.local_path).read_bytes() == payload


def test_fetch_raw_asset_rejects_noncontiguous_range_response(tmp_path: Path) -> None:
    class InvalidRangeSession:
        def __init__(self) -> None:
            self.full_attempts = 0

        def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            del method, url
            if kwargs["headers"].get("Range") is None:
                self.full_attempts += 1
                raise requests.exceptions.ChunkedEncodingError("response ended early")
            return FakeResponse(
                b"wrong",
                status_code=206,
                headers={"Content-Range": "bytes 1-5/10"},
            )

    with pytest.raises(ValueError, match="not contiguous"):
        fetch_raw_asset(
            _transport(InvalidRangeSession()),
            _store(tmp_path),
            "source/report.pdf",
            source="OFFICIAL",
            method="GET",
            url="https://official.example/report.pdf",
            range_fallback_chunk_bytes=5,
        )
    assert not (tmp_path / "source/report.pdf").exists()


def _security_source_payloads() -> tuple[list[bytes], dict[str, bytes]]:
    sse_active = json.dumps(
        {
            "result": [
                {
                    "A_STOCK_CODE": "600000",
                    "FULL_NAME": "上海样本股份有限公司",
                    "LIST_DATE": "19991110",
                    "DELIST_DATE": "-",
                    "STOCK_TYPE": "A股",
                },
                {
                    "A_STOCK_CODE": "688001",
                    "FULL_NAME": "科创样本股份有限公司",
                    "LIST_DATE": "20190722",
                    "DELIST_DATE": "-",
                    "STOCK_TYPE": "8",
                },
            ],
            "pageHelp": {"pageCount": 1, "total": 2},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    sse_delisted = json.dumps(
        {
            "result": [
                {
                    "A_STOCK_CODE": "688086",
                    "FULL_NAME": "退市科创样本股份有限公司",
                    "LIST_DATE": "20200226",
                    "DELIST_DATE": "20230707",
                }
            ],
            "pageHelp": {"pageCount": 1, "total": 1},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    active = _xlsx(
        pd.DataFrame(
            [
                {
                    "板块": "主板",
                    "公司全称": "平安银行股份有限公司",
                    "A股代码": "000001",
                    "A股简称": "平安银行",
                    "A股上市日期": "1991-04-03",
                    "所属行业": "货币金融服务",
                },
                {
                    "板块": "创业板",
                    "公司全称": "中航成飞股份有限公司",
                    "A股代码": "302132",
                    "A股简称": "中航成飞",
                    "A股上市日期": "2010-08-27",
                    "所属行业": "航空制造",
                },
            ]
        ),
        title_rows=2,
    )
    delisted = _xlsx(
        pd.DataFrame(
            [
                {
                    "证券代码": "000002",
                    "证券简称": "退市样本",
                    "上市日期": "1992-01-01",
                    "终止上市日期": "2020-01-01",
                }
            ]
        )
    )
    changes = _xlsx(
        pd.DataFrame(
            [
                {
                    "证券代码": "000001",
                    "原公司全称": "深圳发展银行股份有限公司",
                    "变更后公司全称": "平安银行股份有限公司",
                    "变更日期": "2012-08-02",
                }
            ]
        )
    )
    return [sse_active, sse_delisted], {
        "1110": active,
        "1793_ssgs": delisted,
        "SSGSGMXX": changes,
    }


def test_official_security_master_normalization_alias_timing_and_cache(
    tmp_path: Path,
) -> None:
    sse, workbooks = _security_source_payloads()
    session = RouterSession(sse_pages=sse, szse_workbooks=workbooks)
    store = _store(tmp_path)
    result = acquire_official_security_master(_transport(session), store)

    assert result.securities["security_id"].tolist() == [
        "SSE:600000",
        "SSE:688001",
        "SSE:688086",
        "SZSE:000001",
        "SZSE:000002",
        "SZSE:302132",
    ]
    assert result.securities["ticker"].tolist() == [
        "600000",
        "688001",
        "688086",
        "000001",
        "000002",
        "302132",
    ]
    assert result.securities.loc[0, "board"] == "MAIN"
    assert result.securities.loc[1, "board"] == "STAR"
    assert result.securities.loc[2, "security_status"] == "delisted"
    chinext = result.securities.set_index("security_id").loc["SZSE:302132"]
    assert chinext["board"] == "CHINEXT"
    assert chinext["listing_date"] == "2010-08-27"
    sse_calls = [call for call in session.calls if call[1] == SSE_COMMON_QUERY_URL]
    assert [call[2]["params"]["sqlId"] for call in sse_calls] == [
        "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L",
        "COMMON_SSE_CP_GPJCTPZ_GPLB_ZZGP_L",
    ]
    assert all(call[2]["params"]["STOCK_TYPE"] == "1,8" for call in sse_calls)
    assert all(call[2]["params"]["type"] == "inParams" for call in sse_calls)
    assert result.securities["source_industry_name_current"].notna().sum() == 2
    assert result.securities["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    old_name = result.aliases.loc[
        result.aliases["alias"].eq("深圳发展银行股份有限公司")
    ].iloc[0]
    new_name = result.aliases.loc[
        result.aliases["alias"].eq("平安银行股份有限公司")
    ].iloc[0]
    assert (old_name["valid_from"], old_name["valid_to"]) == (
        "1991-04-03",
        "2012-08-02",
    )
    assert new_name["valid_from"] == "2012-08-02"
    sse_current = result.aliases.loc[
        result.aliases["entity_id"].eq("SSE:600000")
        & result.aliases["alias_type"].eq("legal_name_current_snapshot")
    ].iloc[0]
    assert sse_current["valid_from"] == "2026-08-30"
    assert company_alias_diagnostics(result.aliases)["missing_valid_from"] == 0

    cached_session = RouterSession()
    cached = acquire_official_security_master(_transport(cached_session), store)
    assert not cached_session.calls
    assert cached.securities["cache_hit"].all()
    assert all(asset.cache_hit for asset in cached.assets)


def _announcement(
    document_id: str,
    ticker: str,
    title: str,
    announcement_time: int,
    adjunct_url: str,
    org_id: str | None = None,
) -> dict[str, Any]:
    return {
        "announcementId": document_id,
        "orgId": org_id or f"org-{ticker}",
        "secCode": ticker,
        "secName": f"证券{ticker}",
        "announcementTitle": title,
        "announcementTime": announcement_time,
        "adjunctUrl": adjunct_url,
        "adjunctType": "PDF",
    }


def _cninfo_page(
    announcements: list[dict[str, Any]],
    page_count: int = 1,
    *,
    total_count: int | None = None,
) -> bytes:
    return json.dumps(
        {
            "announcements": announcements,
            "totalpages": page_count,
            "totalAnnouncement": (
                len(announcements) if total_count is None else total_count
            ),
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _metadata_asset() -> RawAsset:
    return RawAsset(
        source="CNINFO",
        source_url_or_identifier="https://www.cninfo.com.cn/query#request_sha256=test",
        retrieval_datetime="2026-08-29T18:00:00Z",
        file_size=1,
        sha256="a" * 64,
        local_path="cninfo/metadata/test.json",
    )


def _stock_map_asset() -> RawAsset:
    return RawAsset(
        source="CNINFO SZSE stock map",
        source_url_or_identifier=CNINFO_SZSE_STOCK_MAP_URL,
        retrieval_datetime="2026-08-30T04:20:00Z",
        file_size=1,
        sha256="b" * 64,
        local_path="cninfo/security_master/szse_stock.json",
    )


def _stock_map_payload() -> bytes:
    return json.dumps(
        {
            "stockList": [
                {"code": "300114", "orgId": "9900013408", "category": "A股"},
                {"code": "302132", "orgId": "9900013408", "category": "A股"},
                {"code": "000001", "orgId": "gssz0000001", "category": "A股"},
                {"code": "200001", "orgId": "gssz0000001", "category": "B股"},
                {"code": "430001", "orgId": "bj-test", "category": "A股"},
            ]
        },
        ensure_ascii=False,
    ).encode("utf-8")


def test_cninfo_stock_map_is_strict_filtered_and_immutable(tmp_path: Path) -> None:
    normalized = normalize_cninfo_stock_map(_stock_map_payload(), _stock_map_asset())
    assert normalized["security_id"].tolist() == [
        "SZSE:000001",
        "SZSE:300114",
        "SZSE:302132",
    ]
    assert (
        normalized.loc[
            normalized["security_id"].isin(["SZSE:300114", "SZSE:302132"]),
            "cninfo_org_id",
        ]
        .eq("9900013408")
        .all()
    )
    assert normalized["category"].eq("A股").all()

    result = acquire_cninfo_stock_map(
        _transport(QueueSession([FakeResponse(_stock_map_payload())])),
        _store(tmp_path),
        url=CNINFO_SZSE_STOCK_MAP_URL,
    )
    raw_path = tmp_path / result.asset.local_path
    assert raw_path.read_bytes() == _stock_map_payload()
    assert raw_path.with_name(f"{raw_path.name}.metadata.json").is_file()


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        json.dumps({"stockList": {}}).encode(),
        json.dumps({"stockList": [{"code": "000001", "category": "A股"}]}).encode(),
        json.dumps(
            {"stockList": [{"code": 1, "orgId": "ORG", "category": "A股"}]}
        ).encode(),
        json.dumps(
            {"stockList": [{"code": "200001", "orgId": "ORG", "category": "B股"}]}
        ).encode(),
    ],
)
def test_cninfo_stock_map_rejects_malformed_or_empty_target_data(
    payload: bytes,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_cninfo_stock_map(payload, _stock_map_asset())


def test_cninfo_stock_map_duplicate_code_org_conflict_fails_closed() -> None:
    payload = json.dumps(
        {
            "stockList": [
                {"code": "302132", "orgId": "ORG-A", "category": "A股"},
                {"code": "302132", "orgId": "ORG-B", "category": "A股"},
            ]
        },
        ensure_ascii=False,
    ).encode()

    with pytest.raises(ValueError, match="multiple orgIds"):
        normalize_cninfo_stock_map(payload, _stock_map_asset())


def test_cninfo_no_year_title_must_identify_issuer_or_be_generic() -> None:
    issuer_report = _announcement(
        "ISSUER",
        "600432",
        "证券600432年报",
        1619740800000,
        "/issuer.PDF",
    )
    generic_report = _announcement(
        "GENERIC",
        "600432",
        "年报全文",
        1619740800000,
        "/generic.PDF",
    )
    unrelated_attachment = _announcement(
        "INVESTEE",
        "600432",
        "NTU年报",
        1619740800000,
        "/investee.PDF",
    )

    metadata = normalize_announcements(
        [issuer_report, generic_report, unrelated_attachment],
        _metadata_asset(),
    )

    assert metadata["document_id"].tolist() == ["ISSUER", "GENERIC"]


def test_cninfo_metadata_preserves_stable_org_id() -> None:
    metadata = normalize_announcements(
        [
            _announcement(
                "D1",
                "000043",
                "2018年年度报告",
                1554048000000,
                "/D1.PDF",
                org_id="gssz0000043",
            )
        ],
        _metadata_asset(),
    )

    assert metadata.loc[0, "cninfo_org_id"] == "gssz0000043"


def test_cninfo_metadata_preserves_missing_org_id_without_guessing() -> None:
    announcement = _announcement(
        "D1", "000043", "2018年年度报告", 1554048000000, "/D1.PDF"
    )
    announcement["orgId"] = None

    metadata = normalize_announcements([announcement], _metadata_asset())

    assert pd.isna(metadata.loc[0, "cninfo_org_id"])
    assert metadata.loc[0, "security_id"] == "SZSE:000043"


def test_cninfo_midnight_timestamp_is_date_only_and_delayed_fail_closed() -> None:
    midnight_ms = int(pd.Timestamp("2021-04-30T00:00:00+08:00").timestamp() * 1000)
    precise_ms = int(pd.Timestamp("2021-04-30T14:59:59+08:00").timestamp() * 1000)
    after_cutoff_ms = int(pd.Timestamp("2021-04-30T15:00:01+08:00").timestamp() * 1000)
    metadata = normalize_announcements(
        [
            _announcement(
                "DATE-ONLY",
                "000001",
                "2020年年度报告",
                midnight_ms,
                "/finalpage/2021-04-30/DATE-ONLY.PDF",
            ),
            _announcement(
                "EXACT",
                "000002",
                "2020年年度报告",
                precise_ms,
                "/finalpage/2021-04-30/EXACT.PDF",
            ),
            _announcement(
                "AFTER-CUTOFF",
                "000003",
                "2020年年度报告",
                after_cutoff_ms,
                "/finalpage/2021-04-30/AFTER-CUTOFF.PDF",
            ),
        ],
        _metadata_asset(),
    ).set_index("document_id")

    date_only = metadata.loc["DATE-ONLY"]
    assert date_only["announcement_time_ms"] == midnight_ms
    assert date_only["source_publication_datetime"] == "2021-04-30T00:00:00+08:00"
    assert date_only["publication_datetime"] == "2021-05-01T00:00:00+08:00"
    assert pd.Timestamp(date_only["publication_datetime"]) > pd.Timestamp(
        "2021-04-30T15:00:00+08:00"
    )
    assert pd.Timestamp(date_only["publication_datetime"]) <= pd.Timestamp(
        "2021-05-01T15:00:00+08:00"
    )
    assert date_only["publication_time_precision"] == "date_only"
    assert date_only["publication_timing_rule"] == "next_calendar_day_midnight"

    exact = metadata.loc["EXACT"]
    assert exact["source_publication_datetime"] == "2021-04-30T14:59:59+08:00"
    assert exact["publication_datetime"] == exact["source_publication_datetime"]
    assert exact["publication_time_precision"] == "exact"
    assert exact["publication_timing_rule"] == "cninfo_source_timestamp"

    timing = publication_timing_diagnostics(metadata.reset_index())
    assert timing["date_only_source_timestamp_count"] == 1
    assert timing["exact_source_timestamp_count"] == 2
    assert timing["conservatively_shifted_count"] == 1
    assert timing["exact_source_timestamp_after_signal_cutoff_count"] == 1
    assert timing["publication_timing_violation_count"] == 0
    assert timing["passed"] is True


def test_cninfo_publication_timing_diagnostics_rejects_early_availability() -> None:
    midnight_ms = int(pd.Timestamp("2021-04-30T00:00:00+08:00").timestamp() * 1000)
    metadata = normalize_announcements(
        [
            _announcement(
                "D1",
                "000001",
                "2020年年度报告",
                midnight_ms,
                "/finalpage/2021-04-30/D1.PDF",
            )
        ],
        _metadata_asset(),
    )
    metadata.loc[0, "publication_datetime"] = metadata.loc[
        0, "source_publication_datetime"
    ]

    timing = publication_timing_diagnostics(metadata)

    assert timing["publication_timing_violation_count"] == 1
    assert timing["passed"] is False


def test_cninfo_query_excludes_date_only_end_boundary_before_download(
    tmp_path: Path,
) -> None:
    midnight_ms = int(pd.Timestamp("2022-12-31T00:00:00+08:00").timestamp() * 1000)
    precise_ms = int(pd.Timestamp("2022-12-31T14:00:00+08:00").timestamp() * 1000)
    session = RouterSession(
        cninfo_pages=[
            _cninfo_page(
                [
                    _announcement(
                        "DATE-ONLY",
                        "000001",
                        "2021年年度报告（修订版）",
                        midnight_ms,
                        "/finalpage/2022-12-31/DATE-ONLY.PDF",
                    ),
                    _announcement(
                        "EXACT",
                        "000002",
                        "2021年年度报告（修订版）",
                        precise_ms,
                        "/finalpage/2022-12-31/EXACT.PDF",
                    ),
                ]
            )
        ]
    )

    documents, _ = query_annual_report_metadata(
        _transport(session),
        _store(tmp_path),
        start_date="2022-12-31",
        end_date="2022-12-31",
    )

    assert documents["document_id"].tolist() == ["EXACT"]
    assert documents.attrs["timing_excluded_after_query_end_count"] == 1


@pytest.mark.parametrize(
    "adjunct_url",
    [
        "https://attacker.example/report.pdf",
        "https://static.cninfo.com.cn@attacker.example/report.pdf",
        "javascript:alert(1)",
    ],
)
def test_cninfo_metadata_rejects_nonofficial_document_url(
    adjunct_url: str,
) -> None:
    announcement = _announcement(
        "D1",
        "000043",
        "2018年年度报告",
        1554048000000,
        adjunct_url,
    )

    with pytest.raises(ValueError, match="approved official host"):
        normalize_announcements([announcement], _metadata_asset())


def test_cninfo_download_public_api_revalidates_document_url(tmp_path: Path) -> None:
    metadata = normalize_announcements(
        [
            _announcement(
                "D1",
                "000043",
                "2018年年度报告",
                1554048000000,
                "/D1.PDF",
            )
        ],
        _metadata_asset(),
    )
    metadata.loc[0, "adjunct_url"] = "https://attacker.example/report.pdf"
    session = RouterSession()

    with pytest.raises(ValueError, match="approved official host"):
        download_documents(metadata, _transport(session), _store(tmp_path))

    assert not session.calls


@pytest.mark.parametrize(
    ("column", "conflicting_value"),
    [
        ("cninfo_org_id", "ORG-B"),
        ("security_id", "SZSE:000044"),
        ("security_name", "另一证券"),
        ("announcement_time_ms", 1554048000001),
        ("adjunct_url", "https://static.cninfo.com.cn/D1-other.PDF"),
    ],
)
def test_cninfo_duplicate_document_identity_conflict_fails_closed(
    column: str,
    conflicting_value: object,
) -> None:
    first = normalize_announcements(
        [
            _announcement(
                "D1",
                "000043",
                "2018年年度报告",
                1554048000000,
                "/D1.PDF",
                org_id="ORG-A",
            )
        ],
        _metadata_asset(),
    )
    second = first.copy()
    second.loc[0, column] = conflicting_value

    with pytest.raises(ValueError, match="disagree on issuer/document evidence"):
        deduplicate_announcements(pd.concat([first, second], ignore_index=True))


def test_cninfo_security_id_reconciliation_is_unique_and_non_collapsing() -> None:
    documents = pd.DataFrame(
        {
            "document_id": [
                "OLD",
                "NEW",
                "OLD-LISTED",
                "NEW-LISTED",
                "NONE",
                "ZERO",
                "AMB",
            ],
            "cninfo_org_id": [
                "ORG-A",
                "ORG-A",
                "ORG-B",
                "ORG-B",
                "ORG-C",
                "ORG-E",
                "ORG-D",
            ],
            "security_id": [
                "SZSE:000043",
                "SZSE:001914",
                "SSE:601268",
                "SSE:601399",
                "SZSE:300114",
                "SZSE:399999",
                "SSE:600099",
            ],
        }
    )
    # ORG-D has two authoritative securities plus one unknown ID.  The two
    # authoritative rows must never be collapsed, and its unknown row must not
    # be guessed to either one.
    documents = pd.concat(
        [
            documents,
            pd.DataFrame(
                {
                    "document_id": [
                        "AMB-A",
                        "AMB-B",
                        "MISSING-ORG-DIRECT",
                        "MISSING-ORG-UNKNOWN",
                    ],
                    "cninfo_org_id": ["ORG-D", "ORG-D", None, None],
                    "security_id": [
                        "SSE:600001",
                        "SSE:600002",
                        "SSE:600003",
                        "SSE:600004",
                    ],
                }
            ),
        ],
        ignore_index=True,
    )
    master = pd.DataFrame(
        {
            "security_id": [
                "SZSE:001914",
                "SSE:601268",
                "SSE:601399",
                "SSE:600001",
                "SSE:600002",
                "SSE:600003",
                "SZSE:302132",
            ]
        }
    )
    stock_map = pd.DataFrame(
        {
            "cninfo_org_id": ["ORG-C", "ORG-C"],
            "security_id": ["SZSE:300114", "SZSE:302132"],
        }
    )

    reconciled, diagnostics = reconcile_security_ids(documents, master, stock_map)
    keyed = reconciled.set_index("document_id")

    assert keyed.loc["OLD", "reported_security_id"] == "SZSE:000043"
    assert keyed.loc["OLD", "security_id"] == "SZSE:001914"
    assert (
        keyed.loc["OLD", "security_id_reconciliation_method"]
        == "org_id_unique_master_match"
    )
    assert keyed.loc["NEW", "security_id"] == "SZSE:001914"
    assert keyed.loc["OLD-LISTED", "security_id"] == "SSE:601268"
    assert keyed.loc["NEW-LISTED", "security_id"] == "SSE:601399"
    assert keyed.loc["NONE", "security_id"] == "SZSE:302132"
    assert (
        keyed.loc["NONE", "security_id_reconciliation_method"]
        == "org_id_unique_master_match"
    )
    assert keyed.loc["ZERO", "security_id"] == "SZSE:399999"
    assert (
        keyed.loc["ZERO", "security_id_reconciliation_method"]
        == "unresolved_no_master_match"
    )
    assert keyed.loc["AMB", "security_id"] == "SSE:600099"
    assert (
        keyed.loc["AMB", "security_id_reconciliation_method"]
        == "unresolved_ambiguous_master_match"
    )
    assert keyed.loc["AMB", "security_id_reconciliation_candidate_count"] == 2
    assert keyed.loc["MISSING-ORG-DIRECT", "security_id"] == "SSE:600003"
    assert (
        keyed.loc["MISSING-ORG-DIRECT", "security_id_reconciliation_method"]
        == "reported_id_in_master_missing_org_id"
    )
    assert keyed.loc["MISSING-ORG-UNKNOWN", "security_id"] == "SSE:600004"
    assert (
        keyed.loc["MISSING-ORG-UNKNOWN", "security_id_reconciliation_method"]
        == "unresolved_missing_org_id"
    )
    assert diagnostics["missing_org_id_count"] == 2
    assert diagnostics["mapped_via_org_id_count"] == 2
    assert diagnostics["unresolved_no_master_match_count"] == 1
    assert diagnostics["unresolved_missing_org_id_count"] == 1
    assert diagnostics["unresolved_ambiguous_master_match_count"] == 1


def test_cninfo_alias_enrichment_starts_at_first_observed_publication() -> None:
    aliases = pd.DataFrame(
        {
            "entity_id": ["SZSE:001914"],
            "canonical_name": ["招商局积余产业运营服务股份有限公司"],
            "alias": ["001914"],
            "alias_type": ["ticker"],
            "valid_from": ["1994-09-28"],
            "valid_to": [None],
            "source": ["SZSE official listing interval"],
        }
    )
    documents = pd.DataFrame(
        {
            "document_id": ["LATE", "EARLY", "CURRENT", "UNRESOLVED"],
            "cninfo_org_id": ["gssz0000043"] * 3 + ["UNRESOLVED"],
            "publication_datetime": [
                "2019-04-01T00:00:00+08:00",
                "2015-03-20T00:00:00+08:00",
                "2020-04-01T00:00:00+08:00",
                "2021-03-17T00:00:00+08:00",
            ],
            "reported_security_id": [
                "SZSE:000043",
                "SZSE:000043",
                "SZSE:001914",
                "SZSE:300114",
            ],
            "security_id": ["SZSE:001914"] * 3 + ["SZSE:300114"],
            "security_name": ["中航善达", "中航善达", "招商积余", "中航电测"],
            "security_id_reconciliation_method": [
                "org_id_unique_master_match",
                "org_id_unique_master_match",
                "reported_id_in_master",
                "unresolved_no_master_match",
            ],
        }
    )
    master = pd.DataFrame(
        {
            "security_id": ["SZSE:001914"],
            "company_name": ["招商局积余产业运营服务股份有限公司"],
        }
    )
    stock_map = pd.DataFrame(
        {
            "cninfo_org_id": ["gssz0000043", "gssz0000043"],
            "security_id": ["SZSE:000043", "SZSE:001914"],
            "retrieval_datetime": ["2026-08-29T18:00:00Z"] * 2,
            "raw_local_path": ["cninfo/security_master/szse_stock.json"] * 2,
        }
    )

    enriched, diagnostics = enrich_company_aliases(
        aliases, documents, master, stock_map
    )
    cninfo = enriched.loc[enriched["alias_type"].eq(CNINFO_SHORT_NAME_ALIAS_TYPE)]

    assert cninfo["alias"].tolist() == ["中航善达", "招商积余"]
    assert cninfo.set_index("alias").loc["中航善达", "valid_from"] == "2015-03-20"
    assert cninfo["valid_to"].isna().all()
    assert (
        cninfo.set_index("alias").loc["中航善达", "source"]
        == f"{CNINFO_SHORT_NAME_ALIAS_SOURCE}EARLY"
    )
    assert "中航电测" not in set(enriched["alias"])
    current_ticker = enriched.loc[
        enriched["alias_type"].eq("ticker") & enriched["alias"].eq("001914")
    ].iloc[0]
    assert current_ticker["valid_from"] == "2020-04-01"
    old_ticker = enriched.loc[
        enriched["alias_type"].eq(CNINFO_OBSERVED_TICKER_ALIAS_TYPE)
        & enriched["alias"].eq("000043")
    ].iloc[0]
    assert old_ticker["valid_from"] == "2015-03-20"
    assert pd.isna(old_ticker["valid_to"])
    assert old_ticker["source"].endswith("EARLY")
    assert diagnostics["constrained_backdated_ticker_alias_rows"] == 1
    assert diagnostics["observed_historical_ticker_alias_rows"] == 1
    assert diagnostics["added_alias_rows"] == 3

    reversed_enriched, reversed_diagnostics = enrich_company_aliases(
        aliases,
        documents.iloc[::-1].reset_index(drop=True),
        master,
        stock_map.iloc[::-1].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(enriched, reversed_enriched)
    assert diagnostics == reversed_diagnostics

    mention = pd.DataFrame(
        {
            "source_company_id": ["SZSE:001914"],
            "counterparty_raw_name": ["招商积余"],
            "relationship_type": ["customer"],
            "source_period_end": ["2018-12-31"],
            "publication_datetime": ["2019-04-01T00:00:00+08:00"],
            "source_document_id": ["MENTION"],
            "source_document_url_or_path": ["fixture://MENTION"],
            "evidence_text": ["客户名称 招商积余"],
            "exposure_value": [None],
            "exposure_share": [None],
        }
    )
    past = resolve_mentions(mention, enriched)
    assert past.loc[0, "resolution_status"] == "unresolved"

    mention.loc[0, "publication_datetime"] = "2021-04-01T00:00:00+08:00"
    future = resolve_mentions(mention, enriched)
    assert future.loc[0, "resolution_status"] == "resolved"
    assert future.loc[0, "resolved_entity_id"] == "SZSE:001914"

    mention.loc[0, "counterparty_raw_name"] = "001914"
    mention.loc[0, "publication_datetime"] = "2019-04-01T00:00:00+08:00"
    assert (
        resolve_mentions(mention, enriched).loc[0, "resolution_status"] == "unresolved"
    )
    mention.loc[0, "publication_datetime"] = "2021-04-01T00:00:00+08:00"
    assert resolve_mentions(mention, enriched).loc[0, "resolution_status"] == "resolved"


def test_unobserved_current_change_code_starts_at_stock_map_retrieval_day() -> None:
    aliases = pd.DataFrame(
        {
            "entity_id": ["SZSE:302132"],
            "canonical_name": ["中航成飞股份有限公司"],
            "alias": ["302132"],
            "alias_type": ["ticker"],
            "valid_from": ["2010-08-27"],
            "valid_to": [None],
            "source": ["SZSE official listing interval"],
        }
    )
    documents = pd.DataFrame(
        {
            "document_id": ["OLD-CODE-EVIDENCE"],
            "cninfo_org_id": ["9900013408"],
            "publication_datetime": ["2021-04-01T00:00:00+08:00"],
            "reported_security_id": ["SZSE:300114"],
            "security_id": ["SZSE:302132"],
            "security_name": ["中航电测"],
            "security_id_reconciliation_method": ["org_id_unique_master_match"],
        }
    )
    master = pd.DataFrame(
        {
            "security_id": ["SZSE:302132"],
            "company_name": ["中航成飞股份有限公司"],
        }
    )
    stock_map = pd.DataFrame(
        {
            "cninfo_org_id": ["9900013408", "9900013408"],
            "security_id": ["SZSE:300114", "SZSE:302132"],
            "retrieval_datetime": ["2026-08-29T18:00:00Z"] * 2,
            "raw_local_path": ["cninfo/security_master/szse_stock.json"] * 2,
        }
    )

    enriched, diagnostics = enrich_company_aliases(
        aliases, documents, master, stock_map
    )
    tickers = enriched.loc[enriched["alias"].isin(["300114", "302132"])]
    assert tickers.set_index("alias").loc["300114", "valid_from"] == "2021-04-01"
    assert tickers.set_index("alias").loc["302132", "valid_from"] == "2026-08-30"
    assert diagnostics["constrained_backdated_ticker_alias_rows"] == 1

    mention = pd.DataFrame(
        {
            "source_company_id": ["SZSE:302132"],
            "counterparty_raw_name": ["302132"],
            "relationship_type": ["customer"],
            "source_period_end": ["2024-12-31"],
            "publication_datetime": ["2025-04-01T00:00:00+08:00"],
            "source_document_id": ["MENTION"],
            "source_document_url_or_path": ["fixture://MENTION"],
            "evidence_text": ["客户代码 302132"],
            "exposure_value": [None],
            "exposure_share": [None],
        }
    )
    assert (
        resolve_mentions(mention, enriched).loc[0, "resolution_status"] == "unresolved"
    )
    mention.loc[0, "publication_datetime"] = "2026-08-31T00:00:00+08:00"
    assert resolve_mentions(mention, enriched).loc[0, "resolution_status"] == "resolved"


def test_stock_map_retrieval_after_closed_ticker_interval_does_not_invert_it() -> None:
    aliases = pd.DataFrame(
        {
            "entity_id": ["SZSE:300114", "SZSE:302132"],
            "canonical_name": ["中航电测股份有限公司", "中航成飞股份有限公司"],
            "alias": ["300114", "302132"],
            "alias_type": ["ticker", "ticker"],
            "valid_from": ["2010-08-27", "2025-02-17"],
            "valid_to": ["2025-02-17", None],
            "source": [
                "SZSE official closed ticker interval",
                "SZSE official current ticker interval",
            ],
        }
    )
    documents = pd.DataFrame(
        columns=[
            "document_id",
            "cninfo_org_id",
            "publication_datetime",
            "reported_security_id",
            "security_id",
            "security_name",
            "security_id_reconciliation_method",
        ]
    )
    master = pd.DataFrame(
        {
            "security_id": ["SZSE:300114", "SZSE:302132"],
            "company_name": ["中航电测股份有限公司", "中航成飞股份有限公司"],
        }
    )
    stock_map = pd.DataFrame(
        {
            "cninfo_org_id": ["9900013408", "9900013408"],
            "security_id": ["SZSE:300114", "SZSE:302132"],
            "retrieval_datetime": ["2026-08-29T18:00:00Z"] * 2,
            "raw_local_path": ["cninfo/security_master/szse_stock.json"] * 2,
        }
    )

    enriched, diagnostics = enrich_company_aliases(
        aliases, documents, master, stock_map
    )

    validate_table(enriched, COMPANY_ALIAS)
    tickers = enriched.loc[enriched["alias_type"].eq("ticker")].set_index("alias")
    old_ticker = tickers.loc["300114"]
    assert old_ticker["valid_from"] == "2010-08-27"
    assert old_ticker["valid_to"] == "2025-02-17"
    assert old_ticker["source"] == "SZSE official closed ticker interval"
    assert tickers.loc["302132", "valid_from"] == "2026-08-30"
    assert diagnostics["constrained_backdated_ticker_alias_rows"] == 1
    assert diagnostics["preserved_closed_ticker_alias_rows"] == 1
    assert diagnostics["observed_historical_ticker_alias_rows"] == 0


def test_parallel_document_downloads_use_thread_local_sessions_and_ordered_results(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(3)
    third_done = threading.Event()
    second_done = threading.Event()
    completion_order: list[str] = []
    created_sessions: list[Any] = []
    created_lock = threading.Lock()

    class WorkerSession:
        def __init__(self) -> None:
            self.thread_ids: set[int] = set()
            self.close_count = 0

        def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            del method, kwargs
            self.thread_ids.add(threading.get_ident())
            barrier.wait(timeout=5)
            document_id = Path(url).stem
            if document_id == "D3":
                completion_order.append(document_id)
                third_done.set()
            elif document_id == "D2":
                assert third_done.wait(timeout=5)
                completion_order.append(document_id)
                second_done.set()
            else:
                assert second_done.wait(timeout=5)
                completion_order.append(document_id)
            return FakeResponse(f"%PDF-1.7\n{document_id}".encode())

        def close(self) -> None:
            self.close_count += 1

    def session_factory() -> WorkerSession:
        session = WorkerSession()
        with created_lock:
            created_sessions.append(session)
        return session

    announcements = [
        _announcement(
            document_id,
            ticker,
            "2019年年度报告",
            1577808000000,
            f"/{document_id}.PDF",
        )
        for document_id, ticker in [
            ("D1", "000001"),
            ("D2", "600000"),
            ("D3", "000002"),
        ]
    ]
    metadata = normalize_announcements(announcements, _metadata_asset())
    store = _store(tmp_path)
    transport = RateLimitedSession(
        RequestPolicy(min_interval_seconds=0),
        session_factory=session_factory,
    )

    documents, assets, failures, failure_log = download_documents(
        metadata,
        transport,
        store,
        clock=lambda: FIXED_NOW,
        max_workers=3,
    )

    assert completion_order == ["D3", "D2", "D1"]
    assert documents["document_id"].tolist() == ["D1", "D2", "D3"]
    assert [Path(asset.local_path).name.split("_", 1)[0] for asset in assets] == [
        "D1",
        "D2",
        "D3",
    ]
    assert not failures and failure_log is None
    assert len(created_sessions) == 3
    assert all(len(session.thread_ids) == 1 for session in created_sessions)
    assert len({next(iter(session.thread_ids)) for session in created_sessions}) == 3
    assert all(session.close_count == 1 for session in created_sessions)

    cached, cached_assets, cached_failures, _ = download_documents(
        metadata,
        _transport(RouterSession()),
        store,
        clock=lambda: FIXED_NOW,
        max_workers=3,
    )
    assert cached["cache_hit"].all()
    assert all(asset.cache_hit for asset in cached_assets)
    assert not cached_failures


def test_parallel_document_failures_are_reported_in_document_order(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(3)
    third_done = threading.Event()
    second_done = threading.Event()
    completion_order: list[str] = []

    class MixedSession:
        def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            del method, kwargs
            barrier.wait(timeout=5)
            document_id = Path(url).stem
            if document_id == "D3":
                completion_order.append(document_id)
                third_done.set()
            elif document_id == "D2":
                assert third_done.wait(timeout=5)
                completion_order.append(document_id)
                second_done.set()
            else:
                assert second_done.wait(timeout=5)
                completion_order.append(document_id)
            content = (
                f"%PDF-1.7\n{document_id}".encode()
                if document_id == "D1"
                else b"not a pdf"
            )
            return FakeResponse(content)

        def close(self) -> None:
            return None

    metadata = normalize_announcements(
        [
            _announcement(
                document_id,
                ticker,
                "2019年年度报告",
                1577808000000,
                f"/{document_id}.PDF",
            )
            for document_id, ticker in [
                ("D1", "000001"),
                ("D2", "600000"),
                ("D3", "000002"),
            ]
        ],
        _metadata_asset(),
    )
    transport = RateLimitedSession(
        RequestPolicy(min_interval_seconds=0),
        session_factory=MixedSession,
    )

    documents, assets, failures, failure_log = download_documents(
        metadata,
        transport,
        _store(tmp_path),
        clock=lambda: FIXED_NOW,
        max_workers=3,
    )

    assert completion_order == ["D3", "D2", "D1"]
    assert [Path(asset.local_path).name.split("_", 1)[0] for asset in assets] == ["D1"]
    assert [failure.document_id for failure in failures] == ["D2", "D3"]
    assert documents["download_status"].tolist() == [
        "DOWNLOADED",
        "FAILED",
        "FAILED",
    ]
    assert failure_log is not None
    failure_lines = (
        (tmp_path / failure_log.local_path).read_text(encoding="utf-8").splitlines()
    )
    assert [json.loads(line)["document_id"] for line in failure_lines] == ["D2", "D3"]


def test_parallel_document_download_propagates_unexpected_worker_error(
    tmp_path: Path,
) -> None:
    created_sessions: list[Any] = []

    class BrokenSession:
        def __init__(self) -> None:
            self.close_count = 0

        def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            del method, url, kwargs
            raise AssertionError("unexpected worker defect")

        def close(self) -> None:
            self.close_count += 1

    def session_factory() -> BrokenSession:
        session = BrokenSession()
        created_sessions.append(session)
        return session

    metadata = normalize_announcements(
        [
            _announcement(
                "D1",
                "000001",
                "2019年年度报告",
                1577808000000,
                "/D1.PDF",
            )
        ],
        _metadata_asset(),
    )
    transport = RateLimitedSession(
        RequestPolicy(min_interval_seconds=0),
        session_factory=session_factory,
    )

    with pytest.raises(AssertionError, match="unexpected worker defect"):
        download_documents(
            metadata,
            transport,
            _store(tmp_path),
            max_workers=2,
        )

    assert created_sessions
    assert all(session.close_count == 1 for session in created_sessions)
    assert not list(tmp_path.rglob("*.metadata.json"))


@pytest.mark.parametrize("workers", [True, 0, -1, 9])
def test_document_download_rejects_invalid_worker_count(
    tmp_path: Path,
    workers: object,
) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        download_documents(
            pd.DataFrame(),
            _transport(RouterSession()),
            _store(tmp_path),
            max_workers=workers,  # type: ignore[arg-type]
        )


def test_cninfo_metadata_query_partitions_cross_year_ranges(tmp_path: Path) -> None:
    session = RouterSession(
        cninfo_pages=[_cninfo_page([]), _cninfo_page([])],
    )

    documents, assets = query_annual_report_metadata(
        _transport(session),
        _store(tmp_path),
        start_date="2020-12-31",
        end_date="2021-01-01",
    )

    assert documents.empty
    assert len(assets) == 2
    forms = [call[2]["data"] for call in session.calls]
    assert [form["seDate"] for form in forms] == [
        "2020-12-31~2020-12-31",
        "2021-01-01~2021-01-01",
    ]
    assert len({asset.local_path for asset in assets}) == 2


def test_cninfo_metadata_query_splits_windows_before_page_101(
    tmp_path: Path,
) -> None:
    session = RouterSession(
        cninfo_pages=[
            _cninfo_page([], page_count=101),
            _cninfo_page([]),
            _cninfo_page([]),
        ],
    )

    documents, assets = query_annual_report_metadata(
        _transport(session),
        _store(tmp_path),
        start_date="2020-01-01",
        end_date="2020-01-02",
    )

    assert documents.empty
    assert len(assets) == 3
    forms = [call[2]["data"] for call in session.calls]
    assert [form["seDate"] for form in forms] == [
        "2020-01-01~2020-01-02",
        "2020-01-01~2020-01-01",
        "2020-01-02~2020-01-02",
    ]
    assert {form["pageNum"] for form in forms} == {"1"}


def test_cninfo_metadata_query_discards_oversized_parent_rows(
    tmp_path: Path,
) -> None:
    parent = _announcement(
        "PARENT",
        "000001",
        "2019年年度报告",
        1577808000000,
        "/parent.PDF",
    )
    left = _announcement(
        "LEFT",
        "000001",
        "2019年年度报告",
        1577844000000,
        "/left.PDF",
    )
    right = _announcement(
        "RIGHT",
        "600000",
        "2019年年度报告",
        1577930400000,
        "/right.PDF",
    )
    session = RouterSession(
        cninfo_pages=[
            _cninfo_page([parent], page_count=101),
            _cninfo_page([left]),
            _cninfo_page([right]),
        ],
    )

    documents, assets = query_annual_report_metadata(
        _transport(session),
        _store(tmp_path),
        start_date="2020-01-01",
        end_date="2020-01-02",
    )

    assert documents["document_id"].tolist() == ["LEFT", "RIGHT"]
    assert len(assets) == 3


def test_cninfo_metadata_query_fails_when_one_day_exceeds_page_limit(
    tmp_path: Path,
) -> None:
    session = RouterSession(cninfo_pages=[_cninfo_page([], page_count=101)])

    with pytest.raises(ValueError, match="one-day result exceeds"):
        query_annual_report_metadata(
            _transport(session),
            _store(tmp_path),
            start_date="2020-01-01",
            end_date="2020-01-01",
        )

    assert session.calls[0][2]["data"]["pageNum"] == "1"


def test_cninfo_metadata_query_uses_total_count_to_recover_final_page(
    tmp_path: Path,
) -> None:
    first = _announcement(
        "FIRST",
        "000001",
        "2019年年度报告",
        1577844000000,
        "/first.PDF",
    )
    second = _announcement(
        "SECOND",
        "600000",
        "2019年年度报告",
        1577847600000,
        "/second.PDF",
    )
    session = RouterSession(
        cninfo_pages=[
            _cninfo_page([first], page_count=1, total_count=2),
            _cninfo_page([second], page_count=1, total_count=2),
        ],
    )

    documents, _ = query_annual_report_metadata(
        _transport(session),
        _store(tmp_path),
        start_date="2020-01-01",
        end_date="2020-01-01",
        page_size=1,
    )

    assert documents["document_id"].tolist() == ["FIRST", "SECOND"]
    assert [call[2]["data"]["pageNum"] for call in session.calls] == ["1", "2"]


def test_cninfo_metadata_query_rejects_empty_expected_page(tmp_path: Path) -> None:
    first = _announcement(
        "FIRST",
        "000001",
        "2019年年度报告",
        1577808000000,
        "/first.PDF",
    )
    session = RouterSession(
        cninfo_pages=[
            _cninfo_page([first], page_count=2, total_count=2),
            _cninfo_page([], page_count=2, total_count=2),
        ],
    )

    with pytest.raises(ValueError, match="empty page"):
        query_annual_report_metadata(
            _transport(session),
            _store(tmp_path),
            start_date="2020-01-01",
            end_date="2020-01-01",
            page_size=1,
        )


def test_cninfo_paging_dedup_actual_timestamp_download_and_cache(
    tmp_path: Path,
) -> None:
    first_url = "https://static.cninfo.com.cn/finalpage/2021/first.PDF"
    second_url = "https://static.cninfo.com.cn/finalpage/2022/second.PDF"
    page_one = _cninfo_page(
        [
            _announcement(
                "D1",
                "000001",
                "<em>2020年年度报告</em>",
                1619827200000,
                "/finalpage/2021/first.PDF",
            ),
            _announcement(
                "ABSTRACT",
                "000001",
                "2020年年度报告摘要",
                1619740800000,
                "/abstract.PDF",
            ),
        ],
        page_count=2,
        total_count=4,
    )
    page_two = _cninfo_page(
        [
            _announcement(
                "D1",
                "000001",
                "2020年年度报告（修订版）",
                1619827200000,
                "/finalpage/2021/first.PDF",
            ),
            _announcement(
                "D2",
                "600000",
                "2021年年度报告",
                1651276800000,
                "/finalpage/2022/second.PDF",
            ),
        ],
        page_count=2,
    )
    session = RouterSession(
        cninfo_pages=[page_one, page_two],
        documents={first_url: b"%PDF-1.7\nfirst", second_url: b"%PDF-1.7\nsecond"},
    )
    store = _store(tmp_path)
    result = acquire_cninfo_disclosures(
        _transport(session),
        store,
        start_date="2022-01-01",
        end_date="2022-12-31",
        page_size=2,
        clock=lambda: FIXED_NOW,
    )

    assert result.documents["document_id"].tolist() == ["D1", "D2"]
    assert result.documents.loc[0, "adjunct_url"] == first_url
    assert result.documents.loc[0, "announcement_time_ms"] == 1619827200000
    assert result.documents["publication_datetime"].str.endswith("+08:00").all()
    assert result.documents["download_status"].eq("DOWNLOADED").all()
    assert result.documents["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert disclosure_acquisition_diagnostics(result)["qa_state"] == "PASS"
    result.documents.loc[result.documents["document_id"].eq("D2"), "cninfo_org_id"] = (
        None
    )
    missing_org_diagnostics = disclosure_acquisition_diagnostics(result)
    assert missing_org_diagnostics["qa_state"] == "PASS"
    assert missing_org_diagnostics["missing_org_id_count"] == 1

    cached_session = RouterSession()
    cached = acquire_cninfo_disclosures(
        _transport(cached_session),
        store,
        start_date="2022-01-01",
        end_date="2022-12-31",
        page_size=2,
        clock=lambda: FIXED_NOW,
    )
    assert not cached_session.calls
    assert cached.documents["cache_hit"].all()


@pytest.mark.parametrize(
    "mode",
    [AcquisitionMode(metadata_only=True), AcquisitionMode(max_documents=1)],
)
def test_limited_disclosure_runs_can_never_report_production_pass(
    tmp_path: Path,
    mode: AcquisitionMode,
) -> None:
    first_url = "https://static.cninfo.com.cn/one.PDF"
    second_url = "https://static.cninfo.com.cn/two.PDF"
    page = _cninfo_page(
        [
            _announcement("D1", "000001", "2020年年度报告", 1619740800000, "/one.PDF"),
            _announcement("D2", "600000", "2021年年度报告", 1651276800000, "/two.PDF"),
        ]
    )
    documents = (
        {}
        if mode.metadata_only
        else {first_url: b"%PDF-1.7\nfirst", second_url: b"%PDF-1.7\nsecond"}
    )
    result = acquire_cninfo_disclosures(
        _transport(RouterSession(cninfo_pages=[page], documents=documents)),
        _store(tmp_path),
        start_date="2022-01-01",
        end_date="2022-12-31",
        mode=mode,
        clock=lambda: FIXED_NOW,
    )
    diagnostics = disclosure_acquisition_diagnostics(result)
    assert not diagnostics["production_qa_eligible"]
    assert diagnostics["qa_state"] == "NOT_EVALUATED_LIMITED_RUN"
    with pytest.raises(ValueError, match="cannot be marked production PASS"):
        mode.guard_qa_state("PASS")


def test_cninfo_failed_download_has_explicit_immutable_log(tmp_path: Path) -> None:
    url = "https://static.cninfo.com.cn/bad.PDF"
    page = _cninfo_page(
        [_announcement("BAD", "000001", "2020年年度报告", 1619740800000, "/bad.PDF")]
    )
    result = acquire_cninfo_disclosures(
        _transport(
            RouterSession(cninfo_pages=[page], documents={url: b"<html>blocked</html>"})
        ),
        _store(tmp_path),
        start_date="2020-01-01",
        end_date="2020-12-31",
        clock=lambda: FIXED_NOW,
    )
    diagnostics = disclosure_acquisition_diagnostics(result)
    assert diagnostics["qa_state"] == "FAIL"
    assert diagnostics["failed_download_count"] == 1
    assert diagnostics["failure_category_counts"] == {
        "SOURCE_UNAVAILABLE": 0,
        "CONTENT_OR_INTEGRITY_ERROR": 1,
    }
    assert "non-PDF content" in result.failures[0].reason
    assert (
        result.failures[0].failure_category
        == DownloadFailureCategory.CONTENT_OR_INTEGRITY_ERROR
    )
    assert result.failure_log_asset is not None
    log_path = tmp_path / result.failure_log_asset.local_path
    failure_record = json.loads(log_path.read_text(encoding="utf-8"))
    assert failure_record["document_id"] == "BAD"
    assert failure_record["reason"]
    assert failure_record["failure_category"] == "CONTENT_OR_INTEGRITY_ERROR"


def test_cninfo_request_exception_is_classified_as_source_unavailable(
    tmp_path: Path,
) -> None:
    url = "https://static.cninfo.com.cn/unavailable.PDF"
    page = _cninfo_page(
        [
            _announcement(
                "UNAVAILABLE",
                "000001",
                "2020年年度报告",
                1619740800000,
                "/unavailable.PDF",
            )
        ]
    )

    class UnavailableDocumentSession(RouterSession):
        def request(self, method: str, request_url: str, **kwargs: Any) -> FakeResponse:
            if request_url == url:
                self.calls.append((method, request_url, kwargs))
                raise requests.ConnectionError("CNINFO document endpoint unavailable")
            return super().request(method, request_url, **kwargs)

    result = acquire_cninfo_disclosures(
        _transport(UnavailableDocumentSession(cninfo_pages=[page])),
        _store(tmp_path),
        start_date="2020-01-01",
        end_date="2020-12-31",
        clock=lambda: FIXED_NOW,
    )

    assert len(result.failures) == 1
    assert (
        result.failures[0].failure_category
        == DownloadFailureCategory.SOURCE_UNAVAILABLE
    )
    diagnostics = disclosure_acquisition_diagnostics(result)
    assert diagnostics["failure_category_counts"] == {
        "SOURCE_UNAVAILABLE": 1,
        "CONTENT_OR_INTEGRITY_ERROR": 0,
    }
    assert result.failure_log_asset is not None
    log_path = tmp_path / result.failure_log_asset.local_path
    failure_record = json.loads(log_path.read_text(encoding="utf-8"))
    assert failure_record["failure_category"] == "SOURCE_UNAVAILABLE"
