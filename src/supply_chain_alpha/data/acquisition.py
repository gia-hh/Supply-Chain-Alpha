"""Shared primitives for auditable, respectful official-source acquisition."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import pandas as pd
import requests

from supply_chain_alpha.utils.hashing import (
    atomic_write_json,
    canonical_json_sha256,
    sha256_bytes,
    sha256_file,
)

RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
LOGGER = logging.getLogger(__name__)
_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_MAX_RANGED_ASSET_BYTES = 2 * 1024**3
RAW_ASSET_REQUIRED_FIELDS = frozenset(
    {
        "source",
        "source_url_or_identifier",
        "retrieval_datetime",
        "file_size",
        "sha256",
    }
)


class ResponseLike(Protocol):
    status_code: int
    headers: Mapping[str, str]
    content: bytes

    def raise_for_status(self) -> None: ...

    def close(self) -> None: ...


class SessionLike(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> ResponseLike: ...


@dataclass(frozen=True)
class RequestPolicy:
    """Deterministic request pacing and bounded retry policy."""

    user_agent: str = "supply-chain-alpha-research/0.2"
    timeout_seconds: float = 30.0
    max_attempts: int = 4
    backoff_seconds: float = 2.0
    min_interval_seconds: float = 0.75

    def __post_init__(self) -> None:
        if not self.user_agent.strip():
            raise ValueError("user_agent must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if self.backoff_seconds < 0 or self.min_interval_seconds < 0:
            raise ValueError("Request delays must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> RequestPolicy:
        """Construct a policy from the ``data_sources.request_policy`` config."""

        return cls(**dict(value or {}))


class RateLimitedSession:
    """A requests-compatible transport with global spacing and retries."""

    def __init__(
        self,
        policy: RequestPolicy | None = None,
        *,
        session: SessionLike | None = None,
        session_factory: Callable[[], SessionLike] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if session is not None and session_factory is not None:
            raise ValueError("session and session_factory are mutually exclusive")
        self.policy = policy or RequestPolicy()
        self._session = session
        self._session_factory = session_factory or requests.Session
        self._thread_local = threading.local()
        self._created_sessions: list[SessionLike] = []
        self._created_sessions_lock = threading.Lock()
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_started: float | None = None
        self._not_before = 0.0
        self._pace_lock = threading.Lock()

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelledError("Request cancelled before the next attempt")

    def _pace(self, cancel_event: threading.Event | None = None) -> float:
        """Reserve one globally spaced attempt start across all worker threads."""

        while True:
            with self._pace_lock:
                self._raise_if_cancelled(cancel_event)
                now = self._monotonic()
                earliest = self._not_before
                if self._last_request_started is not None:
                    earliest = max(
                        earliest,
                        self._last_request_started + self.policy.min_interval_seconds,
                    )
                remaining = earliest - now
                if remaining <= 0:
                    self._last_request_started = now
                    return now

            # Wait outside the lock so a Retry-After received by another worker
            # can extend the shared cooldown before this attempt begins.
            if cancel_event is None:
                self._sleep(remaining)
            elif cancel_event.wait(remaining):
                raise RequestCancelledError(
                    "Request cancelled while waiting for the next attempt"
                )

    def _defer_all_requests(self, delay: float) -> None:
        """Apply retry backoff or Retry-After as a shared source cooldown."""

        with self._pace_lock:
            self._not_before = max(self._not_before, self._monotonic() + delay)

    def _session_for_current_thread(self) -> SessionLike:
        if self._session is not None:
            return self._session
        current = getattr(self._thread_local, "session", None)
        if current is None:
            current = self._session_factory()
            self._thread_local.session = current
            with self._created_sessions_lock:
                self._created_sessions.append(current)
        return current

    def close_thread_sessions(self) -> None:
        """Close factory-created sessions after all worker threads have stopped."""

        with self._created_sessions_lock:
            sessions, self._created_sessions = self._created_sessions, []
        for session in sessions:
            close = getattr(session, "close", None)
            if callable(close):
                close()
        if hasattr(self._thread_local, "session"):
            del self._thread_local.session

    @staticmethod
    def _retry_after(response: ResponseLike) -> float:
        raw_value = response.headers.get("Retry-After")
        if raw_value is None:
            return 0.0
        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError):
            return 0.0

    def request(
        self,
        method: str,
        url: str,
        *,
        cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> ResponseLike:
        """Issue one logical request, retrying only transient failures."""

        headers = {"User-Agent": self.policy.user_agent}
        headers.update(dict(kwargs.pop("headers", {}) or {}))
        timeout = kwargs.pop("timeout", self.policy.timeout_seconds)
        if kwargs.pop("allow_redirects", False) is not False:
            raise ValueError(
                "Cross-host redirects cannot be enabled for raw acquisition"
            )
        # Requests follows redirects by default.  Raw provenance must identify
        # the exact approved origin we contacted, so never let a 3xx silently
        # escape the call-site URL allowlist.
        kwargs["allow_redirects"] = False
        last_error: requests.RequestException | None = None

        for attempt in range(self.policy.max_attempts):
            self._pace(cancel_event)
            try:
                response = self._session_for_current_thread().request(
                    method.upper(),
                    url,
                    headers=headers,
                    timeout=timeout,
                    **kwargs,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 == self.policy.max_attempts:
                    raise
                self._raise_if_cancelled(cancel_event)
                retry_delay = self.policy.backoff_seconds * (2**attempt)
                LOGGER.warning(
                    "Retrying request after %s attempt=%d/%d delay_seconds=%.3f",
                    type(exc).__name__,
                    attempt + 1,
                    self.policy.max_attempts,
                    retry_delay,
                )
                self._defer_all_requests(retry_delay)
                continue

            if 300 <= response.status_code < 400:
                response.close()
                raise requests.TooManyRedirects(
                    f"Official-source redirect is not permitted: {url}",
                    response=response,
                )
            if response.status_code not in RETRYABLE_HTTP_STATUSES:
                response.raise_for_status()
                return response
            if attempt + 1 == self.policy.max_attempts:
                response.raise_for_status()
                return response  # pragma: no cover - raise_for_status normally raises

            retry_delay = max(
                self.policy.backoff_seconds * (2**attempt),
                self._retry_after(response),
            )
            LOGGER.warning(
                "Retrying request after HTTP %d attempt=%d/%d delay_seconds=%.3f",
                response.status_code,
                attempt + 1,
                self.policy.max_attempts,
                retry_delay,
            )
            response.close()
            self._defer_all_requests(retry_delay)

        if last_error is not None:  # pragma: no cover - loop always returns/raises
            raise last_error
        raise RuntimeError("Request retry loop ended unexpectedly")  # pragma: no cover

    def get(self, url: str, **kwargs: Any) -> ResponseLike:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> ResponseLike:
        return self.request("POST", url, **kwargs)


class RequestCancelledError(RuntimeError):
    """Raised inside workers when a coordinator cancels pending retries."""


@dataclass(frozen=True)
class RawAsset:
    """Required provenance for one immutable cached response."""

    source: str
    source_url_or_identifier: str
    retrieval_datetime: str
    file_size: int
    sha256: str
    local_path: str
    cache_hit: bool = False

    def metadata(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("cache_hit")
        return record

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RawAssetIntegrityError(ValueError):
    """Raised rather than mutating an existing raw cache asset."""


class RawAssetRecoveryRequired(RawAssetIntegrityError):
    """Raised when an incomplete raw pair needs a legitimate refetch to recover."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalise_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("retrieval_datetime must include a UTC offset")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_create_bytes(path: Path, content: bytes) -> bool:
    """Create *path* atomically without ever replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(content)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


class RawAssetStore:
    """Append-only content cache with immutable provenance sidecars."""

    def __init__(
        self, root: str | Path, *, clock: Callable[[], datetime] = utc_now
    ) -> None:
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError(f"Raw asset root cannot be a symlink: {self.root}")
        self._resolved_root = self.root.resolve(strict=False)
        self._clock = clock

    @staticmethod
    def _relative_path(value: str | Path) -> PurePosixPath:
        raw = str(value).replace("\\", "/")
        relative = PurePosixPath(raw)
        if (
            not raw
            or relative.is_absolute()
            or ".." in relative.parts
            or ":" in relative.parts[0]
        ):
            raise ValueError(f"Raw asset path must be safely project-relative: {value}")
        return relative

    def _paths(self, relative_path: str | Path) -> tuple[PurePosixPath, Path, Path]:
        relative = self._relative_path(relative_path)
        asset_path = self.root.joinpath(*relative.parts)
        sidecar_path = asset_path.with_name(f"{asset_path.name}.metadata.json")
        if (
            self.root.is_symlink()
            or self.root.resolve(strict=False) != self._resolved_root
        ):
            raise ValueError(f"Raw asset root changed or became a link: {self.root}")
        for candidate in (asset_path, sidecar_path):
            cursor = self.root
            for part in candidate.relative_to(self.root).parts:
                cursor /= part
                if cursor.is_symlink():
                    raise ValueError(
                        f"Raw asset path cannot traverse a link: {candidate}"
                    )
            try:
                # Resolve the parent; resolving a concurrently-created hard-linked
                # file can produce a Windows ``\\?\`` spelling that does not compare
                # lexically with the same directory's ordinary spelling.
                candidate.parent.resolve(strict=False).relative_to(self._resolved_root)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Raw asset path escapes its configured root: {candidate}"
                ) from exc
        return relative, asset_path, sidecar_path

    @staticmethod
    def _read_sidecar(
        relative: PurePosixPath,
        sidecar_path: Path,
        *,
        cache_hit: bool,
    ) -> RawAsset:
        try:
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RawAssetIntegrityError(
                f"Invalid raw metadata sidecar: {sidecar_path}"
            ) from exc
        expected_fields = RAW_ASSET_REQUIRED_FIELDS | {"local_path"}
        if not isinstance(metadata, dict) or set(metadata) != expected_fields:
            raise RawAssetIntegrityError(
                f"Raw metadata has an invalid schema: {sidecar_path}"
            )
        if metadata["local_path"] != relative.as_posix():
            raise RawAssetIntegrityError(
                f"Raw metadata local_path mismatch: {sidecar_path}"
            )
        asset = RawAsset(**metadata, cache_hit=cache_hit)
        _validate_manifest_asset(asset, sidecar_path=sidecar_path)
        return asset

    @staticmethod
    def _assert_asset_identity(
        existing: RawAsset,
        *,
        relative: PurePosixPath,
        content: bytes,
        digest: str,
        source: str,
        source_url_or_identifier: str,
    ) -> None:
        if (
            existing.file_size != len(content)
            or existing.sha256 != digest
            or existing.source != source
            or existing.source_url_or_identifier != source_url_or_identifier
        ):
            raise RawAssetIntegrityError(
                f"Append-only raw asset collision: {relative.as_posix()}"
            )

    def _recover_sidecar_only(
        self,
        *,
        relative: PurePosixPath,
        asset_path: Path,
        sidecar_path: Path,
        content: bytes,
        digest: str,
        source: str,
        source_url_or_identifier: str,
    ) -> RawAsset:
        reserved = self._read_sidecar(relative, sidecar_path, cache_hit=True)
        self._assert_asset_identity(
            reserved,
            relative=relative,
            content=content,
            digest=digest,
            source=source,
            source_url_or_identifier=source_url_or_identifier,
        )
        _atomic_create_bytes(asset_path, content)
        recovered = self.load(relative)
        if recovered is None:  # pragma: no cover - sidecar is known to exist
            raise RawAssetIntegrityError(
                f"Raw cache recovery produced no entry: {relative.as_posix()}"
            )
        self._assert_asset_identity(
            recovered,
            relative=relative,
            content=content,
            digest=digest,
            source=source,
            source_url_or_identifier=source_url_or_identifier,
        )
        return recovered

    def load(self, relative_path: str | Path) -> RawAsset | None:
        relative, asset_path, sidecar_path = self._paths(relative_path)
        asset_exists = asset_path.exists()
        sidecar_exists = sidecar_path.exists()
        if not asset_exists and not sidecar_exists:
            return None
        if sidecar_exists and not asset_exists:
            raise RawAssetRecoveryRequired(
                "Raw cache provenance is reserved but the asset must be refetched: "
                f"{relative.as_posix()}"
            )
        if asset_exists and not sidecar_exists:
            raise RawAssetRecoveryRequired(
                "Raw cache asset exists without provenance and must be refetched: "
                f"{relative.as_posix()}"
            )
        asset = self._read_sidecar(relative, sidecar_path, cache_hit=True)
        if asset_path.stat().st_size != asset.file_size:
            raise RawAssetIntegrityError(f"Raw asset size mismatch: {asset_path}")
        if sha256_file(asset_path) != asset.sha256:
            raise RawAssetIntegrityError(f"Raw asset SHA-256 mismatch: {asset_path}")
        return asset

    def load_bytes(self, relative_path: str | Path) -> tuple[RawAsset, bytes] | None:
        """Load and verify cached bytes in one filesystem pass.

        Hashing the bytes that are returned avoids a second verification read
        followed by a third materialisation read in acquisition cache-hit paths.
        """

        relative, asset_path, sidecar_path = self._paths(relative_path)
        asset_exists = asset_path.exists()
        sidecar_exists = sidecar_path.exists()
        if not asset_exists and not sidecar_exists:
            return None
        if sidecar_exists and not asset_exists:
            raise RawAssetRecoveryRequired(
                "Raw cache provenance is reserved but the asset must be refetched: "
                f"{relative.as_posix()}"
            )
        if asset_exists and not sidecar_exists:
            raise RawAssetRecoveryRequired(
                "Raw cache asset exists without provenance and must be refetched: "
                f"{relative.as_posix()}"
            )
        asset = self._read_sidecar(relative, sidecar_path, cache_hit=True)
        content = asset_path.read_bytes()
        if len(content) != asset.file_size:
            raise RawAssetIntegrityError(f"Raw asset size mismatch: {asset_path}")
        if sha256_bytes(content) != asset.sha256:
            raise RawAssetIntegrityError(f"Raw asset SHA-256 mismatch: {asset_path}")
        return asset, content

    def read_bytes(self, asset: RawAsset) -> bytes:
        cached = self.load(asset.local_path)
        if cached is None:
            raise FileNotFoundError(f"Cached raw asset is missing: {asset.local_path}")
        return self.root.joinpath(*PurePosixPath(asset.local_path).parts).read_bytes()

    def store_bytes(
        self,
        relative_path: str | Path,
        content: bytes,
        *,
        source: str,
        source_url_or_identifier: str,
        retrieval_datetime: datetime | None = None,
    ) -> RawAsset:
        """Cache bytes once; a same-path mutation is a hard integrity error."""

        if not isinstance(content, bytes):
            raise TypeError("Raw asset content must be bytes")
        if (
            not isinstance(source, str)
            or not isinstance(source_url_or_identifier, str)
            or not source.strip()
            or not source_url_or_identifier.strip()
        ):
            raise ValueError(
                "Raw asset source and source_url_or_identifier must be non-empty"
            )
        relative, asset_path, sidecar_path = self._paths(relative_path)
        digest = sha256_bytes(content)
        try:
            existing = self.load(relative)
        except RawAssetRecoveryRequired:
            if sidecar_path.exists() and not asset_path.exists():
                return self._recover_sidecar_only(
                    relative=relative,
                    asset_path=asset_path,
                    sidecar_path=sidecar_path,
                    content=content,
                    digest=digest,
                    source=source,
                    source_url_or_identifier=source_url_or_identifier,
                )
            if asset_path.is_file() and sidecar_path.is_file():
                concurrent = self.load(relative)
                if concurrent is None:  # pragma: no cover - both paths exist
                    raise RawAssetIntegrityError(
                        f"Raw cache recovery produced no entry: {relative.as_posix()}"
                    )
                self._assert_asset_identity(
                    concurrent,
                    relative=relative,
                    content=content,
                    digest=digest,
                    source=source,
                    source_url_or_identifier=source_url_or_identifier,
                )
                return concurrent
            if not asset_path.is_file() or sidecar_path.exists():
                raise RawAssetIntegrityError(
                    f"Raw cache recovery state changed concurrently: {relative.as_posix()}"
                )
            if (
                asset_path.stat().st_size != len(content)
                or sha256_file(asset_path) != digest
            ):
                raise RawAssetIntegrityError(
                    f"Append-only raw asset collision: {relative.as_posix()}"
                )
            existing = None
        if existing is not None:
            self._assert_asset_identity(
                existing,
                relative=relative,
                content=content,
                digest=digest,
                source=source,
                source_url_or_identifier=source_url_or_identifier,
            )
            return existing

        retrieved_at = _normalise_timestamp(retrieval_datetime or self._clock())
        asset = RawAsset(
            source=source,
            source_url_or_identifier=source_url_or_identifier,
            retrieval_datetime=retrieved_at,
            file_size=len(content),
            sha256=digest,
            local_path=relative.as_posix(),
            cache_hit=False,
        )
        metadata_bytes = (
            json.dumps(
                asset.metadata(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        # Reserve the path with complete immutable provenance first.  If the
        # process stops before the asset link is created, a later legitimate
        # refetch can verify against this sidecar and safely finish the pair.
        if not _atomic_create_bytes(sidecar_path, metadata_bytes):
            return self._recover_sidecar_only(
                relative=relative,
                asset_path=asset_path,
                sidecar_path=sidecar_path,
                content=content,
                digest=digest,
                source=source,
                source_url_or_identifier=source_url_or_identifier,
            )
        if not _atomic_create_bytes(asset_path, content):
            concurrent = self.load(relative)
            if concurrent is None:  # pragma: no cover - sidecar was just created
                raise RawAssetIntegrityError(
                    f"Concurrent raw asset creation failed: {relative.as_posix()}"
                )
            self._assert_asset_identity(
                concurrent,
                relative=relative,
                content=content,
                digest=digest,
                source=source,
                source_url_or_identifier=source_url_or_identifier,
            )
            return concurrent
        return asset


def _validate_manifest_asset(asset: RawAsset, *, sidecar_path: Path) -> None:
    """Reject malformed sidecar provenance before it reaches the raw manifest."""

    if not isinstance(asset.source, str) or not asset.source.strip():
        raise RawAssetIntegrityError(
            f"Raw metadata has an invalid source: {sidecar_path}"
        )
    if (
        not isinstance(asset.source_url_or_identifier, str)
        or not asset.source_url_or_identifier.strip()
    ):
        raise RawAssetIntegrityError(
            f"Raw metadata has an invalid source_url_or_identifier: {sidecar_path}"
        )
    if (
        isinstance(asset.file_size, bool)
        or not isinstance(asset.file_size, int)
        or asset.file_size < 0
    ):
        raise RawAssetIntegrityError(
            f"Raw metadata has an invalid file_size: {sidecar_path}"
        )
    if (
        not isinstance(asset.sha256, str)
        or len(asset.sha256) != 64
        or any(character not in "0123456789abcdef" for character in asset.sha256)
    ):
        raise RawAssetIntegrityError(
            f"Raw metadata has an invalid sha256: {sidecar_path}"
        )
    if not isinstance(asset.retrieval_datetime, str):
        raise RawAssetIntegrityError(
            f"Raw metadata has an invalid retrieval_datetime: {sidecar_path}"
        )
    try:
        retrieved_at = datetime.fromisoformat(
            asset.retrieval_datetime.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise RawAssetIntegrityError(
            f"Raw metadata has an invalid retrieval_datetime: {sidecar_path}"
        ) from exc
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise RawAssetIntegrityError(
            f"Raw metadata retrieval_datetime has no UTC offset: {sidecar_path}"
        )


def write_raw_asset_manifest(
    raw_root: str | Path,
    output_path: str | Path | None = None,
    *,
    verify_asset_bytes: bool = True,
) -> Path:
    """Atomically inventory every append-only asset and provenance sidecar.

    Manifest paths are POSIX paths relative to the manifest's parent directory.
    Consequently, a store rooted at ``data/raw/official_sources`` can write to
    ``data/raw/MANIFEST.json`` and produce paths beginning with
    ``official_sources/``.  The store root must equal or be nested below that
    parent so the manifest never contains machine-specific absolute paths.

    Every file under ``raw_root`` (apart from the output manifest itself) must
    be part of a valid :class:`RawAssetStore` asset/sidecar pair.  Untracked or
    malformed files are a hard integrity failure rather than receiving
    invented provenance.  ``verify_asset_bytes=False`` is reserved for callers
    that immediately run :func:`verify_raw_manifest`: it avoids hashing large
    assets twice while retaining pair, schema, path, size, sidecar-hash, and
    exact-inventory checks.  The safe standalone default remains a full byte
    verification.
    """

    if not isinstance(verify_asset_bytes, bool):
        raise TypeError("verify_asset_bytes must be a bool")

    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Raw asset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Raw asset root is not a directory: {root}")
    if root.is_symlink():
        raise RawAssetIntegrityError(f"Raw asset root cannot be a symlink: {root}")

    root = root.resolve(strict=True)
    output = (root / "MANIFEST.json") if output_path is None else Path(output_path)
    output = output.resolve()
    manifest_parent = output.parent.resolve()
    try:
        root.relative_to(manifest_parent)
    except ValueError as exc:
        raise ValueError(
            "raw_root must equal or be nested below the manifest parent directory"
        ) from exc
    if output.exists() and not output.is_file():
        raise ValueError(f"Raw manifest output is not a regular file: {output}")

    files: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise RawAssetIntegrityError(
                f"Raw asset manifest does not permit symlinks: {candidate}"
            )
        if candidate.is_file() and candidate.resolve() != output:
            files.append(candidate.resolve(strict=True))
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    file_set = set(files)

    metadata_fields = RAW_ASSET_REQUIRED_FIELDS | {"local_path"}
    sidecars: list[tuple[Path, str]] = []
    for candidate in files:
        if not candidate.name.endswith(".metadata.json"):
            continue
        try:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(metadata, dict) or set(metadata) != metadata_fields:
            continue
        local_path = metadata.get("local_path")
        if not isinstance(local_path, str):
            raise RawAssetIntegrityError(
                f"Raw metadata has an invalid local_path: {candidate}"
            )
        try:
            relative = RawAssetStore._relative_path(local_path)
        except (TypeError, ValueError) as exc:
            raise RawAssetIntegrityError(
                f"Raw metadata has an invalid local_path: {candidate}"
            ) from exc
        asset_path = root.joinpath(*relative.parts).resolve()
        expected_sidecar = asset_path.with_name(f"{asset_path.name}.metadata.json")
        if expected_sidecar != candidate:
            raise RawAssetIntegrityError(
                f"Raw metadata local_path mismatch: {candidate}"
            )
        if asset_path not in file_set:
            raise RawAssetIntegrityError(
                f"Incomplete raw cache entry (asset/sidecar mismatch): {relative.as_posix()}"
            )
        sidecars.append((candidate, relative.as_posix()))

    store = RawAssetStore(root)
    records: list[dict[str, Any]] = []
    covered: set[Path] = set()
    for sidecar_path, local_path in sidecars:
        if verify_asset_bytes:
            asset = store.load(local_path)
            if asset is None:  # pragma: no cover - existence was checked above
                raise RawAssetIntegrityError(
                    f"Cached raw asset is missing: {local_path}"
                )
        else:
            relative = PurePosixPath(local_path)
            asset = store._read_sidecar(relative, sidecar_path, cache_hit=True)
        _validate_manifest_asset(asset, sidecar_path=sidecar_path)
        asset_path = root.joinpath(*PurePosixPath(local_path).parts).resolve(
            strict=True
        )
        if asset_path.stat().st_size != asset.file_size:
            raise RawAssetIntegrityError(f"Raw asset size mismatch: {asset_path}")
        shared_provenance = {
            "source": asset.source,
            "source_url_or_identifier": asset.source_url_or_identifier,
            "retrieval_datetime": asset.retrieval_datetime,
        }
        records.extend(
            [
                {
                    "path": asset_path.relative_to(manifest_parent).as_posix(),
                    **shared_provenance,
                    "file_size": asset.file_size,
                    "sha256": asset.sha256,
                },
                {
                    "path": sidecar_path.relative_to(manifest_parent).as_posix(),
                    **shared_provenance,
                    "file_size": sidecar_path.stat().st_size,
                    "sha256": sha256_file(sidecar_path),
                },
            ]
        )
        covered.update({asset_path, sidecar_path})

    untracked = sorted(path.relative_to(root).as_posix() for path in file_set - covered)
    if untracked:
        raise RawAssetIntegrityError(
            "Raw asset files lack valid provenance sidecars: " + ", ".join(untracked)
        )

    records.sort(key=lambda record: record["path"])
    atomic_write_json(output, records)
    return output


def request_fingerprint(
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
) -> str:
    """Canonical request identity used in immutable cache filenames."""

    return canonical_json_sha256(
        {
            "method": method.upper(),
            "url": url,
            "params": dict(params or {}),
            "data": dict(data or {}),
        }
    )


def _fetch_ranged_content(
    transport: RateLimitedSession,
    *,
    method: str,
    url: str,
    params: Mapping[str, Any] | None,
    data: Mapping[str, Any] | None,
    headers: Mapping[str, str] | None,
    chunk_bytes: int,
    cancel_event: threading.Event | None,
) -> bytes:
    """Recover one GET response through strict, contiguous byte ranges."""

    if method.upper() != "GET":
        raise ValueError("Range fallback is supported only for GET requests")
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int):
        raise TypeError("range_fallback_chunk_bytes must be an integer")
    if chunk_bytes < 1:
        raise ValueError("range_fallback_chunk_bytes must be positive")

    pieces: list[bytes] = []
    next_start = 0
    expected_total: int | None = None
    while expected_total is None or next_start < expected_total:
        requested_end = next_start + chunk_bytes - 1
        range_headers = dict(headers or {})
        # A range fallback is used only after the ordinary cached response has
        # repeatedly terminated early.  Revalidate against the origin/cache
        # path rather than requesting the same known-bad CDN object fragment.
        range_headers["Cache-Control"] = "no-cache"
        range_headers["Pragma"] = "no-cache"
        range_headers["Range"] = f"bytes={next_start}-{requested_end}"
        response = transport.request(
            method,
            url,
            params=dict(params or {}),
            data=dict(data or {}),
            headers=range_headers,
            cancel_event=cancel_event,
        )
        try:
            if response.status_code != 206:
                raise ValueError(
                    "Range fallback requires HTTP 206 Partial Content; "
                    f"received {response.status_code}"
                )
            raw_content_range = response.headers.get("Content-Range", "")
            match = _CONTENT_RANGE.fullmatch(str(raw_content_range).strip())
            if match is None:
                raise ValueError(
                    "Range fallback response has invalid Content-Range: "
                    f"{raw_content_range!r}"
                )
            returned_start, returned_end, returned_total = map(int, match.groups())
            if returned_total < 1 or returned_total > _MAX_RANGED_ASSET_BYTES:
                raise ValueError(
                    f"Range fallback response has unsafe total size: {returned_total}"
                )
            if returned_start != next_start:
                raise ValueError(
                    "Range fallback response is not contiguous: "
                    f"expected_start={next_start}, returned_start={returned_start}"
                )
            if not returned_start <= returned_end < returned_total:
                raise ValueError(
                    "Range fallback response has invalid byte bounds: "
                    f"{raw_content_range!r}"
                )
            if returned_end > requested_end:
                raise ValueError(
                    "Range fallback response exceeded the requested range: "
                    f"requested_end={requested_end}, returned_end={returned_end}"
                )
            if expected_total is None:
                expected_total = returned_total
            elif returned_total != expected_total:
                raise ValueError(
                    "Range fallback total size changed between responses: "
                    f"expected={expected_total}, returned={returned_total}"
                )

            content = bytes(response.content)
            expected_chunk_size = returned_end - returned_start + 1
            if len(content) != expected_chunk_size:
                raise ValueError(
                    "Range fallback response byte count disagrees with Content-Range: "
                    f"expected={expected_chunk_size}, received={len(content)}"
                )
            pieces.append(content)
            next_start = returned_end + 1
        finally:
            response.close()

    combined = b"".join(pieces)
    if expected_total is None or len(combined) != expected_total:
        raise ValueError(
            "Range fallback did not reconstruct the declared response length: "
            f"expected={expected_total}, received={len(combined)}"
        )
    return combined


def fetch_raw_asset(
    transport: RateLimitedSession,
    store: RawAssetStore,
    relative_path: str | Path,
    *,
    source: str,
    method: str,
    url: str,
    params: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    validate_content: Callable[[bytes], None] | None = None,
    cancel_event: threading.Event | None = None,
    range_fallback_chunk_bytes: int | None = None,
) -> tuple[RawAsset, bytes]:
    """Return a cached response or request, validate, and append it once."""

    fingerprint = request_fingerprint(method, url, params=params, data=data)
    identifier = f"{url}#request_sha256={fingerprint}"
    try:
        cached_payload = store.load_bytes(relative_path)
    except RawAssetRecoveryRequired:
        LOGGER.warning(
            "Refetching raw asset to complete a provenance-reserved cache entry: %s",
            relative_path,
        )
        cached_payload = None
    if cached_payload is not None:
        cached, content = cached_payload
        if cached.source != source or cached.source_url_or_identifier != identifier:
            raise RawAssetIntegrityError(
                f"Cached request identity mismatch for raw asset: {cached.local_path}"
            )
        if validate_content is not None:
            validate_content(content)
        return cached, content

    try:
        response = transport.request(
            method,
            url,
            params=dict(params or {}),
            data=dict(data or {}),
            headers=dict(headers or {}),
            cancel_event=cancel_event,
        )
        try:
            content = bytes(response.content)
        finally:
            response.close()
    except requests.exceptions.ChunkedEncodingError:
        if range_fallback_chunk_bytes is None:
            raise
        LOGGER.warning(
            "Falling back to strict byte-range acquisition chunk_bytes=%d url=%s",
            range_fallback_chunk_bytes,
            url,
        )
        content = _fetch_ranged_content(
            transport,
            method=method,
            url=url,
            params=params,
            data=data,
            headers=headers,
            chunk_bytes=range_fallback_chunk_bytes,
            cancel_event=cancel_event,
        )
    if validate_content is not None:
        validate_content(content)
    asset = store.store_bytes(
        relative_path,
        content,
        source=source,
        source_url_or_identifier=identifier,
    )
    return asset, content


@dataclass(frozen=True)
class AcquisitionMode:
    """Trial controls that can never be presented as production QA."""

    metadata_only: bool = False
    max_documents: int | None = None

    def __post_init__(self) -> None:
        if self.max_documents is not None and (
            isinstance(self.max_documents, bool) or self.max_documents < 1
        ):
            raise ValueError("max_documents must be a positive integer or null")

    @property
    def production_qa_eligible(self) -> bool:
        return not self.metadata_only and self.max_documents is None

    @property
    def label(self) -> str:
        return "PRODUCTION" if self.production_qa_eligible else "LIMITED_TRIAL"

    def guard_qa_state(self, state: str) -> str:
        if not self.production_qa_eligible and state == "PASS":
            raise ValueError(
                "A metadata-only/max-documents run cannot be marked production PASS"
            )
        return state


def provenance_columns(asset: RawAsset, *, prefix: str = "") -> dict[str, Any]:
    """Flatten required raw provenance for a tabular acquisition result."""

    return {
        f"{prefix}source": asset.source,
        f"{prefix}source_url_or_identifier": asset.source_url_or_identifier,
        f"{prefix}retrieval_datetime": asset.retrieval_datetime,
        f"{prefix}file_size": asset.file_size,
        f"{prefix}sha256": asset.sha256,
        f"{prefix}raw_local_path": asset.local_path,
        f"{prefix}cache_hit": asset.cache_hit,
    }


def security_master_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute the Phase-2 security-master acquisition checks."""

    rows = len(frame)

    def column(name: str) -> pd.Series:
        if name in frame:
            return frame[name]
        return pd.Series([None] * rows, index=frame.index, dtype=object)

    duplicate_count = (
        int(frame.duplicated("security_id", keep=False).sum()) if rows else 0
    )
    listing = pd.to_datetime(column("listing_date"), errors="coerce")
    delisting = pd.to_datetime(column("delisting_date"), errors="coerce")
    invalid_interval = int(
        (listing.notna() & delisting.notna() & (listing >= delisting)).sum()
    )
    ticker_non_string = int(
        sum(not isinstance(value, str) for value in column("ticker"))
    )
    invalid_exchange = int((~column("exchange").isin({"SSE", "SZSE"})).sum())
    provenance_fields = [
        "source",
        "source_url_or_identifier",
        "retrieval_datetime",
        "file_size",
        "sha256",
    ]
    missing_provenance = rows
    if rows and all(field in frame for field in provenance_fields):
        invalid_hash = ~frame["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}")
        missing_provenance = int(
            (frame[provenance_fields].isna().any(axis=1) | invalid_hash).sum()
        )
    listing_rate = float(listing.notna().mean()) if rows else 0.0
    passed = bool(
        rows
        and duplicate_count == 0
        and invalid_interval == 0
        and ticker_non_string == 0
        and invalid_exchange == 0
        and missing_provenance == 0
        and listing_rate >= 0.95
    )
    return {
        "rows": rows,
        "duplicate_security_id_count": duplicate_count,
        "invalid_listing_delisting_count": invalid_interval,
        "ticker_non_string_count": ticker_non_string,
        "invalid_exchange_count": invalid_exchange,
        "missing_provenance_count": missing_provenance,
        "listing_date_non_null_rate": listing_rate,
        "listing_date_min": listing.min().date().isoformat()
        if listing.notna().any()
        else None,
        "listing_date_max": listing.max().date().isoformat()
        if listing.notna().any()
        else None,
        "sources": sorted(frame["source"].dropna().astype(str).unique().tolist())
        if "source" in frame
        else [],
        "qa_state": "PASS" if passed else "FAIL",
    }


def assert_security_master_qa(frame: pd.DataFrame) -> dict[str, Any]:
    diagnostics = security_master_diagnostics(frame)
    if diagnostics["qa_state"] != "PASS":
        raise ValueError(f"Security-master acquisition QA failed: {diagnostics}")
    return diagnostics


__all__ = [
    "RAW_ASSET_REQUIRED_FIELDS",
    "RETRYABLE_HTTP_STATUSES",
    "AcquisitionMode",
    "RateLimitedSession",
    "RawAsset",
    "RawAssetIntegrityError",
    "RawAssetRecoveryRequired",
    "RawAssetStore",
    "RequestCancelledError",
    "RequestPolicy",
    "assert_security_master_qa",
    "fetch_raw_asset",
    "provenance_columns",
    "request_fingerprint",
    "security_master_diagnostics",
    "write_raw_asset_manifest",
]
