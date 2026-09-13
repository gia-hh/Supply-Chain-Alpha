"""Small fail-closed boundaries for official-source network locations."""

from __future__ import annotations

from collections.abc import Collection
from urllib.parse import SplitResult, urlsplit


def require_official_https_url(
    value: object,
    *,
    label: str,
    allowed_hosts: Collection[str],
    expected_path: str | None = None,
) -> str:
    """Return *value* only when it is an unambiguous approved HTTPS URL.

    Exact host matching deliberately rejects look-alike subdomains, user-info
    tricks, non-default ports, and network locations supplied through relative
    URL syntax.  Callers may additionally freeze an endpoint path while still
    supplying request parameters separately.
    """

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical URL")
    try:
        parsed: SplitResult = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid URL") from exc

    approved = {host.lower() for host in allowed_hosts}
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname is None
        or parsed.hostname.lower() not in approved
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.fragment
    ):
        raise ValueError(
            f"{label} must use HTTPS on an explicitly approved official host"
        )
    if expected_path is not None and (parsed.path != expected_path or parsed.query):
        raise ValueError(f"{label} must use the frozen official endpoint path")
    return value


__all__ = ["require_official_https_url"]
