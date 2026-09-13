"""Research-freeze fingerprint checks guarding final-holdout access."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from supply_chain_alpha.utils.hashing import atomic_write_json

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = {
    "frozen_at",
    "source_tree_sha256",
    "config_sha256",
    "experiment_registry_sha256",
    "portfolio_candidate",
}


class HoldoutIntegrityError(RuntimeError):
    """Raised before any holdout data may be read."""


def _require_digest(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def write_research_freeze(
    path: str | Path,
    *,
    frozen_at: str,
    source_tree_sha256: str,
    config_sha256: str,
    experiment_registry_sha256: str,
    portfolio_candidate: bool,
) -> dict[str, Any]:
    """Write the immutable decision fingerprint required before holdout access."""

    timestamp = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("frozen_at must include a timezone offset")
    if not isinstance(portfolio_candidate, bool):
        raise TypeError("portfolio_candidate must be boolean")
    payload: dict[str, Any] = {
        "frozen_at": timestamp.isoformat(),
        "source_tree_sha256": _require_digest(source_tree_sha256, "source_tree_sha256"),
        "config_sha256": _require_digest(config_sha256, "config_sha256"),
        "experiment_registry_sha256": _require_digest(
            experiment_registry_sha256, "experiment_registry_sha256"
        ),
        "portfolio_candidate": portfolio_candidate,
    }
    atomic_write_json(Path(path), payload)
    return payload


def authorize_holdout_access(
    freeze_path: str | Path,
    *,
    source_tree_sha256: str,
    config_sha256: str,
    experiment_registry_sha256: str,
) -> dict[str, Any]:
    """Authorize access only when every current fingerprint matches the freeze."""

    path = Path(freeze_path)
    if not path.is_file():
        raise HoldoutIntegrityError("holdout access denied before research freeze")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutIntegrityError("research freeze is unreadable or invalid") from exc
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise HoldoutIntegrityError("research freeze schema is invalid")

    current = {
        "source_tree_sha256": _require_digest(source_tree_sha256, "source_tree_sha256"),
        "config_sha256": _require_digest(config_sha256, "config_sha256"),
        "experiment_registry_sha256": _require_digest(
            experiment_registry_sha256, "experiment_registry_sha256"
        ),
    }
    drifted = [name for name, value in current.items() if payload.get(name) != value]
    if drifted:
        raise HoldoutIntegrityError(
            f"holdout integrity violation: post-freeze drift in {sorted(drifted)}"
        )
    try:
        timestamp = datetime.fromisoformat(
            str(payload["frozen_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise HoldoutIntegrityError("research freeze timestamp is invalid") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise HoldoutIntegrityError("research freeze timestamp lacks timezone")
    if not isinstance(payload["portfolio_candidate"], bool):
        raise HoldoutIntegrityError("research freeze decision is invalid")
    return payload


__all__ = [
    "HoldoutIntegrityError",
    "authorize_holdout_access",
    "write_research_freeze",
]
