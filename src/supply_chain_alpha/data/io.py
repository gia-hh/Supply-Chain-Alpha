from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_canonical_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a canonical table to Parquet; fail loudly if the engine is unavailable."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Canonical outputs require Parquet support. Install project dependencies (pyarrow)."
        ) from exc


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    if suffix == ".json":
        return pd.read_json(p)
    if suffix == ".parquet":
        try:
            return pd.read_parquet(p)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet support requires pyarrow or fastparquet"
            ) from exc
    raise ValueError(f"Unsupported table format: {suffix}")
