"""Deterministic pre-holdout signal evaluation primitives."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

import pandas as pd


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def daily_information_coefficients(
    signals: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    horizon: int = 1,
    signal_column: str = "customer_shock",
) -> pd.DataFrame:
    """Compute daily rank IC against an actual future trading-date return.

    The forward date is selected from the global observed trading calendar,
    never by adding calendar days and never by skipping a target's missing row.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    _require_columns(signals, {"date", "security_id", signal_column}, "signals")
    _require_columns(residuals, {"date", "security_id", "residual_return"}, "residuals")
    signal_frame = signals.loc[:, ["date", "security_id", signal_column]].copy()
    return_frame = residuals.loc[:, ["date", "security_id", "residual_return"]].copy()
    for frame, label in ((signal_frame, "signals"), (return_frame, "residuals")):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date.astype(
            str
        )
        if frame.duplicated(["date", "security_id"]).any():
            raise ValueError(f"{label} contain duplicate date/security_id rows")

    calendar = sorted(return_frame["date"].unique().tolist())
    future_date = {
        day: calendar[position + horizon]
        for position, day in enumerate(calendar)
        if position + horizon < len(calendar)
    }
    signal_frame["forward_return_date"] = signal_frame["date"].map(future_date)
    future_returns = return_frame.rename(
        columns={
            "date": "forward_return_date",
            "residual_return": "forward_residual_return",
        }
    )
    aligned = signal_frame.merge(
        future_returns,
        on=["forward_return_date", "security_id"],
        how="left",
        validate="many_to_one",
    )
    aligned[signal_column] = pd.to_numeric(aligned[signal_column], errors="coerce")
    aligned["forward_residual_return"] = pd.to_numeric(
        aligned["forward_residual_return"], errors="coerce"
    )

    rows: list[dict[str, object]] = []
    for day, group in aligned.groupby("date", sort=True):
        usable = group.dropna(
            subset=[signal_column, "forward_residual_return", "forward_return_date"]
        )
        if (
            len(usable) < 2
            or usable[signal_column].nunique() < 2
            or usable["forward_residual_return"].nunique() < 2
        ):
            continue
        forward_dates = usable["forward_return_date"].unique().tolist()
        if len(forward_dates) != 1:
            raise ValueError("one signal date mapped to multiple forward trading dates")
        rows.append(
            {
                "date": str(day),
                "horizon": horizon,
                "forward_return_date": forward_dates[0],
                "eligible_count": len(usable),
                "rank_ic": float(
                    usable[signal_column]
                    .rank(method="average")
                    .corr(usable["forward_residual_return"].rank(method="average"))
                ),
                "pearson_ic": float(
                    usable[signal_column].corr(usable["forward_residual_return"])
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "date",
            "horizon",
            "forward_return_date",
            "eligible_count",
            "rank_ic",
            "pearson_ic",
        ),
    )


def moving_block_bootstrap_mean(
    values: Iterable[float],
    *,
    replications: int,
    block_length: int,
    seed: int,
) -> pd.Series:
    """Return deterministic moving-block bootstrap means."""

    sample = [float(value) for value in values if math.isfinite(float(value))]
    if not sample:
        raise ValueError("bootstrap requires at least one finite value")
    if isinstance(replications, bool) or replications < 1:
        raise ValueError("replications must be positive")
    if isinstance(block_length, bool) or not 1 <= block_length <= len(sample):
        raise ValueError("block_length must be between 1 and sample length")
    starts = list(range(len(sample) - block_length + 1))
    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(replications):
        draw: list[float] = []
        while len(draw) < len(sample):
            start = generator.choice(starts)
            draw.extend(sample[start : start + block_length])
        means.append(sum(draw[: len(sample)]) / len(sample))
    return pd.Series(means, name="bootstrap_mean", dtype=float)


def newey_west_mean_standard_error(values: Iterable[float], *, lags: int) -> float:
    """Compute a Bartlett-kernel Newey-West standard error for a sample mean."""

    sample = [float(value) for value in values if math.isfinite(float(value))]
    if len(sample) < 2:
        raise ValueError("Newey-West uncertainty requires at least two observations")
    if (
        isinstance(lags, bool)
        or not isinstance(lags, int)
        or not 0 <= lags < len(sample)
    ):
        raise ValueError("lags must be an integer below the sample length")
    mean = sum(sample) / len(sample)
    centered = [value - mean for value in sample]
    long_run_variance = sum(value * value for value in centered) / len(sample)
    for lag in range(1, lags + 1):
        covariance = sum(
            centered[index] * centered[index - lag] for index in range(lag, len(sample))
        ) / len(sample)
        long_run_variance += 2 * (1 - lag / (lags + 1)) * covariance
    return math.sqrt(max(long_run_variance, 0.0) / len(sample))


def degree_preserving_random_graph(
    edges: pd.DataFrame,
    *,
    seed: int,
    max_attempts: int = 1_000,
) -> pd.DataFrame:
    """Randomize customer endpoints while preserving both directed degree multisets."""

    _require_columns(edges, {"supplier_id", "customer_id"}, "edges")
    pairs = edges.loc[:, ["supplier_id", "customer_id"]].astype(str)
    if pairs.duplicated().any() or pairs["supplier_id"].eq(pairs["customer_id"]).any():
        raise ValueError(
            "placebo input must be a simple directed graph without self-edges"
        )
    original = set(pairs.itertuples(index=False, name=None))
    suppliers = pairs["supplier_id"].tolist()
    customers = pairs["customer_id"].tolist()
    generator = random.Random(seed)
    for _ in range(max_attempts):
        shuffled = customers.copy()
        generator.shuffle(shuffled)
        candidate_pairs = list(zip(suppliers, shuffled, strict=True))
        candidate_set = set(candidate_pairs)
        if (
            len(candidate_set) == len(candidate_pairs)
            and all(supplier != customer for supplier, customer in candidate_pairs)
            and candidate_set != original
        ):
            return (
                pd.DataFrame(candidate_pairs, columns=["supplier_id", "customer_id"])
                .sort_values(["supplier_id", "customer_id"], kind="mergesort")
                .reset_index(drop=True)
            )
    raise ValueError("could not construct a distinct degree-preserving placebo graph")


__all__ = [
    "daily_information_coefficients",
    "degree_preserving_random_graph",
    "moving_block_bootstrap_mean",
    "newey_west_mean_standard_error",
]
