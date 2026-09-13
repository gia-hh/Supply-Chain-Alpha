from __future__ import annotations

import numpy as np
import pandas as pd

from supply_chain_alpha.data.schemas import RETURNS_DAILY, validate_table

_KEY = ["date", "security_id"]


def _require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _market_dates(values: pd.Series, *, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any():
        raise ValueError(f"{label} contains null dates")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    normalized = parsed.dt.normalize()
    if not normalized.eq(parsed).all():
        raise ValueError(f"{label} must contain dates, not intraday timestamps")
    return normalized


def _optional_market_dates(values: pd.Series, *, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    normalized = parsed.dt.normalize()
    if (parsed.notna() & ~normalized.eq(parsed)).any():
        raise ValueError(f"{label} must contain dates, not intraday timestamps")
    return normalized


def _normalise_key(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = _market_dates(result["date"], label=f"{label}.date")
    if result["security_id"].isna().any():
        raise ValueError(f"{label}.security_id contains nulls")
    result["security_id"] = result["security_id"].astype(str)
    if result["security_id"].str.strip().eq("").any():
        raise ValueError(f"{label}.security_id contains empty values")
    if result.duplicated(_KEY).any():
        raise ValueError(f"{label} contains duplicate date/security_id values")
    return result


def _finite_numeric(values: pd.Series, *, label: str) -> pd.Series:
    result = pd.to_numeric(values, errors="raise").astype(float)
    if not result.dropna().map(np.isfinite).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def _prepare_industry_history(industry_history: pd.DataFrame) -> pd.DataFrame:
    history = industry_history.copy()
    aliases = {
        "industry_valid_from": "valid_from",
        "industry_valid_to": "valid_to",
    }
    for source, target in aliases.items():
        if source in history and target not in history:
            history = history.rename(columns={source: target})
    _require_columns(
        history,
        {"security_id", "industry_code", "valid_from", "valid_to"},
        label="industry_history",
    )
    if history["security_id"].isna().any() or history["industry_code"].isna().any():
        raise ValueError("industry_history identifiers must be non-null")
    history["security_id"] = history["security_id"].astype(str)
    history["industry_code"] = history["industry_code"].astype(str)
    if (
        history["security_id"].str.strip().eq("").any()
        or history["industry_code"].str.strip().eq("").any()
    ):
        raise ValueError("industry_history identifiers must be non-empty")
    history["valid_from"] = _market_dates(
        history["valid_from"], label="industry_history.valid_from"
    )
    history["valid_to"] = _optional_market_dates(
        history["valid_to"], label="industry_history.valid_to"
    )
    invalid = history["valid_to"].notna() & (
        history["valid_from"] >= history["valid_to"]
    )
    if invalid.any():
        raise ValueError("industry_history intervals must be non-empty and half-open")
    ordered = history.sort_values(
        ["security_id", "valid_from", "valid_to"], kind="stable", na_position="last"
    )
    for _, group in ordered.groupby("security_id", sort=False):
        prior_end = group["valid_to"].shift()
        overlap = prior_end.isna() & prior_end.index.to_series().ne(group.index[0])
        overlap |= group["valid_from"].lt(prior_end).fillna(False)
        if overlap.any():
            raise ValueError("industry_history contains overlapping intervals")
    return ordered.reset_index(drop=True)


def _attach_pit_industry(
    panel: pd.DataFrame, industry_history: pd.DataFrame
) -> pd.Series:
    assigned = pd.Series(pd.NA, index=panel.index, dtype="string")
    by_security = {
        security_id: group
        for security_id, group in panel.groupby("security_id", sort=False)
    }
    for row in industry_history.itertuples(index=False):
        group = by_security.get(row.security_id)
        if group is None:
            continue
        mask = group["date"].ge(row.valid_from)
        if pd.notna(row.valid_to):
            mask &= group["date"].lt(row.valid_to)
        target_index = group.index[mask]
        if assigned.loc[target_index].notna().any():
            raise ValueError("multiple industry histories match one security-day")
        assigned.loc[target_index] = row.industry_code
    return assigned


def compute_leave_one_out_industry_returns(
    panel: pd.DataFrame,
    industry_history: pd.DataFrame,
) -> pd.DataFrame:
    """Compute contemporaneous equal-weight industry returns excluding self."""
    _require_columns(panel, {"date", "security_id", "raw_return"}, label="panel")
    keyed = _normalise_key(panel, label="panel")
    keyed["raw_return"] = _finite_numeric(keyed["raw_return"], label="panel.raw_return")
    history = _prepare_industry_history(industry_history)
    keyed["industry_code"] = _attach_pit_industry(keyed, history)
    valid = keyed["industry_code"].notna() & keyed["raw_return"].notna()
    valid_rows = keyed.loc[valid, ["date", "industry_code", "raw_return"]].copy()
    aggregates = valid_rows.groupby(["date", "industry_code"], sort=True, dropna=False)[
        "raw_return"
    ].agg(["sum", "count"])
    keyed = keyed.join(aggregates, on=["date", "industry_code"])
    keyed["industry_peer_count"] = (keyed["count"] - 1).where(valid).astype("Int64")
    keyed["industry_return"] = (
        (keyed["sum"] - keyed["raw_return"]) / (keyed["count"] - 1)
    ).where(valid & keyed["count"].gt(1))
    return (
        keyed[
            [
                "date",
                "security_id",
                "industry_code",
                "industry_peer_count",
                "industry_return",
            ]
        ]
        .sort_values(_KEY, kind="stable")
        .reset_index(drop=True)
    )


def _rolling_sums(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    window: int,
    lag: int,
) -> pd.DataFrame:
    result: dict[str, pd.Series] = {}
    grouped = frame.groupby("security_id", sort=False)
    for column in columns:
        result[column] = grouped[column].transform(
            lambda series: series.rolling(window=window, min_periods=1).sum().shift(lag)
        )
    return pd.DataFrame(result, index=frame.index)


def _rolling_regressions(
    grid: pd.DataFrame,
    *,
    beta_window: int,
    min_observations: int,
    information_lag_days: int,
) -> pd.DataFrame:
    calc = grid.copy()
    market_valid = calc[["raw_return", "market_return"]].notna().all(axis=1)
    industry_valid = (
        calc[["raw_return", "market_return", "industry_return"]].notna().all(axis=1)
    )
    date_number = (calc["date"].astype("int64") // 86_400_000_000_000).astype(float)

    market_terms = pd.DataFrame(
        {
            "n": market_valid.astype(float),
            "y": calc["raw_return"].where(market_valid, 0.0),
            "x": calc["market_return"].where(market_valid, 0.0),
            "xx": calc["market_return"].pow(2).where(market_valid, 0.0),
            "xy": (calc["market_return"] * calc["raw_return"]).where(market_valid, 0.0),
            "date_max": date_number.where(market_valid, -np.inf),
            "date_min": date_number.where(market_valid, np.inf),
            "security_id": calc["security_id"],
        },
        index=calc.index,
    )
    market_sum = _rolling_sums(
        market_terms,
        ["n", "y", "x", "xx", "xy"],
        window=beta_window,
        lag=information_lag_days,
    )
    # Rolling extrema need min/max rather than sums and remain group-local.
    market_end = market_terms.groupby("security_id", sort=False)["date_max"].transform(
        lambda series: (
            series.rolling(beta_window, min_periods=1).max().shift(information_lag_days)
        )
    )
    market_start = market_terms.groupby("security_id", sort=False)[
        "date_min"
    ].transform(
        lambda series: (
            series.rolling(beta_window, min_periods=1).min().shift(information_lag_days)
        )
    )
    n_m = market_sum["n"]
    centered_xx = market_sum["xx"] - market_sum["x"].pow(2) / n_m
    centered_xy = market_sum["xy"] - market_sum["x"] * market_sum["y"] / n_m
    market_ok = (n_m >= min_observations) & (centered_xx.abs() > 1e-18)
    beta_m_only = (centered_xy / centered_xx).where(market_ok)
    alpha_m_only = ((market_sum["y"] - beta_m_only * market_sum["x"]) / n_m).where(
        market_ok
    )

    industry_terms = pd.DataFrame(
        {
            "n": industry_valid.astype(float),
            "y": calc["raw_return"].where(industry_valid, 0.0),
            "m": calc["market_return"].where(industry_valid, 0.0),
            "i": calc["industry_return"].where(industry_valid, 0.0),
            "mm": calc["market_return"].pow(2).where(industry_valid, 0.0),
            "ii": calc["industry_return"].pow(2).where(industry_valid, 0.0),
            "mi": (calc["market_return"] * calc["industry_return"]).where(
                industry_valid, 0.0
            ),
            "my": (calc["market_return"] * calc["raw_return"]).where(
                industry_valid, 0.0
            ),
            "iy": (calc["industry_return"] * calc["raw_return"]).where(
                industry_valid, 0.0
            ),
            "date_max": date_number.where(industry_valid, -np.inf),
            "date_min": date_number.where(industry_valid, np.inf),
            "security_id": calc["security_id"],
        },
        index=calc.index,
    )
    industry_sum = _rolling_sums(
        industry_terms,
        ["n", "y", "m", "i", "mm", "ii", "mi", "my", "iy"],
        window=beta_window,
        lag=information_lag_days,
    )
    industry_end = industry_terms.groupby("security_id", sort=False)[
        "date_max"
    ].transform(
        lambda series: (
            series.rolling(beta_window, min_periods=1).max().shift(information_lag_days)
        )
    )
    industry_start = industry_terms.groupby("security_id", sort=False)[
        "date_min"
    ].transform(
        lambda series: (
            series.rolling(beta_window, min_periods=1).min().shift(information_lag_days)
        )
    )
    n_i = industry_sum["n"]
    s_mm = industry_sum["mm"] - industry_sum["m"].pow(2) / n_i
    s_ii = industry_sum["ii"] - industry_sum["i"].pow(2) / n_i
    s_mi = industry_sum["mi"] - industry_sum["m"] * industry_sum["i"] / n_i
    c_my = industry_sum["my"] - industry_sum["m"] * industry_sum["y"] / n_i
    c_iy = industry_sum["iy"] - industry_sum["i"] * industry_sum["y"] / n_i
    determinant = s_mm * s_ii - s_mi.pow(2)
    scale = (s_mm.abs() * s_ii.abs()).clip(lower=1.0)
    industry_ok = (n_i >= min_observations) & (determinant.abs() > 1e-14 * scale)
    beta_market_industry = ((c_my * s_ii - c_iy * s_mi) / determinant).where(
        industry_ok
    )
    beta_industry = ((c_iy * s_mm - c_my * s_mi) / determinant).where(industry_ok)
    alpha_industry = (
        (
            industry_sum["y"]
            - beta_market_industry * industry_sum["m"]
            - beta_industry * industry_sum["i"]
        )
        / n_i
    ).where(industry_ok)

    use_industry = industry_ok & calc["industry_return"].notna()
    result = pd.DataFrame(index=calc.index)
    result["residual_model"] = np.where(
        use_industry, "market_plus_pit_industry", "market_only"
    )
    result["alpha"] = alpha_m_only.where(~use_industry, alpha_industry)
    result["beta_market"] = beta_m_only.where(~use_industry, beta_market_industry)
    result["beta_industry"] = beta_industry.where(use_industry)
    result["beta_observation_count"] = n_m.where(~use_industry, n_i).astype("Int64")
    chosen_end = market_end.where(~use_industry, industry_end).replace(
        [-np.inf, np.inf], np.nan
    )
    chosen_start = market_start.where(~use_industry, industry_start).replace(
        [-np.inf, np.inf], np.nan
    )
    result["beta_window_end"] = pd.to_datetime(chosen_end, unit="D", errors="coerce")
    result["beta_window_start"] = pd.to_datetime(
        chosen_start, unit="D", errors="coerce"
    )
    predicted = result["alpha"] + result["beta_market"] * calc["market_return"]
    predicted += result["beta_industry"].fillna(0.0) * calc["industry_return"].fillna(
        0.0
    )
    result["residual_return"] = (calc["raw_return"] - predicted).where(
        result["beta_market"].notna()
    )
    return result


def build_residual_returns(
    market_panel: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    beta_window: int,
    min_observations: int,
    information_lag_days: int,
    industry_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate leakage-safe rolling residual returns.

    Every regression for day ``t`` is shifted by ``information_lag_days`` and
    bounded to ``beta_window`` supplied trading dates.  No full-sample scaling
    is performed.  Industry factors are computed leave-one-out from PIT-valid
    memberships; unavailable or rank-deficient industry models fall back to
    the explicitly labelled market-only model.
    """
    if isinstance(beta_window, bool) or not isinstance(beta_window, (int, np.integer)):
        raise TypeError("beta_window must be a positive integer")
    if isinstance(min_observations, bool) or not isinstance(
        min_observations, (int, np.integer)
    ):
        raise TypeError("min_observations must be a positive integer")
    if isinstance(information_lag_days, bool) or not isinstance(
        information_lag_days, (int, np.integer)
    ):
        raise TypeError("information_lag_days must be a positive integer")
    if beta_window <= 0 or min_observations <= 0:
        raise ValueError("beta_window and min_observations must be positive")
    if min_observations > beta_window:
        raise ValueError("min_observations cannot exceed beta_window")
    if information_lag_days < 1:
        raise ValueError("information_lag_days must be at least one")

    _require_columns(
        market_panel, {"date", "security_id", "raw_return"}, label="market_panel"
    )
    panel = _normalise_key(market_panel, label="market_panel")
    panel["raw_return"] = _finite_numeric(
        panel["raw_return"], label="market_panel.raw_return"
    )
    suspension_column = next(
        (column for column in ("is_suspended", "suspended") if column in panel), None
    )
    if suspension_column is not None:
        flags = panel[suspension_column]
        invalid = flags.dropna().map(
            lambda value: not isinstance(value, (bool, np.bool_))
        )
        if invalid.any():
            raise ValueError(f"market_panel.{suspension_column} must contain booleans")
        panel.loc[flags.fillna(False).astype(bool), "raw_return"] = np.nan

    _require_columns(market_returns, {"date", "market_return"}, label="market_returns")
    factors = market_returns.copy()
    factors["date"] = _market_dates(factors["date"], label="market_returns.date")
    if factors["date"].duplicated().any():
        raise ValueError("market_returns contains duplicate dates")
    factors["market_return"] = _finite_numeric(
        factors["market_return"], label="market_returns.market_return"
    )
    factors = factors.sort_values("date", kind="stable").reset_index(drop=True)
    if factors.empty:
        raise ValueError("market_returns must not be empty")
    missing_factor_dates = sorted(set(panel["date"]).difference(factors["date"]))
    if missing_factor_dates:
        raise ValueError(
            "market_returns does not cover all panel trading dates; "
            f"sample={missing_factor_dates[:5]}"
        )

    securities = panel["security_id"].drop_duplicates().sort_values(kind="stable")
    grid = pd.MultiIndex.from_product(
        [securities, factors["date"]], names=["security_id", "date"]
    ).to_frame(index=False)
    grid = grid.merge(
        panel[_KEY + ["raw_return"]], on=_KEY, how="left", validate="one_to_one"
    )
    grid = grid.merge(factors, on="date", how="left", validate="many_to_one")
    grid = grid.sort_values(["security_id", "date"], kind="stable").reset_index(
        drop=True
    )

    if industry_history is None or industry_history.empty:
        grid["industry_code"] = pd.Series(pd.NA, index=grid.index, dtype="string")
        grid["industry_peer_count"] = pd.Series(pd.NA, index=grid.index, dtype="Int64")
        grid["industry_return"] = np.nan
    else:
        industry = compute_leave_one_out_industry_returns(grid, industry_history)
        grid = grid.merge(industry, on=_KEY, how="left", validate="one_to_one")

    regressions = _rolling_regressions(
        grid,
        beta_window=int(beta_window),
        min_observations=int(min_observations),
        information_lag_days=int(information_lag_days),
    )
    grid = pd.concat([grid, regressions], axis=1)
    original_keys = panel[_KEY]
    output = original_keys.merge(grid, on=_KEY, how="left", validate="one_to_one")
    output = (
        output[
            [
                "date",
                "security_id",
                "raw_return",
                "market_return",
                "industry_return",
                "residual_return",
                "beta_market",
                "beta_industry",
                "residual_model",
                "alpha",
                "beta_observation_count",
                "beta_window_start",
                "beta_window_end",
                "industry_code",
                "industry_peer_count",
            ]
        ]
        .sort_values(_KEY, kind="stable")
        .reset_index(drop=True)
    )
    validate_table(output, RETURNS_DAILY)
    return output
