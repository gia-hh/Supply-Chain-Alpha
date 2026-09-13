from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from supply_chain_alpha.data.schemas import EQUITY_DAILY, validate_table

_KEY = ["date", "security_id"]
_REQUIRED_PRICE_COLUMNS = {
    "date",
    "security_id",
    "open",
    "close",
    "volume",
    "amount",
}
_REQUIRED_MASTER_COLUMNS = {
    "security_id",
    "listing_date",
    "delisting_date",
    "board",
}
_REQUIRED_RULE_COLUMNS = {"valid_from", "valid_to", "board", "ordinary_pct"}


def _as_market_dates(values: pd.Series, *, label: str) -> pd.Series:
    """Return timezone-naive normalized market dates without accepting nulls."""
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any():
        raise ValueError(f"{label} contains null dates")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    normalized = parsed.dt.normalize()
    if not normalized.eq(parsed).all():
        raise ValueError(f"{label} must contain dates, not intraday timestamps")
    return normalized


def _as_optional_market_dates(values: pd.Series, *, label: str) -> pd.Series:
    """Normalize nullable market dates while applying the same date-only rule."""
    parsed = pd.to_datetime(values, errors="raise")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    normalized = parsed.dt.normalize()
    invalid_time = parsed.notna() & ~normalized.eq(parsed)
    if invalid_time.any():
        raise ValueError(f"{label} must contain dates, not intraday timestamps")
    return normalized


def _require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _normalise_ids(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    result = frame.copy()
    if result["security_id"].isna().any():
        raise ValueError(f"{label}.security_id contains nulls")
    result["security_id"] = result["security_id"].astype(str)
    if result["security_id"].str.strip().eq("").any():
        raise ValueError(f"{label}.security_id contains empty values")
    return result


def _validate_numeric_prices(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.copy()
    numeric = [
        column
        for column in (
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "market_cap",
            "free_float_market_cap",
        )
        if column in result.columns
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        finite = result[column].dropna().map(np.isfinite)
        if not finite.all():
            raise ValueError(f"prices.{column} contains non-finite values")
    for column in ("open", "close", "high", "low"):
        if column in result and (result[column].dropna() <= 0).any():
            raise ValueError(f"prices.{column} must be positive when present")
    for column in ("volume", "amount", "market_cap", "free_float_market_cap"):
        if column in result and (result[column].dropna() < 0).any():
            raise ValueError(f"prices.{column} must be non-negative")
    return result


def _coerce_bool_flag(values: pd.Series, *, label: str) -> pd.Series:
    nonnull = values.dropna()
    if not nonnull.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{label} must contain booleans")
    return values.astype("boolean")


def _prepare_calendar(
    trading_calendar: Sequence[Any] | pd.DataFrame,
) -> pd.DatetimeIndex:
    if isinstance(trading_calendar, pd.DataFrame):
        _require_columns(trading_calendar, {"date"}, label="trading_calendar")
        raw_dates = trading_calendar["date"]
    elif isinstance(trading_calendar, (str, bytes)):
        raise TypeError("trading_calendar must be a sequence of dates")
    else:
        raw_dates = pd.Series(list(trading_calendar), dtype="object")
    dates = _as_market_dates(raw_dates, label="trading_calendar")
    if dates.duplicated().any():
        raise ValueError("trading_calendar contains duplicate dates")
    if dates.empty:
        raise ValueError("trading_calendar must not be empty")
    return pd.DatetimeIndex(dates.sort_values(kind="stable"), name="date")


def _prepare_master(security_master: pd.DataFrame) -> pd.DataFrame:
    _require_columns(security_master, _REQUIRED_MASTER_COLUMNS, label="security_master")
    master = _normalise_ids(security_master, label="security_master")
    if master["security_id"].duplicated().any():
        raise ValueError("security_master contains duplicate security_id values")
    master["listing_date"] = _as_market_dates(
        master["listing_date"], label="security_master.listing_date"
    )
    master["delisting_date"] = _as_optional_market_dates(
        master["delisting_date"], label="security_master.delisting_date"
    )
    bad_interval = master["delisting_date"].notna() & (
        master["listing_date"] >= master["delisting_date"]
    )
    if bad_interval.any():
        sample = master.loc[
            bad_interval, ["security_id", "listing_date", "delisting_date"]
        ].head(5)
        raise ValueError(
            "security_master listing intervals must be non-empty and half-open; "
            f"sample={sample.to_dict('records')}"
        )
    if master["board"].isna().any():
        raise ValueError("security_master.board contains nulls")
    master["board"] = master["board"].astype(str).str.strip().str.upper()
    if master["board"].eq("").any():
        raise ValueError("security_master.board contains empty values")
    return master


def _prepare_rules(
    price_limit_rules: pd.DataFrame | Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    rules = (
        price_limit_rules.copy()
        if isinstance(price_limit_rules, pd.DataFrame)
        else pd.DataFrame(list(price_limit_rules))
    )
    _require_columns(rules, _REQUIRED_RULE_COLUMNS, label="price_limit_rules")
    if rules.empty:
        raise ValueError("price_limit_rules must not be empty")
    rules["valid_from"] = _as_market_dates(
        rules["valid_from"], label="price_limit_rules.valid_from"
    )
    rules["valid_to"] = _as_market_dates(
        rules["valid_to"], label="price_limit_rules.valid_to"
    )
    if (rules["valid_from"] > rules["valid_to"]).any():
        raise ValueError("price_limit_rules valid_from must be <= valid_to")
    rules["board"] = rules["board"].astype(str).str.strip().str.upper()
    if rules["board"].eq("").any():
        raise ValueError("price_limit_rules.board contains empty values")
    for column in ("ordinary_pct", "st_pct"):
        if column not in rules:
            rules[column] = np.nan
        rules[column] = pd.to_numeric(rules[column], errors="raise")
        invalid = rules[column].notna() & ~rules[column].between(
            0.0, 1.0, inclusive="neither"
        )
        if invalid.any():
            raise ValueError(f"price_limit_rules.{column} must be in (0, 1)")
    if "ipo_no_limit_trading_days" not in rules:
        rules["ipo_no_limit_trading_days"] = 0
    rules["ipo_no_limit_trading_days"] = pd.to_numeric(
        rules["ipo_no_limit_trading_days"], errors="raise"
    ).fillna(0)
    invalid_days = (rules["ipo_no_limit_trading_days"] < 0) | (
        rules["ipo_no_limit_trading_days"] % 1 != 0
    )
    if invalid_days.any():
        raise ValueError(
            "price_limit_rules.ipo_no_limit_trading_days must be non-negative integers"
        )

    ordered = rules.sort_values(["board", "valid_from", "valid_to"], kind="stable")
    for _, group in ordered.groupby("board", sort=False):
        prior_end = group["valid_to"].shift()
        if (group["valid_from"] <= prior_end).fillna(False).any():
            raise ValueError("price_limit_rules contains overlapping board intervals")
    return ordered.reset_index(drop=True)


def _prepare_actions(corporate_actions: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "date",
        "security_id",
        "cash_dividend",
        "share_multiplier",
        "reference_price",
    ]
    if corporate_actions is None or corporate_actions.empty:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "security_id": pd.Series(dtype="string"),
                "cash_dividend": pd.Series(dtype="float64"),
                "share_multiplier": pd.Series(dtype="float64"),
                "reference_price": pd.Series(dtype="float64"),
            }
        )[columns]
    actions = corporate_actions.copy()
    if "date" not in actions and "ex_date" in actions:
        actions = actions.rename(columns={"ex_date": "date"})
    _require_columns(actions, {"date", "security_id"}, label="corporate_actions")
    actions = _normalise_ids(actions, label="corporate_actions")
    actions["date"] = _as_market_dates(actions["date"], label="corporate_actions.date")
    if actions.duplicated(_KEY).any():
        raise ValueError("corporate_actions contains duplicate date/security_id values")

    aliases = {
        "cash_dividend_per_share": "cash_dividend",
        "cash_dividend_per_pre_share": "cash_dividend",
        "split_factor": "share_multiplier",
    }
    for source, target in aliases.items():
        if source in actions:
            if target in actions:
                raise ValueError(
                    f"corporate_actions cannot contain both {source} and {target}"
                )
            actions = actions.rename(columns={source: target})
    if "cash_dividend" not in actions:
        actions["cash_dividend"] = 0.0
    if "share_multiplier" not in actions:
        actions["share_multiplier"] = 1.0
    if "reference_price" not in actions:
        actions["reference_price"] = np.nan
    for column in ("cash_dividend", "share_multiplier", "reference_price"):
        actions[column] = pd.to_numeric(actions[column], errors="raise").astype(float)
        if not actions[column].dropna().map(np.isfinite).all():
            raise ValueError(f"corporate_actions.{column} contains non-finite values")
    if (actions["cash_dividend"] < 0).any():
        raise ValueError("corporate_actions.cash_dividend must be non-negative")
    if (actions["share_multiplier"] <= 0).any():
        raise ValueError("corporate_actions.share_multiplier must be positive")
    if (actions["reference_price"].dropna() <= 0).any():
        raise ValueError("corporate_actions.reference_price must be positive")
    return actions[columns]


def _assign_price_limit_rules(panel: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["_rule_matched"] = False
    result["_ordinary_pct"] = np.nan
    result["_st_pct"] = np.nan
    result["_ipo_no_limit_days"] = 0
    for rule in rules.itertuples(index=False):
        mask = result["board"].eq(rule.board) & result["date"].between(
            rule.valid_from, rule.valid_to, inclusive="both"
        )
        if (mask & result["_rule_matched"]).any():
            raise ValueError("multiple price-limit rules match one security-day")
        result.loc[mask, "_rule_matched"] = True
        result.loc[mask, "_ordinary_pct"] = rule.ordinary_pct
        result.loc[mask, "_st_pct"] = rule.st_pct
        result.loc[mask, "_ipo_no_limit_days"] = int(rule.ipo_no_limit_trading_days)
    return result


def _round_price(values: pd.Series) -> pd.Series:
    """Round positive A-share prices to cents using exchange-style half-up."""
    return np.floor(values.astype(float) * 100.0 + 0.5 + 1e-12) / 100.0


def build_market_panel(
    prices: pd.DataFrame,
    security_master: pd.DataFrame,
    trading_calendar: Sequence[Any] | pd.DataFrame,
    price_limit_rules: pd.DataFrame | Iterable[Mapping[str, Any]],
    corporate_actions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a deterministic, point-in-time daily market panel.

    Listing and delisting dates form a half-open interval.  Returns use only the
    immediately preceding supplied trading date; missing prices are never
    forward/backward filled.  Explicit corporate actions affect their ex-date
    return only, so appending a later action cannot rewrite earlier returns.

    ``cash_dividend`` is cash per pre-action share and ``share_multiplier`` is
    post-action shares per pre-action share.  A source-provided ex-rights
    ``reference_price`` may be supplied for exact price-limit classification.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if not isinstance(security_master, pd.DataFrame):
        raise TypeError("security_master must be a pandas DataFrame")
    _require_columns(prices, _REQUIRED_PRICE_COLUMNS, label="prices")
    calendar = _prepare_calendar(trading_calendar)
    master = _prepare_master(security_master)
    rules = _prepare_rules(price_limit_rules)
    actions = _prepare_actions(corporate_actions)

    observed = _normalise_ids(prices, label="prices")
    observed["date"] = _as_market_dates(observed["date"], label="prices.date")
    observed = _validate_numeric_prices(observed)
    if observed.duplicated(_KEY).any():
        raise ValueError("prices contains duplicate date/security_id values")
    unknown = sorted(set(observed["security_id"]).difference(master["security_id"]))
    if unknown:
        raise ValueError(
            f"prices contains securities absent from master: {unknown[:5]}"
        )
    outside_calendar = ~observed["date"].isin(calendar)
    if outside_calendar.any():
        sample = observed.loc[outside_calendar, _KEY].head(5).to_dict("records")
        raise ValueError(f"prices contains non-trading-calendar dates: {sample}")

    interval_check = observed[_KEY].merge(
        master[["security_id", "listing_date", "delisting_date"]],
        on="security_id",
        how="left",
        validate="many_to_one",
    )
    outside_listing = (interval_check["date"] < interval_check["listing_date"]) | (
        interval_check["delisting_date"].notna()
        & (interval_check["date"] >= interval_check["delisting_date"])
    )
    if outside_listing.any():
        sample = interval_check.loc[outside_listing, _KEY].head(5).to_dict("records")
        raise ValueError(
            f"prices contains observations outside listing intervals: {sample}"
        )

    grid = pd.MultiIndex.from_product(
        [calendar, master["security_id"].sort_values(kind="stable")], names=_KEY
    ).to_frame(index=False)
    grid = grid.merge(
        master[["security_id", "listing_date", "delisting_date", "board"]],
        on="security_id",
        how="left",
        validate="many_to_one",
    )
    listed = (grid["date"] >= grid["listing_date"]) & (
        grid["delisting_date"].isna() | (grid["date"] < grid["delisting_date"])
    )
    grid = grid.loc[listed].copy()
    panel = grid.merge(observed, on=_KEY, how="left", validate="one_to_one")
    panel = panel.sort_values(["security_id", "date"], kind="stable").reset_index(
        drop=True
    )

    panel["listing_age"] = panel.groupby("security_id", sort=False).cumcount()
    panel["listing_age_trading_days"] = panel["listing_age"]
    panel["has_market_data"] = (
        panel[["open", "close", "volume", "amount"]].notna().all(axis=1)
    )

    provided_flags: list[pd.Series] = []
    for column in ("suspended", "is_suspended"):
        if column in panel:
            provided_flags.append(
                _coerce_bool_flag(panel[column], label=f"prices.{column}")
            )
    if len(provided_flags) == 2:
        conflicts = (
            provided_flags[0].notna()
            & provided_flags[1].notna()
            & provided_flags[0].ne(provided_flags[1])
        )
        if conflicts.any():
            raise ValueError("prices suspended and is_suspended flags conflict")
    explicit_suspended = (
        provided_flags[0]
        if provided_flags
        else pd.Series(pd.NA, index=panel.index, dtype="boolean")
    )
    if len(provided_flags) == 2:
        explicit_suspended = explicit_suspended.fillna(provided_flags[1])
    inferred_suspended = (
        panel["has_market_data"] & panel["volume"].eq(0) & panel["amount"].eq(0)
    )
    panel["is_suspended"] = explicit_suspended.fillna(inferred_suspended).astype(bool)
    panel["suspended"] = panel["is_suspended"]

    panel["previous_close"] = panel.groupby("security_id", sort=False)["close"].shift()
    if not actions.empty:
        unknown_actions = sorted(
            set(actions["security_id"]).difference(master["security_id"])
        )
        if unknown_actions:
            raise ValueError(
                "corporate_actions contains securities absent from master: "
                f"{unknown_actions[:5]}"
            )
        action_calendar_error = ~actions["date"].isin(calendar)
        if action_calendar_error.any():
            raise ValueError("corporate_actions contains non-trading-calendar dates")
        action_intervals = actions[_KEY].merge(
            master[["security_id", "listing_date", "delisting_date"]],
            on="security_id",
            how="left",
            validate="many_to_one",
        )
        action_outside_listing = (
            action_intervals["date"] < action_intervals["listing_date"]
        ) | (
            action_intervals["delisting_date"].notna()
            & (action_intervals["date"] >= action_intervals["delisting_date"])
        )
        if action_outside_listing.any():
            raise ValueError(
                "corporate_actions contains dates outside listing intervals"
            )
    panel = panel.merge(actions, on=_KEY, how="left", validate="one_to_one")
    panel["cash_dividend"] = panel["cash_dividend"].fillna(0.0)
    panel["share_multiplier"] = panel["share_multiplier"].fillna(1.0)
    action_present = panel["cash_dividend"].ne(0) | panel["share_multiplier"].ne(1)
    panel["corporate_action"] = action_present

    derived_reference = (panel["previous_close"] - panel["cash_dividend"]) / panel[
        "share_multiplier"
    ]
    panel["price_limit_reference"] = panel["reference_price"].fillna(derived_reference)
    invalid_reference = panel["price_limit_reference"].notna() & (
        panel["price_limit_reference"] <= 0
    )
    if invalid_reference.any():
        raise ValueError("corporate action implies a non-positive ex-rights reference")

    gross_value = panel["close"] * panel["share_multiplier"] + panel["cash_dividend"]
    return_valid = (
        panel["close"].notna()
        & panel["previous_close"].notna()
        & ~panel["is_suspended"]
    )
    panel["raw_return"] = (gross_value / panel["previous_close"] - 1.0).where(
        return_valid
    )

    panel = _assign_price_limit_rules(panel, rules)
    st_source = (
        panel["st_status"].astype("string")
        if "st_status" in panel
        else pd.Series(pd.NA, index=panel.index, dtype="string")
    )
    is_st = st_source.fillna("").str.upper().str.contains("ST", regex=False)
    missing_st_limit = is_st & panel["_rule_matched"] & panel["_st_pct"].isna()
    if missing_st_limit.any():
        raise ValueError("an ST security-day matched a rule without st_pct")
    panel["price_limit_pct"] = panel["_ordinary_pct"].where(~is_st, panel["_st_pct"])
    ipo_no_limit = panel["_rule_matched"] & (
        panel["listing_age"] < panel["_ipo_no_limit_days"]
    )
    classifiable = (
        panel["has_market_data"]
        & panel["previous_close"].notna()
        & ~panel["is_suspended"]
    )
    missing_rule = classifiable & ~panel["_rule_matched"]
    if missing_rule.any():
        sample = panel.loc[missing_rule, ["date", "security_id", "board"]].head(5)
        raise ValueError(
            "no historical price-limit rule for classifiable security-days; "
            f"sample={sample.to_dict('records')}"
        )

    limited = classifiable & ~ipo_no_limit
    panel["limit_up_price"] = _round_price(
        panel["price_limit_reference"] * (1.0 + panel["price_limit_pct"])
    ).where(limited)
    panel["limit_down_price"] = _round_price(
        panel["price_limit_reference"] * (1.0 - panel["price_limit_pct"])
    ).where(limited)
    panel["limit_up"] = limited & np.isclose(
        panel["close"], panel["limit_up_price"], rtol=0.0, atol=1e-9
    )
    panel["limit_down"] = limited & np.isclose(
        panel["close"], panel["limit_down_price"], rtol=0.0, atol=1e-9
    )
    above_limit = limited & (panel["close"] > panel["limit_up_price"] + 1e-9)
    below_limit = limited & (panel["close"] < panel["limit_down_price"] - 1e-9)
    if (above_limit | below_limit).any():
        sample = panel.loc[
            above_limit | below_limit,
            ["date", "security_id", "close", "limit_up_price", "limit_down_price"],
        ].head(5)
        raise ValueError(
            "observed close violates the applicable historical price limit; "
            f"sample={sample.to_dict('records')}"
        )

    panel["price_limit_state"] = "unavailable"
    panel.loc[classifiable & ipo_no_limit, "price_limit_state"] = "no_limit"
    panel.loc[limited, "price_limit_state"] = "normal"
    panel.loc[panel["limit_up"], "price_limit_state"] = "limit_up"
    panel.loc[panel["limit_down"], "price_limit_state"] = "limit_down"
    panel.loc[panel["is_suspended"], "price_limit_state"] = "suspended"

    close_locked = panel["limit_up"] | panel["limit_down"]
    close_locked &= np.isclose(panel["open"], panel["close"], rtol=0.0, atol=1e-9)
    if "high" in panel and "low" in panel:
        close_locked &= np.isclose(
            panel["high"], panel["close"], rtol=0.0, atol=1e-9
        ) & np.isclose(panel["low"], panel["close"], rtol=0.0, atol=1e-9)
    panel["price_limit_locked"] = close_locked.fillna(False).astype(bool)
    panel["is_tradable"] = (
        panel["has_market_data"]
        & ~panel["is_suspended"]
        & ~panel["price_limit_locked"]
        & panel["volume"].gt(0)
        & panel["amount"].gt(0)
    ).astype(bool)
    panel["tradable"] = panel["is_tradable"]

    output_columns = [
        "date",
        "security_id",
        "open",
        "close",
        "volume",
        "amount",
        "tradable",
    ]
    output_columns += [
        column
        for column in (
            "high",
            "low",
            "market_cap",
            "free_float_market_cap",
            "suspended",
            "limit_up",
            "limit_down",
            "st_status",
        )
        if column in panel and column not in output_columns
    ]
    output_columns += [
        "is_tradable",
        "is_suspended",
        "price_limit_state",
        "price_limit_locked",
        "listing_age",
        "listing_age_trading_days",
        "has_market_data",
        "previous_close",
        "price_limit_reference",
        "price_limit_pct",
        "limit_up_price",
        "limit_down_price",
        "corporate_action",
        "cash_dividend",
        "share_multiplier",
        "raw_return",
        "board",
        "listing_date",
        "delisting_date",
    ]
    output = (
        panel[output_columns].sort_values(_KEY, kind="stable").reset_index(drop=True)
    )
    validate_table(output, EQUITY_DAILY)
    return output
