"""Deterministic portfolio construction, execution, and transaction costs.

The functions in this module are deliberately small research primitives.  They
do not claim live executability or assume that short borrow is available.
Signals observed at a market close can first trade at the next supplied market
open, and an opened position cannot exit before the applicable A-share T+1
settlement offset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_TARGET_COLUMNS = ("signal_date", "security_id", "target_weight")
_MARKET_COLUMNS = {
    "date",
    "security_id",
    "open",
    "amount",
    "tradable",
}
_TRADE_COLUMNS = (
    "signal_date",
    "security_id",
    "target_weight",
    "side",
    "entry_date",
    "scheduled_exit_date",
    "exit_date",
    "status",
    "entry_block_reason",
    "exit_defer_count",
    "entry_price",
    "exit_price",
    "quantity",
    "entry_notional",
    "exit_notional",
    "gross_pnl",
    "entry_commission_cost",
    "exit_commission_cost",
    "commission_cost",
    "entry_slippage_cost",
    "exit_slippage_cost",
    "slippage_cost",
    "entry_stamp_tax_cost",
    "exit_stamp_tax_cost",
    "stamp_tax_cost",
    "entry_venue_fee_cost",
    "exit_venue_fee_cost",
    "venue_fee_cost",
    "total_cost",
    "net_pnl",
    "gross_return",
    "net_return",
    "holding_period_trading_days",
)


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Artifacts produced by the compact deterministic portfolio engine."""

    target_weights: pd.DataFrame
    trades: pd.DataFrame
    daily_pnl: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
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
        raise ValueError(f"{label} must contain market dates, not timestamps")
    return normalized


def _positive_number(value: Any, *, label: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric") from exc
    if not np.isfinite(numeric) or numeric < 0 or (numeric == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return numeric


def _prepare_signals(
    signals: pd.DataFrame,
    *,
    signal_column: str,
) -> pd.DataFrame:
    if not isinstance(signals, pd.DataFrame):
        raise TypeError("signals must be a pandas DataFrame")
    _require_columns(signals, {"date", "security_id", signal_column}, "signals")
    frame = signals[["date", "security_id", signal_column]].copy()
    frame["date"] = _market_dates(frame["date"], label="signals.date")
    if frame["security_id"].isna().any():
        raise ValueError("signals.security_id contains nulls")
    frame["security_id"] = frame["security_id"].astype(str)
    if frame["security_id"].str.strip().eq("").any():
        raise ValueError("signals.security_id contains empty values")
    if frame.duplicated(["date", "security_id"]).any():
        raise ValueError("signals contains duplicate date/security_id rows")
    frame[signal_column] = pd.to_numeric(frame[signal_column], errors="raise")
    finite = frame[signal_column].dropna().map(np.isfinite)
    if not finite.all():
        raise ValueError(f"signals.{signal_column} contains non-finite values")
    return frame


def validate_target_weights(
    target_weights: pd.DataFrame,
    *,
    gross_exposure: float,
    target_net_exposure: float,
    single_name_cap: float,
    tolerance: float = 1e-12,
) -> None:
    """Fail loudly unless every signal-date portfolio meets its constraints."""

    if not isinstance(target_weights, pd.DataFrame):
        raise TypeError("target_weights must be a pandas DataFrame")
    _require_columns(target_weights, set(_TARGET_COLUMNS), "target_weights")
    gross = _positive_number(gross_exposure, label="gross_exposure")
    cap = _positive_number(single_name_cap, label="single_name_cap")
    net = float(target_net_exposure)
    tol = _positive_number(tolerance, label="tolerance", allow_zero=True)
    if not np.isfinite(net) or abs(net) > gross + tol:
        raise ValueError("target_net_exposure must be finite and within gross exposure")
    if cap > gross + tol:
        raise ValueError("single_name_cap cannot exceed gross_exposure")
    if target_weights.empty:
        return

    frame = target_weights[list(_TARGET_COLUMNS)].copy()
    frame["signal_date"] = _market_dates(
        frame["signal_date"], label="target_weights.signal_date"
    )
    if frame["security_id"].isna().any():
        raise ValueError("target_weights.security_id contains nulls")
    frame["security_id"] = frame["security_id"].astype(str)
    if frame["security_id"].str.strip().eq("").any():
        raise ValueError("target_weights.security_id contains empty values")
    if frame.duplicated(["signal_date", "security_id"]).any():
        raise ValueError("target_weights contains duplicate signal-date/security rows")
    frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="raise")
    if not frame["target_weight"].map(np.isfinite).all():
        raise ValueError("target_weights.target_weight must be finite")
    if frame["target_weight"].abs().gt(cap + tol).any():
        raise ValueError("target weight exceeds single_name_cap")

    grouped = frame.groupby("signal_date", sort=True)["target_weight"]
    observed_gross = grouped.apply(lambda values: float(values.abs().sum()))
    observed_net = grouped.sum()
    if not np.allclose(observed_gross, gross, rtol=0.0, atol=tol):
        raise ValueError("target weights do not meet gross_exposure exactly")
    if not np.allclose(observed_net, net, rtol=0.0, atol=tol):
        raise ValueError("target weights do not meet target_net_exposure exactly")


def build_quantile_target_weights(
    signals: pd.DataFrame,
    *,
    signal_column: str = "customer_shock",
    quantiles: int = 5,
    gross_exposure: float = 1.0,
    target_net_exposure: float = 0.0,
    single_name_cap: float = 0.02,
) -> pd.DataFrame:
    """Build deterministic equal-weight long-top/short-bottom portfolios.

    Ties are resolved by ``security_id`` and never by input order.  An
    infeasible single-name cap raises instead of silently returning an
    unconstrained or under-invested portfolio.
    """

    if isinstance(quantiles, bool) or not isinstance(quantiles, (int, np.integer)):
        raise TypeError("quantiles must be an integer")
    if quantiles < 2:
        raise ValueError("quantiles must be at least two")
    gross = _positive_number(gross_exposure, label="gross_exposure")
    cap = _positive_number(single_name_cap, label="single_name_cap")
    net = float(target_net_exposure)
    if not np.isfinite(net) or abs(net) > gross:
        raise ValueError("target_net_exposure must be finite and within gross exposure")

    frame = _prepare_signals(signals, signal_column=signal_column)
    rows: list[dict[str, object]] = []
    long_gross = (gross + net) / 2.0
    short_gross = (gross - net) / 2.0
    for signal_date, group in frame.groupby("date", sort=True):
        eligible = group.dropna(subset=[signal_column]).copy()
        if len(eligible) < 2:
            continue
        tail_count = max(1, len(eligible) // int(quantiles))
        tail_count = min(tail_count, len(eligible) // 2)
        if tail_count == 0:
            continue
        ascending = eligible.sort_values(
            [signal_column, "security_id"],
            ascending=[True, True],
            kind="stable",
        )
        descending = eligible.sort_values(
            [signal_column, "security_id"],
            ascending=[False, True],
            kind="stable",
        )
        short_ids = ascending.head(tail_count)["security_id"].tolist()
        long_ids = descending.head(tail_count)["security_id"].tolist()
        if set(short_ids).intersection(long_ids):
            raise RuntimeError("quantile tails overlap")
        long_weight = long_gross / tail_count if long_gross else 0.0
        short_weight = -(short_gross / tail_count) if short_gross else 0.0
        if max(abs(long_weight), abs(short_weight)) > cap + 1e-12:
            raise ValueError(
                "quantile portfolio is infeasible under single_name_cap; "
                "increase the eligible universe or reduce gross exposure"
            )
        rows.extend(
            {
                "signal_date": signal_date,
                "security_id": security_id,
                "target_weight": long_weight,
            }
            for security_id in long_ids
            if long_weight
        )
        rows.extend(
            {
                "signal_date": signal_date,
                "security_id": security_id,
                "target_weight": short_weight,
            }
            for security_id in short_ids
            if short_weight
        )

    result = pd.DataFrame(rows, columns=_TARGET_COLUMNS)
    if not result.empty:
        result = result.sort_values(
            ["signal_date", "security_id"], kind="stable"
        ).reset_index(drop=True)
        validate_target_weights(
            result,
            gross_exposure=gross,
            target_net_exposure=net,
            single_name_cap=cap,
        )
    return result


def _prepare_market_panel(market_panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(market_panel, pd.DataFrame):
        raise TypeError("market_panel must be a pandas DataFrame")
    _require_columns(market_panel, _MARKET_COLUMNS, "market_panel")
    frame = market_panel.copy()
    frame["date"] = _market_dates(frame["date"], label="market_panel.date")
    if frame["security_id"].isna().any():
        raise ValueError("market_panel.security_id contains nulls")
    frame["security_id"] = frame["security_id"].astype(str)
    if frame["security_id"].str.strip().eq("").any():
        raise ValueError("market_panel.security_id contains empty values")
    if frame.duplicated(["date", "security_id"]).any():
        raise ValueError("market_panel contains duplicate date/security_id rows")
    for column in ("open", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].dropna().map(np.isfinite).all():
            raise ValueError(f"market_panel.{column} contains non-finite values")
    if frame["open"].dropna().le(0).any():
        raise ValueError("market_panel.open must be positive when available")
    if frame["amount"].dropna().lt(0).any():
        raise ValueError("market_panel.amount must be non-negative")
    for column in ("tradable", "suspended", "is_suspended", "limit_up", "limit_down"):
        if column not in frame:
            continue
        nonnull = frame[column].dropna()
        if not nonnull.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"market_panel.{column} must contain booleans")
        frame[column] = frame[column].astype("boolean")
    return frame.sort_values(["date", "security_id"], kind="stable").reset_index(
        drop=True
    )


def _rule_frame(
    records: Any,
    *,
    label: str,
    required: set[str],
    group_column: str,
) -> pd.DataFrame:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError(f"market_rules.{label} must be a sequence")
    frame = pd.DataFrame(list(records))
    _require_columns(frame, {"valid_from", "valid_to", group_column, *required}, label)
    if frame.empty:
        raise ValueError(f"market_rules.{label} must not be empty")
    frame["valid_from"] = _market_dates(
        frame["valid_from"], label=f"market_rules.{label}.valid_from"
    )
    frame["valid_to"] = _market_dates(
        frame["valid_to"], label=f"market_rules.{label}.valid_to"
    )
    if frame["valid_from"].gt(frame["valid_to"]).any():
        raise ValueError(f"market_rules.{label} contains a reversed interval")
    provenance = pd.Series(False, index=frame.index)
    for column in ("source_note", "source_url"):
        if column in frame:
            provenance |= frame[column].notna() & frame[column].astype(
                str
            ).str.strip().ne("")
    if not provenance.all():
        raise ValueError(f"market_rules.{label} rows require source_note or source_url")
    ordered = frame.sort_values(
        [group_column, "valid_from", "valid_to"], kind="stable"
    ).reset_index(drop=True)
    for _, group in ordered.groupby(group_column, sort=False):
        prior_end = group["valid_to"].shift()
        if group["valid_from"].le(prior_end).fillna(False).any():
            raise ValueError(f"market_rules.{label} has overlapping validity intervals")
    return ordered


def _prepare_market_rules(
    market_rules: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, float, list[float]]:
    if not isinstance(market_rules, Mapping):
        raise TypeError("market_rules must be a mapping")
    settlement = _rule_frame(
        market_rules.get("settlement"),
        label="settlement",
        required={"purchased_shares_sellable_from_trading_day_offset"},
        group_column="security_scope",
    )
    settlement["purchased_shares_sellable_from_trading_day_offset"] = pd.to_numeric(
        settlement["purchased_shares_sellable_from_trading_day_offset"],
        errors="raise",
    )
    offsets = settlement["purchased_shares_sellable_from_trading_day_offset"]
    if offsets.mod(1).ne(0).any() or offsets.lt(1).any():
        raise ValueError("A-share settlement offsets must be positive integers")

    fees = _rule_frame(
        market_rules.get("taxes_and_fees"),
        label="taxes_and_fees",
        required={"buy_rate", "sell_rate"},
        group_column="name",
    )
    for column in ("buy_rate", "sell_rate"):
        fees[column] = pd.to_numeric(fees[column], errors="raise")
        if (
            ~fees[column].map(np.isfinite)
            | ~fees[column].between(0.0, 1.0, inclusive="left")
        ).any():
            raise ValueError(f"market_rules.taxes_and_fees.{column} must be in [0, 1)")

    assumptions = market_rules.get("assumptions")
    if not isinstance(assumptions, Mapping):
        raise TypeError("market_rules.assumptions must be a mapping")
    commission = _positive_number(
        assumptions.get("commission_bps_per_side"),
        label="commission_bps_per_side",
        allow_zero=True,
    )
    raw_slippage = assumptions.get("slippage_bps_per_side")
    if isinstance(raw_slippage, (str, bytes)) or not isinstance(raw_slippage, Sequence):
        raise TypeError("slippage_bps_per_side must be a sequence")
    slippage = [
        _positive_number(value, label="slippage_bps_per_side", allow_zero=True)
        for value in raw_slippage
    ]
    if not slippage:
        raise ValueError("slippage_bps_per_side must not be empty")
    return settlement, fees, commission, slippage


def _settlement_offset(settlement: pd.DataFrame, date: pd.Timestamp) -> int:
    active = settlement.loc[
        settlement["security_scope"].astype(str).str.upper().eq("A_SHARE")
        & settlement["valid_from"].le(date)
        & settlement["valid_to"].ge(date)
    ]
    if len(active) != 1:
        raise ValueError(f"expected one A-share settlement rule on {date.date()}")
    return int(active.iloc[0]["purchased_shares_sellable_from_trading_day_offset"])


def _fee_rates(
    fees: pd.DataFrame, date: pd.Timestamp, side: str
) -> tuple[float, float]:
    active = fees.loc[fees["valid_from"].le(date) & fees["valid_to"].ge(date)]
    if active.empty:
        raise ValueError(f"no transaction tax/fee rule covers {date.date()}")
    rate_column = "buy_rate" if side == "buy" else "sell_rate"
    stamp = active["name"].astype(str).str.casefold().str.contains("stamp_tax")
    return float(active.loc[stamp, rate_column].sum()), float(
        active.loc[~stamp, rate_column].sum()
    )


def _row_flag(row: pd.Series, *columns: str) -> bool:
    for column in columns:
        if column in row.index and pd.notna(row[column]) and bool(row[column]):
            return True
    return False


def _fill_block_reason(
    row: pd.Series | None,
    *,
    side: str,
    minimum_liquidity_amount: float,
) -> str | None:
    if row is None:
        return "missing_market_row"
    if _row_flag(row, "suspended", "is_suspended"):
        return "suspended"
    if pd.isna(row["open"]):
        return "missing_open"
    if side == "buy" and _row_flag(row, "limit_up"):
        return "limit_up_buy_locked"
    if side == "sell" and _row_flag(row, "limit_down"):
        return "limit_down_sell_locked"
    if pd.isna(row["tradable"]) or not bool(row["tradable"]):
        return "not_tradable"
    if pd.isna(row["amount"]) or float(row["amount"]) < minimum_liquidity_amount:
        return "below_minimum_liquidity"
    return None


def _empty_trade_row(
    *,
    signal_date: pd.Timestamp,
    security_id: str,
    target_weight: float,
    entry_date: pd.Timestamp | None,
    status: str,
    reason: str,
) -> dict[str, object]:
    row: dict[str, object] = {column: None for column in _TRADE_COLUMNS}
    row.update(
        {
            "signal_date": signal_date,
            "security_id": security_id,
            "target_weight": target_weight,
            "side": "long" if target_weight > 0 else "short",
            "entry_date": entry_date,
            "status": status,
            "entry_block_reason": reason,
            "exit_defer_count": 0,
            "entry_notional": 0.0,
            "exit_notional": 0.0,
            "gross_pnl": 0.0,
            "commission_cost": 0.0,
            "slippage_cost": 0.0,
            "stamp_tax_cost": 0.0,
            "venue_fee_cost": 0.0,
            "total_cost": 0.0,
            "net_pnl": 0.0,
            "gross_return": 0.0,
            "net_return": 0.0,
            "holding_period_trading_days": 0,
        }
    )
    for prefix in ("entry", "exit"):
        for name in (
            "commission_cost",
            "slippage_cost",
            "stamp_tax_cost",
            "venue_fee_cost",
        ):
            row[f"{prefix}_{name}"] = 0.0
    return row


def simulate_open_round_trips(
    target_weights: pd.DataFrame,
    market_panel: pd.DataFrame,
    market_rules: Mapping[str, Any],
    *,
    minimum_liquidity_amount: float = 0.0,
    portfolio_value: float = 1.0,
    holding_period_trading_days: int = 1,
    commission_bps_per_side: float | None = None,
    slippage_bps_per_side: float | None = None,
) -> pd.DataFrame:
    """Execute independent signal cohorts at open with T+1-safe exits.

    Invalid entries are rejected.  Once an entry fills, an invalid exit is
    deferred to the next eligible supplied trading date; if none exists the
    position is reported as ``OPEN`` without inventing an exit price.
    """

    if isinstance(holding_period_trading_days, bool) or not isinstance(
        holding_period_trading_days, (int, np.integer)
    ):
        raise TypeError("holding_period_trading_days must be an integer")
    if holding_period_trading_days < 1:
        raise ValueError("holding_period_trading_days must be at least one")
    liquidity = _positive_number(
        minimum_liquidity_amount,
        label="minimum_liquidity_amount",
        allow_zero=True,
    )
    capital = _positive_number(portfolio_value, label="portfolio_value")
    market = _prepare_market_panel(market_panel)
    settlement, fees, default_commission, slippage_choices = _prepare_market_rules(
        market_rules
    )
    commission_bps = (
        default_commission
        if commission_bps_per_side is None
        else _positive_number(
            commission_bps_per_side,
            label="commission_bps_per_side",
            allow_zero=True,
        )
    )
    slippage_bps = (
        (10.0 if 10.0 in slippage_choices else slippage_choices[0])
        if slippage_bps_per_side is None
        else _positive_number(
            slippage_bps_per_side,
            label="slippage_bps_per_side",
            allow_zero=True,
        )
    )

    _require_columns(target_weights, set(_TARGET_COLUMNS), "target_weights")
    targets = target_weights[list(_TARGET_COLUMNS)].copy()
    if targets.empty:
        return pd.DataFrame(columns=_TRADE_COLUMNS)
    targets["signal_date"] = _market_dates(
        targets["signal_date"], label="target_weights.signal_date"
    )
    if targets["security_id"].isna().any():
        raise ValueError("target_weights.security_id contains nulls")
    targets["security_id"] = targets["security_id"].astype(str)
    if targets["security_id"].str.strip().eq("").any():
        raise ValueError("target_weights.security_id contains empty values")
    targets["target_weight"] = pd.to_numeric(targets["target_weight"], errors="raise")
    if targets.duplicated(["signal_date", "security_id"]).any():
        raise ValueError("target_weights contains duplicate signal-date/security rows")
    if not targets["target_weight"].map(np.isfinite).all():
        raise ValueError("target_weights.target_weight must be finite")
    targets = targets.loc[targets["target_weight"].ne(0)].sort_values(
        ["signal_date", "security_id"], kind="stable"
    )

    trading_dates = pd.DatetimeIndex(sorted(market["date"].unique()))
    date_positions = {date: index for index, date in enumerate(trading_dates)}
    keyed_market = market.set_index(["date", "security_id"], verify_integrity=True)
    rows: list[dict[str, object]] = []
    for target in targets.itertuples(index=False):
        signal_date = pd.Timestamp(target.signal_date)
        signal_index = date_positions.get(signal_date)
        if signal_index is None:
            raise ValueError(
                f"signal date is absent from trading calendar: {signal_date.date()}"
            )
        entry_index = signal_index + 1
        if entry_index >= len(trading_dates):
            rows.append(
                _empty_trade_row(
                    signal_date=signal_date,
                    security_id=target.security_id,
                    target_weight=float(target.target_weight),
                    entry_date=None,
                    status="ENTRY_REJECTED",
                    reason="no_next_trading_day",
                )
            )
            continue
        entry_date = pd.Timestamp(trading_dates[entry_index])
        market_key = (entry_date, target.security_id)
        entry_row = (
            keyed_market.loc[market_key] if market_key in keyed_market.index else None
        )
        entry_side = "buy" if target.target_weight > 0 else "sell"
        entry_reason = _fill_block_reason(
            entry_row,
            side=entry_side,
            minimum_liquidity_amount=liquidity,
        )
        if entry_reason is not None:
            rows.append(
                _empty_trade_row(
                    signal_date=signal_date,
                    security_id=target.security_id,
                    target_weight=float(target.target_weight),
                    entry_date=entry_date,
                    status="ENTRY_REJECTED",
                    reason=entry_reason,
                )
            )
            continue

        settlement_days = _settlement_offset(settlement, entry_date)
        minimum_hold = max(int(holding_period_trading_days), settlement_days)
        scheduled_exit_index = entry_index + minimum_hold
        scheduled_exit_date = (
            pd.Timestamp(trading_dates[scheduled_exit_index])
            if scheduled_exit_index < len(trading_dates)
            else None
        )
        exit_side = "sell" if target.target_weight > 0 else "buy"
        exit_row: pd.Series | None = None
        exit_date: pd.Timestamp | None = None
        exit_index: int | None = None
        for candidate_index in range(scheduled_exit_index, len(trading_dates)):
            candidate_date = pd.Timestamp(trading_dates[candidate_index])
            candidate_key = (candidate_date, target.security_id)
            candidate = (
                keyed_market.loc[candidate_key]
                if candidate_key in keyed_market.index
                else None
            )
            reason = _fill_block_reason(
                candidate,
                side=exit_side,
                minimum_liquidity_amount=liquidity,
            )
            if reason is None:
                exit_row = candidate
                exit_date = candidate_date
                exit_index = candidate_index
                break

        weight = float(target.target_weight)
        direction = 1.0 if weight > 0 else -1.0
        entry_price = float(entry_row["open"])  # type: ignore[index]
        entry_notional = abs(weight) * capital
        quantity = entry_notional / entry_price
        entry_stamp_rate, entry_venue_rate = _fee_rates(fees, entry_date, entry_side)
        entry_commission = entry_notional * commission_bps / 10_000.0
        entry_slippage = entry_notional * slippage_bps / 10_000.0
        entry_stamp = entry_notional * entry_stamp_rate
        entry_venue = entry_notional * entry_venue_rate

        if exit_row is None or exit_date is None or exit_index is None:
            entry_cost = entry_commission + entry_slippage + entry_stamp + entry_venue
            row = _empty_trade_row(
                signal_date=signal_date,
                security_id=target.security_id,
                target_weight=weight,
                entry_date=entry_date,
                status="OPEN",
                reason="",
            )
            row.update(
                {
                    "scheduled_exit_date": scheduled_exit_date,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "entry_notional": entry_notional,
                    "entry_commission_cost": entry_commission,
                    "commission_cost": entry_commission,
                    "entry_slippage_cost": entry_slippage,
                    "slippage_cost": entry_slippage,
                    "entry_stamp_tax_cost": entry_stamp,
                    "stamp_tax_cost": entry_stamp,
                    "entry_venue_fee_cost": entry_venue,
                    "venue_fee_cost": entry_venue,
                    "total_cost": entry_cost,
                    "net_pnl": -entry_cost,
                    "net_return": -entry_cost / capital,
                    "holding_period_trading_days": len(trading_dates) - entry_index - 1,
                }
            )
            rows.append(row)
            continue

        exit_price = float(exit_row["open"])
        exit_notional = quantity * exit_price
        gross_pnl = direction * (exit_notional - entry_notional)
        exit_stamp_rate, exit_venue_rate = _fee_rates(fees, exit_date, exit_side)
        exit_commission = exit_notional * commission_bps / 10_000.0
        exit_slippage = exit_notional * slippage_bps / 10_000.0
        exit_stamp = exit_notional * exit_stamp_rate
        exit_venue = exit_notional * exit_venue_rate
        commission_cost = entry_commission + exit_commission
        slippage_cost = entry_slippage + exit_slippage
        stamp_cost = entry_stamp + exit_stamp
        venue_cost = entry_venue + exit_venue
        total_cost = commission_cost + slippage_cost + stamp_cost + venue_cost
        net_pnl = gross_pnl - total_cost
        rows.append(
            {
                "signal_date": signal_date,
                "security_id": target.security_id,
                "target_weight": weight,
                "side": "long" if weight > 0 else "short",
                "entry_date": entry_date,
                "scheduled_exit_date": scheduled_exit_date,
                "exit_date": exit_date,
                "status": "CLOSED",
                "entry_block_reason": "",
                "exit_defer_count": exit_index - scheduled_exit_index,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "entry_notional": entry_notional,
                "exit_notional": exit_notional,
                "gross_pnl": gross_pnl,
                "entry_commission_cost": entry_commission,
                "exit_commission_cost": exit_commission,
                "commission_cost": commission_cost,
                "entry_slippage_cost": entry_slippage,
                "exit_slippage_cost": exit_slippage,
                "slippage_cost": slippage_cost,
                "entry_stamp_tax_cost": entry_stamp,
                "exit_stamp_tax_cost": exit_stamp,
                "stamp_tax_cost": stamp_cost,
                "entry_venue_fee_cost": entry_venue,
                "exit_venue_fee_cost": exit_venue,
                "venue_fee_cost": venue_cost,
                "total_cost": total_cost,
                "net_pnl": net_pnl,
                "gross_return": gross_pnl / capital,
                "net_return": net_pnl / capital,
                "holding_period_trading_days": exit_index - entry_index,
            }
        )

    return (
        pd.DataFrame(rows, columns=_TRADE_COLUMNS)
        .sort_values(["signal_date", "security_id"], kind="stable")
        .reset_index(drop=True)
    )


def aggregate_daily_pnl(
    trades: pd.DataFrame,
    trading_dates: Sequence[Any],
    *,
    portfolio_value: float = 1.0,
) -> pd.DataFrame:
    """Aggregate transparent side-level costs and realized P&L by market date."""

    capital = _positive_number(portfolio_value, label="portfolio_value")
    calendar = pd.Series(list(trading_dates), dtype="object")
    dates = pd.DatetimeIndex(
        _market_dates(calendar, label="trading_dates").drop_duplicates().sort_values()
    )
    if dates.empty:
        raise ValueError("trading_dates must not be empty")
    output = pd.DataFrame({"date": dates})
    numeric_columns = [
        "gross_pnl",
        "commission_cost",
        "slippage_cost",
        "stamp_tax_cost",
        "venue_fee_cost",
        "total_cost",
        "net_pnl",
        "turnover",
        "gross_exposure",
        "net_exposure",
        "closed_trade_count",
    ]
    for column in numeric_columns:
        output[column] = 0.0
    if trades.empty:
        output["gross_return"] = 0.0
        output["net_return"] = 0.0
        output["closed_trade_count"] = output["closed_trade_count"].astype("Int64")
        return output

    _require_columns(trades, set(_TRADE_COLUMNS), "trades")
    frame = trades.copy()
    for column in ("signal_date", "entry_date", "scheduled_exit_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    for column in (
        "gross_pnl",
        "entry_commission_cost",
        "exit_commission_cost",
        "entry_slippage_cost",
        "exit_slippage_cost",
        "entry_stamp_tax_cost",
        "exit_stamp_tax_cost",
        "entry_venue_fee_cost",
        "exit_venue_fee_cost",
        "entry_notional",
        "exit_notional",
        "target_weight",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    date_index = {date: index for index, date in enumerate(output["date"])}
    for trade in frame.itertuples(index=False):
        entry_index = date_index.get(trade.entry_date)
        if entry_index is not None and trade.status in {"CLOSED", "OPEN"}:
            entry_costs = {
                "commission_cost": trade.entry_commission_cost,
                "slippage_cost": trade.entry_slippage_cost,
                "stamp_tax_cost": trade.entry_stamp_tax_cost,
                "venue_fee_cost": trade.entry_venue_fee_cost,
            }
            for column, value in entry_costs.items():
                output.loc[entry_index, column] += value
            output.loc[entry_index, "turnover"] += trade.entry_notional / capital

        exit_index = date_index.get(trade.exit_date)
        if exit_index is not None and trade.status == "CLOSED":
            output.loc[exit_index, "gross_pnl"] += trade.gross_pnl
            exit_costs = {
                "commission_cost": trade.exit_commission_cost,
                "slippage_cost": trade.exit_slippage_cost,
                "stamp_tax_cost": trade.exit_stamp_tax_cost,
                "venue_fee_cost": trade.exit_venue_fee_cost,
            }
            for column, value in exit_costs.items():
                output.loc[exit_index, column] += value
            output.loc[exit_index, "turnover"] += trade.exit_notional / capital
            output.loc[exit_index, "closed_trade_count"] += 1

        if entry_index is not None and trade.status in {"CLOSED", "OPEN"}:
            end_index = exit_index if exit_index is not None else len(output)
            if end_index > entry_index:
                active = output.index[entry_index:end_index]
                output.loc[active, "gross_exposure"] += abs(trade.target_weight)
                output.loc[active, "net_exposure"] += trade.target_weight

    output["total_cost"] = output[
        ["commission_cost", "slippage_cost", "stamp_tax_cost", "venue_fee_cost"]
    ].sum(axis=1)
    output["net_pnl"] = output["gross_pnl"] - output["total_cost"]
    output["gross_return"] = output["gross_pnl"] / capital
    output["net_return"] = output["net_pnl"] / capital
    output["closed_trade_count"] = output["closed_trade_count"].astype("Int64")
    return output


__all__ = [
    "PortfolioBacktestResult",
    "aggregate_daily_pnl",
    "build_quantile_target_weights",
    "simulate_open_round_trips",
    "validate_target_weights",
]
