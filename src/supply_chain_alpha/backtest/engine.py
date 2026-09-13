"""Compact deterministic wrapper for the portfolio execution primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from supply_chain_alpha.portfolio.execution import (
    PortfolioBacktestResult,
    aggregate_daily_pnl,
    build_quantile_target_weights,
    simulate_open_round_trips,
)


def run_quantile_open_backtest(
    signals: pd.DataFrame,
    market_panel: pd.DataFrame,
    market_rules: Mapping[str, Any],
    *,
    signal_column: str = "customer_shock",
    quantiles: int = 5,
    gross_exposure: float = 1.0,
    target_net_exposure: float = 0.0,
    single_name_cap: float = 0.02,
    minimum_liquidity_amount: float = 0.0,
    portfolio_value: float = 1.0,
    holding_period_trading_days: int = 1,
    commission_bps_per_side: float | None = None,
    slippage_bps_per_side: float | None = None,
) -> PortfolioBacktestResult:
    """Construct signal tails, execute next-open cohorts, and reconcile P&L."""

    weights = build_quantile_target_weights(
        signals,
        signal_column=signal_column,
        quantiles=quantiles,
        gross_exposure=gross_exposure,
        target_net_exposure=target_net_exposure,
        single_name_cap=single_name_cap,
    )
    trades = simulate_open_round_trips(
        weights,
        market_panel,
        market_rules,
        minimum_liquidity_amount=minimum_liquidity_amount,
        portfolio_value=portfolio_value,
        holding_period_trading_days=holding_period_trading_days,
        commission_bps_per_side=commission_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )
    dates = sorted(
        pd.to_datetime(market_panel["date"], errors="raise").dt.date.unique()
    )
    daily = aggregate_daily_pnl(trades, dates, portfolio_value=portfolio_value)
    return PortfolioBacktestResult(weights, trades, daily)


__all__ = ["PortfolioBacktestResult", "run_quantile_open_backtest"]
