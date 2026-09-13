from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from supply_chain_alpha.backtest.engine import run_quantile_open_backtest
from supply_chain_alpha.portfolio.execution import (
    aggregate_daily_pnl,
    build_quantile_target_weights,
    simulate_open_round_trips,
    validate_target_weights,
)

ROOT = Path(__file__).parents[2]


def _market_rules() -> dict[str, object]:
    payload = yaml.safe_load(
        (ROOT / "config" / "market_rules.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _signals(date: str = "2023-08-24") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date] * 10,
            "security_id": [f"S{index:02d}" for index in range(10)],
            "customer_shock": [float(index) for index in range(10)],
        }
    )


def _market_rows(
    dates: list[str],
    securities: list[str],
    *,
    open_by_security: dict[str, list[float | None]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for security_id in securities:
        opens = (
            open_by_security[security_id]
            if open_by_security is not None
            else [100.0] * len(dates)
        )
        for date, open_price in zip(dates, opens, strict=True):
            rows.append(
                {
                    "date": date,
                    "security_id": security_id,
                    "open": open_price,
                    "amount": 10_000_000.0,
                    "tradable": True,
                    "suspended": False,
                    "limit_up": False,
                    "limit_down": False,
                }
            )
    return pd.DataFrame(rows)


def test_quantile_weights_are_deterministic_and_meet_constraints() -> None:
    shuffled = _signals().sample(frac=1.0, random_state=19).reset_index(drop=True)
    weights = build_quantile_target_weights(
        shuffled,
        quantiles=5,
        gross_exposure=1.0,
        target_net_exposure=0.0,
        single_name_cap=0.25,
    )
    keyed = weights.set_index("security_id")["target_weight"]

    assert keyed.to_dict() == {
        "S00": -0.25,
        "S01": -0.25,
        "S08": 0.25,
        "S09": 0.25,
    }
    validate_target_weights(
        weights,
        gross_exposure=1.0,
        target_net_exposure=0.0,
        single_name_cap=0.25,
    )
    with pytest.raises(ValueError, match="infeasible under single_name_cap"):
        build_quantile_target_weights(
            _signals(),
            quantiles=5,
            gross_exposure=1.0,
            target_net_exposure=0.0,
            single_name_cap=0.20,
        )


def test_signal_fills_next_open_t_plus_one_and_costs_reconcile() -> None:
    dates = ["2023-08-24", "2023-08-25", "2023-08-28", "2023-08-29"]
    panel = _market_rows(
        dates,
        ["LONG", "SHORT"],
        open_by_security={
            "LONG": [100.0, 100.0, 110.0, 110.0],
            "SHORT": [100.0, 100.0, 90.0, 90.0],
        },
    )
    targets = pd.DataFrame(
        {
            "signal_date": ["2023-08-24", "2023-08-24"],
            "security_id": ["LONG", "SHORT"],
            "target_weight": [0.5, -0.5],
        }
    )
    trades = simulate_open_round_trips(
        targets,
        panel,
        _market_rules(),
        portfolio_value=1.0,
        commission_bps_per_side=3,
        slippage_bps_per_side=10,
    ).set_index("security_id")

    assert set(trades["status"]) == {"CLOSED"}
    assert set(trades["entry_date"].dt.date.astype(str)) == {"2023-08-25"}
    assert set(trades["exit_date"].dt.date.astype(str)) == {"2023-08-28"}
    assert set(trades["holding_period_trading_days"]) == {1}
    assert trades.loc["LONG", "gross_pnl"] == pytest.approx(0.05)
    assert trades.loc["SHORT", "gross_pnl"] == pytest.approx(0.05)

    # The long sells after the 2023-08-28 tax cut; the short sells on entry
    # under the older rate. Buy-side stamp tax remains zero.
    assert trades.loc["LONG", "stamp_tax_cost"] == pytest.approx(0.55 * 0.0005)
    assert trades.loc["SHORT", "stamp_tax_cost"] == pytest.approx(0.50 * 0.001)
    assert trades.loc["LONG", "commission_cost"] == pytest.approx(
        (0.50 + 0.55) * 0.0003
    )
    assert trades.loc["LONG", "slippage_cost"] == pytest.approx((0.50 + 0.55) * 0.001)
    assert trades["total_cost"].to_numpy() == pytest.approx(
        (
            trades["commission_cost"]
            + trades["slippage_cost"]
            + trades["stamp_tax_cost"]
            + trades["venue_fee_cost"]
        ).to_numpy()
    )
    assert trades["net_pnl"].to_numpy() == pytest.approx(
        (trades["gross_pnl"] - trades["total_cost"]).to_numpy()
    )

    daily = aggregate_daily_pnl(trades.reset_index(), dates, portfolio_value=1.0)
    assert daily["net_pnl"].sum() == pytest.approx(trades["net_pnl"].sum())
    assert daily["gross_pnl"].sum() == pytest.approx(trades["gross_pnl"].sum())
    assert daily["total_cost"].sum() == pytest.approx(trades["total_cost"].sum())
    entry_day = daily.set_index(daily["date"].dt.date.astype(str)).loc["2023-08-25"]
    assert entry_day["gross_exposure"] == pytest.approx(1.0)
    assert entry_day["net_exposure"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("security_id", "weight", "updates", "reason"),
    [
        ("SUSP", 0.1, {"suspended": True}, "suspended"),
        ("MISS", 0.1, {"open": None}, "missing_open"),
        ("LUP", 0.1, {"limit_up": True}, "limit_up_buy_locked"),
        ("LDN", -0.1, {"limit_down": True}, "limit_down_sell_locked"),
        ("ILLIQ", 0.1, {"amount": 99.0}, "below_minimum_liquidity"),
        ("HALT", 0.1, {"tradable": False}, "not_tradable"),
    ],
)
def test_invalid_entry_market_states_do_not_fill(
    security_id: str,
    weight: float,
    updates: dict[str, object],
    reason: str,
) -> None:
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    panel = _market_rows(dates, [security_id])
    entry_mask = panel["date"].eq("2020-01-03")
    for column, value in updates.items():
        panel.loc[entry_mask, column] = value
    targets = pd.DataFrame(
        {
            "signal_date": ["2020-01-02"],
            "security_id": [security_id],
            "target_weight": [weight],
        }
    )

    trade = simulate_open_round_trips(
        targets,
        panel,
        _market_rules(),
        minimum_liquidity_amount=100.0,
    ).iloc[0]
    assert trade["status"] == "ENTRY_REJECTED"
    assert trade["entry_block_reason"] == reason
    assert trade["entry_notional"] == 0.0
    assert trade["total_cost"] == 0.0


def test_t_plus_one_exit_is_deferred_until_sell_is_permitted() -> None:
    dates = [
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
    ]
    panel = _market_rows(dates, ["LONG"])
    panel.loc[panel["date"].eq("2020-01-06"), "limit_down"] = True
    panel.loc[panel["date"].eq("2020-01-07"), "suspended"] = True
    targets = pd.DataFrame(
        {
            "signal_date": ["2020-01-02"],
            "security_id": ["LONG"],
            "target_weight": [0.5],
        }
    )

    trade = simulate_open_round_trips(
        targets, panel, _market_rules(), slippage_bps_per_side=10
    ).iloc[0]
    assert trade["entry_date"].date().isoformat() == "2020-01-03"
    assert trade["scheduled_exit_date"].date().isoformat() == "2020-01-06"
    assert trade["exit_date"].date().isoformat() == "2020-01-08"
    assert trade["exit_defer_count"] == 2
    assert trade["holding_period_trading_days"] == 3


def test_no_positions_produce_zero_return_days() -> None:
    signals = _signals()
    signals["customer_shock"] = None
    dates = ["2023-08-24", "2023-08-25", "2023-08-28"]
    panel = _market_rows(dates, signals["security_id"].tolist())
    result = run_quantile_open_backtest(
        signals,
        panel,
        _market_rules(),
        single_name_cap=0.25,
        slippage_bps_per_side=10,
    )

    assert result.target_weights.empty
    assert result.trades.empty
    assert (result.daily_pnl["gross_return"] == 0.0).all()
    assert (result.daily_pnl["net_return"] == 0.0).all()
    assert (result.daily_pnl["gross_exposure"] == 0.0).all()
    assert (result.daily_pnl["net_exposure"] == 0.0).all()


def test_appending_future_signals_and_market_rows_does_not_change_history() -> None:
    base_dates = ["2023-08-24", "2023-08-25", "2023-08-28", "2023-08-29"]
    securities = _signals()["security_id"].tolist()
    base = run_quantile_open_backtest(
        _signals(),
        _market_rows(base_dates, securities),
        _market_rules(),
        single_name_cap=0.25,
        slippage_bps_per_side=10,
    )
    future_signals = pd.concat([_signals(), _signals("2023-08-29")], ignore_index=True)
    future_dates = [*base_dates, "2023-08-30", "2023-08-31"]
    rerun = run_quantile_open_backtest(
        future_signals,
        _market_rows(future_dates, securities),
        _market_rules(),
        single_name_cap=0.25,
        slippage_bps_per_side=10,
    )

    historical_weights = rerun.target_weights.loc[
        rerun.target_weights["signal_date"].le(pd.Timestamp("2023-08-24"))
    ].reset_index(drop=True)
    historical_trades = rerun.trades.loc[
        rerun.trades["signal_date"].le(pd.Timestamp("2023-08-24"))
    ].reset_index(drop=True)
    historical_daily = rerun.daily_pnl.loc[
        rerun.daily_pnl["date"].le(pd.Timestamp("2023-08-29"))
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(base.target_weights, historical_weights)
    pd.testing.assert_frame_equal(base.trades, historical_trades)
    pd.testing.assert_frame_equal(base.daily_pnl, historical_daily)
