from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from supply_chain_alpha.backtest import run_quantile_open_backtest
from supply_chain_alpha.data.disclosures import mentions_from_text
from supply_chain_alpha.data.market import build_market_panel
from supply_chain_alpha.data.schemas import (
    COMPANY_ALIAS,
    DISCLOSURE_RAW,
    SECURITY_MASTER,
    SUPPLY_CHAIN_EDGE,
    validate_table,
)
from supply_chain_alpha.entities.resolve import ResolutionConfig, resolve_mentions
from supply_chain_alpha.evaluation.statistics import (
    daily_information_coefficients,
    degree_preserving_random_graph,
    moving_block_bootstrap_mean,
    newey_west_mean_standard_error,
)
from supply_chain_alpha.factors.residuals import build_residual_returns
from supply_chain_alpha.graph.edges import build_point_in_time_edges
from supply_chain_alpha.signals import build_diffusion_signals

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "tiny_e2e"


def _extract_fixture_mentions(raw: pd.DataFrame) -> pd.DataFrame:
    extracted: list[pd.DataFrame] = []
    for row in raw.itertuples(index=False):
        mentions = mentions_from_text(
            row.raw_text,
            source_company_id=row.source_company_id,
            relationship_type=row.relationship_type,
            source_period_end=row.source_period_end,
            publication_datetime=row.publication_datetime,
            source_document_id=row.source_document_id,
            source_document_url_or_path=row.source_document_url_or_path,
        )
        mentions["exposure_value"] = float(row.exposure_value)
        mentions["exposure_share"] = float(row.exposure_share)
        extracted.append(mentions)
    result = pd.concat(extracted, ignore_index=True)
    return validate_table(result, DISCLOSURE_RAW)


def _synthetic_prices(
    master: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    scenario: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = np.random.default_rng(int(scenario["random_seed"]))
    common_returns = generator.normal(0.0003, 0.004, size=len(calendar))
    suspension = scenario["suspension"]
    limit_lock = scenario["price_limit_lock"]
    action = scenario["corporate_action"]
    assert isinstance(suspension, dict)
    assert isinstance(limit_lock, dict)
    assert isinstance(action, dict)

    rows: list[dict[str, object]] = []
    for security_number, security in enumerate(master.itertuples(index=False), start=1):
        listing_date = pd.Timestamp(security.listing_date)
        previous_close = 8.0 + security_number
        idiosyncratic = generator.normal(0.0, 0.005, size=len(calendar))
        for day_index, day in enumerate(calendar):
            if day < listing_date:
                continue
            daily_return = float(
                np.clip(
                    common_returns[day_index] + idiosyncratic[day_index], -0.025, 0.025
                )
            )
            is_suspension = security.security_id == suspension[
                "security_id"
            ] and day_index == int(suspension["day_index"])
            is_limit_lock = security.security_id == limit_lock[
                "security_id"
            ] and day_index == int(limit_lock["day_index"])
            is_action = security.security_id == action[
                "security_id"
            ] and day_index == int(action["day_index"])

            if is_suspension:
                close = previous_close
                open_price = close
                high = close
                low = close
                volume = 0.0
                amount = 0.0
            elif is_limit_lock:
                close = np.floor(previous_close * 1.10 * 100.0 + 0.5 + 1e-12) / 100.0
                open_price = close
                high = close
                low = close
                volume = 1_000_000.0
                amount = close * volume
            else:
                action_multiplier = (
                    float(action["share_multiplier"]) if is_action else 1.0
                )
                close = previous_close * (1.0 + daily_return) / action_multiplier
                open_price = (
                    previous_close * (1.0 + 0.25 * daily_return) / action_multiplier
                )
                high = max(open_price, close) * 1.002
                low = min(open_price, close) * 0.998
                volume = 1_000_000.0 + 1_000.0 * security_number
                amount = volume * (open_price + close) / 2.0

            rows.append(
                {
                    "date": day,
                    "security_id": security.security_id,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                    "is_suspended": bool(is_suspension),
                }
            )
            previous_close = close

    action_day = calendar[int(action["day_index"])]
    actions = pd.DataFrame(
        [
            {
                "date": action_day,
                "security_id": action["security_id"],
                "share_multiplier": float(action["share_multiplier"]),
                "cash_dividend": 0.0,
            }
        ]
    )
    return pd.DataFrame(rows), actions


def _run_tiny_chain() -> dict[str, object]:
    scenario = yaml.safe_load((FIXTURE / "scenario.yaml").read_text(encoding="utf-8"))
    master = pd.read_csv(FIXTURE / "security_master.csv", dtype=str)
    aliases = pd.read_csv(FIXTURE / "company_alias.csv", dtype=str)
    raw = pd.read_csv(FIXTURE / "raw_disclosures.csv", dtype=str)
    validate_table(master, SECURITY_MASTER)
    validate_table(aliases, COMPANY_ALIAS)

    mentions = _extract_fixture_mentions(raw)
    audit = resolve_mentions(
        mentions,
        aliases,
        config=ResolutionConfig(fuzzy_enabled=False),
    )
    edges = build_point_in_time_edges(mentions, audit, max_age_days=550)
    validate_table(edges, SUPPLY_CHAIN_EDGE)

    calendar = pd.bdate_range(
        start=str(scenario["trading_start"]),
        periods=int(scenario["trading_days"]),
    )
    prices, actions = _synthetic_prices(master, calendar, scenario)
    market_rules = yaml.safe_load(
        (ROOT / "config" / "market_rules.yaml").read_text(encoding="utf-8")
    )
    market = build_market_panel(
        prices,
        master,
        calendar,
        market_rules["price_limits"],
        corporate_actions=actions,
    )
    market_returns = (
        market.groupby("date", as_index=False, sort=True)["raw_return"]
        .mean()
        .rename(columns={"raw_return": "market_return"})
    )
    industry = master[["security_id"]].copy()
    industry["industry_code"] = [
        "IND_A" if index % 2 else "IND_B" for index in range(len(industry))
    ]
    industry["valid_from"] = "2019-01-01"
    industry["valid_to"] = None
    residuals = build_residual_returns(
        market,
        market_returns,
        beta_window=20,
        min_observations=10,
        information_lag_days=1,
        industry_history=industry,
    )
    signals = build_diffusion_signals(edges, residuals)
    ic = daily_information_coefficients(
        signals, residuals, horizon=1, signal_column="customer_shock"
    )
    bootstrap = moving_block_bootstrap_mean(
        ic["rank_ic"],
        replications=128,
        block_length=5,
        seed=int(scenario["random_seed"]),
    )
    nw_se = newey_west_mean_standard_error(ic["rank_ic"], lags=3)
    placebo = degree_preserving_random_graph(
        edges[["supplier_id", "customer_id"]], seed=int(scenario["random_seed"])
    )

    fully_active = pd.Timestamp("2020-02-11")
    portfolio_signals = signals.loc[pd.to_datetime(signals["date"]).ge(fully_active)]
    backtest = run_quantile_open_backtest(
        portfolio_signals,
        market,
        market_rules,
        signal_column="customer_shock",
        quantiles=3,
        gross_exposure=1.0,
        target_net_exposure=0.0,
        single_name_cap=0.5,
        minimum_liquidity_amount=1_000_000.0,
        portfolio_value=1_000_000.0,
        holding_period_trading_days=1,
        commission_bps_per_side=3.0,
        slippage_bps_per_side=10.0,
    )
    return {
        "scenario": scenario,
        "master": master,
        "mentions": mentions,
        "audit": audit,
        "edges": edges,
        "calendar": calendar,
        "market": market,
        "residuals": residuals,
        "signals": signals,
        "ic": ic,
        "bootstrap": bootstrap,
        "nw_se": nw_se,
        "placebo": placebo,
        "backtest": backtest,
    }


def test_required_tiny_end_to_end_fixture() -> None:
    result = _run_tiny_chain()
    scenario = result["scenario"]
    master = result["master"]
    audit = result["audit"]
    edges = result["edges"]
    calendar = result["calendar"]
    market = result["market"]
    residuals = result["residuals"]
    signals = result["signals"]
    ic = result["ic"]
    bootstrap = result["bootstrap"]
    placebo = result["placebo"]
    backtest = result["backtest"]
    assert isinstance(scenario, dict)
    assert isinstance(master, pd.DataFrame)
    assert isinstance(audit, pd.DataFrame)
    assert isinstance(edges, pd.DataFrame)
    assert isinstance(calendar, pd.DatetimeIndex)
    assert isinstance(market, pd.DataFrame)
    assert isinstance(residuals, pd.DataFrame)
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(ic, pd.DataFrame)
    assert isinstance(bootstrap, pd.Series)
    assert isinstance(placebo, pd.DataFrame)

    assert len(master) == int(scenario["security_count"]) == 12
    assert len(calendar) == int(scenario["trading_days"]) == 160
    assert len(edges) == int(scenario["directed_named_relationships"]) == 6
    assert audit["resolution_status"].value_counts().to_dict() == {
        "resolved": 6,
        "anonymous": 1,
        "unresolved": 1,
    }
    renamed = audit.loc[audit["source_document_id"].eq("D06")].iloc[0]
    assert renamed["resolved_entity_id"] == "T05"
    future = audit.loc[audit["source_document_id"].eq("D08")].iloc[0]
    assert future["resolution_status"] == "unresolved"

    suspension = scenario["suspension"]
    limit_lock = scenario["price_limit_lock"]
    action = scenario["corporate_action"]
    assert isinstance(suspension, dict)
    assert isinstance(limit_lock, dict)
    assert isinstance(action, dict)
    suspension_row = market.loc[
        market["security_id"].eq(suspension["security_id"])
        & market["date"].eq(calendar[int(suspension["day_index"])])
    ].iloc[0]
    limit_row = market.loc[
        market["security_id"].eq(limit_lock["security_id"])
        & market["date"].eq(calendar[int(limit_lock["day_index"])])
    ].iloc[0]
    action_row = market.loc[
        market["security_id"].eq(action["security_id"])
        & market["date"].eq(calendar[int(action["day_index"])])
    ].iloc[0]
    assert suspension_row["price_limit_state"] == "suspended"
    assert not suspension_row["is_tradable"]
    assert limit_row["price_limit_state"] == "limit_up"
    assert limit_row["price_limit_locked"]
    assert not limit_row["is_tradable"]
    assert action_row["corporate_action"]
    assert action_row["share_multiplier"] == 2.0
    future_listing = pd.Timestamp(str(scenario["future_listing"]["listing_date"]))
    assert market.loc[market["security_id"].eq("T12"), "date"].min() == future_listing

    assert residuals["residual_return"].notna().sum() > 1_000
    assert not signals.empty
    assert signals["date"].equals(signals["graph_snapshot_date"])
    assert len(ic) > 100
    assert len(bootstrap) == 128
    pd.testing.assert_series_equal(
        bootstrap,
        moving_block_bootstrap_mean(
            ic["rank_ic"], replications=128, block_length=5, seed=42
        ),
    )
    assert np.isfinite(float(result["nw_se"]))
    assert (
        placebo.groupby("supplier_id").size().to_dict()
        == edges.groupby("supplier_id").size().to_dict()
    )
    assert (
        placebo.groupby("customer_id").size().to_dict()
        == edges.groupby("customer_id").size().to_dict()
    )

    assert not backtest.target_weights.empty
    filled = backtest.trades.loc[backtest.trades["status"].isin(["CLOSED", "OPEN"])]
    assert not filled.empty
    assert (filled["entry_date"] > filled["signal_date"]).all()
    closed = backtest.trades.loc[backtest.trades["status"].eq("CLOSED")]
    assert not closed.empty
    assert (closed["exit_date"] > closed["entry_date"]).all()
    assert np.isclose(
        backtest.trades["net_pnl"].sum(), backtest.daily_pnl["net_pnl"].sum()
    )
    assert np.isclose(
        backtest.trades["total_cost"].sum(), backtest.daily_pnl["total_cost"].sum()
    )
