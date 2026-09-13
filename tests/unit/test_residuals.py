from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from supply_chain_alpha.factors.residuals import (
    build_residual_returns,
    compute_leave_one_out_industry_returns,
)


def _synthetic_market(days: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=days)
    market = np.array(
        [0.010, -0.004, 0.006, -0.008, 0.003, 0.012, -0.006, 0.005, 0.009, -0.002] * 2,
        dtype=float,
    )[:days]
    noise = np.array(
        [0.001, -0.002, 0.0005, 0.0015, -0.001, 0.002, -0.0005, 0.001, -0.0015, 0.0002]
        * 2,
        dtype=float,
    )[:days]
    panel = pd.DataFrame(
        {
            "date": dates,
            "security_id": "A",
            "raw_return": 0.001 + 1.5 * market + noise,
            "is_suspended": False,
        }
    )
    factors = pd.DataFrame({"date": dates, "market_return": market})
    return panel, factors


def test_leave_one_out_industry_factor_excludes_target() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2020-01-02"] * 3,
            "security_id": ["A", "B", "C"],
            "raw_return": [0.10, 0.20, 0.30],
        }
    )
    history = pd.DataFrame(
        {
            "security_id": ["A", "B", "C"],
            "industry_code": ["X", "X", "Y"],
            "valid_from": ["2019-01-01"] * 3,
            "valid_to": [None] * 3,
        }
    )

    result = compute_leave_one_out_industry_returns(panel, history).set_index(
        "security_id"
    )
    assert result.loc["A", "industry_return"] == pytest.approx(0.20)
    assert result.loc["B", "industry_return"] == pytest.approx(0.10)
    assert pd.isna(result.loc["C", "industry_return"])
    assert result.loc["A", "industry_peer_count"] == 1


def test_beta_window_ends_at_t_minus_one_and_ignores_day_t() -> None:
    panel, factors = _synthetic_market()
    baseline = build_residual_returns(
        panel,
        factors,
        beta_window=5,
        min_observations=5,
        information_lag_days=1,
    )
    target_date = factors.loc[10, "date"]
    target = baseline.loc[baseline["date"].eq(target_date)].iloc[0]
    expected_x = factors.loc[5:9, "market_return"].to_numpy()
    expected_y = panel.loc[5:9, "raw_return"].to_numpy()
    expected_beta = np.linalg.lstsq(
        np.column_stack([np.ones(5), expected_x]), expected_y, rcond=None
    )[0][1]
    assert target["beta_market"] == pytest.approx(expected_beta)
    assert target["beta_window_end"] == factors.loc[9, "date"]
    assert target["beta_observation_count"] == 5
    assert target["residual_model"] == "market_only"

    changed = panel.copy()
    changed.loc[10, "raw_return"] += 0.25
    rerun = build_residual_returns(
        changed,
        factors,
        beta_window=5,
        min_observations=5,
        information_lag_days=1,
    )
    changed_target = rerun.loc[rerun["date"].eq(target_date)].iloc[0]
    assert changed_target["beta_market"] == pytest.approx(target["beta_market"])
    assert changed_target["alpha"] == pytest.approx(target["alpha"])
    assert changed_target["residual_return"] - target[
        "residual_return"
    ] == pytest.approx(0.25)


def test_residuals_are_future_data_invariant() -> None:
    panel, factors = _synthetic_market()
    base = build_residual_returns(
        panel.iloc[:15],
        factors.iloc[:15],
        beta_window=6,
        min_observations=4,
        information_lag_days=1,
    )
    extended = build_residual_returns(
        panel,
        factors,
        beta_window=6,
        min_observations=4,
        information_lag_days=1,
    )
    historical = extended.loc[extended["date"] <= factors.loc[14, "date"]]
    pdt.assert_frame_equal(
        base.reset_index(drop=True), historical.reset_index(drop=True), check_exact=True
    )


def test_industry_history_is_point_in_time_and_market_only_is_fallback() -> None:
    panel_a, factors = _synthetic_market(12)
    panel_b = panel_a.copy()
    panel_b["security_id"] = "B"
    panel_b["raw_return"] = panel_b["raw_return"] * 0.7 + 0.0003
    panel = pd.concat([panel_a, panel_b], ignore_index=True)
    history = pd.DataFrame(
        [
            {
                "security_id": "A",
                "industry_code": "X",
                "valid_from": "2020-01-02",
                "valid_to": None,
            },
            {
                "security_id": "B",
                "industry_code": "X",
                "valid_from": "2020-01-20",
                "valid_to": None,
            },
        ]
    )

    result = build_residual_returns(
        panel,
        factors,
        beta_window=5,
        min_observations=4,
        information_lag_days=1,
        industry_history=history,
    )
    before_b_membership = result["date"] < pd.Timestamp("2020-01-20")
    assert result.loc[before_b_membership, "industry_return"].isna().all()
    assert result.loc[before_b_membership, "residual_model"].eq("market_only").all()


def test_available_pit_industry_model_is_used_and_future_membership_is_invariant() -> (
    None
):
    dates = pd.bdate_range("2020-01-02", periods=18)
    market_values = np.resize(
        np.array([0.008, -0.004, 0.006, -0.009, 0.003, 0.011, -0.005]),
        len(dates),
    )
    industry_shock = np.resize(
        np.array([-0.003, 0.007, -0.002, 0.005, -0.006, 0.004, 0.001]),
        len(dates),
    )
    rows: list[dict[str, object]] = []
    for offset, security_id in enumerate(("A", "B", "C")):
        raw = (
            0.0002 * offset
            + (1.0 + 0.1 * offset) * market_values
            + (0.7 + 0.05 * offset) * industry_shock
            + np.resize(np.array([0.0001, -0.0002, 0.0003]), len(dates))
        )
        rows.extend(
            {"date": day, "security_id": security_id, "raw_return": value}
            for day, value in zip(dates, raw, strict=True)
        )
    panel = pd.DataFrame(rows)
    factors = pd.DataFrame({"date": dates, "market_return": market_values})
    history = pd.DataFrame(
        {
            "security_id": ["A", "B", "C"],
            "industry_code": ["X", "X", "X"],
            "valid_from": [dates[0]] * 3,
            "valid_to": [None] * 3,
        }
    )
    base = build_residual_returns(
        panel,
        factors,
        beta_window=10,
        min_observations=8,
        information_lag_days=1,
        industry_history=history,
    )
    mature = base["date"] >= dates[10]
    assert base.loc[mature, "residual_model"].eq("market_plus_pit_industry").all()
    assert base.loc[mature, "beta_industry"].notna().all()

    future_membership = pd.concat(
        [
            history,
            pd.DataFrame(
                [
                    {
                        "security_id": "D",
                        "industry_code": "X",
                        "valid_from": dates[-1] + pd.offsets.BDay(1),
                        "valid_to": None,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    rerun = build_residual_returns(
        panel,
        factors,
        beta_window=10,
        min_observations=8,
        information_lag_days=1,
        industry_history=future_membership,
    )
    pdt.assert_frame_equal(base, rerun, check_exact=True)


def test_residual_parameter_validation_rejects_zero_lag() -> None:
    panel, factors = _synthetic_market(5)
    with pytest.raises(ValueError, match="at least one"):
        build_residual_returns(
            panel,
            factors,
            beta_window=3,
            min_observations=2,
            information_lag_days=0,
        )


def test_residuals_are_deterministic_under_input_reordering() -> None:
    panel, factors = _synthetic_market(12)
    first = build_residual_returns(
        panel,
        factors,
        beta_window=5,
        min_observations=4,
        information_lag_days=1,
    )
    reordered = build_residual_returns(
        panel.iloc[::-1].reset_index(drop=True),
        factors.iloc[::-1].reset_index(drop=True),
        beta_window=5,
        min_observations=4,
        information_lag_days=1,
    )
    pdt.assert_frame_equal(first, reordered, check_exact=True)
