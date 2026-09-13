from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from supply_chain_alpha.evaluation.statistics import (
    daily_information_coefficients,
    degree_preserving_random_graph,
    moving_block_bootstrap_mean,
    newey_west_mean_standard_error,
)


def test_hand_computed_rank_ic_uses_next_observed_trading_day() -> None:
    signals = pd.DataFrame(
        {
            "date": ["2020-01-03"] * 3,
            "security_id": ["A", "B", "C"],
            "customer_shock": [1.0, 2.0, 3.0],
        }
    )
    residuals = pd.DataFrame(
        {
            "date": ["2020-01-03"] * 3 + ["2020-01-06"] * 3,
            "security_id": ["A", "B", "C"] * 2,
            "residual_return": [9.0, -9.0, 0.0, -0.1, 0.0, 0.2],
        }
    )

    result = daily_information_coefficients(signals, residuals)

    assert result.loc[0, "forward_return_date"] == "2020-01-06"
    assert result.loc[0, "rank_ic"] == pytest.approx(1.0)


def test_missing_next_day_return_is_not_replaced_by_a_later_return() -> None:
    signals = pd.DataFrame(
        {
            "date": ["2020-01-03"] * 3,
            "security_id": ["A", "B", "C"],
            "customer_shock": [1.0, 2.0, 3.0],
        }
    )
    residuals = pd.DataFrame(
        {
            "date": ["2020-01-03"] * 3 + ["2020-01-06"] * 3 + ["2020-01-07"] * 3,
            "security_id": ["A", "B", "C"] * 3,
            "residual_return": [0.0, 0.0, 0.0, None, 0.1, 0.2, 9.0, 0.2, 0.3],
        }
    )

    result = daily_information_coefficients(signals, residuals)

    assert result.loc[0, "eligible_count"] == 2
    assert result.loc[0, "forward_return_date"] == "2020-01-06"


def test_bootstrap_seed_is_reproducible_and_newey_west_is_finite() -> None:
    first = moving_block_bootstrap_mean(
        [0.1, -0.2, 0.3, 0.4], replications=20, block_length=2, seed=42001
    )
    second = moving_block_bootstrap_mean(
        [0.1, -0.2, 0.3, 0.4], replications=20, block_length=2, seed=42001
    )
    pd.testing.assert_series_equal(first, second)
    assert newey_west_mean_standard_error([0.1, -0.2, 0.3, 0.4], lags=2) >= 0


def test_random_graph_placebo_preserves_directed_degrees_and_seed() -> None:
    edges = pd.DataFrame(
        {
            "supplier_id": ["S1", "S1", "S2", "S3", "S4", "S4"],
            "customer_id": ["C1", "C2", "C2", "C3", "C1", "C4"],
        }
    )
    first = degree_preserving_random_graph(edges, seed=43000)
    second = degree_preserving_random_graph(edges, seed=43000)

    pd.testing.assert_frame_equal(first, second)
    assert set(first.itertuples(index=False, name=None)) != set(
        edges.itertuples(index=False, name=None)
    )
    assert Counter(first["supplier_id"]) == Counter(edges["supplier_id"])
    assert Counter(first["customer_id"]) == Counter(edges["customer_id"])
    assert not first["supplier_id"].eq(first["customer_id"]).any()
    assert not first.duplicated().any()
