from __future__ import annotations

import pandas as pd
import pytest

from supply_chain_alpha.signals.diffusion import build_diffusion_signals


def _edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "supplier_id": ["S1", "S1", "S1", "S2"],
            "customer_id": ["C1", "C2", "C2", "C2"],
            "effective_start": [
                "2020-01-02T07:00:00Z",
                "2020-01-03T08:00:01Z",
                "2020-01-03T08:00:01Z",
                "2020-01-02T07:00:00Z",
            ],
            "effective_end": [None, None, None, "2020-01-04T07:00:00Z"],
        }
    )


def _residuals() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in ("2020-01-02", "2020-01-03", "2020-01-06"):
        for security_id, value in {
            "S1": 0.01,
            "S2": 0.03,
            "C1": 0.02,
            "C2": 0.04,
        }.items():
            rows.append(
                {"date": day, "security_id": security_id, "residual_return": value}
            )
    return pd.DataFrame(rows)


def test_hand_computed_direction_timing_and_duplicate_neighbor_collapse() -> None:
    signals = build_diffusion_signals(_edges(), _residuals())
    keyed = signals.set_index(["date", "security_id"])

    assert keyed.loc[("2020-01-02", "S1"), "customer_shock"] == pytest.approx(0.02)
    assert keyed.loc[("2020-01-03", "S1"), "customer_shock"] == pytest.approx(0.02)
    assert keyed.loc[("2020-01-03", "S1"), "customer_neighbor_count"] == 1
    assert keyed.loc[("2020-01-03", "C2"), "supplier_shock"] == pytest.approx(0.03)
    assert keyed.loc[("2020-01-06", "S1"), "customer_shock"] == pytest.approx(0.03)
    assert keyed.loc[("2020-01-06", "S1"), "customer_neighbor_count"] == 2
    assert keyed.loc[("2020-01-06", "C2"), "supplier_shock"] == pytest.approx(0.01)


def test_missing_neighbor_is_excluded_and_never_zero_filled() -> None:
    residuals = _residuals()
    residuals.loc[
        residuals["security_id"].eq("C2") & residuals["date"].eq("2020-01-03"),
        "residual_return",
    ] = None
    signals = build_diffusion_signals(_edges(), residuals)
    row = signals.set_index(["date", "security_id"]).loc[("2020-01-03", "S1")]
    assert row["customer_shock"] == pytest.approx(0.02)
    assert row["customer_neighbor_count"] == 1


def test_future_rows_and_edges_do_not_change_historical_signals() -> None:
    original = build_diffusion_signals(_edges(), _residuals())
    future_returns = pd.concat(
        [
            _residuals(),
            pd.DataFrame(
                {
                    "date": ["2021-01-04", "2021-01-04"],
                    "security_id": ["S1", "C3"],
                    "residual_return": [0.9, -0.9],
                }
            ),
        ],
        ignore_index=True,
    )
    future_edges = pd.concat(
        [
            _edges(),
            pd.DataFrame(
                {
                    "supplier_id": ["S1"],
                    "customer_id": ["C3"],
                    "effective_start": ["2021-01-04T07:00:00Z"],
                    "effective_end": [None],
                }
            ),
        ],
        ignore_index=True,
    )
    rerun = build_diffusion_signals(future_edges, future_returns)
    historical = rerun.loc[rerun["date"].le("2020-01-06")].reset_index(drop=True)
    pd.testing.assert_frame_equal(original, historical)


def test_self_neighbor_is_rejected() -> None:
    edge = _edges().iloc[[0]].copy()
    edge.loc[:, "customer_id"] = "S1"
    with pytest.raises(ValueError, match="self-neighbor"):
        build_diffusion_signals(edge, _residuals())
