from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from supply_chain_alpha.data.market import build_market_panel


def _master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": "A",
                "listing_date": "2020-01-02",
                "delisting_date": None,
                "board": "MAIN",
            },
            {
                "security_id": "B",
                "listing_date": "2020-01-06",
                "delisting_date": None,
                "board": "MAIN",
            },
        ]
    )


def _rules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "valid_from": "2020-01-01",
                "valid_to": "2020-01-05",
                "board": "MAIN",
                "ordinary_pct": 0.10,
                "st_pct": 0.05,
            },
            {
                "valid_from": "2020-01-06",
                "valid_to": "2020-12-31",
                "board": "MAIN",
                "ordinary_pct": 0.20,
                "st_pct": 0.05,
            },
        ]
    )


def _quote(day: str, close: float, **extra: object) -> dict[str, object]:
    return {
        "date": day,
        "security_id": "A",
        "open": extra.pop("open", close),
        "high": extra.pop("high", close),
        "low": extra.pop("low", close),
        "close": close,
        "volume": extra.pop("volume", 100.0),
        "amount": extra.pop("amount", 10_000.0),
        **extra,
    }


def test_market_panel_enforces_listing_calendar_and_historical_rules() -> None:
    calendar = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    prices = pd.DataFrame(
        [
            _quote("2020-01-02", 100.0),
            _quote("2020-01-03", 110.0),
            _quote(
                "2020-01-06",
                110.0,
                volume=0.0,
                amount=0.0,
                is_suspended=True,
            ),
            _quote("2020-01-07", 132.0),
        ]
    )

    result = build_market_panel(prices, _master(), calendar, _rules())
    a = result.loc[result["security_id"].eq("A")].set_index("date")

    assert a.loc[pd.Timestamp("2020-01-03"), "price_limit_state"] == "limit_up"
    assert not a.loc[pd.Timestamp("2020-01-03"), "is_tradable"]
    assert a.loc[pd.Timestamp("2020-01-06"), "price_limit_state"] == "suspended"
    assert pd.isna(a.loc[pd.Timestamp("2020-01-06"), "raw_return"])
    assert a.loc[pd.Timestamp("2020-01-07"), "price_limit_state"] == "limit_up"
    assert a.loc[pd.Timestamp("2020-01-07"), "price_limit_pct"] == pytest.approx(0.20)
    assert a["listing_age"].tolist() == [0, 1, 2, 3]

    b_dates = result.loc[result["security_id"].eq("B"), "date"].tolist()
    assert b_dates == [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-07")]


def test_market_panel_rejects_noncalendar_and_outside_listing_quotes() -> None:
    with pytest.raises(ValueError, match="non-trading-calendar"):
        build_market_panel(
            pd.DataFrame([_quote("2020-01-04", 100.0)]),
            _master(),
            ["2020-01-02", "2020-01-03"],
            _rules(),
        )

    before_listing = _quote("2020-01-02", 100.0)
    before_listing["security_id"] = "B"
    with pytest.raises(ValueError, match="outside listing intervals"):
        build_market_panel(
            pd.DataFrame([before_listing]),
            _master(),
            ["2020-01-02", "2020-01-03", "2020-01-06"],
            _rules(),
        )


def test_missing_trading_day_is_not_filled_or_bridged() -> None:
    result = build_market_panel(
        pd.DataFrame(
            [
                _quote("2020-01-02", 100.0),
                _quote("2020-01-06", 101.0),
            ]
        ),
        _master().iloc[[0]],
        ["2020-01-02", "2020-01-03", "2020-01-06"],
        _rules(),
    ).set_index("date")

    assert not result.loc[pd.Timestamp("2020-01-03"), "has_market_data"]
    assert not result.loc[pd.Timestamp("2020-01-03"), "is_tradable"]
    assert pd.isna(result.loc[pd.Timestamp("2020-01-06"), "previous_close"])
    assert pd.isna(result.loc[pd.Timestamp("2020-01-06"), "raw_return"])


def test_corporate_actions_are_local_and_future_invariant() -> None:
    master = _master().iloc[[0]]
    rules = pd.DataFrame(
        [
            {
                "valid_from": "2020-01-01",
                "valid_to": "2020-12-31",
                "board": "MAIN",
                "ordinary_pct": 0.90,
                "st_pct": 0.05,
            }
        ]
    )
    base_prices = pd.DataFrame(
        [
            _quote("2020-01-02", 100.0),
            _quote("2020-01-03", 50.0, open=49.0, high=51.0, low=49.0),
            _quote("2020-01-06", 51.0, open=50.0, high=52.0, low=50.0),
        ]
    )
    base_actions = pd.DataFrame(
        [
            {
                "date": "2020-01-03",
                "security_id": "A",
                "share_multiplier": 2.0,
                "cash_dividend": 0.0,
            }
        ]
    )
    base = build_market_panel(
        base_prices,
        master,
        ["2020-01-02", "2020-01-03", "2020-01-06"],
        rules,
        base_actions,
    )
    assert base.loc[base["date"].eq("2020-01-03"), "raw_return"].iat[
        0
    ] == pytest.approx(0.0)

    future_prices = pd.concat(
        [base_prices, pd.DataFrame([_quote("2020-01-07", 26.0)])],
        ignore_index=True,
    )
    future_actions = pd.concat(
        [
            base_actions,
            pd.DataFrame(
                [
                    {
                        "date": "2020-01-07",
                        "security_id": "A",
                        "share_multiplier": 2.0,
                        "cash_dividend": 0.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    extended = build_market_panel(
        future_prices,
        master,
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
        rules,
        future_actions,
    )
    historical = extended.loc[extended["date"] <= pd.Timestamp("2020-01-06")]
    pdt.assert_frame_equal(
        base.reset_index(drop=True), historical.reset_index(drop=True), check_exact=True
    )


def test_market_panel_is_deterministic_under_input_reordering() -> None:
    prices = pd.DataFrame(
        [
            _quote("2020-01-02", 100.0),
            _quote("2020-01-03", 105.0, open=101.0, high=106.0, low=100.0),
            _quote("2020-01-06", 106.0, open=105.0, high=107.0, low=104.0),
        ]
    )
    calendar = ["2020-01-02", "2020-01-03", "2020-01-06"]
    first = build_market_panel(prices, _master(), calendar, _rules())
    reordered = build_market_panel(
        prices.iloc[::-1].reset_index(drop=True),
        _master().iloc[::-1].reset_index(drop=True),
        list(reversed(calendar)),
        _rules().iloc[::-1].reset_index(drop=True),
    )
    pdt.assert_frame_equal(first, reordered, check_exact=True)
