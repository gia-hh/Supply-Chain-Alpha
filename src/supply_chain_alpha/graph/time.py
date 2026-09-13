from __future__ import annotations

from typing import Literal

import pandas as pd

MARKET_TIMEZONE = "Asia/Shanghai"


def market_timestamp_utc(
    value: object,
    *,
    timezone: str = MARKET_TIMEZONE,
    errors: Literal["raise", "coerce"] = "raise",
) -> pd.Timestamp:
    """Interpret a timestamp on the China-market clock and return a UTC instant.

    Source timestamps with an explicit offset retain their instant.  Legacy
    timezone-naive values are interpreted as Shanghai local time rather than as
    UTC, which keeps disclosure cutoffs and calendar-year boundaries consistent.
    """
    if pd.isna(value):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone)
        return timestamp.tz_convert("UTC")
    except (TypeError, ValueError):
        if errors == "coerce":
            return pd.NaT
        raise


def market_instants_utc(
    values: pd.Series,
    *,
    timezone: str = MARKET_TIMEZONE,
    errors: Literal["raise", "coerce"] = "raise",
) -> pd.Series:
    """Vector form of :func:`market_timestamp_utc`, preserving the input index."""
    converted = values.map(
        lambda value: market_timestamp_utc(
            value,
            timezone=timezone,
            errors=errors,
        )
    )
    return pd.Series(
        pd.to_datetime(converted, errors=errors, utc=True),
        index=values.index,
        name=values.name,
    )


def market_dates_utc(
    values: pd.Series,
    *,
    timezone: str = MARKET_TIMEZONE,
    errors: Literal["raise", "coerce"] = "raise",
) -> pd.Series:
    """Convert market-calendar dates to their Shanghai-midnight UTC instants."""

    def convert(value: object) -> pd.Timestamp:
        if pd.isna(value):
            return pd.NaT
        try:
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(timezone)
            else:
                timestamp = timestamp.tz_convert(timezone)
            return timestamp.normalize().tz_convert("UTC")
        except (TypeError, ValueError):
            if errors == "coerce":
                return pd.NaT
            raise

    converted = values.map(convert)
    return pd.Series(
        pd.to_datetime(converted, errors=errors, utc=True),
        index=values.index,
        name=values.name,
    )


def market_year_start_utc(
    year: int,
    *,
    timezone: str = MARKET_TIMEZONE,
) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=1, day=1, tz=timezone).tz_convert("UTC")


def market_year_end_utc(
    year: int,
    *,
    timezone: str = MARKET_TIMEZONE,
) -> pd.Timestamp:
    return market_year_start_utc(int(year) + 1, timezone=timezone) - pd.Timedelta(
        nanoseconds=1
    )


def market_calendar_year(
    values: pd.Series,
    *,
    timezone: str = MARKET_TIMEZONE,
    errors: Literal["raise", "coerce"] = "raise",
) -> pd.Series:
    instants = market_instants_utc(values, timezone=timezone, errors=errors)
    return instants.dt.tz_convert(timezone).dt.year
