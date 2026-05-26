"""
Data loader for NO2 day-ahead spot prices from ENTSO-E Transparency Platform.

This module is the single entry point for getting prices into a clean pandas
DataFrame the rest of the project can use. It does three things:

1. Fetches hourly day-ahead prices from ENTSO-E via the `entsoe-py` wrapper,
   one year at a time (ENTSO-E rate limits + URL length make multi-year
   single requests fragile).
2. Caches both the raw hourly series and a daily-aggregated version as
   parquet files under `data/processed/`, so notebooks don't re-hit the API.
3. Exposes a small set of convenience functions used by the notebooks.

Why ENTSO-E and not Nord Pool directly:
    Nord Pool's public site does not offer a documented free REST API; you'd
    have to scrape or use their commercial feed. ENTSO-E aggregates the same
    auction results (day-ahead spot in NO2 comes from Nord Pool, but it's
    re-published by the TSOs to ENTSO-E under regulatory requirement) and
    provides a clean REST API for free, once your account has API access.

Units returned by ENTSO-E for NO_2 day-ahead prices: EUR/MWh.
Timezone in the returned index: Europe/Oslo (local Norwegian time, handles DST).
"""

from __future__ import annotations

# Standard-library imports
import os
from pathlib import Path
from typing import Optional

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient


# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

# Project root is two levels up from this file: src/data_loader.py -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Where processed (cached) data lives. We don't cache anything under data/raw/
# because raw/ is reserved for manually downloaded files (kept out of git).
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

# File names for the cached parquet datasets
HOURLY_PARQUET: Path = PROCESSED_DIR / "no2_prices_hourly.parquet"
DAILY_PARQUET: Path = PROCESSED_DIR / "no2_prices_daily.parquet"
NVE_RESERVOIR_PARQUET: Path = PROCESSED_DIR / "nve_reservoir_no2.parquet"

# ENTSO-E area code for the NO2 bidding zone. `entsoe-py` uses underscore form.
NO2_AREA_CODE: str = "NO_2"

# NVE elspot-area number for NO2 (1=NO1, 2=NO2, ... 5=NO5).
NVE_NO2_OMRNR: int = 2

# Public NVE magasinstatistikk endpoint — no auth required, weekly resolution
# per elspot area going back to 1995.
NVE_API_URL: str = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"

# Norwegian local time. ENTSO-E will return data indexed in this timezone when
# we pass tz-aware Timestamps in this zone to the query.
TZ_OSLO: str = "Europe/Oslo"


# -----------------------------------------------------------------------------
# Token handling
# -----------------------------------------------------------------------------

def load_entsoe_token() -> str:
    """
    Load the ENTSO-E API token from the environment.

    Looks for an env var ``ENTSOE_API_TOKEN``. If a ``.env`` file exists at the
    project root, it is loaded first (via python-dotenv), so the env var can be
    set there during local development without exporting it shell-wide.

    Returns
    -------
    str
        The API token.

    Raises
    ------
    RuntimeError
        If no token is found. The error message points to .env.example.
    """
    # Load variables from .env into os.environ. `override=False` means a value
    # already set in the real shell environment wins — useful for CI later.
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    token = os.environ.get("ENTSOE_API_TOKEN", "").strip()
    if not token or token == "paste-your-token-here":
        raise RuntimeError(
            "ENTSOE_API_TOKEN is not set. Copy .env.example to .env and "
            "paste your token from https://transparency.entsoe.eu "
            "(My Account Settings → Web API Security Token)."
        )
    return token


# -----------------------------------------------------------------------------
# Fetching from ENTSO-E
# -----------------------------------------------------------------------------

def _year_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Break [start, end) into ~1-year chunks aligned to calendar years.

    ENTSO-E enforces a maximum lookback / response size on most endpoints.
    Day-ahead prices for a single zone fit comfortably in a year, so we use
    that as our chunk size. Aligning to calendar years also makes caching
    and partial re-fetches more intuitive later if we want to add them.

    Parameters
    ----------
    start, end : pd.Timestamp
        Timezone-aware bounds. ``end`` is exclusive in the same sense ENTSO-E
        treats it: a request for [2020-01-01, 2021-01-01) returns all of 2020.

    Returns
    -------
    list of (chunk_start, chunk_end) tuples covering the full range.
    """
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start
    while current < end:
        # Jump to the start of next calendar year, then take the min with `end`
        # so the last chunk doesn't run past the requested range.
        next_year_start = pd.Timestamp(
            year=current.year + 1, month=1, day=1, tz=current.tz
        )
        chunk_end = min(next_year_start, end)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


def fetch_no2_day_ahead_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    token: Optional[str] = None,
) -> pd.Series:
    """
    Fetch hourly NO2 day-ahead spot prices from ENTSO-E.

    Parameters
    ----------
    start : pd.Timestamp
        Inclusive lower bound. Must be timezone-aware (typically Europe/Oslo).
    end : pd.Timestamp
        Exclusive upper bound, same tz as ``start``.
    token : str, optional
        ENTSO-E API token. If None, read from environment via
        :func:`load_entsoe_token`.

    Returns
    -------
    pd.Series
        Hourly EUR/MWh prices indexed by tz-aware DatetimeIndex (Europe/Oslo).
        Series name is ``"price_eur_mwh"`` for readability downstream.
    """
    if token is None:
        token = load_entsoe_token()

    client = EntsoePandasClient(api_key=token)

    # Query one year at a time and concatenate. We log a brief line per chunk
    # so the user sees progress when this runs in a notebook cell.
    pieces: list[pd.Series] = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        print(f"  Fetching {chunk_start.date()} → {chunk_end.date()} ...", flush=True)
        # `query_day_ahead_prices` returns a tz-aware Series in the local tz
        # of the area (Europe/Oslo for NO_2), in EUR/MWh, hourly resolution.
        piece = client.query_day_ahead_prices(
            country_code=NO2_AREA_CODE,
            start=chunk_start,
            end=chunk_end,
        )
        pieces.append(piece)

    # Concatenate; sort just in case (chunks should already be in order) and
    # drop duplicates that can occur at year boundaries.
    series = pd.concat(pieces).sort_index()
    series = series[~series.index.duplicated(keep="first")]
    series.name = "price_eur_mwh"
    return series


# -----------------------------------------------------------------------------
# Aggregation and caching
# -----------------------------------------------------------------------------

def to_daily_mean(hourly: pd.Series) -> pd.DataFrame:
    """
    Aggregate an hourly price series to a daily-mean DataFrame.

    The Schwartz 1-factor model works on daily resolution; intra-day pattern
    is not interesting for valuing a reservoir that we're treating as making
    one go/no-go decision per day.

    Parameters
    ----------
    hourly : pd.Series
        Hourly EUR/MWh series with tz-aware DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (date, not Timestamp), ``price_eur_mwh`` (daily mean),
        ``n_hours`` (count of observations contributing to that day — useful
        for sanity-checking DST days, which have 23 or 25 hours).
    """
    # Resample to daily frequency in the series' own timezone. Using "D" with
    # a tz-aware index respects local midnight boundaries (important: a
    # "Norwegian day" runs midnight-to-midnight Oslo time, not UTC).
    daily_mean = hourly.resample("D").mean()
    daily_count = hourly.resample("D").count()

    df = pd.DataFrame({
        "date": daily_mean.index.date,
        "price_eur_mwh": daily_mean.values,
        "n_hours": daily_count.values,
    })
    # Drop any days where ENTSO-E returned no observations at all (very rare,
    # but possible at the bleeding edge of the most recent data).
    df = df[df["n_hours"] > 0].reset_index(drop=True)
    return df


def cache_prices(hourly: pd.Series, daily: pd.DataFrame) -> None:
    """Write both hourly and daily datasets to parquet under data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Hourly: write as a DataFrame so timezone info survives a round-trip
    # cleanly (parquet doesn't store Series names + tz combinations as nicely).
    hourly.to_frame().to_parquet(HOURLY_PARQUET)
    daily.to_parquet(DAILY_PARQUET)
    print(f"Wrote {HOURLY_PARQUET.relative_to(PROJECT_ROOT)} ({len(hourly):,} rows)")
    print(f"Wrote {DAILY_PARQUET.relative_to(PROJECT_ROOT)} ({len(daily):,} rows)")


def load_cached_daily() -> pd.DataFrame:
    """Read the daily-mean parquet. Raises FileNotFoundError if not yet cached."""
    if not DAILY_PARQUET.exists():
        raise FileNotFoundError(
            f"No cached daily data at {DAILY_PARQUET}. "
            "Run data_loader.py as a script first, or call refresh_cache()."
        )
    return pd.read_parquet(DAILY_PARQUET)


def load_cached_hourly() -> pd.Series:
    """Read the hourly parquet back to a tz-aware Series."""
    if not HOURLY_PARQUET.exists():
        raise FileNotFoundError(
            f"No cached hourly data at {HOURLY_PARQUET}. "
            "Run data_loader.py as a script first, or call refresh_cache()."
        )
    df = pd.read_parquet(HOURLY_PARQUET)
    # We wrote a single-column DataFrame; restore the Series.
    return df.iloc[:, 0].rename("price_eur_mwh")


def refresh_cache(start_year: int = 2020, end_year: Optional[int] = None) -> None:
    """
    End-to-end: fetch fresh data, aggregate to daily, write parquet caches.

    Parameters
    ----------
    start_year : int, default 2020
        First calendar year to include.
    end_year : int, optional
        Exclusive upper bound. Defaults to next calendar year (so "today" is
        always included).
    """
    if end_year is None:
        end_year = pd.Timestamp.now(tz=TZ_OSLO).year + 1

    start = pd.Timestamp(year=start_year, month=1, day=1, tz=TZ_OSLO)
    end = pd.Timestamp(year=end_year, month=1, day=1, tz=TZ_OSLO)

    print(f"Fetching NO2 day-ahead prices for {start_year}–{end_year - 1} ...")
    hourly = fetch_no2_day_ahead_prices(start, end)
    daily = to_daily_mean(hourly)
    cache_prices(hourly, daily)


# -----------------------------------------------------------------------------
# NVE magasinstatistikk (reservoir filling per elspot area)
# -----------------------------------------------------------------------------
#
# Used as *context* only — not as input to the option model in v1. The
# correlation between reservoir filling and spot price is one of the most
# basic stylized facts of the Norwegian power market, so we plot them
# alongside in notebook 01 to motivate the v2 extension that would model
# inflow as stochastic.

def fetch_nve_reservoir(
    area_no: int = NVE_NO2_OMRNR,
    start_year: int = 2018,
) -> pd.DataFrame:
    """
    Fetch weekly reservoir filling for one Norwegian elspot area from NVE.

    Hits the public NVE Magasinstatistikk API (no auth). The endpoint returns
    a single JSON payload with all areas; we filter to ``omrType == 'EL'``
    (= elspot areas) and the requested area number.

    Parameters
    ----------
    area_no : int, default 2
        NVE elspot area number. 1 = NO1, 2 = NO2, ..., 5 = NO5.
    start_year : int, default 2018
        Filter to records on or after January 1 of this year. NVE has data
        back to 1995, but for the project context (NO2 prices 2020-now) we
        want roughly the same window.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime), iso_year, iso_week, fyllingsgrad (0-1),
        kapasitet_twh, fylling_twh, endring_fyllingsgrad (week-over-week
        change in filling ratio). Sorted by date, indexed by integer.
    """
    import requests
    response = requests.get(NVE_API_URL, timeout=30)
    response.raise_for_status()
    records = response.json()

    df = pd.DataFrame(records)
    # Keep only elspot areas (omrType='EL') for the requested area number.
    df = df[(df["omrType"] == "EL") & (df["omrnr"] == area_no)].copy()
    df["date"] = pd.to_datetime(df["dato_Id"])
    df = df[df["date"] >= pd.Timestamp(start_year, 1, 1)].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Trim and rename to a tidy schema. The original camel/snake mix is a bit
    # ugly; we standardise on snake_case English-flavoured field names.
    out = pd.DataFrame({
        "date": df["date"],
        "iso_year": df["iso_aar"].astype(int),
        "iso_week": df["iso_uke"].astype(int),
        "fyllingsgrad": df["fyllingsgrad"].astype(float),
        "kapasitet_twh": df["kapasitet_TWh"].astype(float),
        "fylling_twh": df["fylling_TWh"].astype(float),
        "endring_fyllingsgrad": df["endring_fyllingsgrad"].astype(float),
    })
    return out


def cache_nve_reservoir(df: pd.DataFrame) -> None:
    """Write the NVE reservoir DataFrame to parquet under data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(NVE_RESERVOIR_PARQUET)
    print(f"Wrote {NVE_RESERVOIR_PARQUET.relative_to(PROJECT_ROOT)} ({len(df):,} rows)")


def load_cached_nve_reservoir() -> pd.DataFrame:
    """Read the NVE reservoir parquet. Raises FileNotFoundError if not cached."""
    if not NVE_RESERVOIR_PARQUET.exists():
        raise FileNotFoundError(
            f"No cached NVE data at {NVE_RESERVOIR_PARQUET}. "
            "Run `refresh_nve_cache()` or `uv run python -m src.data_loader` first."
        )
    return pd.read_parquet(NVE_RESERVOIR_PARQUET)


def refresh_nve_cache(area_no: int = NVE_NO2_OMRNR, start_year: int = 2018) -> None:
    """Fetch + cache NVE reservoir data for one area."""
    print(f"Fetching NVE reservoir filling for area {area_no} (from {start_year})...")
    df = fetch_nve_reservoir(area_no=area_no, start_year=start_year)
    cache_nve_reservoir(df)


# -----------------------------------------------------------------------------
# Script entry point — `uv run python -m src.data_loader` does a full refresh
# of both ENTSO-E prices and NVE reservoir data.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    refresh_cache()
    refresh_nve_cache()
