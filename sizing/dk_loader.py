"""Energinet DK1 day-ahead spot price loader.

Pulls hourly DA spot prices for DK1 (West Denmark) from the public
Energinet data service. Free, no auth.

Persistence forecast at t-24h is constructed as
  forecast[t] = realized[t - 24]
which is the canonical baseline in price-forecasting literature. The
forecast error distribution is fat-tailed and regime-conditional.

Per PREREGISTRATION_AMEND_01.md (commit ebb035c), this replaces the
ERCOT North Hub setup that was originally pre-registered.

Cache layout:
  data/dk/{year}.csv   columns: timestamp_utc, da_eur_per_mwh
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


DATA_DIR = Path(__file__).parent / "data" / "dk"
ENERGINET_URL = "https://api.energidataservice.dk/dataset/Elspotprices"


def _fetch_year(year: int, area: str = "DK1") -> pd.DataFrame:
    """Pull all hourly DA prices for a single year. Energinet caps queries
    at ~10000 records per request, so we paginate by month."""
    rows = []
    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        if month == 12:
            end = f"{year + 1}-01-01"
        else:
            end = f"{year}-{month + 1:02d}-01"
        params = {
            "start": start,
            "end": end,
            "filter": f'{{"PriceArea":["{area}"]}}',
            "limit": 10000,
        }
        for attempt in range(3):
            try:
                r = requests.get(ENERGINET_URL, params=params, timeout=30)
                r.raise_for_status()
                payload = r.json()
                rows.extend(payload["records"])
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Failed {year}-{month}: {e}")
                time.sleep(2)
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["HourUTC"], utc=True)
    df = df.rename(columns={"SpotPriceEUR": "da_eur_per_mwh"})
    df = df[["timestamp_utc", "da_eur_per_mwh"]].sort_values("timestamp_utc")
    df = df.drop_duplicates(subset="timestamp_utc").reset_index(drop=True)
    return df


def load_dk_year(year: int, area: str = "DK1", refresh: bool = False) -> pd.DataFrame:
    """Load DA prices for a year, cached locally."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fp = DATA_DIR / f"{area}_{year}.csv"
    if fp.exists() and not refresh:
        return pd.read_csv(fp, parse_dates=["timestamp_utc"])
    df = _fetch_year(year, area=area)
    df.to_csv(fp, index=False)
    return df


def realized_and_persistence(year: int, area: str = "DK1"):
    """Return (realized, forecast) hourly arrays of shape (T,).

    realized[t] = DA[t]
    forecast[t] = DA[t - 24]   (persistence at -24h)
    First 24 hours are dropped to align indices.
    """
    df = load_dk_year(year, area=area)
    da = df["da_eur_per_mwh"].to_numpy()
    n = len(da)
    if n < 25:
        raise ValueError(f"too few hours ({n}) for {year}/{area}")
    realized = da[24:]
    forecast = da[:-24]
    return realized, forecast


def empirical_residuals(year: int, area: str = "DK1") -> np.ndarray:
    realized, forecast = realized_and_persistence(year, area)
    return realized - forecast


def ensemble_forecasts(realized: np.ndarray, base_forecast: np.ndarray,
                       residuals_pool: np.ndarray, K: int,
                       seed: int = 0) -> np.ndarray:
    """DEPRECATED: see PREREGISTRATION_AMEND_02.md.

    The original implementation perturbed `base_forecast` with iid samples
    from the empirical residual pool. That is a no-op information-wise
    (ensemble mean -> base_forecast as K grows; no Jensen lift). Use
    multi_lag_ensemble() instead.
    """
    rng = np.random.default_rng(seed)
    T = len(base_forecast)
    out = np.zeros((K, T))
    for k in range(K):
        idx = rng.integers(0, len(residuals_pool), size=T)
        out[k] = base_forecast + residuals_pool[idx]
    return out


def load_dk_resources(year: int, area: str = "DK1") -> pd.DataFrame:
    """Pull hourly DK1 wind + solar + gross consumption. Cached locally.

    Returns a DataFrame with timestamp_utc, wind_mwh, solar_mwh, load_mwh.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fp = DATA_DIR / f"{area}_resources_{year}.csv"
    if fp.exists():
        return pd.read_csv(fp, parse_dates=["timestamp_utc"])
    rows = []
    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        end = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"
        params = {
            "start": start, "end": end,
            "filter": f'{{"PriceArea":["{area}"]}}',
            "limit": 10000,
        }
        for attempt in range(3):
            try:
                r = requests.get(
                    "https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement",
                    params=params, timeout=30)
                r.raise_for_status()
                rows.extend(r.json()["records"])
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Failed {year}-{month}: {e}")
                time.sleep(2)
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["HourUTC"], utc=True)
    wind_cols = ["OffshoreWindLt100MW_MWh", "OffshoreWindGe100MW_MWh",
                 "OnshoreWindLt50kW_MWh", "OnshoreWindGe50kW_MWh"]
    solar_cols = ["SolarPowerLt10kW_MWh", "SolarPowerGe10Lt40kW_MWh",
                  "SolarPowerGe40kW_MWh", "SolarPowerSelfConMWh"]
    df["wind_mwh"] = df[wind_cols].fillna(0).sum(axis=1)
    df["solar_mwh"] = df[solar_cols].fillna(0).sum(axis=1)
    df["load_mwh"] = df["GrossConsumptionMWh"].fillna(0)
    df = df[["timestamp_utc", "wind_mwh", "solar_mwh", "load_mwh"]]
    df = df.sort_values("timestamp_utc").drop_duplicates(subset="timestamp_utc")
    df.to_csv(fp, index=False)
    return df.reset_index(drop=True)


def multi_lag_persistence(year: int, area: str = "DK1",
                          lags_hours: tuple[int, ...] = (24, 48, 168, 336)
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Multi-lag persistence ensemble. Returns (realized, forecasts_K).

    realized: shape (T,) realized DA prices, after the longest lag is
              dropped to align indices.
    forecasts_K: shape (K, T) where row k is the persistence forecast at
                 lag `lags_hours[k]`.

    Per AMEND_02: this is the canonical "K=4 ensemble" the paper uses.
    """
    df = load_dk_year(year, area=area)
    da = df["da_eur_per_mwh"].to_numpy()
    max_lag = max(lags_hours)
    K = len(lags_hours)
    T = len(da) - max_lag
    if T < 1:
        raise ValueError(f"too few hours ({len(da)}) for lags {lags_hours}")
    realized = da[max_lag:]
    forecasts = np.zeros((K, T))
    for k, lag in enumerate(lags_hours):
        forecasts[k] = da[max_lag - lag : len(da) - lag]
    return realized, forecasts


if __name__ == "__main__":
    # Smoke test: pull 2022 and report basic stats
    df = load_dk_year(2022)
    print(f"DK1 2022: {len(df)} hourly records")
    print(f"  range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
    print(f"  price stats EUR/MWh:")
    print(f"    mean   = {df['da_eur_per_mwh'].mean():.1f}")
    print(f"    std    = {df['da_eur_per_mwh'].std():.1f}")
    print(f"    min    = {df['da_eur_per_mwh'].min():.1f}")
    print(f"    max    = {df['da_eur_per_mwh'].max():.1f}")
    print(f"    p5     = {df['da_eur_per_mwh'].quantile(0.05):.1f}")
    print(f"    p95    = {df['da_eur_per_mwh'].quantile(0.95):.1f}")

    realized, forecast = realized_and_persistence(2022)
    res = realized - forecast
    print(f"\nPersistence (t-24h) residuals:")
    print(f"  mean   = {res.mean():.2f}")
    print(f"  std    = {res.std():.2f}")
    print(f"  min    = {res.min():.1f}")
    print(f"  max    = {res.max():.1f}")
    print(f"  q01    = {np.quantile(res, 0.01):.1f}")
    print(f"  q99    = {np.quantile(res, 0.99):.1f}")
    print(f"  kurtosis (excess) = {((res - res.mean())**4).mean() / (res.var()**2) - 3:.1f}")
