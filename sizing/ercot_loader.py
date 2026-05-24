"""ERCOT North Hub day-ahead SPP loader via gridstatus.

Pre-registration commitment (originally PREREGISTRATION_ERCOT.md, then
amended to DK1 in AMEND_01 due to perceived data inaccessibility) is
re-activated here: the gridstatus 0.36 `get_dam_spp(year)` method does
return the full annual archive for free (verified empirically).

Multi-lag persistence ensemble construction is the same as DK1 per
PREREGISTRATION_AMEND_02.md (commit 0b5c9f1).

Cache layout:
  data/ercot/{year}_north.csv   columns: timestamp_utc, da_usd_per_mwh
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data" / "ercot"
HUB = "HB_NORTH"


def load_ercot_year(year: int, hub: str = HUB) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fp = DATA_DIR / f"{year}_{hub.lower()}.csv"
    if fp.exists():
        return pd.read_csv(fp, parse_dates=["timestamp_utc"])
    import gridstatus
    ercot = gridstatus.Ercot()
    raw = ercot.get_dam_spp(year, verbose=False)
    hub_df = raw[raw["Location"] == hub].copy()
    hub_df["timestamp_utc"] = pd.to_datetime(hub_df["Interval Start"], utc=True)
    hub_df = hub_df.rename(columns={"SPP": "da_usd_per_mwh"})
    hub_df = hub_df[["timestamp_utc", "da_usd_per_mwh"]]
    hub_df = hub_df.sort_values("timestamp_utc").drop_duplicates(subset="timestamp_utc")
    hub_df = hub_df.reset_index(drop=True)
    hub_df.to_csv(fp, index=False)
    return hub_df


def multi_lag_persistence_ercot(year: int, hub: str = HUB,
                                lags_hours: tuple[int, ...] = (24, 48, 168, 336)
                                ) -> tuple[np.ndarray, np.ndarray]:
    df = load_ercot_year(year, hub)
    da = df["da_usd_per_mwh"].to_numpy()
    max_lag = max(lags_hours)
    K = len(lags_hours)
    T = len(da) - max_lag
    realized = da[max_lag:]
    forecasts = np.zeros((K, T))
    for k, lag in enumerate(lags_hours):
        forecasts[k] = da[max_lag - lag : len(da) - lag]
    return realized, forecasts


if __name__ == "__main__":
    for year in [2021, 2022, 2023]:
        df = load_ercot_year(year)
        prices = df["da_usd_per_mwh"].to_numpy()
        realized, forecasts = multi_lag_persistence_ercot(year)
        res = realized - forecasts[0]
        kurt = ((res - res.mean()) ** 4).mean() / (res.var() ** 2) - 3
        print(f"ERCOT HB_NORTH {year}: T={len(prices)} h, "
              f"mean=${prices.mean():.1f} std=${prices.std():.1f} "
              f"min=${prices.min():.1f} max=${prices.max():.1f} "
              f"res_std=${res.std():.1f} kurt={kurt:.1f}")
