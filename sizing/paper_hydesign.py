"""Sweep b_E with hydesign-merchant DEFAULT operational constraints.

Runs locally in topfarm-env (CPLEX). Compares two operating policies:
  - hydesign-default: DoD=0.9, batched 110-h chunks with terminal SoC=0.5*b_E
  - hydesign-relaxed: DoD=1.0, no terminal SoC (== our LP, sanity baseline)

For 6 (market, year) and each b_E in B_E_GRID. Output is saved to
`results_hydesign/<source>_<year>.json` with the same NPV-comparable
schema as `paper_<year>.json`.

Run as (from /tmp to avoid hydesign namespace shadowing):
    cd /tmp && conda run -n topfarm-env python \
      /Users/julianquick/portfolio_copy/battery_gym/sizing/paper_hydesign.py \
      --source dk1 --year 2022 \
      --out /Users/julianquick/portfolio_copy/battery_gym/results/hydesign/dk1_2022.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SIZING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIZING_DIR)
os.chdir("/tmp")  # avoid hydesign namespace shadowing

from hydesign_merchant_fork import ems_cplex_merchant  # noqa: E402
from dk_loader import load_dk_year  # noqa: E402
from ercot_loader import load_ercot_year  # noqa: E402

B_E_GRID = [0.5, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
B_P = 1.0
GRID_MW = 100.0  # large enough to be non-binding for merchant battery
DOD_DEFAULT = 0.9


def _build_inputs(prices: np.ndarray, b_E: float):
    N = len(prices)
    ts = pd.date_range("2022-01-01", periods=N, freq="h")
    wind_ts = pd.Series(np.zeros(N), index=ts)
    solar_ts = pd.Series(np.zeros(N), index=ts)
    price_ts = pd.Series(prices, index=ts)
    E_batt_ts = pd.Series(np.full(N, b_E), index=ts)
    return wind_ts, solar_ts, price_ts, E_batt_ts


def run_policy(prices, b_E, b_P, dod, enforce_terminal):
    wind_ts, solar_ts, price_ts, E_batt_ts = _build_inputs(prices, b_E)
    P_HPP, P_curt, P_chg_dis, E_SOC, _ = ems_cplex_merchant(
        wind_ts=wind_ts, solar_ts=solar_ts, price_ts=price_ts,
        P_batt_MW=b_P, E_batt_MWh_t=E_batt_ts,
        hpp_grid_connection=GRID_MW,
        battery_depth_of_discharge=dod, charge_efficiency=1.0,
        peak_hr_quantile=0.9, cost_of_battery_P_fluct_in_peak_price_ratio=0.0,
        n_full_power_hours_expected_per_day_at_peak_price=0,
        enforce_terminal_soc=enforce_terminal,
    )
    actions = np.asarray(P_chg_dis)
    R = float((actions * prices).sum())
    return {"R": R}


def run_year(source: str, year: int):
    if source == "dk1":
        df = load_dk_year(year, area="DK1")
        prices = df["da_eur_per_mwh"].to_numpy()
    else:
        df = load_ercot_year(year)
        prices = df["da_usd_per_mwh"].to_numpy()
    n_full_year = 8760
    if len(prices) > n_full_year:
        prices = prices[:n_full_year]
    print(f"  prices: {len(prices)} h, mean={prices.mean():.1f}")

    default_rows, relaxed_rows = [], []
    for b_E in B_E_GRID:
        t0 = time.time()
        d = run_policy(prices, b_E, B_P, dod=DOD_DEFAULT, enforce_terminal=True)
        r = run_policy(prices, b_E, B_P, dod=1.0, enforce_terminal=False)
        gap = (r["R"] - d["R"]) / max(abs(r["R"]), 1e-6)
        print(f"  b_E={b_E:6.1f}  default={d['R']:>10.0f}  relaxed={r['R']:>10.0f}  "
              f"gap={gap*100:+.2f}%  ({time.time()-t0:.0f}s)")
        default_rows.append({"b_E": b_E, "R": d["R"]})
        relaxed_rows.append({"b_E": b_E, "R": r["R"]})
    return {"hydesign_default": default_rows, "hydesign_relaxed": relaxed_rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["dk1", "ercot"], required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    print(f"Hydesign-merchant: {args.source} {args.year}")
    t0 = time.time()
    result = {
        "meta": {"source": args.source, "year": args.year,
                  "dod_default": DOD_DEFAULT,
                  "batch_size": 110},
        "by_year": {str(args.year): run_year(args.source, args.year)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
