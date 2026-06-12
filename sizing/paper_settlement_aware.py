"""Referee items M5 + M6: settlement-aware dispatch and single-site wind.

M5 (settlement-blind dispatch): the imbalance extension's policies never
see lambda. A settlement-aware scheduler would reserve SoC headroom for
absorption. We implement the simplest version: schedule the arbitrage
LP inside a reserve band SoC in [rho*b_E, (1-rho)*b_E], rho in
{0, 0.1, 0.2, 0.3}; real-time absorption uses the full [0, b_E]. Per
lambda the best rho forms the settlement-aware envelope; question: does
the break-point lambda* move when both policies are allowed headroom?

M6 (aggregate wind too smooth): DK1 system wind scaled to W_peak is far
more persistence-forecastable than a single plant. We inflate forecast
errors: w_fcst_gamma = clip(w + gamma*(w_fcst - w), 0, inf), gamma in
{1, 2, 3} (gamma=2..3 spans published single-site/aggregate error
ratios); question: lambda*(gamma)?

Combos run: (gamma=1, rho in {0,.1,.2,.3}) + (gamma in {2,3}, rho=0).
Outputs are lambda-independent (arb_rev, wind_da_rev, imb_abs per
cell); the lambda sweep is post-processing.

GBAR usage (one job per year):
  python sizing/paper_settlement_aware.py --year 2022 \
      --out results/imbalance/aware_dk1_2022.json
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

from arbitrage_agents import lp_linear_actions  # noqa: E402
from dk_loader import load_dk_year, load_dk_resources  # noqa: E402
from paper_imbalance import (  # noqa: E402
    B_E_GRID, B_P, WIND_SCALE_MW, LAGS_K4, CHUNK_HOURS,
    build_persistence_forecasts, absorb_wind_error, _chunk_indices,
)

COMBOS = [(1.0, 0.0), (1.0, 0.1), (1.0, 0.2), (1.0, 0.3),
          (2.0, 0.0), (3.0, 0.0)]  # (gamma, rho)


def dispatch_chunked_reserve(prices_real, p_fcst, w_real, w_fcst,
                             b_E: float, b_P: float, rho: float):
    """Like paper_imbalance.dispatch_chunked, but the arbitrage schedule
    is confined to SoC in [rho*b_E, (1-rho)*b_E]; absorption uses the
    full band."""
    T = len(prices_real)
    a_sched_full = np.zeros(T)
    residual_full = np.zeros(T)
    lo = rho * b_E
    b_eff = max(b_E * (1.0 - 2.0 * rho), 1e-9)
    soc_carry = b_E / 2.0
    for s, e in _chunk_indices(T):
        soc0_eff = min(max(soc_carry - lo, 0.0), b_eff)
        a = lp_linear_actions(p_fcst[s:e], b_eff, b_P, soc0_eff, mu=0.0)
        a_sched_full[s:e] = a
        delta, res = absorb_wind_error(a, w_real[s:e], w_fcst[s:e],
                                        b_E, b_P, soc_carry)
        residual_full[s:e] = res
        soc_carry = soc_carry - float(np.sum(a + delta))
        soc_carry = max(0.0, min(b_E, soc_carry))
    return a_sched_full, residual_full


def run_year(year: int):
    print(f"DK1 {year}: load price + wind")
    df_p = load_dk_year(year, area="DK1")
    df_r = load_dk_resources(year, area="DK1")
    df = pd.merge(df_p, df_r, on="timestamp_utc", how="inner")
    p = df["da_eur_per_mwh"].to_numpy()
    w = (df["wind_mwh"].to_numpy() / df["wind_mwh"].max()) * WIND_SCALE_MW

    p_real, p_F = build_persistence_forecasts(p, LAGS_K4)
    w_real, w_F = build_persistence_forecasts(w, LAGS_K4)
    policies = {"single": (p_F[0], w_F[0]),
                "ensemble": (p_F.mean(axis=0), w_F.mean(axis=0))}

    rows = []
    for gamma, rho in COMBOS:
        for pol, (p_fcst, w_fcst_base) in policies.items():
            w_fcst = np.clip(w_real + gamma * (w_fcst_base - w_real),
                             0.0, None)
            err = float(np.abs(w_real - w_fcst).mean())
            for b_E in B_E_GRID:
                t0 = time.time()
                a_sched, residual = dispatch_chunked_reserve(
                    p_real, p_fcst, w_real, w_fcst, b_E, B_P, rho)
                rows.append({
                    "gamma": gamma, "rho": rho, "b_E": b_E, "policy": pol,
                    "arb_rev": float(np.sum(p_real * a_sched)),
                    "wind_da_rev": float(np.sum(p_real * w_fcst)),
                    "imb_abs": float(np.sum(np.abs(residual))),
                    "p_dot_r": float(np.sum(p_real * residual)),
                })
            print(f"  g={gamma} rho={rho} {pol:8s} |w_err|={err:.3f} "
                  f"({time.time()-t0:.1f}s last cell)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rows = run_year(args.year)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": {"year": args.year, "combos": COMBOS,
                            "B_E": B_E_GRID, "B_P": B_P,
                            "wind_scale_MW": WIND_SCALE_MW,
                            "lags": list(LAGS_K4),
                            "chunk_hours": CHUNK_HOURS},
                   "rows": rows}, f)
    print(f"Wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
