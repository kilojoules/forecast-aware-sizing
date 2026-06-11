"""Real-settlement imbalance study (DK1 wind + battery).

paper_imbalance.py established a synthetic break-point: at constant
penalty lambda* ~ 100 EUR/MWh, single-forecast and ensemble policies
pick different b_E*. This script answers the follow-up that determines
whether the break-point matters in practice: under ACTUAL DK1
imbalance settlement prices (Energinet RegulatingBalancePowerdata),
which side of lambda* did 2021/2022/2023 sit, and do the two policies
pick different b_E* under real settlement?

Settlement variants applied post-hoc to the same hourly residual
r_t (+ = surplus delivered vs schedule) used in paper_imbalance.py:

  two-price (production BRP, conservative; DK regime pre-Nov-2021):
    cost_t = max(P_up,t - P_DA,t, 0) * max(-r_t, 0)
           + max(P_DA,t - P_down,t, 0) * max(r_t, 0)
    Never profitable; helpful imbalance settles at DA.

  one-price (DK regime post-Nov-2021):
    cost_t = r_t * (P_DA,t - P_imb,t)
    Can be negative (imbalance in the system-helping direction earns).

  constant-lambda (paper baseline, for reference):
    cost = lambda * sum |r_t|

Effective penalty per cell: lam_eff = total_cost / total |r|, directly
comparable to the synthetic lambda axis.

Output: results/imbalance/dk1_{year}_real.json
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

from dk_loader import load_dk_year, load_dk_resources, load_dk_balancing  # noqa: E402
from paper_imbalance import (  # noqa: E402
    B_E_GRID, B_P, WIND_SCALE_MW, LAGS_K4, DISC, CAPEX_E, CAPEX_P,
    build_persistence_forecasts, dispatch_chunked,
)


def settlement_stats(p_da, bal_up, bal_down, imb):
    """Year-level penalty distributions, EUR/MWh."""
    up_pen = np.maximum(bal_up - p_da, 0.0)     # cost of being short
    down_pen = np.maximum(p_da - bal_down, 0.0)  # cost of being long
    one_spread = np.abs(imb - p_da)
    def q(x):
        x = x[np.isfinite(x)]
        return {"mean": float(x.mean()), "median": float(np.median(x)),
                "p75": float(np.quantile(x, 0.75)),
                "p90": float(np.quantile(x, 0.90)),
                "p99": float(np.quantile(x, 0.99)),
                "frac_nonzero": float((x > 0.01).mean())}
    return {"up_penalty": q(up_pen), "down_penalty": q(down_pen),
            "oneprice_abs_spread": q(one_spread)}


def run_year(year: int):
    print(f"DK1 {year}: load price + wind + balancing")
    df_p = load_dk_year(year, area="DK1")
    df_r = load_dk_resources(year, area="DK1")
    df_b = load_dk_balancing(year, area="DK1")
    df = pd.merge(df_p, df_r, on="timestamp_utc", how="inner")
    df = pd.merge(df, df_b, on="timestamp_utc", how="inner")
    # Balancing gaps (rare): settle at DA, i.e. zero penalty that hour
    for c in ("bal_up_eur_per_mwh", "bal_down_eur_per_mwh", "imb_eur_per_mwh"):
        df[c] = df[c].fillna(df["da_eur_per_mwh"])
    p = df["da_eur_per_mwh"].to_numpy()
    w_raw = df["wind_mwh"].to_numpy()
    w = (w_raw / w_raw.max()) * WIND_SCALE_MW
    bal_up = df["bal_up_eur_per_mwh"].to_numpy()
    bal_down = df["bal_down_eur_per_mwh"].to_numpy()
    imb = df["imb_eur_per_mwh"].to_numpy()
    print(f"  N={len(df)} h after merge")

    max_lag = max(LAGS_K4)
    p_real, p_F = build_persistence_forecasts(p, LAGS_K4)
    w_real, w_F = build_persistence_forecasts(w, LAGS_K4)
    # Align settlement arrays with the max_lag trim
    bal_up = bal_up[max_lag:]
    bal_down = bal_down[max_lag:]
    imb = imb[max_lag:]
    p_da = p_real

    stats = settlement_stats(p_da, bal_up, bal_down, imb)
    print(f"  up-penalty mean={stats['up_penalty']['mean']:.1f} "
          f"p90={stats['up_penalty']['p90']:.1f}; "
          f"down-penalty mean={stats['down_penalty']['mean']:.1f} "
          f"p90={stats['down_penalty']['p90']:.1f} EUR/MWh")

    policies = {
        "single": (p_F[0], w_F[0]),
        "ensemble": (p_F.mean(axis=0), w_F.mean(axis=0)),
    }
    rows = []
    for b_E in B_E_GRID:
        t0 = time.time()
        for pol, (p_fcst, w_fcst) in policies.items():
            a_sched, residual = dispatch_chunked(p_real, p_fcst, w_real,
                                                  w_fcst, b_E, B_P)
            arb_rev = float(np.sum(p_real * a_sched))
            wind_da_rev = float(np.sum(p_real * w_fcst))
            short = np.maximum(-residual, 0.0)
            long_ = np.maximum(residual, 0.0)
            cost_2p = float(np.sum(np.maximum(bal_up - p_da, 0.0) * short
                                   + np.maximum(p_da - bal_down, 0.0) * long_))
            cost_1p = float(np.sum(residual * (p_da - imb)))
            imb_abs = float(np.sum(np.abs(residual)))
            row = {"b_E": b_E, "policy": pol, "arb_rev": arb_rev,
                   "wind_da_rev": wind_da_rev, "imb_abs": imb_abs,
                   "cost_2p": cost_2p, "cost_1p": cost_1p,
                   "lam_eff_2p": cost_2p / imb_abs if imb_abs > 0 else 0.0,
                   "lam_eff_1p": cost_1p / imb_abs if imb_abs > 0 else 0.0}
            for variant, cost in (("two_price", cost_2p), ("one_price", cost_1p)):
                op = arb_rev + wind_da_rev - cost
                row[f"npv_{variant}"] = DISC * op - CAPEX_E * b_E - CAPEX_P * B_P
            rows.append(row)
        print(f"  b_E={b_E:6.1f}  ({time.time()-t0:.1f}s)")

    # Argmax comparison per settlement variant
    summary = {}
    for variant in ("two_price", "one_price"):
        summary[variant] = {}
        for pol in ("single", "ensemble"):
            sub = [r for r in rows if r["policy"] == pol]
            best = max(sub, key=lambda r: r[f"npv_{variant}"])
            summary[variant][pol] = {"b_E_star": best["b_E"],
                                     "npv": best[f"npv_{variant}"],
                                     "lam_eff": best["lam_eff_2p" if variant == "two_price" else "lam_eff_1p"]}
        s, e = summary[variant]["single"], summary[variant]["ensemble"]
        summary[variant]["diverge"] = s["b_E_star"] != e["b_E_star"]
        print(f"  {variant}: single b_E*={s['b_E_star']} "
              f"ensemble b_E*={e['b_E_star']} diverge={summary[variant]['diverge']}")
    return rows, stats, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rows, stats, summary = run_year(args.year)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "meta": {"year": args.year, "source": "dk1_real_settlement",
                     "B_E": B_E_GRID, "B_P": B_P, "lags": list(LAGS_K4),
                     "wind_scale_MW": WIND_SCALE_MW, "DISC": DISC,
                     "CAPEX_E": CAPEX_E, "CAPEX_P": CAPEX_P},
            "settlement_stats": stats,
            "summary": summary,
            "rows": rows,
        }, f, indent=2)
    print(f"Wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
