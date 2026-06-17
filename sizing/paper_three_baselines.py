"""The three co-located dispatch baselines (slide framework).

Implements, for the wind + battery HPP with imbalance settlement, the
three reference strategies and evaluates each over the b_E x lambda
grid so their optimal sizing can be compared:

  1. PERFECT FORESIGHT (upper bound).
     Bid = realised wind, dispatch on realised prices. Residual = 0,
     zero imbalance cost, full arbitrage. The co-located optimizer of
     paper_imbalance expanded with settlement, given clairvoyance.

  2. HONEST forecast-error baseline (single, ensemble).
     Trader bids forecast power at gate closure; real-time BESS
     corrects error at PoC. Residual r_t = (w_t - hat_w_t) + dAa_t,
     settled. Identical to paper_imbalance; reproduced here so all
     three baselines sit in one comparable table.

  3. ADVERSARIAL forecast-error baseline (worst-case / pessimist).
     Two flavours, both feeding a DETERMINISTIC worst-case input:
       adv_design  -- pessimistic, IMPLEMENTABLE (no peeking): bid the
                      per-hour lower bound of the plausible band
                      (min over the K ensemble members). Designs around
                      a lower bound on delivered power. This is the
                      "worst-case / most pessimist forecast, made a
                      deterministic input" candidate strategy.
       adv_stress  -- the slide's adversary: within a physical-
                      plausibility (SCADA) band [min_k, max_k], pick the
                      per-hour bid that MAXIMISES post-correction
                      imbalance volume (peeks at realised; an upper
                      bound on imbalance cost / lower bound on profit /
                      upper bound on TSO balancing procurement).

Operating profit uses the corrected settlement (residual valued at DA):
  op = arb_rev + wind_da_rev + p_dot_r - lambda * imb_abs.

Run (memory-capped):
  scripts/memrun.sh 1000 .pixi/envs/default/bin/python -u \
      sizing/paper_three_baselines.py --year 2022 --wind 5 \
      --out results/imbalance/baselines_dk1_2022_w5.json
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

from dk_loader import load_dk_year, load_dk_resources  # noqa: E402
from paper_imbalance import (  # noqa: E402
    B_E_GRID, B_P, WIND_SCALE_MW, LAGS_K4, LAMBDA_GRID, DISC,
    CAPEX_E, CAPEX_P, build_persistence_forecasts, dispatch_chunked,
)


def cell_metrics(prices_real, p_fcst, w_real, w_fcst, b_E, b_P):
    a_sched, residual = dispatch_chunked(prices_real, p_fcst, w_real,
                                         w_fcst, b_E, b_P)
    return {
        "arb_rev": float(np.sum(prices_real * a_sched)),
        "wind_da_rev": float(np.sum(prices_real * w_fcst)),
        "imb_abs": float(np.sum(np.abs(residual))),
        "p_dot_r": float(np.sum(prices_real * residual)),
    }


def adv_stress_bid(w_real, w_lo, w_hi):
    """Per-hour band endpoint farthest from realised (max pre-correction
    error). Adversary peeks at realised -> upper bound on imbalance."""
    d_lo = np.abs(w_lo - w_real)
    d_hi = np.abs(w_hi - w_real)
    return np.where(d_hi >= d_lo, w_hi, w_lo)


def run_year(year: int, wind_mw: float):
    df = pd.merge(load_dk_year(year, area="DK1"),
                  load_dk_resources(year, area="DK1"), on="timestamp_utc")
    p = df["da_eur_per_mwh"].to_numpy()
    w = (df["wind_mwh"].to_numpy() / df["wind_mwh"].max()) * wind_mw

    p_real, p_F = build_persistence_forecasts(p, LAGS_K4)
    w_real, w_F = build_persistence_forecasts(w, LAGS_K4)
    p_single, p_ens = p_F[0], p_F.mean(axis=0)
    w_lo, w_hi = w_F.min(axis=0), w_F.max(axis=0)

    # (label, price forecast, wind bid)
    bidders = {
        "perfect":    (p_real,   w_real),
        "single":     (p_single, w_F[0]),
        "ensemble":   (p_ens,    w_F.mean(axis=0)),
        "adv_design": (p_ens,    w_lo),                       # lower bound
        "adv_stress": (p_ens,    adv_stress_bid(w_real, w_lo, w_hi)),
    }
    print(f"DK1 {year} wind={wind_mw}MW: "
          f"|w_err| single={np.abs(w_real-w_F[0]).mean():.3f} "
          f"adv_design={np.abs(w_real-w_lo).mean():.3f} "
          f"adv_stress={np.abs(w_real-adv_stress_bid(w_real,w_lo,w_hi)).mean():.3f}")

    rows = []
    for b_E in B_E_GRID:
        t0 = time.time()
        cells = {name: cell_metrics(p_real, pf, w_real, wf, b_E, B_P)
                 for name, (pf, wf) in bidders.items()}
        for lam in LAMBDA_GRID:
            for name, c in cells.items():
                op = c["arb_rev"] + c["wind_da_rev"] + c["p_dot_r"] - lam * c["imb_abs"]
                rows.append({"b_E": b_E, "lambda": lam, "policy": name,
                             "npv": DISC * op - CAPEX_E * b_E - CAPEX_P * B_P,
                             **c})
        print(f"  b_E={b_E:6.1f} ({time.time()-t0:.1f}s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--wind", type=float, default=WIND_SCALE_MW)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = run_year(args.year, args.wind)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"year": args.year, "wind_scale_MW": args.wind,
                        "B_E": B_E_GRID, "lambda": LAMBDA_GRID, "B_P": B_P,
                        "baselines": ["perfect", "single", "ensemble",
                                      "adv_design", "adv_stress"]},
               "rows": rows}, open(args.out, "w"))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
