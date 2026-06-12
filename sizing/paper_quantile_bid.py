"""Newsvendor / quantile-optimal wind bidding for the ensemble policy (S2).

Stochastic-programming objection: under asymmetric imbalance penalties
the optimal DA wind bid is a quantile of the predictive distribution
(Pinson et al. 2007), not the mean; forcing both policies to bid the
point forecast measures the value of a better mean, not of
distributional information. Here the ensemble policy bids the per-hour
quantile q of its K=4 members, q in {0.25, 0.4, 0.5, 0.6, 0.75}; the
best q per (lambda, b_E) forms the quantile-optimal envelope. The
single-forecast policy has no distribution and is unchanged.

Outputs per (q, b_E): lambda-independent (arb_rev, wind_da_rev,
imb_abs) for the synthetic symmetric-lambda axis, plus real eSett
two-price / one-price settlement costs. Bands and argmaxes are
post-processing.

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python -u \
         sizing/paper_quantile_bid.py --year 2022 \
         --out results/imbalance/qbid_dk1_2022.json
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
    B_E_GRID, B_P, WIND_SCALE_MW, LAGS_K4,
    build_persistence_forecasts, dispatch_chunked,
)

QS = [0.25, 0.4, 0.5, 0.6, 0.75]


def run_year(year: int):
    print(f"DK1 {year}: load price + wind + balancing")
    df = pd.merge(load_dk_year(year, area="DK1"),
                  load_dk_resources(year, area="DK1"), on="timestamp_utc")
    df = pd.merge(df, load_dk_balancing(year, area="DK1"), on="timestamp_utc")
    for c in ("bal_up_eur_per_mwh", "bal_down_eur_per_mwh", "imb_eur_per_mwh"):
        df[c] = df[c].fillna(df["da_eur_per_mwh"])
    p = df["da_eur_per_mwh"].to_numpy()
    w = (df["wind_mwh"].to_numpy() / df["wind_mwh"].max()) * WIND_SCALE_MW

    max_lag = max(LAGS_K4)
    p_real, p_F = build_persistence_forecasts(p, LAGS_K4)
    w_real, w_F = build_persistence_forecasts(w, LAGS_K4)
    bal_up = df["bal_up_eur_per_mwh"].to_numpy()[max_lag:]
    bal_down = df["bal_down_eur_per_mwh"].to_numpy()[max_lag:]
    imb = df["imb_eur_per_mwh"].to_numpy()[max_lag:]
    p_ens = p_F.mean(axis=0)

    # bidders: single (lag-24, no distribution), ensemble-mean (baseline),
    # ensemble-quantile q (newsvendor candidates)
    bidders = {"single": (p_F[0], w_F[0]),
               "ens_mean": (p_ens, w_F.mean(axis=0))}
    for q in QS:
        bidders[f"ens_q{q:g}"] = (p_ens, np.quantile(w_F, q, axis=0))

    rows = []
    for name, (p_fcst, w_fcst) in bidders.items():
        t0 = time.time()
        for b_E in B_E_GRID:
            a_sched, residual = dispatch_chunked(p_real, p_fcst, w_real,
                                                  w_fcst, b_E, B_P)
            short = np.maximum(-residual, 0.0)
            long_ = np.maximum(residual, 0.0)
            rows.append({
                "bidder": name, "b_E": b_E,
                "arb_rev": float(np.sum(p_real * a_sched)),
                "wind_da_rev": float(np.sum(p_real * w_fcst)),
                "imb_abs": float(np.sum(np.abs(residual))),
                "p_dot_r": float(np.sum(p_real * residual)),
                "cost_2p": float(np.sum(
                    np.maximum(bal_up - p_real, 0.0) * short
                    + np.maximum(p_real - bal_down, 0.0) * long_)),
                "cost_1p": float(np.sum(residual * (p_real - imb))),
            })
        print(f"  {name:10s} ({time.time()-t0:.0f}s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = run_year(args.year)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": {"year": args.year, "quantiles": QS,
                            "B_E": B_E_GRID, "B_P": B_P,
                            "wind_scale_MW": WIND_SCALE_MW},
                   "rows": rows}, f)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
