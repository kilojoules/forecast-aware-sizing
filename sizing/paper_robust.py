"""Max-min robust dispatch policy through the merchant sizing sweep.

The 2x2 factorial plans on a point trajectory (single lag-24h or
ensemble mean). For a linear objective, planning on the mean IS the
expectation-optimal stochastic plan, so the factorial spans forecast
quality but only one decision attitude. This script adds a genuinely
different decision rule: worst-case-robust dispatch (maximize the
minimum revenue across the K=4 multi-lag members), and asks the
paper's question of it: does b_E* move?

Same chunking, cycling cost, and evaluation (PriceEnv linear reward on
realized prices) as paper_benchmark.py; NPV form matches
paper_stress_figures.py.

Run (memory-capped):
  scripts/memrun.sh 1000 .pixi/envs/default/bin/python -u \
      sizing/paper_robust.py --source dk1 --year 2022 \
      --out results/main/robust_dk1_2022.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from arbitrage_agents import lp_maxmin_actions, run_actions
from dk_loader import multi_lag_persistence
from env import PriceEnv
from paper_benchmark import B_E_GRID, B_P, CHUNK_HOURS, MU_LP, chunk_indices, safe_D

DISC = sum(1.07 ** -y for y in range(15))
CAPEX_E = 100_000.0
CAPEX_P = 75_000.0


def dispatch_chunked_maxmin(realized: np.ndarray, forecasts_K: np.ndarray,
                            b_E: float, mu: float) -> dict:
    R_total = 0.0
    soc_full = []
    soc_carry = b_E / 2.0
    for s, e in chunk_indices(len(realized)):
        env = PriceEnv(b_E, B_P, prices=realized[s:e], soc0=soc_carry,
                       mu=mu, reward_mode="linear")
        a = lp_maxmin_actions(forecasts_K[:, s:e], b_E, B_P, soc_carry, mu=mu)
        rec = run_actions(env, a)
        R_total += rec["R"]
        soc_full.extend(rec["soc_log"][1:])
        soc_carry = rec["soc_log"][-1]
    soc_full = [b_E / 2.0] + soc_full
    return {"R": R_total, "D": safe_D(np.asarray(soc_full), b_E)}


def npv(R: float, b_E: float) -> float:
    return R * DISC - CAPEX_E * b_E - CAPEX_P * B_P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dk1", "ercot"], required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.source == "dk1":
        realized, forecasts_K = multi_lag_persistence(args.year)
    else:
        from ercot_loader import multi_lag_persistence_ercot
        realized, forecasts_K = multi_lag_persistence_ercot(args.year)

    rows = []
    for b_E in B_E_GRID:
        t0 = time.time()
        res = dispatch_chunked_maxmin(realized, forecasts_K, b_E, mu=MU_LP)
        rows.append({"b_E": b_E, "R": res["R"], "D": res["D"],
                     "npv": npv(res["R"], b_E)})
        print(f"  b_E={b_E:6.1f}  R={res['R']:12.0f}  ({time.time()-t0:.1f}s)")

    best = max(rows, key=lambda r: r["npv"])
    print(f"robust b_E* = {best['b_E']} (npv {best['npv']/1e6:.2f}M)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": {"source": args.source, "year": args.year,
                            "policy": "linear_maxmin", "mu": MU_LP,
                            "B_E": B_E_GRID, "K": forecasts_K.shape[0],
                            "chunk_hours": CHUNK_HOURS},
                   "rows": rows, "b_E_star": best["b_E"]}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
