"""Single (b_E, b_P) array task for LUMI 2-D sweep.

Per AMEND_03 (committed alongside this code).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from arbitrage_agents import (lp_linear_actions, qp_ensemble_actions,
                              qp_quadratic_actions, run_actions)
from degradation import cycle_degradation
from dk_loader import multi_lag_persistence
from env import PriceEnv
from ercot_loader import multi_lag_persistence_ercot


B_E_GRID = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
B_P_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
ALPHA_QP = 0.005
MU_LP = 5.0
CHUNK_HOURS = 24 * 7 * 8


def safe_D(soc, b):
    try:
        D, _, _ = cycle_degradation(soc, b)
        return D
    except Exception:
        return 0.0


def chunks(T, c=CHUNK_HOURS):
    s = 0
    while s < T:
        yield s, min(s + c, T)
        s += c


def dispatch(realized, fc, b_E, b_P, cost, mu_or_alpha, ensemble=False):
    R_total = 0.0
    soc_full = []
    soc_carry = b_E / 2
    for s, e in chunks(len(realized) if not ensemble else realized.shape[0]):
        r_chunk = realized[s:e] if not ensemble else realized[s:e]
        env = PriceEnv(b_E, b_P, prices=r_chunk, soc0=soc_carry,
                       alpha=mu_or_alpha if cost == "quadratic" else 0,
                       mu=mu_or_alpha if cost == "linear" else 0,
                       reward_mode=cost)
        if ensemble:
            f = fc[:, s:e]
            if cost == "linear":
                a = lp_linear_actions(f.mean(axis=0), b_E, b_P, soc_carry, mu=mu_or_alpha)
            else:
                a = qp_ensemble_actions(f, b_E, b_P, soc_carry, alpha=mu_or_alpha)
        else:
            f = fc[s:e]
            if cost == "linear":
                a = lp_linear_actions(f, b_E, b_P, soc_carry, mu=mu_or_alpha)
            else:
                a = qp_quadratic_actions(f, b_E, b_P, soc_carry, alpha=mu_or_alpha)
        rec = run_actions(env, a)
        R_total += rec["R"]
        soc_full.extend(rec["soc_log"][1:])
        soc_carry = rec["soc_log"][-1]
    soc_full = [b_E / 2] + soc_full
    return R_total, safe_D(np.asarray(soc_full), b_E)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--out_dir",
                   default=str(Path(__file__).resolve().parent.parent / "results" / "2d"))
    args = p.parse_args()

    n_be = len(B_E_GRID)
    n_bp = len(B_P_GRID)
    n_total = n_be * n_bp
    if args.task_id >= n_total:
        print(f"task {args.task_id} out of range (max {n_total - 1})")
        return

    bp_i = args.task_id // n_be
    be_i = args.task_id % n_be
    b_P = B_P_GRID[bp_i]
    b_E = B_E_GRID[be_i]
    os.makedirs(args.out_dir, exist_ok=True)
    out_fp = Path(args.out_dir) / f"2d_bP{b_P}_bE{b_E}.json"
    if out_fp.exists():
        print(f"already done: {out_fp}")
        return

    print(f"Task {args.task_id}: b_E={b_E} b_P={b_P}")
    t0 = time.time()
    result = {"b_E": b_E, "b_P": b_P, "by_market": {}}

    for source in ["dk1", "ercot"]:
        result["by_market"][source] = {}
        for year in [2021, 2022, 2023]:
            try:
                if source == "dk1":
                    realized, fc_K = multi_lag_persistence(year, lags_hours=(24, 48, 168, 336))
                else:
                    realized, fc_K = multi_lag_persistence_ercot(year, lags_hours=(24, 48, 168, 336))
            except Exception as e:
                print(f"  skip {source} {year}: {e}")
                continue
            fc_single = fc_K[0]
            year_out = {}
            for cost, m in [("linear", MU_LP), ("quadratic", ALPHA_QP)]:
                Rs, Ds = dispatch(realized, fc_single, b_E, b_P, cost, m, ensemble=False)
                Re, De = dispatch(realized, fc_K, b_E, b_P, cost, m, ensemble=True)
                year_out[f"{cost}_single"] = {"R": Rs, "D": Ds}
                year_out[f"{cost}_ensemble"] = {"R": Re, "D": De}
            result["by_market"][source][str(year)] = year_out
            print(f"  {source} {year}: linear-s R={year_out['linear_single']['R']:.0f}  "
                  f"linear-e R={year_out['linear_ensemble']['R']:.0f}  "
                  f"qp-s R={year_out['quadratic_single']['R']:.0f}  "
                  f"qp-e R={year_out['quadratic_ensemble']['R']:.0f}")

    with open(out_fp, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_fp}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
