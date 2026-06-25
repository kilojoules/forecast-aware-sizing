"""Continuous forecast-uncertainty sweep: operations cost vs sizing invariance.

Parameterized forecast:
    forecast_gamma(t) = realized(t) + gamma * (persistence(t) - realized(t))
  gamma=0 -> perfect foresight (oracle); gamma=1 -> persistence skill;
  gamma>1 -> error amplified beyond persistence.

For each (market, year) and gamma, dispatch the full b_E grid (8-week
chunks, SoC carry, linear LP) on realized prices and record:
  - revenue at a fixed reference battery (operations panel)
  - argmax b_E* and its NPV (sizing panel -- is it flat in gamma?)

This is the continuous version of the single-vs-ensemble invariance
claim: does optimal capacity stay put as the forecast is continuously
degraded, even as operating revenue collapses?

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python -u \
         sizing/paper_gamma_sweep.py --out results/main/gamma_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from arbitrage_agents import lp_linear_actions, run_actions
from dk_loader import multi_lag_persistence
from ercot_loader import multi_lag_persistence_ercot
from env import PriceEnv
from paper_benchmark import B_E_GRID, B_P, MU_LP, chunk_indices

DISC = sum(1.07 ** -y for y in range(15))
CE, CP = 100_000.0, 75_000.0
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
REF_BE = 16.0  # operations reference battery


def revenue_for(realized, plan, b_E):
    R, soc = 0.0, b_E / 2.0
    for s, e in chunk_indices(len(realized)):
        env = PriceEnv(b_E, B_P, prices=realized[s:e], soc0=soc,
                       mu=MU_LP, reward_mode="linear")
        a = lp_linear_actions(plan[s:e], b_E, B_P, soc, mu=MU_LP)
        rec = run_actions(env, a)
        R += rec["R"]
        soc = rec["soc_log"][-1]
    return R


def analyze(source, year):
    realized, F = (multi_lag_persistence(year) if source == "dk1"
                   else multi_lag_persistence_ercot(year))
    persistence = F[0]
    rows = []
    for gamma in GAMMAS:
        plan = realized + gamma * (persistence - realized)
        npvs = []
        for b_E in B_E_GRID:
            R = revenue_for(realized, plan, b_E)
            npvs.append((R * DISC - CE * b_E - CP, b_E, R))
        best = max(npvs)
        ref = [n for n in npvs if n[1] == REF_BE][0]
        rows.append({"gamma": gamma, "b_E_star": best[1],
                     "npv_star": best[0], "rev_ref": ref[2],
                     "npv_curve": [n[0] for n in npvs]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}
    for src in ("dk1", "ercot"):
        for y in (2021, 2022, 2023):
            t0 = time.time()
            rows = analyze(src, y)
            res[f"{src}_{y}"] = rows
            bstars = [r["b_E_star"] for r in rows]
            rev0 = rows[0]["rev_ref"]
            revpct = [f"{r['rev_ref']/rev0*100:.0f}" for r in rows]
            print(f"{src} {y}: b*(gamma)={[f'{b:g}' for b in bstars]}  "
                  f"rev%@16={revpct}  ({time.time()-t0:.0f}s)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"gammas": GAMMAS, "B_E": B_E_GRID,
                        "ref_b_E": REF_BE,
                        "note": "forecast=realized+gamma*(persistence-realized)"},
               "regimes": res}, open(args.out, "w"))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
