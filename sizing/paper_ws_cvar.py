"""WS / RP / EEV decomposition table (S4) + CVaR risk sweep (S6).

S4: the canonical stochastic-programming decomposition, in this
paper's setting. WS = wait-and-see (dispatch on realized prices,
perfect foresight); RP = the recourse-problem solution, which for the
linear objective equals the LP on the ensemble mean (planning on the
mean IS the scenario-mean optimum); EEV = the mean-value/point-forecast
solution evaluated against realized prices (the lag-24h single
forecast). EVPI = WS - RP, operational VSS = RP - EEV, plus the
capacity-level analogue (b* under each information structure).

S6: risk-measure sweep. Per-chunk revenues give an empirical
distribution of period revenue; the sizing objective becomes
n_chunks * CVaR_alpha(chunk revenue) discounted, alpha in
{0 (=mean), 0.5, 0.85}. Question: does b_E* move with risk aversion,
and does single-vs-ensemble invariance survive at each alpha?

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python -u \
         sizing/paper_ws_cvar.py --out results/main/ws_cvar.json
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
from env import PriceEnv
from paper_benchmark import B_E_GRID, B_P, MU_LP, chunk_indices

DISC = sum(1.07 ** -y for y in range(15))
CAPEX_E = 100_000.0
CAPEX_P = 75_000.0
ALPHAS = [0.0, 0.5, 0.85]


def chunk_revenues(realized, plan, b_E):
    revs, soc = [], b_E / 2.0
    for s, e in chunk_indices(len(realized)):
        env = PriceEnv(b_E, B_P, prices=realized[s:e], soc0=soc,
                       mu=MU_LP, reward_mode="linear")
        a = lp_linear_actions(plan[s:e], b_E, B_P, soc, mu=MU_LP)
        rec = run_actions(env, a)
        revs.append(rec["R"])
        soc = rec["soc_log"][-1]
    return np.array(revs)


def cvar(x: np.ndarray, alpha: float) -> float:
    """Mean of the worst (1-alpha) fraction (alpha=0 -> plain mean)."""
    if alpha <= 0:
        return float(x.mean())
    k = max(1, int(np.ceil((1 - alpha) * len(x))))
    return float(np.sort(x)[:k].mean())


def analyze(source: str, year: int) -> dict:
    if source == "dk1":
        realized, F = multi_lag_persistence(year)
    else:
        from ercot_loader import multi_lag_persistence_ercot
        realized, F = multi_lag_persistence_ercot(year)
    plans = {"EEV_single": F[0],
             "RP_ensemble": F.mean(axis=0),
             "WS_oracle": realized}
    bE = np.array(B_E_GRID)
    Rc = {name: np.array([chunk_revenues(realized, plan, b) for b in bE])
          for name, plan in plans.items()}
    n_chunks = Rc["WS_oracle"].shape[1]

    out = {"npv_at_common_b": {}, "b_star": {}, "cvar_b_star": {}}
    # capacity argmax per information structure (mean objective)
    npvs = {}
    for name in plans:
        R_year = Rc[name].sum(axis=1)
        npvs[name] = R_year * DISC - CAPEX_E * bE - CAPEX_P * B_P
        out["b_star"][name] = float(bE[npvs[name].argmax()])
    # decomposition at the RP argmax capacity
    j = int(npvs["RP_ensemble"].argmax())
    for name in plans:
        out["npv_at_common_b"][name] = float(npvs[name][j])
    out["common_b"] = float(bE[j])
    out["EVPI"] = out["npv_at_common_b"]["WS_oracle"] - \
        out["npv_at_common_b"]["RP_ensemble"]
    out["VSS_op"] = out["npv_at_common_b"]["RP_ensemble"] - \
        out["npv_at_common_b"]["EEV_single"]
    # CVaR sizing sweep (single + ensemble)
    for name in ("EEV_single", "RP_ensemble"):
        out["cvar_b_star"][name] = {}
        for a in ALPHAS:
            obj = np.array([n_chunks * cvar(Rc[name][i], a) for i in
                            range(len(bE))]) * DISC - CAPEX_E * bE - CAPEX_P * B_P
            out["cvar_b_star"][name][f"{a:g}"] = float(bE[obj.argmax()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}
    for src in ("dk1", "ercot"):
        for y in (2021, 2022, 2023):
            t0 = time.time()
            r = analyze(src, y)
            res[f"{src}_{y}"] = r
            cv_s = r["cvar_b_star"]["EEV_single"]
            cv_e = r["cvar_b_star"]["RP_ensemble"]
            print(f"{src} {y}: b*(WS/RP/EEV)="
                  f"{r['b_star']['WS_oracle']:g}/{r['b_star']['RP_ensemble']:g}/"
                  f"{r['b_star']['EEV_single']:g}  "
                  f"EVPI={r['EVPI']/1e6:.2f}M VSS_op={r['VSS_op']/1e6:.2f}M  "
                  f"CVaR b*(single)={list(cv_s.values())} "
                  f"b*(ens)={list(cv_e.values())}  ({time.time()-t0:.0f}s)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": {"alphas": ALPHAS, "B_E": B_E_GRID,
                            "note": "CVaR over 8-week chunk revenues"},
                   "regimes": res}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
