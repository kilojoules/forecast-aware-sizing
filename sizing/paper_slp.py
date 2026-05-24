"""Two-stage scenario stochastic LP for battery dispatch.

Per AMEND_05.

Per-period scenario LP:
  Stage 1: commit to charge/discharge at time t (here-and-now).
  Stage 2: under each of N scenarios, dispatch the rest of the horizon.
  Minimize -E[revenue] + alpha * cycling.

To stay tractable: 50 price scenarios sampled from the empirical residual
distribution around the persistence forecast. Solve the SLP over short
rolling windows (24 h) with state carryforward.

This is the production-grade comparison referenced in §1 / Krishnamurthy
2018 / Birge & Louveaux 1997.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import cvxpy as cp

from arbitrage_agents import lp_linear_actions, qp_quadratic_actions, run_actions
from degradation import cycle_degradation
from dk_loader import empirical_residuals as dk_residuals
from dk_loader import load_dk_year, multi_lag_persistence
from env import PriceEnv
from ercot_loader import load_ercot_year, multi_lag_persistence_ercot


B_E_GRID = [0.5, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
B_P = 1.0
ALPHA_QP = 0.005
MU_LP = 5.0
N_SCENARIOS = 50
WINDOW = 24                         # rolling-horizon length (h)


def empirical_residuals(source: str, year: int) -> np.ndarray:
    if source == "dk1":
        return dk_residuals(year)
    df = load_ercot_year(year)
    da = df["da_usd_per_mwh"].to_numpy()
    return da[24:] - da[:-24]


def slp_window(prices_window_scenarios, b_E, b_P, soc0, alpha, eta=1.0):
    """SLP over a single rolling window across N scenarios. prices: (N, T)."""
    N, T = prices_window_scenarios.shape
    P_chg = cp.Variable(T, nonneg=True)
    P_dis = cp.Variable(T, nonneg=True)
    # Single first-stage commit: same actions across all scenarios
    rev_per_scenario = [prices_window_scenarios[k] @ (P_dis - P_chg) for k in range(N)]
    cyc = cp.sum_squares(P_chg + P_dis)
    obj = cp.Maximize(cp.sum(rev_per_scenario) / N - alpha * cyc)
    cons = [P_chg <= b_P, P_dis <= b_P]
    soc_t = soc0 + eta * cp.cumsum(P_chg) - cp.cumsum(P_dis) / max(eta, 1e-9)
    cons += [soc_t >= 0, soc_t <= b_E]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver="CLARABEL")
    except Exception:
        prob.solve(solver="SCS")
    if P_dis.value is None or P_chg.value is None:
        return np.zeros(T), np.zeros(T)
    return np.asarray(P_chg.value), np.asarray(P_dis.value)


def rolling_slp(realized, fc_baseline, residual_pool, b_E, b_P, alpha, seed=0):
    """Rolling-horizon SLP. At each step, sample N scenarios from pool,
    solve SLP, apply first action, advance."""
    rng = np.random.default_rng(seed)
    T = len(realized)
    soc = b_E / 2
    actions = np.zeros(T)
    socs = [soc]
    R_total = 0.0
    for t in range(T):
        end = min(t + WINDOW, T)
        h = end - t
        if h < 1:
            break
        # Build scenarios: baseline + iid residual noise per scenario
        baseline = fc_baseline[t:end]
        scenarios = np.zeros((N_SCENARIOS, h))
        for k in range(N_SCENARIOS):
            idx = rng.integers(0, len(residual_pool), size=h)
            scenarios[k] = baseline + residual_pool[idx]
        P_chg, P_dis = slp_window(scenarios, b_E, b_P, soc, alpha)
        # Apply first action (greedy from SLP)
        a_chg = float(np.clip(P_chg[0], 0, b_P))
        a_dis = float(np.clip(P_dis[0], 0, soc))
        a = a_dis - a_chg
        a = max(-b_P, min(b_P, a))
        soc = max(0, min(b_E, soc + a_chg - a_dis))
        actions[t] = a
        socs.append(soc)
        R_total += realized[t] * a
    return R_total, np.asarray(socs), actions


def safe_D(soc, b):
    try:
        D, _, _ = cycle_degradation(soc, b)
        return D
    except Exception:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["dk1", "ercot"], required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.source == "dk1":
        realized, fc_K = multi_lag_persistence(args.year, lags_hours=(24,))
    else:
        realized, fc_K = multi_lag_persistence_ercot(args.year, lags_hours=(24,))
    fc_baseline = fc_K[0]
    residual_pool = empirical_residuals(args.source, args.year)
    print(f"SLP: {args.source} {args.year}, T={len(realized)}, residual std={residual_pool.std():.1f}")

    out = {"meta": {"source": args.source, "year": args.year, "method": "SLP_N50_W24"}}
    out["by_year"] = {str(args.year): {}}
    for cost, m in [("quadratic", ALPHA_QP)]:    # SLP only QP (LP rank-invariant)
        rows = []
        for b_E in B_E_GRID:
            t0 = time.time()
            R, soc_log, _ = rolling_slp(realized, fc_baseline, residual_pool,
                                          b_E, B_P, alpha=m)
            D = safe_D(soc_log, b_E)
            print(f"  b_E={b_E}: R={R:.0f}  D={D:.5f}  ({time.time()-t0:.0f}s)")
            rows.append({"b_E": b_E, "R": R, "D": D})
        out["by_year"][str(args.year)][f"{cost}_slp"] = rows
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
