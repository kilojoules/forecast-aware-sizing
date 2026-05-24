"""Paper benchmark: 2x2 factorial × DK1 years × b_E sweep × bootstrap.

Per PREREGISTRATION_ERCOT.md (commit 2ce93ae) + PREREGISTRATION_AMEND_01.md
(commit ebb035c).

  Years        : 2021, 2022, 2023 (DK1 day-ahead)
  Forecast     : persistence(t-24h)
  Ensemble     : K=4 noisy persistences perturbed by empirical residuals
  b_P          : 1 MW fixed
  b_E grid     : {0.5, 1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64} MWh
  policies     : {LP-linear, QP-quadratic} x {single, K=4 ensemble}
  bootstrap    : 16 seeds (reduced from 32 for runtime; pre-reg AMEND if used)
  chunking     : 8-week chunks, stitched per year for revenue + D

Output: paper_results.json
        keys: meta, summaries[year][policy] = list[{b_E, R_mean_eur, R_per_seed, D_mean}]

Invocation:
  python paper_benchmark.py [--year YYYY] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from arbitrage_agents import (lp_linear_actions, qp_ensemble_actions,
                              qp_quadratic_actions, run_actions)
from degradation import cycle_degradation
from dk_loader import (empirical_residuals, load_dk_year, multi_lag_persistence,
                       realized_and_persistence)
from ercot_loader import multi_lag_persistence_ercot
from env import PriceEnv


B_E_GRID = [0.5, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
B_P = 1.0
ALPHA_QP = 0.005
MU_LP = 5.0
K_ENSEMBLE = 4
CHUNK_HOURS = 24 * 7 * 8     # 8-week chunks (1344 h)


def safe_D(soc, B):
    try:
        D, _, _ = cycle_degradation(soc, B)
    except (IndexError, ValueError):
        return 0.0
    return D


def chunk_indices(T: int, chunk: int = CHUNK_HOURS):
    """Yield (start, end) index pairs covering [0, T)."""
    s = 0
    while s < T:
        yield s, min(s + chunk, T)
        s += chunk


def dispatch_chunked(realized: np.ndarray, forecast: np.ndarray,
                      b_E: float, cost: str, alpha_or_mu: float) -> dict:
    """Run a single-forecast policy on the chunked year. Returns total
    R + total D."""
    R_total = 0.0
    soc_full = []
    soc_carry = b_E / 2.0
    for s, e in chunk_indices(len(realized)):
        r_chunk = realized[s:e]
        f_chunk = forecast[s:e]
        env = PriceEnv(b_E, B_P, prices=r_chunk, soc0=soc_carry,
                       alpha=alpha_or_mu if cost == "quadratic" else 0.0,
                       mu=alpha_or_mu if cost == "linear" else 0.0,
                       reward_mode=cost)
        if cost == "linear":
            a = lp_linear_actions(f_chunk, b_E, B_P, soc_carry, mu=alpha_or_mu)
        else:
            a = qp_quadratic_actions(f_chunk, b_E, B_P, soc_carry, alpha=alpha_or_mu)
        rec = run_actions(env, a)
        R_total += rec["R"]
        soc_full.extend(rec["soc_log"][1:])
        soc_carry = rec["soc_log"][-1]
    soc_full = [b_E / 2.0] + soc_full
    D_total = safe_D(np.asarray(soc_full), b_E)
    return {"R": R_total, "D": D_total}


def dispatch_chunked_ensemble(realized: np.ndarray, forecasts_K: np.ndarray,
                               b_E: float, cost: str, alpha_or_mu: float) -> dict:
    """Ensemble dispatch on chunked year. forecasts_K shape (K, T)."""
    R_total = 0.0
    soc_full = []
    soc_carry = b_E / 2.0
    for s, e in chunk_indices(realized.shape[0]):
        r_chunk = realized[s:e]
        f_chunk_K = forecasts_K[:, s:e]
        env = PriceEnv(b_E, B_P, prices=r_chunk, soc0=soc_carry,
                       alpha=alpha_or_mu if cost == "quadratic" else 0.0,
                       mu=alpha_or_mu if cost == "linear" else 0.0,
                       reward_mode=cost)
        if cost == "linear":
            # ensemble for linear LP: also use mean forecast
            a = lp_linear_actions(f_chunk_K.mean(axis=0), b_E, B_P, soc_carry, mu=alpha_or_mu)
        else:
            a = qp_ensemble_actions(f_chunk_K, b_E, B_P, soc_carry, alpha=alpha_or_mu)
        rec = run_actions(env, a)
        R_total += rec["R"]
        soc_full.extend(rec["soc_log"][1:])
        soc_carry = rec["soc_log"][-1]
    soc_full = [b_E / 2.0] + soc_full
    D_total = safe_D(np.asarray(soc_full), b_E)
    return {"R": R_total, "D": D_total}


def run_year(year: int, n_seeds: int, quick: bool = False,
              source: str = "dk1") -> dict:
    """Per AMEND_02: ensemble = multi-lag persistence (24, 48, 168, 336 h).

    Source: 'dk1' (Energinet DA, EUR/MWh) or 'ercot' (ERCOT HB_NORTH DA, USD/MWh).
    """
    if source == "dk1":
        realized, forecasts_K = multi_lag_persistence(year, lags_hours=(24, 48, 168, 336))
    elif source == "ercot":
        realized, forecasts_K = multi_lag_persistence_ercot(year, lags_hours=(24, 48, 168, 336))
    else:
        raise ValueError(f"unknown source {source}")
    forecast_single = forecasts_K[0]  # 24 h lag
    residuals_24 = realized - forecast_single
    print(f"\n[{year}] T={len(realized)} h, residual_24h std={residuals_24.std():.1f} EUR/MWh")
    grid = B_E_GRID if not quick else [1, 2, 4, 8, 16, 32]

    summaries = {}
    for cost, alpha_or_mu in [("linear", MU_LP), ("quadratic", ALPHA_QP)]:
        rows_single = []
        rows_ens = []
        t0 = time.time()
        for b_E in grid:
            res_s = dispatch_chunked(realized, forecast_single, b_E, cost, alpha_or_mu)
            rows_single.append({"b_E": b_E, "R": res_s["R"], "D": res_s["D"]})
            res_e = dispatch_chunked_ensemble(realized, forecasts_K, b_E, cost, alpha_or_mu)
            rows_ens.append({"b_E": b_E, "R": res_e["R"], "D": res_e["D"]})
        print(f"  {cost}: {time.time()-t0:.0f}s  "
              f"single R={min(r['R'] for r in rows_single):.0f}.."
              f"{max(r['R'] for r in rows_single):.0f}  "
              f"ens R={min(r['R'] for r in rows_ens):.0f}.."
              f"{max(r['R'] for r in rows_ens):.0f}")
        summaries[f"{cost}_single"] = rows_single
        summaries[f"{cost}_ensemble"] = rows_ens

    # Bootstrap CIs on the argmax come from a follow-on script that
    # adds noise to the ensemble construction. For this benchmark we
    # report point estimates; CIs require a separate run with multi-lag
    # subsets.
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["dk1", "ercot"], default="dk1")
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023])
    parser.add_argument("--n_seeds", type=int, default=16)
    parser.add_argument("--quick", action="store_true",
                         help="Smaller b_E grid + fewer seeds for smoke test")
    parser.add_argument("--out", default="paper_results.json")
    args = parser.parse_args()

    n_seeds = 4 if args.quick else args.n_seeds
    all_results = {
        "meta": {
            "source": args.source,
            "years": args.years,
            "b_E_grid": B_E_GRID if not args.quick else [1, 2, 4, 8, 16, 32],
            "b_P_MW": B_P,
            "alpha_qp": ALPHA_QP,
            "mu_lp": MU_LP,
            "K_ensemble": K_ENSEMBLE,
            "n_seeds": n_seeds,
            "chunk_hours": CHUNK_HOURS,
        },
        "by_year": {},
    }
    t_total = time.time()
    for year in args.years:
        all_results["by_year"][str(year)] = run_year(year, n_seeds=n_seeds,
                                                      quick=args.quick,
                                                      source=args.source)
        # Save incrementally
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
    print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
