"""b_sat^eps diagnostic: regime-classification statistic.

Per THEORY_DRAFT.md: b_sat^eps(pi) is the smallest battery energy
capacity at which the marginal revenue R'(b, pi) drops below epsilon.
Computed by finite differences on a fine sweep, with bootstrap CIs
across forecast realizations.

Per PREREGISTRATION_ERCOT.md (preserved in AMEND_01):
  - eps/c_E in {0.01, 0.05, 0.10}, headline at 0.05
  - 32 bootstrap seeds
  - 95% percentile CIs

Standalone tool: works for any (price_realized, forecast_builder, policy)
triple. Used both in synthetic AR(1) validation and DK1 empirical regime
characterization.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from arbitrage_agents import qp_ensemble_actions, qp_quadratic_actions, run_actions
from env import PriceEnv


def revenue_curve(b_E_grid: np.ndarray,
                  realized: np.ndarray,
                  forecast_builder: Callable[[int], np.ndarray | tuple[np.ndarray, ...]],
                  policy: str,
                  alpha: float,
                  b_P: float = 1.0,
                  K: int = 4,
                  n_seeds: int = 32,
                  seed_base: int = 0) -> dict:
    """Sweep b_E, compute mean realized revenue per b_E with bootstrap.

    forecast_builder(seed) -> either:
       a 1-D ndarray (single forecast, used by 'single' policies)
       a 2-D ndarray of shape (K, T) (used by 'ensemble' policies)
    The same callable should support both shapes via the K argument.

    Returns:
       {b_E_grid, R_mean, R_std, R_per_seed (n_seeds, len(b_E_grid))}
    """
    R_per_seed = np.zeros((n_seeds, len(b_E_grid)))
    for s in range(n_seeds):
        for j, b_E in enumerate(b_E_grid):
            if policy == "qp_single":
                fc = forecast_builder(seed_base + s, K=1)
                if fc.ndim == 2:
                    fc = fc[0]
                env = PriceEnv(b_E, b_P, prices=realized, alpha=alpha,
                               reward_mode="quadratic")
                a = qp_quadratic_actions(fc, b_E, b_P, env.soc0, alpha=alpha)
                rec = run_actions(env, a)
            elif policy == "qp_ensemble":
                fc = forecast_builder(seed_base + s, K=K)
                env = PriceEnv(b_E, b_P, prices=realized, alpha=alpha,
                               reward_mode="quadratic")
                a = qp_ensemble_actions(fc, b_E, b_P, env.soc0, alpha=alpha)
                rec = run_actions(env, a)
            else:
                raise ValueError(f"unknown policy {policy}")
            R_per_seed[s, j] = rec["R"]
    R_mean = R_per_seed.mean(axis=0)
    R_std = R_per_seed.std(axis=0)
    return {
        "b_E_grid": b_E_grid,
        "R_mean": R_mean,
        "R_std": R_std,
        "R_per_seed": R_per_seed,
    }


def b_sat_eps(b_E_grid: np.ndarray, R_per_seed: np.ndarray,
              eps: float, n_bootstrap: int = 1000,
              ci: float = 0.95) -> dict:
    """Estimate b_sat^eps and percentile bootstrap CI.

    For each bootstrap resample of seeds, compute R_mean(b_E),
    finite-difference R'(b_E), find smallest b_E where R'(b_E) < eps.

    Returns: {point, ci_low, ci_high, samples}
    """
    n_seeds = R_per_seed.shape[0]
    rng = np.random.default_rng(0)
    samples = np.zeros(n_bootstrap)
    for k in range(n_bootstrap):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        R = R_per_seed[idx].mean(axis=0)
        # Forward finite diff
        dR = np.diff(R) / np.diff(b_E_grid)
        # Find smallest b_E where dR < eps
        # dR has length n-1; corresponds to midpoint b values
        below = np.where(dR < eps)[0]
        if len(below) == 0:
            samples[k] = b_E_grid[-1]   # never saturates within range
        else:
            samples[k] = b_E_grid[below[0]]
    point_R = R_per_seed.mean(axis=0)
    dR_point = np.diff(point_R) / np.diff(b_E_grid)
    below = np.where(dR_point < eps)[0]
    point = b_E_grid[below[0]] if len(below) else b_E_grid[-1]
    alpha = (1 - ci) / 2
    return {
        "point": float(point),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1 - alpha)),
        "samples": samples,
    }


def overlap_pct(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> float:
    """% overlap relative to the smaller CI's width.

    Special cases:
      - Both CIs degenerate to the same point: 100%.
      - Both degenerate but at different points: 0%.
      - One degenerate, contained in the other's CI: 100%.
    """
    lo = max(ci_a[0], ci_b[0])
    hi = min(ci_a[1], ci_b[1])
    overlap = max(0.0, hi - lo)
    width_a = ci_a[1] - ci_a[0]
    width_b = ci_b[1] - ci_b[0]
    # Degenerate cases
    if width_a == 0 and width_b == 0:
        return 100.0 if abs(ci_a[0] - ci_b[0]) < 1e-9 else 0.0
    if width_a == 0:
        return 100.0 if ci_b[0] <= ci_a[0] <= ci_b[1] else 0.0
    if width_b == 0:
        return 100.0 if ci_a[0] <= ci_b[0] <= ci_a[1] else 0.0
    smaller_width = min(width_a, width_b)
    return 100.0 * overlap / smaller_width


# -----------------------------------------------------------------------------
# Synthetic validation
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from price_signal import make_forecast, synth_diurnal

    NOISE = 8
    ALPHA = 0.005
    B_P = 1.0

    # Synthetic AR(1) week (Pilot S1 setup)
    realized = synth_diurnal(168, seed=42)
    pool_rng = np.random.default_rng(11)

    def synthetic_forecast_builder(seed: int, K: int = 1) -> np.ndarray:
        if K == 1:
            return make_forecast(realized, NOISE, seed=seed)
        return np.stack([
            make_forecast(realized, NOISE, seed=int(s))
            for s in pool_rng.integers(0, 1_000_000, size=K)
        ])

    # Fine sweep around the apparent argmax
    b_E_grid = np.array([0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20])
    print(f"Synthetic AR(1) b_sat^eps validation (n_seeds=16 for speed)")
    t0 = time.time()
    out_single = revenue_curve(b_E_grid, realized, synthetic_forecast_builder,
                                policy="qp_single", alpha=ALPHA, b_P=B_P,
                                K=4, n_seeds=16)
    out_ens = revenue_curve(b_E_grid, realized, synthetic_forecast_builder,
                             policy="qp_ensemble", alpha=ALPHA, b_P=B_P,
                             K=4, n_seeds=16)
    print(f"  swept in {time.time()-t0:.0f}s")

    print(f"\n{'b_E':>5} | {'R_single':>8} | {'R_ens':>8}")
    for j, b in enumerate(b_E_grid):
        print(f"{b:5.1f} | {out_single['R_mean'][j]:8.0f} | {out_ens['R_mean'][j]:8.0f}")

    # b_sat^eps with eps relative to a typical c_E. For synthetic with
    # weekly revenue ~3000 EUR, annual ~150k, we approximate marginal-rev
    # threshold at eps ~ 50 EUR/week per MWh (rough proxy for c_E in
    # weekly amortization; we'll use eps absolute).
    for eps in [10.0, 50.0, 100.0]:
        sat_s = b_sat_eps(b_E_grid, out_single["R_per_seed"], eps=eps)
        sat_e = b_sat_eps(b_E_grid, out_ens["R_per_seed"], eps=eps)
        ov = overlap_pct((sat_s["ci_low"], sat_s["ci_high"]),
                         (sat_e["ci_low"], sat_e["ci_high"]))
        print(f"\n  eps={eps:6.1f} EUR/week-per-MWh:")
        print(f"    b_sat (single):   {sat_s['point']:5.1f}  CI [{sat_s['ci_low']:.1f}, {sat_s['ci_high']:.1f}]")
        print(f"    b_sat (ensemble): {sat_e['point']:5.1f}  CI [{sat_e['ci_low']:.1f}, {sat_e['ci_high']:.1f}]")
        print(f"    overlap: {ov:.0f}%   (>=50% -> invariance survives)")
