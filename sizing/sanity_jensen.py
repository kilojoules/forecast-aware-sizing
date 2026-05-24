"""Pilot (a) day 3-5: Jensen-gap test under nonlinear cycling cost.

The two prior pilots used LINEAR cycling cost. LP linear-cost solutions
are rank-invariant in prices: shrinkage / scaling of forecast cannot
change decisions. This is why pilot (a) day 2 found zero shrinkage lift.

With QUADRATIC (or otherwise convex nonlinear) cycling cost, this
property breaks. Optimal cycling depth depends on price *magnitudes*,
not just ranks. Now Jensen's inequality should let stochastic-LP /
ensemble-forecast policies beat deterministic-LP under noise.

Setup:
  Quadratic cycling cost: alpha * sum (P_chg[t] + P_dis[t])^2.
  Solve via scipy.optimize.minimize (SLSQP) -- convex QP.
  Compare: deterministic-QP-with-forecast vs ensemble-forecast-QP vs oracle.

Headline test: does ensemble-forecast (averaging K forecast realizations
into a single 'mean forecast' before solve) beat single-forecast at
nonlinear cost?

If yes: Jensen-gap angle is alive. RL trained on the joint distribution
could plausibly do at least this well.
If no: even at nonlinear cost, the angle dies. Kill the rework outright.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds

from sanity_forecast import (
    B_E, B_P, SOC0, SEED,
    synth_price, make_forecast,
    simulate_with_actions, revenue_realized, D_of,
    lp_dispatch, policy_lp_oracle,
)
from degradation import cycle_degradation


T_QP = 72  # 3 days (keeps SLSQP tractable)


def solve_qp(price: np.ndarray, alpha: float, soc0: float = SOC0):
    """Convex QP: maximize sum price * (P_dis - P_chg) - alpha * sum (P_chg + P_dis)^2.
    Variables: x = [P_chg, P_dis], length 2T.
    Returns (P_chg, P_dis).
    """
    n_t = len(price)
    n = 2 * n_t

    def neg_obj(x):
        P_chg = x[:n_t]
        P_dis = x[n_t:]
        rev = np.sum(price * (P_dis - P_chg))
        cyc = np.sum((P_chg + P_dis) ** 2)
        return -(rev - alpha * cyc)

    def neg_obj_grad(x):
        P_chg = x[:n_t]
        P_dis = x[n_t:]
        d_chg = -(-price - 2 * alpha * (P_chg + P_dis))   # d(neg_obj)/d_P_chg
        d_dis = -(price - 2 * alpha * (P_chg + P_dis))    # d(neg_obj)/d_P_dis
        return np.concatenate([d_chg, d_dis])

    # Linear constraints: 0 <= soc0 + cumsum(P_chg - P_dis) <= B_E
    A = np.zeros((2 * n_t, n))
    lb = np.zeros(2 * n_t)
    ub = np.zeros(2 * n_t)
    for t in range(n_t):
        # Upper-bound row: soc0 + cumsum(P_chg) - cumsum(P_dis) <= B_E
        A[t, 0:t+1] = 1.0
        A[t, n_t:n_t+t+1] = -1.0
        lb[t] = -np.inf
        ub[t] = B_E - soc0
        # Lower-bound row: soc0 + cumsum(P_chg) - cumsum(P_dis) >= 0
        A[n_t+t, 0:t+1] = 1.0
        A[n_t+t, n_t:n_t+t+1] = -1.0
        lb[n_t+t] = -soc0
        ub[n_t+t] = np.inf

    constr = LinearConstraint(A, lb, ub)
    bounds = Bounds(np.zeros(n), B_P * np.ones(n))
    x0 = np.zeros(n)
    res = minimize(neg_obj, x0, jac=neg_obj_grad, method="SLSQP",
                   bounds=bounds, constraints=constr,
                   options={"maxiter": 200, "ftol": 1e-6})
    if not res.success:
        # Fallback: trust-constr
        res = minimize(neg_obj, x0, jac=neg_obj_grad, method="trust-constr",
                       bounds=bounds, constraints=constr,
                       options={"maxiter": 500, "gtol": 1e-6})
    return res.x[:n_t], res.x[n_t:]


def main():
    print("Pilot (a) day 3-5: Jensen-gap test under quadratic cycling cost")
    print(f"T={T_QP}h  B_E={B_E}  B_P={B_P}\n")

    realized_full = synth_price(168, seed=SEED)
    realized = realized_full[:T_QP]
    print(f"Realized: mean={realized.mean():.1f} std={realized.std():.1f}")

    # Solve oracle QP at one alpha to get a sane reference revenue
    ALPHA = 0.005   # cycling penalty -- tune so QP cycles a few times not zero
    print(f"\nQuadratic cycling alpha = {ALPHA}\n")

    # Oracle (perfect foresight)
    t0 = time.time()
    P_chg, P_dis = solve_qp(realized, ALPHA)
    print(f"  oracle QP solve: {time.time()-t0:.1f}s")
    soc_or, chg_or, dis_or = simulate_with_actions(P_chg, P_dis)
    R_oracle = revenue_realized(realized, chg_or, dis_or)
    D_oracle = D_of(soc_or, B_E)
    print(f"Oracle QP: R={R_oracle:.1f}  D={D_oracle:.5f}\n")

    # Sweep noise + compare deterministic-with-forecast vs ensemble-mean-forecast
    noise_levels = [3, 8, 18]
    K_ENSEMBLE = 16   # forecast realizations averaged for ensemble policy
    fc_seeds_per_test = list(range(8))  # outer loop: 8 single-forecast realizations
    rng_pool = np.random.default_rng(123)

    print("Comparing single-forecast QP vs ensemble-forecast QP:")
    print(f"{'noise':>6} | {'R_single':>9} | {'R_ensem':>8} | {'D_single':>9} | {'D_ensem':>8} | lift_R | lift_(R-aD)")
    print("-" * 92)
    rows = []
    for noise_std in noise_levels:
        Rs_s, Rs_e = [], []
        Ds_s, Ds_e = [], []
        for outer_seed in fc_seeds_per_test:
            # Single forecast
            forecast_single = make_forecast(realized_full, noise_std, seed=outer_seed)[:T_QP]
            P_chg_s, P_dis_s = solve_qp(forecast_single, ALPHA)
            soc_s, chg_s, dis_s = simulate_with_actions(P_chg_s, P_dis_s)
            Rs_s.append(revenue_realized(realized, chg_s, dis_s))
            Ds_s.append(D_of(soc_s, B_E))

            # Ensemble forecast: average K independent forecasts of THE SAME realized
            inner_seeds = rng_pool.integers(0, 100000, size=K_ENSEMBLE)
            forecasts = np.stack([
                make_forecast(realized_full, noise_std, seed=int(s))[:T_QP]
                for s in inner_seeds])
            forecast_mean = forecasts.mean(axis=0)
            P_chg_e, P_dis_e = solve_qp(forecast_mean, ALPHA)
            soc_e, chg_e, dis_e = simulate_with_actions(P_chg_e, P_dis_e)
            Rs_e.append(revenue_realized(realized, chg_e, dis_e))
            Ds_e.append(D_of(soc_e, B_E))

        R_s, R_e = np.mean(Rs_s), np.mean(Rs_e)
        D_s, D_e = np.mean(Ds_s), np.mean(Ds_e)
        lift_R = (R_e - R_s) / max(abs(R_s), 1e-6) * 100
        # Joint reward: revenue - alpha * cycling cost (proxy for net value with degradation in loop)
        score_s = R_s - 1000 * D_s   # weight degradation heavily for visibility
        score_e = R_e - 1000 * D_e
        lift_score = (score_e - score_s) / max(abs(score_s), 1e-6) * 100
        rows.append((noise_std, R_s, R_e, D_s, D_e, lift_R, lift_score))
        print(f"{noise_std:6.1f} | {R_s:9.1f} | {R_e:8.1f} | "
              f"{D_s:9.5f} | {D_e:8.5f} | {lift_R:+5.1f}% | {lift_score:+5.1f}%")

    # Plot
    rows = np.array(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(rows[:, 0], rows[:, 1], "o-", color="#cc6677", label="Single-forecast QP")
    ax.plot(rows[:, 0], rows[:, 2], "s-", color="#4477aa", label=f"Ensemble (K={K_ENSEMBLE}) QP")
    ax.axhline(R_oracle, color="black", lw=1, ls="--", label=f"Oracle={R_oracle:.0f}")
    ax.set_xlabel("Forecast AR(1) noise std")
    ax.set_ylabel("Realized revenue")
    ax.set_title(f"QP with alpha={ALPHA} (quadratic cycling cost)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(rows[:, 0], rows[:, 5], "o-", color="#cc6677", label="Lift in revenue (%)")
    ax.plot(rows[:, 0], rows[:, 6], "s-", color="#4477aa", label="Lift in (R - 1000D) (%)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Forecast AR(1) noise std")
    ax.set_ylabel("Ensemble lift over single forecast (%)")
    ax.set_title("Jensen lift: ensemble vs single forecast")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Pilot (a) day 3-5: Jensen-gap test (T={T_QP}h, alpha={ALPHA})", y=1.02)
    fig.tight_layout()
    out = "fig_jensen_gap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")

    # Verdict
    max_lift = max(rows[:, 5].max(), rows[:, 6].max())
    print(f"\n{'='*70}")
    print(f"Max ensemble lift: {max_lift:.1f}%")
    if max_lift > 5:
        print("Ensemble forecast meaningfully beats single forecast under nonlinear cost.")
        print("-> Jensen-gap angle alive. RL trained on joint distribution should do ~this well.")
    elif max_lift > 1:
        print(f"Marginal lift ({max_lift:.1f}%). Effect exists but small.")
    else:
        print(f"No measurable lift ({max_lift:.1f}%). Even at nonlinear cost, the gap doesnt open.")
        print("-> KILL the rework. All three angles dead.")


if __name__ == "__main__":
    main()
