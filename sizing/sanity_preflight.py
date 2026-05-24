"""Pilot (a) pre-flight: two cheap tests before committing to the rework.

(a) K-sweep: does Jensen lift survive at REALISTIC ensemble size (K=2, 4)?
    Operators have a few forecasts, not 16. K=16 in sanity_jensen.py was generous.

(b) Pareto: does QP-quadratic-cost strictly dominate LP-linear-cost on the
    (revenue, rainflow D) plane? If LP-linear's Pareto frontier (sweep mu)
    matches QP-quadratic's (sweep alpha), the nonlinear-cost premise fails.

If both pass: commit to refactor.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from sanity_forecast import (
    B_E, B_P, SOC0, SEED,
    synth_price, make_forecast,
    simulate_with_actions, revenue_realized, D_of,
    lp_dispatch,
)
from sanity_jensen import T_QP, solve_qp


# =============================================================================
# (a) K-sweep
# =============================================================================
def part_a_k_sweep():
    print("=" * 70)
    print("(a) K-sweep: Jensen lift at realistic ensemble sizes")
    print("=" * 70)
    realized_full = synth_price(168, seed=SEED)
    realized = realized_full[:T_QP]
    ALPHA = 0.005
    NOISE = 8     # realistic-poor regime where Jensen was strongest
    K_VALUES = [1, 2, 4, 8, 16, 32]
    fc_seeds_per_test = list(range(8))
    rng_pool = np.random.default_rng(456)

    # Oracle
    P_chg, P_dis = solve_qp(realized, ALPHA)
    soc_or, chg_or, dis_or = simulate_with_actions(P_chg, P_dis)
    R_oracle = revenue_realized(realized, chg_or, dis_or)
    D_oracle = D_of(soc_or, B_E)
    print(f"\nOracle QP at alpha={ALPHA}: R={R_oracle:.1f}  D={D_oracle:.5f}")
    print(f"Forecast noise std = {NOISE} (realistic-poor)\n")

    rows = []
    print(f"{'K':>4} | {'R_mean':>8} | {'D_mean':>8} | {'lift_R':>7} | {'lift_score':>9}")
    print("-" * 60)
    for K in K_VALUES:
        Rs, Ds = [], []
        for outer_seed in fc_seeds_per_test:
            inner_seeds = rng_pool.integers(0, 1_000_000, size=K)
            forecasts = np.stack([
                make_forecast(realized_full, NOISE, seed=int(s))[:T_QP]
                for s in inner_seeds])
            forecast_mean = forecasts.mean(axis=0)
            P_chg, P_dis = solve_qp(forecast_mean, ALPHA)
            soc, chg, dis = simulate_with_actions(P_chg, P_dis)
            Rs.append(revenue_realized(realized, chg, dis))
            Ds.append(D_of(soc, B_E))
        R_mean, D_mean = np.mean(Rs), np.mean(Ds)
        rows.append((K, R_mean, D_mean))
        print(f"{K:>4d} | {R_mean:8.1f} | {D_mean:8.5f}")

    rows = np.array(rows)
    R_K1 = rows[0, 1]
    print(f"\nLift over K=1 (single forecast):")
    for K, R, D in rows:
        if K == 1:
            continue
        lift = (R - R_K1) / max(abs(R_K1), 1e-6) * 100
        print(f"  K={int(K)}: +{lift:.1f}%")

    # K=2 is the realistic minimum -- if no lift here, kill
    K2_lift = (rows[1, 1] - R_K1) / max(abs(R_K1), 1e-6) * 100
    K4_lift = (rows[2, 1] - R_K1) / max(abs(R_K1), 1e-6) * 100
    print(f"\n(a) GO threshold: K=2 lift > 5%, K=4 lift > 8%")
    print(f"    K=2 actual: {K2_lift:+.1f}%  -> {'PASS' if K2_lift > 5 else 'FAIL'}")
    print(f"    K=4 actual: {K4_lift:+.1f}%  -> {'PASS' if K4_lift > 8 else 'FAIL'}")

    return rows


# =============================================================================
# (b) Pareto: LP-linear vs QP-quadratic
# =============================================================================
def part_b_pareto():
    print("\n" + "=" * 70)
    print("(b) Pareto: LP-linear-cost frontier vs QP-quadratic-cost frontier")
    print("=" * 70)
    realized_full = synth_price(168, seed=SEED)
    realized = realized_full[:T_QP]

    # LP-linear sweep mu
    mus = np.concatenate([[0.0], np.logspace(-1, np.log10(realized.max() * 2), 18)])
    lp_pts = []
    for mu in mus:
        P_chg, P_dis = lp_dispatch(realized, mu)
        soc, chg, dis = simulate_with_actions(P_chg, P_dis)
        R = revenue_realized(realized, chg, dis)
        D = D_of(soc, B_E)
        lp_pts.append((R, D, mu))
    lp_pts = np.array(lp_pts)
    print(f"LP-linear: swept {len(mus)} mu values. R in [{lp_pts[:,0].min():.0f}, {lp_pts[:,0].max():.0f}], "
          f"D in [{lp_pts[:,1].min():.5f}, {lp_pts[:,1].max():.5f}]")

    # QP-quadratic sweep alpha
    alphas = np.concatenate([[0.0], np.logspace(-4, 0, 12)])
    qp_pts = []
    for alpha in alphas:
        try:
            P_chg, P_dis = solve_qp(realized, alpha)
        except Exception as e:
            print(f"  alpha={alpha} skipped ({e})")
            continue
        soc, chg, dis = simulate_with_actions(P_chg, P_dis)
        R = revenue_realized(realized, chg, dis)
        D = D_of(soc, B_E)
        qp_pts.append((R, D, alpha))
    qp_pts = np.array(qp_pts)
    print(f"QP-quadratic: swept {len(qp_pts)} alpha values. R in [{qp_pts[:,0].min():.0f}, {qp_pts[:,0].max():.0f}], "
          f"D in [{qp_pts[:,1].min():.5f}, {qp_pts[:,1].max():.5f}]")

    # Pareto-dominance: at each LP point, is there a QP point strictly better?
    # And vice versa.
    def strict_dom(pt, others):
        """Does any point in `others` strictly dominate pt = (R, D)?
        Strict: other_R >= R AND other_D <= D AND (other_R > R OR other_D < D)."""
        R, D = pt[0], pt[1]
        return np.any((others[:, 0] >= R - 1e-6) & (others[:, 1] <= D + 1e-9)
                       & ((others[:, 0] > R + 1e-6) | (others[:, 1] < D - 1e-9)))

    qp_dominated_by_lp = sum(strict_dom(p, lp_pts) for p in qp_pts)
    lp_dominated_by_qp = sum(strict_dom(p, qp_pts) for p in lp_pts)
    print(f"\nQP points dominated by LP: {qp_dominated_by_lp}/{len(qp_pts)}")
    print(f"LP points dominated by QP: {lp_dominated_by_qp}/{len(lp_pts)}")
    if lp_dominated_by_qp > qp_dominated_by_lp:
        print(f"-> QP frontier dominates LP at more points. Nonlinear cost has Pareto value.")
    elif qp_dominated_by_lp > lp_dominated_by_qp:
        print(f"-> LP frontier dominates QP at more points. Linear cost is fine.")
    else:
        print(f"-> Frontiers are equivalent. Sweeping mu in linear LP captures the QP frontier.")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(lp_pts[:, 0], lp_pts[:, 1], c="#cc6677", s=60, marker="o",
               label=f"LP linear-cost (sweep mu, n={len(lp_pts)})", zorder=3)
    ax.plot(np.sort(lp_pts[:, 0]), lp_pts[lp_pts[:, 0].argsort(), 1],
            color="#cc6677", lw=1, alpha=0.5)
    ax.scatter(qp_pts[:, 0], qp_pts[:, 1], c="#4477aa", s=60, marker="s",
               label=f"QP quadratic-cost (sweep alpha, n={len(qp_pts)})", zorder=3)
    ax.plot(np.sort(qp_pts[:, 0]), qp_pts[qp_pts[:, 0].argsort(), 1],
            color="#4477aa", lw=1, alpha=0.5)
    ax.set_xlabel("Cumulative revenue")
    ax.set_ylabel("Rainflow D")
    ax.set_title("Perfect-foresight Pareto: LP-linear vs QP-quadratic cycling cost")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "fig_pareto_lp_vs_qp.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")

    return lp_pts, qp_pts, qp_dominated_by_lp, lp_dominated_by_qp


# =============================================================================
# (c) Noise-Pareto: LP-linear-with-forecast vs QP-ensemble
# =============================================================================
def part_c_noise_pareto():
    print("\n" + "=" * 70)
    print("(c) Noise-Pareto: LP-with-forecast frontier vs QP-ensemble frontier")
    print("=" * 70)
    realized_full = synth_price(168, seed=SEED)
    realized = realized_full[:T_QP]
    NOISE = 8
    K = 4   # realistic ensemble size
    fc_seeds = list(range(8))
    rng_pool = np.random.default_rng(789)

    # LP-linear with forecast: sweep mu
    mus = np.concatenate([[0.0], np.logspace(-1, np.log10(realized.max() * 2), 12)])
    lp_pts = []
    for mu in mus:
        Rs, Ds = [], []
        for outer in fc_seeds:
            forecast = make_forecast(realized_full, NOISE, seed=outer)[:T_QP]
            P_chg, P_dis = lp_dispatch(forecast, mu)
            soc, chg, dis = simulate_with_actions(P_chg, P_dis)
            Rs.append(revenue_realized(realized, chg, dis))
            Ds.append(D_of(soc, B_E))
        lp_pts.append((np.mean(Rs), np.mean(Ds), mu))
    lp_pts = np.array(lp_pts)

    # QP-quadratic with K=4 ensemble forecast: sweep alpha
    alphas = np.concatenate([[0.0], np.logspace(-4, 0, 8)])
    qp_pts = []
    for alpha in alphas:
        Rs, Ds = [], []
        for outer in fc_seeds:
            inner_seeds = rng_pool.integers(0, 1_000_000, size=K)
            forecasts = np.stack([
                make_forecast(realized_full, NOISE, seed=int(s))[:T_QP]
                for s in inner_seeds])
            forecast_mean = forecasts.mean(axis=0)
            try:
                P_chg, P_dis = solve_qp(forecast_mean, alpha)
            except Exception:
                continue
            soc, chg, dis = simulate_with_actions(P_chg, P_dis)
            Rs.append(revenue_realized(realized, chg, dis))
            Ds.append(D_of(soc, B_E))
        if Rs:
            qp_pts.append((np.mean(Rs), np.mean(Ds), alpha))
    qp_pts = np.array(qp_pts)

    print(f"\nNoise={NOISE}, K={K}")
    print(f"LP-linear-with-forecast: R in [{lp_pts[:,0].min():.0f}, {lp_pts[:,0].max():.0f}], "
          f"D in [{lp_pts[:,1].min():.5f}, {lp_pts[:,1].max():.5f}]")
    print(f"QP-quadratic-ensemble:   R in [{qp_pts[:,0].min():.0f}, {qp_pts[:,0].max():.0f}], "
          f"D in [{qp_pts[:,1].min():.5f}, {qp_pts[:,1].max():.5f}]")

    def strict_dom(pt, others):
        R, D = pt[0], pt[1]
        return np.any((others[:, 0] >= R - 1e-6) & (others[:, 1] <= D + 1e-9)
                      & ((others[:, 0] > R + 1e-6) | (others[:, 1] < D - 1e-9)))

    qp_dom_lp = sum(strict_dom(p, lp_pts) for p in qp_pts)
    lp_dom_qp = sum(strict_dom(p, qp_pts) for p in lp_pts)
    print(f"\nQP-ensemble dominated by LP-with-forecast: {qp_dom_lp}/{len(qp_pts)}")
    print(f"LP-with-forecast dominated by QP-ensemble: {lp_dom_qp}/{len(lp_pts)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(lp_pts[:, 0], lp_pts[:, 1], c="#cc6677", s=60, marker="o",
               label=f"LP linear-cost with single forecast (n={len(lp_pts)})", zorder=3)
    ax.plot(np.sort(lp_pts[:, 0]), lp_pts[lp_pts[:, 0].argsort(), 1],
            color="#cc6677", lw=1, alpha=0.5)
    ax.scatter(qp_pts[:, 0], qp_pts[:, 1], c="#4477aa", s=60, marker="s",
               label=f"QP quadratic-cost K={K} ensemble (n={len(qp_pts)})", zorder=3)
    ax.plot(np.sort(qp_pts[:, 0]), qp_pts[qp_pts[:, 0].argsort(), 1],
            color="#4477aa", lw=1, alpha=0.5)
    ax.set_xlabel("Realized revenue (averaged over 8 forecast seeds)")
    ax.set_ylabel("Rainflow D")
    ax.set_title(f"Noise-Pareto: LP-with-forecast vs QP-ensemble (noise={NOISE}, K={K})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_pareto_noise.png", dpi=120, bbox_inches="tight")
    print("Wrote fig_pareto_noise.png")
    return lp_pts, qp_pts, qp_dom_lp, lp_dom_qp


def main():
    t0 = time.time()
    a_rows = part_a_k_sweep()
    b_lp, b_qp, b_qp_dom, b_lp_dom = part_b_pareto()
    c_lp, c_qp, c_qp_dom, c_lp_dom = part_c_noise_pareto()
    print(f"\nTotal time: {time.time()-t0:.0f}s")

    # Combined verdict
    K2_lift = (a_rows[1, 1] - a_rows[0, 1]) / max(abs(a_rows[0, 1]), 1e-6) * 100
    K4_lift = (a_rows[2, 1] - a_rows[0, 1]) / max(abs(a_rows[0, 1]), 1e-6) * 100
    print("\n" + "=" * 70)
    print("PRE-FLIGHT VERDICT")
    print("=" * 70)
    print(f"(a) K=2 lift: {K2_lift:+.1f}%  K=4 lift: {K4_lift:+.1f}%")
    print(f"(b) PERFECT-foresight: QP dom by LP {b_qp_dom}/{len(b_qp)}, LP dom by QP {b_lp_dom}/{len(b_lp)}")
    print(f"(c) NOISY-foresight (n=8, K=4): QP-ens dom by LP-fc {c_qp_dom}/{len(c_qp)}, "
          f"LP-fc dom by QP-ens {c_lp_dom}/{len(c_lp)}")

    a_pass = K2_lift > 5 and K4_lift > 8
    b_pass = b_lp_dom > b_qp_dom + 1   # QP dominates LP at more points
    if a_pass and b_pass:
        print("\nBOTH PASS -> commit to refactor.")
    elif a_pass and not b_pass:
        print("\n(a) passes but (b) fails: linear LP captures the Pareto. Quadratic-cost premise weaker.")
        print("    Refactor still defensible but the headline is weaker.")
    elif not a_pass and b_pass:
        print("\n(a) fails but (b) passes: realistic K=2-4 ensembles dont give meaningful lift.")
        print("    Theory works at K>>1 but unlikely deployable.")
    else:
        print("\nBOTH FAIL -> kill the rework.")


if __name__ == "__main__":
    main()
