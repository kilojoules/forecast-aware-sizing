"""Phase 1 benchmark: reproduce the Jensen lift at full T=168 scale.

Uses the new PriceEnv + arbitrage_agents API. Sweeps forecast noise,
runs three policies (LP-linear-with-forecast, QP-quadratic-with-single
forecast, QP-quadratic-with-K=4-ensemble), reports realized revenue +
post-hoc rainflow D + Pareto plot.

This is the headline figure for the rework. If it reproduces the
sanity_jensen.py result at T=168 with the new API, phase 1 is done.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from arbitrage_agents import (lp_linear_actions, qp_ensemble_actions,
                              qp_quadratic_actions, run_actions)
from degradation import cycle_degradation
from env import PriceEnv
from price_signal import make_forecast, synth_diurnal


def D_safe(soc, B):
    try:
        D, _, _ = cycle_degradation(soc, B)
    except (IndexError, ValueError):
        return 0.0
    return D


def main():
    T = 168
    B_E, B_P = 2.0, 2.0
    SEED = 42
    NOISES = [3, 5, 8, 12, 18]
    K = 4
    ALPHA = 0.005
    MU = 5.0    # tuned to get LP cycling
    FC_SEEDS = list(range(8))
    rng = np.random.default_rng(11)

    print(f"Phase 1 benchmark: T={T} B_E={B_E} B_P={B_P} alpha={ALPHA} mu={MU}\n")
    realized = synth_diurnal(T, seed=SEED)

    # Oracle baselines for context
    env = PriceEnv(B_E, B_P, prices=realized, mu=MU, alpha=ALPHA,
                   reward_mode="quadratic")
    a_oracle_qp = qp_quadratic_actions(realized, B_E, B_P, env.soc0, ALPHA)
    rec = run_actions(env, a_oracle_qp)
    R_oracle_qp = rec["R"]
    D_oracle_qp = D_safe(rec["soc_log"], B_E)
    print(f"QP oracle (perfect foresight): R={R_oracle_qp:.1f}  D={D_oracle_qp:.5f}")

    a_oracle_lp = lp_linear_actions(realized, B_E, B_P, env.soc0, MU)
    rec = run_actions(env, a_oracle_lp)
    R_oracle_lp = rec["R"]
    D_oracle_lp = D_safe(rec["soc_log"], B_E)
    print(f"LP oracle (perfect foresight): R={R_oracle_lp:.1f}  D={D_oracle_lp:.5f}\n")

    # Sweep noise: compare three policies under noisy forecasts
    rows = []
    print(f"{'noise':>5} | {'R_lp_fc':>8} | {'R_qp_single':>12} | {'R_qp_ensem':>11} | "
          f"{'D_lp':>7} | {'D_qp_s':>7} | {'D_qp_e':>7}")
    print("-" * 90)
    for noise in NOISES:
        Rs_lp, Rs_qps, Rs_qpe = [], [], []
        Ds_lp, Ds_qps, Ds_qpe = [], [], []
        for outer in FC_SEEDS:
            forecast_single = make_forecast(realized, noise, seed=outer)
            inner_seeds = rng.integers(0, 1_000_000, size=K)
            forecasts_K = np.stack([
                make_forecast(realized, noise, seed=int(s)) for s in inner_seeds])

            # LP linear with single forecast
            a_lp = lp_linear_actions(forecast_single, B_E, B_P, env.soc0, MU)
            rec = run_actions(env, a_lp)
            Rs_lp.append(rec["R"])
            Ds_lp.append(D_safe(rec["soc_log"], B_E))

            # QP quadratic with single forecast
            a_qps = qp_quadratic_actions(forecast_single, B_E, B_P, env.soc0, ALPHA)
            rec = run_actions(env, a_qps)
            Rs_qps.append(rec["R"])
            Ds_qps.append(D_safe(rec["soc_log"], B_E))

            # QP quadratic with K-ensemble forecast
            a_qpe = qp_ensemble_actions(forecasts_K, B_E, B_P, env.soc0, ALPHA)
            rec = run_actions(env, a_qpe)
            Rs_qpe.append(rec["R"])
            Ds_qpe.append(D_safe(rec["soc_log"], B_E))

        row = (noise,
               np.mean(Rs_lp), np.mean(Rs_qps), np.mean(Rs_qpe),
               np.mean(Ds_lp), np.mean(Ds_qps), np.mean(Ds_qpe))
        rows.append(row)
        print(f"{noise:5.1f} | {row[1]:8.1f} | {row[2]:12.1f} | {row[3]:11.1f} | "
              f"{row[4]:.5f} | {row[5]:.5f} | {row[6]:.5f}")

    rows = np.array(rows)

    # Headline plot: revenue vs noise per policy
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(rows[:, 0], rows[:, 1], "o-", color="#aaaaaa", label="LP-linear-with-forecast")
    ax.plot(rows[:, 0], rows[:, 2], "s-", color="#cc6677", label="QP-quadratic-single-forecast")
    ax.plot(rows[:, 0], rows[:, 3], "^-", color="#4477aa", label=f"QP-quadratic-K={K}-ensemble")
    ax.axhline(R_oracle_lp, color="black", lw=0.8, ls="--", label=f"LP oracle={R_oracle_lp:.0f}")
    ax.axhline(R_oracle_qp, color="black", lw=0.8, ls=":", label=f"QP oracle={R_oracle_qp:.0f}")
    ax.set_xlabel("Forecast AR(1) noise std")
    ax.set_ylabel("Realized revenue")
    ax.set_title(f"Phase 1: revenue vs noise (T={T}, alpha={ALPHA}, mu={MU})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Lift plot: (qp-ensemble - qp-single) / qp-single
    ax = axes[1]
    lift_ens = (rows[:, 3] - rows[:, 2]) / np.maximum(np.abs(rows[:, 2]), 1e-6) * 100
    lift_lp_vs_qpe = (rows[:, 3] - rows[:, 1]) / np.maximum(np.abs(rows[:, 1]), 1e-6) * 100
    ax.plot(rows[:, 0], lift_ens, "^-", color="#4477aa", label="QP-ensemble vs QP-single")
    ax.plot(rows[:, 0], lift_lp_vs_qpe, "o-", color="#117733", label="QP-ensemble vs LP-with-fc")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Forecast AR(1) noise std")
    ax.set_ylabel("Lift (%)")
    ax.set_title("Ensemble lift over single-forecast baselines")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Phase 1 benchmark: PriceEnv + arbitrage_agents at full scale", y=1.02)
    fig.tight_layout()
    out = "fig_phase1_benchmark.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")

    # Pareto plot at chosen mid-noise level
    PARETO_NOISE = 8
    print(f"\nPareto sweep at noise={PARETO_NOISE} (sweep mu, alpha)...")
    mus = np.concatenate([[0.0], np.logspace(-1, 2, 10)])
    alphas = np.concatenate([[0.0], np.logspace(-4, -1, 8)])
    fc_seeds_pareto = list(range(4))

    lp_pts, qp_pts, qpe_pts = [], [], []
    for mu in mus:
        Rs, Ds = [], []
        for outer in fc_seeds_pareto:
            fc = make_forecast(realized, PARETO_NOISE, seed=outer)
            a = lp_linear_actions(fc, B_E, B_P, env.soc0, mu)
            rec = run_actions(env, a)
            Rs.append(rec["R"])
            Ds.append(D_safe(rec["soc_log"], B_E))
        lp_pts.append((np.mean(Rs), np.mean(Ds)))
    for alpha in alphas:
        Rs, Ds = [], []
        Rs_e, Ds_e = [], []
        for outer in fc_seeds_pareto:
            fc = make_forecast(realized, PARETO_NOISE, seed=outer)
            a = qp_quadratic_actions(fc, B_E, B_P, env.soc0, alpha)
            rec = run_actions(env, a)
            Rs.append(rec["R"])
            Ds.append(D_safe(rec["soc_log"], B_E))
            inner_seeds = rng.integers(0, 1_000_000, size=K)
            fc_K = np.stack([make_forecast(realized, PARETO_NOISE, seed=int(s)) for s in inner_seeds])
            a_e = qp_ensemble_actions(fc_K, B_E, B_P, env.soc0, alpha)
            rec_e = run_actions(env, a_e)
            Rs_e.append(rec_e["R"])
            Ds_e.append(D_safe(rec_e["soc_log"], B_E))
        qp_pts.append((np.mean(Rs), np.mean(Ds)))
        qpe_pts.append((np.mean(Rs_e), np.mean(Ds_e)))
    lp_pts = np.array(lp_pts)
    qp_pts = np.array(qp_pts)
    qpe_pts = np.array(qpe_pts)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(lp_pts[:, 0], lp_pts[:, 1], c="#aaaaaa", s=60, marker="o",
               label=f"LP-linear-with-forecast ({len(lp_pts)} mu)", zorder=3)
    ax.scatter(qp_pts[:, 0], qp_pts[:, 1], c="#cc6677", s=60, marker="s",
               label=f"QP-quadratic-single ({len(qp_pts)} alpha)", zorder=3)
    ax.scatter(qpe_pts[:, 0], qpe_pts[:, 1], c="#4477aa", s=60, marker="^",
               label=f"QP-quadratic-K={K}-ensemble ({len(qpe_pts)} alpha)", zorder=3)
    ax.set_xlabel("Realized revenue (averaged)")
    ax.set_ylabel("Rainflow D")
    ax.set_title(f"Phase 1 Pareto: noise={PARETO_NOISE} on T={T}h trace")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_phase1_pareto.png", dpi=120, bbox_inches="tight")
    print("Wrote fig_phase1_pareto.png")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.0f}s")
