"""Phase 2: train + evaluate behavior-cloning RL agent on hourly arbitrage.

Trains a 2-layer MLP on (state, target_action) pairs from QP-ensemble
trajectories at noise=8, K=4. Evaluates on held-out (realized, single-forecast)
pairs against three baselines: LP-with-forecast, QP-single, QP-ensemble.

If learned policy beats QP-single under noise, the rework's RL claim is
empirically supported. If it matches QP-ensemble, the value is inference
cost (1 NN forward pass vs K QP solves).

Run:
  pixi run python phase2_benchmark.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
import torch

from arbitrage_agents import (lp_linear_actions, qp_ensemble_actions,
                              qp_quadratic_actions, run_actions)
from degradation import cycle_degradation
from env import PriceEnv
from price_signal import make_forecast, synth_diurnal
from rl_agent import (build_dataset, load_policy, policy_dispatch,
                      save_policy, train_policy)


# Config
T = 168
B_E, B_P = 2.0, 2.0
SOC0 = 1.0
ALPHA = 0.005
MU = 5.0
TRAIN_NOISE = 8
EVAL_NOISES = [3, 5, 8, 12, 18]
K_ENSEMBLE = 4
LOOKAHEAD = 72
N_TRAIN_TRACES = 400
EVAL_TRACES = 30
EPOCHS = 200


def D_safe(soc, B):
    try:
        D, _, _ = cycle_degradation(soc, B)
    except (IndexError, ValueError):
        return 0.0
    return D


def main():
    print(f"Phase 2: train BC-RL on T={T}, alpha={ALPHA}, train_noise={TRAIN_NOISE}, K={K_ENSEMBLE}")
    print(f"  N_train={N_TRAIN_TRACES} traces, lookahead={LOOKAHEAD}h, epochs={EPOCHS}")

    # 1) Generate training set
    t0 = time.time()
    print(f"\n[1/4] Building training dataset ({N_TRAIN_TRACES} traces)...")
    X, y = build_dataset(N_traces=N_TRAIN_TRACES, T=T, b_E=B_E, b_P=B_P, soc0=SOC0,
                         noise_std=TRAIN_NOISE, K_ensemble=K_ENSEMBLE,
                         alpha=ALPHA, lookahead=LOOKAHEAD, target="ensemble",
                         base_seed=42, verbose=True)
    print(f"  X={X.shape}, y={y.shape}  ({time.time()-t0:.0f}s)")

    # 2) Train policy
    t0 = time.time()
    print(f"\n[2/4] Training MLP policy...")
    pi, hist = train_policy(X, y, b_P=B_P, epochs=EPOCHS, batch=512,
                            lr=1e-3, hidden=128, layers=3, seed=0,
                            val_frac=0.1, verbose=True)
    save_policy(pi, "rl_policy.pt", meta={"trained_on_noise": TRAIN_NOISE,
                                            "K_ensemble": K_ENSEMBLE,
                                            "lookahead": LOOKAHEAD,
                                            "alpha": ALPHA,
                                            "T": T,
                                            "b_E": B_E, "b_P": B_P,
                                            "N_train": N_TRAIN_TRACES})
    print(f"  Saved rl_policy.pt  ({time.time()-t0:.0f}s)")

    # 3) Evaluate vs baselines on held-out traces, sweep noise
    print(f"\n[3/4] Evaluating on {EVAL_TRACES} held-out traces, sweep noise...")
    rng = np.random.default_rng(7)
    held_out_seeds = rng.integers(10_000, 1_000_000, size=EVAL_TRACES)
    results = {n: {"lp": [], "qps": [], "qpe": [], "rl": []} for n in EVAL_NOISES}
    Ds = {n: {"lp": [], "qps": [], "qpe": [], "rl": []} for n in EVAL_NOISES}

    for noise in EVAL_NOISES:
        t0 = time.time()
        for tr_i, real_seed in enumerate(held_out_seeds):
            realized = synth_diurnal(T, seed=int(real_seed))
            fc_seed = int(rng.integers(10_000, 1_000_000))
            forecast_single = make_forecast(realized, noise, seed=fc_seed)

            inner = rng.integers(10_000, 1_000_000, size=K_ENSEMBLE)
            forecasts_K = np.stack([
                make_forecast(realized, noise, seed=int(s)) for s in inner])

            env = PriceEnv(B_E, B_P, prices=realized, soc0=SOC0, alpha=ALPHA,
                           reward_mode="quadratic")

            # LP-linear with single forecast
            a = lp_linear_actions(forecast_single, B_E, B_P, SOC0, mu=MU)
            rec = run_actions(env, a)
            results[noise]["lp"].append(rec["R"])
            Ds[noise]["lp"].append(D_safe(rec["soc_log"], B_E))
            # QP-quadratic single forecast
            a = qp_quadratic_actions(forecast_single, B_E, B_P, SOC0, alpha=ALPHA)
            rec = run_actions(env, a)
            results[noise]["qps"].append(rec["R"])
            Ds[noise]["qps"].append(D_safe(rec["soc_log"], B_E))
            # QP-ensemble (K)
            a = qp_ensemble_actions(forecasts_K, B_E, B_P, SOC0, alpha=ALPHA)
            rec = run_actions(env, a)
            results[noise]["qpe"].append(rec["R"])
            Ds[noise]["qpe"].append(D_safe(rec["soc_log"], B_E))
            # RL policy on single forecast
            rec = policy_dispatch(pi, env, forecast_single, lookahead=LOOKAHEAD)
            results[noise]["rl"].append(rec["R"])
            Ds[noise]["rl"].append(D_safe(rec["soc_log"], B_E))
        print(f"  noise={noise}: done in {time.time()-t0:.0f}s")

    # 4) Report + plot
    print(f"\n[4/4] Results:")
    print(f"{'noise':>5} | {'R_lp':>7} | {'R_qps':>7} | {'R_qpe':>7} | {'R_rl':>7} | "
          f"{'lift_rl_vs_qps':>15} | {'lift_rl_vs_qpe':>14}")
    print("-" * 95)
    rows = []
    for noise in EVAL_NOISES:
        r_lp = np.mean(results[noise]["lp"])
        r_qps = np.mean(results[noise]["qps"])
        r_qpe = np.mean(results[noise]["qpe"])
        r_rl = np.mean(results[noise]["rl"])
        lift_qps = (r_rl - r_qps) / max(abs(r_qps), 1e-6) * 100
        lift_qpe = (r_rl - r_qpe) / max(abs(r_qpe), 1e-6) * 100
        rows.append((noise, r_lp, r_qps, r_qpe, r_rl, lift_qps, lift_qpe))
        print(f"{noise:5d} | {r_lp:7.1f} | {r_qps:7.1f} | {r_qpe:7.1f} | {r_rl:7.1f} | "
              f"{lift_qps:+13.1f}%  | {lift_qpe:+12.1f}%")

    rows = np.array(rows, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(rows[:, 0], rows[:, 1], "o-", color="#aaaaaa", label="LP-linear-with-fc")
    ax.plot(rows[:, 0], rows[:, 2], "s-", color="#cc6677", label="QP-quadratic-single")
    ax.plot(rows[:, 0], rows[:, 3], "^-", color="#4477aa", label=f"QP-quadratic-K={K_ENSEMBLE}")
    ax.plot(rows[:, 0], rows[:, 4], "D-", color="#117733", label="RL (BC-MLP)")
    ax.set_xlabel("Forecast AR(1) noise std (test)")
    ax.set_ylabel(f"Realized revenue (mean over {EVAL_TRACES} traces)")
    ax.set_title("Phase 2: BC-RL vs three baselines")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(rows[:, 0], rows[:, 5], "o-", color="#117733", label="RL vs QP-single")
    ax.plot(rows[:, 0], rows[:, 6], "s-", color="#cc6677", label="RL vs QP-ensemble")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Forecast AR(1) noise std (test)")
    ax.set_ylabel("Lift (%)")
    ax.set_title("RL lift over single and ensemble baselines")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_phase2_benchmark.png", dpi=120, bbox_inches="tight")
    print("\nWrote fig_phase2_benchmark.png")

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["train_loss"], label="train")
    ax.plot(hist["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("BC-MLP training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_phase2_train.png", dpi=120, bbox_inches="tight")
    print("Wrote fig_phase2_train.png")


if __name__ == "__main__":
    main()
