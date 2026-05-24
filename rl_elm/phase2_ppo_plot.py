"""Render phase 2 PPO results.

Reads gbar_results/ppo_logs/phase2_ppo_results.json (or path passed as argv1),
plots three figures:
  - fig_phase2_ppo_revenue.png : revenue vs noise, all four policies
  - fig_phase2_ppo_lift.png    : lift of RL vs QP-single and QP-ensemble
  - fig_phase2_ppo_pareto.png  : (R, D) at noise=8 for all policies

Usage:
  pixi run python phase2_ppo_plot.py [path/to/phase2_ppo_results.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else "gbar_results/ppo_logs/phase2_ppo_results.json"
    p = Path(p)
    if not p.exists():
        raise SystemExit(f"results file not found: {p}")
    rows = json.loads(p.read_text())
    rows.sort(key=lambda r: r["noise"])
    n = np.array([r["noise"] for r in rows])
    R_lp = np.array([r["R_lp"] for r in rows])
    R_qps = np.array([r["R_qps"] for r in rows])
    R_qpe = np.array([r["R_qpe"] for r in rows])
    R_rl = np.array([r["R_rl"] for r in rows])
    D_lp = np.array([r["D_lp"] for r in rows])
    D_qps = np.array([r["D_qps"] for r in rows])
    D_qpe = np.array([r["D_qpe"] for r in rows])
    D_rl = np.array([r["D_rl"] for r in rows])

    # 1) Revenue vs noise
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n, R_lp, "o-", color="#aaaaaa", label="LP-linear-with-fc")
    ax.plot(n, R_qps, "s-", color="#cc6677", label="QP-quadratic-single")
    ax.plot(n, R_qpe, "^-", color="#4477aa", label="QP-quadratic-K=4 ensemble")
    ax.plot(n, R_rl, "D-", color="#117733", label="PPO (gbar-trained)")
    ax.set_xlabel("Forecast AR(1) noise std (test)")
    ax.set_ylabel("Realized revenue (mean)")
    ax.set_title("Phase 2 v2 (PPO): revenue vs forecast noise")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_phase2_ppo_revenue.png", dpi=120, bbox_inches="tight")
    print("Wrote fig_phase2_ppo_revenue.png")

    # 2) Lift
    eps = 1e-6
    fig, ax = plt.subplots(figsize=(10, 5))
    lift_qps = (R_rl - R_qps) / np.maximum(np.abs(R_qps), eps) * 100
    lift_qpe = (R_rl - R_qpe) / np.maximum(np.abs(R_qpe), eps) * 100
    lift_lp = (R_rl - R_lp) / np.maximum(np.abs(R_lp), eps) * 100
    ax.plot(n, lift_qps, "o-", color="#117733", label="PPO vs QP-single")
    ax.plot(n, lift_qpe, "s-", color="#cc6677", label="PPO vs QP-ensemble")
    ax.plot(n, lift_lp, "^-", color="#aaaaaa", label="PPO vs LP-linear")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Forecast AR(1) noise std (test)")
    ax.set_ylabel("Lift (%)")
    ax.set_title("PPO lift over baselines")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_phase2_ppo_lift.png", dpi=120, bbox_inches="tight")
    print("Wrote fig_phase2_ppo_lift.png")

    # 3) Pareto at noise=8
    idx_8 = int(np.argmin(np.abs(n - 8)))
    fig, ax = plt.subplots(figsize=(8, 6))
    pts = [
        ("LP-linear-with-fc", R_lp[idx_8], D_lp[idx_8], "#aaaaaa", "o"),
        ("QP-quad-single",     R_qps[idx_8], D_qps[idx_8], "#cc6677", "s"),
        ("QP-quad-K=4 ensem",  R_qpe[idx_8], D_qpe[idx_8], "#4477aa", "^"),
        ("PPO",                R_rl[idx_8],  D_rl[idx_8],  "#117733", "D"),
    ]
    for label, R, D, c, m in pts:
        ax.scatter(R, D, color=c, marker=m, s=200, edgecolor="black",
                   lw=0.8, label=f"{label}  R={R:.0f} D={D:.5f}")
    ax.set_xlabel("Realized revenue")
    ax.set_ylabel("Rainflow D")
    ax.set_title(f"(R, D) at test noise={n[idx_8]:.0f}, mean over held-out traces")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_phase2_ppo_pareto.png", dpi=120, bbox_inches="tight")
    print("Wrote fig_phase2_ppo_pareto.png")

    # Summary table
    print("\nSummary:")
    print(f"{'noise':>5} | {'R_lp':>7} | {'R_qps':>7} | {'R_qpe':>7} | {'R_rl':>7} | "
          f"{'lift_vs_qps':>11} | {'lift_vs_qpe':>11}")
    print("-" * 80)
    for i in range(len(n)):
        print(f"{n[i]:5.0f} | {R_lp[i]:7.1f} | {R_qps[i]:7.1f} | {R_qpe[i]:7.1f} | "
              f"{R_rl[i]:7.1f} | {lift_qps[i]:+10.1f}% | {lift_qpe[i]:+10.1f}%")


if __name__ == "__main__":
    main()
