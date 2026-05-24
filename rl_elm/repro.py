"""Reproduce the headline result on a laptop in ~5 minutes.

Trains ELM-RL at B=(10,100), c=(5,50), d=(5,50), T=200k, then plots
per-battery DoD histograms (Naive vs ELM). Naive over-cycles the small
battery; ELM redistributes regulation.

Usage:
    python repro.py            # run + save fig_headline.png + print numbers
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from agents import ELMQAgent, NaiveAgent
from degradation import fleet_degradation
from env import FleetEnv, FleetSpec
from reg_signal import MarkovSignal


SPEC = FleetSpec(B=(10, 100), c=(5, 50), d=(5, 50))
T_TRAIN = 200_000
T_EVAL = 50_000
SEED = 42


def collect_dod(soc_log, B):
    from degradation import cycle_degradation
    D, dods, counts = cycle_degradation(soc_log, B)
    return D, np.asarray(dods), np.asarray(counts)


def main():
    print(f"Headline reproduction: B={SPEC.B}  T_train={T_TRAIN:,}  T_eval={T_EVAL:,}")
    print("This trains ELM-RL once locally (~5 min) and plots per-battery DoD.\n")

    # Naive baseline
    sig_n = MarkovSignal(seed=SEED + 100)
    env_n = FleetEnv(SPEC, sig_n, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode="quad_growth")
    env_n.reset()
    naive = NaiveAgent(SPEC)
    for _ in range(T_EVAL):
        a = naive.act(env_n.b, env_n.r)
        env_n.step(a)
    Ds_naive, _ = fleet_degradation(env_n.soc_log, list(SPEC.B))

    # Train ELM
    sig = MarkovSignal(seed=SEED)
    env = FleetEnv(SPEC, sig, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode="quad_growth")
    elm = ELMQAgent(env, hidden=200, gamma=0.95, alpha=0.01, eps0=0.6,
                    eps_decay=1e-5, batch=128, replay=5000, seed=SEED)
    t0 = time.time()
    elm.train(T_TRAIN)
    print(f"  trained ELM in {time.time()-t0:.0f}s\n")

    # Eval ELM on the same held-out trajectory
    sig_e = MarkovSignal(seed=SEED + 100)
    env_e = FleetEnv(SPEC, sig_e, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode="quad_growth")
    env_e.reset()
    elm.env = env_e
    for _ in range(T_EVAL):
        a = elm.act(env_e.feasible(), greedy=True)
        env_e.step(a)
    Ds_elm, _ = fleet_degradation(env_e.soc_log, list(SPEC.B))

    print(f"Naive  D per battery: {[f'{d:.4f}' for d in Ds_naive]}  sum={sum(Ds_naive):.4f}")
    print(f"ELM-RL D per battery: {[f'{d:.4f}' for d in Ds_elm]}  sum={sum(Ds_elm):.4f}")
    print(f"ELM vs Naive: {(sum(Ds_naive)-sum(Ds_elm))/sum(Ds_naive)*100:+.1f}% lower D")

    # Two-panel plot: per-battery DoD histogram (Naive vs ELM)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, 1, 21)
    for i, ax in enumerate(axes):
        D_n, dods_n, counts_n = collect_dod(env_n.soc_log[i], SPEC.B[i])
        D_e, dods_e, counts_e = collect_dod(env_e.soc_log[i], SPEC.B[i])
        ax.hist(dods_n, bins=bins, weights=counts_n, alpha=0.6, label=f"Naive  D={D_n:.4f}", color="#aaaaaa")
        ax.hist(dods_e, bins=bins, weights=counts_e, alpha=0.6, label=f"ELM-RL D={D_e:.4f}", color="#cc6677")
        ax.set_title(f"Battery {i+1}  (capacity B={SPEC.B[i]})")
        ax.set_xlabel("DoD (fraction of capacity)")
        ax.set_ylabel("Cycle count")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Naive over-cycles the small battery; ELM-RL redistributes regulation."
        f"\nELM total D = {sum(Ds_elm):.3f} vs Naive {sum(Ds_naive):.3f}  "
        f"(reduction {(sum(Ds_naive)-sum(Ds_elm))/sum(Ds_naive)*100:.1f}%)",
        y=1.02,
    )
    fig.tight_layout()
    out = "fig_headline.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
