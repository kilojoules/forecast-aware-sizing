"""Proxy reward vs rainflow D correlation: is the proxy a good surrogate?

For each (config, policy, seed), record (cumulative proxy reward, rainflow D).
A working proxy gives a tight monotone scatter; a noisy/non-monotone scatter
explains why D-best and reward-best policies disagree.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from agents import ELMQAgent, GreedyAgent, NaiveAgent
from degradation import fleet_degradation
from env import FleetEnv, FleetSpec
from reg_signal import MarkovSignal


CONFIGS = [
    FleetSpec(B=(2, 20), c=(2, 10), d=(2, 10)),
    FleetSpec(B=(5, 20), c=(2, 10), d=(2, 10)),
    FleetSpec(B=(2, 50), c=(2, 25), d=(2, 25)),
    FleetSpec(B=(5, 50), c=(2, 25), d=(2, 25)),
    FleetSpec(B=(10, 100), c=(5, 50), d=(5, 50)),
]


def eval_naive(spec, seed, T):
    sig = MarkovSignal(seed=seed)
    env = FleetEnv(spec, sig, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode="quad_growth")
    env.reset()
    naive = NaiveAgent(spec)
    R = 0.0
    for _ in range(T):
        a = naive.act(env.b, env.r)
        _, r = env.step(a)
        R += r
    Ds, _ = fleet_degradation(env.soc_log, list(spec.B))
    return R, sum(Ds)


def eval_greedy(spec, seed, T):
    sig = MarkovSignal(seed=seed)
    env = FleetEnv(spec, sig, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode="quad_growth")
    env.reset()
    g = GreedyAgent(env)
    R = 0.0
    for _ in range(T):
        g.env = env
        a = g.act(env.b, env.r)
        _, r = env.step(a)
        R += r
    Ds, _ = fleet_degradation(env.soc_log, list(spec.B))
    return R, sum(Ds)


def eval_elm(spec, seed, T_train, T_eval, **kwargs):
    sig = MarkovSignal(seed=seed)
    env_train = FleetEnv(spec, sig, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode="quad_growth")
    elm = ELMQAgent(env_train, hidden=kwargs.get("hidden", 200), gamma=0.95,
                    alpha=kwargs.get("lr", 0.01), eps0=kwargs.get("eps0", 0.6),
                    eps_decay=kwargs.get("eps_decay", 1e-5), batch=128, replay=5000, seed=seed)
    elm.train(T_train)
    sig_e = MarkovSignal(seed=seed + 100)
    env = FleetEnv(spec, sig_e, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode="quad_growth")
    env.reset()
    elm.env = env
    R = 0.0
    for _ in range(T_eval):
        a = elm.act(env.feasible(), greedy=True)
        _, r = env.step(a)
        R += r
    Ds, _ = fleet_degradation(env.soc_log, list(spec.B))
    return R, sum(Ds)


def main():
    T_eval = 30_000
    T_train = 100_000
    seeds = [42, 7, 123]
    points = []
    t0 = time.time()
    for spec in CONFIGS:
        for seed in seeds:
            for name, fn in [("Naive", lambda: eval_naive(spec, seed, T_eval)),
                             ("Greedy", lambda: eval_greedy(spec, seed, T_eval))]:
                R, D = fn()
                points.append({"config": str(spec.B), "seed": seed, "agent": name, "R": R, "D": D})
                print(f"  {name:7s} B={spec.B} seed={seed}: R={R:.1f} D={D:.4f}", flush=True)
            # Vary ELM hyperparams to span the policy space
            for tag, kw in [("ELM_default", dict()),
                            ("ELM_smalllr", dict(lr=1e-3)),
                            ("ELM_largeeps", dict(eps0=0.9, eps_decay=5e-6))]:
                R, D = eval_elm(spec, seed, T_train, T_eval, **kw)
                points.append({"config": str(spec.B), "seed": seed, "agent": tag, "R": R, "D": D})
                print(f"  {tag:14s} B={spec.B} seed={seed}: R={R:.1f} D={D:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    Path("proxy_d_corr.json").write_text(json.dumps(points, indent=2))

    # Scatter
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4 * len(CONFIGS), 4), sharey=False)
    for ax, spec in zip(axes, CONFIGS):
        cfg = str(spec.B)
        sub = [p for p in points if p["config"] == cfg]
        agents = sorted({p["agent"] for p in sub})
        colors = plt.cm.tab10.colors
        for ai, agent in enumerate(agents):
            xs = [p["R"] for p in sub if p["agent"] == agent]
            ys = [p["D"] for p in sub if p["agent"] == agent]
            ax.scatter(xs, ys, label=agent, color=colors[ai % len(colors)], s=40)
        ax.set_xlabel("Cumulative proxy reward")
        ax.set_ylabel("Rainflow D")
        ax.set_title(f"B={cfg}")
        ax.grid(alpha=0.3)
    axes[-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Proxy reward vs final rainflow D (T_eval=30k, 3 seeds, 5 policies)", y=1.02)
    fig.tight_layout()
    fig.savefig("fig_proxy_d_corr.png", dpi=120, bbox_inches="tight")
    print("\nWrote fig_proxy_d_corr.png + proxy_d_corr.json")


if __name__ == "__main__":
    main()
