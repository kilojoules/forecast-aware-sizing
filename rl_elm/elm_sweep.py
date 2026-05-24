"""Grid search ELM-RL hyperparameters at B=(2,3). Find any combo beating Naive on D.

Naive baseline at B=(2,3), T=10^5: D_sum ~= 0.532. Goal: ELM beats this on multi-seed average.
"""
from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import ELMQAgent, NaiveAgent
from degradation import fleet_degradation
from env import FleetEnv, FleetSpec
from reg_signal import MarkovSignal


SEEDS = [42, 7, 123]
T_TRAIN = 30_000
T_EVAL = 30_000
# Naive D_sum at T=30k, B=(2,3) ~ 0.16 (1/3 of T=100k baseline 0.532).
NAIVE_BASELINE_T30K = 0.16


def evaluate(agent_factory, seed: int, T_train: int, T_eval: int, spec: FleetSpec) -> tuple[float, float]:
    sig = MarkovSignal(seed=seed)
    env = FleetEnv(spec, sig, alpha_d=1.0, beta=0.1, sp_mode="last_extremum")
    agent = agent_factory(env)
    if hasattr(agent, "train"):
        agent.train(T_train)
    sig_eval = MarkovSignal(seed=seed + 100)
    env_eval = FleetEnv(spec, sig_eval, alpha_d=1.0, beta=0.1, sp_mode="last_extremum")
    env_eval.reset()
    total_R = 0.0
    for _ in range(T_eval):
        actions = env_eval.feasible()
        if isinstance(agent, NaiveAgent):
            a = agent.act(env_eval.b, env_eval.r)
        else:
            agent.env = env_eval
            a = agent.act(actions, greedy=True)
        _, r = env_eval.step(a)
        total_R += r
    Ds, _ = fleet_degradation(env_eval.soc_log, list(spec.B))
    return total_R, sum(Ds)


def main():
    spec = FleetSpec(B=(2, 3), c=(2, 3), d=(2, 3))

    # Coarse grid: 3 hidden x 3 lr x 2 eps decay = 18 combos
    HIDDEN = [50, 100, 200]
    LR = [1e-4, 1e-3, 1e-2]
    EPS_DECAY = [2.5e-4, 5.5e-5]  # slow vs very-slow
    EPS0 = 0.6
    BATCH = 128
    REPLAY = 5000
    GAMMA = 0.95

    grid = list(itertools.product(HIDDEN, LR, EPS_DECAY))
    # First compute Naive baseline at this T.
    naive_Ds = []
    for s in SEEDS:
        _, Dsum = evaluate(lambda env: NaiveAgent(spec), s, T_TRAIN, T_EVAL, spec)
        naive_Ds.append(Dsum)
    naive_baseline = statistics.mean(naive_Ds)
    print(f"Naive baseline D_sum at T={T_EVAL}: {naive_baseline:.4f} ± {statistics.stdev(naive_Ds):.4f}\n", flush=True)
    print(f"Sweep: {len(grid)} configs x {len(SEEDS)} seeds = {len(grid)*len(SEEDS)} runs", flush=True)

    results = []
    for h, lr, edec in grid:
        D_sums = []
        rewards = []
        for s in SEEDS:
            def make(env, h=h, lr=lr, edec=edec, s=s):
                return ELMQAgent(env, hidden=h, gamma=GAMMA, alpha=lr, eps0=EPS0,
                                 eps_decay=edec, batch=BATCH, replay=REPLAY, seed=s)
            R, Dsum = evaluate(make, s, T_TRAIN, T_EVAL, spec)
            D_sums.append(Dsum)
            rewards.append(R)
        m_D = statistics.mean(D_sums)
        s_D = statistics.stdev(D_sums)
        m_R = statistics.mean(rewards)
        beat = "YES" if m_D < naive_baseline else "no"
        print(f"hidden={h:3d}  lr={lr:.0e}  decay={edec:.1e}  -> D={m_D:.4f}±{s_D:.4f}  R={m_R:.0f}  beats_Naive={beat}", flush=True)
        results.append({"hidden": h, "lr": lr, "decay": edec, "D_mean": m_D, "D_std": s_D, "R_mean": m_R, "beats": bool(m_D < naive_baseline)})

    results.sort(key=lambda r: r["D_mean"])
    print(f"\nBest combo: {results[0]}")
    Path("elm_sweep_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
