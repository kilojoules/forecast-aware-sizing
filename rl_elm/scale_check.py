"""Sanity-scale ELM-RL at mid-size configs before committing to LUMI.

Trains ELM-RL at B=(10,10) and B=(20,20). Reports D_sum and timing per 100k steps.
Uses the best ELM hyperparams from elm_sweep.py output (or override via CLI).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import ELMQAgent, GreedyAgent, NaiveAgent
from degradation import fleet_degradation
from env import FleetEnv, FleetSpec
from reg_signal import MarkovSignal

REWARD_MODE = "exp_growth"


def run_one(spec: FleetSpec, T: int, hidden: int, lr: float, eps0: float,
            eps_decay: float, seed: int) -> dict:
    rng = MarkovSignal(seed=seed)
    env = FleetEnv(spec, rng, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode=REWARD_MODE)
    elm = ELMQAgent(env, hidden=hidden, gamma=0.95, alpha=lr, eps0=eps0,
                    eps_decay=eps_decay, batch=128, replay=5000, seed=seed)
    t0 = time.time()
    elm.train(T)
    train_time = time.time() - t0

    sig_eval = MarkovSignal(seed=seed + 100)
    env_eval = FleetEnv(spec, sig_eval, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode=REWARD_MODE)
    env_eval.reset()
    elm.env = env_eval
    R_elm = 0.0
    for _ in range(T):
        actions = env_eval.feasible()
        a = elm.act(actions, greedy=True)
        _, r = env_eval.step(a)
        R_elm += r
    Ds_elm, _ = fleet_degradation(env_eval.soc_log, list(spec.B))

    naive = NaiveAgent(spec)
    sig_eval2 = MarkovSignal(seed=seed + 100)
    env_n = FleetEnv(spec, sig_eval2, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode=REWARD_MODE)
    env_n.reset()
    R_naive = 0.0
    for _ in range(T):
        a = naive.act(env_n.b, env_n.r)
        _, r = env_n.step(a)
        R_naive += r
    Ds_naive, _ = fleet_degradation(env_n.soc_log, list(spec.B))

    sig_eval3 = MarkovSignal(seed=seed + 100)
    env_g = FleetEnv(spec, sig_eval3, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode=REWARD_MODE)
    env_g.reset()
    greedy = GreedyAgent(env_g)
    R_greedy = 0.0
    for _ in range(T):
        greedy.env = env_g
        a = greedy.act(env_g.b, env_g.r)
        _, r = env_g.step(a)
        R_greedy += r
    Ds_greedy, _ = fleet_degradation(env_g.soc_log, list(spec.B))

    return {
        "B": spec.B, "c": spec.c, "d": spec.d, "T": T, "hidden": hidden, "seed": seed,
        "elm": {"R": R_elm, "D_per": [float(d) for d in Ds_elm], "D_sum": float(sum(Ds_elm)), "train_s": train_time},
        "naive": {"R": R_naive, "D_per": [float(d) for d in Ds_naive], "D_sum": float(sum(Ds_naive))},
        "greedy": {"R": R_greedy, "D_per": [float(d) for d in Ds_greedy], "D_sum": float(sum(Ds_greedy))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100_000)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eps0", type=float, default=0.6)
    ap.add_argument("--eps-decay", type=float, default=2.5e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    ap.add_argument("--reward", choices=["exp_growth", "quad_growth"], default="exp_growth")
    ap.add_argument("--config", choices=["sym", "het", "all"], default="sym")
    args = ap.parse_args()
    global REWARD_MODE
    REWARD_MODE = args.reward

    sym_configs = [
        FleetSpec(B=(10, 10), c=(5, 5), d=(5, 5)),
        FleetSpec(B=(20, 20), c=(10, 10), d=(10, 10)),
    ]
    het_configs = [
        FleetSpec(B=(5, 20), c=(2, 10), d=(2, 10)),
        FleetSpec(B=(2, 20), c=(2, 10), d=(2, 10)),
    ]
    configs = {"sym": sym_configs, "het": het_configs, "all": sym_configs + het_configs}[args.config]
    results = []
    for spec in configs:
        per_seed = []
        for s in args.seeds:
            r = run_one(spec, args.T, args.hidden, args.lr, args.eps0, args.eps_decay, s)
            per_seed.append(r)
            print(
                f"B={spec.B}  seed={s}  ELM D={r['elm']['D_sum']:.4f}  "
                f"Greedy D={r['greedy']['D_sum']:.4f}  "
                f"Naive D={r['naive']['D_sum']:.4f}  "
                f"ELMvsNaive={(r['naive']['D_sum']-r['elm']['D_sum'])/r['naive']['D_sum']*100:+.1f}%  "
                f"ELMvsGreedy={(r['greedy']['D_sum']-r['elm']['D_sum'])/r['greedy']['D_sum']*100:+.1f}%  "
                f"train_s={r['elm']['train_s']:.0f}",
                flush=True,
            )
        elm_Ds = [x["elm"]["D_sum"] for x in per_seed]
        naive_Ds = [x["naive"]["D_sum"] for x in per_seed]
        elm_m = statistics.mean(elm_Ds)
        naive_m = statistics.mean(naive_Ds)
        improvement = (naive_m - elm_m) / naive_m * 100 if naive_m else 0
        print(f"  -> B={spec.B}: ELM mean={elm_m:.4f}, Naive mean={naive_m:.4f}, ELM is {improvement:+.1f}% better\n", flush=True)
        results.append({"B": spec.B, "elm_mean": elm_m, "naive_mean": naive_m, "improvement_pct": improvement, "per_seed": per_seed})

    Path("scale_check_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
