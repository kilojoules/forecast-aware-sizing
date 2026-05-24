"""Reproduce toy Markov experiment from §5.2: compare Naive, Greedy, RL agents.

Validation criterion: on (B1,B2)=(2,3), RL produces lower per-battery rainflow
degradation AND higher accumulated reward than both baselines, AND DoD
histograms of RL are concentrated at smaller cycle depths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agents import ELMQAgent, GreedyAgent, NaiveAgent, TabularQ
from degradation import fleet_degradation
from env import FleetEnv, FleetSpec, feasible_actions
from reg_signal import MarkovSignal


def sanity_check_action_count(spec: FleetSpec, sig_support: np.ndarray) -> int:
    total = 0
    for b1 in range(spec.B[0] + 1):
        for b2 in range(spec.B[1] + 1):
            for r in sig_support:
                total += len(feasible_actions(spec, (b1, b2), int(r)))
    return total


def run_agent(name, agent_factory, spec, sig_kwargs, T, seed_run=42, sp_mode="last_extremum",
              reward_mode="quad_growth", alpha_d=1.0, beta=0.1, log_traces=True, train_kwargs=None):
    sig = MarkovSignal(seed=seed_run, **sig_kwargs)
    env = FleetEnv(spec, sig, alpha_d=alpha_d, beta=beta, sp_mode=sp_mode, reward_mode=reward_mode)
    agent = agent_factory(env)
    if hasattr(agent, "train"):
        agent.train(T, **(train_kwargs or {}))
    sig_eval = MarkovSignal(seed=seed_run + 100, **sig_kwargs)
    env_eval = FleetEnv(spec, sig_eval, alpha_d=alpha_d, beta=beta, sp_mode=sp_mode, reward_mode=reward_mode)
    env_eval.reset()
    total_R = 0.0
    for _ in range(T):
        actions = env_eval.feasible()
        if isinstance(agent, NaiveAgent):
            a = agent.act(env_eval.b, env_eval.r)
        elif isinstance(agent, GreedyAgent):
            agent.env = env_eval  # greedy uses env.b_sp directly
            a = agent.act(env_eval.b, env_eval.r)
        elif isinstance(agent, TabularQ):
            dirs = tuple(t.dir for t in env_eval.trackers)
            key = agent._state_key(env_eval.b, env_eval.r, tuple(env_eval.b_sp), dirs)
            a = agent.act(key, actions, greedy=True)
        elif isinstance(agent, ELMQAgent):
            agent.env = env_eval
            a = agent.act(actions, greedy=True)
        else:
            raise TypeError(name)
        _, r_t = env_eval.step(a)
        total_R += r_t
    Ds, histos = fleet_degradation(env_eval.soc_log, list(spec.B))
    out = {
        "name": name,
        "reward": float(total_R),
        "D_per_battery": [float(d) for d in Ds],
        "dod_hist": [(list(map(float, dods)), list(map(float, counts))) for dods, counts in histos],
    }
    if log_traces:
        out["soc_log"] = [list(map(int, traj)) for traj in env_eval.soc_log]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, nargs=2, default=[2, 3])
    ap.add_argument("--c", type=int, nargs=2, default=[2, 3])
    ap.add_argument("--d", type=int, nargs=2, default=[2, 3])
    ap.add_argument("--T", type=int, default=20000)
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--sp", choices=["last_extremum", "anchor"], default="last_extremum",
                    help="switching-point definition (paper-faithful=last_extremum)")
    ap.add_argument("--reward", choices=["exp_growth", "quad_growth"], default="quad_growth",
                    help="proxy reward shape (quad_growth scales better at large fleets)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--skip-tabular", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--elm-hidden", type=int, default=200)
    ap.add_argument("--elm-lr", type=float, default=1e-2)
    ap.add_argument("--elm-eps0", type=float, default=0.6)
    ap.add_argument("--elm-eps-decay", type=float, default=2.5e-4)
    ap.add_argument("--elm-batch", type=int, default=128)
    ap.add_argument("--elm-replay", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=0,
                    help="ELM training log interval (0 = silent)")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="ELM checkpoint path (npz)")
    ap.add_argument("--checkpoint-every", type=int, default=0)
    args = ap.parse_args()

    spec = FleetSpec(B=tuple(args.B), c=tuple(args.c), d=tuple(args.d))
    sig_kwargs = dict(states=(-4, -1, 1, 5))

    sig_support = MarkovSignal(**sig_kwargs).support
    sxa = sanity_check_action_count(spec, sig_support)
    print(f"Feasible (s,a) count B={args.B}: {sxa}; paper Table-1 |SxA| is the unconstrained box prod(B+1)*prod(c+d+1).")
    print(f"SP mode: {args.sp}; alpha={args.alpha}, beta={args.beta}")

    results = []
    common = dict(sp_mode=args.sp, reward_mode=args.reward, alpha_d=args.alpha, beta=args.beta, seed_run=args.seed)

    print("\n[Naive]")
    results.append(run_agent("Naive", lambda env: NaiveAgent(spec), spec, sig_kwargs, args.T, **common))
    print("  reward=", results[-1]["reward"], " D=", results[-1]["D_per_battery"])

    print("\n[Greedy]")
    results.append(run_agent("Greedy", lambda env: GreedyAgent(env), spec, sig_kwargs, args.T, **common))
    print("  reward=", results[-1]["reward"], " D=", results[-1]["D_per_battery"])

    if not args.skip_tabular:
        print("\n[Tabular Q]")
        results.append(run_agent(
            "TabularQ",
            lambda env: TabularQ(env, alpha=0.1, gamma=0.95, eps0=0.6, eps_decay=1e-4),
            spec, sig_kwargs, args.T, **common,
        ))
        print("  reward=", results[-1]["reward"], " D=", results[-1]["D_per_battery"])

    print("\n[ELM-RL]")
    elm_train_kwargs = {}
    if args.log_every:
        elm_train_kwargs["log_every"] = args.log_every
    if args.checkpoint:
        elm_train_kwargs["checkpoint_path"] = args.checkpoint
        elm_train_kwargs["checkpoint_every"] = args.checkpoint_every or args.T // 10
    results.append(run_agent(
        "ELM-RL",
        lambda env: ELMQAgent(env, hidden=args.elm_hidden, gamma=0.95, alpha=args.elm_lr,
                              eps0=args.elm_eps0, eps_decay=args.elm_eps_decay,
                              batch=args.elm_batch, replay=args.elm_replay, seed=args.seed),
        spec, sig_kwargs, args.T, **common, train_kwargs=elm_train_kwargs,
    ))
    print("  reward=", results[-1]["reward"], " D=", results[-1]["D_per_battery"])

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
