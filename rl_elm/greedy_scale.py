"""Discriminator: Greedy vs Naive at B=(2,3), (10,10), (20,20).

If Greedy <= Naive on rainflow D at scale, the proxy reward is misaligned with
the true degradation objective. No learner of the proxy can beat Naive in that
case -- the bug is the reward, not the learner.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

from agents import GreedyAgent, NaiveAgent
from degradation import fleet_degradation
from env import FleetEnv, FleetSpec
from reg_signal import MarkovSignal

REWARD_MODE = "exp_growth"  # set by --reward in main()


def eval_agent(agent_cls_or_factory, spec, T, seed):
    sig = MarkovSignal(seed=seed)
    env = FleetEnv(spec, sig, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode=REWARD_MODE)
    env.reset()
    if agent_cls_or_factory is NaiveAgent:
        agent = NaiveAgent(spec)
    elif agent_cls_or_factory is GreedyAgent:
        agent = GreedyAgent(env)
    else:
        agent = agent_cls_or_factory(env)
    R = 0.0
    for _ in range(T):
        if isinstance(agent, NaiveAgent):
            a = agent.act(env.b, env.r)
        elif isinstance(agent, GreedyAgent):
            agent.env = env
            a = agent.act(env.b, env.r)
        else:
            raise TypeError
        _, r = env.step(a)
        R += r
    Ds, _ = fleet_degradation(env.soc_log, list(spec.B))
    return R, sum(Ds), Ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["exp_growth", "quad_growth"], default="exp_growth")
    ap.add_argument("--T", type=int, default=100_000)
    args = ap.parse_args()
    global REWARD_MODE
    REWARD_MODE = args.reward
    print(f"reward_mode={REWARD_MODE}, T={args.T}\n", flush=True)
    T = args.T
    seeds = [42, 7, 123]
    configs = [
        FleetSpec(B=(2, 3), c=(2, 3), d=(2, 3)),
        FleetSpec(B=(10, 10), c=(5, 5), d=(5, 5)),
        FleetSpec(B=(20, 20), c=(10, 10), d=(10, 10)),
        # Heterogeneous fleets -- Naive's proportional allocation may be sub-optimal here
        FleetSpec(B=(5, 20), c=(2, 10), d=(2, 10)),
        FleetSpec(B=(2, 20), c=(2, 10), d=(2, 10)),
    ]
    for spec in configs:
        n_Ds, g_Ds = [], []
        n_Rs, g_Rs = [], []
        for s in seeds:
            t0 = time.time()
            Rn, Dn, _ = eval_agent(NaiveAgent, spec, T, s)
            t1 = time.time()
            Rg, Dg, _ = eval_agent(GreedyAgent, spec, T, s)
            t2 = time.time()
            n_Ds.append(Dn); g_Ds.append(Dg)
            n_Rs.append(Rn); g_Rs.append(Rg)
            print(f"B={spec.B} seed={s}  Naive D={Dn:.4f} ({t1-t0:.0f}s)  Greedy D={Dg:.4f} ({t2-t1:.0f}s)  diff={(Dn-Dg)/Dn*100:+.1f}%", flush=True)
        n_m, g_m = statistics.mean(n_Ds), statistics.mean(g_Ds)
        print(f"  -> B={spec.B}: Naive mean={n_m:.4f}, Greedy mean={g_m:.4f}, Greedy is {(n_m-g_m)/n_m*100:+.1f}% better\n", flush=True)


if __name__ == "__main__":
    main()
