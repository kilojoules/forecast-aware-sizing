"""Action-match analysis: does ELM-RL imitate Greedy or beat it?

Train ELM at B=(2,50) and B=(10,100), then on a held-out signal trajectory
log step-by-step (state, ELM action, Greedy action, match?). If match-rate
>= 80%, ELM is "smooth Greedy" -- it has reproduced the proxy-greedy policy
via function approximation rather than learning multi-step credit assignment.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import ELMQAgent, GreedyAgent
from degradation import fleet_degradation
from env import FleetEnv, FleetSpec
from reg_signal import MarkovSignal


def run(spec: FleetSpec, T_train: int = 300_000, T_eval: int = 30_000, seed: int = 42):
    print(f"\n=== B={spec.B}  T_train={T_train}  T_eval={T_eval}  seed={seed} ===", flush=True)

    # Train ELM
    sig = MarkovSignal(seed=seed)
    env = FleetEnv(spec, sig, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode="quad_growth")
    elm = ELMQAgent(env, hidden=200, gamma=0.95, alpha=0.01, eps0=0.6,
                    eps_decay=1e-5, batch=128, replay=5000, seed=seed)
    t0 = time.time()
    elm.train(T_train)
    print(f"  trained ELM in {time.time()-t0:.0f}s", flush=True)

    # Held-out eval: identical trajectory for ELM and Greedy.
    # Both agents see the SAME state at each step; the env only advances when
    # we step it. We use two parallel envs that consume the same Markov signal.
    sig1 = MarkovSignal(seed=seed + 100)
    sig2 = MarkovSignal(seed=seed + 100)  # same seed -> same realization
    env_elm = FleetEnv(spec, sig1, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode="quad_growth")
    env_grd = FleetEnv(spec, sig2, alpha_d=1.0, beta=0.1, sp_mode="last_extremum", reward_mode="quad_growth")
    env_elm.reset(); env_grd.reset()
    greedy = GreedyAgent(env_grd)

    matches = 0
    cosine_signs = 0  # rough "direction agreement"
    R_elm = 0.0
    R_grd = 0.0
    for _ in range(T_eval):
        # Force same state by overwriting greedy's state to ELM's (deterministic env).
        env_grd.b = env_elm.b
        env_grd.r = env_elm.r
        # Sync the SP trackers too so Greedy uses the same SP values.
        for i in range(spec.N):
            env_grd.trackers[i] = env_elm.trackers[i].clone()

        elm.env = env_elm
        actions_elm = env_elm.feasible()
        actions_grd = env_grd.feasible()
        a_elm = elm.act(actions_elm, greedy=True)
        greedy.env = env_grd
        a_grd = greedy.act(env_grd.b, env_grd.r)

        if a_elm == a_grd:
            matches += 1
        if all(((a_elm[i] > 0) == (a_grd[i] > 0)) or (a_elm[i] == 0 == a_grd[i]) for i in range(spec.N)):
            cosine_signs += 1

        # advance ELM env (Greedy env is overwritten next step from ELM)
        _, r1 = env_elm.step(a_elm)
        _, r2 = env_grd.step(a_grd)
        R_elm += r1
        R_grd += r2

    rate = matches / T_eval * 100
    sign_rate = cosine_signs / T_eval * 100
    Ds_elm, _ = fleet_degradation(env_elm.soc_log, list(spec.B))
    Ds_grd, _ = fleet_degradation(env_grd.soc_log, list(spec.B))
    print(
        f"  match_rate={rate:.1f}%  sign_match={sign_rate:.1f}%  "
        f"ELM D={sum(Ds_elm):.4f}  Greedy D={sum(Ds_grd):.4f}",
        flush=True,
    )
    return rate, sign_rate, sum(Ds_elm), sum(Ds_grd)


def main():
    configs = [
        FleetSpec(B=(2, 50), c=(2, 25), d=(2, 25)),
        FleetSpec(B=(10, 100), c=(5, 50), d=(5, 50)),
    ]
    summary = []
    for spec in configs:
        for seed in [42, 7]:
            rate, signs, D_e, D_g = run(spec, T_train=300_000, T_eval=20_000, seed=seed)
            summary.append((spec.B, seed, rate, signs, D_e, D_g))
    print("\n--- Summary ---")
    print(f"{'B':<14} {'seed':<6} {'match%':>8} {'sign%':>8} {'D_ELM':>8} {'D_Greedy':>10}")
    for B, s, r, sg, de, dg in summary:
        print(f"{str(B):<14} {s:<6} {r:>8.1f} {sg:>8.1f} {de:>8.4f} {dg:>10.4f}")


if __name__ == "__main__":
    main()
