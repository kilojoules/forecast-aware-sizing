"""Phase 2 v2: PPO training on PriceEnv (gymnasium-wrapped) under forecast noise.

Why PPO instead of behavior cloning:
  BC of QP-ensemble actions from a single-forecast state was shown
  information-limited (PHASE2_BC_NEGATIVE_MEMO.md). PPO samples realized
  trajectories during training, learns the distributional optimum from
  reward signal directly. Test-time policy still takes only single
  forecast as input.

Setup:
  Wrap PriceEnv in a gymnasium-compatible env. At reset(), sample a fresh
  realized trace + a fresh single forecast. Step the env with continuous
  action; reward is the per-step (revenue - alpha*action^2). At episode
  end (T steps), done.

  PPO trains for total_timesteps with continuous action policy + value
  network. Standard MlpPolicy from sb3.

  Evaluate against QP-single, QP-ensemble, LP-with-forecast on held-out
  (realized, forecast) pairs.

Run:
  pixi run python phase2_ppo.py
  # or on GBAR via gbar_submit_phase2_ppo.sh
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Quiet TF / mujoco-py warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MUJOCO_GL", "disabled")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from arbitrage_agents import (lp_linear_actions, qp_ensemble_actions,
                              qp_quadratic_actions, run_actions)
from degradation import cycle_degradation
from env import PriceEnv
from price_signal import make_forecast, synth_diurnal


# -----------------------------------------------------------------------------
# Gymnasium wrapper
# -----------------------------------------------------------------------------
class ArbitrageGymEnv(gym.Env):
    """gymnasium env wrapping PriceEnv with forecast-noise + state featurization.

    Reset -> sample new realized + new single forecast.
    Step  -> apply action, return (obs, reward, terminated, truncated, info).

    Observation: [soc/b_E, normalized_forecast_window(lookahead)] -> dim = 1+lookahead
    Action: scalar in [-b_P, +b_P] (positive=discharge).
    Reward: revenue at step - alpha * a^2  (matches PriceEnv 'quadratic').
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 T: int = 168,
                 b_E: float = 2.0,
                 b_P: float = 2.0,
                 alpha: float = 0.005,
                 noise_std: float = 8.0,
                 lookahead: int = 72,
                 soc0_frac: float = 0.5,
                 base_seed: int | None = None):
        super().__init__()
        self.T = T
        self.b_E = b_E
        self.b_P = b_P
        self.alpha = alpha
        self.noise_std = noise_std
        self.lookahead = lookahead
        self.soc0 = soc0_frac * b_E
        self.action_space = spaces.Box(low=-b_P, high=b_P, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1 + lookahead,), dtype=np.float32)
        self._rng = np.random.default_rng(base_seed)
        self._env: PriceEnv | None = None
        self._forecast: np.ndarray | None = None
        self._fc_mean = 0.0
        self._fc_std = 1.0

    def _sample_episode(self):
        seed_real = int(self._rng.integers(0, 2**31 - 1))
        seed_fc = int(self._rng.integers(0, 2**31 - 1))
        realized = synth_diurnal(self.T, seed=seed_real)
        forecast = make_forecast(realized, self.noise_std, seed=seed_fc)
        return realized, forecast

    def _obs(self) -> np.ndarray:
        t = self._env.t
        end = min(t + self.lookahead, self.T)
        window = self._forecast[t:end]
        if len(window) < self.lookahead:
            pad = np.full(self.lookahead - len(window), self._fc_mean)
            window = np.concatenate([window, pad])
        z_window = (window - self._fc_mean) / self._fc_std
        return np.concatenate([[self._env.soc / self.b_E], z_window]).astype(np.float32)

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        realized, forecast = self._sample_episode()
        self._env = PriceEnv(self.b_E, self.b_P, prices=realized, soc0=self.soc0,
                              alpha=self.alpha, reward_mode="quadratic")
        self._env.reset()
        self._forecast = forecast
        self._fc_mean = float(forecast.mean())
        self._fc_std = max(float(forecast.std()), 1e-6)
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -self.b_P, self.b_P))
        _, r, done, info = self._env.step(a)
        terminated = done
        truncated = False
        return self._obs(), float(r), terminated, truncated, info


# -----------------------------------------------------------------------------
# Eval helpers (use the wrapper env's observation, but evaluate REWARDS using
# realized prices from the wrapped PriceEnv).
# -----------------------------------------------------------------------------
def policy_dispatch_ppo(model, env_kwargs: dict, realized: np.ndarray,
                         forecast: np.ndarray) -> dict:
    """Apply trained PPO policy to a SPECIFIC realized + forecast pair."""
    env = ArbitrageGymEnv(**env_kwargs)
    # Override the random sampling -- inject the trace we want.
    env._env = PriceEnv(env.b_E, env.b_P, prices=realized, soc0=env.soc0,
                         alpha=env.alpha, reward_mode="quadratic")
    env._env.reset()
    env._forecast = forecast
    env._fc_mean = float(forecast.mean())
    env._fc_std = max(float(forecast.std()), 1e-6)
    obs = env._obs()
    while env._env.t < env.T:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            break
    R = float(np.dot(realized, np.asarray(env._env.action_log)))
    return {
        "R": R,
        "soc_log": np.asarray(env._env.soc_log),
        "actions": np.asarray(env._env.action_log),
        "rewards": np.asarray(env._env.reward_log),
    }


def D_safe(soc, B):
    try:
        D, _, _ = cycle_degradation(soc, B)
    except (IndexError, ValueError):
        return 0.0
    return D


# -----------------------------------------------------------------------------
# Train + benchmark
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--T", type=int, default=168)
    parser.add_argument("--lookahead", type=int, default=72)
    parser.add_argument("--noise_train", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--b_E", type=float, default=2.0)
    parser.add_argument("--b_P", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="ppo_policy.zip")
    parser.add_argument("--log_dir", default="ppo_logs")
    parser.add_argument("--n_eval", type=int, default=30)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--load", default=None)
    parser.add_argument("--no_subproc", action="store_true",
                         help="Force DummyVecEnv even when n_envs > 1")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # Lazy imports so the file imports cleanly even if sb3 is missing
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                                  VecMonitor)

    env_kwargs = dict(T=args.T, b_E=args.b_E, b_P=args.b_P, alpha=args.alpha,
                      noise_std=args.noise_train, lookahead=args.lookahead)

    if args.eval_only:
        if args.load is None:
            raise SystemExit("--eval_only requires --load")
        model = PPO.load(args.load)
    else:
        def make_env(rank):
            def _thunk():
                env = ArbitrageGymEnv(**env_kwargs, base_seed=args.seed + rank)
                return env
            return _thunk

        # Smoke-test single env first so we fail fast with a real traceback
        print("[PPO] env smoke test...", flush=True)
        e = ArbitrageGymEnv(**env_kwargs, base_seed=args.seed)
        obs, info = e.reset()
        print(f"  obs shape: {obs.shape}  dtype: {obs.dtype}  action_space: {e.action_space}", flush=True)
        for _ in range(3):
            obs, r, term, trunc, _ = e.step(np.zeros(1, dtype=np.float32))
        print(f"  3 zero-steps OK; reward={r:.3f}", flush=True)
        del e

        # Default to DummyVecEnv (in-process); SubprocVecEnv only if user explicitly
        # asks AND env pickles. Avoids fork+pickle issues.
        if args.n_envs > 1 and not args.no_subproc:
            try:
                venv = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
                print(f"[PPO] using SubprocVecEnv (n_envs={args.n_envs})", flush=True)
            except Exception as ex:
                print(f"[PPO] SubprocVecEnv failed ({ex}); falling back to DummyVecEnv", flush=True)
                venv = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
        else:
            venv = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
            print(f"[PPO] using DummyVecEnv (n_envs={args.n_envs})", flush=True)
        venv = VecMonitor(venv, filename=os.path.join(args.log_dir, "monitor.csv"))

        print(f"[PPO] training: total_timesteps={args.total_timesteps}, T={args.T}", flush=True)
        model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed,
                     n_steps=max(2048 // max(args.n_envs, 1), 64), batch_size=256,
                     learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                     ent_coef=0.0, n_epochs=10,
                     policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                     tensorboard_log=args.log_dir)
        print("[PPO] model constructed; starting learn()", flush=True)
        t0 = time.time()
        model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
        print(f"[PPO] trained in {time.time()-t0:.0f}s", flush=True)
        model.save(args.save)
        venv.close()

    # Evaluation
    print(f"\n[EVAL] {args.n_eval} held-out traces, sweep noise...")
    rng = np.random.default_rng(7)
    held_out_seeds = rng.integers(10_000, 10**8, size=args.n_eval)
    EVAL_NOISES = [3, 5, 8, 12, 18]
    K = 4
    rows = []
    for noise in EVAL_NOISES:
        Rs = {"lp": [], "qps": [], "qpe": [], "rl": []}
        Ds = {"lp": [], "qps": [], "qpe": [], "rl": []}
        for real_seed in held_out_seeds:
            realized = synth_diurnal(args.T, seed=int(real_seed))
            fc_seed = int(rng.integers(10_000, 10**8))
            forecast_single = make_forecast(realized, noise, seed=fc_seed)
            inner = rng.integers(10_000, 10**8, size=K)
            forecasts_K = np.stack([
                make_forecast(realized, noise, seed=int(s)) for s in inner])
            base_env = PriceEnv(args.b_E, args.b_P, prices=realized,
                                 soc0=args.b_E * 0.5, alpha=args.alpha,
                                 reward_mode="quadratic")
            a = lp_linear_actions(forecast_single, args.b_E, args.b_P,
                                   base_env.soc0, mu=5.0)
            rec = run_actions(base_env, a)
            Rs["lp"].append(rec["R"]); Ds["lp"].append(D_safe(rec["soc_log"], args.b_E))
            a = qp_quadratic_actions(forecast_single, args.b_E, args.b_P,
                                      base_env.soc0, alpha=args.alpha)
            rec = run_actions(base_env, a)
            Rs["qps"].append(rec["R"]); Ds["qps"].append(D_safe(rec["soc_log"], args.b_E))
            a = qp_ensemble_actions(forecasts_K, args.b_E, args.b_P,
                                     base_env.soc0, alpha=args.alpha)
            rec = run_actions(base_env, a)
            Rs["qpe"].append(rec["R"]); Ds["qpe"].append(D_safe(rec["soc_log"], args.b_E))
            rec = policy_dispatch_ppo(model, env_kwargs, realized, forecast_single)
            Rs["rl"].append(rec["R"]); Ds["rl"].append(D_safe(rec["soc_log"], args.b_E))

        rows.append((noise,
                     np.mean(Rs["lp"]), np.mean(Rs["qps"]),
                     np.mean(Rs["qpe"]), np.mean(Rs["rl"]),
                     np.mean(Ds["lp"]), np.mean(Ds["qps"]),
                     np.mean(Ds["qpe"]), np.mean(Ds["rl"])))
        print(f"  noise={noise}: R lp={rows[-1][1]:.0f} qps={rows[-1][2]:.0f} "
              f"qpe={rows[-1][3]:.0f} rl={rows[-1][4]:.0f}")

    # Save results JSON for later plotting
    import json
    with open(os.path.join(args.log_dir, "phase2_ppo_results.json"), "w") as f:
        json.dump([{"noise": r[0],
                     "R_lp": r[1], "R_qps": r[2], "R_qpe": r[3], "R_rl": r[4],
                     "D_lp": r[5], "D_qps": r[6], "D_qpe": r[7], "D_rl": r[8]}
                    for r in rows], f, indent=2)
    print(f"\nSaved results to {args.log_dir}/phase2_ppo_results.json")


if __name__ == "__main__":
    main()
