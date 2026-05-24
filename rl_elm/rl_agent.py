"""Behavior-cloning RL agent for hourly arbitrage under forecast uncertainty.

Approach:
  Train a small MLP policy pi(state) -> action via supervised learning on
  (state, target_action) pairs. The target action is the QP-ensemble (or
  perfect-foresight oracle) action at that step. The policy at test time
  receives only a single noisy forecast and learns to produce robust actions.

Architecture:
  Input  : (1 + lookahead) features:
              soc_t / b_E,
              (forecast[t : t+lookahead] - mean) / std
  Hidden : MLP (default 2 layers x 64 units), ReLU
  Output : tanh, scaled to [-b_P, +b_P]

Training data is generated from the same synth_diurnal price model used in
phase 1; train on N_train traces, evaluate on N_eval held-out traces.

Why behavior cloning (not full RL):
  1. Sample-efficient: needs no env rollouts during training.
  2. Faithful to the Jensen-gap thesis: the policy maps single-forecast
     observations to actions a hindsight ensemble would have chosen.
  3. Cheap: a 2-layer MLP trains in seconds for <30k pairs.

If a learned policy beats QP-single under noise, the rework's RL claim
is supported. If it matches QP-ensemble, the value is RL inference cost
(one forward pass) vs solving K QPs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from arbitrage_agents import (qp_ensemble_actions, qp_quadratic_actions,
                              run_actions)
from env import PriceEnv
from price_signal import make_forecast, synth_diurnal


# -----------------------------------------------------------------------------
# State featurization
# -----------------------------------------------------------------------------
def featurize(soc_t: float, b_E: float, forecast: np.ndarray, t: int,
              lookahead: int = 24,
              forecast_mean: float | None = None,
              forecast_std: float | None = None) -> np.ndarray:
    """State features at step t given a single-forecast view."""
    end = min(t + lookahead, len(forecast))
    window = forecast[t:end]
    if len(window) < lookahead:
        # pad right with mean
        pad = np.full(lookahead - len(window),
                      forecast_mean if forecast_mean is not None else window.mean())
        window = np.concatenate([window, pad])
    if forecast_mean is None:
        forecast_mean = forecast.mean()
    if forecast_std is None:
        forecast_std = max(forecast.std(), 1e-6)
    z_window = (window - forecast_mean) / forecast_std
    return np.concatenate([[soc_t / max(b_E, 1e-9)], z_window])


# -----------------------------------------------------------------------------
# MLP policy
# -----------------------------------------------------------------------------
class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, b_P: float, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.b_P = b_P
        nets = []
        d = in_dim
        for _ in range(layers):
            nets += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        nets += [nn.Linear(d, 1), nn.Tanh()]
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x) * self.b_P


# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------
def build_dataset(N_traces: int,
                  T: int, b_E: float, b_P: float, soc0: float,
                  noise_std: float, K_ensemble: int,
                  alpha: float, lookahead: int = 24,
                  target: str = "ensemble",     # 'ensemble' or 'oracle'
                  base_seed: int = 0,
                  verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) supervised dataset.

    For each of N_traces traces:
      - generate realized prices (independent seed)
      - generate single forecast (target features come from this)
      - generate K ensemble forecasts (used to compute target action vector
        if target='ensemble')
      - target actions: QP-ensemble actions on the K forecasts (or oracle
        on realized).
      - features: at each t in 0..T-1, (soc_t reached by playing target up
        to t, single_forecast windowed at t)
    """
    rng = np.random.default_rng(base_seed)
    Xs, ys = [], []
    for tr in range(N_traces):
        seed_real = int(rng.integers(0, 1_000_000))
        seed_fc = int(rng.integers(0, 1_000_000))
        realized = synth_diurnal(T, seed=seed_real)
        forecast_single = make_forecast(realized, noise_std, seed=seed_fc)
        if target == "ensemble":
            inner = rng.integers(0, 1_000_000, size=K_ensemble)
            forecasts = np.stack([
                make_forecast(realized, noise_std, seed=int(s)) for s in inner])
            target_actions = qp_ensemble_actions(forecasts, b_E, b_P, soc0, alpha)
        elif target == "oracle":
            target_actions = qp_quadratic_actions(realized, b_E, b_P, soc0, alpha)
        else:
            raise ValueError(target)
        # Roll through to get SoCs at each t
        env = PriceEnv(b_E, b_P, prices=realized, soc0=soc0, alpha=alpha,
                       reward_mode="quadratic")
        env.reset()
        # Use target actions to advance state (this is the "expert" trajectory)
        fc_mean, fc_std = forecast_single.mean(), max(forecast_single.std(), 1e-6)
        for t in range(T):
            # Build state from CURRENT soc and single forecast windowed at t
            x = featurize(env.soc, b_E, forecast_single, t, lookahead,
                          forecast_mean=fc_mean, forecast_std=fc_std)
            Xs.append(x)
            ys.append(target_actions[t])
            env.step(target_actions[t])
        if verbose and (tr + 1) % 50 == 0:
            print(f"  built {tr+1}/{N_traces} traces")
    X = np.array(Xs, dtype=np.float32)
    y = np.array(ys, dtype=np.float32).reshape(-1, 1)
    return X, y


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_policy(X: np.ndarray, y: np.ndarray, b_P: float,
                 epochs: int = 50, batch: int = 256,
                 lr: float = 1e-3, hidden: int = 64, layers: int = 2,
                 device: str = "cpu", seed: int = 0,
                 val_frac: float = 0.1,
                 verbose: bool = True) -> tuple[MLPPolicy, dict]:
    torch.manual_seed(seed)
    in_dim = X.shape[1]
    n = len(X)
    n_val = int(n * val_frac)
    n_train = n - n_val
    perm = np.random.default_rng(seed).permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    X_tr, y_tr = torch.tensor(X[train_idx], device=device), torch.tensor(y[train_idx], device=device)
    X_va, y_va = torch.tensor(X[val_idx], device=device), torch.tensor(y[val_idx], device=device)

    pi = MLPPolicy(in_dim, b_P, hidden=hidden, layers=layers).to(device)
    opt = optim.Adam(pi.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        pi.train()
        order = torch.randperm(n_train)
        ep_loss = 0.0
        for i in range(0, n_train, batch):
            idx = order[i:i+batch]
            xb, yb = X_tr[idx], y_tr[idx]
            pred = pi(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n_train
        pi.eval()
        with torch.no_grad():
            v_loss = loss_fn(pi(X_va), y_va).item()
        history["train_loss"].append(ep_loss)
        history["val_loss"].append(v_loss)
        if verbose and (ep + 1) % max(1, epochs // 10) == 0:
            print(f"  ep {ep+1}/{epochs}  train={ep_loss:.5f}  val={v_loss:.5f}")
    return pi, history


# -----------------------------------------------------------------------------
# Inference / evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def policy_dispatch(pi: MLPPolicy, env: PriceEnv, forecast: np.ndarray,
                    lookahead: int = 24) -> dict:
    """Roll policy on env using a single forecast view. Returns same dict as run_actions."""
    pi.eval()
    env.reset()
    fc_mean, fc_std = forecast.mean(), max(forecast.std(), 1e-6)
    for t in range(env.T):
        x = featurize(env.soc, env.b_E, forecast, t, lookahead,
                      forecast_mean=fc_mean, forecast_std=fc_std)
        a = float(pi(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item())
        env.step(a)
    R = float(np.dot(env.prices[:env.T], np.asarray(env.action_log)))
    return {
        "R": R,
        "soc_log": np.asarray(env.soc_log),
        "actions": np.asarray(env.action_log),
        "rewards": np.asarray(env.reward_log),
    }


# -----------------------------------------------------------------------------
# Save / load
# -----------------------------------------------------------------------------
def save_policy(pi: MLPPolicy, path: str | Path, meta: dict | None = None):
    payload = {"state_dict": pi.state_dict(),
               "in_dim": pi.net[0].in_features,
               "b_P": pi.b_P,
               "meta": meta or {}}
    torch.save(payload, str(path))


def load_policy(path: str | Path, hidden: int = 64, layers: int = 2) -> MLPPolicy:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    pi = MLPPolicy(payload["in_dim"], payload["b_P"], hidden=hidden, layers=layers)
    pi.load_state_dict(payload["state_dict"])
    pi.eval()
    return pi
