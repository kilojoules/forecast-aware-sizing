"""Naive, Greedy, Tabular Q-learning, ELM-RL agents."""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Callable

import numpy as np

from env import FleetEnv, FleetSpec, feasible_actions


# ---- Naive (eq 19) ----------------------------------------------------------
class NaiveAgent:
    def __init__(self, spec: FleetSpec, rng: np.random.Generator | None = None):
        self.spec = spec
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def act(self, b: tuple[int, ...], r: int) -> tuple[int, ...]:
        spec = self.spec
        N = spec.N
        # served regulation
        a_min_i = [-min(spec.d[i], b[i]) for i in range(N)]
        a_max_i = [min(spec.c[i], spec.B[i] - b[i]) for i in range(N)]
        A_max, A_min = sum(a_max_i), sum(a_min_i)
        r_bar = max(min(r, A_max), A_min)

        total_B = sum(spec.B)
        # proportional allocation, then floor + integer remainder distribution
        raw = [spec.B[i] / total_B * r_bar for i in range(N)]
        a = [int(math.floor(x)) if r_bar >= 0 else int(math.ceil(x)) for x in raw]
        # clip to per-battery bounds
        a = [max(min(a[i], a_max_i[i]), a_min_i[i]) for i in range(N)]
        # fix mismatch one unit at a time among feasible batteries
        sign = 1 if (r_bar - sum(a)) > 0 else -1
        while sum(a) != r_bar:
            feasible = [
                i for i in range(N)
                if (sign > 0 and a[i] < a_max_i[i]) or (sign < 0 and a[i] > a_min_i[i])
            ]
            if not feasible:
                break
            i = int(self.rng.choice(feasible))
            a[i] += sign
        return tuple(a)


# ---- Greedy (eq 20: argmax one-step proxy reward) ---------------------------
def proxy_reward_for_action(env: FleetEnv, a: tuple[int, ...]) -> float:
    """One-step proxy reward using the post-update SP, mirroring env.step()."""
    spec = env.spec
    rew = 0.0
    for i in range(spec.N):
        new_b = env.b[i] + a[i]
        clone = env.trackers[i].clone()
        clone.update(new_b)
        if env.sp_mode == "anchor":
            sp_i = clone.sp_anchor()
        else:
            sp_i = clone.sp_last_extremum()
        cap = max(spec.B[i], 1)
        old_dev = abs(env.b[i] - sp_i)
        new_dev = abs(new_b - sp_i)
        if env.reward_mode == "quad_growth":
            rew -= env.alpha_d * ((new_dev / cap) ** 2 - (old_dev / cap) ** 2)
        else:
            rew -= env.alpha_d * (np.exp(env.beta * new_dev) - np.exp(env.beta * old_dev))
    return rew


class GreedyAgent:
    def __init__(self, env: FleetEnv, rng: np.random.Generator | None = None):
        self.env = env
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def act(self, b: tuple[int, ...], r: int) -> tuple[int, ...]:
        actions = feasible_actions(self.env.spec, b, r)
        rewards = [proxy_reward_for_action(self.env, a) for a in actions]
        best = max(rewards)
        ties = [a for a, rr in zip(actions, rewards) if abs(rr - best) < 1e-9]
        return ties[int(self.rng.integers(0, len(ties)))]


# ---- Tabular Q-learning -----------------------------------------------------
class TabularQ:
    def __init__(self, env: FleetEnv, alpha: float = 0.1, gamma: float = 0.95,
                 eps0: float = 0.6, eps_decay: float = 1e-5, seed: int = 0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps0 = eps0
        self.eps_decay = eps_decay
        self.Q: dict[tuple, dict[tuple[int, ...], float]] = defaultdict(dict)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.visit: dict[tuple, int] = defaultdict(int)

    def _state_key(self, b: tuple[int, ...], r: int, b_sp: tuple[int, ...], dirs: tuple[int, ...]) -> tuple:
        # Paper §3.1 defines state s = (b_1,...,b_N, r). b_sp/dir are bookkeeping for reward,
        # not part of MDP state. Tabular Q learns the marginal over b_sp, which is the
        # paper's setup; the resulting MDP is partially observable but tractable.
        return (b, r)

    def _q(self, key, a):
        return self.Q[key].get(a, 0.0)

    def act(self, key, actions: list[tuple[int, ...]], greedy: bool = False) -> tuple[int, ...]:
        eps = 0.0 if greedy else self.eps0 / (1 + self.eps_decay * self.t)
        if self.rng.random() < eps:
            return actions[int(self.rng.integers(0, len(actions)))]
        qs = [self._q(key, a) for a in actions]
        best = max(qs)
        ties = [a for a, q in zip(actions, qs) if abs(q - best) < 1e-9]
        return ties[int(self.rng.integers(0, len(ties)))]

    def train(self, T: int):
        env = self.env
        env.reset()
        for _ in range(T):
            b, r = env.b, env.r
            b_sp = tuple(env.b_sp)
            dirs = tuple(t.dir for t in env.trackers)
            key = self._state_key(b, r, b_sp, dirs)
            actions = env.feasible()
            a = self.act(key, actions)
            (b2, r2), reward = env.step(a)
            b_sp2 = tuple(env.b_sp)
            dirs2 = tuple(t.dir for t in env.trackers)
            key2 = self._state_key(b2, r2, b_sp2, dirs2)
            actions2 = env.feasible()
            q_next = max((self._q(key2, ap) for ap in actions2), default=0.0)
            self.visit[(key, a)] += 1
            lr = self.alpha / (1 + 0.001 * self.visit[(key, a)])
            self.Q[key][a] = self._q(key, a) + lr * (reward + self.gamma * q_next - self._q(key, a))
            self.t += 1


# ---- ELM-RL: random feature map + linear semi-gradient TD -------------------
def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class ELMFeatureMap:
    def __init__(self, in_dim: int, hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-1.0, 1.0, size=(hidden, in_dim))
        self.b = rng.uniform(-1.0, 1.0, size=(hidden,))
        self.hidden = hidden

    def phi(self, x: np.ndarray) -> np.ndarray:
        z = self.W @ x + self.b
        return silu(z)


class ELMQAgent:
    """Linear Q over ELM features.

    Input vector x = [s, a, r_proxy, b_sp]  (dim 3N + 2 per the paper, eq 15).
    Per the paper, x includes the most recent reward `r` (not the regulation
    request r); we follow that convention.
    """

    def __init__(self, env: FleetEnv, hidden: int = 50, gamma: float = 0.95,
                 alpha: float = 1e-3, eps0: float = 0.6, eps_decay: float = 5.5e-4,
                 batch: int = 128, replay: int = 2000, seed: int = 0,
                 rich_features: bool = True):
        self.env = env
        spec = env.spec
        N = spec.N
        self.rich_features = rich_features
        # Base 3N+2: b_i, a_i, b_sp_i, r_signal, r_proxy.
        # Rich: + (b_i - sp_i), |b_i - sp_i|, sign(b_i - sp_i), (b_i+a_i - sp_i),
        # |b_i+a_i - sp_i| per battery; + |S_r| one-hot bits for r_signal.
        sig_card = len(env.signal.support)
        if rich_features:
            in_dim = 3 * N + 2 + 5 * N + sig_card
        else:
            in_dim = 3 * N + 2
        self.fmap = ELMFeatureMap(in_dim, hidden, seed=seed)
        self.w = np.zeros(hidden)
        self.gamma = gamma
        self.alpha = alpha
        self.eps0 = eps0
        self.eps_decay = eps_decay
        self.batch = batch
        self.replay: deque = deque(maxlen=replay)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        # Per-battery normalization: each SoC, action, and SP is divided by that
        # battery's capacity, so all inputs are in [-1, 1] regardless of fleet
        # heterogeneity. Earlier impl used a global max(B) which compressed the
        # smaller battery's signal to a fraction of the input range.
        self._caps = np.array([max(spec.B[i], 1) for i in range(N)], dtype=float)
        self._cs = np.array([max(spec.c[i], 1) for i in range(N)], dtype=float)
        self._sig_norm = max(abs(s) for s in env.signal.support)
        self._sig_support = list(env.signal.support)
        self._N = N

    def _x(self, b, r_signal, a, b_sp, r_proxy=0.0) -> np.ndarray:
        N = self._N
        sig = self._sig_norm
        sup = self._sig_support
        sig_card = len(sup)
        if self.rich_features:
            out = np.zeros(3 * N + 2 + 5 * N + sig_card)
        else:
            out = np.empty(3 * N + 2)
        for i in range(N):
            B = self._caps[i]
            c = self._cs[i]
            out[i] = b[i] / B
            out[N + i] = a[i] / c
            out[2 * N + 1 + i] = b_sp[i] / B
        out[2 * N] = r_signal / sig
        out[3 * N + 1] = r_proxy
        if self.rich_features:
            base = 3 * N + 2
            for i in range(N):
                B = self._caps[i]
                dev_now = (b[i] - b_sp[i]) / B
                dev_next = (b[i] + a[i] - b_sp[i]) / B
                out[base + 5 * i + 0] = dev_now
                out[base + 5 * i + 1] = abs(dev_now)
                out[base + 5 * i + 2] = np.sign(dev_now)
                out[base + 5 * i + 3] = dev_next
                out[base + 5 * i + 4] = abs(dev_next)
            r_idx = sup.index(r_signal) if r_signal in sup else 0
            out[base + 5 * N + r_idx] = 1.0
        return out

    def q(self, b, r_signal, a, b_sp, r_proxy=0.0) -> float:
        x = self._x(b, r_signal, a, b_sp, r_proxy)
        return float(self.w @ self.fmap.phi(x))

    def act(self, actions, greedy: bool = False) -> tuple[int, ...]:
        eps = 0.0 if greedy else self.eps0 / (1.0 + self.eps_decay * self.t)
        if self.rng.random() < eps:
            return actions[int(self.rng.integers(0, len(actions)))]
        env = self.env
        b_sp = tuple(env.b_sp)
        qs = [self.q(env.b, env.r, a, b_sp) for a in actions]
        best = max(qs)
        ties = [a for a, q in zip(actions, qs) if abs(q - best) < 1e-9]
        return ties[int(self.rng.integers(0, len(ties)))]

    def _td_update(self):
        if len(self.replay) < self.batch:
            return
        idxs = self.rng.choice(len(self.replay), size=self.batch, replace=False)
        grad = np.zeros_like(self.w)
        for k in idxs:
            b, r_sig, a, r_rew, b2, r_sig2, b_sp, b_sp2, acts2 = self.replay[k]
            x = self._x(b, r_sig, a, b_sp)
            phi = self.fmap.phi(x)
            q_sa = self.w @ phi
            q_next = max(
                (self.w @ self.fmap.phi(self._x(b2, r_sig2, ap, b_sp2)) for ap in acts2),
                default=0.0,
            )
            td = r_rew + self.gamma * q_next - q_sa
            grad += td * phi
        self.w += (self.alpha / self.batch) * grad

    def train(self, T: int, update_every: int = 8, log_every: int = 0,
              checkpoint_path: str | None = None, checkpoint_every: int = 0):
        """Train ELM-RL via semi-gradient TD with replay.

        Optional logging:
          log_every: print rolling reward / |w| every N steps (0 = silent)
          checkpoint_path: save np.savez(self.w, fmap.W, fmap.b) every checkpoint_every
        """
        import time
        env = self.env
        env.reset()
        rolling_R = 0.0
        t0 = time.time()
        for step in range(T):
            b = env.b
            r_sig = env.r
            b_sp = tuple(env.b_sp)
            actions = env.feasible()
            a = self.act(actions)
            (b2, r_sig2), reward = env.step(a)
            b_sp2 = tuple(env.b_sp)
            actions2 = env.feasible()
            self.replay.append((b, r_sig, a, reward, b2, r_sig2, b_sp, b_sp2, tuple(actions2)))
            rolling_R += reward
            self.t += 1
            if step % update_every == 0:
                self._td_update()
            if log_every and step and step % log_every == 0:
                eps = self.eps0 / (1.0 + self.eps_decay * self.t)
                rate = step / max(time.time() - t0, 1e-6)
                print(f"  [ELM] step={step}  rolling_R={rolling_R:.1f}  |w|={np.linalg.norm(self.w):.3f}  eps={eps:.3f}  rate={rate:.0f}/s",
                      flush=True)
                rolling_R = 0.0
            if checkpoint_path and checkpoint_every and step and step % checkpoint_every == 0:
                np.savez(checkpoint_path, w=self.w, W=self.fmap.W, b=self.fmap.b, t=self.t)

    def save(self, path: str):
        np.savez(path, w=self.w, W=self.fmap.W, b=self.fmap.b, t=self.t)

    def load(self, path: str):
        data = np.load(path)
        self.w = data["w"]
        self.fmap.W = data["W"]
        self.fmap.b = data["b"]
        self.t = int(data["t"])
