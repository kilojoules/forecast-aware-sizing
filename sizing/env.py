"""Hourly price-driven arbitrage env for the workshop sizing study.

Extracted from the original `env.py` (which also contained the fleet
frequency-regulation env; see `rl_elm/env.py` for that).

State: continuous SoC in [0, b_E].
Action: continuous a in [-b_P, +b_P].  Sign convention: +a = discharge to grid.
Dynamics: SoC[t+1] = SoC[t] - a[t] / eta if a >= 0 else SoC[t] - a[t] * eta
          (clipped to [0, b_E]). Phase-1 default eta=1 keeps LP/QP clean.
Reward modes:
  - 'linear':         r = p[t] * a - mu * |a|
  - 'quadratic':      r = p[t] * a - alpha * a^2
  - 'rainflow_proxy': r = p[t] * a - alpha * (rainflow proxy)
"""
from __future__ import annotations

from typing import Literal

import numpy as np


PriceRewardMode = Literal["linear", "quadratic", "rainflow_proxy"]


class PriceEnv:
    def __init__(
        self,
        b_E: float,
        b_P: float,
        prices: np.ndarray,
        soc0: float | None = None,
        mu: float = 0.0,
        alpha: float = 0.0,
        eta: float = 1.0,
        reward_mode: PriceRewardMode = "linear",
    ):
        self.b_E = float(b_E)
        self.b_P = float(b_P)
        self.prices = np.asarray(prices, dtype=float)
        self.T = len(self.prices)
        self.soc0 = float(b_E) / 2 if soc0 is None else float(soc0)
        if not 0 <= self.soc0 <= self.b_E:
            raise ValueError(f"soc0 {self.soc0} out of [0, {self.b_E}]")
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.eta = float(eta)
        if reward_mode not in ("linear", "quadratic", "rainflow_proxy"):
            raise ValueError(f"unknown reward_mode {reward_mode}")
        self.reward_mode = reward_mode
        self.reset()

    def reset(self) -> tuple[float, int]:
        self.soc = self.soc0
        self.t = 0
        self.tracker = ContinuousRainflowStack(self.soc0)
        self.soc_log: list[float] = [self.soc0]
        self.action_log: list[float] = []
        self.reward_log: list[float] = []
        return self.soc, self.t

    def feasible_range(self) -> tuple[float, float]:
        a_min = max(-self.b_P, -(self.b_E - self.soc) * self.eta)
        a_max = min(self.b_P, self.soc / max(self.eta, 1e-9))
        return a_min, a_max

    def step(self, a: float):
        if self.t >= self.T:
            raise RuntimeError("episode already done")
        a_min, a_max = self.feasible_range()
        a = max(a_min, min(a_max, float(a)))
        if a >= 0:
            soc_new = self.soc - a / max(self.eta, 1e-9)
        else:
            soc_new = self.soc - a * self.eta
        soc_new = max(0.0, min(self.b_E, soc_new))
        self.tracker.update(soc_new)
        revenue = self.prices[self.t] * a
        if self.reward_mode == "linear":
            cost = self.mu * abs(a)
        elif self.reward_mode == "quadratic":
            cost = self.alpha * a * a
        else:
            sp = self.tracker.sp_last_extremum()
            cap = max(self.b_E, 1e-9)
            old_dev = abs(self.soc - sp)
            new_dev = abs(soc_new - sp)
            cost = self.alpha * ((new_dev / cap) ** 2 - (old_dev / cap) ** 2)
        r = revenue - cost
        self.soc = soc_new
        self.t += 1
        self.soc_log.append(self.soc)
        self.action_log.append(a)
        self.reward_log.append(r)
        done = self.t >= self.T
        return self.soc, r, done, {"a_clipped": a}


class ContinuousRainflowStack:
    """Float-valued mirror of the integer RainflowStack used by FleetEnv."""

    def __init__(self, x0: float):
        self.stack: list[float] = [x0]
        self.last: float = x0
        self.dir: int = 0
        self.completed_cycles: list[float] = []

    def update(self, x_new: float) -> float:
        cycle_range = 0.0
        if x_new > self.last:
            new_dir = 1
        elif x_new < self.last:
            new_dir = -1
        else:
            new_dir = self.dir
        if self.dir != 0 and new_dir != 0 and new_dir != self.dir:
            self.stack.append(self.last)
            cycle_range = self._simplify()
        self.last = x_new
        self.dir = new_dir
        return cycle_range

    def _simplify(self) -> float:
        cycle_range = 0.0
        while len(self.stack) >= 4:
            a, b, c, d = self.stack[-4], self.stack[-3], self.stack[-2], self.stack[-1]
            inner = abs(c - b)
            outer_l = abs(b - a)
            outer_r = abs(d - c)
            if inner <= outer_l and inner <= outer_r:
                cycle_range = max(cycle_range, inner)
                self.completed_cycles.append(inner)
                del self.stack[-3:-1]
            else:
                break
        return cycle_range

    def sp_last_extremum(self) -> float:
        return self.stack[-1] if self.stack else self.last

    def sp_anchor(self) -> float:
        return self.stack[0] if self.stack else self.last
