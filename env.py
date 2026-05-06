"""Battery fleet environment for degradation-aware frequency regulation.

Reproduces Section 2 of Srinivasa et al. 2026 (arXiv:2601.22865v2).

State: s = (b_1, ..., b_N, r), b_i in {0,...,B_i}, r in S_r.
Action: a = (a_1, ..., a_N), integer charge/discharge per battery.
Constraints:
  - per-battery: -min(d_i,b_i) <= a_i <= min(c_i, B_i - b_i)
  - collective: sum a_i = clip(r, A_min, A_max)

Switching-point tracking (Section 3.5):
  - 'last_extremum' (paper-faithful): SP_i(t) = top of online rainflow stack.
    The reward in eq (9) using this SP is structurally gameable by full-amplitude
    oscillation -- after a reversal the SP slides to the new extremum, so the
    next-step deviation collapses to 0 and reversing always rewards.
  - 'anchor' (alternative): SP_i(t) = bottom of online rainflow stack. Stable
    reference point that does not move with reversal. Removes gaming.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


SPMode = Literal["last_extremum", "anchor"]
RewardMode = Literal["exp_growth", "quad_growth"]


@dataclass(frozen=True)
class FleetSpec:
    B: tuple[int, ...]
    c: tuple[int, ...]
    d: tuple[int, ...]

    @property
    def N(self) -> int:
        return len(self.B)


def feasible_actions(spec: FleetSpec, b: tuple[int, ...], r: int) -> list[tuple[int, ...]]:
    """Enumerate integer action vectors satisfying all constraints in eqs (1),(3),(6)."""
    N = spec.N
    a_min_i = [-min(spec.d[i], b[i]) for i in range(N)]
    a_max_i = [min(spec.c[i], spec.B[i] - b[i]) for i in range(N)]
    A_max = sum(a_max_i)
    A_min = sum(a_min_i)
    r_bar = max(min(r, A_max), A_min)

    # enumerate via cartesian product across per-battery ranges
    ranges = [range(a_min_i[i], a_max_i[i] + 1) for i in range(N)]
    return [tuple(a) for a in itertools.product(*ranges) if sum(a) == r_bar]


def step_soc(b: tuple[int, ...], a: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(b[i] + a[i] for i in range(len(b)))


class RainflowStack:
    """Online rainflow simplification per battery.

    Maintains a stack of confirmed extrema of the SoC trajectory so far. After
    each direction reversal we push the just-passed value as a new extremum,
    then apply the 4-point rainflow rule: if the inner two points form a cycle
    smaller than both outer ranges, extract them. Cycles are also collected for
    optional online-degradation reward use.
    """

    def __init__(self, b0: int):
        self.stack: list[int] = [b0]
        self.last_b: int = b0
        self.dir: int = 0
        self.completed_cycles: list[int] = []  # ranges (peak-trough magnitudes)

    def update(self, b_new: int) -> int:
        """Return the range of any completed inner cycle (0 if none)."""
        cycle_range = 0
        if b_new > self.last_b:
            new_dir = 1
        elif b_new < self.last_b:
            new_dir = -1
        else:
            new_dir = self.dir
        if new_dir != self.dir and self.dir != 0:
            self.stack.append(self.last_b)
            cycle_range = self._simplify()
        self.dir = new_dir
        self.last_b = b_new
        return cycle_range

    def _simplify(self) -> int:
        # Apply 4-point rainflow rule until no more inner cycles can be extracted.
        total_extracted = 0
        while len(self.stack) >= 4:
            X1, X2, X3, X4 = self.stack[-4:]
            inner = abs(X2 - X3)
            left = abs(X1 - X2)
            right = abs(X3 - X4)
            if inner <= left and inner <= right:
                self.completed_cycles.append(inner)
                total_extracted = max(total_extracted, inner)
                # remove X2 and X3
                del self.stack[-3:-1]
            else:
                break
        return total_extracted

    def sp_last_extremum(self) -> int:
        return self.stack[-1]

    def sp_anchor(self) -> int:
        return self.stack[0]

    def clone(self) -> "RainflowStack":
        c = RainflowStack(self.last_b)
        c.stack = list(self.stack)
        c.last_b = self.last_b
        c.dir = self.dir
        c.completed_cycles = list(self.completed_cycles)
        return c


class FleetEnv:
    """Episode runner with switching-point tracking for the proxy reward.

    Reward (eq 9-10):
        r_i(t) = -(alpha_d * exp(beta * |b_i(t) + a_i(t) - b_i^SP(t)|)
                  - alpha_d * exp(beta * |b_i(t) - b_i^SP(t)|))
        R = sum_i r_i

    The switching point b_i^SP(t) is the most recent local extremum of the SoC
    trajectory of battery i, updated incrementally per Section 3.5.
    """

    def __init__(
        self,
        spec: FleetSpec,
        signal,
        alpha_d: float = 1.0,
        beta: float = 0.1,
        b0: tuple[int, ...] | None = None,
        sp_mode: SPMode = "last_extremum",
        reward_mode: RewardMode = "exp_growth",
    ):
        self.spec = spec
        self.signal = signal
        self.alpha_d = alpha_d
        self.beta = beta
        self.b0 = b0 if b0 is not None else tuple(B // 2 for B in spec.B)
        self.sp_mode = sp_mode
        self.reward_mode = reward_mode
        self.reset()

    def reset(self) -> tuple[tuple[int, ...], int]:
        self.b = self.b0
        self.r = self.signal.reset()
        self.trackers = [RainflowStack(b) for b in self.b0]
        self.soc_log: list[list[int]] = [[b] for b in self.b0]
        self.signal_log: list[int] = [self.r]
        self.action_log: list[tuple[int, ...]] = []
        self.reward_log: list[float] = []
        return self.b, self.r

    def sp(self, i: int) -> int:
        if self.sp_mode == "anchor":
            return self.trackers[i].sp_anchor()
        return self.trackers[i].sp_last_extremum()

    @property
    def b_sp(self) -> list[int]:
        return [self.sp(i) for i in range(self.spec.N)]

    def step(self, a: tuple[int, ...]):
        spec = self.spec
        N = spec.N
        b_new = step_soc(self.b, a)
        # Update SP tracker FIRST so the reward uses the just-confirmed extremum:
        # at a reversal step the freshly-pushed extremum becomes the SP, and the
        # reward at that step measures cycle-depth growth from that new SP.
        for i in range(N):
            self.trackers[i].update(b_new[i])
            self.soc_log[i].append(b_new[i])
        r_t = 0.0
        for i in range(N):
            sp_i = self.sp(i)
            cap = max(spec.B[i], 1)
            old_dev = abs(self.b[i] - sp_i)
            new_dev = abs(b_new[i] - sp_i)
            if self.reward_mode == "quad_growth":
                # Squared normalized deviation; penalizes deep excursions super-linearly
                # (aligned with rainflow stress f(δ) which is super-linear in cycle depth).
                r_t -= self.alpha_d * ((new_dev / cap) ** 2 - (old_dev / cap) ** 2)
            else:
                r_t -= self.alpha_d * (np.exp(self.beta * new_dev) - np.exp(self.beta * old_dev))
        self.b = b_new
        self.r = self.signal.step()
        self.signal_log.append(self.r)
        self.action_log.append(a)
        self.reward_log.append(r_t)
        return (self.b, self.r), r_t

    def feasible(self) -> list[tuple[int, ...]]:
        return feasible_actions(self.spec, self.b, self.r)


# =============================================================================
# Hourly price-driven arbitrage env (phase 1 of the rework, see HYDESIGN_PILOTS)
# =============================================================================
PriceRewardMode = Literal["linear", "quadratic", "rainflow_proxy"]


class PriceEnv:
    """Single-battery hourly-arbitrage MDP.

    State: continuous SoC in [0, b_E].
    Action: continuous a in [-b_P, +b_P].  Sign convention: +a = discharge to grid.
    Dynamics: SoC[t+1] = SoC[t] - a[t] / eta_charge if a<0 else SoC[t] - a[t] * eta_discharge
              (clipped to [0, b_E]).
              For phase 1 default eta=1 keeps the LP/QP comparison clean.
    Reward modes:
      - 'linear':         r = p[t] * a - mu * |a|
      - 'quadratic':      r = p[t] * a - alpha * a^2
      - 'rainflow_proxy': r = p[t] * a - alpha * (rainflow proxy, see below)

    Rainflow proxy reward (in-loop incremental approximation):
      Maintain a continuous rainflow stack on SoC. At each step, the proxy is
      alpha * ((|SoC[t+1] - sp|/b_E)^2 - (|SoC[t] - sp|/b_E)^2),
      with sp = top of stack (most recent confirmed extremum, updated FIRST).
      This mirrors FleetEnvs quad_growth mode at fleet size 1.

    Inputs:
      prices  : np.ndarray length T, exogenous price trace per hour.
      b_E     : energy capacity (MWh), scalar float.
      b_P     : power limit (MW), scalar float.
      soc0    : initial SoC in [0, b_E] (default b_E/2).
      mu, alpha: penalty coefficients for the chosen reward mode.
      eta     : single round-trip efficiency (default 1.0 = ideal).
    """

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
        """Action bounds at current state. a >= -b_P (max charge),
        a <= +b_P (max discharge), 0 <= soc - a/eta <= b_E (SoC bounds)."""
        a_min = max(-self.b_P, -(self.b_E - self.soc) * self.eta)
        a_max = min(self.b_P, self.soc / max(self.eta, 1e-9))
        return a_min, a_max

    def step(self, a: float):
        if self.t >= self.T:
            raise RuntimeError("episode already done")
        a_min, a_max = self.feasible_range()
        a = max(a_min, min(a_max, float(a)))
        # Charge if a<0 (battery gains energy = -a*eta), discharge if a>0 (battery loses a/eta).
        if a >= 0:
            soc_new = self.soc - a / max(self.eta, 1e-9)
        else:
            soc_new = self.soc - a * self.eta   # -a > 0, soc grows
        soc_new = max(0.0, min(self.b_E, soc_new))
        # Rainflow tracker updated BEFORE reward (matches FleetEnv convention)
        self.tracker.update(soc_new)
        # Reward
        revenue = self.prices[self.t] * a
        if self.reward_mode == "linear":
            cost = self.mu * abs(a)
        elif self.reward_mode == "quadratic":
            cost = self.alpha * a * a
        else:  # rainflow_proxy
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
    """Float-valued mirror of RainflowStack used by PriceEnv.

    Same algorithm as the integer stack: tracks confirmed extrema, applies
    the 4-point rule on direction reversals.
    """

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
                # remove the inner pair
                del self.stack[-3:-1]
            else:
                break
        return cycle_range

    def sp_last_extremum(self) -> float:
        return self.stack[-1] if self.stack else self.last

    def sp_anchor(self) -> float:
        return self.stack[0] if self.stack else self.last
