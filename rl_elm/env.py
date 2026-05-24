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

