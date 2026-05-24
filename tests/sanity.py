"""Closed-form sanity checks on env, SP tracker, and rainflow degradation.

Run: python3 tests/sanity.py (from battery_gym/)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl_elm"))

import numpy as np

from degradation import cycle_degradation, stress
from env import FleetEnv, FleetSpec, RainflowStack, feasible_actions


def approx(a: float, b: float, rel: float = 1e-3, abs_: float = 1e-9) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


# ---- Stress fn checks --------------------------------------------------------
def test_stress_monotone():
    deltas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    vals = [stress(d) for d in deltas]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), f"stress not monotone: {vals}"
    print(f"  stress(δ) monotone in δ: {[f'{v:.2e}' for v in vals]}")


# ---- Rainflow degradation closed-form ---------------------------------------
def test_constant_trajectory():
    """No cycles, no degradation."""
    D, dods, _ = cycle_degradation([5] * 100, capacity=10)
    assert D == 0.0, f"constant SoC should give D=0, got {D}"
    assert len(dods) == 0
    print("  constant trajectory: D=0 ✓")


def test_single_full_cycle():
    """0 -> B -> 0 -> B should give one full cycle of range B."""
    B = 10
    traj = [0, B, 0, B]
    D, dods, counts = cycle_degradation(traj, B)
    expected = stress(1.0)  # δ = B/B = 1
    # closed cycle [B,0]; residue [0, B] -> open half cycle of range B (count 0.5)
    # so D = 1 * stress(1) + 0.5 * stress(1) = 1.5 * stress(1)
    expected_total = 1.5 * expected
    assert approx(D, expected_total, rel=1e-3), f"expected ~{expected_total:.3e}, got {D:.3e}"
    print(f"  single full cycle: D={D:.3e} matches 1.5*stress(1)={expected_total:.3e} ✓")


def test_N_fixed_depth_cycles():
    """Repeated depth-d oscillation: D = N * stress(d/B) approximately (modulo residue)."""
    B = 4
    d = 2
    N = 50
    # construct traj 0,2,0,2,0,2,...
    traj = [0, d] * N + [0]
    D, dods, _ = cycle_degradation(traj, B)
    expected = (N - 0.5) * stress(d / B)  # roughly N closed cycles
    rel_err = abs(D - expected) / expected
    assert rel_err < 0.1, f"depth-{d} x {N} cycles: expected~{expected:.3e}, got {D:.3e}, rel_err={rel_err:.3f}"
    print(f"  N={N} depth-{d} cycles, B={B}: D={D:.3e} ≈ N*stress(0.5)={N*stress(d/B):.3e} ✓")


# ---- SP tracker state checks ------------------------------------------------
def test_sp_tracker_simple_oscillation():
    """For 0,2,0,2,0,2,0,2 the rainflow stack should remain bounded."""
    rs = RainflowStack(0)
    for b in [2, 0, 2, 0, 2, 0, 2]:
        rs.update(b)
    # Stack after long oscillation should be O(1) — rainflow extracts inner cycles.
    assert len(rs.stack) <= 3, f"stack should remain bounded, got {rs.stack}"
    print(f"  SP tracker stack bounded under oscillation: final={rs.stack} ✓")


def test_sp_tracker_monotone():
    """Strictly increasing trajectory pushes nothing (no reversal)."""
    rs = RainflowStack(0)
    for b in [1, 2, 3, 4, 5]:
        rs.update(b)
    assert rs.stack == [0], f"monotone increase should keep stack=[0], got {rs.stack}"
    assert rs.dir == 1
    print(f"  SP tracker monotone increase: stack={rs.stack} ✓")


def test_sp_tracker_clone_independence():
    rs = RainflowStack(0)
    for b in [1, 2, 1, 0]:
        rs.update(b)
    clone = rs.clone()
    clone.update(1)
    clone.update(2)
    assert rs.last_b == 0 and rs.dir == -1, "clone mutation leaked into original"
    print(f"  SP tracker clone independent of original ✓")


# ---- Env step + feasibility -------------------------------------------------
class _DummySignal:
    def __init__(self, vals):
        self.vals = list(vals)
        self.i = 0
        self.support = np.array([-1, 0, 1])

    def reset(self):
        self.i = 0
        return self.vals[0] if self.vals else 0

    def step(self):
        self.i = min(self.i + 1, len(self.vals) - 1)
        return self.vals[self.i]


def test_env_soc_update():
    spec = FleetSpec(B=(4, 4), c=(2, 2), d=(2, 2))
    env = FleetEnv(spec, _DummySignal([0, 0, 0, 0]), alpha_d=1.0, beta=0.1)
    env.reset()
    assert env.b == (2, 2)  # default b0 is B/2
    (b_new, _), _ = env.step((1, -1))
    assert b_new == (3, 1), f"step (1,-1) from (2,2) should give (3,1), got {b_new}"
    print(f"  env step SoC update correct ✓")


def test_env_feasible_zero_signal():
    """With r=0 and symmetric capacity, all actions summing to 0 should be feasible."""
    spec = FleetSpec(B=(2, 2), c=(2, 2), d=(2, 2))
    feas = feasible_actions(spec, (1, 1), 0)
    expected = {(0, 0), (1, -1), (-1, 1)}
    assert set(feas) == expected, f"expected {expected}, got {set(feas)}"
    print(f"  feasible actions for r=0, b=(1,1): {set(feas)} ✓")


def test_env_feasible_extremes():
    """At full charge, can't accept positive r beyond max discharge."""
    spec = FleetSpec(B=(2, 3), c=(2, 3), d=(2, 3))
    feas = feasible_actions(spec, (2, 3), 5)  # batteries full, request +5
    A_max = min(2, 0) + min(3, 0)  # both at boundary, can only discharge
    A_min = -2 - 3
    r_bar = max(min(5, A_max), A_min)  # = 0
    # actions summing to 0 with a_i in [-min(d,b), 0]: a1 in [-2,0], a2 in [-3,0]
    expected_count = sum(
        1
        for a1 in range(-2, 1)
        for a2 in range(-3, 1)
        if a1 + a2 == 0
    )
    assert len(feas) == expected_count, f"expected {expected_count}, got {len(feas)}"
    print(f"  feasibility at full SoC, r=5: served r̄={r_bar}, |A|={len(feas)} ✓")


# ---- End-to-end: zero-power policy gives D=0 in env -------------------------
def test_zero_policy_zero_degradation():
    """With a signal that always allows zero action, no SoC motion -> D=0."""
    spec = FleetSpec(B=(4, 4), c=(2, 2), d=(2, 2))
    sig = _DummySignal([0] * 100)
    env = FleetEnv(spec, sig, alpha_d=1.0, beta=0.1)
    env.reset()
    for _ in range(99):
        env.step((0, 0))
    from degradation import fleet_degradation
    Ds, _ = fleet_degradation(env.soc_log, list(spec.B))
    assert all(d == 0 for d in Ds), f"zero-action policy should give D=0, got {Ds}"
    assert sum(env.reward_log) == 0, f"zero-action gives no reward, got {sum(env.reward_log)}"
    print(f"  zero-action policy: D={Ds}, total_R={sum(env.reward_log)} ✓")


def main():
    print("Running sanity tests...\n")
    tests = [
        test_stress_monotone,
        test_constant_trajectory,
        test_single_full_cycle,
        test_N_fixed_depth_cycles,
        test_sp_tracker_simple_oscillation,
        test_sp_tracker_monotone,
        test_sp_tracker_clone_independence,
        test_env_soc_update,
        test_env_feasible_zero_signal,
        test_env_feasible_extremes,
        test_zero_policy_zero_degradation,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e!r}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
