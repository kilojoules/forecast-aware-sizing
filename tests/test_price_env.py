"""Tests for PriceEnv + arbitrage_agents.

Run: python tests/test_price_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from arbitrage_agents import (lp_linear_actions, qp_ensemble_actions,
                              qp_quadratic_actions, run_actions)
from env import ContinuousRainflowStack, PriceEnv
from price_signal import make_forecast, synth_diurnal


def approx(a: float, b: float, rel: float = 1e-3, abs_: float = 1e-9) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


# ---- PriceEnv basics ---------------------------------------------------------
def test_price_env_charge_discharge():
    """Action sign convention: +a discharges, -a charges. Round-trip preserves SoC at eta=1."""
    prices = np.array([10.0, 10.0])
    env = PriceEnv(b_E=2.0, b_P=1.0, prices=prices, soc0=0.5, mu=0.0, eta=1.0,
                   reward_mode="linear")
    env.step(-0.5)   # charge: SoC 0.5 -> 1.0
    env.step(+0.5)   # discharge: SoC 1.0 -> 0.5
    assert approx(env.soc, 0.5), f"round-trip SoC failed: {env.soc}"
    print(f"  PriceEnv round-trip preserves SoC: {env.soc}")


def test_price_env_revenue_linear():
    """Revenue = price * dispatch. Cost zero at mu=0."""
    prices = np.array([10.0, 100.0])
    env = PriceEnv(b_E=2.0, b_P=1.0, prices=prices, soc0=0.5, mu=0.0,
                   reward_mode="linear")
    env.step(-1.0)   # buy 1 MWh at 10 -> revenue -10
    env.step(+1.0)   # sell 1 MWh at 100 -> revenue +100
    total = sum(env.reward_log)
    assert approx(total, 90.0), f"expected 90.0, got {total}"
    print(f"  Linear revenue correct: total={total}")


def test_price_env_quadratic_cost():
    """Quadratic cost = alpha * a^2."""
    prices = np.array([0.0])
    # Big battery so a=1.5 is feasible (soc=2.0 supports 1.5 discharge)
    env = PriceEnv(b_E=4.0, b_P=2.0, prices=prices, soc0=2.0, alpha=2.0,
                   reward_mode="quadratic")
    env.step(1.5)
    expected_cost = 2.0 * 1.5 ** 2
    expected_reward = 0.0 - expected_cost
    assert approx(env.reward_log[0], expected_reward), \
        f"expected {expected_reward}, got {env.reward_log[0]} (action={env.action_log[0]})"
    print(f"  Quadratic cost correct: {env.reward_log[0]} = -{expected_cost}")


def test_price_env_clipping():
    """Action clipped to feasible range based on SoC and b_P."""
    prices = np.array([10.0])
    env = PriceEnv(b_E=2.0, b_P=1.0, prices=prices, soc0=0.0, mu=0.0,
                   reward_mode="linear")
    # Try to discharge from empty -> should clip to 0
    env.step(+0.5)
    assert approx(env.soc, 0.0), f"discharge from empty should clip: {env.soc}"
    assert approx(env.action_log[0], 0.0), f"action should clip to 0: {env.action_log[0]}"
    print(f"  Discharge from empty clipped: a={env.action_log[0]}")


def test_continuous_rainflow_stack():
    """4-point rule extracts an inner cycle from a [0,2,1,3,0] trajectory."""
    s = ContinuousRainflowStack(0.0)
    # Stack progression: [0] -> [0] -> [0,2] -> [0,2,1] -> [0,2,1,3] -> simplify -> [0,3]
    # The (b,c)=(2,1) inner cycle has range 1, sandwiched between outer ranges 2 and 2.
    seq = [2.0, 1.0, 3.0, 0.0]
    for x in seq:
        s.update(x)
    assert len(s.completed_cycles) >= 1, f"expected >=1 completed cycle, got {s.completed_cycles}"
    assert approx(s.completed_cycles[0], 1.0), f"expected range 1.0, got {s.completed_cycles}"
    print(f"  Continuous rainflow stack works: cycles={s.completed_cycles}")


# ---- Arbitrage agents --------------------------------------------------------
def test_lp_linear_arbitrage():
    """LP with mu=0 should max revenue: charge low, discharge high."""
    prices = np.array([10.0, 50.0, 100.0])
    a = lp_linear_actions(prices, b_E=2.0, b_P=1.0, soc0=0.0, mu=0.0)
    # Optimal: charge at 10 (a<0), idle or discharge at 50, discharge at 100
    revenue = float(np.dot(prices, a))
    # Charge 1 MWh at 10 (revenue -10), discharge 1 at 100 (revenue +100), net +90
    assert revenue >= 80, f"LP did not arbitrage: revenue={revenue}, actions={a}"
    print(f"  LP arbitrage: actions={a}, revenue={revenue}")


def test_lp_high_mu_zero_dispatch():
    """Very high mu -> LP forces zero dispatch."""
    prices = np.array([10.0, 100.0])
    a = lp_linear_actions(prices, b_E=2.0, b_P=1.0, soc0=1.0, mu=1000.0)
    assert np.allclose(a, 0.0, atol=1e-3), f"high mu should zero dispatch: {a}"
    print(f"  Hi-mu LP zero dispatch: max|a|={abs(a).max():.4f}")


def test_qp_quadratic_smaller_actions_than_lp():
    """QP at moderate alpha gives smaller actions than mu=0 LP."""
    prices = synth_diurnal(24, seed=0)
    a_lp = lp_linear_actions(prices, 2.0, 1.0, 1.0, mu=0.0)
    a_qp = qp_quadratic_actions(prices, 2.0, 1.0, 1.0, alpha=1.0)
    # QP penalizes magnitudes -> max action should be lower
    assert abs(a_qp).max() < abs(a_lp).max() + 1e-6, \
        f"QP alpha=1 should reduce max action: lp={abs(a_lp).max()}, qp={abs(a_qp).max()}"
    print(f"  QP shrinks actions: max|a_lp|={abs(a_lp).max():.3f}, max|a_qp|={abs(a_qp).max():.3f}")


def test_qp_ensemble_jensen_lift():
    """Ensemble of K=4 forecasts beats single forecast under noise."""
    realized = synth_diurnal(72, seed=42)
    NOISE = 8
    K = 4
    rng = np.random.default_rng(0)
    Rs_s, Rs_e = [], []
    for outer in range(4):
        fc_single = make_forecast(realized, NOISE, seed=outer)
        a_s = qp_quadratic_actions(fc_single, 2.0, 2.0, 1.0, alpha=0.005)
        Rs_s.append(float(np.dot(realized, a_s)))
        inner = rng.integers(0, 1_000_000, size=K)
        forecasts = np.stack([
            make_forecast(realized, NOISE, seed=int(s)) for s in inner])
        a_e = qp_ensemble_actions(forecasts, 2.0, 2.0, 1.0, alpha=0.005)
        Rs_e.append(float(np.dot(realized, a_e)))
    R_s = np.mean(Rs_s)
    R_e = np.mean(Rs_e)
    lift = (R_e - R_s) / abs(R_s) * 100
    assert lift > 5, f"Ensemble lift expected >5% under noise=8, got {lift:.1f}%"
    print(f"  Jensen-lift test: ensemble +{lift:.1f}% over single (R_s={R_s:.0f}, R_e={R_e:.0f})")


def test_run_actions_helper():
    """run_actions applies a precomputed action vector and returns summary."""
    prices = np.array([10.0, 100.0])
    env = PriceEnv(2.0, 1.0, prices=prices, soc0=1.0, mu=0.0,
                   reward_mode="linear")
    rec = run_actions(env, np.array([-1.0, 1.0]))
    assert approx(rec["R"], 90.0), f"expected R=90, got {rec['R']}"
    print(f"  run_actions summary: R={rec['R']}, soc_log={rec['soc_log']}")


def main():
    tests = [
        test_price_env_charge_discharge,
        test_price_env_revenue_linear,
        test_price_env_quadratic_cost,
        test_price_env_clipping,
        test_continuous_rainflow_stack,
        test_lp_linear_arbitrage,
        test_lp_high_mu_zero_dispatch,
        test_qp_quadratic_smaller_actions_than_lp,
        test_qp_ensemble_jensen_lift,
        test_run_actions_helper,
    ]
    for t in tests:
        print(f"-- {t.__name__}")
        t()
    print(f"\nAll {len(tests)} PriceEnv tests passed.")


if __name__ == "__main__":
    main()
