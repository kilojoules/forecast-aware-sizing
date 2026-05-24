"""Pilot (a) day 1: measure value-of-perfect-foresight on hourly arbitrage.

Setup:
  Realized price trace = synth diurnal AR(1) (same as sanity_arbitrage.py).
  Forecast = realized + AR(1) noise with varying std (= forecast MAE proxy).

Policies:
  1. LP-oracle: solves on realized (perfect foresight). Upper bound.
  2. LP-rolling-horizon-with-forecast: at each step, solves k-step LP on
     forecast slice [t..t+k], applies first action, advances. Re-solves.
  3. LP-deterministic-full-horizon-with-forecast: solves once on the
     full forecast trace (this is what hydesign-style does). Score on realized.

For each forecast-noise level, score all three on realized prices.
Sweep noise; plot R(noise) for each policy.

Headline number: value-of-perfect-foresight = (R_oracle - R_with_forecast) / R_oracle.
This is the upside ceiling for any forecast-robust RL policy.

If ceiling is small (<10%), the forecast-uncertainty angle is dead.
If ceiling is meaningful (>15%), pilot day 2-3 builds RL to close it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from degradation import cycle_degradation


T = 168
B_E = 2.0
B_P = 2.0
SOC0 = 0.5 * B_E
SEED = 42


def synth_price(T: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hours = np.arange(T)
    diurnal = (
        50.0
        + 35.0 * np.sin(2 * np.pi * (hours - 8) / 24)
        + 15.0 * np.sin(2 * np.pi * (hours - 19) / 24 * 2)
    )
    noise = np.zeros(T)
    rho = 0.7
    for t in range(1, T):
        noise[t] = rho * noise[t-1] + rng.normal(0, 8.0)
    return diurnal + noise


def make_forecast(realized: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    """Forecast = realized + AR(1) noise (rho=0.5) with given std."""
    rng = np.random.default_rng(seed + 1000)
    fc_noise = np.zeros_like(realized)
    rho = 0.5
    sigma_innov = noise_std * np.sqrt(1 - rho**2)
    for t in range(1, len(realized)):
        fc_noise[t] = rho * fc_noise[t-1] + rng.normal(0, sigma_innov)
    return realized + fc_noise


def lp_dispatch(price: np.ndarray, mu: float, soc0: float = SOC0):
    """Full-horizon LP on given price array. Returns (P_chg, P_dis) of length len(price)."""
    n_t = len(price)
    n = 2 * n_t
    c = np.concatenate([price + mu, -price + mu])
    bounds = [(0, B_P)] * n
    A_ub = np.zeros((2 * n_t, n))
    b_ub = np.zeros(2 * n_t)
    for t in range(n_t):
        A_ub[t, 0:t+1] = 1.0
        A_ub[t, n_t:n_t+t+1] = -1.0
        b_ub[t] = B_E - soc0
        A_ub[n_t+t, 0:t+1] = -1.0
        A_ub[n_t+t, n_t:n_t+t+1] = 1.0
        b_ub[n_t+t] = soc0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x[:n_t], res.x[n_t:]


def simulate_with_actions(P_chg: np.ndarray, P_dis: np.ndarray, soc0: float = SOC0):
    """Apply actions, clipping at SoC bounds. Returns SoC trajectory length T+1, effective P_chg, P_dis."""
    soc = soc0
    socs = [soc]
    eff_chg = np.zeros_like(P_chg)
    eff_dis = np.zeros_like(P_dis)
    for t in range(len(P_chg)):
        chg = min(P_chg[t], B_E - soc)
        dis = min(P_dis[t], soc)
        soc = soc + chg - dis
        socs.append(soc)
        eff_chg[t] = chg
        eff_dis[t] = dis
    return np.array(socs), eff_chg, eff_dis


def revenue_realized(realized: np.ndarray, P_chg: np.ndarray, P_dis: np.ndarray) -> float:
    return float(np.sum(realized * (P_dis - P_chg)))


def D_of(soc: np.ndarray, B: float) -> float:
    try:
        D, _, _ = cycle_degradation(soc, B)
    except (IndexError, ValueError):
        return 0.0
    return D


# -----------------------------------------------------------------------------
# Policies
# -----------------------------------------------------------------------------
def policy_lp_oracle(realized, mu):
    """Full-horizon LP on realized. Upper bound."""
    P_chg, P_dis = lp_dispatch(realized, mu)
    soc, P_chg_e, P_dis_e = simulate_with_actions(P_chg, P_dis)
    return soc, P_chg_e, P_dis_e


def policy_lp_full_with_forecast(realized, forecast, mu):
    """Full-horizon LP on forecast, but actions executed on realized.
    Realized SoC dynamics may clip differently than the LP planned."""
    P_chg, P_dis = lp_dispatch(forecast, mu)
    soc, P_chg_e, P_dis_e = simulate_with_actions(P_chg, P_dis)
    return soc, P_chg_e, P_dis_e


def policy_lp_rolling(realized, forecast, mu, lookahead: int = 24):
    """Rolling horizon: at each step t, solve k-step LP on forecast[t..t+k-1],
    apply first action, advance state.
    """
    soc = SOC0
    socs = [soc]
    eff_chg = np.zeros(len(realized))
    eff_dis = np.zeros(len(realized))
    for t in range(len(realized)):
        end = min(t + lookahead, len(forecast))
        fc_slice = forecast[t:end]
        if len(fc_slice) == 0:
            break
        P_chg, P_dis = lp_dispatch(fc_slice, mu, soc0=soc)
        chg = min(P_chg[0], B_E - soc)
        dis = min(P_dis[0], soc)
        soc = soc + chg - dis
        socs.append(soc)
        eff_chg[t] = chg
        eff_dis[t] = dis
    return np.array(socs), eff_chg, eff_dis


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
def main():
    print("Pilot (a) day 1: value-of-perfect-foresight ceiling for hourly arbitrage")
    print(f"T={T}h  B_E={B_E}  B_P={B_P}  seed={SEED}\n")
    realized = synth_price(T, seed=SEED)
    print(f"Realized: mean={realized.mean():.1f} std={realized.std():.1f} "
          f"range=[{realized.min():.1f}, {realized.max():.1f}]")

    # Use modest mu so the policy actually cycles; in mu=0 oracle is max-revenue.
    MU = 0.0
    print(f"\nFixing mu={MU}. (If mu=0, pure revenue maximization, ignore degradation.)\n")

    # Oracle
    soc_or, chg_or, dis_or = policy_lp_oracle(realized, MU)
    R_or = revenue_realized(realized, chg_or, dis_or)
    D_or = D_of(soc_or, B_E)
    print(f"LP-oracle (perfect foresight):           R={R_or:7.1f}  D={D_or:.5f}")

    # Sweep forecast noise std
    noise_levels = [1, 3, 5, 8, 12, 18, 25]   # std of AR(1) forecast noise
    print(f"\nForecast noise std sweep: {noise_levels}\n")

    rows = []
    for noise_std in noise_levels:
        # Average over a few seeds for the forecast noise (NOT the realized trace)
        Rs_full, Rs_roll = [], []
        Ds_full, Ds_roll = [], []
        for fc_seed in range(5):
            forecast = make_forecast(realized, noise_std, seed=fc_seed)
            mae = np.mean(np.abs(forecast - realized))
            # Full-horizon LP with forecast, executed on realized
            soc_f, chg_f, dis_f = policy_lp_full_with_forecast(realized, forecast, MU)
            Rs_full.append(revenue_realized(realized, chg_f, dis_f))
            Ds_full.append(D_of(soc_f, B_E))
            # Rolling horizon LP (lookahead=24h)
            soc_r, chg_r, dis_r = policy_lp_rolling(realized, forecast, MU, lookahead=24)
            Rs_roll.append(revenue_realized(realized, chg_r, dis_r))
            Ds_roll.append(D_of(soc_r, B_E))
        R_full, R_roll = np.mean(Rs_full), np.mean(Rs_roll)
        D_full, D_roll = np.mean(Ds_full), np.mean(Ds_roll)
        gap_full = (R_or - R_full) / R_or * 100 if R_or > 0 else 0.0
        gap_roll = (R_or - R_roll) / R_or * 100 if R_or > 0 else 0.0
        rows.append((noise_std, R_full, gap_full, R_roll, gap_roll, D_full, D_roll))
        print(f"  noise={noise_std:5.1f}: "
              f"full-LP R={R_full:7.1f} ({gap_full:+5.1f}% vs oracle)  "
              f"roll-LP R={R_roll:7.1f} ({gap_roll:+5.1f}% vs oracle)  "
              f"D_full={D_full:.4f}  D_roll={D_roll:.4f}")

    # ---- Plot ---------------------------------------------------------------
    rows = np.array(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(rows[:, 0], rows[:, 2], "o-", color="#cc6677",
            label="Full-horizon LP-with-forecast vs oracle")
    ax.plot(rows[:, 0], rows[:, 4], "s-", color="#4477aa",
            label="Rolling-horizon LP (k=24h) vs oracle")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Forecast AR(1) noise std")
    ax.set_ylabel("Revenue gap vs oracle (%)")
    ax.set_title("Value-of-perfect-foresight ceiling vs forecast quality")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(rows[:, 0], rows[:, 1], "o-", color="#cc6677", label="Full-horizon w/ forecast")
    ax.plot(rows[:, 0], rows[:, 3], "s-", color="#4477aa", label="Rolling 24h w/ forecast")
    ax.axhline(R_or, color="black", lw=1, ls="--", label=f"Oracle R={R_or:.0f}")
    ax.set_xlabel("Forecast AR(1) noise std")
    ax.set_ylabel("Realized revenue")
    ax.set_title("Realized revenue vs forecast noise (averaged over 5 noise seeds)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Pilot (a) day 1: forecast-quality ceiling for hourly arbitrage\n"
        f"T={T}h, B_E={B_E}MWh, B_P={B_P}MW, mu=0",
        y=1.02)
    fig.tight_layout()
    out = "fig_forecast_ceiling.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")

    # ---- Verdict ---------------------------------------------------------------
    max_gap_full = rows[:, 2].max()
    max_gap_roll = rows[:, 4].max()
    print(f"\n{'='*70}")
    print(f"Max gap: full={max_gap_full:+.1f}%, roll={max_gap_roll:+.1f}%")
    if max(max_gap_full, max_gap_roll) > 15:
        print("Forecast gap >15% at high noise. Forecast-uncertainty angle has room.")
        print("-> day 2-3: build forecast-robust policy and try to close the gap.")
    elif max(max_gap_full, max_gap_roll) > 5:
        print(f"Modest gap (5-15%). Marginal upside for RL.")
    else:
        print("Gap <5% across all noise levels. Forecast-uncertainty angle dead.")
        print("-> KILL pilot (a). Lit review said 11-16%; this trace says less.")


if __name__ == "__main__":
    main()
