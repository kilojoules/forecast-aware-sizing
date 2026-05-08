"""Pilot S1: smoke test for sizing-under-uncertainty thesis.

Question: does the optimal battery energy capacity b_E* shift between
(a) deterministic-perfect-foresight dispatch and (b) stochastic-aware
ensemble dispatch, when both are scored on realized prices?

Setup:
  - 1-week realized synthetic diurnal price trace (T=168 h).
  - Sweep b_E in {1, 2, 4, 8, 16} MWh, fixed b_P = b_E / 4 (4-hour battery
    -- a typical hybrid plant ratio).
  - For each b_E, solve weekly dispatch under:
      (a) deterministic: single-forecast QP (8 forecast seeds avg)
      (b) stochastic:    K=4 ensemble QP (same 8 seeds avg)
  - Project 1-week revenue + degradation to 10-year lifetime NPV.
    Replacement triggered when cumulative D consumes the LoH budget.
  - Find b_E* under each regime; compare.

Go/no-go (per SIZING_PILOTS.md):
  |b_E*_det - b_E*_sto| / b_E*_det >= 10%   -> proceed to S2
  < 5%                                       -> kill, sizing robust

Run:
  python3 pilot_s1.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from arbitrage_agents import (qp_ensemble_actions, qp_quadratic_actions,
                              run_actions)
from degradation import cycle_degradation
from env import PriceEnv
from price_signal import make_forecast, synth_diurnal


# Config
T = 168                      # 1 representative week
WEEKS_PER_YEAR = 52
LIFETIME_YEARS = 15          # plant service life (battery may be replaced ~mid-life)
DISCOUNT_RATE = 0.07
CAPEX_PER_MWH = 100_000      # currency units / MWh (~$100/kWh, 2026 estimate)
CAPEX_PER_MW = 75_000        # PCS/inverter/EMS overhead -- fixed by power, not energy.
                             # Without this, NPV trivially favors the smallest battery
                             # because both revenue and energy CAPEX scale linearly.
LOH_BUDGET = 0.20            # 20% capacity loss => replace
NOISE = 8.0                  # forecast AR(1) noise std (realistic-poor)
ALPHA = 0.005
SEED = 42
K_ENSEMBLE = 4
N_FORECAST_SEEDS = 8         # outer-seed averaging for variance reduction
B_E_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
B_P_FIXED = 1.0              # fix MW; sweep duration only.
                             # Real systems have inverter sizing decoupled
                             # from cell-stack sizing. With b_P fixed, longer-
                             # duration batteries capture longer-period
                             # arbitrage spreads with diminishing returns,
                             # giving a real interior optimum.


def safe_D(soc, b):
    try:
        D, _, _ = cycle_degradation(soc, b)
    except (IndexError, ValueError):
        return 0.0
    return D


def lifetime_npv(R_week: float, D_week: float, b_E: float, b_P: float) -> dict:
    """Project 1-week dispatch metrics to lifetime NPV.

    revenue: assume identical week repeats 52 times/year for LIFETIME_YEARS,
             discounted at DISCOUNT_RATE.
    degradation: same projection. When cumulative D crosses LoH_BUDGET, a
                 replacement event happens (battery cost re-paid, fresh battery
                 starts the next cycle).
    Replacement timing: equal-spaced across lifetime (D growing linearly under
                        the repeating-week assumption is exact).
    """
    annual_revenue = R_week * WEEKS_PER_YEAR
    annual_D = D_week * WEEKS_PER_YEAR
    # Discount factors per year (year 0 = present)
    df = np.array([(1.0 / (1 + DISCOUNT_RATE)) ** y for y in range(LIFETIME_YEARS)])
    lifetime_revenue = annual_revenue * df.sum()

    lifetime_D = annual_D * LIFETIME_YEARS
    # n_replacements: how many times the battery dies (LoH_budget consumed).
    if annual_D > 0:
        years_per_replacement = LOH_BUDGET / annual_D
        n_replacements = int(np.floor(LIFETIME_YEARS / years_per_replacement))
    else:
        years_per_replacement = float("inf")
        n_replacements = 0
    # Initial CAPEX = energy + power components.
    # Replacements only re-pay the energy component (PCS / inverter is reused).
    capex_initial = b_E * CAPEX_PER_MWH + b_P * CAPEX_PER_MW
    capex_replacement = 0.0
    for r in range(n_replacements):
        y = (r + 1) * years_per_replacement
        if y < LIFETIME_YEARS:
            capex_replacement += b_E * CAPEX_PER_MWH * (1.0 / (1 + DISCOUNT_RATE)) ** y
    total_capex = capex_initial + capex_replacement
    npv = lifetime_revenue - total_capex
    return {
        "lifetime_revenue": lifetime_revenue,
        "lifetime_D": lifetime_D,
        "annual_D": annual_D,
        "n_replacements": n_replacements,
        "capex_initial": capex_initial,
        "capex_replacement": capex_replacement,
        "total_capex": total_capex,
        "npv": npv,
    }


def main():
    print(f"Pilot S1: sizing sweep, T={T}h week, lifetime={LIFETIME_YEARS}yr, "
          f"noise={NOISE}, K={K_ENSEMBLE}, alpha={ALPHA}\n")

    realized = synth_diurnal(T, seed=SEED)
    print(f"Realized week: mean={realized.mean():.1f} std={realized.std():.1f} "
          f"range=[{realized.min():.1f}, {realized.max():.1f}]\n")

    rng = np.random.default_rng(123)
    rows = []
    print(f"{'b_E':>5} | {'b_P':>5} | "
          f"{'R_det':>9} | {'D_det':>8} | {'rep_d':>5} | {'NPV_det':>11} | "
          f"{'R_sto':>9} | {'D_sto':>8} | {'rep_s':>5} | {'NPV_sto':>11} | "
          f"{'NPV_dif':>9}")
    print("-" * 120)
    t0 = time.time()
    for b_E in B_E_VALUES:
        b_P = B_P_FIXED
        Rs_det, Ds_det = [], []
        Rs_sto, Ds_sto = [], []
        for outer_seed in range(N_FORECAST_SEEDS):
            forecast_single = make_forecast(realized, NOISE, seed=outer_seed)
            inner_seeds = rng.integers(0, 1_000_000, size=K_ENSEMBLE)
            forecasts_K = np.stack([
                make_forecast(realized, NOISE, seed=int(s)) for s in inner_seeds])

            env = PriceEnv(b_E, b_P, prices=realized,
                           alpha=ALPHA, reward_mode="quadratic")
            a_det = qp_quadratic_actions(forecast_single, b_E, b_P, env.soc0,
                                          alpha=ALPHA)
            rec = run_actions(env, a_det)
            Rs_det.append(rec["R"]); Ds_det.append(safe_D(rec["soc_log"], b_E))

            env = PriceEnv(b_E, b_P, prices=realized,
                           alpha=ALPHA, reward_mode="quadratic")
            a_sto = qp_ensemble_actions(forecasts_K, b_E, b_P, env.soc0,
                                         alpha=ALPHA)
            rec = run_actions(env, a_sto)
            Rs_sto.append(rec["R"]); Ds_sto.append(safe_D(rec["soc_log"], b_E))

        R_det = float(np.mean(Rs_det)); D_det = float(np.mean(Ds_det))
        R_sto = float(np.mean(Rs_sto)); D_sto = float(np.mean(Ds_sto))
        npv_det = lifetime_npv(R_det, D_det, b_E, b_P)
        npv_sto = lifetime_npv(R_sto, D_sto, b_E, b_P)
        diff = npv_sto["npv"] - npv_det["npv"]
        rows.append({
            "b_E": b_E, "b_P": b_P,
            "R_det": R_det, "D_det": D_det,
            "rep_det": npv_det["n_replacements"], "NPV_det": npv_det["npv"],
            "R_sto": R_sto, "D_sto": D_sto,
            "rep_sto": npv_sto["n_replacements"], "NPV_sto": npv_sto["npv"],
            "NPV_diff": diff,
        })
        print(f"{b_E:5.2f} | {b_P:5.2f} | "
              f"{R_det:9.1f} | {D_det:.5f} | {npv_det['n_replacements']:5d} | {npv_det['npv']:11.0f} | "
              f"{R_sto:9.1f} | {D_sto:.5f} | {npv_sto['n_replacements']:5d} | {npv_sto['npv']:11.0f} | "
              f"{diff:+9.0f}")

    print(f"\nDone in {time.time()-t0:.0f}s")

    # Find optimal b_E for each regime
    npv_det_arr = np.array([r["NPV_det"] for r in rows])
    npv_sto_arr = np.array([r["NPV_sto"] for r in rows])
    b_E_arr = np.array([r["b_E"] for r in rows])

    bE_star_det = b_E_arr[int(np.argmax(npv_det_arr))]
    bE_star_sto = b_E_arr[int(np.argmax(npv_sto_arr))]
    npv_max_det = float(npv_det_arr.max())
    npv_max_sto = float(npv_sto_arr.max())

    if bE_star_det > 0:
        shift_pct = (bE_star_sto - bE_star_det) / bE_star_det * 100
    else:
        shift_pct = 0.0

    # Gap in realized NPV between regimes at det's chosen size
    npv_at_det_choice = rows[int(np.argmax(npv_det_arr))]["NPV_sto"]
    npv_uplift_pct = ((npv_max_sto - npv_at_det_choice)
                      / max(abs(npv_at_det_choice), 1.0) * 100)

    print(f"\nOptimal sizing:")
    print(f"  Deterministic dispatch: b_E* = {bE_star_det:.2f} MWh, NPV = {npv_max_det:.0f}")
    print(f"  Stochastic dispatch:    b_E* = {bE_star_sto:.2f} MWh, NPV = {npv_max_sto:.0f}")
    print(f"\n  Shift in b_E*: {shift_pct:+.1f}%")
    print(f"  NPV uplift from re-sizing under stochastic dispatch: {npv_uplift_pct:+.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    ax.plot(b_E_arr, npv_det_arr, "o-", color="#cc6677", label="Deterministic dispatch")
    ax.plot(b_E_arr, npv_sto_arr, "s-", color="#4477aa", label=f"Stochastic K={K_ENSEMBLE} dispatch")
    ax.axvline(bE_star_det, color="#cc6677", ls=":", alpha=0.5)
    ax.axvline(bE_star_sto, color="#4477aa", ls=":", alpha=0.5)
    ax.set_xlabel("Battery energy capacity b_E (MWh)")
    ax.set_ylabel("Lifetime NPV")
    ax.set_xscale("log")
    ax.set_title(f"NPV vs b_E (shift={shift_pct:+.1f}%)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    rep_det = np.array([r["rep_det"] for r in rows])
    rep_sto = np.array([r["rep_sto"] for r in rows])
    ax.plot(b_E_arr, rep_det, "o-", color="#cc6677", label="Deterministic")
    ax.plot(b_E_arr, rep_sto, "s-", color="#4477aa", label="Stochastic")
    ax.set_xlabel("Battery energy capacity b_E (MWh)")
    ax.set_ylabel("# replacements over lifetime")
    ax.set_xscale("log")
    ax.set_title("Replacement count")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    R_det_arr = np.array([r["R_det"] for r in rows])
    R_sto_arr = np.array([r["R_sto"] for r in rows])
    ax.plot(b_E_arr, R_det_arr, "o-", color="#cc6677", label="Deterministic")
    ax.plot(b_E_arr, R_sto_arr, "s-", color="#4477aa", label="Stochastic")
    ax.set_xlabel("Battery energy capacity b_E (MWh)")
    ax.set_ylabel("Weekly revenue")
    ax.set_xscale("log")
    ax.set_title("Weekly arbitrage revenue")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Pilot S1: sizing under deterministic vs stochastic dispatch  "
        f"(T={T}h week, lifetime={LIFETIME_YEARS}yr, noise={NOISE}, K={K_ENSEMBLE})",
        y=1.02)
    fig.tight_layout()
    out = "fig_pilot_s1.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")

    # Verdict
    print(f"\n{'='*72}")
    if abs(shift_pct) >= 10:
        print(f"VERDICT: shift {shift_pct:+.1f}% >= 10% -> GO. Run pilot S2 (noise sweep).")
    elif abs(shift_pct) < 5:
        print(f"VERDICT: shift {shift_pct:+.1f}% < 5% -> KILL. Sizing robust to dispatch fidelity.")
    else:
        print(f"VERDICT: shift {shift_pct:+.1f}% in 5-10% range -> marginal. Investigate before S2.")
    print(f"NPV uplift from stochastic-aware sizing on the optimal config: {npv_uplift_pct:+.1f}%")

    # Quick noise sweep: does sizing shift appear at higher forecast noise?
    print(f"\n--- Quick noise sweep at b_E in {B_E_VALUES} ---")
    print(f"{'noise':>5} | {'bE*_det':>7} | {'bE*_sto':>7} | {'shift':>6} | {'NPV_det':>9} | {'NPV_sto':>9} | {'uplift':>6}")
    print("-" * 75)
    rng2 = np.random.default_rng(456)
    for noise_level in [3.0, 8.0, 18.0]:
        npv_det_vec, npv_sto_vec = [], []
        for b_E in B_E_VALUES:
            b_P = B_P_FIXED
            Rs_det, Ds_det = [], []
            Rs_sto, Ds_sto = [], []
            for outer_seed in range(N_FORECAST_SEEDS):
                forecast_single = make_forecast(realized, noise_level, seed=outer_seed)
                inner_seeds = rng2.integers(0, 1_000_000, size=K_ENSEMBLE)
                forecasts_K = np.stack([
                    make_forecast(realized, noise_level, seed=int(s)) for s in inner_seeds])
                env = PriceEnv(b_E, b_P, prices=realized, alpha=ALPHA, reward_mode="quadratic")
                a_det = qp_quadratic_actions(forecast_single, b_E, b_P, env.soc0, alpha=ALPHA)
                rec = run_actions(env, a_det)
                Rs_det.append(rec["R"]); Ds_det.append(safe_D(rec["soc_log"], b_E))
                env = PriceEnv(b_E, b_P, prices=realized, alpha=ALPHA, reward_mode="quadratic")
                a_sto = qp_ensemble_actions(forecasts_K, b_E, b_P, env.soc0, alpha=ALPHA)
                rec = run_actions(env, a_sto)
                Rs_sto.append(rec["R"]); Ds_sto.append(safe_D(rec["soc_log"], b_E))
            npv_d = lifetime_npv(float(np.mean(Rs_det)), float(np.mean(Ds_det)), b_E, b_P)["npv"]
            npv_s = lifetime_npv(float(np.mean(Rs_sto)), float(np.mean(Ds_sto)), b_E, b_P)["npv"]
            npv_det_vec.append(npv_d); npv_sto_vec.append(npv_s)
        npv_det_vec = np.array(npv_det_vec); npv_sto_vec = np.array(npv_sto_vec)
        bE_d = B_E_VALUES[int(npv_det_vec.argmax())]
        bE_s = B_E_VALUES[int(npv_sto_vec.argmax())]
        shift = (bE_s - bE_d) / max(bE_d, 1e-9) * 100
        uplift = (npv_sto_vec.max() - npv_det_vec[int(npv_det_vec.argmax())]) / max(abs(npv_det_vec[int(npv_det_vec.argmax())]), 1.0) * 100
        print(f"{noise_level:5.1f} | {bE_d:7.2f} | {bE_s:7.2f} | {shift:+5.1f}% | "
              f"{npv_det_vec.max():9.0f} | {npv_sto_vec.max():9.0f} | {uplift:+5.1f}%")


if __name__ == "__main__":
    main()
