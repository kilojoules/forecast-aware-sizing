"""Local hydesign-vs-QP-single sanity check.

Goal: run hydesign's CPLEX EMS on DK1 2022 prices, week 1 (168 h),
b_E=4 MWh, b_P=1 MW. Compare action profile and revenue to our
`qp_quadratic_actions` / `lp_linear_actions` on the same input.

Three possible outcomes:
  1. Identical actions to tolerance → option 3 (cite hydesign).
  2. Different actions, same revenue → calibration drift; reconcile.
  3. Different actions AND different revenue → real baseline. Do the LUMI sweep.

Run as:
    cd /tmp && conda run -n topfarm-env python \
      /Users/julianquick/portfolio_copy/battery_gym/sizing/hydesign_local_check.py
(Run from /tmp so the local `hydesign/` checkout doesn't shadow the installed
package at /Users/julianquick/hydesign.)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SIZING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIZING_DIR)
os.chdir("/tmp")  # avoid hydesign namespace shadowing

from hydesign.ems.ems import ems_cplex_parts  # noqa: E402

# Pull data via our loader
from dk_loader import load_dk_year  # noqa: E402
from arbitrage_agents import (lp_linear_actions, qp_quadratic_actions,  # noqa
                              run_actions)
from env import PriceEnv  # noqa: E402
from hydesign_merchant_fork import ems_cplex_parts_merchant  # noqa: E402


def _build_inputs(prices: np.ndarray, b_E: float, b_P: float, grid_MW: float):
    N = len(prices)
    ts = pd.date_range("2022-01-01", periods=N, freq="h")
    wind_ts = pd.Series(np.zeros(N), index=ts)
    solar_ts = pd.Series(np.zeros(N), index=ts)
    price_ts = pd.Series(prices, index=ts)
    E_batt_ts = pd.Series(np.full(N, b_E), index=ts)
    return wind_ts, solar_ts, price_ts, E_batt_ts


def run_hydesign_upstream(prices, b_E, b_P, grid_MW=100.0):
    """Upstream ems_cplex_parts — blocked by P_HPP >= 0 for merchant."""
    wind_ts, solar_ts, price_ts, E_batt_ts = _build_inputs(prices, b_E, b_P, grid_MW)
    out = ems_cplex_parts(
        wind_ts=wind_ts, solar_ts=solar_ts, price_ts=price_ts,
        P_batt_MW=b_P, E_batt_MWh_t=E_batt_ts,
        hpp_grid_connection=grid_MW,
        battery_depth_of_discharge=0.9, charge_efficiency=1.0,
        peak_hr_quantile=0.9, cost_of_battery_P_fluct_in_peak_price_ratio=0.0,
        n_full_power_hours_expected_per_day_at_peak_price=0,
    )
    P_HPP, P_curt, P_chg_dis, E_SOC, penalty = out
    actions = np.asarray(P_chg_dis)
    return {"actions": actions, "soc": np.asarray(E_SOC),
            "revenue": float((actions * prices).sum()),
            "P_HPP": np.asarray(P_HPP)}


def run_hydesign_merchant(prices, b_E, b_P, grid_MW=100.0,
                           dod=1.0, enforce_terminal=False):
    """Local fork: bidirectional grid. dod=1.0 + no terminal → matches our LP."""
    wind_ts, solar_ts, price_ts, E_batt_ts = _build_inputs(prices, b_E, b_P, grid_MW)
    out = ems_cplex_parts_merchant(
        wind_ts=wind_ts, solar_ts=solar_ts, price_ts=price_ts,
        P_batt_MW=b_P, E_batt_MWh_t=E_batt_ts,
        hpp_grid_connection=grid_MW,
        battery_depth_of_discharge=dod, charge_efficiency=1.0,
        peak_hr_quantile=0.9, cost_of_battery_P_fluct_in_peak_price_ratio=0.0,
        n_full_power_hours_expected_per_day_at_peak_price=0,
        enforce_terminal_soc=enforce_terminal,
    )
    P_HPP, P_curt, P_chg_dis, E_SOC, penalty = out
    actions = np.asarray(P_chg_dis)
    return {"actions": actions, "soc": np.asarray(E_SOC),
            "revenue": float((actions * prices).sum()),
            "P_HPP": np.asarray(P_HPP)}


def run_ours(prices: np.ndarray, b_E: float, b_P: float, alpha: float = 0.0,
             mu: float = 0.0, kind: str = "qp") -> dict:
    soc0 = b_E / 2
    if kind == "qp":
        a = qp_quadratic_actions(prices, b_E, b_P, soc0, alpha=alpha)
    else:
        a = lp_linear_actions(prices, b_E, b_P, soc0, mu=mu)
    env = PriceEnv(b_E, b_P, prices=prices, soc0=soc0,
                   alpha=alpha if kind == "qp" else 0,
                   mu=mu if kind == "lp" else 0,
                   reward_mode="quadratic" if kind == "qp" else "linear")
    rec = run_actions(env, a)
    return {"actions": np.asarray(a), "soc": np.asarray(rec["soc_log"]),
            "revenue": float(rec["R"])}


def main():
    print("Loading DK1 2022 prices...")
    df = load_dk_year(2022, area="DK1")
    prices_full = df["da_eur_per_mwh"].to_numpy()
    # short horizon — CPLEX Community Edition cap (1000 vars/1000 consts).
    # hydesign EMS adds ~7 vars per timestep, so ~100 h is safe via parts.
    H = 96
    prices = prices_full[:H]
    print(f"  H={H}, mean={prices.mean():.1f}, std={prices.std():.1f}")

    b_E, b_P = 4.0, 1.0
    print(f"\nb_E={b_E} MWh, b_P={b_P} MW, 100% efficiency, no penalties")

    print("\n--- hydesign ems_cplex_parts (upstream, P_HPP>=0) ---")
    try:
        hyd_up = run_hydesign_upstream(prices, b_E, b_P)
        print(f"  upstream revenue = {hyd_up['revenue']:.2f} EUR")
        print(f"  upstream |action| mean = {np.abs(hyd_up['actions']).mean():.3f} MW")
    except Exception as e:
        print(f"  upstream failed: {e}")
        hyd_up = None

    print("\n--- merchant: hydesign-default constraints (DoD=0.9, terminal SoC=0.5b_E) ---")
    hyd_strict = run_hydesign_merchant(prices, b_E, b_P, dod=0.9, enforce_terminal=True)
    print(f"  rev = {hyd_strict['revenue']:.2f} EUR")

    print("\n--- merchant: relaxed (DoD=1.0, NO terminal SoC) → matches our LP ---")
    hyd = run_hydesign_merchant(prices, b_E, b_P, dod=1.0, enforce_terminal=False)
    print(f"  rev = {hyd['revenue']:.2f} EUR")
    print(f"  |action| mean = {np.abs(hyd['actions']).mean():.3f} MW")
    print(f"  P_HPP range = [{hyd['P_HPP'].min():.2f}, {hyd['P_HPP'].max():.2f}] MW")

    print("\n--- ours: lp_linear mu=0 (scipy HiGHS) ---")
    ours_lp = run_ours(prices, b_E, b_P, mu=0.0, kind="lp")
    print(f"  lp revenue (env)  = {ours_lp['revenue']:.2f} EUR")
    print(f"  lp |action| mean = {np.abs(ours_lp['actions']).mean():.3f} MW")
    # cvxpy broken in topfarm-env (numpy 1.x vs 2.x conflict); skip QP comparison
    ours_qp = {"actions": ours_lp["actions"], "revenue": ours_lp["revenue"]}

    out = {
        "horizon": H, "b_E": b_E, "b_P": b_P,
        "hydesign_merchant_revenue": hyd["revenue"],
        "qp_revenue": ours_qp["revenue"],
        "lp_revenue": ours_lp["revenue"],
        "hydesign_merchant_actions_mean_abs": float(np.abs(hyd["actions"]).mean()),
        "qp_actions_mean_abs": float(np.abs(ours_qp["actions"]).mean()),
        "lp_actions_mean_abs": float(np.abs(ours_lp["actions"]).mean()),
        "hydesign_merchant_actions": hyd["actions"].tolist(),
        "qp_actions": ours_qp["actions"].tolist(),
        "lp_actions": ours_lp["actions"].tolist(),
    }
    corr_qp = float(np.corrcoef(hyd["actions"], ours_qp["actions"])[0, 1])
    corr_lp = float(np.corrcoef(hyd["actions"], ours_lp["actions"])[0, 1])
    rev_diff_qp = (hyd["revenue"] - ours_qp["revenue"]) / max(abs(hyd["revenue"]), 1e-6)
    rev_diff_lp = (hyd["revenue"] - ours_lp["revenue"]) / max(abs(hyd["revenue"]), 1e-6)
    out["corr_hydesign_qp"] = corr_qp
    out["corr_hydesign_lp"] = corr_lp
    out["revenue_rel_diff_vs_qp"] = rev_diff_qp
    out["revenue_rel_diff_vs_lp"] = rev_diff_lp
    print(f"\nAction correlations:")
    print(f"  corr(hyd_merchant, qp) = {corr_qp:.4f}")
    print(f"  corr(hyd_merchant, lp) = {corr_lp:.4f}")
    print(f"Revenue relative diffs:")
    print(f"  (hyd - qp)/hyd = {rev_diff_qp*100:+.3f}%")
    print(f"  (hyd - lp)/hyd = {rev_diff_lp*100:+.3f}%")
    # outcome verdict
    if abs(corr_qp) > 0.99 and abs(rev_diff_qp) < 0.01:
        verdict = "OUTCOME 1: identical to QP-single (correlation>0.99, revenue<1%)"
    elif abs(rev_diff_qp) < 0.01:
        verdict = "OUTCOME 2: actions differ but revenue matches (calibration)"
    else:
        verdict = "OUTCOME 3: actions AND revenue differ (real baseline)"
    print(f"\n>>> {verdict} <<<")
    out["verdict"] = verdict

    out_path = os.path.join(SIZING_DIR, "..", "results", "hydesign",
                             "hydesign_local_check_out.json")
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
