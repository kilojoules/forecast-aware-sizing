"""Imbalance-penalty break-point study (DK1 wind + battery).

Question: forecast errors are free in the pure-arbitrage model
(no production commitment). Real merchants face imbalance settlement.
At what imbalance-price level lambda does forecast quality begin to
change sizing decisions?

Setup (DK1 only, 3 years):
  - Wind 1 MW + battery b_P=1 MW co-located, single grid commit.
  - Day-ahead: policy commits a_sched_t (price-forecast LP) and
    hat_w_t (wind forecast, persistence or multi-lag ensemble).
  - Real-time: realized w_t known. Battery absorbs wind error within
    feasibility: delta_a_t = clip(hat_w_t - w_t, ...).
  - Imbalance residual r_t = w_t - hat_w_t + delta_a_t.
  - Cost = lambda * sum |r_t|.
  - NPV(b_E, lambda) = DISC * (arb_rev + wind_DA_rev - imbalance_cost)
                       - CAPEX_E * b_E - CAPEX_P * b_P.

Sweep:
  - b_E in B_E_GRID (paper default).
  - lambda in {0, 5, 10, 25, 50, 100} EUR/MWh.
  - Two policies: single (persistence lag 24h) and K=4 multi-lag ensemble.
    Wind forecast follows same lag(s) as price forecast.

Output: results_imbalance/dk1_{year}.json
        per (lambda, policy, b_E): {arb_rev, wind_da_rev, imb_cost, npv}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BG = "/Users/julianquick/portfolio_copy/battery_gym"
sys.path.insert(0, BG)

from arbitrage_agents import lp_linear_actions  # noqa: E402
from dk_loader import load_dk_year, load_dk_resources  # noqa: E402

B_E_GRID = [0.5, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
B_P = 1.0
WIND_SCALE_MW = 5.0  # peak wind = 5 MW (typical HPP overbuild: wind > battery)
LAGS_K4 = (24, 48, 168, 336)
LAMBDA_GRID = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0]  # EUR/MWh
DISC = sum(1.07 ** -y for y in range(15))  # ~9.108
CAPEX_E = 100_000.0
CAPEX_P = 75_000.0
CHUNK_HOURS = 24 * 7 * 8  # 8 weeks; matches paper_benchmark stitch


def build_persistence_forecasts(series: np.ndarray, lags: tuple[int, ...]):
    """Return (realized, forecasts_K) aligned at max(lags)."""
    max_lag = max(lags)
    T = len(series) - max_lag
    realized = series[max_lag:]
    K = len(lags)
    F = np.zeros((K, T))
    for k, lag in enumerate(lags):
        F[k] = series[max_lag - lag: len(series) - lag]
    return realized, F


def absorb_wind_error(a_sched: np.ndarray, w_realized: np.ndarray,
                      w_forecast: np.ndarray, b_E: float, b_P: float,
                      soc0: float):
    """Walk SoC, deviate battery to absorb wind error within feasibility.

    Convention: action > 0 = discharge (subtract from SoC).
    Wind enters as injection; surplus wind (w > hat_w) wants battery to
    absorb (charge harder, action more negative).
    """
    T = len(a_sched)
    soc = float(soc0)
    delta = np.zeros(T)
    residual = np.zeros(T)
    for t in range(T):
        e = w_realized[t] - w_forecast[t]  # surplus if positive
        # battery wants delta = -e (charge surplus, discharge shortfall)
        target = -e
        # power feasibility: a_sched + delta in [-b_P, b_P]
        dlo_p = -b_P - a_sched[t]
        dhi_p = b_P - a_sched[t]
        # SoC feasibility: soc - (a_sched + delta) in [0, b_E]
        # => delta in [soc - b_E - a_sched, soc - a_sched]
        dlo_s = soc - b_E - a_sched[t]
        dhi_s = soc - a_sched[t]
        dlo = max(dlo_p, dlo_s)
        dhi = min(dhi_p, dhi_s)
        if dhi < dlo:
            d = 0.0  # infeasible, no correction
        else:
            d = float(np.clip(target, dlo, dhi))
        delta[t] = d
        residual[t] = e + d  # uncompensated imbalance
        soc = soc - (a_sched[t] + d)
        soc = max(0.0, min(b_E, soc))  # numerical safety
    return delta, residual


def _chunk_indices(T: int, chunk: int = CHUNK_HOURS):
    s = 0
    while s < T:
        yield s, min(s + chunk, T)
        s += chunk


def dispatch_chunked(prices_real, p_fcst, w_real, w_fcst, b_E, b_P):
    """Solve LP per 8-week chunk; absorb wind error inside chunk.

    Returns concatenated arrays plus stats: a_sched, residual.
    SoC continuity across chunks (start of next chunk = end of prev real SoC).
    """
    T = len(prices_real)
    a_sched_full = np.zeros(T)
    residual_full = np.zeros(T)
    soc_carry = b_E / 2.0
    for s, e in _chunk_indices(T):
        a = lp_linear_actions(p_fcst[s:e], b_E, b_P, soc_carry, mu=0.0)
        a_sched_full[s:e] = a
        # Absorb wind error within chunk; SoC carry update reflects realized
        # battery action (a_sched + delta).
        delta, res = absorb_wind_error(a, w_real[s:e], w_fcst[s:e],
                                        b_E, b_P, soc_carry)
        residual_full[s:e] = res
        soc_carry = soc_carry - float(np.sum(a + delta))
        soc_carry = max(0.0, min(b_E, soc_carry))
    return a_sched_full, residual_full


def run_cell(prices_real, p_fcst_single, p_fcst_ens_mean,
             w_real, w_fcst_single, w_fcst_ens_mean,
             b_E: float, b_P: float):
    """One b_E, both policies. Returns dict policy -> {arb, wind_da, imb_abs}."""
    out = {}
    for name, p_fcst, w_fcst in [
        ("single", p_fcst_single, w_fcst_single),
        ("ensemble", p_fcst_ens_mean, w_fcst_ens_mean),
    ]:
        a_sched, residual = dispatch_chunked(prices_real, p_fcst, w_real,
                                              w_fcst, b_E, b_P)
        arb_rev = float(np.sum(prices_real * a_sched))
        wind_da_rev = float(np.sum(prices_real * w_fcst))
        imb_abs = float(np.sum(np.abs(residual)))
        out[name] = {"arb_rev": arb_rev, "wind_da_rev": wind_da_rev,
                     "imb_abs": imb_abs}
    return out


def run_year(year: int):
    print(f"DK1 {year}: load data")
    df_p = load_dk_year(year, area="DK1")
    df_r = load_dk_resources(year, area="DK1")
    # align on hour
    df = pd.merge(df_p, df_r, on="timestamp_utc", how="inner")
    p = df["da_eur_per_mwh"].to_numpy()
    w_raw = df["wind_mwh"].to_numpy()
    # Normalize wind to WIND_SCALE_MW peak (HPP-style: wind > battery)
    w = (w_raw / w_raw.max()) * WIND_SCALE_MW
    print(f"  N={len(df)} h, p mean={p.mean():.1f}, "
          f"w peak (norm)={w.max():.2f} MW, w mean={w.mean():.2f} MW")

    # Persistence ensemble at K4 lags
    max_lag = max(LAGS_K4)
    p_real, p_F = build_persistence_forecasts(p, LAGS_K4)
    w_real, w_F = build_persistence_forecasts(w, LAGS_K4)
    p_single = p_F[0]
    p_ens = p_F.mean(axis=0)
    w_single = w_F[0]
    w_ens = w_F.mean(axis=0)
    # Forecast error stats for diagnostic
    print(f"  |p_err| single mean={np.abs(p_real - p_single).mean():.1f}, "
          f"ens={np.abs(p_real - p_ens).mean():.1f}")
    print(f"  |w_err| single mean={np.abs(w_real - w_single).mean():.4f}, "
          f"ens={np.abs(w_real - w_ens).mean():.4f}")

    rows = []
    for b_E in B_E_GRID:
        t0 = time.time()
        cell = run_cell(p_real, p_single, p_ens, w_real, w_single, w_ens,
                        b_E=b_E, b_P=B_P)
        for lam in LAMBDA_GRID:
            for pol in ("single", "ensemble"):
                c = cell[pol]
                op = c["arb_rev"] + c["wind_da_rev"] - lam * c["imb_abs"]
                npv = DISC * op - CAPEX_E * b_E - CAPEX_P * B_P
                rows.append({
                    "b_E": b_E, "lambda": lam, "policy": pol,
                    "arb_rev": c["arb_rev"],
                    "wind_da_rev": c["wind_da_rev"],
                    "imb_abs": c["imb_abs"],
                    "imb_cost": lam * c["imb_abs"],
                    "op_profit": op, "npv": npv,
                })
        print(f"  b_E={b_E:6.1f}  ({time.time()-t0:.1f}s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rows = run_year(args.year)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "meta": {"year": args.year, "source": "dk1",
                     "B_E": B_E_GRID, "lambda": LAMBDA_GRID, "B_P": B_P,
                     "lags": list(LAGS_K4), "wind_scale_MW": WIND_SCALE_MW,
                     "DISC": DISC, "CAPEX_E": CAPEX_E, "CAPEX_P": CAPEX_P},
            "rows": rows,
        }, f, indent=2)
    print(f"Wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
