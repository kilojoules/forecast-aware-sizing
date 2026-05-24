"""Quantile-regression ensemble forecast (replaces multi-lag persistence).

Trains a gradient-boosted quantile regressor on past hourly prices using
calendar features (hour, day-of-week, month) + lagged prices (24h, 48h,
168h, 336h). Outputs K=20 quantile-sample forecasts. Run dispatch under
the K-ensemble.

Per AMEND_04 (committed alongside).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from arbitrage_agents import (lp_linear_actions, qp_ensemble_actions,
                              qp_quadratic_actions, run_actions)
from degradation import cycle_degradation
from dk_loader import load_dk_year
from env import PriceEnv
from ercot_loader import load_ercot_year


B_E_GRID = [0.5, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
B_P = 1.0
ALPHA_QP = 0.005
MU_LP = 5.0
CHUNK_HOURS = 24 * 7 * 8
QUANTILES = np.linspace(0.05, 0.95, 20)  # K=20


def build_features(prices: np.ndarray, timestamps: pd.DatetimeIndex):
    """Calendar features + lagged prices. Drop first 336 h (max lag)."""
    n = len(prices)
    lags = [24, 48, 168, 336]
    max_lag = max(lags)
    valid = n - max_lag
    X = np.zeros((valid, len(lags) + 4))
    for i, lag in enumerate(lags):
        X[:, i] = prices[max_lag - lag : n - lag]
    ts = timestamps[max_lag:]
    hour = ts.dt.hour.values if hasattr(ts, "dt") else ts.hour
    dow = ts.dt.dayofweek.values if hasattr(ts, "dt") else ts.dayofweek
    X[:, len(lags)]     = np.sin(2 * np.pi * hour / 24)
    X[:, len(lags) + 1] = np.cos(2 * np.pi * hour / 24)
    X[:, len(lags) + 2] = np.sin(2 * np.pi * dow / 7)
    X[:, len(lags) + 3] = np.cos(2 * np.pi * dow / 7)
    y = prices[max_lag:]
    return X, y


def fit_quantile_models(X_train, y_train, quantiles=QUANTILES):
    models = {}
    for q in quantiles:
        m = GradientBoostingRegressor(loss="quantile", alpha=float(q),
                                       n_estimators=120, max_depth=4,
                                       learning_rate=0.05, random_state=0)
        m.fit(X_train, y_train)
        models[float(q)] = m
    return models


def predict_quantiles(models, X) -> np.ndarray:
    """Returns shape (K, T) quantile forecasts."""
    out = np.zeros((len(models), len(X)))
    for i, q in enumerate(sorted(models.keys())):
        out[i] = models[q].predict(X)
    return out


def chunks(T, c=CHUNK_HOURS):
    s = 0
    while s < T:
        yield s, min(s + c, T)
        s += c


def safe_D(soc, b):
    try:
        D, _, _ = cycle_degradation(soc, b)
        return D
    except Exception:
        return 0.0


def dispatch(realized, fc, b_E, b_P, cost, m, ensemble=False):
    R_total = 0.0
    soc_full = []
    soc_carry = b_E / 2
    T = realized.shape[0] if not ensemble else realized.shape[0]
    for s, e in chunks(T):
        r_chunk = realized[s:e]
        env = PriceEnv(b_E, b_P, prices=r_chunk, soc0=soc_carry,
                       alpha=m if cost == "quadratic" else 0,
                       mu=m if cost == "linear" else 0,
                       reward_mode=cost)
        if ensemble:
            f = fc[:, s:e]
            if cost == "linear":
                a = lp_linear_actions(f.mean(axis=0), b_E, b_P, soc_carry, mu=m)
            else:
                a = qp_ensemble_actions(f, b_E, b_P, soc_carry, alpha=m)
        else:
            f = fc[s:e]
            if cost == "linear":
                a = lp_linear_actions(f, b_E, b_P, soc_carry, mu=m)
            else:
                a = qp_quadratic_actions(f, b_E, b_P, soc_carry, alpha=m)
        rec = run_actions(env, a)
        R_total += rec["R"]
        soc_full.extend(rec["soc_log"][1:])
        soc_carry = rec["soc_log"][-1]
    soc_full = [b_E / 2] + soc_full
    return R_total, safe_D(np.asarray(soc_full), b_E)


def run_year(source: str, year: int, n_estimators: int = 120) -> dict:
    """Train forecast model on the OTHER two years (out-of-sample),
    apply to target year. Avoids needing year-1 data not in our cache."""
    other_years = [y for y in [2021, 2022, 2023] if y != year]
    if source == "dk1":
        df_train = pd.concat([load_dk_year(y) for y in other_years], ignore_index=True)
        df_test = load_dk_year(year)
        col = "da_eur_per_mwh"
    else:
        df_train = pd.concat([load_ercot_year(y) for y in other_years], ignore_index=True)
        df_test = load_ercot_year(year)
        col = "da_usd_per_mwh"

    train_prices = df_train[col].to_numpy()
    train_ts = pd.to_datetime(df_train["timestamp_utc"]).reset_index(drop=True)
    test_prices = df_test[col].to_numpy()
    test_ts = pd.to_datetime(df_test["timestamp_utc"]).reset_index(drop=True)

    X_train, y_train = build_features(train_prices, train_ts)
    X_test, y_test = build_features(test_prices, test_ts)

    print(f"  fitting {len(QUANTILES)} quantile models...")
    t0 = time.time()
    models = fit_quantile_models(X_train, y_train)
    print(f"  fitted in {time.time()-t0:.0f}s")

    fc_K = predict_quantiles(models, X_test)
    fc_median = fc_K[len(QUANTILES) // 2]   # median = single-forecast baseline

    realized = y_test
    print(f"  median forecast residual std: {(realized - fc_median).std():.1f}")

    out = {}
    for cost, m in [("linear", MU_LP), ("quadratic", ALPHA_QP)]:
        rows_single, rows_ens = [], []
        for b_E in B_E_GRID:
            Rs, Ds = dispatch(realized, fc_median, b_E, B_P, cost, m, ensemble=False)
            Re, De = dispatch(realized, fc_K, b_E, B_P, cost, m, ensemble=True)
            rows_single.append({"b_E": b_E, "R": Rs, "D": Ds})
            rows_ens.append({"b_E": b_E, "R": Re, "D": De})
        out[f"{cost}_single"] = rows_single
        out[f"{cost}_ensemble"] = rows_ens
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["dk1", "ercot"], required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"Quantile-regression ensemble: {args.source} {args.year}")
    t0 = time.time()
    result = {"meta": {"source": args.source, "year": args.year,
                        "K": len(QUANTILES), "ensemble": "quantile_regression"},
              "by_year": {str(args.year): run_year(args.source, args.year)}}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
