"""Paper figure: charge/discharge schedules under the 4 policies on a
high-volatility DK1 2022 week.

Picks the week containing the year's max price spike. Runs each policy
on that 7-day slice (using the multi-lag persistence forecast set), plots:

  Panel 1 (top):  realized price + 4 forecast lags + ensemble mean
  Panel 2:        SoC trajectory under each of the 4 policies
  Panel 3:        net dispatch (positive=discharge, negative=charge) under each policy

The plot makes the intuition concrete: ensemble dispatch smooths
phantom-spike reactions; deterministic single-forecast over-cycles.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
import pandas as pd

from arbitrage_agents import lp_linear_actions, qp_quadratic_actions
from dk_loader import load_dk_resources, load_dk_year, multi_lag_persistence
from env import PriceEnv


B_E = 8.0   # NPV-optimal capacity for 2022
B_P = 1.0
ALPHA = 0.005
MU = 5.0


def find_spike_week(year: int) -> tuple[int, int]:
    """Return (start_hour, end_hour) for the 7-day window containing
    the year's price maximum, after accounting for the 336h dropped-prefix
    that multi_lag_persistence uses."""
    df = load_dk_year(year)
    da = df["da_eur_per_mwh"].to_numpy()
    # Shift so we operate in the same index space as multi_lag_persistence
    # (drops first 336 h)
    after_drop = da[336:]
    # find argmax
    spike_idx = int(after_drop.argmax())
    # 7-day window centered on spike
    half = 24 * 3
    s = max(0, spike_idx - half)
    e = min(len(after_drop), spike_idx + half + 24)
    return s, e


def run_policy_on_slice(realized_slice: np.ndarray, forecast_slice: np.ndarray,
                         cost: str, alpha_or_mu: float) -> dict:
    env = PriceEnv(B_E, B_P, prices=realized_slice, soc0=B_E / 2,
                   alpha=alpha_or_mu if cost == "quadratic" else 0.0,
                   mu=alpha_or_mu if cost == "linear" else 0.0,
                   reward_mode=cost)
    env.reset()
    if cost == "linear":
        a = lp_linear_actions(forecast_slice, B_E, B_P, env.soc0, mu=alpha_or_mu)
    else:
        a = qp_quadratic_actions(forecast_slice, B_E, B_P, env.soc0, alpha=alpha_or_mu)
    for ai in a:
        env.step(ai)
    return {"soc": np.asarray(env.soc_log), "actions": np.asarray(env.action_log)}


def main():
    year = 2022
    realized, forecasts_K = multi_lag_persistence(year, lags_hours=(24, 48, 168, 336))
    df = load_dk_year(year)
    timestamps = pd.to_datetime(df["timestamp_utc"].iloc[336:].reset_index(drop=True))

    s, e = find_spike_week(year)
    print(f"Spike window: hours {s}..{e} ({e-s} h)")
    realized_slice = realized[s:e]
    forecasts_slice = forecasts_K[:, s:e]
    forecast_single = forecasts_slice[0]    # 24h lag
    forecast_mean = forecasts_slice.mean(axis=0)
    times = timestamps.iloc[s:e].reset_index(drop=True)

    print(f"Slice price range: {realized_slice.min():.0f}..{realized_slice.max():.0f} EUR/MWh")

    # Run all 4 policies on the slice
    results = {
        "LP-linear single":   run_policy_on_slice(realized_slice, forecast_single, "linear", MU),
        "LP-linear K=4 ens":  run_policy_on_slice(realized_slice, forecast_mean, "linear", MU),
        "QP-quadratic single": run_policy_on_slice(realized_slice, forecast_single, "quadratic", ALPHA),
        "QP-quadratic K=4 ens": run_policy_on_slice(realized_slice, forecast_mean, "quadratic", ALPHA),
    }
    cmap = {
        "LP-linear single":     "#aaaaaa",
        "LP-linear K=4 ens":    "#117733",
        "QP-quadratic single":  "#cc6677",
        "QP-quadratic K=4 ens": "#4477aa",
    }

    # Load resource data + slice
    res_df = load_dk_resources(year)
    res_df["timestamp_utc"] = pd.to_datetime(res_df["timestamp_utc"], utc=True)
    df_main = load_dk_year(year)
    df_main["timestamp_utc"] = pd.to_datetime(df_main["timestamp_utc"], utc=True)
    res_df_aligned = res_df.set_index("timestamp_utc").reindex(
        df_main["timestamp_utc"].iloc[336:].reset_index(drop=True)).reset_index(drop=True)
    wind_slice = res_df_aligned["wind_mwh"].to_numpy()[s:e]
    solar_slice = res_df_aligned["solar_mwh"].to_numpy()[s:e]
    load_slice = res_df_aligned["load_mwh"].to_numpy()[s:e]
    residual_slice = load_slice - wind_slice - solar_slice  # what thermal+import covers

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                              gridspec_kw={"height_ratios": [1.2, 1.2, 1, 1]})

    # Panel 1: prices + forecasts
    ax = axes[0]
    ax.plot(times, realized_slice, color="black", lw=1.6, label="Realised")
    lag_styles = [(":", "#dd7777"), ("--", "#dd9988"), ("-.", "#dd99aa"), ("-", "#cc77bb")]
    lag_hours = [24, 48, 168, 336]
    for k, (ls, c) in enumerate(lag_styles):
        ax.plot(times, forecasts_slice[k], color=c, lw=0.8, ls=ls,
                 label=f"Persistence(t-{lag_hours[k]}h)", alpha=0.6)
    ax.plot(times, forecast_mean, color="#4477aa", lw=1.4, ls="--",
             label="K=4 ensemble mean")
    ax.set_ylabel("DA price (EUR/MWh)")
    ax.set_title(f"DK1 {year}: spike-week + persistence forecasts at lags 24/48/168/336 h")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 2: DK1 wind / solar / load (the physical context the price reflects)
    ax = axes[1]
    ax.fill_between(times, 0, wind_slice, color="#117733", alpha=0.5, label="Wind production")
    ax.fill_between(times, wind_slice, wind_slice + solar_slice, color="#ddcc77",
                     alpha=0.6, label="Solar production")
    ax.plot(times, load_slice, color="black", lw=1.4, label="Gross load")
    ax.plot(times, residual_slice, color="#882255", lw=1.0, ls="--",
             label="Residual load (load - wind - solar)")
    ax.set_ylabel("DK1 system MWh")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("Underlying physical context: DK1 wind, solar, load. Battery dispatch is on the price signal these drive.",
                  fontsize=9, loc="left")

    # Panel 3: SoC trajectories
    ax = axes[2]
    for label, res in results.items():
        soc = res["soc"]
        # soc is len T+1; align to times by taking [1:]
        ax.plot(times, soc[1:1 + len(times)], color=cmap[label], lw=1.2, label=label)
    ax.axhline(0, color="black", lw=0.4, ls=":")
    ax.axhline(B_E, color="black", lw=0.4, ls=":")
    ax.set_ylabel("SoC (MWh)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 4: net dispatch
    ax = axes[3]
    for label, res in results.items():
        a = res["actions"]
        ax.plot(times, a, color=cmap[label], lw=1.2, alpha=0.85, label=label)
    ax.axhline(0, color="black", lw=0.4, ls="-")
    ax.axhline(B_P, color="black", lw=0.4, ls=":")
    ax.axhline(-B_P, color="black", lw=0.4, ls=":")
    ax.set_ylabel("Net dispatch (MW)\n(+ = discharge)")
    ax.set_xlabel("Hour")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Charge/discharge schedule at $b_E^*$={B_E} MWh, $b_P$={B_P} MW. "
                  f"Single-forecast policies react to phantom spikes; ensemble smooths.",
                  y=1.005, fontsize=11)
    fig.tight_layout()
    out = "fig_paper_timeseries.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
