"""Pilot (a) day 2: can a forecast-robust policy close the LP-with-forecast gap?

Day 1 (sanity_forecast.py) showed the LP-with-forecast vs LP-oracle gap
grows from ~7% (realistic forecasts) to ~50% (poor forecasts).

Day 2 question: can a simple shrinkage policy beat raw-forecast LP at high noise?

Shrinkage policy:
  forecast_used = w * forecast + (1 - w) * forecast_mean
  Solve LP on shrunken forecast. As w -> 0, ignore forecast and use only mean.
  Sweep w in [0, 1]. Identifies whether trusting forecast less helps under noise.

Also test: a "robust threshold" rule that uses ROLLING REALIZED stats
(no forecast at all). Bench against LP-with-forecast at each noise level.

Headline claim to test: at realistic noise (5-12), can w<1 shrinkage or
the realized-rolling threshold close >2% of the gap?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from sanity_forecast import (
    T, B_E, B_P, SOC0, SEED,
    synth_price, make_forecast,
    lp_dispatch, simulate_with_actions, revenue_realized, D_of,
    policy_lp_oracle, policy_lp_full_with_forecast,
)


def policy_shrunk(realized, forecast, w: float, mu: float = 0.0):
    """LP on a shrunken forecast: w * forecast + (1-w) * mean(forecast)."""
    fc_mean = np.mean(forecast)
    used = w * forecast + (1.0 - w) * fc_mean
    P_chg, P_dis = lp_dispatch(used, mu)
    soc, P_chg_e, P_dis_e = simulate_with_actions(P_chg, P_dis)
    return soc, P_chg_e, P_dis_e


def policy_realized_rolling_threshold(realized, low_pct: float = 25,
                                       high_pct: float = 75, window: int = 48):
    """Use a 48-h rolling window of REALIZED prices to set thresholds.
    No forecast used. (Cheats slightly: at time t we know prices [t-window..t-1]
    -- a 'past prices only' percentile.)
    """
    soc = SOC0
    socs = [soc]
    P_chg = np.zeros(T)
    P_dis = np.zeros(T)
    for t in range(T):
        start = max(0, t - window)
        if start == t:
            # not enough history -- be conservative: do nothing
            socs.append(soc)
            continue
        hist = realized[start:t]
        p_low = np.percentile(hist, low_pct)
        p_high = np.percentile(hist, high_pct)
        if realized[t] < p_low and soc < B_E:
            chg = min(B_P, B_E - soc)
            P_chg[t] = chg
            soc += chg
        elif realized[t] > p_high and soc > 0:
            dis = min(B_P, soc)
            P_dis[t] = dis
            soc -= dis
        socs.append(soc)
    return np.array(socs), P_chg, P_dis


def main():
    print("Pilot (a) day 2: forecast-robust policies vs LP-with-forecast")
    print(f"T={T}h  B_E={B_E}  B_P={B_P}  seed={SEED}\n")

    realized = synth_price(T, seed=SEED)
    MU = 0.0

    # Oracle baseline
    soc_or, chg_or, dis_or = policy_lp_oracle(realized, MU)
    R_oracle = revenue_realized(realized, chg_or, dis_or)
    print(f"Oracle R={R_oracle:.1f}\n")

    # Realized-rolling threshold (forecast-free)
    soc_thr, chg_thr, dis_thr = policy_realized_rolling_threshold(realized)
    R_thr = revenue_realized(realized, chg_thr, dis_thr)
    print(f"Realized-rolling threshold (forecast-free): R={R_thr:.1f} "
          f"({(R_oracle - R_thr)/R_oracle*100:+.1f}% gap)\n")

    # Sweep noise + shrinkage w
    noise_levels = [3, 5, 8, 12, 18, 25]
    ws = [0.0, 0.25, 0.5, 0.75, 1.0]
    fc_seeds = list(range(8))  # 8 forecast realizations per noise level

    matrix = np.zeros((len(noise_levels), len(ws)))
    print("Shrinkage sweep (revenue averaged over 8 forecast seeds):")
    print(f"{'noise':>6} | " + " | ".join(f"w={w:4.2f}" for w in ws))
    print("-" * 60)
    for i, noise_std in enumerate(noise_levels):
        row = []
        for j, w in enumerate(ws):
            Rs = []
            for fc_seed in fc_seeds:
                forecast = make_forecast(realized, noise_std, seed=fc_seed)
                soc_s, chg_s, dis_s = policy_shrunk(realized, forecast, w, MU)
                Rs.append(revenue_realized(realized, chg_s, dis_s))
            matrix[i, j] = np.mean(Rs)
            row.append(np.mean(Rs))
        print(f"{noise_std:6.1f} | " + " | ".join(f"{r:7.1f}" for r in row))

    # Find best w per noise level vs raw LP (w=1)
    print(f"\nBest-w lift over raw LP-with-forecast (w=1):")
    print(f"{'noise':>6} | best w | R_best - R_w1 | rel %")
    print("-" * 50)
    for i, noise_std in enumerate(noise_levels):
        best_j = matrix[i].argmax()
        best_w = ws[best_j]
        R_raw = matrix[i, -1]   # w=1.0 column
        R_best = matrix[i, best_j]
        delta = R_best - R_raw
        rel = delta / R_raw * 100 if R_raw > 0 else 0.0
        marker = "  <-- WIN" if rel > 2 and best_w != 1.0 else ""
        print(f"{noise_std:6.1f} | {best_w:6.2f} | {delta:13.1f} | {rel:+5.1f}%{marker}")

    # ---- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for j, w in enumerate(ws):
        ax.plot(noise_levels, matrix[:, j], "o-",
                label=f"w={w:.2f} (shrink-to-mean)")
    ax.axhline(R_oracle, color="black", lw=1, ls="--", label=f"Oracle={R_oracle:.0f}")
    ax.axhline(R_thr, color="grey", lw=1, ls=":", label=f"Realized-rolling thr={R_thr:.0f}")
    ax.set_xlabel("Forecast AR(1) noise std")
    ax.set_ylabel("Realized revenue")
    ax.set_title("Shrinkage policy vs forecast noise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    pct_oracle = (matrix - matrix[:, -1:]) / matrix[:, -1:] * 100
    for j, w in enumerate(ws[:-1]):  # skip w=1.0 since it's the baseline
        ax.plot(noise_levels, pct_oracle[:, j], "o-",
                label=f"w={w:.2f} vs raw LP")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Forecast AR(1) noise std")
    ax.set_ylabel("Lift over raw LP-with-forecast (%)")
    ax.set_title("Shrinkage lift over raw LP")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Pilot (a) day 2: can shrinkage close the LP-with-forecast gap?",
        y=1.02)
    fig.tight_layout()
    out = "fig_forecast_robust.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")

    # Verdict
    max_lift = pct_oracle[:, :-1].max() if len(ws) > 1 else 0.0
    print(f"\n{'='*70}")
    print(f"Max shrinkage lift over raw LP-with-forecast: {max_lift:.1f}%")
    if max_lift > 5:
        print("Shrinkage closes >5% of gap. Forecast-robust policy has real value.")
        print("-> day 3+: replace shrinkage with learned RL policy. Continue rework.")
    elif max_lift > 2:
        print(f"Shrinkage helps by {max_lift:.1f}%. Marginal but real.")
    else:
        print(f"Shrinkage barely helps ({max_lift:.1f}%). LP-with-raw-forecast is fine.")
        print("-> KILL pilot (a). Forecast-uncertainty angle is dead at this scale.")


if __name__ == "__main__":
    main()
