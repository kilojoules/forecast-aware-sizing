"""Operations-to-sizing lead triptych under a forecast-uncertainty dial.

Parameterized forecast:
    forecast_gamma(t) = realized(t) + gamma * (persistence(t) - realized(t))
  gamma=0 -> perfect foresight; gamma=1 -> persistence skill (the honest
  operational baseline); gamma>1 -> worse than persistence (stress).

Three panels share the gamma axis:
  A. SoC traces over an illustrative DK1-2022 spike week (b_E = b_E* = 8 MWh,
     the optimum) at gamma=0/1/2. The degradation is MISTIMING, not
     over-cycling: cycling volume is ~constant in gamma, but the
     action-price correlation collapses, so the battery sells into the
     wrong hours.
  B. Money (NPV at b_E=8) vs gamma, raw EUR with a zero line, all six
     market-years. Falls monotonically; turns loss-making past gamma~2.5.
  C. Optimal capacity b_E* vs gamma, with the within-1%-of-optimum
     tolerance band. FLAT only on the realizable skill range gamma in
     [0,1] (and ercot_2021 already moves at gamma=1); collapses toward the
     grid floor for gamma>1. Not a flat line -- the honest claim is a
     bounded plateau.

Reads left->right: dispatch degrades by mistiming -> money falls steeply
-> optimal size is flat across the realistic skill range, then shrinks
once forecasts are worse than persistence.

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python \
         sizing/fig_ops_uncertainty.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SIZING_DIR = Path(__file__).resolve().parent
RESULTS = SIZING_DIR.parent / "results"
FIGURES = SIZING_DIR.parent / "paper" / "figures"
sys.path.insert(0, str(SIZING_DIR))

import matplotlib.pyplot as plt
import numpy as np

from arbitrage_agents import lp_linear_actions
from dk_loader import multi_lag_persistence

B_E_STAR = 8.0          # b_E* for DK1 2022 (the optimum, not the 2x-oversized 16)
B_P = 1.0
B_E_GRID = [0.5, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
DISC = sum(1.07 ** -y for y in range(15))
CE, CP = 100_000.0, 75_000.0
WEEK_GAMMAS = [0.0, 1.0, 2.0]
REGIME_COLOR = {"dk1_2021": "#1f77b4", "dk1_2022": "#d62728",
                "dk1_2023": "#2ca02c", "ercot_2021": "#9467bd",
                "ercot_2022": "#8c564b", "ercot_2023": "#e377c2"}


def forecast_gamma(realized, persistence, gamma):
    return realized + gamma * (persistence - realized)


def walk(actions, soc0, b_E):
    soc = [soc0]
    for a in actions:
        soc.append(max(0.0, min(b_E, soc[-1] - a)))
    return np.array(soc)


def panel_A(ax, axp):
    realized, F = multi_lag_persistence(2022, area="DK1")
    persistence = F[0]
    spike = int(realized.argmax())
    half = 24 * 2  # +-2 days: the window where dispatch differences are real
    s, e = max(0, spike - half), min(len(realized), spike + half + 12)
    r, pers = realized[s:e], persistence[s:e]
    soc0 = B_E_STAR / 2
    axp.plot(np.arange(len(r)), r, color="0.6", lw=0.9, alpha=0.7, zorder=0)
    axp.set_ylabel("price €/MWh", fontsize=8, color="0.5")
    axp.tick_params(axis="y", labelsize=7, colors="0.5")
    styles = {0.0: ("#00204d", 2.4, "-", "γ=0  perfect foresight"),
              1.0: ("#bb3754", 2.4, "--", "γ=1  persistence (honest baseline)"),
              2.0: ("#f9c932", 2.0, "-", "γ=2  worse than persistence")}
    for g in WEEK_GAMMAS:
        f = forecast_gamma(r, pers, g)
        a = lp_linear_actions(f, B_E_STAR, B_P, soc0, mu=0.0)
        corr = np.corrcoef(a, r)[0, 1]
        c, lw, ls, lab = styles[g]
        ax.plot(np.arange(len(a) + 1), walk(a, soc0, B_E_STAR), color=c,
                lw=lw, ls=ls, label=f"{lab}  (corr$_{{a,p}}$={corr:+.2f})",
                zorder=3)
    ax.axhline(B_E_STAR, color="grey", lw=0.5, ls=":")
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("hour of week", fontsize=9)
    ax.set_ylabel(f"state of charge (MWh, on $b_E^*$={int(B_E_STAR)} MWh)",
                  fontsize=9)
    ax.set_title("A. dispatch degrades by mis-timing\n"
                 "(DK1 2022 spike week; cycling volume ~constant, "
                 "timing corr collapses)", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.2)


def panel_B(ax, gs):
    gammas = gs["meta"]["gammas"]
    i8 = B_E_GRID.index(8)
    for reg, rows in gs["regimes"].items():
        npv8 = [r["npv_curve"][i8] / 1e6 for r in rows]
        ax.plot(gammas, npv8, "-o", ms=2.5, lw=1.3,
                color=REGIME_COLOR[reg], label=reg.replace("_", " "))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(1.0, color="grey", lw=1, ls="--")
    ax.text(1.02, ax.get_ylim()[1] * 0.92, "persistence\nskill", fontsize=7,
            color="0.3")
    ax.axvspan(1.0, 3.0, color="0.85", alpha=0.5, zorder=0)
    ax.set_xlabel("forecast-error dial  γ", fontsize=9)
    ax.set_ylabel("NPV at $b_E$=8 MWh  (M€, 15y/7%)", fontsize=9)
    ax.set_title("B. money falls with forecast error\n"
                 "(turns loss-making past γ≈2.5; γ>1 = stress)", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=6.3, ncol=2)


def panel_C(ax, gs):
    gammas = gs["meta"]["gammas"]
    grid = np.array(B_E_GRID)
    for reg, rows in gs["regimes"].items():
        bstar = [r["b_E_star"] for r in rows]
        # within-1%-of-optimum tolerance band
        lo, hi = [], []
        for r in rows:
            npvc = np.array(r["npv_curve"])
            best = npvc.max()
            tol = abs(best) * 0.01
            ok = grid[npvc >= best - tol]
            lo.append(ok.min()); hi.append(ok.max())
        ax.fill_between(gammas, lo, hi, color=REGIME_COLOR[reg], alpha=0.10,
                        zorder=0)
        ax.plot(gammas, bstar, "-o", ms=2.5, lw=1.3,
                color=REGIME_COLOR[reg], label=reg.replace("_", " "))
    ax.axvline(1.0, color="grey", lw=1, ls="--")
    ax.axvspan(1.0, 3.0, color="0.85", alpha=0.5, zorder=0)
    ax.text(0.18, 70, "flat on the\nrealizable skill\nrange γ≤1", fontsize=7.5,
            color="0.25")
    ax.annotate("ercot 2021 moves\n8→4 already at γ=1",
                xy=(1.0, 4), xytext=(1.35, 18), fontsize=7, color="#9467bd",
                arrowprops=dict(arrowstyle="->", color="#9467bd", lw=0.8))
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_yticklabels([0.5, 1, 2, 4, 8, 16, 32, 64, 128], fontsize=7)
    ax.set_xlabel("forecast-error dial  γ", fontsize=9)
    ax.set_ylabel("optimal capacity $b_E^*$ (MWh, shaded: within 1% NPV)",
                  fontsize=9)
    ax.set_title("C. optimal SIZE is flat across the skill range\n"
                 "then collapses once forecasts beat-by persistence fails",
                 fontsize=9.5)
    ax.legend(loc="lower left", fontsize=6.3, ncol=2)


def main(out: Path):
    gs = json.load(open(RESULTS / "main" / "gamma_sweep.json"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axp = axes[0].twinx()
    panel_A(axes[0], axp)
    panel_B(axes[1], gs)
    panel_C(axes[2], gs)
    fig.suptitle("Turn up the forecast-uncertainty dial: operations degrade and "
                 "the money falls, but the optimal SIZE holds across the realistic "
                 "forecast-skill range", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    main(FIGURES / "fig_ops_uncertainty.png")
