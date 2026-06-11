"""README hero figures.

Two compact panels for repo top-of-README:
  fig_readme_npv.png  - NPV vs b_E at lambda=100 EUR/MWh, single vs
                        ensemble, DK1 2022. Shows ensemble peaks at
                        16 MWh; single needs 24 MWh for same NPV.
  fig_readme_soc.png  - SoC traces on DK1 2022 spike week, single vs
                        K=4 ensemble. Shows ensemble dispatches more
                        smoothly; single over-reacts to phantom spikes.

Run from repo root: python sizing/readme_figures.py
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
from dk_loader import load_dk_year, multi_lag_persistence


def figure_npv(out: Path):
    """NPV vs imbalance penalty lambda, optimally-sized battery per policy."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    for ax, year in zip(axes, [2021, 2022, 2023]):
        d = json.load(open(RESULTS / "imbalance" / f"dk1_{year}.json"))
        rows = d["rows"]
        lambdas = sorted({r["lambda"] for r in rows})
        for pol, color, marker, label in [
            ("single", "#1f77b4", "o", "single forecast (lag-24h)"),
            ("ensemble", "#ff7f0e", "s", "ensemble (K=4 multi-lag)"),
        ]:
            best_npv = []
            best_b = []
            for lam in lambdas:
                sub = [r for r in rows
                       if r["policy"] == pol and r["lambda"] == lam]
                best = max(sub, key=lambda r: r["npv"])
                best_npv.append(best["npv"] / 1e6)
                best_b.append(best["b_E"])
            ax.plot(lambdas, best_npv, "-", marker=marker, color=color,
                     lw=2, ms=7, label=label)
        # Shade band where the two policies' argmax-b_E disagree (forecast
        # quality matters for sizing)
        b_single = [
            max([r for r in rows if r["policy"] == "single" and r["lambda"] == lam],
                key=lambda r: r["npv"])["b_E"] for lam in lambdas
        ]
        b_ens = [
            max([r for r in rows if r["policy"] == "ensemble" and r["lambda"] == lam],
                key=lambda r: r["npv"])["b_E"] for lam in lambdas
        ]
        diverge = [b_single[j] != b_ens[j] for j in range(len(lambdas))]
        for j, lam in enumerate(lambdas):
            if diverge[j]:
                ax.axvspan(lam * 0.7, lam * 1.4, color="grey", alpha=0.12,
                            lw=0)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel(r"imbalance penalty $\lambda$ (€/MWh)", fontsize=10)
        if year == 2021:
            ax.set_ylabel("NPV at optimal $b_E$ (M€, 15-yr, 7%)",
                           fontsize=10)
            ax.legend(loc="lower left", fontsize=9)
        ax.set_title(f"DK1 {year}", fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "NPV vs imbalance penalty for energy promised but not delivered. "
        "Shaded $\\lambda$ bands: single and ensemble pick different $b_E^*$.",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_penalty(out: Path):
    """One-message number line: real DK1 penalties vs the break-point."""
    lam_eff = {2021: 10.9, 2022: 27.7, 2023: 14.9}  # two-price, paper_real_imbalance
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axvspan(50, 100, color="grey", alpha=0.3)
    ax.text(75, 0.78, "break-point zone:\nforecast quality starts\ndriving battery size",
            ha="center", va="center", fontsize=9, color="0.25")
    ax.axvspan(92, 123, color="#d62728", alpha=0.18)
    ax.text(107.5, 0.28, "DK1 after\nMar-2025\nreforms", ha="center",
            va="center", fontsize=9, color="#a02020")
    offsets = {2021: (-10, 14), 2022: (0, 14), 2023: (10, 14)}
    for y, x in lam_eff.items():
        ax.plot([x], [0.5], "o", color="#1f77b4", ms=11, zorder=3)
        dx, dy = offsets[y]
        ax.annotate(str(y), (x, 0.5), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=10,
                    color="#1f77b4")
    ax.text(17, 0.22, "what Danish wind+battery plants\nactually paid, 2021–2023",
            ha="center", va="center", fontsize=9, color="#1f77b4")
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("penalty for energy promised but not delivered (€/MWh)",
                  fontsize=10)
    ax.set_title("Below the zone: cheap-forecast sizing is safe. "
                 "The 2025 reforms ended that.", fontsize=11)
    ax.spines[["left", "right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def year_revenue_stats(b_E: float = 16.0, b_P: float = 1.0,
                       year: int = 2022, chunk: int = 24 * 7 * 8):
    """Full-year arbitrage revenue: oracle vs ensemble vs single.

    Plan on {realized | ensemble | single}, always evaluated on realized
    prices. 'Plan on realized' = perfect foresight (note: an ensemble of
    the truth IS the truth, so 'robust plan + perfectly known future'
    coincides with this oracle).
    """
    realized, forecasts_K = multi_lag_persistence(year, area="DK1")
    plans = {"oracle": realized,
             "ensemble": forecasts_K.mean(axis=0),
             "single": forecasts_K[0]}
    rev = {}
    for name, plan in plans.items():
        total, soc = 0.0, b_E / 2
        for s in range(0, len(realized), chunk):
            e = min(s + chunk, len(realized))
            a = lp_linear_actions(plan[s:e], b_E, b_P, soc, mu=0.0)
            total += float(np.sum(realized[s:e] * a))
            soc = max(0.0, min(b_E, soc - float(np.sum(a))))
        rev[name] = total
    return rev


def figure_soc(out: Path):
    b_E, b_P = 16.0, 1.0
    realized, forecasts_K = multi_lag_persistence(2022, area="DK1")
    # Find spike week
    spike_idx = int(realized.argmax())
    half = 24 * 3
    s = max(0, spike_idx - half)
    e = min(len(realized), spike_idx + half + 24)
    r_slice = realized[s:e]
    f_single = forecasts_K[0, s:e]   # lag-24h
    f_ens = forecasts_K.mean(axis=0)[s:e]
    hours = np.arange(len(r_slice))

    soc0 = b_E / 2
    a_oracle = lp_linear_actions(r_slice, b_E, b_P, soc0, mu=0.0)
    a_single = lp_linear_actions(f_single, b_E, b_P, soc0, mu=0.0)
    a_ens = lp_linear_actions(f_ens, b_E, b_P, soc0, mu=0.0)

    def walk(actions):
        soc = [soc0]
        for a in actions:
            soc.append(max(0.0, min(b_E, soc[-1] - a)))
        return np.array(soc)

    def week_rev(actions):
        return float(np.sum(r_slice * actions))

    yr = year_revenue_stats(b_E=b_E, b_P=b_P, year=2022)
    print("DK1 2022 full-year arbitrage revenue (b_E=16):")
    for k, v in yr.items():
        print(f"  {k:9s} €{v:,.0f}  ({v / yr['oracle'] * 100:.0f}% of oracle)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.4),
                                     sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    ax1.plot(hours, r_slice, color="black", lw=1.2, label="realized DA price")
    ax1.set_ylabel("€/MWh", fontsize=10)
    ax1.set_title("Same battery, same crisis week, three information levels "
                  "(DK1 2022)", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    for actions, color, ls, label in [
        (a_oracle, "0.45", "--",
         f"perfect foresight (upper bound)  week €{week_rev(a_oracle)/1e3:.1f}k"),
        (a_single, "#1f77b4", "-",
         f"cheap forecast  week €{week_rev(a_single)/1e3:.1f}k"),
        (a_ens, "#ff7f0e", "-",
         f"ensemble forecast  week €{week_rev(a_ens)/1e3:.1f}k"),
    ]:
        ax2.plot(np.arange(len(actions) + 1), walk(actions), color=color,
                 ls=ls, lw=1.8, label=label)
    ax2.axhline(b_E, color="grey", lw=0.5, ls="--")
    ax2.axhline(0, color="grey", lw=0.5, ls="--")
    ax2.set_xlabel("hour of week", fontsize=10)
    ax2.set_ylabel(f"state of charge (MWh, {int(b_E)} MWh battery)",
                   fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_npv(FIGURES / "fig_readme_npv.png")
    figure_penalty(FIGURES / "fig_readme_penalty.png")
    figure_soc(FIGURES / "fig_readme_soc.png")
