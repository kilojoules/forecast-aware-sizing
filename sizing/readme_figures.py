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
    d = json.load(open(RESULTS / "imbalance" / "dk1_2022.json"))
    rows = d["rows"]
    lam = 100.0
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for pol, color, marker, label in [
        ("single", "#1f77b4", "o", "single forecast (lag-24h)"),
        ("ensemble", "#ff7f0e", "s", "ensemble (K=4 multi-lag)"),
    ]:
        sub = sorted(
            [r for r in rows if r["policy"] == pol and r["lambda"] == lam],
            key=lambda r: r["b_E"],
        )
        b = [r["b_E"] for r in sub]
        npv = [r["npv"] / 1e6 for r in sub]
        ax.plot(b, npv, "-", marker=marker, color=color, lw=2, ms=7,
                 label=label)
        ai = int(np.argmax(npv))
        ax.axvline(b[ai], ls=":", color=color, alpha=0.6, lw=1.5)
        ax.annotate(f"$b_E^* = {int(b[ai])}$ MWh\nNPV = {npv[ai]:.1f}M€",
                     xy=(b[ai], npv[ai]),
                     xytext=(10 if pol == "single" else -110,
                              -25 if pol == "single" else 15),
                     textcoords="offset points",
                     fontsize=10, color=color,
                     arrowprops=dict(arrowstyle="-", color=color, alpha=0.6))
    ax.set_xscale("log")
    ax.set_xlabel(r"battery energy capacity $b_E$ (MWh)", fontsize=11)
    ax.set_ylabel("NPV (M€, 15-yr, 7%)", fontsize=11)
    ax.set_title(
        "DK1 2022 + 5-MW wind + 1-MW battery, imbalance penalty $\\lambda = 100$ €/MWh",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


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
    a_single = lp_linear_actions(f_single, b_E, b_P, soc0, mu=0.0)
    a_ens = lp_linear_actions(f_ens, b_E, b_P, soc0, mu=0.0)

    def walk(actions):
        soc = [soc0]
        for a in actions:
            soc.append(max(0.0, min(b_E, soc[-1] - a)))
        return np.array(soc)
    soc_single = walk(a_single)
    soc_ens = walk(a_ens)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5),
                                     sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    ax1.plot(hours, r_slice, color="black", lw=1.2, label="realized DA price")
    ax1.set_ylabel("€/MWh", fontsize=10)
    ax1.set_title("DK1 2022 spike week: single-forecast over-cycles, ensemble smooths",
                   fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.plot(np.arange(len(soc_single)), soc_single, color="#1f77b4",
             lw=1.8, label="single forecast (over-cycles, hits both rails)")
    ax2.plot(np.arange(len(soc_ens)), soc_ens, color="#ff7f0e",
             lw=1.8, label="ensemble (shallower, stays in middle band)")
    ax2.axhline(b_E, color="grey", lw=0.5, ls="--")
    ax2.axhline(0, color="grey", lw=0.5, ls="--")
    ax2.set_xlabel("hour of week", fontsize=10)
    ax2.set_ylabel(f"SoC (MWh, on $b_E = {int(b_E)}$ probe)", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_npv(FIGURES / "fig_readme_npv.png")
    figure_soc(FIGURES / "fig_readme_soc.png")
