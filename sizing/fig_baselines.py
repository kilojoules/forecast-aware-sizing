"""Adversarial baselines in battery-incremental NPV space.

The five co-located strategies (perfect / ensemble / single / adv_design
pessimist / adv_stress adversary) plotted as a categorical scatter in
(optimal size, battery-incremental NPV). 'Incremental' = NPV at the
strategy's b_E* minus its no-battery NPV, which strips the ~95 M EUR
wind DA revenue that otherwise dwarfs the 6-10 M battery span and hides
the sizing gap. Shown at lambda=200 (wind/battery ratio 20, DK1 2022)
where the decision-attitude spread is widest; a note marks that at
lambda=0 all five collapse to the same point.

Message: the pessimist (adv_design), which designs around the worst-case
forecast, sizes SMALLEST and earns LEAST -- worst-case design is a
mistake, not caution. Decision attitude spans the size axis far wider
than the single-vs-ensemble forecast gap.

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python sizing/fig_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SIZING_DIR = Path(__file__).resolve().parent
RESULTS = SIZING_DIR.parent / "results"
FIGURES = SIZING_DIR.parent / "paper" / "figures"

import matplotlib.pyplot as plt
import numpy as np

STRAT = {
    "perfect":    ("perfect foresight", "#08519c", "*", 260),
    "ensemble":   ("ensemble (honest)", "#ff7f0e", "s", 130),
    "single":     ("single (honest)", "#1f77b4", "o", 130),
    "adv_design": ("pessimist\n(worst-case design)", "#d62728", "v", 150),
    "adv_stress": ("adversary\n(max imbalance)", "#7f7f7f", "X", 130),
}
YEAR, RATIO, LAM = 2022, 20, 200.0


def best_at(rows, pol, lam):
    sub = [r for r in rows if r["policy"] == pol and r["lambda"] == lam]
    best = max(sub, key=lambda r: r["npv"])
    return best["b_E"], best["npv"]


def main(out: Path):
    rows = json.load(open(RESULTS / "imbalance"
                          / f"baselines_dk1_{YEAR}_w{RATIO}.json"))["rows"]
    order = ["perfect", "ensemble", "single", "adv_design", "adv_stress"]
    perf_npv = best_at(rows, "perfect", LAM)[1]
    sizes, gaps, labels, colors = [], [], [], []
    for pol in order:
        b, npv = best_at(rows, pol, LAM)
        sizes.append(b); gaps.append((npv - perf_npv) / 1e6)
        labels.append(STRAT[pol][0]); colors.append(STRAT[pol][1])

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(order))
    ax.bar(x, gaps, color=colors, alpha=0.85, zorder=3)
    for xi, (g, b) in enumerate(zip(gaps, sizes)):
        ax.annotate(f"$b_E^*$={b:g} MWh", (xi, g), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=9,
                    fontweight="bold", color="0.15")
    ax.axhline(0, color="#08519c", lw=1.5)
    ax.text(-0.45, 0, "perfect\nforesight", fontsize=8, color="#08519c",
            va="center", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("NPV shortfall vs perfect foresight (M€)", fontsize=10)
    ax.set_title(f"Adversarial baselines — DK1 {YEAR}, {RATIO}:1 wind/battery, "
                 f"λ={LAM:g} €/MWh\n"
                 "the pessimist (worst-case design) sizes SMALLEST (24 MWh) "
                 "yet loses the MOST of the honest policies", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate("worst-case design is a mistake, not caution: a delivery-floor "
                "bid makes\nimbalance chronic and one-signed, which no battery "
                "can buffer.\nAt λ=0 all five collapse to one point (merchant "
                "invariance).",
                xy=(0.98, 0.04), xycoords="axes fraction", ha="right",
                fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    for pol, b, g in zip(order, sizes, gaps):
        print(f"  {pol:11s} b*={b:g}  gap_vs_perfect={g:+.2f}M")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    main(FIGURES / "fig_baselines.png")
