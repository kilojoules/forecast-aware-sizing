"""Why more revenue doesn't mean a bigger battery: sizing is a MARGINAL
decision.

Panel A: lifetime NPV vs b_E for single vs ensemble dispatch (DK1 2022).
Better forecasts lift the whole curve (more money) but the peak stays at
the same b_E* -- the revenue is a level shift, not a tilt, and adding a
constant to a curve does not move its argmax.

Panel B: the reason. Marginal revenue of the next MWh block vs b_E,
against the flat capex line (100 k EUR/MWh). Optimal size is where
marginal revenue crosses capex. Better forecasts lift marginal revenue
on the inframarginal MWh but not enough at the margin to move the
crossing -- so the size is unchanged. The extra money accrues to
capacity you would build anyway.

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python sizing/fig_marginal.py
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

DISC = sum(1.07 ** -y for y in range(15))
CE, CP = 100_000.0, 75_000.0
YEAR = 2022
GRID = [0.5, 1, 2, 4, 8, 16, 24, 32, 48]


def main(out: Path):
    d = json.load(open(RESULTS / "main" / f"paper_{YEAR}.json"))["by_year"][str(YEAR)]
    R = {"single": {r["b_E"]: r["R"] for r in d["linear_single"]},
         "ensemble": {r["b_E"]: r["R"] for r in d["linear_ensemble"]}}
    b = np.array(GRID)
    npv = {p: np.array([R[p][x] * DISC - CE * x - CP for x in GRID]) / 1e6
           for p in R}
    bstar = {p: GRID[int(npv[p].argmax())] for p in R}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    C = {"single": "#1f77b4", "ensemble": "#ff7f0e"}
    LAB = {"single": "cheap (single) forecast", "ensemble": "better (ensemble) forecast"}

    # Panel A: NPV vs b_E, peaks aligned
    for p in ("single", "ensemble"):
        axA.plot(b, npv[p], "-o", color=C[p], lw=2, ms=5, label=LAB[p])
        i = GRID.index(bstar[p])
        axA.plot([bstar[p]], [npv[p][i]], "*", color=C[p], ms=16, zorder=5)
    axA.axvline(bstar["ensemble"], color="0.5", ls="--", lw=1)
    # vertical "more money, same size" arrow at the argmax
    i = GRID.index(bstar["ensemble"])
    axA.annotate("", xy=(bstar["ensemble"], npv["ensemble"][i]),
                 xytext=(bstar["ensemble"], npv["single"][i]),
                 arrowprops=dict(arrowstyle="<->", color="green", lw=1.8))
    axA.annotate("better forecast:\nmore money, same size",
                 xy=(bstar["ensemble"], (npv["ensemble"][i] + npv["single"][i]) / 2),
                 xytext=(bstar["ensemble"] * 2.2, (npv["ensemble"][i] + npv["single"][i]) / 2 - 0.55),
                 color="green", fontsize=9, va="center",
                 arrowprops=dict(arrowstyle="->", color="green", lw=1))
    axA.text(bstar["ensemble"], axA.get_ylim()[0] + 0.05,
             f"same $b_E^*$={bstar['ensemble']:g} MWh", ha="center", fontsize=9,
             color="0.3")
    axA.set_xscale("log")
    axA.set_xticks(GRID); axA.set_xticklabels([f"{x:g}" for x in GRID], fontsize=8)
    axA.set_xlabel("battery size $b_E$ (MWh)", fontsize=10)
    axA.set_ylabel("lifetime NPV (M€, 15y/7%)", fontsize=10)
    axA.set_title("A. better forecasts lift the peak UP, not sideways\n"
                  "(revenue is a level shift; the argmax doesn't move)", fontsize=10)
    axA.legend(loc="upper left", fontsize=9)
    axA.grid(alpha=0.3)

    # Panel B: marginal revenue of the next MWh vs capex
    mid = [(GRID[i] + GRID[i - 1]) / 2 for i in range(1, len(GRID))]
    for p in ("single", "ensemble"):
        marg = [(R[p][GRID[i]] - R[p][GRID[i - 1]]) / (GRID[i] - GRID[i - 1]) * DISC / 1e3
                for i in range(1, len(GRID))]
        axB.plot(mid, marg, "-o", color=C[p], lw=2, ms=4, label=LAB[p])
    axB.axhline(CE / 1e3, color="black", lw=1.6, ls="-")
    axB.text(GRID[-1] * 0.5, CE / 1e3 + 6, "capex = 100 k€/MWh", fontsize=9)
    axB.axvline(bstar["ensemble"], color="0.5", ls="--", lw=1)
    axB.text(bstar["ensemble"], axB.get_ylim()[0] if False else 5,
             f"$b_E^*$≈{bstar['ensemble']:g}", ha="center", fontsize=9, color="0.3")
    axB.set_xscale("log")
    axB.set_xticks([1, 2, 4, 8, 16, 32]); axB.set_xticklabels([1, 2, 4, 8, 16, 32], fontsize=8)
    axB.set_xlabel("battery size $b_E$ (MWh)", fontsize=10)
    axB.set_ylabel("marginal revenue of the next MWh (k€/MWh, 15y)", fontsize=10)
    axB.set_title("B. size = where the marginal MWh meets capex\n"
                  "(better forecasts don't lift it enough at the margin to move the crossing)",
                  fontsize=10)
    axB.legend(loc="upper right", fontsize=9)
    axB.grid(alpha=0.3)

    fig.suptitle("More revenue justifies BUILDING the battery (and more of them), "
                 "not a BIGGER one — sizing is set by the marginal MWh",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}  (argmax single={bstar['single']} ensemble={bstar['ensemble']})")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    main(FIGURES / "fig_marginal.png")
