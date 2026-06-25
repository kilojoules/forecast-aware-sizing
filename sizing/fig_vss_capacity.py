"""The figure that convinces a sizing-tool builder: operating value rises
with forecast information, but the optimal capacity does not move.

Left: for each market-year, NPV at the fixed optimal capacity under three
information structures -- EEV (point forecast), RP (ensemble mean), WS
(perfect foresight / wait-and-see). The bars climb (EVPI, VSS_op > 0):
better information is worth real money for OPERATION.

Right: the optimal capacity argmax b_E* under the same three structures.
The markers sit on top of each other in 5/6 regimes (VSS-for-capacity = 0);
only ERCOT 2021 (Storm Uri) moves, and only for the clairvoyant oracle.

Message: pay for a stochastic dispatcher to make money, not to size --
the cheap deterministic inner loop already picks the right capacity.

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python \
         sizing/fig_vss_capacity.py
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

LEVELS = ["EEV_single", "RP_ensemble", "WS_oracle"]
LABELS = ["point\nforecast", "ensemble\nmean", "perfect\nforesight"]
LCOLOR = ["#bdbdbd", "#6baed6", "#08519c"]


def main(out: Path):
    ws = json.load(open(RESULTS / "main" / "ws_cvar.json"))["regimes"]
    regimes = list(ws.keys())
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # LEFT: NPV at the common (RP) capacity climbs with information
    x = np.arange(len(regimes))
    w = 0.26
    for j, lvl in enumerate(LEVELS):
        vals = [ws[r]["npv_at_common_b"][lvl] / 1e6 for r in regimes]
        axL.bar(x + (j - 1) * w, vals, w, color=LCOLOR[j], label=LABELS[j].replace("\n", " "))
    axL.set_xticks(x)
    axL.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8)
    axL.set_ylabel("NPV at the optimal capacity (M€)", fontsize=10)
    axL.set_title("Better information is worth real money — for OPERATION\n"
                  "(EVPI 0.08–0.68 M€, VSS$_{op}$ 0.03–0.33 M€ > 0 everywhere)",
                  fontsize=10)
    axL.legend(loc="upper left", fontsize=8, title="dispatch information")
    axL.grid(axis="y", alpha=0.3)

    # RIGHT: optimal capacity argmax does NOT move with information
    for j, lvl in enumerate(LEVELS):
        bstar = [ws[r]["b_star"][lvl] for r in regimes]
        axR.scatter(x, bstar, s=140, color=LCOLOR[j], zorder=3 + j,
                    edgecolor="white", linewidth=1.2, label=LABELS[j].replace("\n", " "))
    axR.set_xticks(x)
    axR.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8)
    axR.set_yscale("log")
    axR.set_yticks([4, 8])
    axR.set_yticklabels([4, 8], fontsize=9)
    axR.set_ylim(2.5, 12)
    axR.set_ylabel("optimal capacity $b_E^*$ (MWh)", fontsize=10)
    axR.set_title("...but the optimal SIZE does not move\n"
                  "(markers coincide in 5/6 regimes; VSS-for-capacity = 0)",
                  fontsize=10)
    axR.annotate("only ERCOT 2021 (Uri)\nmoves, and only for the\nclairvoyant oracle",
                 xy=(3, 8), xytext=(3.3, 9.6), fontsize=7.5, color="#08519c",
                 arrowprops=dict(arrowstyle="->", color="#08519c", lw=0.8))
    axR.legend(loc="lower right", fontsize=8, title="dispatch information")
    axR.grid(axis="y", alpha=0.3)

    fig.suptitle("Pay for stochastic dispatch to earn more, not to size: "
                 "the cheap deterministic inner loop already picks the right capacity",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    main(FIGURES / "fig_vss_capacity.png")
