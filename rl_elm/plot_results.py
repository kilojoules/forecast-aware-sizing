"""Plot DoD histograms (Fig 2 of paper) and a results-summary table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(in_path: str, out_path: str = "fig_dod.png"):
    results = json.load(open(in_path))
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 21)
    for ax, res in zip(axes, results):
        all_dod = []
        all_count = []
        for dods, counts in res["dod_hist"]:
            all_dod.extend(dods)
            all_count.extend(counts)
        all_dod = np.array(all_dod)
        all_count = np.array(all_count)
        ax.hist(all_dod, bins=bins, weights=all_count)
        D = res["D_per_battery"]
        R = res["reward"]
        ax.set_title(f"{res['name']}\nR={R:.1f}, D=({D[0]:.3f},{D[1]:.3f})")
        ax.set_xlabel("DoD (fraction)")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Cycle frequency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_paperfaithful.json",
         sys.argv[2] if len(sys.argv) > 2 else "fig_dod.png")
