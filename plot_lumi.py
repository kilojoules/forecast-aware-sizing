"""Aggregate LUMI het-fleet results into a summary table and bar chart."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONFIGS = [
    ((2, 20), (2, 10), (2, 10)),
    ((5, 20), (2, 10), (2, 10)),
    ((2, 50), (2, 25), (2, 25)),
    ((5, 50), (2, 25), (2, 25)),
    ((10, 100), (5, 50), (5, 50)),
]
SEEDS = [42, 7]


def main():
    results_dir = Path("lumi_results")
    rows = []
    for ti, (B, c, d) in enumerate(CONFIGS):
        per_seed = {"naive": [], "greedy": [], "elm": []}
        for si, seed in enumerate(SEEDS):
            task_idx = 2 * ti + si
            f = results_dir / f"het_{task_idx}_seed{seed}.json"
            if not f.exists():
                print(f"  missing: {f}")
                continue
            data = json.loads(f.read_text())
            for r in data:
                k = r["name"].lower().replace("-rl", "").replace("greedy", "greedy")
                if "naive" in k:
                    per_seed["naive"].append(sum(r["D_per_battery"]))
                elif "greedy" in k:
                    per_seed["greedy"].append(sum(r["D_per_battery"]))
                elif "elm" in k:
                    per_seed["elm"].append(sum(r["D_per_battery"]))
        m = {k: statistics.mean(v) for k, v in per_seed.items() if v}
        s = {k: (statistics.stdev(v) if len(v) > 1 else 0.0) for k, v in per_seed.items() if v}
        rows.append({"B": B, "naive_m": m["naive"], "naive_s": s["naive"],
                     "greedy_m": m["greedy"], "greedy_s": s["greedy"],
                     "elm_m": m["elm"], "elm_s": s["elm"]})
        print(
            f"B={B}: Naive={m['naive']:.3f}±{s['naive']:.3f}  "
            f"Greedy={m['greedy']:.3f}±{s['greedy']:.3f}  "
            f"ELM={m['elm']:.3f}±{s['elm']:.3f}  "
            f"ELMvsNaive={(m['naive']-m['elm'])/m['naive']*100:+.1f}%  "
            f"ELMvsGreedy={(m['greedy']-m['elm'])/m['greedy']*100:+.1f}%"
        )

    # Bar chart: D_sum normalized by Naive (Naive=1.0 reference)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    width = 0.27
    naive_vals = [1.0 for _ in rows]
    greedy_vals = [r["greedy_m"] / r["naive_m"] for r in rows]
    elm_vals = [r["elm_m"] / r["naive_m"] for r in rows]
    elm_err = [r["elm_s"] / r["naive_m"] for r in rows]

    ax.bar(x - width, naive_vals, width, label="Naive (baseline=1.0)", color="#aaaaaa")
    ax.bar(x, greedy_vals, width, label="Greedy (proxy optimum)", color="#4477aa")
    ax.bar(x + width, elm_vals, width, yerr=elm_err, label="ELM-RL (T=1M)", color="#cc6677", capsize=4)

    labels = [f"B={r['B']}" for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total rainflow D (normalized by Naive)")
    ax.set_title("Heterogeneous fleet: ELM-RL vs Greedy vs Naive (LUMI, T=1M, 2 seeds)")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_lumi_summary.png", dpi=120)
    print("\nWrote fig_lumi_summary.png")

    # Markdown table
    md = ["| Config | Naive | Greedy | ELM-RL | ELM vs Naive | ELM vs Greedy |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| B={r['B']} | {r['naive_m']:.3f} | {r['greedy_m']:.3f} | "
            f"{r['elm_m']:.3f}±{r['elm_s']:.3f} | "
            f"**{(r['naive_m']-r['elm_m'])/r['naive_m']*100:+.1f}%** | "
            f"{(r['greedy_m']-r['elm_m'])/r['greedy_m']*100:+.1f}% |"
        )
    Path("lumi_summary.md").write_text("\n".join(md) + "\n")
    print("Wrote lumi_summary.md")


if __name__ == "__main__":
    main()
