"""Generate the full plot suite from local + LUMI artifacts.

Outputs:
  fig_headline.png       -- already exists from repro.py (B=(10,100) DoD)
  fig_lumi_summary.png   -- already exists from plot_lumi.py (5-config bars)
  fig_soc_traces.png     -- time-domain SoC traces, Naive vs ELM, B=(10,100)
  fig_dod_grid.png       -- 5-config grid of DoD per battery (Naive vs ELM)
  fig_action_match.png   -- bar chart of ELM-vs-Greedy match rate
  fig_proxy_vs_d.png     -- scatter of cumulative proxy reward vs rainflow D
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from degradation import cycle_degradation


CONFIGS = [
    ((2, 20),  (2, 10),  (2, 10)),
    ((5, 20),  (2, 10),  (2, 10)),
    ((2, 50),  (2, 25),  (2, 25)),
    ((5, 50),  (2, 25),  (2, 25)),
    ((10, 100),(5, 50),  (5, 50)),
]
SEEDS = [42, 7]


def load_lumi(task_idx: int, seed: int):
    f = Path("lumi_results") / f"het_{task_idx}_seed{seed}.json"
    return json.loads(f.read_text())


# ---- 1) SoC traces ----------------------------------------------------------
def plot_soc_traces(window: int = 1000, out: str = "fig_soc_traces.png"):
    data = load_lumi(8, 42)  # B=(10,100), seed=42
    by_name = {r["name"]: r for r in data}
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex="col")
    cmap = {"Naive": "#aaaaaa", "Greedy": "#4477aa", "ELM-RL": "#cc6677"}
    for col, batt_idx in enumerate([0, 1]):
        for row, name in enumerate(["Naive", "ELM-RL"]):
            ax = axes[row, col]
            soc = by_name[name]["soc_log"][batt_idx][:window]
            ax.plot(soc, color=cmap[name], lw=0.7)
            ax.set_title(f"{name} -- battery {batt_idx+1} (B={[10,100][batt_idx]})")
            ax.set_ylabel("SoC")
            ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("time step")
    fig.suptitle("SoC traces, first 1000 steps, B=(10,100) -- ELM keeps small battery shallow", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  wrote {out}")


# ---- 2) DoD grid across 5 configs -------------------------------------------
def plot_dod_grid(out: str = "fig_dod_grid.png"):
    fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharex="col")
    bins = np.linspace(0, 1, 21)
    for col, ((B, c, d), task_offset) in enumerate(zip(CONFIGS, range(0, 10, 2))):
        data = load_lumi(task_offset, 42)
        by_name = {r["name"]: r for r in data}
        for row, batt_idx in enumerate([0, 1]):
            ax = axes[row, col]
            for name, color in [("Naive", "#aaaaaa"), ("ELM-RL", "#cc6677")]:
                soc = by_name[name]["soc_log"][batt_idx]
                D, dods, counts = cycle_degradation(soc, B[batt_idx])
                ax.hist(dods, bins=bins, weights=counts, alpha=0.6, label=f"{name} D={D:.3f}", color=color)
            ax.legend(fontsize=7)
            ax.set_title(f"B={B}  batt {batt_idx+1} (B_i={B[batt_idx]})")
            if row == 1:
                ax.set_xlabel("DoD (frac)")
            if col == 0:
                ax.set_ylabel("cycle count")
    fig.suptitle("DoD distributions across 5 het configs (LUMI seed=42). ELM consistently kills mid/deep cycles on small battery.", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  wrote {out}")


# ---- 3) Action-match bar -----------------------------------------------------
def plot_action_match(out: str = "fig_action_match.png"):
    matches = [
        ("B=(2,50) seed=42", 99.5),
        ("B=(2,50) seed=7",  99.6),
        ("B=(10,100) seed=42", 95.7),
        ("B=(10,100) seed=7",  97.4),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [m[0] for m in matches]
    rates = [m[1] for m in matches]
    bars = ax.barh(labels, rates, color="#cc6677")
    ax.axvline(80, color="black", linestyle="--", lw=1, label="80% threshold (smooth-Greedy)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("ELM action == Greedy action (%)")
    ax.set_title("Action-match rate: ELM-RL is a smooth function-approximator copy of Greedy")
    for bar, r in zip(bars, rates):
        ax.text(r - 2, bar.get_y() + bar.get_height()/2, f"{r:.1f}%", ha="right", va="center", color="white", fontsize=10)
    ax.legend(loc="lower left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  wrote {out}")


# ---- 4) Proxy-vs-D scatter from LUMI runs -----------------------------------
def plot_proxy_vs_d(out: str = "fig_proxy_vs_d.png"):
    pts = []
    for ti, (B, c, d) in enumerate(CONFIGS):
        for si, seed in enumerate(SEEDS):
            data = load_lumi(2 * ti + si, seed)
            for r in data:
                pts.append({"B": B, "name": r["name"], "R": r["reward"], "D": sum(r["D_per_battery"])})
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4 * len(CONFIGS), 4))
    cmap = {"Naive": "#aaaaaa", "Greedy": "#4477aa", "ELM-RL": "#cc6677"}
    for ax, (B, c, d) in zip(axes, CONFIGS):
        sub = [p for p in pts if p["B"] == B]
        for name in ["Naive", "Greedy", "ELM-RL"]:
            xs = [p["R"] for p in sub if p["name"] == name]
            ys = [p["D"] for p in sub if p["name"] == name]
            ax.scatter(xs, ys, color=cmap[name], s=80, edgecolors="black", lw=0.5, label=name)
        ax.set_xlabel("cumulative proxy R")
        ax.set_ylabel("rainflow D")
        ax.set_title(f"B={B}")
        ax.grid(alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Proxy reward and rainflow D agree (higher R ↔ lower D). Across 5 configs × 2 seeds.", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  wrote {out}")


def main():
    print("Generating plot suite...")
    plot_soc_traces()
    plot_dod_grid()
    plot_action_match()
    plot_proxy_vs_d()
    print("Done.")


if __name__ == "__main__":
    main()
