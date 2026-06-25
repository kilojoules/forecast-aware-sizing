"""Stress-gauntlet scoreboard: does merchant sizing-invariance survive
each adversarial attack?

Replaces four near-redundant invariance figures (npv / lift / 2d / slp)
with one scoreboard. Rows = 6 market-years; columns = attacks ordered by
increasing strength; each cell tests whether the optimal capacity argmax
is invariant (green check) or shifts (red x) under that attack. The few
red cells -- DK1 2022 under the K=20 quantile ensemble, and ERCOT 2021
(Storm Uri) under several attacks -- are exactly where the paper says
invariance is fragile.

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python sizing/fig_gauntlet.py
"""
from __future__ import annotations

import glob
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
B_P = 1.0
REGIMES = ["dk1_2021", "dk1_2022", "dk1_2023",
           "ercot_2021", "ercot_2022", "ercot_2023"]


def argmax_be(rows, key="R"):
    return max(rows, key=lambda r: r[key] * DISC - CE * r["b_E"] - CP * B_P)["b_E"]


def main_file(reg):
    src, yr = reg.split("_")
    fp = (RESULTS / "main" / f"paper_{yr}.json") if src == "dk1" \
        else (RESULTS / "main" / "paper_ercot.json")
    return json.load(open(fp))["by_year"][yr]


def col_persistence(reg):
    d = main_file(reg)
    s, e = argmax_be(d["linear_single"]), argmax_be(d["linear_ensemble"])
    return s, e, s == e


def col_slp(reg):
    d = main_file(reg)
    src, yr = reg.split("_")
    slp = json.load(open(RESULTS / "slp" / f"{reg}.json"))["by_year"][yr]
    e = argmax_be(d["quadratic_ensemble"])
    s = argmax_be(slp["quadratic_slp"])
    return s, e, s == e


def col_maxmin(reg):
    d = main_file(reg)
    e = argmax_be(d["linear_ensemble"])
    rob = json.load(open(RESULTS / "main" / f"robust_{reg}.json"))
    s = max(rob["rows"], key=lambda r: r["npv"])["b_E"]
    return s, e, s == e


def col_quantile(reg):
    src, yr = reg.split("_")
    q = json.load(open(RESULTS / "quantile" / f"{reg}.json"))["by_year"][yr]
    s, e = argmax_be(q["linear_single"]), argmax_be(q["linear_ensemble"])
    return s, e, s == e


def col_2d(reg):
    src, yr = reg.split("_")
    grid = {}
    for fp in glob.glob(str(RESULTS / "2d" / "2d_bP*.json")):
        d = json.load(open(fp))
        bP, bE = d["b_P"], d["b_E"]
        m = d["by_market"].get(src, {}).get(yr)
        if m:
            grid[(bP, bE)] = m
    def amax(pol):
        return max(grid.items(),
                   key=lambda kv: kv[1][pol]["R"] * DISC - CE * kv[0][1] - CP * kv[0][0])[0][1]
    s, e = amax("linear_single"), amax("linear_ensemble")
    return s, e, s == e


def col_gamma(reg):
    gs = json.load(open(RESULTS / "main" / "gamma_sweep.json"))["regimes"][reg]
    # invariant on the realizable skill range gamma in [0,1]
    on01 = [r["b_E_star"] for r in gs if r["gamma"] <= 1.0]
    flat = len(set(on01)) == 1
    return on01[0], on01[-1], flat


ATTACKS = [
    ("K=4 persistence\nensemble", col_persistence),
    ("2-D $(b_E,b_P)$\nsurface", col_2d),
    ("scenario SLP\n(N=50)", col_slp),
    ("max-min robust\n(worst-case)", col_maxmin),
    ("continuous $\\gamma$\n(skill range)", col_gamma),
    ("K=20 quantile\nensemble", col_quantile),
]


def main(out: Path):
    nR, nC = len(REGIMES), len(ATTACKS)
    inv = np.zeros((nR, nC), dtype=bool)
    annot = [[""] * nC for _ in range(nR)]
    for i, reg in enumerate(REGIMES):
        for j, (_, fn) in enumerate(ATTACKS):
            s, e, ok = fn(reg)
            inv[i, j] = ok
            annot[i][j] = ("✓" if ok else "✗") + f"\n{s:g}/{e:g}"

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = np.where(inv, 0.82, 0.30)  # green-ish vs red-ish via two colors
    for i in range(nR):
        for j in range(nC):
            color = "#2ca02c" if inv[i, j] else "#d62728"
            ax.add_patch(plt.Rectangle((j, nR - 1 - i), 1, 1, color=color,
                                       alpha=0.22, ec="white", lw=2))
            ax.text(j + 0.5, nR - 1 - i + 0.5, annot[i][j], ha="center",
                    va="center", fontsize=8.5,
                    color="#1a7a1a" if inv[i, j] else "#a01010")
    ax.set_xlim(0, nC); ax.set_ylim(0, nR)
    ax.set_xticks(np.arange(nC) + 0.5)
    ax.set_xticklabels([a[0] for a in ATTACKS], fontsize=8.5)
    ax.set_yticks(np.arange(nR) + 0.5)
    ax.set_yticklabels([r.replace("_", " ") for r in REGIMES[::-1]], fontsize=9)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    n_break = int((~inv).sum())
    ax.set_title("Stress gauntlet: does merchant sizing-invariance survive each attack?\n"
                 f"cell = argmax(single / comparator); ✓ invariant, ✗ shifts. "
                 f"{nR*nC - n_break}/{nR*nC} survive.",
                 fontsize=10.5)
    ax.annotate("attacks ordered by increasing strength →",
                xy=(0.5, -0.13), xycoords="axes fraction", ha="center",
                fontsize=8.5, color="0.4")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}  ({nR*nC - n_break}/{nR*nC} invariant)")
    for i, reg in enumerate(REGIMES):
        print(f"  {reg}: " + " ".join(
            f"{ATTACKS[j][0].split(chr(10))[0]}={'OK' if inv[i,j] else 'BREAK'}"
            for j in range(nC)))


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    main(FIGURES / "fig_gauntlet.png")
