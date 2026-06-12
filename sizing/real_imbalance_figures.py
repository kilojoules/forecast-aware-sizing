"""Figure for the real-settlement + ratio-map extension (paper Sec 5.2).

Two panels:
  left  - effective imbalance penalty lambda_eff of real DK1 settlement
          (eSett, two-price and one-price) per study year, against the
          measured break-point band and the post-March-2025 up-regulation
          spread range.
  right - break-point lambda* vs wind/battery ratio (first argmax
          divergence), three DK1 years, showing saturation at ~50.

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python \
         sizing/real_imbalance_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SIZING_DIR = Path(__file__).resolve().parent
RESULTS = SIZING_DIR.parent / "results"
FIGURES = SIZING_DIR.parent / "paper" / "figures"

YEARS = (2021, 2022, 2023)
COLORS = {2021: "#1f77b4", 2022: "#d62728", 2023: "#2ca02c"}
# Post-March-2025 DK1 mean up-regulation spread vs DA (Blue Power Partners)
POST2025_BAND = (92.0, 123.0)


DISC = sum(1.07 ** -y for y in range(15))
CAPEX_E, CAPEX_P = 100_000.0, 75_000.0
FINE_LAMBDAS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80,
                100, 125, 150, 200, 350, 500]


def divergence_segments(rows):
    """Contiguous lambda ranges where the single-forecast policy sizes
    ABOVE the ensemble. Residuals are lambda-independent, so NPV is
    recomputed on the fine lambda grid from stored
    (arb_rev, wind_da_rev, imb_abs)."""
    cells = {}
    for r in rows:
        cells[(r["b_E"], r["policy"])] = (r["arb_rev"], r["wind_da_rev"],
                                          r["imb_abs"])
    bEs = sorted({k[0] for k in cells})
    diverged = []
    for lam in FINE_LAMBDAS:
        b = {}
        for pol in ("single", "ensemble"):
            b[pol] = max(
                (DISC * (cells[(bE, pol)][0] + cells[(bE, pol)][1]
                         - lam * cells[(bE, pol)][2])
                 - CAPEX_E * bE - CAPEX_P, bE) for bE in bEs)[1]
        diverged.append(b["single"] > b["ensemble"])
    segs, start = [], None
    for i, d in enumerate(diverged):
        if d and start is None:
            start = FINE_LAMBDAS[i]
        if not d and start is not None:
            segs.append((start, FINE_LAMBDAS[i]))
            start = None
    if start is not None:
        segs.append((start, FINE_LAMBDAS[-1]))
    return segs


def main(out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- left: lambda_eff of real settlement vs break-point band
    for i, y in enumerate(YEARS):
        d = json.load(open(RESULTS / "imbalance" / f"dk1_{y}_real.json"))
        le2 = d["summary"]["two_price"]["ensemble"]["lam_eff"]
        le1 = d["summary"]["one_price"]["ensemble"]["lam_eff"]
        ax1.plot([le2], [i], "D", color=COLORS[y], ms=9)
        ax1.plot([le1], [i], "o", mfc="none", mec=COLORS[y], ms=9)
        ax1.text(le2 + 3, i, f"{y}", color=COLORS[y], fontsize=10,
                 va="center")
    ax1.set_ylim(-0.6, len(YEARS) - 0.4)
    ax1.axvspan(50, 100, color="grey", alpha=0.25,
                label=r"measured $\lambda^*$ band")
    ax1.axvspan(*POST2025_BAND, color="#d62728", alpha=0.15,
                label="post-2025 up-reg spread")
    ax1.set_xlim(0, 140)
    ax1.set_yticks([])
    ax1.set_xlabel(r"effective imbalance penalty $\lambda$ (€/MWh)",
                   fontsize=10)
    ax1.set_title("Real DK1 settlement sat below the break-point\n"
                  r"(filled $\diamond$ two-price, open $\circ$ one-price"
                  r" $\lambda_{\mathrm{eff}}$; no argmax divergence any year)",
                  fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)

    # --- right: divergence bands (single sizes above ensemble) vs ratio
    ratios = [2, 5, 10, 20]
    yoff = {2021: -0.22, 2022: 0.0, 2023: 0.22}
    for iy, W in enumerate(ratios):
        for y in YEARS:
            if W == 5:
                fp = RESULTS / "imbalance" / f"dk1_{y}.json"
            else:
                fp = RESULTS / "imbalance" / "ratio" / f"dk1_{y}_w{W}.json"
            segs = divergence_segments(json.load(open(fp))["rows"])
            for (a, b) in segs:
                ax2.plot([max(a, 5), b], [iy + yoff[y]] * 2, "-",
                         color=COLORS[y], lw=5, solid_capstyle="butt",
                         label=f"DK1 {y}" if (iy, (a, b)) == (1, segs[0]) else None)
    ax2.axvspan(10.9, 27.7, color="grey", alpha=0.25,
                label=r"real DK1 $\lambda_{\mathrm{eff}}$ 2021--23")
    ax2.axvspan(*POST2025_BAND, color="#d62728", alpha=0.15,
                label="post-2025 up-reg spread")
    ax2.set_xscale("log")
    ax2.set_xlim(5, 550)
    ax2.set_yticks(range(len(ratios)))
    ax2.set_yticklabels([f"{r}:1" for r in ratios])
    ax2.set_xlabel(r"imbalance penalty $\lambda$ (€/MWh)", fontsize=10)
    ax2.set_ylabel("wind/battery capacity ratio", fontsize=10)
    ax2.set_title("Where single-forecast sizing exceeds ensemble sizing\n"
                  "(thick segments; persistent band opens at 45–70 for "
                  "ratios ≥ 10)", fontsize=10)
    handles, labels = ax2.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax2.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    main(FIGURES / "fig_paper_real_imbalance.png")
