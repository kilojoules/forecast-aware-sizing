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


def lambda_star(rows) -> float | None:
    """Smallest lambda with argmax divergence, or None."""
    for lam in sorted({r["lambda"] for r in rows}):
        b = {}
        for pol in ("single", "ensemble"):
            sub = [r for r in rows if r["policy"] == pol and r["lambda"] == lam]
            b[pol] = max(sub, key=lambda r: r["npv"])["b_E"]
        if b["single"] != b["ensemble"]:
            return lam
    return None


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

    # --- right: lambda* vs wind/battery ratio
    ratios = [2, 5, 10, 20]
    jitter = {2021: 0.97, 2022: 1.0, 2023: 1.03}
    for y in YEARS:
        stars = []
        for W in ratios:
            if W == 5:
                fp = RESULTS / "imbalance" / f"dk1_{y}.json"
            else:
                fp = RESULTS / "imbalance" / "ratio" / f"dk1_{y}_w{W}.json"
            rows = json.load(open(fp))["rows"]
            stars.append(lambda_star(rows))
        xs = [r * jitter[y] for r, s in zip(ratios, stars) if s is not None]
        ys = [s for s in stars if s is not None]
        ax2.plot(xs, ys, "-o", color=COLORS[y], lw=2, ms=7, label=f"DK1 {y}")
        # mark "no divergence <= 500" with an open upward marker at top
        for r, s in zip(ratios, stars):
            if s is None:
                ax2.plot([r * jitter[y]], [500], "^", mfc="none",
                         mec=COLORS[y], ms=9)
    ax2.axhspan(10.9, 27.7, color="grey", alpha=0.25,
                label=r"real DK1 $\lambda_{\mathrm{eff}}$ 2021--23")
    ax2.axhspan(*POST2025_BAND, color="#d62728", alpha=0.15,
                label="post-2025 up-reg spread")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(ratios)
    ax2.set_xticklabels([str(r) for r in ratios])
    ax2.set_xlabel("wind/battery capacity ratio $W_{\\mathrm{peak}}/b_P$",
                   fontsize=10)
    ax2.set_ylabel(r"break-point $\lambda^*$ (€/MWh)", fontsize=10)
    ax2.set_title("Break-point saturates near 50 €/MWh\n"
                  r"(open $\triangle$: no divergence up to $\lambda=500$)",
                  fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    main(FIGURES / "fig_paper_real_imbalance.png")
