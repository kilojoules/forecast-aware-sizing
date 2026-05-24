"""Generate paper figures from paper_results.json artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np


CAPEX_E = 100_000   # EUR/MWh
CAPEX_P = 75_000    # EUR/MW
DISCOUNT = 0.07
LIFETIME = 15
B_P = 1.0


def lifetime_npv(R_year: float, b_E: float, b_P: float = B_P) -> float:
    df_sum = sum((1 + DISCOUNT) ** -y for y in range(LIFETIME))
    return R_year * df_sum - CAPEX_E * b_E - CAPEX_P * b_P


def load_results(*paths) -> dict:
    """Merge paper_results JSONs from multiple files. Tags each year with
    its source (dk1 / ercot) into key 'market_year' to avoid collision."""
    merged = {"meta": {}, "by_market": {}}
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        src = r["meta"].get("source", "dk1")
        if src not in merged["by_market"]:
            merged["by_market"][src] = {}
        merged["by_market"][src].update(r["by_year"])
        if not merged["meta"]:
            merged["meta"] = r["meta"]
    return merged


def _market_year_panels(merged: dict):
    """Return list of (market, year, ye_dict, currency_label, market_label)
    suitable for panel iteration."""
    out = []
    for src in ["dk1", "ercot"]:
        if src not in merged["by_market"]:
            continue
        cur = "EUR" if src == "dk1" else "USD"
        label = "DK1" if src == "dk1" else "ERCOT N."
        for year in sorted(merged["by_market"][src]):
            out.append((src, year, merged["by_market"][src][year], cur, label))
    return out


CMAP = {
    "linear_single":     ("#aaaaaa", "o", "LP-linear single"),
    "linear_ensemble":   ("#117733", "s", "LP-linear K=4-lag"),
    "quadratic_single":  ("#cc6677", "D", "QP-quadratic single"),
    "quadratic_ensemble": ("#4477aa", "^", "QP-quadratic K=4-lag"),
}


def figure_revenue_vs_bE(merged: dict, out: str = "fig_paper_revenue.png"):
    panels = _market_year_panels(merged)
    n = len(panels)
    if n == 0:
        return
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), squeeze=False)
    for i, (src, year, ye, cur, label) in enumerate(panels):
        ax = axes[i // cols, i % cols]
        for key, (color, marker, lbl) in CMAP.items():
            rs = ye[key]
            ax.plot([r["b_E"] for r in rs], [r["R"] for r in rs],
                     marker=marker, color=color, label=lbl, lw=1.5, ms=6)
        ax.set_xscale("log")
        ax.set_xlabel(r"$b_E$ (MWh)")
        ax.set_ylabel(f"Annual revenue ({cur})")
        ax.set_title(f"{label} {year}")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="lower right")
    # blank unused axes
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("Realised arbitrage revenue vs battery capacity. Multi-lag ensemble dominates single-forecast.", y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


def figure_npv_vs_bE(merged: dict, out: str = "fig_paper_npv.png"):
    panels = _market_year_panels(merged)
    n = len(panels)
    if n == 0:
        return
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), squeeze=False)
    for i, (src, year, ye, cur, label) in enumerate(panels):
        ax = axes[i // cols, i % cols]
        bE_stars = []
        for key, (color, marker, lbl) in CMAP.items():
            rs = ye[key]
            b_E = np.array([r["b_E"] for r in rs])
            R = np.array([r["R"] for r in rs])
            npv = np.array([lifetime_npv(r, b) for r, b in zip(R, b_E)])
            ax.plot(b_E, npv, marker=marker, color=color, label=lbl, lw=1.5, ms=6)
            bE_stars.append(b_E[int(np.argmax(npv))])
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$b_E$ (MWh)")
        ax.set_ylabel(f"Lifetime NPV ({cur}, 15 yr 7%)")
        title = f"{label} {year}   $b_E^*$ = " + "/".join(f"{s:.0f}" for s in bE_stars)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="lower left")
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("Lifetime NPV vs battery capacity. Title shows $b_E^*$ for each policy in order LP-s/LP-e/QP-s/QP-e (all identical).", y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


def figure_lift_vs_bE(merged: dict, out: str = "fig_paper_lift.png"):
    panels = _market_year_panels(merged)
    n = len(panels)
    if n == 0:
        return
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.0 * rows), squeeze=False)
    for i, (src, year, ye, cur, label) in enumerate(panels):
        ax = axes[i // cols, i % cols]
        for cost, color, marker in [("linear", "#cc6677", "o"), ("quadratic", "#4477aa", "s")]:
            single = ye[f"{cost}_single"]
            ens = ye[f"{cost}_ensemble"]
            b_E = np.array([r["b_E"] for r in single])
            R_s = np.array([r["R"] for r in single])
            R_e = np.array([r["R"] for r in ens])
            lift = (R_e - R_s) / np.maximum(np.abs(R_s), 1e-6) * 100
            ax.plot(b_E, lift, marker=marker, color=color, label=cost.title(), lw=1.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$b_E$ (MWh)")
        ax.set_ylabel("Ensemble lift over single (%)")
        ax.set_title(f"{label} {year}")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("Ensemble lift vs $b_E$. Approximately constant-in-$b_E$ -> argmax invariant.", y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


def figure_spectrum(merged: dict, out: str = "fig_paper_spectrum.png"):
    """Plot welch PSDs of synthetic AR(1) + DK1 + ERCOT yearly traces."""
    from price_signal import synth_diurnal
    from dk_loader import load_dk_year
    from ercot_loader import load_ercot_year
    from spectrum import welch_psd

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    # DK1 panel
    ax = axes[0]
    x = synth_diurnal(168 * 4, seed=0)
    p, P = welch_psd(x)
    ax.loglog(p, P, label="Synthetic diurnal AR(1)", color="#aaaaaa", lw=1.5)
    colors = {"2021": "#117733", "2022": "#cc6677", "2023": "#4477aa"}
    for year in ["2021", "2022", "2023"]:
        try:
            df = load_dk_year(int(year))
            x = df["da_eur_per_mwh"].to_numpy()
            p, P = welch_psd(x)
            ax.loglog(p, P, label=f"DK1 {year}", color=colors[year], lw=1.5)
        except Exception as e:
            print(f"  DK1 {year} skip: {e}")
    for tau, label in [(12, "12h"), (24, "1d"), (168, "1 wk"), (720, "1 mo")]:
        ax.axvline(tau, color="black", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("PSD (EUR/MWh)$^2$ / cyc/day")
    ax.set_title("DK1 day-ahead spectra")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # ERCOT panel
    ax = axes[1]
    p, P = welch_psd(synth_diurnal(168 * 4, seed=0))
    ax.loglog(p, P, label="Synthetic diurnal AR(1)", color="#aaaaaa", lw=1.5)
    for year in ["2021", "2022", "2023"]:
        try:
            df = load_ercot_year(int(year))
            x = df["da_usd_per_mwh"].to_numpy()
            p, P = welch_psd(x)
            ax.loglog(p, P, label=f"ERCOT {year}", color=colors[year], lw=1.5)
        except Exception as e:
            print(f"  ERCOT {year} skip: {e}")
    for tau, label in [(12, "12h"), (24, "1d"), (168, "1 wk"), (720, "1 mo")]:
        ax.axvline(tau, color="black", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("PSD (USD/MWh)$^2$ / cyc/day")
    ax.set_title("ERCOT North Hub day-ahead spectra")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Spectral content: synthetic AR(1) (single timescale) vs DK1 + ERCOT (multi-timescale)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


def summary_table(merged: dict) -> None:
    panels = _market_year_panels(merged)
    print("\nSummary table (lifetime NPV at b_E* / cost-of-capital adjusted):")
    print(f"{'market':>10} {'year':>6} | {'b_E*':>5} | {'NPV_lin_s':>10} | {'NPV_lin_e':>10} | "
          f"{'NPV_qp_s':>10} | {'NPV_qp_e':>10} | {'lift_qp':>8}")
    print("-" * 100)
    for src, year, ye, cur, label in panels:
        best = {}
        for policy in ["linear_single", "linear_ensemble", "quadratic_single", "quadratic_ensemble"]:
            rows = ye[policy]
            bn = -1e18; bb = 0
            for row in rows:
                npv = lifetime_npv(row["R"], row["b_E"])
                if npv > bn:
                    bn = npv; bb = row["b_E"]
            best[policy] = (bb, bn)
        lift = (best["quadratic_ensemble"][1] - best["quadratic_single"][1]) / max(abs(best["quadratic_single"][1]), 1.0) * 100
        print(f"{label:>10} {year:>6} | {best['linear_single'][0]:5.0f} | "
              f"{best['linear_single'][1]:10.0f} | {best['linear_ensemble'][1]:10.0f} | "
              f"{best['quadratic_single'][1]:10.0f} | {best['quadratic_ensemble'][1]:10.0f} | {lift:+7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="paper_results JSON files")
    args = parser.parse_args()
    merged = load_results(*args.paths)
    figure_revenue_vs_bE(merged)
    figure_npv_vs_bE(merged)
    figure_lift_vs_bE(merged)
    figure_spectrum(merged)
    summary_table(merged)


if __name__ == "__main__":
    main()
