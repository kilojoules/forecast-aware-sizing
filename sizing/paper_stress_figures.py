"""Stress-test figures: 2-D heatmap, quantile-vs-persistence, SLP comparison."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

SIZING_DIR = Path(__file__).resolve().parent
RESULTS = SIZING_DIR.parent / "results"
FIGURES = SIZING_DIR.parent / "paper" / "figures"
sys.path.insert(0, str(SIZING_DIR))

import matplotlib.pyplot as plt
import numpy as np


CAPEX_E = 100_000
CAPEX_P = 75_000
DISC = sum((1.07) ** -y for y in range(15))
B_P_DEFAULT = 1.0


def lifetime_npv(R, b_E, b_P=B_P_DEFAULT):
    return R * DISC - CAPEX_E * b_E - CAPEX_P * b_P


# -----------------------------------------------------------------------------
# 2-D heatmap from ../results/2d/*.json
# -----------------------------------------------------------------------------
def figure_2d_heatmap(out: str = str(FIGURES / "fig_paper_2d.png")):
    files = sorted(glob.glob(str(RESULTS / "2d" / "2d_*.json")))
    if not files:
        print("no results_2d files; skipping")
        return
    bymyp = {}
    for fp in files:
        r = json.load(open(fp))
        bP, bE = r["b_P"], r["b_E"]
        for src, ye_dict in r["by_market"].items():
            for year, pol_dict in ye_dict.items():
                for pol, vals in pol_dict.items():
                    npv = lifetime_npv(vals["R"], bE, bP)
                    bymyp.setdefault((src, year, pol), {})[(bP, bE)] = npv

    bP_grid = sorted({k[0] for d in bymyp.values() for k in d.keys()})
    bE_grid = sorted({k[1] for d in bymyp.values() for k in d.keys()})

    panels = [(s, y) for s in ["dk1", "ercot"] for y in ["2021", "2022", "2023"]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for i, (src, year) in enumerate(panels):
        ax = axes[i // 3, i % 3]
        # Use QP-ensemble NPV surface
        d = bymyp.get((src, year, "quadratic_ensemble"), {})
        if not d:
            ax.axis("off"); continue
        Z = np.zeros((len(bP_grid), len(bE_grid)))
        for ip, bP in enumerate(bP_grid):
            for ib, bE in enumerate(bE_grid):
                Z[ip, ib] = d.get((bP, bE), np.nan) / 1e6
        im = ax.pcolormesh(bE_grid, bP_grid, Z, shading="nearest", cmap="viridis")
        # Mark argmaxes for each policy
        markers = {"linear_single": ("o", "white"), "linear_ensemble": ("s", "yellow"),
                    "quadratic_single": ("D", "magenta"), "quadratic_ensemble": ("^", "red")}
        for pol, (mk, color) in markers.items():
            d2 = bymyp.get((src, year, pol), {})
            if d2:
                best = max(d2.items(), key=lambda kv: kv[1])
                ax.plot(best[0][1], best[0][0], marker=mk, color=color,
                        ms=10, mew=1.5, mec="black", label=pol[:9])
        ax.set_xscale("log")
        ax.set_yscale("log")
        market_label = "DK1" if src == "dk1" else "ERCOT N."
        ax.set_title(f"{market_label} {year}")
        if i // 3 == 1:
            ax.set_xlabel(r"$b_E$ (MWh)")
        if i % 3 == 0:
            ax.set_ylabel(r"$b_P$ (MW)")
        plt.colorbar(im, ax=ax, label="NPV (M)")
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("2-D NPV surfaces (QP-ensemble dispatch). Markers: argmax per policy. Colocation -> argmax invariant.", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Quantile-vs-persistence comparison
# -----------------------------------------------------------------------------
def figure_quantile_vs_persistence(out: str = str(FIGURES / "fig_paper_quantile.png")):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    panels = [(s, y) for s in ["dk1", "ercot"] for y in [2021, 2022, 2023]]
    for i, (src, year) in enumerate(panels):
        ax = axes[i // 3, i % 3]
        # Persistence
        try:
            if src == "dk1":
                p = json.load(open(str(RESULTS / "main" / f"paper_{year}.json")))
                ye_p = p["by_year"][str(year)]
            else:
                p = json.load(open(str(RESULTS / "main" / "paper_ercot.json")))
                ye_p = p["by_year"][str(year)]
        except FileNotFoundError:
            ax.axis("off"); continue
        # Quantile
        try:
            q = json.load(open(str(RESULTS / "quantile" / f"{src}_{year}.json")))
            ye_q = q["by_year"][str(year)]
        except FileNotFoundError:
            ye_q = None
        for label, ye, ls in [("persistence-K=4", ye_p, "-"),
                                 ("quantile-K=20", ye_q, "--")]:
            if ye is None: continue
            for cost, color in [("linear", "#cc6677"), ("quadratic", "#4477aa")]:
                rows_s = ye[f"{cost}_single"]
                rows_e = ye[f"{cost}_ensemble"]
                bE = np.array([r["b_E"] for r in rows_s])
                R_s = np.array([r["R"] for r in rows_s])
                R_e = np.array([r["R"] for r in rows_e])
                npv_s = np.array([lifetime_npv(R, b) for R, b in zip(R_s, bE)])
                npv_e = np.array([lifetime_npv(R, b) for R, b in zip(R_e, bE)])
                ax.plot(bE, npv_s / 1e6, color=color, ls=ls, alpha=0.55, lw=1.0)
                ax.plot(bE, npv_e / 1e6, color=color, ls=ls, lw=2.0,
                        label=f"{cost}-ens {label}" if i == 0 else None)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xscale("log")
        market_label = "DK1" if src == "dk1" else "ERCOT N."
        ax.set_title(f"{market_label} {year}", fontsize=10)
        if i // 3 == 1:
            ax.set_xlabel(r"$b_E$ (MWh)")
        if i % 3 == 0:
            ax.set_ylabel("NPV (M, 15y, 7%)")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Persistence-K=4 (solid) vs Quantile-K=20 (dashed) ensembles. DK1 2022 quantile breaks invariance.", y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# SLP comparison
# -----------------------------------------------------------------------------
def figure_slp(out: str = str(FIGURES / "fig_paper_slp.png")):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    panels = [(s, y) for s in ["dk1", "ercot"] for y in [2021, 2022, 2023]]
    for i, (src, year) in enumerate(panels):
        ax = axes[i // 3, i % 3]
        try:
            if src == "dk1":
                p = json.load(open(str(RESULTS / "main" / f"paper_{year}.json")))
                ye_p = p["by_year"][str(year)]
            else:
                p = json.load(open(str(RESULTS / "main" / "paper_ercot.json")))
                ye_p = p["by_year"][str(year)]
        except FileNotFoundError:
            ax.axis("off"); continue
        try:
            s_data = json.load(open(str(RESULTS / "slp" / f"{src}_{year}.json")))
            ye_slp = s_data["by_year"][str(year)]
        except FileNotFoundError:
            ye_slp = None
        # Plot QP-single, QP-ensemble, SLP
        for key, color, label in [("quadratic_single", "#cc6677", "QP-single"),
                                   ("quadratic_ensemble", "#4477aa", "QP-ens K=4 lag")]:
            rows = ye_p[key]
            bE = np.array([r["b_E"] for r in rows])
            npv = np.array([lifetime_npv(r["R"], b) for r, b in zip(rows, bE)])
            ax.plot(bE, npv / 1e6, "-", color=color, lw=2,
                     label=label if i == 0 else None)
        if ye_slp:
            rows = ye_slp.get("quadratic_slp", [])
            bE = np.array([r["b_E"] for r in rows])
            npv = np.array([lifetime_npv(r["R"], b) for r, b in zip(rows, bE)])
            ax.plot(bE, npv / 1e6, "D-", color="#117733", lw=2,
                     label="SLP rolling N=50" if i == 0 else None, ms=6)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xscale("log")
        market_label = "DK1" if src == "dk1" else "ERCOT N."
        ax.set_title(f"{market_label} {year}", fontsize=10)
        if i // 3 == 1:
            ax.set_xlabel(r"$b_E$ (MWh)")
        if i % 3 == 0:
            ax.set_ylabel("NPV (M, 15y, 7%)")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("SLP rolling-window dispatch vs full-horizon QP-single + QP-ensemble. SLP NPV is lower (rolling vs perfect-foresight gap) but $b_E^*$ matches.", y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


def figure_hydesign(out: str = str(FIGURES / "fig_paper_hydesign.png")):
    """Hydesign-default (DoD=0.9, batched 110-h cycle-balance) vs relaxed
    (= our LP) NPV across b_E, all 6 (market, year)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    panels = [(s, y) for s in ["dk1", "ercot"] for y in [2021, 2022, 2023]]
    summary = []
    for i, (src, year) in enumerate(panels):
        ax = axes[i // 3, i % 3]
        try:
            h = json.load(open(str(RESULTS / "hydesign" / f"{src}_{year}.json")))
            ye = h["by_year"][str(year)]
        except FileNotFoundError:
            ax.axis("off"); continue
        for key, color, label, mk in [
            ("hydesign_default", "#cc6677", "hydesign default (DoD=0.9, batched cycle balance)", "o"),
            ("hydesign_relaxed", "#4477aa", "merchant relaxed (= our QP-single)", "s"),
        ]:
            rows = ye[key]
            bE = np.array([r["b_E"] for r in rows])
            R = np.array([r["R"] for r in rows])
            npv = np.array([lifetime_npv(r, b) for r, b in zip(R, bE)])
            ax.plot(bE, npv / 1e6, mk + "-", color=color, lw=1.6,
                     label=label if i == 0 else None, ms=4)
            argmax_idx = int(np.argmax(npv))
            argmax_b = float(bE[argmax_idx])
            argmax_npv = float(npv[argmax_idx])
            ax.axvline(argmax_b, color=color, ls=":", lw=0.8, alpha=0.6)
            if key == "hydesign_default":
                summary.append({"src": src, "year": year,
                                 "default_argmax": argmax_b,
                                 "default_npv_at_argmax": argmax_npv,
                                 "default_R_array": R.tolist()})
            else:
                summary[-1]["relaxed_argmax"] = argmax_b
                summary[-1]["relaxed_npv_at_argmax"] = argmax_npv
                summary[-1]["npv_gap_M"] = (argmax_npv
                                             - summary[-1]["default_npv_at_argmax"]) / 1e6
                summary[-1]["npv_gap_pct"] = float(
                    100 * (argmax_npv - summary[-1]["default_npv_at_argmax"])
                    / max(abs(argmax_npv), 1e-6)
                )
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xscale("log")
        market_label = "DK1" if src == "dk1" else "ERCOT N."
        ax.set_title(f"{market_label} {year}", fontsize=10)
        if i // 3 == 1:
            ax.set_xlabel(r"$b_E$ (MWh)")
        if i % 3 == 0:
            ax.set_ylabel("NPV (M, 15y, 7%)")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Hydesign-default off-the-shelf vs unrestricted LP. Defaults leave 17--65\\% on the table; "
                  r"$b_E^*$ shifts down under defaults.", y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")
    print("\nHydesign summary table (NPV at argmax):")
    print(f"  {'market':5s} {'year':4s}  {'def b_E*':>9s} {'def NPV':>10s}  "
          f"{'rel b_E*':>9s} {'rel NPV':>10s}  {'NPV gap':>8s}  argmax_shift")
    for r in summary:
        shift = "→ shift" if r['default_argmax'] != r['relaxed_argmax'] else ""
        print(f"  {r['src']:5s} {r['year']}  "
              f"{r['default_argmax']:>9.1f} {r['default_npv_at_argmax']/1e6:>9.2f}M  "
              f"{r['relaxed_argmax']:>9.1f} {r['relaxed_npv_at_argmax']/1e6:>9.2f}M  "
              f"{r['npv_gap_pct']:>+7.1f}%  {shift}")
    return summary


# -----------------------------------------------------------------------------
# Imbalance-penalty sweep from ../results/imbalance/dk1_*.json
# -----------------------------------------------------------------------------
def figure_imbalance(out: str = str(FIGURES / "fig_paper_imbalance.png")):
    files = sorted(glob.glob(str(RESULTS / "imbalance" / "dk1_*.json")))
    if not files:
        print("no results_imbalance files; skipping")
        return
    data = {}
    for fp in files:
        r = json.load(open(fp))
        year = r["meta"]["year"]
        data[year] = r
    years = sorted(data.keys())
    if not years:
        return

    # For each (year, policy, lambda): find argmax b_E by NPV.
    lambdas = data[years[0]]["meta"]["lambda"]
    b_grid = data[years[0]]["meta"]["B_E"]

    def argmax_b(rows, policy, lam):
        sub = [r for r in rows if r["policy"] == policy and r["lambda"] == lam]
        if not sub:
            return None, None
        best = max(sub, key=lambda r: r["npv"])
        return best["b_E"], best["npv"]

    # Figure: 2 rows × 3 cols.
    # Row 1: argmax b_E* vs lambda for each policy, one panel per year.
    # Row 2: NPV(b_E) curves at largest lambda where divergence appears.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    summary = []
    for i, year in enumerate(years):
        ax = axes[0, i]
        rows = data[year]["rows"]
        b_single = [argmax_b(rows, "single", lam)[0] for lam in lambdas]
        b_ens = [argmax_b(rows, "ensemble", lam)[0] for lam in lambdas]
        ax.plot(lambdas, b_single, "o-", color="C0", label="single (lag-24h)")
        ax.plot(lambdas, b_ens, "s-", color="C1", label="ensemble (K=4)")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$ (EUR/MWh imbalance)")
        if i == 0:
            ax.set_ylabel(r"$b_E^*$ (MWh)")
        ax.set_title(f"DK1 {year}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if i == 0:
            ax.legend(fontsize=8, loc="best")
        # find first lambda where single and ensemble diverge
        shift_lam = None
        for j, lam in enumerate(lambdas):
            if b_single[j] != b_ens[j]:
                shift_lam = lam
                break
        summary.append({"year": year, "b_single": b_single, "b_ens": b_ens,
                         "shift_lambda": shift_lam})

        # row 2: NPV curves at the break-point lambda (100 EUR/MWh in all 3 yrs)
        ax2 = axes[1, i]
        lam_show = 100.0 if 100.0 in lambdas else lambdas[-1]
        for pol, color, mk in [("single", "C0", "o"), ("ensemble", "C1", "s")]:
            sub = sorted(
                [r for r in rows if r["policy"] == pol and r["lambda"] == lam_show],
                key=lambda r: r["b_E"],
            )
            b_arr = [r["b_E"] for r in sub]
            n_arr = [r["npv"] / 1e6 for r in sub]
            ax2.plot(b_arr, n_arr, "-", marker=mk, color=color,
                      label=f"{pol}")
            # mark argmax
            ai = int(np.argmax(n_arr))
            ax2.axvline(b_arr[ai], ls=":", color=color, alpha=0.5)
        ax2.set_xscale("log")
        ax2.set_xlabel(r"$b_E$ (MWh)")
        if i == 0:
            ax2.set_ylabel("NPV (M)")
        ax2.set_title(rf"DK1 {year}, $\lambda$ = {lam_show:.0f}", fontsize=10)
        ax2.grid(alpha=0.3)
        if i == 0:
            ax2.legend(fontsize=8, loc="best")

    fig.suptitle(r"Imbalance penalty $\lambda$ on wind forecast error: when does "
                  r"$b_E^*$ depend on forecast quality?", y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")
    print("\nImbalance break-point summary:")
    print(f"  {'year':4s}  {'shift λ':>8s}  {'b* single (per λ)':<40s}  ensemble")
    for r in summary:
        print(f"  {r['year']}  "
              f"{str(r['shift_lambda']) if r['shift_lambda'] is not None else 'none':>8s}  "
              f"{str(r['b_single']):<40s}  {r['b_ens']}")
    return summary


if __name__ == "__main__":
    figure_2d_heatmap()
    figure_quantile_vs_persistence()
    figure_slp()
    figure_hydesign()
    figure_imbalance()
