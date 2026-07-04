"""#5 Effective-sample-size (n_eff) data-sufficiency diagnostic.

The invariance is weakest exactly where the data are thinnest -- ERCOT
2021/Uri (mean bootstrap regret 10%, p95 47%) is effectively one spike
episode. This ships a second practitioner check beside the b_sat test:
n_eff, the effective number of independent earning episodes in a
market-year, from the participation ratio of daily arbitrage revenue

    n_eff = (sum_d w_d)^2 / sum_d w_d^2 ,  w_d = max(0, daily revenue)

n_eff is ~365 when value is spread across the year and small when it
concentrates in a few spike days. We show sizing fragility (bootstrap
regret / P(argmax shift), from results/main/regret_ci.json) scales with
1/n_eff, so ERCOT 2021 flags 'thin' as a PREDICTED consequence of low
sample rather than an unexplained exception. Verdict thresholds:
n_eff >= 60 sufficient, 20-60 thin, < 20 insufficient.

Output: figure + printed table.

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python sizing/n_eff_diag.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SIZING_DIR = Path(__file__).resolve().parent
RESULTS = SIZING_DIR.parent / "results"
FIGURES = SIZING_DIR.parent / "paper" / "figures"
sys.path.insert(0, str(SIZING_DIR))

import matplotlib.pyplot as plt
import numpy as np

from arbitrage_agents import lp_linear_actions
from dk_loader import multi_lag_persistence
from ercot_loader import multi_lag_persistence_ercot
from paper_benchmark import B_P, MU_LP, chunk_indices

REF_BE = 8.0
SUFFICIENT, THIN = 60.0, 20.0


def daily_revenue(realized, plan, b_E):
    acts, soc = [], b_E / 2.0
    for s, e in chunk_indices(len(realized)):
        a = lp_linear_actions(plan[s:e], b_E, B_P, soc, mu=MU_LP)
        acts.append(a)
        soc = max(0.0, min(b_E, soc - float(np.sum(a))))
    a = np.concatenate(acts)
    rev_h = realized * a
    T = len(rev_h) - (len(rev_h) % 24)
    return rev_h[:T].reshape(-1, 24).sum(axis=1)   # per-day net revenue


def n_eff(daily):
    w = np.maximum(daily, 0.0)                      # earning days
    return float(w.sum() ** 2 / np.sum(w ** 2)) if w.sum() > 0 else 0.0


def verdict(n):
    return "sufficient" if n >= SUFFICIENT else ("thin" if n >= THIN else "insufficient")


def main(out: Path):
    reg = json.load(open(RESULTS / "main" / "regret_ci.json"))["regimes"]
    rows = []
    for src in ("dk1", "ercot"):
        for y in (2021, 2022, 2023):
            realized, F = (multi_lag_persistence(y) if src == "dk1"
                           else multi_lag_persistence_ercot(y))
            d = daily_revenue(realized, F.mean(axis=0), REF_BE)
            ne = n_eff(d)
            key = f"{src}_{y}"
            br = reg[key]["boot_cross_regret"]
            rows.append({"regime": key, "n_eff": ne, "verdict": verdict(ne),
                         "regret_mean": br["mean"], "regret_p95": br["p95"],
                         "p_equal": reg[key]["p_equal"]})
            print(f"  {key:11s} n_eff={ne:6.1f} [{verdict(ne):11s}]  "
                  f"regret mean={br['mean']:.1%} p95={br['p95']:.1%}  "
                  f"P(argmax eq)={reg[key]['p_equal']:.2f}")

    # correlation check: 1/n_eff vs regret p95
    inv = np.array([1.0 / r["n_eff"] for r in rows])
    p95 = np.array([r["regret_p95"] for r in rows])
    rho = float(np.corrcoef(inv, p95)[0, 1])
    print(f"  corr(1/n_eff, regret_p95) = {rho:+.2f}")

    fig, ax = plt.subplots(figsize=(7.4, 5))
    XLO, XHI = 5.0, 400.0
    ax.axvspan(XLO, THIN, color="#d62728", alpha=0.07)
    ax.axvspan(THIN, SUFFICIENT, color="#ff7f0e", alpha=0.07)
    ax.axvspan(SUFFICIENT, XHI, color="#2ca02c", alpha=0.07)
    dodge = {"ercot_2023": (6, -12), "ercot_2022": (6, 8), "dk1_2023": (6, -6)}
    for r in rows:
        c = {"sufficient": "#2ca02c", "thin": "#ff7f0e",
             "insufficient": "#d62728"}[r["verdict"]]
        ax.scatter(r["n_eff"], r["regret_p95"] * 100, s=90, color=c, zorder=3)
        dx, dy = dodge.get(r["regime"], (6, 4))
        ax.annotate(r["regime"].replace("_", " "), (r["n_eff"], r["regret_p95"] * 100),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(XLO, XHI)
    ax.set_xlabel("effective earning episodes  $n_{\\mathrm{eff}}$  "
                  "(participation ratio of daily revenue)", fontsize=10)
    ax.set_ylabel("sizing fragility: bootstrap regret p95 (% of NPV)", fontsize=10)
    ax.set_title("Fewer independent earning episodes → more fragile sizing\n"
                 "ERCOT 2021 (Storm Uri) flags 'thin' as a prediction, not an excuse "
                 f"(corr(1/$n_{{eff}}$, regret)={rho:+.2f})", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    json.dump({"ref_b_E": REF_BE, "thresholds": {"sufficient": SUFFICIENT, "thin": THIN},
               "rows": rows, "corr_inv_neff_regret": rho},
              open(RESULTS / "main" / "n_eff.json", "w"), indent=1)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    main(FIGURES / "fig_n_eff.png")
