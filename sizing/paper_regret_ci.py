"""Regret + argmax-CI analysis for the merchant invariance result (M2),
plus the replacement-NPV robustness audit (M4).

Addresses two referee objections:
  M2: argmax equality on a doubling b_E grid, with no uncertainty
      quantification, is weak evidence. Remedy: (a) paired block
      bootstrap over 8-week chunks -> distribution of (b_E*_single,
      b_E*_ensemble) and P(equal); (b) plateau regret -- the NPV cost
      of mis-sizing by a factor of 2 -- as the decision-relevant
      magnitude behind argmax statements.
  M4: section 2.4 defines NPV with a replacement term that no reported
      experiment includes. Remedy: recompute argmaxes from stored
      (R, D) with replacements scheduled per the 20% loss-of-health
      budget and report whether invariance survives both NPV forms.

Linear (LP) policies only: the LP pair is the headline comparison and
the QP pair shares its forecasts. ~1200 LP solves, minutes with the
state-variable LP.

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python -u \
         sizing/paper_regret_ci.py --out results/main/regret_ci.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from arbitrage_agents import lp_linear_actions, run_actions
from dk_loader import multi_lag_persistence
from env import PriceEnv
from paper_benchmark import B_E_GRID, B_P, MU_LP, chunk_indices

DISC = sum(1.07 ** -y for y in range(15))
CAPEX_E = 100_000.0
CAPEX_P = 75_000.0
LIFETIME = 15
LOH_BUDGET = 0.2  # replacement when cumulative rainflow D hits 20%
N_BOOT = 2000
SEED = 0


def npv(R_year: float, b_E: float) -> float:
    return R_year * DISC - CAPEX_E * b_E - CAPEX_P * B_P


def repl_cost(D_year: float, b_E: float) -> float:
    """Discounted replacement cost: replace each time cumulative D
    consumes the loss-of-health budget; energy component only."""
    if D_year <= 0:
        return 0.0
    period = LOH_BUDGET / D_year  # years between replacements
    cost, y = 0.0, period
    while y < LIFETIME:
        cost += CAPEX_E * b_E * (1.07 ** -y)
        y += period
    return cost


def chunk_revenues(realized, plan, b_E):
    """Per-chunk linear-reward revenue for one plan trajectory."""
    revs, soc = [], b_E / 2.0
    for s, e in chunk_indices(len(realized)):
        env = PriceEnv(b_E, B_P, prices=realized[s:e], soc0=soc,
                       mu=MU_LP, reward_mode="linear")
        a = lp_linear_actions(plan[s:e], b_E, B_P, soc, mu=MU_LP)
        rec = run_actions(env, a)
        revs.append(rec["R"])
        soc = rec["soc_log"][-1]
    return np.array(revs)


def analyze_regime(source: str, year: int) -> dict:
    if source == "dk1":
        realized, F = multi_lag_persistence(year)
    else:
        from ercot_loader import multi_lag_persistence_ercot
        realized, F = multi_lag_persistence_ercot(year)
    plans = {"single": F[0], "ensemble": F.mean(axis=0)}

    # per-chunk revenue matrix: policy -> (n_bE, n_chunks)
    Rc = {}
    for pol, plan in plans.items():
        rows = []
        for b_E in B_E_GRID:
            rows.append(chunk_revenues(realized, plan, b_E))
        Rc[pol] = np.array(rows)
    n_chunks = Rc["single"].shape[1]
    bE = np.array(B_E_GRID)

    # point estimates
    point = {}
    for pol in plans:
        R_year = Rc[pol].sum(axis=1)
        npvs = np.array([npv(r, b) for r, b in zip(R_year, bE)])
        point[pol] = {"b_E_star": float(bE[npvs.argmax()]),
                      "npv": npvs.tolist()}

    # plateau regret on the ensemble policy: cost of mis-sizing 2x
    ens_npv = np.array(point["ensemble"]["npv"])
    i_star = int(ens_npv.argmax())
    best = ens_npv[i_star]
    regret = {}
    for label, j in (("half", i_star - 1), ("double", i_star + 1)):
        if 0 <= j < len(bE):
            regret[label] = float((best - ens_npv[j]) / abs(best))
    # cross-policy regret: operate ensemble at single's argmax
    j_single = int(np.argmax([npv(r, b) for r, b in
                              zip(Rc["single"].sum(axis=1), bE)]))
    regret["cross_policy"] = float((best - ens_npv[j_single]) / abs(best))

    # paired block bootstrap (same chunk resample for both policies)
    rng = np.random.default_rng(SEED)
    stars = {"single": [], "ensemble": []}
    equal = 0
    boot_regret = []  # NPV cost of adopting single's b_E* under ensemble ops
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_chunks, size=n_chunks)
        b_star, npvs_by = {}, {}
        for pol in plans:
            R_year = Rc[pol][:, idx].sum(axis=1)
            npvs = R_year * DISC - CAPEX_E * bE - CAPEX_P * B_P
            npvs_by[pol] = npvs
            b_star[pol] = bE[npvs.argmax()]
            stars[pol].append(float(b_star[pol]))
        equal += b_star["single"] == b_star["ensemble"]
        ne = npvs_by["ensemble"]
        boot_regret.append(
            float((ne.max() - ne[npvs_by["single"].argmax()]) / abs(ne.max())))
    ci = {}
    for pol in plans:
        v, c = np.unique(stars[pol], return_counts=True)
        ci[pol] = {f"{x:g}": float(n / N_BOOT) for x, n in zip(v, c)}
    br = np.array(boot_regret)

    return {"point": {p: point[p]["b_E_star"] for p in point},
            "p_equal": equal / N_BOOT,
            "argmax_dist": ci,
            "regret": regret,
            "boot_cross_regret": {"mean": float(br.mean()),
                                   "p95": float(np.quantile(br, 0.95)),
                                   "max": float(br.max())}}


def repl_audit() -> dict:
    """M4: argmax with vs without replacement term, from stored (R, D)."""
    out = {}
    files = {("dk1", y): (f"results/main/paper_{y}.json", str(y))
             for y in (2021, 2022, 2023)}
    for y in (2021, 2022, 2023):
        files[("ercot", y)] = ("results/main/paper_ercot.json", str(y))
    for (src, y), (fp, key) in files.items():
        d = json.load(open(fp))["by_year"][key]
        for pol, rows in d.items():
            bE = np.array([r["b_E"] for r in rows])
            R = np.array([r["R"] for r in rows])
            D = np.array([r["D"] for r in rows])
            base = R * DISC - CAPEX_E * bE - CAPEX_P * B_P
            withr = base - np.array([repl_cost(dd, b)
                                     for dd, b in zip(D, bE)])
            out[f"{src}_{y}_{pol}"] = {
                "b_E_star": float(bE[base.argmax()]),
                "b_E_star_repl": float(bE[withr.argmax()]),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    audit = repl_audit()
    shifts = {k: v for k, v in audit.items()
              if v["b_E_star"] != v["b_E_star_repl"]}
    print(f"repl audit: {len(audit)} cells, "
          f"{len(shifts)} argmax shifts when replacement term added")
    for k, v in shifts.items():
        print(f"  {k}: {v['b_E_star']} -> {v['b_E_star_repl']}")

    regimes = {}
    for src in ("dk1", "ercot"):
        for y in (2021, 2022, 2023):
            t0 = time.time()
            r = analyze_regime(src, y)
            regimes[f"{src}_{y}"] = r
            print(f"{src} {y}: point {r['point']}, "
                  f"P(argmax equal)={r['p_equal']:.3f}, "
                  f"regret half/double/cross="
                  f"{r['regret'].get('half', float('nan')):.3%}/"
                  f"{r['regret'].get('double', float('nan')):.3%}/"
                  f"{r['regret']['cross_policy']:.3%}  "
                  f"({time.time()-t0:.0f}s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": {"n_boot": N_BOOT, "seed": SEED,
                            "policies": "linear single/ensemble",
                            "B_E": B_E_GRID},
                   "repl_audit": audit,
                   "regimes": regimes}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
