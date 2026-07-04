"""#1 Continuous argmax + equivalence (TOST) test.

The headline 'same size' has been argmax equality on a near-doubling
b_E grid, and DK1 2022's 8-vs-4 call turns on a 0.28% NPV edge -- one
grid step from flipping. This replaces the coarse-grid argmax with a
continuous one (parabola-vertex on a dense b_E grid, the standard
sub-grid-precision peak estimate) and wraps the invariance claim in a
two-one-sided-tests (TOST) equivalence test on the continuous capacity
gap delta = b_E*_single - b_E*_ensemble, against a decision-relevant
margin: the +-b_E offset that costs 1% of optimal NPV.

Output: results/main/continuous_argmax.json + a figure. Linear single
vs ensemble (the headline pair), all six market-years, paired block
bootstrap over the eight-week chunks.

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python -u \
         sizing/continuous_argmax.py --out results/main/continuous_argmax.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from dk_loader import multi_lag_persistence
from ercot_loader import multi_lag_persistence_ercot
from paper_regret_ci import chunk_revenues, DISC, CAPEX_E as CE, CAPEX_P as CP, B_P  # noqa: E402

DENSE_BE = np.geomspace(0.5, 96.0, 46)     # dense, ~geometric
N_BOOT = 2000
SEED = 0
NPV_MARGIN = 0.01                          # equivalence margin = 1% of optimal NPV


def npv_curve(chunk_rev_by_b, idx=None):
    """chunk_rev_by_b: (n_b, n_chunks). Return NPV(b) vector."""
    R = (chunk_rev_by_b[:, idx].sum(1) if idx is not None
         else chunk_rev_by_b.sum(1))
    return R * DISC - CE * DENSE_BE - CP


def continuous_peak(npv):
    """Sub-grid argmax + local curvature via 3-point parabola in log(b_E).
    Returns (b_star, npv_star, curvature_H_wrt_logb)."""
    i = int(np.argmax(npv))
    if i == 0 or i == len(npv) - 1:
        return float(DENSE_BE[i]), float(npv[i]), None   # boundary
    x = np.log(DENSE_BE[i - 1:i + 2])
    y = npv[i - 1:i + 2]
    # parabola y = a x^2 + b x + c
    a, b, _ = np.polyfit(x, y, 2)
    if a >= 0:
        return float(DENSE_BE[i]), float(npv[i]), None
    xstar = -b / (2 * a)
    ystar = np.polyval([a, b, _], xstar)
    return float(np.exp(xstar)), float(ystar), float(a)  # H (d2/dlogb2) = 2a


def margin_be(b_star, npv_star, H):
    """b_E offset (MWh) that costs NPV_MARGIN of optimal NPV.
    NPV ~ npv_star + H*(logb - logb*)^2 (H<0). Solve for |dlogb|, map to db."""
    if H is None or H >= 0:
        return float("nan")
    dlogb = np.sqrt(NPV_MARGIN * abs(npv_star) / abs(H))
    # symmetric-ish in MWh: average the +/- excursions
    return float(0.5 * (b_star * np.exp(dlogb) - b_star * np.exp(-dlogb)))


def analyze(source, year):
    realized, F = (multi_lag_persistence(year) if source == "dk1"
                   else multi_lag_persistence_ercot(year))
    plans = {"single": F[0], "ensemble": F.mean(axis=0)}
    # per-chunk revenue at each dense b_E, per policy
    Rc = {}
    for name, plan in plans.items():
        Rc[name] = np.array([chunk_revenues(realized, plan, b) for b in DENSE_BE])
    n_chunks = Rc["single"].shape[1]

    point = {}
    for name in plans:
        bstar, nstar, H = continuous_peak(npv_curve(Rc[name]))
        point[name] = {"b_star": bstar, "npv_star": nstar,
                       "margin_be": margin_be(bstar, nstar, H),
                       "boundary": H is None}
    delta0 = point["single"]["b_star"] - point["ensemble"]["b_star"]
    # equivalence margin = the tighter of the two policies' 1%-NPV margins
    margins = [point[p]["margin_be"] for p in plans
               if not np.isnan(point[p]["margin_be"])]
    delta_margin = float(min(margins)) if margins else float("nan")

    rng = np.random.default_rng(SEED)
    deltas = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = rng.integers(0, n_chunks, size=n_chunks)
        bs, _, _ = continuous_peak(npv_curve(Rc["single"], idx))
        be, _, _ = continuous_peak(npv_curve(Rc["ensemble"], idx))
        deltas[k] = bs - be
    lo, hi = np.quantile(deltas, [0.05, 0.95])   # 90% CI for TOST at 5%
    tost_equiv = bool(hi < delta_margin and lo > -delta_margin)
    return {
        "point": point, "delta_point": float(delta0),
        "delta_margin_be": delta_margin,
        "delta_ci90": [float(lo), float(hi)],
        "delta_mean": float(deltas.mean()), "delta_std": float(deltas.std()),
        "tost_equivalent": tost_equiv,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}
    for src in ("dk1", "ercot"):
        for y in (2021, 2022, 2023):
            t0 = time.time()
            r = analyze(src, y)
            res[f"{src}_{y}"] = r
            p = r["point"]
            print(f"{src} {y}: b* single={p['single']['b_star']:.2f} "
                  f"ens={p['ensemble']['b_star']:.2f}  "
                  f"delta={r['delta_point']:+.2f} MWh  margin=+-{r['delta_margin_be']:.2f}  "
                  f"CI90=[{r['delta_ci90'][0]:+.2f},{r['delta_ci90'][1]:+.2f}]  "
                  f"TOST-equivalent={r['tost_equivalent']}  ({time.time()-t0:.0f}s)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"dense_be": DENSE_BE.tolist(), "n_boot": N_BOOT,
                        "npv_margin_frac": NPV_MARGIN}, "regimes": res},
              open(args.out, "w"), indent=1)
    n_eq = sum(v["tost_equivalent"] for v in res.values())
    print(f"Wrote {args.out}  ({n_eq}/6 TOST-equivalent within 1%-NPV margin)")


if __name__ == "__main__":
    main()
