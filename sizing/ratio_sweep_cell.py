"""One (wind-scale, year) cell of the lambda*-vs-ratio map.

Run under scripts/memrun.sh to enforce the local <1 GB memory cap:
  scripts/memrun.sh 1000 .pixi/envs/default/bin/python -u \
      sizing/ratio_sweep_cell.py --wind 10 --year 2021
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SIZING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIZING_DIR)

import paper_imbalance as pi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind", type=float, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--b-e", type=float, default=None, dest="b_e",
                    help="run a single b_E and write a partial file "
                         "(memory-bounded mode)")
    ap.add_argument("--merge", action="store_true",
                    help="merge partial files into the cell file")
    ap.add_argument("--outdir", default="results/imbalance/ratio")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"dk1_{args.year}_w{int(args.wind)}.json"
    if fp.exists():
        print(f"skip {fp}")
        return
    part_dir = out / "partial"

    if args.merge:
        rows = []
        for b_E in pi.B_E_GRID:
            pf = part_dir / f"dk1_{args.year}_w{int(args.wind)}_b{float(b_E):g}.json"
            rows.extend(json.load(open(pf))["rows"])
        with open(fp, "w") as f:
            json.dump({"meta": {"year": args.year, "wind_scale_MW": args.wind,
                                "B_E": pi.B_E_GRID, "lambda": pi.LAMBDA_GRID},
                       "rows": rows}, f)
        print(f"Wrote {fp}")
        return

    pi.WIND_SCALE_MW = args.wind
    if args.b_e is not None:
        part_dir.mkdir(parents=True, exist_ok=True)
        pf = part_dir / f"dk1_{args.year}_w{int(args.wind)}_b{float(args.b_e):g}.json"
        if pf.exists():
            print(f"skip {pf}")
            return
        pi.B_E_GRID = [args.b_e]
        rows = pi.run_year(args.year)
        with open(pf, "w") as f:
            json.dump({"rows": rows}, f)
        print(f"Wrote {pf}")
        return

    rows = pi.run_year(args.year)
    with open(fp, "w") as f:
        json.dump({"meta": {"year": args.year, "wind_scale_MW": args.wind,
                            "B_E": pi.B_E_GRID, "lambda": pi.LAMBDA_GRID},
                   "rows": rows}, f)
    print(f"Wrote {fp}")


if __name__ == "__main__":
    main()
