"""Run all agents across multiple signal/training seeds; report mean ± std per battery."""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path


SEEDS = [42, 7, 123]
T = 100000


def run_one(seed: int) -> list[dict]:
    out_path = f"_seed{seed}.json"
    cmd = [
        sys.executable, "run.py",
        "--B", "2", "3", "--c", "2", "3", "--d", "2", "3",
        "--T", str(T), "--seed", str(seed),
        "--out", out_path,
    ]
    print(f"[seed={seed}] running...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.load(open(out_path))


def main():
    runs = [run_one(s) for s in SEEDS]
    by_agent: dict[str, list[dict]] = {}
    for run in runs:
        for r in run:
            by_agent.setdefault(r["name"], []).append(r)
    print(f"\nResults across {len(SEEDS)} seeds (mean ± std):\n")
    print(f"{'Agent':<10}  {'Reward':>14}  {'D1':>15}  {'D2':>15}  {'D1+D2':>15}")
    for agent, rs in by_agent.items():
        rwd = [r["reward"] for r in rs]
        d1 = [r["D_per_battery"][0] for r in rs]
        d2 = [r["D_per_battery"][1] for r in rs]
        ds = [r["D_per_battery"][0] + r["D_per_battery"][1] for r in rs]

        def s(xs):
            return f"{statistics.mean(xs):.3f}±{statistics.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"

        print(f"{agent:<10}  {s(rwd):>14}  {s(d1):>15}  {s(d2):>15}  {s(ds):>15}")
    Path("multi_seed_summary.json").write_text(json.dumps(by_agent, indent=2, default=lambda o: list(o) if hasattr(o, "__iter__") else o))


if __name__ == "__main__":
    main()
