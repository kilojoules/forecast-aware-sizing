#!/bin/bash
# Audit the workshop-paper plan state.
# Emits one line per detected change in repo state. Used by Monitor.

set -u

cd "$(dirname "$0")"

prev_hash=""
while true; do
    artifacts=$(
        ls -la data/ercot/*.csv 2>/dev/null | wc -l | tr -d ' '
        echo -n " "
        ls -la spectrum.py 2>/dev/null | wc -l | tr -d ' '
        echo -n " "
        ls -la b_sat_classifier.py 2>/dev/null | wc -l | tr -d ' '
        echo -n " "
        ls -la paper_benchmark.py 2>/dev/null | wc -l | tr -d ' '
        echo -n " "
        ls -la fig_bsat_*.png 2>/dev/null | wc -l | tr -d ' '
        echo -n " "
        ls -la fig_ercot_*.png 2>/dev/null | wc -l | tr -d ' '
        echo -n " "
        ls -la paper.pdf 2>/dev/null | wc -l | tr -d ' '
    )
    last_commit=$(git log -1 --format=%h-%s 2>/dev/null | head -c 60)
    plan_unchecked=$(grep -c '^- \[ \]' PLAN.md 2>/dev/null || echo 0)
    plan_checked=$(grep -c '^- \[x\]' PLAN.md 2>/dev/null || echo 0)
    blocker_count=$(ls BLOCKER_*.md 2>/dev/null | wc -l | tr -d ' ')

    cur_hash="$artifacts | $last_commit | done=$plan_checked todo=$plan_unchecked block=$blocker_count"
    if [ "$cur_hash" != "$prev_hash" ]; then
        echo "$(date +%H:%M:%S) | $cur_hash"
        prev_hash="$cur_hash"
    fi
    sleep 300
done
