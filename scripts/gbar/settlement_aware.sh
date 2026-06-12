#!/bin/bash
#BSUB -J aware[1-3]
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 2:00
#BSUB -o aware_%J_%I.out
#BSUB -e aware_%J_%I.err

# Settlement-aware + wind-error-inflation sweep (referee M5/M6).
# Job array: index 1->2021, 2->2022, 3->2023.
# Submit from the repo root on gbar:  bsub < scripts/gbar/settlement_aware.sh

set -e
source /etc/profile 2>/dev/null || true
module load python3/3.11.9

cd "$HOME/projects/battery_gym"
VENV=".venv311aware"
if [ ! -f "$VENV/ok" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" -q install numpy "scipy>=1.13" "pandas>=2" requests fatpack
    touch "$VENV/ok"
fi

YEAR=$((2020 + LSB_JOBINDEX))
echo "[aware] year $YEAR on $(hostname)"
"$VENV/bin/python" -u sizing/paper_settlement_aware.py --year "$YEAR" \
    --out "results/imbalance/aware_dk1_${YEAR}.json"
