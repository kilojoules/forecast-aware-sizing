#!/bin/bash
#BSUB -J imbfix[1-21]
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 2:00
#BSUB -o imbfix_%J_%I.out
#BSUB -e imbfix_%J_%I.err

# Regenerate the full imbalance program with corrected settlement
# accounting (p_dot_r: residual energy valued at DA). 21 cells:
#   1-3   paper_imbalance W=5        (results/imbalance/dk1_Y.json)
#   4-12  ratio W in {2,10,20}       (results/imbalance/ratio/...)
#   13-15 real eSett settlement      (results/imbalance/dk1_Y_real.json)
#   16-18 settlement-aware + gamma   (results/imbalance/aware_dk1_Y.json)
#   19-21 quantile bidding           (results/imbalance/qbid_dk1_Y.json)
# Requires .venv311aware to exist (built by settlement_aware.sh).
# Submit: bsub < scripts/gbar/imbalance_rerun.sh

set -e
source /etc/profile 2>/dev/null || true
module load python3/3.11.9
cd "$HOME/projects/battery_gym"
PY=".venv311aware/bin/python"

I=$LSB_JOBINDEX
Y=$((2021 + (I - 1) % 3))
case $(( (I - 1) / 3 )) in
  0) $PY -u sizing/paper_imbalance.py --year $Y --out results/imbalance/dk1_${Y}.json ;;
  1) $PY -u sizing/ratio_sweep_cell.py --wind 2  --year $Y ;;
  2) $PY -u sizing/ratio_sweep_cell.py --wind 10 --year $Y ;;
  3) $PY -u sizing/ratio_sweep_cell.py --wind 20 --year $Y ;;
  4) $PY -u sizing/paper_real_imbalance.py --year $Y --out results/imbalance/dk1_${Y}_real.json ;;
  5) $PY -u sizing/paper_settlement_aware.py --year $Y --out results/imbalance/aware_dk1_${Y}.json ;;
  6) $PY -u sizing/paper_quantile_bid.py --year $Y --out results/imbalance/qbid_dk1_${Y}.json ;;
esac
echo "[imbfix] index $I (year $Y) done"
