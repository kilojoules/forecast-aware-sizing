#!/bin/bash -l
#SBATCH --job-name=battery_gym
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-9%2

# CPU-bound numpy code; no GPU needed.
# Array indices map to (B1,B2,c1,c2,d1,d2,seed) tuples below.

set -euo pipefail
mkdir -p logs results checkpoints

cd $SLURM_SUBMIT_DIR

# Resolve pixi (binary lives in user home on LUMI)
if ! command -v pixi >/dev/null 2>&1; then
    if [ -x "$HOME/.pixi/bin/pixi" ]; then
        export PATH="$HOME/.pixi/bin:$PATH"
    fi
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Heterogeneous configs (where Greedy beats Naive 25-50%):
#  - small/large mix at multiple scales
#  - 2 seeds per config
CONFIGS=(
  "2 20  2 10  2 10  42"
  "2 20  2 10  2 10  7"
  "5 20  2 10  2 10  42"
  "5 20  2 10  2 10  7"
  "2 50  2 25  2 25  42"
  "2 50  2 25  2 25  7"
  "5 50  2 25  2 25  42"
  "5 50  2 25  2 25  7"
  "10 100 5 50 5 50  42"
  "10 100 5 50 5 50  7"
)

read B1 B2 c1 c2 d1 d2 SEED <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

T=${T:-1000000}
HIDDEN=${HIDDEN:-200}
LR=${LR:-0.01}
EPS_DECAY=${EPS_DECAY:-1e-5}

echo "Task $SLURM_ARRAY_TASK_ID: B=($B1,$B2) c=($c1,$c2) d=($d1,$d2) T=$T hidden=$HIDDEN seed=$SEED"

pixi run python rl_elm/run.py \
    --B $B1 $B2 --c $c1 $c2 --d $d1 $d2 \
    --T $T \
    --seed $SEED \
    --reward quad_growth \
    --elm-hidden $HIDDEN \
    --elm-lr $LR \
    --elm-eps-decay $EPS_DECAY \
    --skip-tabular \
    --log-every 100000 \
    --checkpoint checkpoints/het_${SLURM_ARRAY_TASK_ID}.npz \
    --checkpoint-every 200000 \
    --out results/het_${SLURM_ARRAY_TASK_ID}_seed${SEED}.json
