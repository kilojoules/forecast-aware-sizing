#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-29%2
#SBATCH --output=logs/2d_%A_%a.out
#SBATCH --error=logs/2d_%A_%a.err
#SBATCH --job-name=batt_2d

# 2-D (b_E, b_P) heatmap. 30 array tasks, one per (b_P, b_E) pair.
# Each task runs 2 markets * 3 years * 4 policies on its (b_E, b_P).

set -e
cd /scratch/project_465002609/julian/battery_gym
mkdir -p logs results_2d
pixi run python -u paper_2d_task.py --task_id ${SLURM_ARRAY_TASK_ID} \
    --out_dir results_2d
