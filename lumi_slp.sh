#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/slp_%A_%a.out
#SBATCH --error=logs/slp_%A_%a.err
#SBATCH --job-name=batt_slp

# 6 array tasks: (dk1, ercot) x (2021, 2022, 2023). SLP is heavy.
set -e
cd /scratch/project_465002609/julian/battery_gym
mkdir -p logs results_slp
SOURCES=(dk1 dk1 dk1 ercot ercot ercot)
YEARS=(2021 2022 2023 2021 2022 2023)
src=${SOURCES[$SLURM_ARRAY_TASK_ID]}
year=${YEARS[$SLURM_ARRAY_TASK_ID]}
pixi run python -u paper_slp.py --source $src --year $year \
    --out results_slp/${src}_${year}.json
