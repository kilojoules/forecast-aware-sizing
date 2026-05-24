#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/q_%A_%a.out
#SBATCH --error=logs/q_%A_%a.err
#SBATCH --job-name=batt_q

# 6 array tasks: (dk1, ercot) x (2021, 2022, 2023)
set -e
cd /scratch/project_465002609/julian/battery_gym
mkdir -p logs results_quantile
SOURCES=(dk1 dk1 dk1 ercot ercot ercot)
YEARS=(2021 2022 2023 2021 2022 2023)
src=${SOURCES[$SLURM_ARRAY_TASK_ID]}
year=${YEARS[$SLURM_ARRAY_TASK_ID]}
pixi run python -u paper_quantile.py --source $src --year $year \
    --out results_quantile/${src}_${year}.json
