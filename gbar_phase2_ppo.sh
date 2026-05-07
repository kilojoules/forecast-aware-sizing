#!/bin/bash
#BSUB -J battery_phase2_ppo
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 4:00
#BSUB -o phase2_ppo_%J.out
#BSUB -e phase2_ppo_%J.err

# GBAR submission for phase 2 PPO training. Uses GBAR module system + a
# project-local venv (no pixi -- pixi is not installed on gbar).
#
# Usage:
#   bsub < gbar_phase2_ppo.sh

set -e
echo "[gbar] starting phase2 PPO at $(date)"
echo "[gbar] hostname:  $(hostname)"
echo "[gbar] cwd:       $(pwd)"
echo "[gbar] cores:     ${LSB_DJOB_NUMPROC:-?}"

# 1. Bring up the module system + load python
source /etc/profile 2>/dev/null || true
[ -f ~/.bashrc ] && source ~/.bashrc 2>/dev/null || true
module load python3/3.11.9

# 2. Build / reuse a venv in the project dir
VENV="$HOME/projects/battery_gym/.venv311"
if [ ! -d "$VENV" ]; then
    echo "[gbar] creating venv at $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet \
        'numpy<2' matplotlib scipy fatpack cvxpy
    "$VENV/bin/pip" install --quiet \
        torch --index-url https://download.pytorch.org/whl/cpu
    "$VENV/bin/pip" install --quiet stable-baselines3 gymnasium
    echo "[gbar] venv ready"
fi
source "$VENV/bin/activate"
python3 -c "import numpy, scipy, cvxpy, torch, stable_baselines3, gymnasium; print('imports OK')"

# 3. Run training + benchmark
python3 -u phase2_ppo.py \
    --total_timesteps 1000000 \
    --n_envs 4 \
    --T 168 \
    --lookahead 72 \
    --noise_train 8.0 \
    --alpha 0.005 \
    --b_E 2.0 \
    --b_P 2.0 \
    --seed 42 \
    --save ppo_policy.zip \
    --log_dir ppo_logs \
    --n_eval 50

echo "[gbar] phase2 PPO done at $(date)"
