#!/bin/bash

#SBATCH --job-name=box_base_eval
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --exclude=n0139.savio4,n0141.savio4,n0142.savio4

# Base ACT BC success rate on TwoArmBoxCleanup. No residual.
# Optional first argument is n_action_steps (open-loop chunk before replan).
# Defaults to the checkpoint value (20).
#
#   sbatch resfit/rl_finetuning/shell/box_base_eval.sh
#   sbatch --job-name=box_qf10 resfit/rl_finetuning/shell/box_base_eval.sh 10
#   sbatch --job-name=box_qf20 resfit/rl_finetuning/shell/box_base_eval.sh 20

source ~/.bashrc
set -eo pipefail

export CONDA_PREFIX=/global/scratch/users/r112358/miniconda3/envs/residual_rl
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PY="${CONDA_PREFIX}/bin/python"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Avoid ~/robosuite shadowing the editable install.
cd /tmp
REPO=/global/home/users/r112358/residual-offpolicy-rl
export PYTHONPATH="${REPO}:${REPO}/deps/lerobot:${PYTHONPATH:-}"

mkdir -p "${REPO}/logs"
OUT_ROOT=/global/scratch/users/r112358/box_base_eval
mkdir -p "${OUT_ROOT}"

CKPT=/global/scratch/users/r112358/dexmg-act-ckpts/TwoArmBoxCleanup/37745620/policy_step_20000
N_ACTION_STEPS="${1:-}"
RESULT="${OUT_ROOT}/success_rate.txt"
ARGS=(
  --ckpt "${CKPT}"
  --env TwoArmBoxCleanup
  --num_episodes 100
  --no_video
)
if [[ -n "${N_ACTION_STEPS}" ]]; then
  ARGS+=(--n_action_steps "${N_ACTION_STEPS}")
  RESULT="${OUT_ROOT}/success_rate_qf${N_ACTION_STEPS}.txt"
fi
ARGS+=(--result "${RESULT}")

if [[ -n "${CUDA_VISIBLE_DEVICES}" && "${CUDA_VISIBLE_DEVICES}" != *","* ]]; then
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES}"
else
  export MUJOCO_EGL_DEVICE_ID=0
fi

echo "host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset} egl=${MUJOCO_EGL_DEVICE_ID}"
echo "ckpt=${CKPT}"
echo "n_action_steps=${N_ACTION_STEPS:-checkpoint_default} result=${RESULT}"
echo "progress=${RESULT%.txt}.progress"
nvidia-smi -L

"${PY}" -u -m resfit.lerobot.scripts.eval_bc_dexmg "${ARGS[@]}"
