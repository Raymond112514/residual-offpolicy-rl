#!/bin/bash

#SBATCH --job-name=box_pca_h10
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=/global/scratch/users/r112358/resfit-logs/%x_%A_%a.out
#SBATCH --error=/global/scratch/users/r112358/resfit-logs/%x_%A_%a.err
#SBATCH --exclude=n0120.savio4,n0121.savio4,n0129.savio4,n0141.savio4,n0142.savio4

# Finish TwoArmBoxCleanup PCA h=10. Seed 1 is already logged in
# dexmg-box-clean-residual-td3 / TwoArmBoxCleanup_pca_h10.
#   0  seed 2
#   1  seed 3
#   2  seed 4
# Same flags as slurm_pca_h20.sh array tasks 9-11. Q stays on the executed action.
# n0120 and n0121 died in the first second on the previous submit (39687718).
#
#   cd /global/home/users/r112358/residual-offpolicy-rl
#   sbatch resfit/rl_finetuning/shell/slurm_pca_h10_box_missing.sh

source ~/.bashrc
set -euo pipefail

# Call the env interpreter directly. `conda activate` falls through to the
# base install on some nodes.
unset PYTHONHOME
export CONDA_PREFIX=/global/scratch/users/r112358/miniconda3/envs/residual_rl
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PY="${CONDA_PREFIX}/bin/python"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_DIR=/global/scratch/users/r112358/wandb
export CACHE_DIR=/global/scratch/users/r112358/resfit-cache

cd /global/home/users/r112358/residual-offpolicy-rl
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/deps/lerobot:${PYTHONPATH:-}"

mkdir -p logs "${WANDB_DIR}" "${CACHE_DIR}" /global/scratch/users/r112358/resfit-logs

SEEDS=(2 3 4)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
HORIZON=10

CONFIG=residual_td3_box_clean_config
CKPT=/global/scratch/users/r112358/dexmg-act-ckpts/TwoArmBoxCleanup/37745620/policy_step_20000
WANDB_PROJECT=dexmg-box-clean-residual-td3
WANDB_GROUP=TwoArmBoxCleanup_pca_h10
WANDB_NAME="${WANDB_GROUP}_seed${SEED}"

if [[ ! -x "${PY}" ]]; then
  echo "Missing env python: ${PY}" >&2
  exit 1
fi
if [[ ! -f "${CKPT}/policy/config.json" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

echo "Array task ${SLURM_ARRAY_TASK_ID}"
echo "  task=TwoArmBoxCleanup method=pca horizon=${HORIZON} seed=${SEED}"
echo "  ckpt=${CKPT}"
echo "  wandb=${WANDB_PROJECT}/${WANDB_NAME} group=${WANDB_GROUP}"
echo "  host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"

"${PY}" -m resfit.rl_finetuning.scripts.train_residual_td3 \
  --config-name="${CONFIG}" \
  base_policy.ckpt_path="${CKPT}" \
  algo.offline_fraction=0.0 \
  algo.prefetch_batches=4 \
  algo.total_timesteps=200_000 \
  algo.chunk_residual.enabled=true \
  algo.chunk_residual.use_pca=true \
  algo.chunk_residual.k=8 \
  algo.chunk_residual.horizon="${HORIZON}" \
  agent.actor.action_scale=0.2 \
  agent.actor_lr=5e-6 \
  seed="${SEED}" \
  eval_num_envs=1 \
  eval_first=false \
  debug=false \
  wandb.project="${WANDB_PROJECT}" \
  wandb.name="${WANDB_NAME}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.mode=online
rc=$?
if (( rc != 0 )); then
  echo "Training failed status=${rc}" >&2
  exit "${rc}"
fi

echo "Finished ${WANDB_NAME}"
