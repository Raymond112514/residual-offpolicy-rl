#!/bin/bash

#SBATCH --job-name=pca_sub_box
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# TwoArmBoxCleanup. Full residual actor + PCA-subspace regularizer.
# 6 weights x 2 seeds = 12 jobs. Horizon=20, k=8, Q on residual.
#   0-1    w=0.1  seeds 1-2
#   2-3    w=0.5  seeds 1-2
#   4-5    w=1.0  seeds 1-2
#   6-7    w=2.0  seeds 1-2
#   8-9    w=3.0  seeds 1-2
#  10-11   w=5.0  seeds 1-2
#
#   mkdir -p logs
#   sbatch resfit/rl_finetuning/shell/slurm_pca_subspace_reg_box.sh

source ~/.bashrc
conda activate residual_rl
set -eo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export WANDB_DIR=/global/scratch/users/r112358/wandb
export CACHE_DIR=/global/scratch/users/r112358/resfit-cache

cd /global/home/users/r112358/residual-offpolicy-rl
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/deps/lerobot:${PYTHONPATH:-}"

mkdir -p logs "${WANDB_DIR}" "${CACHE_DIR}"

SEEDS=(1 2)
WEIGHTS=(0.1 0.5 1.0 2.0 3.0 5.0)
NSEED=${#SEEDS[@]}

g=${SLURM_ARRAY_TASK_ID}
seed_i=$((g % NSEED))
w_i=$((g / NSEED))
SEED=${SEEDS[${seed_i}]}
WEIGHT=${WEIGHTS[${w_i}]}
WEIGHT_TAG=${WEIGHT/./p}

TASK=TwoArmBoxCleanup
CONFIG=residual_td3_box_clean_config
CKPT=/global/scratch/users/r112358/dexmg-act-ckpts/TwoArmBoxCleanup/37745620/policy_step_20000
WANDB_PROJECT=dexmg-box-clean-residual-td3
WANDB_GROUP="${TASK}_pca_sub_w${WEIGHT_TAG}"
WANDB_NAME="${WANDB_GROUP}_seed${SEED}"

if [[ ! -f "${CKPT}/policy/config.json" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

echo "Array task ${g}"
echo "  task=${TASK} pca_subspace_reg_weight=${WEIGHT} seed=${SEED}"
echo "  ckpt=${CKPT}"
echo "  wandb=${WANDB_PROJECT}/${WANDB_NAME} group=${WANDB_GROUP}"
echo "  host=$(hostname)"

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
  --config-name="${CONFIG}" \
  base_policy.ckpt_path="${CKPT}" \
  algo.offline_fraction=0.0 \
  algo.prefetch_batches=4 \
  algo.total_timesteps=200_000 \
  algo.critic_on_residual=true \
  algo.chunk_residual.enabled=true \
  algo.chunk_residual.use_pca=false \
  algo.chunk_residual.k=8 \
  algo.chunk_residual.horizon=20 \
  algo.chunk_residual.pca_subspace_reg_weight="${WEIGHT}" \
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

echo "Finished ${WANDB_NAME}"
