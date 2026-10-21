#!/bin/bash

#SBATCH --job-name=qres_box
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# TwoArmBoxCleanup only. 2 methods x 4 horizons x 4 seeds = 32 jobs.
# Q is trained on the actor residual (algo.critic_on_residual=true).
# Baseline: full flattened chunk residual. PCA: z @ P with k=8.
#   0-3    baseline  h=1   seeds 1-4
#   4-7    baseline  h=4   seeds 1-4
#   8-11   baseline  h=10  seeds 1-4
#  12-15   baseline  h=20  seeds 1-4
#  16-19   pca       h=1   seeds 1-4
#  20-23   pca       h=4   seeds 1-4
#  24-27   pca       h=10  seeds 1-4
#  28-31   pca       h=20  seeds 1-4
#
#   mkdir -p logs
#   sbatch resfit/rl_finetuning/shell/slurm_qres_baseline_pca_box.sh

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

CKPT_ROOT=/global/scratch/users/r112358/dexmg-act-ckpts
SEEDS=(1 2 3 4)
HORIZONS=(1 4 10 20)
NSEED=${#SEEDS[@]}
NH=${#HORIZONS[@]}

g=${SLURM_ARRAY_TASK_ID}
seed_i=$((g % NSEED))
h_i=$(((g / NSEED) % NH))
method_i=$((g / (NSEED * NH)))
SEED=${SEEDS[${seed_i}]}
HORIZON=${HORIZONS[${h_i}]}

TASK=TwoArmBoxCleanup
CONFIG=residual_td3_box_clean_config
CKPT="${CKPT_ROOT}/TwoArmBoxCleanup/37745620/policy_step_20000"
WANDB_PROJECT=dexmg-box-clean-residual-td3

if [[ "${method_i}" -eq 0 ]]; then
  METHOD=baseline
  METHOD_FLAGS=(algo.chunk_residual.enabled=true algo.chunk_residual.use_pca=false)
else
  METHOD=pca
  METHOD_FLAGS=(algo.chunk_residual.enabled=true algo.chunk_residual.use_pca=true algo.chunk_residual.k=8)
fi

WANDB_GROUP="${TASK}_${METHOD}_qres_h${HORIZON}"
WANDB_NAME="${WANDB_GROUP}_seed${SEED}"

if [[ ! -f "${CKPT}/policy/config.json" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

echo "Array task ${g}"
echo "  task=${TASK} method=${METHOD} q=residual horizon=${HORIZON} seed=${SEED}"
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
  wandb.mode=online \
  "${METHOD_FLAGS[@]}"

echo "Finished ${WANDB_NAME}"
