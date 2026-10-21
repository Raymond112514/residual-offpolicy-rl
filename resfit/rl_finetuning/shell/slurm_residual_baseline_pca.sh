#!/bin/bash

#SBATCH --job-name=res_base_pca
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# 2 tasks x 2 methods x 4 seeds = 16 jobs.
# Both methods use the same open-loop ACT chunk residual wrapper.
# Baseline: full flattened chunk residual. PCA: z @ P with k=8.
# Eval is logged at 10k / 20k / ... env steps on every run.
#   0-3   TwoArmBoxCleanup  baseline  seeds 1-4
#   4-7   TwoArmBoxCleanup  pca       seeds 1-4
#   8-11  TwoArmCoffee      baseline  seeds 1-4
#  12-15  TwoArmCoffee      pca       seeds 1-4
#
# Checkpoint: policy_step_20000. No demo mixing. W&B online.
#
#   mkdir -p logs
#   sbatch resfit/rl_finetuning/shell/slurm_residual_baseline_pca.sh

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
NSEED=${#SEEDS[@]}

g=${SLURM_ARRAY_TASK_ID}
task_i=$((g / (2 * NSEED)))
method_i=$(((g / NSEED) % 2))
seed_i=$((g % NSEED))
SEED=${SEEDS[${seed_i}]}

if [[ "${task_i}" -eq 0 ]]; then
  TASK=TwoArmBoxCleanup
  CONFIG=residual_td3_box_clean_config
  CKPT="${CKPT_ROOT}/TwoArmBoxCleanup/37745620/policy_step_20000"
  WANDB_PROJECT=dexmg-box-clean-residual-td3
  EXTRA=(agent.actor_lr=5e-6)
else
  TASK=TwoArmCoffee
  CONFIG=residual_td3_coffee_config
  CKPT="${CKPT_ROOT}/TwoArmCoffee/37745619/policy_step_20000"
  WANDB_PROJECT=dexmg-coffee-residual-td3
  EXTRA=(
    agent.actor_lr=1e-6
    algo.gamma=0.995
    algo.learning_starts=10_000
    algo.critic_warmup_steps=10_000
    algo.num_updates_per_iteration=4
    algo.stddev_max=0.025
    algo.stddev_min=0.025
    algo.buffer_size=300_000
  )
fi

if [[ "${method_i}" -eq 0 ]]; then
  METHOD=baseline
  METHOD_FLAGS=(algo.chunk_residual.enabled=true algo.chunk_residual.use_pca=false)
else
  METHOD=pca
  METHOD_FLAGS=(algo.chunk_residual.enabled=true algo.chunk_residual.use_pca=true algo.chunk_residual.k=8)
fi

WANDB_NAME="${TASK}_${METHOD}_seed${SEED}"

if [[ ! -f "${CKPT}/policy/config.json" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

echo "Array task ${g}"
echo "  task=${TASK} method=${METHOD} seed=${SEED}"
echo "  ckpt=${CKPT}"
echo "  wandb=${WANDB_PROJECT}/${WANDB_NAME}"
echo "  host=$(hostname)"

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
  --config-name="${CONFIG}" \
  base_policy.ckpt_path="${CKPT}" \
  algo.offline_fraction=0.0 \
  algo.prefetch_batches=4 \
  agent.actor.action_scale=0.2 \
  seed="${SEED}" \
  eval_num_envs=1 \
  eval_first=false \
  debug=false \
  wandb.project="${WANDB_PROJECT}" \
  wandb.name="${WANDB_NAME}" \
  wandb.group="${TASK}_${METHOD}" \
  wandb.mode=online \
  "${EXTRA[@]}" \
  "${METHOD_FLAGS[@]}"

echo "Finished ${WANDB_NAME}"
