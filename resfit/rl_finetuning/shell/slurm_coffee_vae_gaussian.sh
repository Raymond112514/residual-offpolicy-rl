#!/bin/bash

#SBATCH --job-name=cof_vae_g
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-35
#SBATCH --output=/global/scratch/users/r112358/resfit-logs/%x_%A_%a.out
#SBATCH --error=/global/scratch/users/r112358/resfit-logs/%x_%A_%a.err

# TwoArmCoffee VAE residual, unsquashed Gaussian π(z).
#   horizon ∈ {20, 10, 5}
#   beta    ∈ {1e-2, 1e-3, 1e-4}
#   epochs  ∈ {20, 100}
#   seeds   ∈ {1, 2}
# 3 h × 3 β × 2 epochs × 2 seeds = 36 jobs.
# Index: seed-fast, then beta, then epochs, then horizon.
#   0-11  h=20
#  12-23  h=10
#  24-35  h=5
# Split: --array=0-17 high, --array=18-35 normal.
#
#   sbatch --qos=rail_gpu4_high   --array=0-17 --job-name=cof_vae_h \
#     --exclude=n0142.savio4 resfit/rl_finetuning/shell/slurm_coffee_vae_gaussian.sh
#   sbatch --qos=rail_gpu4_normal --array=18-35 --job-name=cof_vae_n \
#     --exclude=n0142.savio4 resfit/rl_finetuning/shell/slurm_coffee_vae_gaussian.sh

CONDA_ROOT="${CONDA_ROOT:-/global/scratch/users/r112358/miniconda3}"
ENV_PREFIX="${CONDA_ROOT}/envs/residual_rl"
if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  echo "Missing env python: ${ENV_PREFIX}/bin/python" >&2
  exit 1
fi
export PATH="${ENV_PREFIX}/bin:${CONDA_ROOT}/condabin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
export CONDA_DEFAULT_ENV=residual_rl
hash -r
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

mkdir -p logs "${WANDB_DIR}" "${CACHE_DIR}" /global/scratch/users/r112358/resfit-logs

SEEDS=(1 2)
EPOCHS=(20 100)
BETAS=(1e-2 1e-3 1e-4)
HORIZONS=(20 10 5)
NSEED=${#SEEDS[@]}
NBETA=${#BETAS[@]}
NEP=${#EPOCHS[@]}

g=${SLURM_ARRAY_TASK_ID}
seed_i=$((g % NSEED))
rest=$((g / NSEED))
b_i=$((rest % NBETA))
rest=$((rest / NBETA))
e_i=$((rest % NEP))
h_i=$((rest / NEP))
SEED=${SEEDS[${seed_i}]}
AE_EPOCHS=${EPOCHS[${e_i}]}
BETA=${BETAS[${b_i}]}
HORIZON=${HORIZONS[${h_i}]}

TASK=TwoArmCoffee
CONFIG=residual_td3_coffee_config
CKPT=/global/scratch/users/r112358/dexmg-act-ckpts/TwoArmCoffee/37745619/policy_step_20000
WANDB_PROJECT=dexmg-coffee-residual-td3
WANDB_GROUP="${TASK}_vae_k8_h${HORIZON}_ep${AE_EPOCHS}_beta${BETA}_gaussian"
WANDB_NAME="${WANDB_GROUP}_seed${SEED}"

if [[ ! -f "${CKPT}/policy/config.json" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

echo "Array task ${g}"
echo "  task=${TASK} vae k=8 h=${HORIZON} epochs=${AE_EPOCHS} beta=${BETA} squash=gaussian seed=${SEED}"
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
  algo.chunk_residual.use_vae=true \
  algo.chunk_residual.use_pca=false \
  algo.chunk_residual.k=8 \
  algo.chunk_residual.horizon="${HORIZON}" \
  algo.chunk_residual.vae=true \
  algo.chunk_residual.vae_beta="${BETA}" \
  algo.chunk_residual.vae_epochs="${AE_EPOCHS}" \
  algo.chunk_residual.pca_fit_episodes=20 \
  agent.actor.action_scale=0.2 \
  agent.actor.squash=gaussian \
  agent.actor_lr=1e-6 \
  algo.gamma=0.995 \
  algo.learning_starts=10_000 \
  algo.critic_warmup_steps=10_000 \
  algo.num_updates_per_iteration=4 \
  algo.stddev_max=0.025 \
  algo.stddev_min=0.025 \
  algo.buffer_size=300_000 \
  seed="${SEED}" \
  eval_num_envs=1 \
  eval_first=false \
  debug=false \
  wandb.project="${WANDB_PROJECT}" \
  wandb.name="${WANDB_NAME}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.mode=online

echo "Finished ${WANDB_NAME}"
