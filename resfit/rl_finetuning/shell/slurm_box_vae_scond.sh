#!/bin/bash

#SBATCH --job-name=box_vae_scond
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-35
#SBATCH --output=/global/scratch/users/r112358/resfit-logs/%x_%A_%a.out
#SBATCH --error=/global/scratch/users/r112358/resfit-logs/%x_%A_%a.err
#SBATCH --exclude=n0120.savio4,n0121.savio4,n0129.savio4,n0141.savio4,n0142.savio4

# TwoArmBoxCleanup state-conditioned VAE residual. Same grid as the Aloha
# scond sweep, with horizon in place of query_freq.
#   horizon ∈ {10, 20}   (both divide ACT n_action_steps=20)
#   beta    ∈ {1e-2, 1e-3, 1e-4}
#   epochs  ∈ {20, 50, 100}
#   seeds   ∈ {1, 2}
# 2 h × 3 β × 3 epochs × 2 seeds = 36 jobs.
# Index: seed-fast, then epochs, then beta, then horizon.
#   0-17  h=10
#  18-35  h=20
# Gaussian π(z), Q on VAE coords, residual = decoder(z, state)*std.
# State is the query-time proprio (observation.vae_state).

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
set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
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

SEEDS=(1 2)
EPOCHS=(20 50 100)
BETAS=(1e-2 1e-3 1e-4)
HORIZONS=(10 20)
NSEED=${#SEEDS[@]}
NEP=${#EPOCHS[@]}
NBETA=${#BETAS[@]}

g=${SLURM_ARRAY_TASK_ID}
seed_i=$((g % NSEED))
rest=$((g / NSEED))
e_i=$((rest % NEP))
rest=$((rest / NEP))
b_i=$((rest % NBETA))
h_i=$((rest / NBETA))
SEED=${SEEDS[${seed_i}]}
AE_EPOCHS=${EPOCHS[${e_i}]}
BETA=${BETAS[${b_i}]}
HORIZON=${HORIZONS[${h_i}]}

TASK=TwoArmBoxCleanup
CONFIG=residual_td3_box_clean_config
CKPT=/global/scratch/users/r112358/dexmg-act-ckpts/TwoArmBoxCleanup/37745620/policy_step_20000
WANDB_PROJECT=dexmg_box_vae_scond
WANDB_GROUP="${TASK}_vae_scond_k8_h${HORIZON}_ep${AE_EPOCHS}_beta${BETA}"
WANDB_NAME="${WANDB_GROUP}_seed${SEED}"

if [[ ! -f "${CKPT}/policy/config.json" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

echo "Array task ${g}"
echo "  task=${TASK} vae scond k=8 h=${HORIZON} epochs=${AE_EPOCHS} beta=${BETA} squash=gaussian seed=${SEED}"
echo "  ckpt=${CKPT}"
echo "  wandb=${WANDB_PROJECT}/${WANDB_NAME} group=${WANDB_GROUP}"
echo "  host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"

set +e
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
  algo.chunk_residual.vae_condition_on_state=true \
  algo.chunk_residual.vae_beta="${BETA}" \
  algo.chunk_residual.vae_epochs="${AE_EPOCHS}" \
  algo.chunk_residual.pca_fit_episodes=20 \
  agent.actor.action_scale=0.2 \
  agent.actor.squash=gaussian \
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
set -e
if (( rc != 0 )); then
  echo "Failed ${WANDB_NAME} rc=${rc}"
  exit "${rc}"
fi
echo "Finished ${WANDB_NAME}"
