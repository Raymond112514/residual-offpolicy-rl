#!/bin/bash

#SBATCH --job-name=res_box_0res
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --exclude=n0139.savio4
#SBATCH --output=/global/scratch/users/r112358/resfit-logs/%x_%A_%a.out
#SBATCH --error=/global/scratch/users/r112358/resfit-logs/%x_%A_%a.err

# Baseline with zero warmup residual (random_action_noise_scale=0).
# One run per node. 2 chunks x 2 scales x 2 seeds = 8 normal-queue nodes. UTD=1.
#   0-3  chunk 20, scales 0.05 then 0.1, seeds 101 then 10101
#   4-7  chunk 10, scales 0.05 then 0.1, seeds 101 then 10101
#
#   mkdir -p /global/scratch/users/r112358/resfit-logs
#   sbatch resfit/rl_finetuning/shell/slurm_box_baseline_zero_residual.sh

activate_ok=0
for attempt in 1 2 3 4 5; do
  if source ~/.bashrc && conda activate residual_rl; then
    activate_ok=1
    break
  fi
  echo "conda activate failed (attempt ${attempt}), retrying in 20s"
  sleep 20
done
if [[ "${activate_ok}" -ne 1 ]]; then
  echo "conda activate failed"
  exit 1
fi
set -uo pipefail

LOCAL_ROOT="/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${LOCAL_ROOT}/wandb" "${LOCAL_ROOT}/replay"
export TMPDIR="${LOCAL_ROOT}"
export TEMP="${LOCAL_ROOT}"
export TMP="${LOCAL_ROOT}"
export REPLAY_SCRATCH_DIR="${LOCAL_ROOT}/replay"
export WANDB_DIR="${LOCAL_ROOT}/wandb"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"
export CACHE_DIR=/global/scratch/users/r112358/resfit-cache

cd /global/home/users/r112358/residual-offpolicy-rl
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/deps/lerobot:${PYTHONPATH:-}"

LOG_DIR=/global/scratch/users/r112358/resfit-logs
mkdir -p "${LOG_DIR}" "${WANDB_DIR}" "${CACHE_DIR}"

CKPT_ROOT=/global/scratch/users/r112358/dexmg-act-ckpts
SEEDS=(101 10101)
SCALES=(0.05 0.1)
CHUNKS=(20 10)
K=1
N_UPDATES=$((4 * K))
N_ACTOR=${K}
NSEED=${#SEEDS[@]}
NSCALE=${#SCALES[@]}

g=${SLURM_ARRAY_TASK_ID}
seed_i=$((g % NSEED))
scale_i=$(((g / NSEED) % NSCALE))
chunk_i=$((g / (NSEED * NSCALE)))
SEED=${SEEDS[${seed_i}]}
SCALE=${SCALES[${scale_i}]}
CHUNK=${CHUNKS[${chunk_i}]}

CONFIG=residual_td3_box_clean_config
CKPT="${CKPT_ROOT}/TwoArmBoxCleanup/37745620/policy_step_20000"
WANDB_PROJECT=dexmg_box_baseline_sweep
WANDB_NAME="box_0residual_chunk${CHUNK}_scale${SCALE}_utd${K}_seed${SEED}_binary_reward"

if [[ ! -f "${CKPT}/policy/config.json" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

echo "Array task ${g} on $(hostname)"
echo "  chunk=${CHUNK} action_scale=${SCALE} utd=${K} noise_scale=0 seed=${SEED} reward_discount=false"
echo "  critic_updates=${N_UPDATES} actor_updates=${N_ACTOR} per transition"
echo "  gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  wandb=${WANDB_PROJECT}/${WANDB_NAME}"
echo "  ckpt=${CKPT}"

run_log="${LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_seed${SEED}.log"
echo "  log=${run_log}"
train_ok=0
for attempt in 1 2 3; do
python -m resfit.rl_finetuning.scripts.train_residual_td3 \
  --config-name="${CONFIG}" \
  base_policy.ckpt_path="${CKPT}" \
  algo.action_chunk="${CHUNK}" \
  algo.requery=false \
  algo.offline_fraction=0.0 \
  algo.action_chunk_reward_discount=false \
  algo.random_action_noise_scale=0 \
  algo.total_timesteps=200_000 \
  algo.num_updates_per_iteration="${N_UPDATES}" \
  algo.actor_updates_per_iteration="${N_ACTOR}" \
  agent.actor_lr=5e-6 \
  agent.actor.action_scale="${SCALE}" \
  save_video=false \
  seed="${SEED}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.group="box_0residual_chunk${CHUNK}_scale${SCALE}_utd${K}_binary_reward" \
  wandb.name="${WANDB_NAME}" \
  wandb.mode=online \
  > "${run_log}" 2>&1 && train_ok=1 && break
  echo "train failed (attempt ${attempt}), retrying in 30s" | tee -a "${run_log}"
  sleep 30
done
exit $((1 - train_ok))
