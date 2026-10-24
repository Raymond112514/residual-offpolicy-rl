#!/bin/bash

#SBATCH --job-name=res_box_pca
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-23
#SBATCH --exclude=n0139.savio4
#SBATCH --output=/global/scratch/users/r112358/resfit-logs/%x_%A_%a.out
#SBATCH --error=/global/scratch/users/r112358/resfit-logs/%x_%A_%a.err

# TwoArmBoxCleanup PCA residual, 8 basis, 0/1 chunk reward.
# Same grid as the baseline sweep: 2 chunks x 4 action scales x 3 UTD = 24 jobs.
# Each job uses 1 GPU and runs seeds 101 and 10101 in parallel on it. Chunk 20 starts first.
#   0-11   chunk 20
#  12-23   chunk 10
# UTD critic/actor steps per transition: 4*K critic updates, then K actor updates.
#   action_scale: 0.05, 0.1, 0.2, 0.3
#   K: 1, 2, 4
#   seeds: 101, 10101
#   PCA k: 8
#
#   mkdir -p /global/scratch/users/r112358/resfit-logs
#   sbatch resfit/rl_finetuning/shell/slurm_box_pca_sweep.sh

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
if [[ -n "${ONLY_SEED:-}" ]]; then
  SEEDS=("${ONLY_SEED}")
fi
CHUNKS=(20 10)
SCALES=(0.05 0.1 0.2 0.3)
KS=(1 2 4)
PCA_K=8
NK=${#KS[@]}
NSCALE=${#SCALES[@]}

g=${SLURM_ARRAY_TASK_ID}
k_i=$((g % NK))
scale_i=$(((g / NK) % NSCALE))
chunk_i=$((g / (NK * NSCALE)))
K=${KS[${k_i}]}
SCALE=${SCALES[${scale_i}]}
CHUNK=${CHUNKS[${chunk_i}]}
N_UPDATES=$((4 * K))
N_ACTOR=${K}

CONFIG=residual_td3_box_clean_config
CKPT="${CKPT_ROOT}/TwoArmBoxCleanup/37745620/policy_step_20000"
WANDB_PROJECT=dexmg_box_pca_sweep
WANDB_NAME="box_pca${PCA_K}_chunk${CHUNK}_scale${SCALE}_utd${K}_binary_reward"

if [[ ! -f "${CKPT}/policy/config.json" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

echo "Array task ${g} on $(hostname)"
echo "  chunk=${CHUNK} action_scale=${SCALE} utd=${K} pca_k=${PCA_K} seeds=${SEEDS[*]} reward_discount=false"
echo "  critic_updates=${N_UPDATES} actor_updates=${N_ACTOR} per transition"
echo "  gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  wandb=${WANDB_PROJECT}/${WANDB_NAME}"
echo "  ckpt=${CKPT}"

run_train() {
  local run_log="$1"
  shift
  local attempt
  for attempt in 1 2 3; do
    python -m resfit.rl_finetuning.scripts.train_residual_td3 "$@" > "${run_log}" 2>&1 && return 0
    echo "train failed (attempt ${attempt}), retrying in 30s" | tee -a "${run_log}"
    sleep 30
  done
  return 1
}

pids=()
for seed in "${SEEDS[@]}"; do
  run_log="${LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_seed${seed}.log"
  echo "  seed=${seed} log=${run_log}"
  run_train "${run_log}" \
    --config-name="${CONFIG}" \
    base_policy.ckpt_path="${CKPT}" \
    algo.action_chunk="${CHUNK}" \
    algo.requery=false \
    algo.offline_fraction=0.0 \
    algo.action_chunk_reward_discount=false \
    algo.total_timesteps=200_000 \
    algo.num_updates_per_iteration="${N_UPDATES}" \
    algo.actor_updates_per_iteration="${N_ACTOR}" \
    algo.latent.name=pca \
    algo.latent.k="${PCA_K}" \
    agent.actor_lr=5e-6 \
    agent.actor.action_scale="${SCALE}" \
    save_video=false \
    seed="${seed}" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.group="${WANDB_NAME}" \
    wandb.name="${WANDB_NAME}" \
    wandb.mode=online &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
