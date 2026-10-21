#!/bin/bash

#SBATCH --job-name=box_base
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-71
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=n0141.savio4,n0142.savio4

# TwoArmBoxCleanup standard chunk residual. No PCA, no VAE.
# Q is on the executed action. offline_fraction=0. n_step is forced to 1
# for these horizons. actor_lr=5e-6. 200k steps. One seed per GPU.
#
#   horizon       5, 10, 20
#   action_scale  0.05, 0.1, 0.2, 0.5
#   utd           1, 4, 8
#   seeds         1001, 1456
#   36 configs * 2 seeds = 72 nodes
#
# Index: config major (horizon, then scale, then utd), then seed.
# QoS caps are 16 / 8 / 8 GPUs, so each band drains in two waves:
#   0-31   rail_gpu4_high    (32)
#  32-47   rail_gpu4_normal  (16)
#  48-63   rail_gpu4_low     (16)
#  64-71   rail_gpu4_lowest  (8)
#
#   cd /global/home/users/r112358/residual-offpolicy-rl
#   sbatch --qos=rail_gpu4_high    --array=0-31  resfit/rl_finetuning/shell/slurm_box_baseline_sweep.sh
#   sbatch --qos=rail_gpu4_normal  --array=32-47 resfit/rl_finetuning/shell/slurm_box_baseline_sweep.sh
#   sbatch --qos=rail_gpu4_low     --array=48-63 resfit/rl_finetuning/shell/slurm_box_baseline_sweep.sh
#   sbatch --qos=rail_gpu4_lowest  --array=64-71 resfit/rl_finetuning/shell/slurm_box_baseline_sweep.sh

source ~/.bashrc
set -eo pipefail

# Call the env interpreter directly. `conda activate` goes through the base
# conda install, which is broken on n0142 and then falls through to Python 3.13.
export CONDA_PREFIX=/global/scratch/users/r112358/miniconda3/envs/residual_rl
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PY="${CONDA_PREFIX}/bin/python"
export OMP_NUM_THREADS=4
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

mkdir -p logs "${WANDB_DIR}" "${CACHE_DIR}"

HORIZONS=(5 10 20)
SCALES=(0.05 0.1 0.2 0.5)
UTDS=(1 4 8)
SEEDS=(1001 1456)

n_u=${#UTDS[@]}
n_s=${#SCALES[@]}
n_h=${#HORIZONS[@]}
n_seed=${#SEEDS[@]}
TOTAL_CFGS=$((n_h * n_s * n_u))
TOTAL=$((TOTAL_CFGS * n_seed))

g=${SLURM_ARRAY_TASK_ID}
if (( g >= TOTAL )); then
  echo "Array index ${g} has no config (TOTAL=${TOTAL})."
  exit 1
fi

seed_i=$((g % n_seed))
cfg=$((g / n_seed))
u_i=$((cfg % n_u))
s_i=$(((cfg / n_u) % n_s))
h_i=$((cfg / (n_u * n_s)))

HORIZON=${HORIZONS[$h_i]}
SCALE=${SCALES[$s_i]}
UTD=${UTDS[$u_i]}
SEED=${SEEDS[$seed_i]}

CONFIG=residual_td3_box_clean_config
CKPT=/global/scratch/users/r112358/dexmg-act-ckpts/TwoArmBoxCleanup/37745620/policy_step_20000
WANDB_PROJECT=dexmg_box_baseline_sweep
WANDB_GROUP="box_base_h${HORIZON}_as${SCALE}_utd${UTD}"
RUN_NAME="${WANDB_GROUP}_seed${SEED}"

if [[ ! -f "${CKPT}/policy/config.json" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

echo "Array ${g} / ${TOTAL}"
echo "  horizon=${HORIZON} action_scale=${SCALE} utd=${UTD} seed=${SEED}"
echo "  run=${RUN_NAME}"
echo "  ckpt=${CKPT}"
echo "  host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"

"${PY}" -m resfit.rl_finetuning.scripts.train_residual_td3 \
  --config-name="${CONFIG}" \
  base_policy.ckpt_path="${CKPT}" \
  algo.offline_fraction=0.0 \
  algo.prefetch_batches=4 \
  algo.total_timesteps=200_000 \
  algo.n_step=1 \
  algo.critic_on_residual=false \
  algo.num_updates_per_iteration="${UTD}" \
  algo.update_every_n_steps=1 \
  algo.chunk_residual.enabled=true \
  algo.chunk_residual.use_pca=false \
  algo.chunk_residual.horizon="${HORIZON}" \
  agent.actor.action_scale="${SCALE}" \
  agent.actor_lr=5e-6 \
  seed="${SEED}" \
  eval_num_envs=1 \
  eval_first=false \
  debug=false \
  wandb.project="${WANDB_PROJECT}" \
  wandb.name="${RUN_NAME}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.mode=online

echo "Completed array ${g}: h=${HORIZON} as=${SCALE} utd=${UTD} seed=${SEED}"
