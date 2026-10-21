#!/usr/bin/env python
"""Train a LeRobot policy on a dataset hosted on the Hugging Face Hub.


Key features
============
1.  Works with any dataset created with `LeRobotDataset` and uploaded to the Hub.
2.  Supports the main LeRobot policies ("diffusion", "act", "pi0", …) with
    their default configs.
3.  Delta-timestamps are **inferred automatically** from the policy's
    *_delta_indices* properties so that the dataset delivers exactly what the
    policy expects.  The utility that does the heavy-lifting lives in
    `lerobot.common.datasets.factory.resolve_delta_timestamps`.
4.  Uses AdamW for optimisation (learning-rate & weight-decay are taken from the
    policy's config when available).
5.  Optional Weights & Biases logging -- enabled via `--wandb_enable` and the
    `--wandb_project` name.
6.  Saves checkpoints using `policy.save_pretrained(…)` so that they can later
    be re-loaded with `PreTrainedPolicy.from_pretrained(…)` **or** retrieved
    directly from W&B artefacts for evaluation.
7.  At step 0 and every checkpoint (`--save_freq`), rolls out the policy,
    fits PCA on action chunks, and logs the spectrum to W&B as
    `spectrum/eval_step_{step}`.

Example
=======
python train_hf.py \
  --dataset ankile/franka-lift-dataset \
  --policy diffusion \
  --wandb_enable --wandb_project my_robot_runs
"""

from __future__ import annotations

import os
import sys

# Savio's /lib64/libstdc++ is too old for conda-forge FFmpeg (libvpl needs
# GLIBCXX_3.4.29). LD_LIBRARY_PATH is only read at process start, so re-exec
# once with conda lib first. DataLoader spawn workers inherit this.
_CONDA_LIB = os.path.join(sys.prefix, "lib")
_LD_FLAG = "RESFIT_CONDA_LIBSTDCXX"
if os.path.isdir(_CONDA_LIB):
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    _parts = [p for p in _ld.split(":") if p]
    if not _parts or os.path.abspath(_parts[0]) != os.path.abspath(_CONDA_LIB):
        os.environ["LD_LIBRARY_PATH"] = f"{_CONDA_LIB}:{_ld}" if _ld else _CONDA_LIB
        if os.environ.get(_LD_FLAG) != "1":
            os.environ[_LD_FLAG] = "1"
            os.execv(sys.executable, [sys.executable, *sys.argv])

# Spawned AsyncVectorEnv workers re-import this file with sys.path[0] =
# resfit/lerobot/scripts/, so `import resfit` fails unless the repo root is
# on PYTHONPATH / sys.path.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_py_path = os.environ.get("PYTHONPATH", "")
if _REPO_ROOT not in _py_path.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{_REPO_ROOT}{os.pathsep}{_py_path}" if _py_path else _REPO_ROOT

import argparse
import json
import logging
import multiprocessing as mp
import re
import shutil
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.common.datasets.utils import cycle
from lerobot.common.utils.random_utils import set_seed
from termcolor import colored

import wandb
from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper, create_vectorized_env
from resfit.lerobot.policies.factory import make_policy, make_policy_config
from resfit.lerobot.policies.pretrained import PreTrainedPolicy
from resfit.lerobot.utils.load_policy import load_checkpoint, save_checkpoint

# Set multiprocessing start method for CUDA compatibility
# This must be done before any other multiprocessing operations
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # Start method already set, which is fine
    pass

# -----------------------------------------------------------------------------
# Caching configuration ------------------------------------------------------
# -----------------------------------------------------------------------------
# Generic environment variable (shared across algorithms) -------------------
# ``CACHE_DIR`` specifies the root folder for **all** local caches.
# Falls back to the current directory if unset.
_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()


parser = argparse.ArgumentParser(description="Offline training on a HF Hub dataset with LeRobot policies")

# Required args
parser.add_argument(
    "--dataset", type=str, required=True, help="HF Hub dataset repo-id e.g. `ankile/franka-lift-dataset`"
)
parser.add_argument(
    "--policy",
    type=str,
    default="diffusion",
    choices=[
        "diffusion",
        "act",
        "latent_act",
        "pi0",
        "pi0fast",
        "tdmpc",
        "vqbet",
    ],
    help="Which policy architecture to train",
)

# Training hyper-parameters
parser.add_argument("--steps", type=int, default=100_000, help="Total optimization steps")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--grad_clip_norm", type=float, default=10.0)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument(
    "--video_backend",
    type=str,
    default="torchcodec",
    choices=["pyav", "torchcodec", "video_reader"],
    help="Video decoder backend. torchcodec is faster; pyav is the fallback.",
)

# Reproducibility / device
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

# Logging & checkpoints
parser.add_argument("--output_dir", type=str, default="outputs/train_hf")
parser.add_argument("--log_freq", type=int, default=100, help="How often to print & log to W&B (in steps)")
parser.add_argument("--save_freq", type=int, default=1_000, help="How often to save checkpoints (in steps)")

# WandB
parser.add_argument("--wandb_enable", action="store_true", help="Enable Weights & Biases logging")
parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (required when --wandb_enable)")
parser.add_argument("--wandb_entity", type=str, default=None)

# Resume
parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to a local checkpoint directory to resume from")
parser.add_argument(
    "--resume_run_id",
    type=str,
    default=None,
    help="WandB run ID to resume from (grabs the 'latest' artifact to restore trainer state).",
)

# ------------------------------------------------------------------
# PCA spectrum eval (rollouts at each checkpoint, including step 0)
# ------------------------------------------------------------------
parser.add_argument(
    "--rollout_freq",
    type=int,
    default=None,
    help=(
        "Deprecated / unused. PCA spectrum eval runs at step 0 and at every "
        "checkpoint (--save_freq), not on a separate success-rate schedule."
    ),
)
parser.add_argument(
    "--eval_env",
    type=str,
    default=None,
    help="DexMimicGen environment name used for PCA spectrum rollouts.",
)
parser.add_argument("--eval_num_envs", type=int, default=5, help="Number of parallel environments for PCA rollouts.")
parser.add_argument(
    "--eval_num_episodes",
    type=int,
    default=20,
    help="Episodes to roll out for PCA at step 0 and at every checkpoint.",
)
parser.add_argument(
    "--eval_camera_size", type=int, default=84, help="Camera image size for evaluation rollouts (should match dataset)."
)
parser.add_argument(
    "--eval_render_size",
    type=int,
    default=None,
    help="High-resolution camera size for video recording (if different from eval_camera_size).",
)
parser.add_argument(
    "--eval_video_key",
    type=str,
    default=None,
    help="Observation key for the camera to use for video recording (e.g., 'observation.images.frontview').",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode (uses synchronous vec envs instead of async multiprocessing for debugging).",
)

# -------------------------------------------------
# Policy configuration overrides
# -------------------------------------------------
# Allow users to pass a JSON-formatted string that will be forwarded as keyword
# arguments to the policy configuration dataclass (e.g. ACTConfig, DiffusionConfig, …).
# Example:
#   --policy_kwargs '{"dim_model": 1024, "chunk_size": 100}'
# This will result in `ACTConfig(dim_model=1024, chunk_size=100, …)`
parser.add_argument(
    "--policy_kwargs",
    type=str,
    default=None,
    help=(
        "Overrides for the policy configuration. Accepts either: "
        '1) A JSON dictionary string, e.g. \'{"dim_model": 1024, "chunk_size": 100}\', or '
        "2) A compact 'key=value' list separated by commas or spaces, e.g. "
        "   'dim_model=1024,chunk_size=100 optimizer_lr=3e-4'. "
        "All pairs are forwarded directly to make_policy_config(...)."
    ),
)

# Camera selection
parser.add_argument(
    "--policy_cameras",
    type=str,
    nargs="*",
    default=None,
    help=(
        "List of camera names to use for the policy. If not specified, all cameras from the dataset will be used. "
        "Example: --policy_cameras agentview robot0_eye_in_hand"
    ),
)

# Proprioceptive observations
parser.add_argument(
    "--disable_proprioceptive_obs",
    action="store_true",
    help=(
        "Disable proprioceptive observations (observation.state) during training. "
        "Only visual observations will be used."
    ),
)

# -------------------------------------------------
# Setup logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)


# Create a named logger
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PCA spectrum eval (rollouts → action-chunk PCA → wandb)
# -----------------------------------------------------------------------------


def _collect_action_chunks(
    policy: PreTrainedPolicy,
    env: VectorizedEnvWrapper,
    *,
    num_episodes: int,
    chunk_size: int,
) -> tuple[np.ndarray, float, int, int]:
    """Roll out *num_episodes* and return (chunks, success_rate, n_success, n_episodes).

    Chunks have shape (N, chunk_size, action_dim). Success is reward == 1 at episode end.
    """
    policy_was_training = policy.training
    policy.eval()

    num_envs = env.num_envs
    env_name = getattr(env, "env_name", "unknown")
    logger.info(
        f"PCA rollouts in {env_name}: {num_episodes} episodes, {num_envs} parallel envs, chunk_size={chunk_size}"
    )

    ep_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    chunks: list[np.ndarray] = []
    done_episodes = 0
    n_success = 0
    total_steps = 0
    start_time = time.perf_counter()

    obs, _ = env.reset()
    policy.reset()

    while done_episodes < num_episodes:
        with torch.inference_mode():
            action = policy.select_action(obs)
        action_np = action.detach().float().cpu().numpy()
        for i in range(num_envs):
            ep_actions[i].append(action_np[i])

        obs, reward, terminated, truncated, _info = env.step(action)
        total_steps += num_envs
        done = terminated | truncated

        if bool(done.any().item()):
            terminated_envs = torch.where(done)[0]
            policy.reset(env_ids=terminated_envs)
            for env_idx in terminated_envs:
                i = int(env_idx.item())
                if done_episodes >= num_episodes:
                    ep_actions[i] = []
                    continue
                a = np.stack(ep_actions[i], axis=0)
                n = (len(a) // chunk_size) * chunk_size
                if n == 0:
                    logger.warning(
                        "Episode %d too short (%d steps) for chunk_size=%d",
                        done_episodes + 1,
                        len(a),
                        chunk_size,
                    )
                else:
                    chunks.append(a[:n].reshape(-1, chunk_size, a.shape[-1]))
                is_success = bool(reward[i].item() == 1.0)
                n_success += int(is_success)
                done_episodes += 1
                ep_actions[i] = []
            if done_episodes >= num_episodes:
                break

        if total_steps % 1_000 == 0:
            logger.info(
                f"PCA rollouts: {total_steps} steps, {done_episodes}/{num_episodes} episodes, "
                f"{n_success} successes, "
                f"{total_steps / max(time.perf_counter() - start_time, 1e-6):.1f} fps"
            )

    if policy_was_training:
        policy.train()

    if not chunks:
        raise RuntimeError("No action chunks collected for PCA")
    out = np.concatenate(chunks, axis=0)
    elapsed = time.perf_counter() - start_time
    success_rate = n_success / done_episodes if done_episodes > 0 else 0.0
    logger.info(
        f"Collected {out.shape[0]} chunks of shape ({out.shape[1]}, {out.shape[2]}) "
        f"from {done_episodes} episodes in {elapsed:.1f}s "
        f"({n_success}/{done_episodes} success, {success_rate * 100:.1f}%)"
    )
    return out, success_rate, n_success, done_episodes


def _explained_variance_from_chunks(chunks: np.ndarray) -> np.ndarray:
    """PCA explained-variance ratio of flattened action chunks, largest first."""
    n, t, d = chunks.shape
    x = chunks.reshape(n, t * d).astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(x, full_matrices=False)
    eig = (s**2) / max(n - 1, 1)
    total = float(eig.sum())
    return (eig / max(total, 1e-12)).astype(np.float64)


def _plot_spectrum(explained: np.ndarray, path: Path, *, step: int, max_show: int = 50) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(max_show, len(explained))
    xs = np.arange(1, n_show + 1)
    cum = np.cumsum(explained)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].bar(xs, explained[:n_show], color="steelblue", width=0.8)
    axes[0].set_ylabel("Explained variance")
    axes[0].set_title(f"PCA spectrum of action chunks (step {step})")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].plot(xs, cum[:n_show], color="darkorange", marker="o", markersize=3)
    axes[1].axhline(0.9, color="gray", ls="--", lw=1, label="90%")
    axes[1].axhline(0.99, color="gray", ls=":", lw=1, label="99%")
    axes[1].set_xlabel("Principal component")
    axes[1].set_ylabel("Cumulative")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _n_pcs_to_threshold(explained: np.ndarray, threshold: float) -> int:
    if len(explained) == 0:
        return 0
    cum = np.cumsum(explained)
    idx = int(np.searchsorted(cum, threshold, side="left"))
    return min(idx + 1, len(explained))


def _run_pca_eval(
    *,
    policy: PreTrainedPolicy,
    env: VectorizedEnvWrapper,
    save_dir: Path,
    step: int,
    num_episodes: int,
    log_wandb: bool,
) -> None:
    """Roll out the policy, fit PCA on action chunks, save + log the spectrum."""
    chunk_size = int(getattr(policy.config, "chunk_size", 20))
    t0 = time.perf_counter()
    chunks, success_rate, n_success, n_done = _collect_action_chunks(
        policy, env, num_episodes=num_episodes, chunk_size=chunk_size
    )
    explained = _explained_variance_from_chunks(chunks)

    spectrum_dir = save_dir / "spectrum"
    plot_path = spectrum_dir / f"eval_step_{step}.png"
    npz_path = spectrum_dir / f"eval_step_{step}.npz"
    _plot_spectrum(explained, plot_path, step=step)
    np.savez_compressed(
        npz_path,
        explained_variance_ratio=explained.astype(np.float32),
        n_chunks=np.array([chunks.shape[0]], dtype=np.int32),
        chunk_size=np.array([chunks.shape[1]], dtype=np.int32),
        action_dim=np.array([chunks.shape[2]], dtype=np.int32),
    )

    n_pcs_90 = _n_pcs_to_threshold(explained, 0.9)
    n_pcs_99 = _n_pcs_to_threshold(explained, 0.99)
    top8 = float(explained[: min(8, len(explained))].sum())
    elapsed = time.perf_counter() - t0
    logger.info(
        colored(
            f"[step {step:>6d}] eval success-rate: {success_rate * 100:.1f}% "
            f"({n_success}/{n_done}) | PCA: n_chunks={chunks.shape[0]} "
            f"pc1={explained[0]:.3f} top8={top8:.3f} n_pcs_90={n_pcs_90} n_pcs_99={n_pcs_99} "
            f"({elapsed:.1f}s) -> {plot_path}",
            "cyan",
        )
    )

    if log_wandb:
        n_show = min(50, len(explained))
        cum = np.cumsum(explained)
        table = wandb.Table(columns=["pc", "explained_variance_ratio", "cumulative"])
        for i in range(n_show):
            table.add_data(i + 1, float(explained[i]), float(cum[i]))
        payload: dict[str, Any] = {
            "eval/success_rate": success_rate,
            "eval/n_success": n_success,
            "eval/n_episodes": n_done,
            f"spectrum/eval_step_{step}": wandb.Image(str(plot_path), caption=f"PCA spectrum @ step {step}"),
            "spectrum/values": table,
            "spectrum/n_chunks": int(chunks.shape[0]),
            "spectrum/pc1_explained": float(explained[0]),
            "spectrum/top8_explained": top8,
            "spectrum/n_pcs_90": n_pcs_90,
            "spectrum/n_pcs_99": n_pcs_99,
            "time/pca_eval_s": elapsed,
        }
        wandb.log(payload, step=step)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(cfg: argparse.Namespace):
    # ---------------------------------------------------------------------
    # Logging / device setup
    # ---------------------------------------------------------------------
    device = torch.device(cfg.device)
    logger.info(colored(f"Using device: {device}", "green"))

    # Create a timestamped folder in CACHE_DIR for all outputs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_cache_dir = _CACHE_ROOT / f"bc_run_{timestamp}_{Path(cfg.dataset).name}_{cfg.policy}"
    run_cache_dir.mkdir(parents=True, exist_ok=True)

    if cfg.seed is not None:
        set_seed(cfg.seed)
        logger.info(colored(f"Random seed set to {cfg.seed}", "yellow"))

    # ---------------------------------------------------------------------
    # Dataset (metadata first, then actual dataset with resolved timestamps)
    # ---------------------------------------------------------------------
    logger.info("Fetching dataset metadata from the Hub…")
    ds_meta = LeRobotDatasetMetadata(cfg.dataset)

    # ---------------------------------------------------------------------
    # Build the policy configuration, applying any CLI-specified overrides
    # ---------------------------------------------------------------------

    def _infer_type(val: str):
        """Try to cast *val* to int, float or bool if possible, else return str."""
        if val.lower() in {"true", "false"}:
            return val.lower() == "true"
        try:
            if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
                return int(val)
            return float(val)
        except ValueError:
            return val  # leave as string

    if cfg.policy_kwargs is not None:
        # Initialize policy_kwargs as empty dict
        policy_kwargs = {}

        # First attempt JSON parsing ----------------------------------
        try:
            policy_kwargs = json.loads(cfg.policy_kwargs)
            if not isinstance(policy_kwargs, dict):
                raise TypeError
        except Exception:
            # Fallback: parse "key1=val1,key2=val2 ..." ----------------
            text = cfg.policy_kwargs.strip()
            # Allow comma or whitespace as separators
            tokens = re.split(r"[ ,]+", text)
            for token in filter(None, tokens):
                if "=" not in token:
                    raise ValueError(f"Could not parse --policy_kwargs token '{token}'. Expected 'key=value'.")
                k, v = token.split("=", 1)
                policy_kwargs[k] = _infer_type(v)
    else:
        policy_kwargs = {}

    # Build the policy config and set the target device.  When resuming from
    # checkpoints the device string can include an explicit index (e.g.
    # "cuda:0").  The underlying config validator only understands the bare
    # device types ("cuda", "cpu", "mps"), so we drop any optional suffix
    # before passing it on.

    policy_cfg = make_policy_config(cfg.policy, **policy_kwargs)

    # Set the chunk size to 20 (env is at 20 fps)
    policy_cfg.chunk_size = 20
    policy_cfg.n_action_steps = 20

    if isinstance(cfg.device, str):
        # e.g. "cuda:0" -> "cuda"
        policy_cfg.device = cfg.device.split(":", 1)[0]
    else:
        # Fall back to original value if somehow not a string
        policy_cfg.device = cfg.device

    # Filter dataset features based on selected cameras if specified
    if cfg.policy_cameras is not None:
        logger.info(f"Filtering dataset to use only cameras: {cfg.policy_cameras}")

        # Create a filtered features dict that only includes selected cameras
        filtered_features = {
            key: feature for key, feature in ds_meta.features.items() if not key.startswith("observation.images.")
        }

        # Only include selected camera features
        available_cameras = []
        for key, feature in ds_meta.features.items():
            if key.startswith("observation.images."):
                camera_name = key.replace("observation.images.", "")
                available_cameras.append(camera_name)
                if camera_name in cfg.policy_cameras:
                    filtered_features[key] = feature

        # Validate that all requested cameras exist
        missing_cameras = [cam for cam in cfg.policy_cameras if cam not in available_cameras]
        if missing_cameras:
            raise ValueError(
                f"Requested cameras not found in dataset: {missing_cameras}. Available cameras: {available_cameras}"
            )

        logger.info(f"Available cameras: {available_cameras}")
        logger.info(f"Selected cameras: {cfg.policy_cameras}")

        # Update the metadata object with filtered features
        ds_meta.info["features"] = filtered_features

    # Filter dataset features to remove proprioceptive observations if specified
    if cfg.disable_proprioceptive_obs:
        logger.info("Filtering dataset to remove proprioceptive observations (observation.state)")

        # Create a filtered features dict that excludes observation.state
        filtered_features = {key: feature for key, feature in ds_meta.features.items() if key != "observation.state"}

        # Validate that we still have some observations
        remaining_obs_keys = [key for key in filtered_features if key.startswith("observation")]
        if not remaining_obs_keys:
            raise ValueError(
                "Cannot disable proprioceptive observations: no other observation types found in dataset. "
                "Dataset must contain at least one image observation or environment state."
            )

        logger.info(f"Remaining observation keys after filtering: {remaining_obs_keys}")

        # Update the metadata object with filtered features
        ds_meta.info["features"] = filtered_features

    # Determine delta-timestamps from policy indices & dataset fps
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)

    logger.info("Building LeRobotDataset with inferred delta-timestamps…")
    logger.info(f"Using video backend: {cfg.video_backend}")

    image_transforms_config = ImageTransformsConfig(enable=True)
    image_transforms = ImageTransforms(image_transforms_config)

    dataset = LeRobotDataset(
        cfg.dataset,
        delta_timestamps=delta_timestamps,
        download_videos=True,
        image_transforms=image_transforms,
        video_backend=cfg.video_backend,
    )

    # ---------------------------------------------------------------------
    # Dataloader
    # ---------------------------------------------------------------------
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    dl_iter = cycle(dataloader)

    # ---------------------------------------------------------------------
    # Policy + optimizer
    # ---------------------------------------------------------------------
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.train()

    # Print the policy config
    print(policy_cfg)

    # Learning-rate & weight-decay fallbacks
    lr_default = getattr(policy_cfg, "optimizer_lr", 1e-4)
    wd_default = getattr(policy_cfg, "optimizer_weight_decay", 0.0)

    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=lr_default, weight_decay=wd_default)

    # ---------------------------------------------------------------------
    # Optional WandB
    # ---------------------------------------------------------------------
    if cfg.wandb_enable:
        if cfg.wandb_project is None:
            raise ValueError("--wandb_project is required when --wandb_enable is set")

        wandb_run_id = cfg.resume_run_id if cfg.resume_run_id else None

        # -----------------------
        # Prepare an exhaustive config dict for W&B for improved reproducibility
        # -----------------------
        extra_cfg: dict[str, Any] = {}

        # 1) CLI / script arguments (already cover basics)
        extra_cfg.update(vars(cfg))

        # 2) Policy (actor) configuration
        try:
            extra_cfg["policy_config"] = asdict(policy_cfg) if is_dataclass(policy_cfg) else policy_cfg.__dict__
        except Exception:
            # Fallback to string repr if conversion failed (e.g. non-serialisable fields)
            extra_cfg["policy_config"] = str(policy_cfg)

        # 3) Dataset metadata (lightweight summary only to avoid huge nested structures)
        extra_cfg["dataset_meta"] = {
            "repo_id": ds_meta.repo_id,
            "fps": ds_meta.fps,
            "robot_type": ds_meta.robot_type,
            "total_episodes": ds_meta.total_episodes,
            "total_frames": ds_meta.total_frames,
            "feature_keys": list(ds_meta.features.keys()),
        }

        # 4) Delta-timestamps actually used when building the dataset
        extra_cfg["delta_timestamps"] = delta_timestamps

        # NEW: Log image transforms configuration
        extra_cfg["image_transforms"] = asdict(image_transforms_config)

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=extra_cfg,
            name=f"{cfg.policy}_{Path(cfg.dataset).name}",
            id=wandb_run_id,
            resume="must" if wandb_run_id else None,
        )
        logger.info(colored("W&B logging enabled", "blue"))

    # ---------------------------------------------------------------------
    # Optionally resume from checkpoint (local folder or WandB artifact)
    # ---------------------------------------------------------------------
    start_step = 0
    if cfg.resume_run_id is not None:
        logger.info(colored(f"Resuming from WandB run {cfg.resume_run_id}", "cyan"))
        api = wandb.Api()
        # Compose the artifact path: <entity>/<project>/run_<id>_latest:latest
        artifact_path = (
            f"{(cfg.wandb_entity + '/' if cfg.wandb_entity else '')}"
            f"{cfg.wandb_project}/run_{cfg.resume_run_id}_latest:latest"
        )
        artifact = api.artifact(artifact_path)
        artifact_dir = Path(artifact.download())

        start_step, policy, optimizer = load_checkpoint(artifact_dir, policy, optimizer)
        policy.to(device)
    elif cfg.resume_ckpt is not None:
        logger.info(colored(f"Resuming from local checkpoint {cfg.resume_ckpt}", "cyan"))
        start_step, policy, optimizer = load_checkpoint(Path(cfg.resume_ckpt), policy, optimizer)
        policy.to(device)

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    # Checkpoints go to --output_dir (kept on disk). run_cache_dir is only for
    # transient files and is deleted at the end if it is a different path.
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(colored(f"Checkpoints will be saved to {output_dir}", "yellow"))

    step = start_step
    eval_env = None  # type: ignore  # will hold the evaluation environment if created

    if cfg.eval_env:
        # ------------------------------------------------------------------
        # Create the DexMimicGen evaluation environment
        # ------------------------------------------------------------------
        device_str = "cpu" if cfg.device == "cpu" else "cuda"

        # Check if this is a DexMimicGen environment
        dexmimicgen_envs = [
            "TwoArmCoffee",
            "TwoArmThreading",
            "TwoArmThreePieceAssembly",
            "TwoArmTransport",
            "TwoArmLiftTray",
            "TwoArmBoxCleanup",
            "TwoArmDrawerCleanup",
            "TwoArmPouring",
            "TwoArmCanSortRandom",
        ]
        robomimic_envs = [
            "Lift",
            "Can",
            "Square",
            "Transport",
        ]
        mimicgen_envs = [
            "Threading",  # Single-arm threading task from MimicGen
        ]

        envs = dexmimicgen_envs + robomimic_envs + mimicgen_envs

        if cfg.eval_env in envs:
            # SyncVectorEnv only: AsyncVectorEnv spawn re-executes this file as
            # __main__ (python -m) and either nests extra trainers or dies with
            # BrokenPipe. PCA eval does not need async workers.
            logger.info(f"Creating evaluation environment: {cfg.eval_env} (SyncVectorEnv, {cfg.eval_num_envs} envs)")
            eval_env = create_vectorized_env(
                env_name=cfg.eval_env,
                num_envs=cfg.eval_num_envs,
                device=device_str,
                camera_size=cfg.eval_camera_size,
                render_size=cfg.eval_render_size,
                video_key=cfg.eval_video_key,
                debug=True,
            )
        else:
            raise ValueError(f"Unknown environment: {cfg.eval_env}. Supported environments are: {envs}")

    def _pca_at_step(eval_step: int) -> None:
        if eval_env is None:
            return
        _run_pca_eval(
            policy=policy,
            env=eval_env,
            save_dir=output_dir,
            step=eval_step,
            num_episodes=cfg.eval_num_episodes,
            log_wandb=bool(cfg.wandb_enable),
        )

    # Untrained policy, before any updates in this run.
    if start_step == 0:
        if eval_env is None:
            logger.info("No --eval_env; skipping PCA spectrum eval")
        else:
            logger.info(colored("Running PCA spectrum eval at step 0 (before training)", "cyan"))
            _pca_at_step(0)

    while step < cfg.steps:
        # ------------------------------------------------------------------
        # Measure data loading time ----------------------------------------
        # ------------------------------------------------------------------
        iter_start_t = time.perf_counter()

        data_t0 = time.perf_counter()
        batch: dict[str, Any] = next(dl_iter)

        # Move tensors to device
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device, non_blocking=True)
        data_load_ms = (time.perf_counter() - data_t0) * 1000

        # ------------------------------------------------------------------
        # Measure policy update time ---------------------------------------
        # ------------------------------------------------------------------
        update_t0 = time.perf_counter()

        loss, _ = policy.forward(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        update_ms = (time.perf_counter() - update_t0) * 1000

        iter_ms = (time.perf_counter() - iter_start_t) * 1000

        # ------------------------------------------------------------------
        # Logging ----------------------------------------------------------
        # ------------------------------------------------------------------
        loss_val = loss.item()

        if step % cfg.log_freq == 0:
            msg = (
                f"[step {step:>6d}/{cfg.steps}]"
                f" loss: {loss_val:.4f}"
                f" | data: {data_load_ms:.1f} ms"
                f" | update: {update_ms:.1f} ms"
                f" | iter: {iter_ms:.1f} ms"
            )
            logger.info(msg)
            if wandb is not None:
                wandb.log(
                    {
                        "train/loss": loss_val,
                        "time/data_load_ms": data_load_ms,
                        "time/update_ms": update_ms,
                        "time/iter_ms": iter_ms,
                    },
                    step=step,
                )

        # Checkpointing ----------------------------------------------
        if (step % cfg.save_freq == 0 and step != start_step) or step + 1 == cfg.steps:
            # 1) Save model-only weights (space-efficient, keeps history)
            #    └── policy_step_<n>
            model_dir = output_dir / f"policy_step_{step}"
            model_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(model_dir / "policy")

            # 2) Save full training state once under a constant "latest" directory
            latest_dir = output_dir / "latest"
            if latest_dir.exists():
                # Remove previous to avoid stale files
                shutil.rmtree(latest_dir)
            save_checkpoint(latest_dir, step, policy, optimizer)

            logger.info(
                colored(
                    f"Checkpoint saved (model-only @ {model_dir}, full state @ {latest_dir})",
                    "magenta",
                )
            )

            if wandb is not None:
                # Log model-only artifact (no optimizer)
                art_model = wandb.Artifact(name=f"run_{wandb.run.id}_model_step_{step}", type="model")
                art_model.add_dir(str(model_dir))
                wandb.log_artifact(art_model)

                # Log/overwrite the "latest" artifact with full state for resume
                art_latest = wandb.Artifact(name=f"run_{wandb.run.id}_latest", type="model")
                art_latest.add_dir(str(latest_dir))
                wandb.log_artifact(art_latest, aliases=["latest"])

            _pca_at_step(step)

        step += 1

    logger.info(colored("Training finished!", "green", attrs=["bold"]))
    if wandb is not None:
        wandb.finish()

    if eval_env is not None:
        eval_env.close()

    # ---------------------------------------------------------------------
    # Cleanup --------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Do not delete checkpoints. Only remove the transient cache dir if it is
    # separate from --output_dir.
    if run_cache_dir.exists() and run_cache_dir.resolve() != output_dir.resolve():
        logger.info(f"Cleaning up run cache directory: {run_cache_dir}")
        shutil.rmtree(run_cache_dir)


if __name__ == "__main__":
    """
    Example commands:

    PCA spectrum eval runs at step 0 and at every checkpoint (--save_freq, default 1000):
    20 env rollouts → action-chunk PCA → wandb.Image at spectrum/eval_step_{step}.

    python -m resfit.lerobot.scripts.train_bc_dexmg \
        --dataset ankile/dexmimicgen-coffee-v1 \
        --policy act \
        --batch_size 64 --num_workers 8 \
        --wandb_enable --wandb_project dexmimicgen-test \
        --eval_env TwoArmCoffee \
        --eval_num_envs 8 \
        --eval_num_episodes 20 \
        --save_freq 1000 \
        --eval_camera_size 84 --eval_render_size 224

    # Debug mode: uses synchronous environments for easier debugging with ipdb
    python -m resfit.lerobot.scripts.train_bc_dexmg \
        --dataset ankile/dexmimicgen-coffee-v1 \
        --policy act \
        --batch_size 64 --num_workers 8 \
        --wandb_enable --wandb_project dexmimicgen-test \
        --eval_env TwoArmCoffee \
        --eval_num_envs 4 \
        --eval_num_episodes 12 \
        --eval_camera_size 84 --eval_render_size 224 \
        --debug

    # Train without proprioceptive observations (vision-only)
    python -m resfit.lerobot.scripts.train_bc_dexmg \
        --dataset ankile/dexmimicgen-coffee-v1 \
        --policy act \
        --batch_size 64 --num_workers 8 \
        --wandb_enable --wandb_project dexmimicgen-test \
        --disable_proprioceptive_obs
    """
    main(parser.parse_args())
