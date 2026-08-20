#!/usr/bin/env python
"""Collect ACT rollouts, fit PCA on action chunks, plot the spectrum, play eigenactions.

Collects --num_episodes of policy rollouts, groups executed actions into
(chunk_size, action_dim) windows (default 20 x 24), fits PCA, writes:

  spectrum.png              explained-variance spectrum
  pca.npz                   mean, components, explained variance
  eigenaction_00.mp4 ...    open-loop playback of top-K PCs (no overlay text)

Each eigenaction video starts from a fresh reset and applies
  mean_chunk + scale * sqrt(lambda_k) * PC_k
for --play_steps env steps (the chunk is tiled if play_steps > chunk_size).

Example
=======
python -m resfit.lerobot.scripts.plot_act_eigenactions \
  --ckpt /global/scratch/users/r112358/dexmg-act-ckpts/TwoArmCoffee/37745619/latest \
  --env TwoArmCoffee \
  --num_episodes 20 \
  --k 8 \
  --play_steps 40 \
  --output_dir ~/residual-offpolicy-rl/evals/eigenactions_coffee
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from resfit.lerobot.scripts.eval_bc_dexmg import DEFAULT_VIDEO_KEYS, resolve_policy_dir
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.utils.load_policy import load_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _to_env_action(action_nd: np.ndarray, device: torch.device) -> torch.Tensor:
    a = torch.as_tensor(np.clip(action_nd, -1.0, 1.0), device=device, dtype=torch.float32)
    if a.ndim == 1:
        a = a.unsqueeze(0)
    return a


def collect_action_chunks(
    policy,
    env,
    *,
    num_episodes: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run the policy and return action chunks of shape (N, chunk_size, action_dim)."""
    chunks: list[np.ndarray] = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        policy.reset()
        ep_actions: list[np.ndarray] = []
        while True:
            with torch.inference_mode():
                action = policy.select_action(obs)
            ep_actions.append(action[0].detach().float().cpu().numpy())
            obs, _reward, terminated, truncated, _info = env.step(action)
            if bool((terminated | truncated)[0].item()):
                break
        a = np.stack(ep_actions, axis=0)
        n = (len(a) // chunk_size) * chunk_size
        if n == 0:
            logger.warning("Episode %d too short (%d steps) for chunk_size=%d", ep + 1, len(a), chunk_size)
            continue
        ep_chunks = a[:n].reshape(-1, chunk_size, a.shape[-1])
        chunks.append(ep_chunks)
        logger.info(
            "Episode %d/%d: %d steps -> %d chunks",
            ep + 1,
            num_episodes,
            len(a),
            ep_chunks.shape[0],
        )
    if not chunks:
        raise RuntimeError("No action chunks collected")
    out = np.concatenate(chunks, axis=0)
    logger.info("Collected %d chunks of shape (%d, %d)", out.shape[0], out.shape[1], out.shape[2])
    return out


def plot_spectrum(explained: np.ndarray, path: Path, *, max_show: int = 50) -> None:
    n_show = min(max_show, len(explained))
    xs = np.arange(1, n_show + 1)
    cum = np.cumsum(explained)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].bar(xs, explained[:n_show], color="steelblue", width=0.8)
    axes[0].set_ylabel("Explained variance")
    axes[0].set_title("PCA spectrum of ACT action chunks")
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
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote spectrum to %s", path)


def play_eigenaction(
    env,
    seq: np.ndarray,
    *,
    play_steps: int,
    device: torch.device,
    video_path: Path,
    fps: int = 20,
) -> None:
    """Open-loop: reset, then tile `seq` (T, D) for `play_steps`."""
    t_chunk = seq.shape[0]
    env.reset()
    writer = imageio.get_writer(video_path.as_posix(), fps=fps)
    for t in range(play_steps):
        action_t = seq[t % t_chunk]
        env.step(_to_env_action(action_t, device))
        writer.append_data(env.render()[0])
    writer.close()
    logger.info("Wrote %s", video_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA eigenactions from ACT BC rollouts.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--k", type=int, default=8, help="Top-K eigenactions to play and save")
    parser.add_argument("--play_steps", type=int, default=40, help="Open-loop env steps per eigenaction video")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiplier on sqrt(lambda_k) * PC_k added to the mean chunk",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--camera_size", type=int, default=84)
    parser.add_argument("--render_size", type=int, default=224)
    parser.add_argument("--video_key", type=str, default=None)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    policy_dir = resolve_policy_dir(Path(args.ckpt))
    video_key = args.video_key or DEFAULT_VIDEO_KEYS.get(args.env, "observation.images.agentview")
    if args.output_dir is None:
        out_dir = Path.home() / "residual-offpolicy-rl" / "evals" / f"eigenactions_{args.env}"
    else:
        out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info("Loading policy from %s", policy_dir)
    policy = load_policy(policy_dir)
    policy.to(device)
    policy.eval()
    chunk_size = int(getattr(policy.config, "chunk_size", 20))

    env = create_vectorized_env(
        env_name=args.env,
        num_envs=1,
        device=str(device),
        camera_size=args.camera_size,
        render_size=args.render_size,
        debug=True,
        video_key=video_key,
    )

    chunks = collect_action_chunks(
        policy, env, num_episodes=args.num_episodes, chunk_size=chunk_size, device=device
    )
    n, t, d = chunks.shape
    x = chunks.reshape(n, t * d).astype(np.float64)
    n_comp = min(x.shape[0], x.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(x)
    explained = pca.explained_variance_ratio_.astype(np.float64)
    mean_chunk = pca.mean_.reshape(t, d).astype(np.float32)

    np.savez_compressed(
        out_dir / "pca.npz",
        mean_chunk=mean_chunk,
        components=pca.components_.astype(np.float32),
        explained_variance_ratio=explained.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        chunk_size=np.array([t], dtype=np.int32),
        action_dim=np.array([d], dtype=np.int32),
    )
    (out_dir / "pca.json").write_text(
        json.dumps(
            {
                "n_chunks": int(n),
                "chunk_size": int(t),
                "action_dim": int(d),
                "k": int(args.k),
                "total_explained_top_k": float(explained[: args.k].sum()),
                "explained_variance_ratio": explained[: max(args.k, 32)].tolist(),
            },
            indent=2,
        )
    )
    plot_spectrum(explained, out_dir / "spectrum.png")

    k = min(int(args.k), pca.components_.shape[0])
    for i in range(k):
        pc = pca.components_[i].reshape(t, d).astype(np.float32)
        amp = args.scale * float(np.sqrt(max(pca.explained_variance_[i], 0.0)))
        seq = mean_chunk + amp * pc
        play_eigenaction(
            env,
            seq,
            play_steps=args.play_steps,
            device=device,
            video_path=out_dir / f"eigenaction_{i:02d}.mp4",
        )

    env.close()
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
