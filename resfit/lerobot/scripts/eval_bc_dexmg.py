#!/usr/bin/env python
"""Evaluate a saved ACT BC checkpoint on a DexMimicGen env and write one MP4.

Uses a single synchronous env (no AsyncVectorEnv spawn) so this is safe on
Savio's 60G GPU allocations.

Example (from residual-offpolicy-rl, conda env residual_rl):

python -m resfit.lerobot.scripts.eval_bc_dexmg \
  --ckpt /global/scratch/users/r112358/dexmg-act-ckpts/TwoArmCoffee/37745619/latest \
  --env TwoArmCoffee \
  --num_episodes 20
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.utils.load_policy import load_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_VIDEO_KEYS = {
    "TwoArmCoffee": "observation.images.agentview",
    "TwoArmPouring": "observation.images.agentview",
    "TwoArmCanSortRandom": "observation.images.frontview",
    "TwoArmBoxCleanup": "observation.images.agentview",
    "TwoArmDrawerCleanup": "observation.images.agentview",
    "TwoArmLiftTray": "observation.images.agentview",
}


def resolve_policy_dir(ckpt: Path) -> Path:
    ckpt = ckpt.expanduser().resolve()
    if (ckpt / "policy" / "config.json").exists():
        return ckpt / "policy"
    if (ckpt / "config.json").exists():
        return ckpt
    raise FileNotFoundError(
        f"No policy config.json under {ckpt} or {ckpt / 'policy'}. "
        "Pass a policy_step_<n> dir, latest/, or the policy/ folder itself."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a DexMG ACT BC checkpoint and save one MP4.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint dir (policy_step_N, latest, or policy/)")
    parser.add_argument("--env", type=str, required=True, help="DexMimicGen env, e.g. TwoArmCoffee")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--output", type=str, default=None, help="Output mp4 path (default: next to ckpt)")
    parser.add_argument("--no_video", action="store_true", help="Skip MP4 recording")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--camera_size", type=int, default=84, help="Policy camera size (must match training)")
    parser.add_argument("--render_size", type=int, default=224, help="Video render resolution")
    parser.add_argument(
        "--video_key",
        type=str,
        default=None,
        help="Camera for the mp4, e.g. observation.images.agentview",
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Optional training step to log to W&B (inferred from policy_step_N if omitted)",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    policy_dir = resolve_policy_dir(ckpt_path)
    video_key = args.video_key or DEFAULT_VIDEO_KEYS.get(args.env, "observation.images.agentview")

    checkpoint_step = args.checkpoint_step
    if checkpoint_step is None:
        name = ckpt_path.name
        if name.startswith("policy_step_"):
            try:
                checkpoint_step = int(name.split("policy_step_", 1)[1])
            except ValueError:
                checkpoint_step = None

    out_path = None
    if not args.no_video:
        if args.output is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = policy_dir.parent / f"eval_{args.env}_{args.num_episodes}ep_{stamp}.mp4"
        else:
            out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info("Loading policy from %s", policy_dir)
    policy = load_policy(policy_dir)
    policy.to(device)
    policy.eval()

    logger.info("Creating sync eval env %s (1 env, no multiprocessing)", args.env)
    env = create_vectorized_env(
        env_name=args.env,
        num_envs=1,
        device=str(device),
        camera_size=args.camera_size,
        render_size=args.render_size,
        debug=True,
        video_key=video_key,
    )

    writer = None
    if out_path is not None:
        import imageio

        writer = imageio.get_writer(out_path.as_posix(), fps=20)

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            mode=args.wandb_mode,
            config={
                "env": args.env,
                "ckpt": str(ckpt_path),
                "num_episodes": args.num_episodes,
                "checkpoint_step": checkpoint_step,
                "no_video": args.no_video,
            },
        )

    successes = 0
    done_episodes = 0
    total_steps = 0
    t0 = time.perf_counter()

    obs, _ = env.reset()
    policy.reset()
    episode_step = 0

    if out_path is None:
        logger.info("Running %d episodes (no video)", args.num_episodes)
    else:
        logger.info("Running %d episodes -> %s", args.num_episodes, out_path)
    while done_episodes < args.num_episodes:
        with torch.inference_mode():
            action = policy.select_action(obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        if writer is not None:
            writer.append_data(env.render()[0])
        episode_step += 1
        total_steps += 1

        done = bool((terminated | truncated)[0].item())
        if not done:
            continue

        is_success = bool(reward[0].item() == 1.0)
        done_episodes += 1
        successes += int(is_success)
        logger.info(
            "Episode %d/%d: %s (%d steps)",
            done_episodes,
            args.num_episodes,
            "SUCCESS" if is_success else "FAIL",
            episode_step,
        )
        policy.reset()
        episode_step = 0

    if writer is not None:
        writer.close()
    env.close()

    elapsed = time.perf_counter() - t0
    success_rate = successes / done_episodes if done_episodes else 0.0
    logger.info(
        "Done: %d/%d success (%.1f%%) | %d steps in %.1fs%s",
        successes,
        done_episodes,
        100 * success_rate,
        total_steps,
        elapsed,
        f" | video: {out_path}" if out_path is not None else "",
    )

    if wandb_run is not None:
        log = {
            "eval/success_rate": success_rate,
            "eval/successes": successes,
            "eval/num_episodes": done_episodes,
            "eval/total_steps": total_steps,
        }
        wandb_run.log(log, step=checkpoint_step)
        wandb_run.summary["eval/success_rate"] = success_rate
        wandb_run.finish()


if __name__ == "__main__":
    main()
