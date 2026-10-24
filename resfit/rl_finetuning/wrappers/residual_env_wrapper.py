# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Environment wrapper that includes a base policy, enabling residual RL training
to be done in a standard way without explicit base policy handling in the training loop.

This wrapper assumes:
- The environment is already vectorized (batched)
- All inputs/outputs are torch tensors
- The environment comes from create_vectorized_env
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper
from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.rl_finetuning.latent_rl.base import LatentModel, decode_latent


class BasePolicyVecEnvWrapper:
    """
    Wraps a vectorized environment with a base policy to enable standard RL training of residual policies.

    This wrapper:
    1. Takes raw observations from the vectorized environment
    2. Passes them through the base policy to get base actions
    3. Augments observations with base actions for the residual policy
    4. Combines base + residual actions before stepping the environment
    5. Returns augmented observations that include base actions

    Assumes the environment is already vectorized and works with torch tensors.
    """

    def __init__(
        self,
        vec_env: VectorizedEnvWrapper,
        base_policy: ACTPolicy,
        action_scaler,
        state_standardizer,
        action_chunk: int = 1,
        requery: bool = False,
        gamma: float = 0.99,
        reward_discount: bool = True,
    ):
        """
        Args:
            vec_env: Vectorized environment from create_vectorized_env
            base_policy: Base policy (e.g., ACTPolicy) to augment with residual actions
            action_scaler: ActionScaler object for scaling/unscaling actions (REQUIRED)
            state_standardizer: StateStandardizer object for standardizing states (REQUIRED)
            action_chunk: RL action chunk size. 
            requery: False walks one ACT prediction. True queries again after every window.
            gamma: Per-environment-step discount used inside a chunk.
            reward_discount: If True, store sum_i gamma^i r_{t+i}. If False, store 0/1 success.
        """
        assert action_scaler is not None, "action_scaler is required for consistent normalization"
        assert state_standardizer is not None, "state_standardizer is required for consistent normalization"

        self.vec_env = vec_env
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.action_chunk = int(action_chunk)
        self.requery = bool(requery)
        self.gamma = float(gamma)
        self.reward_discount = bool(reward_discount)
        self.n_action_steps = int(getattr(base_policy.config, "n_action_steps", 1))
        self.latent: LatentModel | None = None
        self._last_state: torch.Tensor | None = None

        # Get action dimension from the environment
        self.action_dim = vec_env.action_space.shape[-1]
        self.rl_action_dim = self.action_dim * self.action_chunk

        # Store image keys from base policy config
        self.image_keys = list(base_policy.config.image_features.keys())

        # Create modified observation space that includes base actions
        self._setup_observation_space()
        if self.action_chunk > 1:
            batch_shape = vec_env.action_space.shape[:-1]
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(*batch_shape, self.rl_action_dim),
                dtype=np.float32,
            )

    def _setup_observation_space(self):
        """Setup observation space to include base actions in the state."""

        # Get original observation space
        orig_obs_space = self.vec_env.observation_space

        # Copy the action space
        self.action_space = self.vec_env.action_space

        # Create new observation space with augmented state
        obs_spaces = {}
        for key, space in orig_obs_space.spaces.items():
            if key == "observation.state":
                # Augment state dimension with base actions
                orig_shape = list(space.shape)
                new_shape = orig_shape.copy()
                # new_shape[-1] += self.action_dim  # Not anymore
                obs_spaces[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=tuple(new_shape), dtype=space.dtype)
            else:
                # Keep other observations unchanged
                obs_spaces[key] = space

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset environment and base policy."""
        # Reset the underlying vectorized environment
        raw_obs, info = self.vec_env.reset(**kwargs)

        # Reset base policy
        self.base_policy.reset()
        base_naction = self._refresh_chunk(raw_obs)

        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, base_naction)

        # Store for later use in step
        self._last_base_naction = base_naction

        return augmented_obs, info

    def step(
        self, residual_naction: torch.Tensor, on_env_step=None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment with residual action.

        Args:
            residual_action: The residual action from the residual policy

        Returns:
            augmented_obs: Observations augmented with base actions
            reward: Reward tensor
            terminated: Terminated tensor
            truncated: Truncated tensor
            info: Info dict
        """
        # Combine base and residual actions
        # Residual action is already scaled inside the Actor class
        # To ensure that we can use the same exploration for all dimensions,
        # we use the normalized actions as the action space
        # The normalized base action is stored as [-1, 1] in the replay buffer
        # and the residual action is predicted as action_scale * [-1, 1]
        batch = residual_naction.shape[0]
        base = self._last_base_naction.view(batch, self.action_chunk, self.action_dim)
        with torch.no_grad():
            delta = decode_latent(self.latent, residual_naction, self._last_state)
        delta = delta.view(batch, self.action_chunk, self.action_dim)
        combined = torch.clamp(base + delta, -1.0, 1.0)

        discounted_reward = torch.zeros(combined.shape[0], device=combined.device, dtype=torch.float32)
        step_discount = 1.0
        raw_obs = None
        reward = None
        terminated = truncated = None
        last_info: dict = {}
        n_inner = 0
        for i in range(self.action_chunk):
            # Unscale back to original action space for environment execution
            env_action = self.action_scaler.unscale(combined[:, i])
            # Step the underlying vectorized environment
            raw_obs, reward, terminated, truncated, last_info = self.vec_env.step(
                env_action
            )
            r = reward.reshape(reward.shape[0], -1)
            if self.reward_discount:
                discounted_reward = discounted_reward + r.sum(dim=-1) * step_discount
                step_discount *= self.gamma
            else:
                discounted_reward = torch.maximum(
                    discounted_reward, (r == 1.0).any(dim=-1).to(dtype=discounted_reward.dtype)
                )
            n_inner += 1
            if on_env_step is not None:
                on_env_step()
            if (terminated | truncated).any():
                break

        last_info = dict(last_info)
        # Store the scaled action for replay buffer (already computed above)
        last_info["scaled_action"] = combined.reshape(combined.shape[0], -1).detach()
        last_info["chunk_env_steps"] = int(n_inner)

        done_mask = terminated | truncated
        if done_mask.any():
            # Handle policy reset for terminated environments
            self.base_policy.reset(env_ids=torch.where(done_mask)[0].tolist())
            # Get next base action from the base policy
            base_naction = self._refresh_chunk(raw_obs)
        elif self.requery or self._t + self.action_chunk >= self.n_action_steps:
            base_naction = self._refresh_chunk(raw_obs)
        else:
            self._t += self.action_chunk
            base_naction = self._window_flat()
        # Store for next step
        self._last_base_naction = base_naction

        # Handle final_obs in info dict to ensure consistent shapes
        if "final_obs" in last_info:
            last_info = self._process_final_obs_in_info(last_info, combined.device)

        reward_out = discounted_reward.to(dtype=reward.dtype)
        # Augment observations with base action and apply state standardization
        return self._augment_obs(raw_obs, base_naction), reward_out, terminated, truncated, last_info

    def _augment_obs(self, raw_obs: dict[str, torch.Tensor], base_naction: torch.Tensor) -> dict[str, torch.Tensor]:
        """Augment observations with base actions."""

        # New way to do this is to just add the base action to the state under its own key
        augmented_obs = raw_obs.copy()
        augmented_obs["observation.base_action"] = base_naction
        augmented_obs["observation.state"] = self.state_standardizer.standardize(augmented_obs["observation.state"])
        self._last_state = augmented_obs["observation.state"]

        return augmented_obs

    def set_latent(self, latent: LatentModel | None) -> None:
        """Attach a fitted latent model. ``step`` decodes actor outputs before adding them to the base chunk."""
        self.latent = latent

    def _process_final_obs_in_info(self, info: dict, device: torch.device) -> dict:
        """Pad final_obs state with zeros to match augmented observation format."""
        if "final_obs" not in info or info["final_obs"] is None:
            return info

        for final_obs_dict in info["final_obs"]:
            if final_obs_dict is not None and "observation.state" in final_obs_dict:
                # Pad with zeros (no action taken at terminal state)
                final_obs_dict["observation.base_action"] = torch.zeros(
                    self.rl_action_dim, device=device, dtype=torch.float32
                )

        return info

    def _refresh_chunk(self, raw_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Query ACT and return the first action_chunk scaled actions, flattened."""
        with torch.no_grad():
            chunk = self.base_policy.predict_action_chunk(raw_obs)
        self._chunk = self.action_scaler.scale(chunk)
        self._t = 0
        return self._window_flat()

    def _window_flat(self) -> torch.Tensor:
        window = self._chunk[:, self._t : self._t + self.action_chunk]
        return window.reshape(window.shape[0], -1)

    def render(self):
        """Pass through to underlying environment."""
        return self.vec_env.render()

    def close(self):
        """Close the environment."""
        return self.vec_env.close()

    # Pass through any other attributes/methods to the underlying environment
    def __getattr__(self, name: str):
        """Delegate unknown attributes to the underlying vectorized environment."""
        return getattr(self.vec_env, name)
