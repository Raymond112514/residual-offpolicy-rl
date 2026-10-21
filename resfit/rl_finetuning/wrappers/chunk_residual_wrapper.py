# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Chunk residual: one RL step is an h-step window of an ACT chunk + residual."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper
from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.rl_finetuning.autoencoder.action_ae import ResidualActionAE


def _flat_state(obs) -> np.ndarray:
    state = obs["observation.state"]
    if torch.is_tensor(state):
        state = state.detach().float().cpu().numpy()
    return np.asarray(state, dtype=np.float32).reshape(-1)


def collect_act_chunk_windows(
    *,
    vec_env: VectorizedEnvWrapper,
    base_policy: ACTPolicy,
    action_scaler,
    act_chunk_size: int,
    horizon: int,
    n_episodes: int,
    return_states: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Collect scaled action-chunk windows from initial base-policy rollouts.

    Returns X with shape (N, horizon * action_dim). These are actions, not
    pairwise differences X-X'. When `return_states` is set, also return the
    proprio vector from the observation that produced the ACT chunk. Every
    window sliced from that chunk shares that query-time state.
    """
    chunks: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for ep in range(n_episodes):
        obs, _ = vec_env.reset()
        base_policy.reset()
        n_windows = 0
        t = 0
        act_chunk = None
        query_state = None
        while True:
            if act_chunk is None or t >= act_chunk_size:
                with torch.no_grad():
                    act_chunk = action_scaler.scale(base_policy.predict_action_chunk(obs))
                t = 0
                if return_states:
                    query_state = _flat_state(obs)
            window = act_chunk[0, t : t + horizon].detach().float().cpu().numpy()
            chunks.append(window)
            if return_states:
                states.append(query_state)
            n_windows += 1
            done_ep = False
            for i in range(horizon):
                obs, _reward, terminated, truncated, _info = vec_env.step(
                    action_scaler.unscale(act_chunk[:, t + i])
                )
                if bool((terminated | truncated)[0].item()):
                    done_ep = True
                    break
            if done_ep:
                break
            t += horizon
        print(f"  action-chunk collect episode {ep + 1}/{n_episodes}: {n_windows} windows")
    if not chunks:
        raise ValueError("No action-chunk windows collected")
    actions = np.stack(chunks, axis=0).reshape(len(chunks), -1).astype(np.float64)
    if return_states:
        return actions, np.stack(states, axis=0).astype(np.float32)
    return actions


def fit_or_load_act_chunk_pca(
    *,
    vec_env: VectorizedEnvWrapper,
    base_policy: ACTPolicy,
    action_scaler,
    act_chunk_size: int,
    horizon: int,
    k: int,
    pca_path: str | None,
    pca_fit_episodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Return P with shape (k, horizon * action_dim). Residual is z @ P (no mean)."""
    if pca_path:
        path = Path(pca_path).expanduser().resolve()
        components = np.asarray(np.load(path)["components"], dtype=np.float32)
        if components.shape[0] < k:
            raise ValueError(f"PCA file {path} has {components.shape[0]} components, need k={k}")
        P = components[:k]
        print(f"Loaded ACT-chunk PCA from {path}: P={tuple(P.shape)}")
        return torch.as_tensor(P, device=device, dtype=torch.float32)

    print(
        f"Fitting ACT-chunk PCA on {pca_fit_episodes} episodes "
        f"(act_chunk={act_chunk_size}, horizon={horizon}, k={k})..."
    )
    X = collect_act_chunk_windows(
        vec_env=vec_env,
        base_policy=base_policy,
        action_scaler=action_scaler,
        act_chunk_size=act_chunk_size,
        horizon=horizon,
        n_episodes=pca_fit_episodes,
    )
    _u, s, vt = np.linalg.svd(X - X.mean(axis=0, keepdims=True), full_matrices=False)
    n_keep = min(k, vt.shape[0])
    if n_keep < k:
        print(f"WARNING: reducing k {k} -> {n_keep}")
        k = n_keep
    P = vt[:k].astype(np.float32)
    eig = (s[:k] ** 2) / max(X.shape[0] - 1, 1)
    total = float((s**2).sum() / max(X.shape[0] - 1, 1))
    ratios = eig / max(total, 1e-12)
    print(
        "  PCA explained variance (top k): "
        + ", ".join(f"{r:.3f}" for r in ratios)
        + f" (cumsum={float(np.cumsum(ratios)[-1]):.3f})"
    )
    return torch.as_tensor(P, device=device, dtype=torch.float32)


def fit_or_load_act_chunk_vae(
    *,
    vec_env: VectorizedEnvWrapper,
    base_policy: ACTPolicy,
    action_scaler,
    act_chunk_size: int,
    horizon: int,
    k: int,
    vae_path: str | None,
    pca_fit_episodes: int,
    device: torch.device,
    vae: bool = True,
    beta: float = 1e-3,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    condition_on_state: bool = False,
    state_standardizer=None,
) -> ResidualActionAE:
    """Train or load a SERL-style VAE on action chunks (not X-X' diffs).

    `condition_on_state` concatenates the standardized query-time proprio
    vector into the encoder and decoder.
    """
    if vae_path:
        path = Path(vae_path).expanduser().resolve()
        ae = ResidualActionAE.load(str(path), map_location="cpu")
        if ae.latent_dim != int(k):
            raise ValueError(f"VAE file {path} has k={ae.latent_dim}, need k={k}")
        expected = int(horizon) * int(vec_env.action_space.shape[-1])
        if ae.ac_chunk * ae.ac_dim != expected:
            raise ValueError(
                f"VAE file {path} raw dim {ae.ac_chunk}*{ae.ac_dim} != horizon*act_dim={expected}"
            )
        if bool(condition_on_state) and not ae.condition_on_state:
            print(f"WARNING: {path} is unconditional but vae_condition_on_state is set")
        if ae.condition_on_state and not bool(condition_on_state):
            print(f"WARNING: {path} conditions on state but vae_condition_on_state is false")
        return ae.to(device)

    if condition_on_state and state_standardizer is None:
        raise ValueError("condition_on_state requires state_standardizer")
    print(
        f"Fitting ACT-chunk {'VAE' if vae else 'AE'} on {pca_fit_episodes} episodes "
        f"(act_chunk={act_chunk_size}, horizon={horizon}, k={k}, beta={beta:g}, epochs={epochs}, "
        f"condition_on_state={bool(condition_on_state)})..."
    )
    collected = collect_act_chunk_windows(
        vec_env=vec_env,
        base_policy=base_policy,
        action_scaler=action_scaler,
        act_chunk_size=act_chunk_size,
        horizon=horizon,
        n_episodes=pca_fit_episodes,
        return_states=bool(condition_on_state),
    )
    states = None
    if condition_on_state:
        X, raw_states = collected
        state_t = torch.as_tensor(raw_states, dtype=torch.float32)
        states = state_standardizer.standardize(state_t).detach().cpu().numpy().astype(np.float32)
    else:
        X = collected
    env_action_dim = int(vec_env.action_space.shape[-1])
    ae = ResidualActionAE.fit(
        chunks=X.astype(np.float32),
        k=int(k),
        ac_chunk=int(horizon),
        ac_dim=env_action_dim,
        vae=bool(vae),
        beta=float(beta),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        seed=int(seed),
        device=device,
        states=states,
    )
    return ae.to(device)


class ChunkResidualVecEnvWrapper:
    """One RL step: residual over ACT[t:t+horizon], executed open-loop.

    ACT predicts act_chunk_size actions; residual windows of `horizon` walk that plan.
    If pca_basis is set, RL action is z in R^k and residual = z @ P.
    If vae_decoder is set, RL action is z in R^k and residual = decoder(z)*std
    (no mean). Reward is 1 if any inner step succeeded, else 0.
    """

    def __init__(
        self,
        vec_env: VectorizedEnvWrapper,
        base_policy: ACTPolicy,
        action_scaler,
        state_standardizer,
        act_chunk_size: int,
        horizon: int,
        pca_basis: torch.Tensor | None = None,
        vae_decoder: ResidualActionAE | None = None,
    ):
        if horizon <= 0 or act_chunk_size % horizon != 0:
            raise ValueError(f"horizon={horizon} must be a positive divisor of act_chunk_size={act_chunk_size}")
        if pca_basis is not None and vae_decoder is not None:
            raise ValueError("pca_basis and vae_decoder cannot be used together")
        self.vec_env = vec_env
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.act_chunk_size = int(act_chunk_size)
        self.horizon = int(horizon)
        self.env_action_dim = vec_env.action_space.shape[-1]
        self.base_action_dim = self.horizon * self.env_action_dim
        self.vae_decoder = vae_decoder
        if pca_basis is not None:
            if pca_basis.shape[1] != self.base_action_dim:
                raise ValueError(
                    f"PCA basis has shape {tuple(pca_basis.shape)}, expected (k, {self.base_action_dim})"
                )
            self.pca_basis = pca_basis.detach().float()
            self.rl_action_dim = int(pca_basis.shape[0])
        elif vae_decoder is not None:
            self.pca_basis = None
            self.rl_action_dim = int(vae_decoder.latent_dim)
        else:
            self.pca_basis = None
            self.rl_action_dim = self.base_action_dim

        batch_shape = vec_env.action_space.shape[:-1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(*batch_shape, self.rl_action_dim), dtype=np.float32
        )
        self.observation_space = vec_env.observation_space
        self._act_chunk_n: torch.Tensor | None = None
        self._t = 0
        self._condition_on_state = bool(
            vae_decoder is not None and getattr(vae_decoder, "condition_on_state", False)
        )
        self._query_vae_state: torch.Tensor | None = None

    def _decode_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim == 1:
            residual = residual.unsqueeze(0)
        if self.vae_decoder is not None:
            dec = self.vae_decoder
            if next(dec.parameters()).device != residual.device:
                dec = dec.to(residual.device)
            state = None
            if self._condition_on_state:
                if self._query_vae_state is None:
                    raise ValueError("State-conditioned VAE decode has no query state")
                state = self._query_vae_state.to(device=residual.device, dtype=residual.dtype)
                if state.shape[0] != residual.shape[0]:
                    raise ValueError(
                        f"Query state batch {state.shape[0]} != residual batch {residual.shape[0]}"
                    )
            flat = dec(residual, state)
        elif self.pca_basis is not None:
            P = self.pca_basis.to(device=residual.device, dtype=residual.dtype)
            flat = residual @ P
        else:
            flat = residual
        if flat.shape[-1] != self.base_action_dim:
            raise ValueError(f"Residual has last dim {flat.shape[-1]}, expected {self.base_action_dim}")
        return flat.view(flat.shape[0], self.horizon, self.env_action_dim)

    def _current_slice(self) -> torch.Tensor:
        assert self._act_chunk_n is not None
        return self._act_chunk_n[:, self._t : self._t + self.horizon]

    def _augment_obs(self, raw_obs: dict[str, torch.Tensor], base_n: torch.Tensor) -> dict[str, torch.Tensor]:
        augmented_obs = raw_obs.copy()
        augmented_obs["observation.base_action"] = base_n.reshape(base_n.shape[0], -1)
        augmented_obs["observation.state"] = self.state_standardizer.standardize(augmented_obs["observation.state"])
        if self._condition_on_state:
            if self._query_vae_state is None:
                raise ValueError("State-conditioned VAE has no query state")
            augmented_obs["observation.vae_state"] = self._query_vae_state
        return augmented_obs

    def _refresh_act_chunk(self, raw_obs: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            chunk_env = self.base_policy.predict_action_chunk(raw_obs)
        self._act_chunk_n = self.action_scaler.scale(chunk_env)
        self._t = 0
        if self._condition_on_state:
            self._query_vae_state = self.state_standardizer.standardize(raw_obs["observation.state"])

    def _process_final_obs_in_info(
        self, info: dict, device: torch.device, query_vae_state: torch.Tensor | None = None
    ) -> dict:
        if "final_obs" not in info or info["final_obs"] is None:
            return info
        for final_obs_dict in info["final_obs"]:
            if final_obs_dict is not None and "observation.state" in final_obs_dict:
                final_obs_dict["observation.base_action"] = torch.zeros(
                    self.base_action_dim, device=device, dtype=torch.float32
                )
                if self._condition_on_state and query_vae_state is not None:
                    final_obs_dict["observation.vae_state"] = query_vae_state[0].detach().to(device)
        return info

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        raw_obs, info = self.vec_env.reset(**kwargs)
        self.base_policy.reset()
        self._refresh_act_chunk(raw_obs)
        return self._augment_obs(raw_obs, self._current_slice()), info

    def step(self, residual_action: torch.Tensor):
        if residual_action.ndim == 1:
            residual_action = residual_action.unsqueeze(0)
        query_vae_state = self._query_vae_state
        base_n = self._current_slice()
        delta = self._decode_residual(residual_action)

        n_inner = 0
        success = torch.zeros(residual_action.shape[0], device=residual_action.device, dtype=torch.bool)
        last_info: dict = {}
        raw_obs = None
        terminated = truncated = None
        for i in range(self.horizon):
            combined_n = torch.clamp(base_n[:, i] + delta[:, i], -1.0, 1.0)
            raw_obs, reward, terminated, truncated, last_info = self.vec_env.step(
                self.action_scaler.unscale(combined_n)
            )
            success = success | (reward == 1.0)
            n_inner += 1
            if (terminated | truncated).any():
                break

        last_info["chunk_env_steps"] = int(n_inner)
        last_info["scaled_action"] = torch.clamp(base_n + delta, -1.0, 1.0).reshape(
            residual_action.shape[0], -1
        ).detach()

        done_mask = terminated | truncated
        if done_mask.any():
            self.base_policy.reset(env_ids=torch.where(done_mask)[0].tolist())
            self._refresh_act_chunk(raw_obs)
        else:
            self._t += self.horizon
            if self._t >= self.act_chunk_size:
                self._refresh_act_chunk(raw_obs)

        if "final_obs" in last_info:
            last_info = self._process_final_obs_in_info(
                last_info, residual_action.device, query_vae_state
            )
        return self._augment_obs(raw_obs, self._current_slice()), success.float(), terminated, truncated, last_info

    def render(self):
        return self.vec_env.render()

    def close(self):
        return self.vec_env.close()

    def __getattr__(self, name: str):
        return getattr(self.vec_env, name)
