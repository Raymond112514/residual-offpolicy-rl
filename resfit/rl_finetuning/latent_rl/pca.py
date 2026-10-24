"""PCA on base-policy action chunks.

Fit centers the windows and keeps the top-k right singular vectors. 
Decode does not add the mean back. The actor moves along principal directions of the chunks,
and the residual is z @ P.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from resfit.rl_finetuning.latent_rl.base import LatentModel
from resfit.rl_finetuning.latent_rl.registry import register


@register("pca")
class PCALatent(LatentModel):
    def __init__(self, cfg: Any, *, action_dim: int):
        super().__init__()
        k = int(cfg.k)
        if k < 1:
            raise ValueError(f"PCA k must be positive, got {k}")
        self.action_dim = int(action_dim)
        self.latent_dim = k
        self.explained_variance_ratio: np.ndarray | None = None
        self.register_buffer("components", torch.zeros(k, self.action_dim))
        self._fitted = False

    def fit(self, chunks: np.ndarray, states: np.ndarray | None = None) -> None:
        del states  # PCA is unconditional. The argument is here for the shared contract.
        data = np.asarray(chunks, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected chunks of shape (N, D), got {data.shape}")
        n, d = data.shape
        if d != self.action_dim:
            raise ValueError(f"Chunk dim {d} does not match action dim {self.action_dim}")
        if n < 2:
            raise ValueError(f"Need at least 2 action windows for PCA, got {n}")

        k = min(self.latent_dim, n - 1, d)
        if k < 1:
            raise ValueError(f"Invalid PCA k={self.latent_dim} for data shape {data.shape}")
        if k != self.latent_dim:
            print(f"[pca] reducing k from {self.latent_dim} to {k} for shape {data.shape}")
            self.latent_dim = k

        centered = data - data.mean(axis=0, keepdims=True)
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
        components = torch.as_tensor(vt[:k].astype(np.float32))
        self.register_buffer("components", components)

        eig = (s[:k] ** 2) / max(n - 1, 1)
        total = float((s**2).sum() / max(n - 1, 1))
        ratios = eig / max(total, 1e-12)
        self.explained_variance_ratio = ratios.astype(np.float32)
        self._fitted = True
        print(
            "[pca] explained variance (top k): "
            + ", ".join(f"{r:.3f}" for r in ratios)
            + f" (cumsum={float(np.cumsum(ratios)[-1]):.3f}) on N={n} D={d}"
        )

    def decode(self, z: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        del state
        if not self._fitted:
            raise RuntimeError("PCALatent.decode called before fit")
        squeezed = z.ndim == 1
        if squeezed:
            z = z.unsqueeze(0)
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {tuple(z.shape)}")
        basis = self.components[: self.latent_dim].to(device=z.device, dtype=z.dtype)
        residual = z @ basis
        return residual.squeeze(0) if squeezed else residual

    def load_ckpt(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        components = torch.as_tensor(ckpt["components"], dtype=torch.float32)
        if components.ndim != 2 or int(components.shape[1]) != self.action_dim:
            raise ValueError(
                f"PCA checkpoint {path} has components {tuple(components.shape)}, "
                f"expected (*, {self.action_dim})"
            )
        self.latent_dim = int(components.shape[0])
        self.register_buffer("components", components)
        ratios = ckpt.get("explained_variance_ratio")
        self.explained_variance_ratio = None if ratios is None else np.asarray(ratios, dtype=np.float32)
        self._fitted = True
