"""Shared model class for residual latent models.

A new method subclasses LatentModel, implements fit and decode, and
registers itself. The trainer, wrapper, and critic only call decode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn


class LatentModel(nn.Module, ABC):
    """Maps actor latents z to a chunk residual in scaled action space."""

    name: str = ""
    latent_dim: int
    action_dim: int

    @abstractmethod
    def fit(self, chunks: np.ndarray, states: np.ndarray | None = None) -> None:
        """Fit on base-policy windows chunks of shape (N, H * A).

        states is the query-time proprio for each window, shape (N, S).
        Models that ignore state still accept it.
        """

    @abstractmethod
    def decode(self, z: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Map latents (B, K) to residuals (B, H * A)."""

    def load_ckpt(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement load_ckpt")


def decode_latent(
    model: LatentModel | None,
    z: torch.Tensor,
    state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Identity when no latent model is fitted."""
    if model is None:
        return z
    return model.decode(z, state)
