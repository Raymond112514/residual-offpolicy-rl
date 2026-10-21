"""Torch port of serl/autoencoder/model.py: Dense+ReLU MLP, Xavier, optional VAE."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _xavier_linear(in_dim: int, out_dim: int) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """Flax-style MLP: ReLU after every layer except the last."""

    def __init__(self, in_dim: int, dims: Sequence[int], activate_final: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = int(in_dim)
        n = len(dims)
        for i, d_out in enumerate(dims):
            layers.append(_xavier_linear(d_in, int(d_out)))
            if i + 1 < n or activate_final:
                layers.append(nn.ReLU())
            d_in = int(d_out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPAutoencoder(nn.Module):
    """Same architecture as serl.autoencoder.model.MLPAutoencoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        vae: bool = False,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        self.vae = bool(vae)
        self.cond_dim = int(cond_dim)
        enc_out = 2 * self.latent_dim if self.vae else self.latent_dim
        self.encoder = MLP(self.input_dim + self.cond_dim, list(self.hidden_dims) + [enc_out])
        self.decoder = MLP(
            self.latent_dim + self.cond_dim, list(reversed(self.hidden_dims)) + [self.input_dim]
        )

    def _concat_cond(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if self.cond_dim == 0:
            return x
        if cond is None:
            raise ValueError("State conditioning is enabled but no state was passed")
        return torch.cat([x, cond], dim=-1)

    def encode_mu(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(self._concat_cond(x, cond))
        if self.vae:
            mu, _logvar = torch.chunk(h, 2, dim=-1)
            return mu
        return h

    def decode(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(self._concat_cond(z, cond))

    def forward(
        self, x: torch.Tensor, sample: bool = False, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(self._concat_cond(x, cond))
        if self.vae:
            mu, logvar = torch.chunk(h, 2, dim=-1)
            logvar = logvar.clamp(-20.0, 10.0)
            if sample:
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            else:
                z = mu
        else:
            mu = h
            logvar = torch.zeros_like(mu)
            z = mu
        return self.decode(z, cond), mu, logvar


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
