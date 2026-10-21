"""Action-chunk AE/VAE trained on actions (not X-X' diffs).

Residual decode is decoder(z) * std. Mean is omitted because a residual has
mean 0 by definition; training still uses per-dim mean/std for stability.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from resfit.rl_finetuning.autoencoder.model import MLPAutoencoder, gaussian_kl


class ResidualActionAE(nn.Module):
    """Fitted SERL-style action AE/VAE. ``forward(z)`` returns a residual."""

    def __init__(
        self,
        model: MLPAutoencoder,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
        ac_chunk: int,
        ac_dim: int,
        vae: bool = True,
        beta: float = 0.0,
        epochs: int = 0,
        condition_on_state: bool = False,
        state_mean: np.ndarray | torch.Tensor | None = None,
        state_std: np.ndarray | torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(-1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).reshape(-1))
        self.ac_chunk = int(ac_chunk)
        self.ac_dim = int(ac_dim)
        self.vae = bool(vae)
        self.beta = float(beta)
        self.epochs = int(epochs)
        self.condition_on_state = bool(condition_on_state)
        if self.condition_on_state:
            if state_mean is None or state_std is None:
                raise ValueError("condition_on_state requires state_mean and state_std")
            s_mean = torch.as_tensor(state_mean, dtype=torch.float32).reshape(-1)
            s_std = torch.as_tensor(state_std, dtype=torch.float32).reshape(-1)
        else:
            s_mean = torch.zeros(0, dtype=torch.float32)
            s_std = torch.ones(0, dtype=torch.float32)
        self.register_buffer("state_mean", s_mean)
        self.register_buffer("state_std", s_std)
        raw_dim = self.ac_chunk * self.ac_dim
        if int(self.mean.numel()) != raw_dim or int(self.std.numel()) != raw_dim:
            raise ValueError(
                f"AE dim mismatch: mean {tuple(self.mean.shape)} std {tuple(self.std.shape)} "
                f"expected raw_dim={raw_dim}"
            )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def latent_dim(self) -> int:
        return int(self.model.latent_dim)

    @property
    def cond_dim(self) -> int:
        return int(self.state_mean.numel()) if self.condition_on_state else 0

    def forward(self, z: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Decode latent coords to a residual: decoder(z[, state]) * std (no mean)."""
        if z.ndim == 1:
            z = z.unsqueeze(0)
        cond = None
        if self.condition_on_state:
            if state is None:
                raise ValueError("state is required when vae_condition_on_state is set")
            if state.ndim == 1:
                state = state.unsqueeze(0)
            state = state.to(device=z.device, dtype=z.dtype)
            if state.shape[-1] != self.cond_dim:
                raise ValueError(f"State dim {state.shape[-1]} != VAE cond_dim {self.cond_dim}")
            cond = (state - self.state_mean.to(device=z.device, dtype=z.dtype)) / self.state_std.to(
                device=z.device, dtype=z.dtype
            )
        recon = self.model.decode(z, cond)
        return recon * self.std.to(device=z.device, dtype=z.dtype)

    @classmethod
    def fit(
        cls,
        chunks: np.ndarray,
        k: int,
        ac_chunk: int,
        ac_dim: int,
        vae: bool = True,
        beta: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        val_frac: float = 0.1,
        seed: int = 0,
        hidden_dims: Sequence[int] = (256, 256),
        device: Optional[str | torch.device] = None,
        states: np.ndarray | None = None,
    ) -> "ResidualActionAE":
        x = np.asarray(chunks, dtype=np.float32)
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        if x.ndim != 2:
            raise ValueError(f"chunks must be (N, D) or (N, T, A), got {x.shape}")
        ac_chunk, ac_dim, k = int(ac_chunk), int(ac_dim), int(k)
        raw_dim = ac_chunk * ac_dim
        if x.shape[1] != raw_dim:
            raise ValueError(f"chunk dim {x.shape[1]} != {ac_chunk}*{ac_dim}={raw_dim}")
        n = int(x.shape[0])
        if n < 2:
            raise ValueError(f"Need at least 2 action chunks to fit AE, got {n}")

        rng = np.random.default_rng(int(seed))
        order = rng.permutation(n)
        n_val = min(max(int(round(n * float(val_frac))), 1), n - 1) if val_frac > 0 else 0
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        train = x[train_idx]
        val = x[val_idx] if n_val else np.zeros((0, raw_dim), dtype=np.float32)
        mean = train.mean(axis=0).astype(np.float32)
        std = np.maximum(train.std(axis=0).astype(np.float32), 1e-6)
        train_n = (train - mean) / std
        val_n = (val - mean) / std if n_val else val

        condition_on_state = states is not None
        state_mean = np.zeros((0,), dtype=np.float32)
        state_std = np.ones((0,), dtype=np.float32)
        train_s = val_s = None
        cond_dim = 0
        if condition_on_state:
            states = np.asarray(states, dtype=np.float32)
            if states.ndim == 1:
                states = states.reshape(-1, 1)
            if states.shape[0] != n:
                raise ValueError(f"Got {states.shape[0]} states for {n} chunks")
            state_mean = states[train_idx].mean(axis=0).astype(np.float32)
            state_std = np.maximum(states[train_idx].std(axis=0).astype(np.float32), 1e-6)
            states_n = (states - state_mean) / state_std
            train_s = states_n[train_idx]
            val_s = states_n[val_idx] if n_val else np.zeros((0, states.shape[1]), dtype=np.float32)
            cond_dim = int(state_mean.shape[0])

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        torch.manual_seed(int(seed))
        model = MLPAutoencoder(
            input_dim=raw_dim,
            latent_dim=k,
            hidden_dims=hidden_dims,
            vae=bool(vae),
            cond_dim=cond_dim,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        kl_weight = float(beta) if vae else 0.0
        kind = "VAE" if vae else "AE"
        mean_t = torch.from_numpy(mean).to(device)
        std_t = torch.from_numpy(std).to(device)

        def _run_epoch(data: np.ndarray, cond_data: np.ndarray | None, train_mode: bool) -> dict:
            if data.shape[0] == 0:
                return {}
            model.train(mode=train_mode)
            acc = []
            idx = rng.permutation(len(data)) if train_mode else np.arange(len(data))
            for i in range(0, len(data), int(batch_size)):
                sel = idx[i : i + int(batch_size)]
                batch = torch.from_numpy(data[sel]).to(device)
                cond = None if cond_data is None else torch.from_numpy(cond_data[sel]).to(device)
                with torch.set_grad_enabled(train_mode):
                    recon, mu, logvar = model(batch, sample=bool(vae and train_mode), cond=cond)
                    mse = F.mse_loss(recon, batch)
                    kl = gaussian_kl(mu, logvar) if vae else recon.new_zeros(())
                    loss = mse + kl_weight * kl
                    if train_mode:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                recon_raw = recon.detach() * std_t + mean_t
                x_raw = batch.detach() * std_t + mean_t
                err = recon_raw - x_raw
                var_x = torch.var(x_raw)
                acc.append(
                    dict(
                        loss=float(loss.detach()),
                        recon_loss=float(torch.mean(err**2)),
                        kl=float(kl.detach()),
                        explained_var=float(1.0 - torch.var(err) / torch.clamp(var_x, min=1e-12)),
                        n=int(batch.shape[0]),
                    )
                )
            total = sum(m["n"] for m in acc)
            keys = [key for key in acc[0] if key != "n"]
            return {key: sum(m[key] * m["n"] for m in acc) / total for key in keys}

        cond_msg = f"  state_dim={cond_dim}" if condition_on_state else ""
        print(
            f"pretrain {kind} on actions (not diffs)  train/val={len(train_n)}/{len(val_n)}  "
            f"dim={raw_dim} K={k} epochs={int(epochs)} beta={kl_weight:g} device={device}  "
            f"residual decode = decoder(z)*std (mean=0){cond_msg}"
        )
        for epoch in range(1, int(epochs) + 1):
            tr = _run_epoch(train_n, train_s, train_mode=True)
            va = _run_epoch(val_n, val_s, train_mode=False)
            print(
                f"epoch {epoch:4d}  "
                f"train_loss={tr.get('loss', float('nan')):.5f}  "
                f"val_loss={va.get('loss', float('nan')):.5f}  "
                f"val_recon={va.get('recon_loss', float('nan')):.5f}  "
                f"val_kl={va.get('kl', float('nan')):.4f}  "
                f"val_ev={va.get('explained_var', float('nan')):.4f}"
            )

        model.eval()
        model.cpu()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(
            model=model,
            mean=mean,
            std=std,
            ac_chunk=ac_chunk,
            ac_dim=ac_dim,
            vae=bool(vae),
            beta=kl_weight,
            epochs=int(epochs),
            condition_on_state=condition_on_state,
            state_mean=state_mean,
            state_std=state_std,
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(
            dict(
                state_dict=self.model.state_dict(),
                mean=self.mean.detach().cpu().numpy(),
                std=self.std.detach().cpu().numpy(),
                latent_dim=np.int32(self.latent_dim),
                input_dim=np.int32(int(self.mean.numel())),
                ac_chunk=np.int32(self.ac_chunk),
                ac_dim=np.int32(self.ac_dim),
                hidden_dims=np.asarray(self.model.hidden_dims, dtype=np.int32),
                vae=bool(self.vae),
                beta=np.float32(self.beta),
                epochs=np.int32(self.epochs),
                condition_on_state=bool(self.condition_on_state),
                state_mean=self.state_mean.detach().cpu().numpy(),
                state_std=self.state_std.detach().cpu().numpy(),
                cond_dim=np.int32(self.cond_dim),
            ),
            path,
        )
        print(f"Saved ResidualActionAE to {path}")

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> "ResidualActionAE":
        data = torch.load(path, map_location=map_location, weights_only=False)
        hidden = tuple(int(x) for x in np.asarray(data["hidden_dims"]).tolist())
        condition_on_state = bool(data.get("condition_on_state", False))
        model = MLPAutoencoder(
            input_dim=int(data["input_dim"]),
            latent_dim=int(data["latent_dim"]),
            hidden_dims=hidden,
            vae=bool(data["vae"]),
            cond_dim=int(data.get("cond_dim", 0)),
        )
        model.load_state_dict(data["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        ae = cls(
            model=model,
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            ac_chunk=int(data["ac_chunk"]),
            ac_dim=int(data["ac_dim"]),
            vae=bool(data["vae"]),
            beta=float(data.get("beta", 0.0)),
            epochs=int(data.get("epochs", 0)),
            condition_on_state=condition_on_state,
            state_mean=np.asarray(data["state_mean"], dtype=np.float32) if condition_on_state else None,
            state_std=np.asarray(data["state_std"], dtype=np.float32) if condition_on_state else None,
        )
        print(
            f"Loaded ResidualActionAE from {path}  k={ae.latent_dim} "
            f"raw=({ae.ac_chunk},{ae.ac_dim}) vae={ae.vae} beta={ae.beta:g}  "
            f"condition_on_state={ae.condition_on_state} state_dim={ae.cond_dim}  "
            f"residual decode = decoder(z)*std (mean=0)"
        )
        return ae
