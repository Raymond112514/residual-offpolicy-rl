from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LatentConfig:
    # name specifies the latent model to use, so far none and pca are supported
    name: str = "none"
    k: int = 8
    ckpt: str | None = None
