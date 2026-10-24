"""Latent residual models.
"""

from resfit.rl_finetuning.latent_rl.base import LatentModel, decode_latent
from resfit.rl_finetuning.latent_rl.registry import build_latent, register

__all__ = ["LatentModel", "build_latent", "decode_latent", "register"]
