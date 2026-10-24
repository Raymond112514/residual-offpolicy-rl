"""Register latent models and construct the one named in the config."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from resfit.rl_finetuning.latent_rl.base import LatentModel

_REGISTRY: dict[str, type[LatentModel]] = {}
_imported = False
_SKIP = {"base", "registry", "config"}


def register(name: str):
    """Decorator: ``@register("pca")`` on a ``LatentModel`` subclass."""

    def deco(cls: type[LatentModel]) -> type[LatentModel]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"latent model {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _import_models() -> None:
    global _imported
    if _imported:
        return
    package = importlib.import_module("resfit.rl_finetuning.latent_rl")
    for mod in pkgutil.iter_modules(package.__path__):
        if mod.name in _SKIP or mod.name.startswith("_"):
            continue
        importlib.import_module(f"resfit.rl_finetuning.latent_rl.{mod.name}")
    _imported = True


def build_latent(cfg: Any, *, action_dim: int) -> LatentModel | None:
    """Construct an unfitted model, or None when ``cfg.name`` is ``none``."""
    name = str(getattr(cfg, "name", "none") or "none")
    if name in ("none", "None"):
        return None
    _import_models()
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown latent model {name!r}. Known models: {known}")
    return _REGISTRY[name](cfg, action_dim=int(action_dim))
