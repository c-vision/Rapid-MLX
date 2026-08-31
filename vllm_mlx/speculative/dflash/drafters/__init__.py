# SPDX-License-Identifier: Apache-2.0
"""Vendored DFlash / DFlash2 drafter loader.

Loads the Qwen3.8-27B-DFlash2 (and DFlash1) draft model straight from a
local checkpoint without depending on upstream mlx-vlm's incomplete
``qwen3_dflash`` drafter. ``DFlashDraftModel`` auto-detects DFlash2 from
the checkpoint config (``dflash_config.conv_group_size``) and builds the
conv + selector-lattice path.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx

from .config import DFlashConfig
from .qwen3_dflash import DFlashDraftModel

logger = logging.getLogger(__name__)

_WEIGHT_NAMES = ("model.safetensors", "model-*.safetensors")


def is_dflash2_repo(model_path: str | Path) -> bool:
    """True if ``model_path`` is a DFlash2 checkpoint (has conv_group_size)."""
    cfg_path = Path(model_path) / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        return False
    dflash_cfg = cfg.get("dflash_config") or {}
    return int(dflash_cfg.get("conv_group_size", 0) or 0) > 0


def load_drafter(model_path: str | Path, kind: str = "dflash"):
    """Build and load a DFlash/DFlash2 draft model from a local checkpoint.

    Returns ``(model, resolved_kind)`` matching mlx-vlm's ``load_drafter``
    contract, so the existing DFlash runtime and ``BatchedEngine`` path need
    no changes: the model exposes ``draft_block``, ``config.target_layer_ids``,
    ``reset`` and ``make_cache``, which the (mlx-vlm) round loop drives.
    """
    model_path = Path(model_path)
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"DFlash drafter missing config.json at {model_path}")
    config = DFlashConfig.from_dict(json.loads((model_path / "config.json").read_text()))

    model = DFlashDraftModel(config)
    weights = _load_weights(model_path)
    model.load_weights(model.sanitize(weights), strict=True)
    # Materialize eagerly so the server doesn't pay a lazy-eval spike on the
    # first draft forward.
    mx.eval(model.parameters())
    resolved = "dflash2" if is_dflash2_repo(model_path) else kind
    logger.info(
        "Vendored DFlash drafter loaded: %s (%s, %d tensors)",
        model_path,
        resolved,
        len(weights),
    )
    # Keep ``kind`` (e.g. "dflash") as the resolved round-loop kind — the
    # dispatcher only knows dflash/eagle3/mtp; DFlash2 uses the DFlash1 loop.
    return model, kind


def _load_weights(model_path: Path) -> dict[str, mx.array]:
    """Load safetensors weights, handling single-file and sharded layouts."""
    shards = sorted(model_path.glob("model-*.safetensors"))
    if shards:
        return mx.load_shard(str(model_path), list_of_shards=[p.name for p in shards])
    single = model_path / "model.safetensors"
    if single.exists():
        return mx.load(str(single))
    raise FileNotFoundError(
        f"DFlash drafter has no safetensors weights at {model_path}"
    )


__all__ = [
    "DFlashConfig",
    "DFlashDraftModel",
    "load_drafter",
    "is_dflash2_repo",
]
