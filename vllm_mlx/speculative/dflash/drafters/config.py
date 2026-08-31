# SPDX-License-Identifier: Apache-2.0
"""DFlash / DFlash2 drafter config — vendored into Rapid-MLX.

Vendored (with DFlash2 fields added) from mlx-vlm 0.6.x
``mlx_vlm/speculative/drafters/qwen3_dflash/config.py``.

**Why vendor:** the DFlash2 drafter (Qwen3.8-27B-DFlash2) uses two-tap
dynamic causal convs and a selector-lattice head that upstream mlx-vlm's
``qwen3_dflash`` did not implement — it rejected the checkpoint's
``attention_conv``/``mlp_conv`` and ``candidate_selector`` weights with
"not in model". Rather than patch site-packages (which a pip reinstall
of mlx-vlm would clobber), Rapid-MLX owns the DFlash2 drafter here.

Only the class used by the drafter loader is vendored; everything else
(``KVCache``, ``RotatingKVCache``, qwen3 ``MLP``, rope) is imported from
mlx-lm / mlx-vlm, which are already core/optional deps.
"""
import inspect
from dataclasses import dataclass, field
from typing import Any, List, Optional

from mlx_vlm.models.base import BaseModelConfig


@dataclass
class DFlashConfig(BaseModelConfig):
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rope_theta: float = 10000000.0
    rope_scaling: Optional[dict[str, Any]] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    block_size: int = 16
    mask_token_id: int = 248070
    target_layer_ids: List[int] = field(default_factory=lambda: [1, 8, 15, 22, 29])
    num_target_layers: int = 32
    layer_types: List[str] = field(default_factory=list)
    sliding_window: Optional[int] = None
    final_logit_softcapping: Optional[float] = None
    runtime_block_size: int | None = None
    draft_window_size: int | None = None
    # DFlash2 fields (Qwen3.8-27B-DFlash2). Present only on DFlash2
    # checkpoints; their absence selects the DFlash1 path.
    is_causal: bool = False
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 256
    selector_top_k: int = 16

    @classmethod
    def from_dict(cls, params: dict) -> "DFlashConfig":
        flat = dict(params)
        dflash_cfg = flat.pop("dflash_config", None) or {}
        if "mask_token_id" in dflash_cfg:
            flat["mask_token_id"] = dflash_cfg["mask_token_id"]
        if "target_layer_ids" in dflash_cfg:
            flat["target_layer_ids"] = list(dflash_cfg["target_layer_ids"])
        if "runtime_block_size" in dflash_cfg:
            flat["runtime_block_size"] = dflash_cfg["runtime_block_size"]
        if "draft_window_size" in dflash_cfg:
            flat["draft_window_size"] = dflash_cfg["draft_window_size"]
        # DFlash2: pull block_size + conv/selector fields out of dflash_config.
        for key in (
            "block_size",
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ):
            if key in dflash_cfg:
                flat[key] = dflash_cfg[key]
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
