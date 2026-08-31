# SPDX-License-Identifier: Apache-2.0
"""DFlash / DFlash2 drafter model — vendored into Rapid-MLX.

Vendored and extended from mlx-vlm 0.6.x
``mlx_vlm/speculative/drafters/qwen3_dflash/dflash.py``.

Upstream's ``qwen3_dflash`` implements the **DFlash1** draft (a plain
``fc`` context projection + lm_head logits) but not DFlash2. The
Qwen3.8-27B-DFlash2 checkpoint additionally needs:
  - two-tap **GroupedDynamicCausalConv** (``attention_conv``/``mlp_conv``)
  - a **CandidateSelector** lattice head (predecessor/successor codebooks
    + hidden_projection, top-K candidates, greedy path walk)

This file adds those. ``DFlashDraftModel`` auto-detects DFlash2 from the
config (``conv_group_size > 0``) and builds the DFlash2 path (conv layers
+ candidate_selector) without affecting DFlash1 (gemma4-style) drafters.

The round-loop orchestration (``_dflash_rounds``) is still mlx-vlm's and
is agnostic to the drafter internals — it only needs ``draft_block``,
``config.target_layer_ids``, ``reset`` and ``make_cache``, which this
model provides.
"""
from typing import List

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import MLP as Qwen3MLP
from mlx_lm.models.rope_utils import initialize_rope

from mlx_vlm.models.cache import BufferedRotatingKVCache, RotatingKVCache

from .config import DFlashConfig


def _build_rope(config: DFlashConfig):
    return initialize_rope(
        dims=config.head_dim,
        base=config.rope_theta,
        traditional=False,
        scaling_config=config.rope_scaling,
        max_position_embeddings=config.max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        layer_types = (
            config.layer_types or ["full_attention"] * config.num_hidden_layers
        )
        self.is_sliding = layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, x_ctx: mx.array, rope, cache: KVCache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        if self.is_sliding:
            if self.sliding_window is None:
                raise ValueError(
                    "DFlash draft config must define sliding_window for sliding layers."
                )
            keep_ctx = self.sliding_window - 1
            if S > keep_ctx:
                skip = S - keep_ctx
                x_ctx = x_ctx[:, skip:]
                S = x_ctx.shape[1]
                cache.offset += skip

        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        # DFlash denoises the whole proposed block at once, so draft-block
        # self-attention is intentionally non-causal. Sliding layers already
        # limit resident prefix context through the rotating cache above.
        mask = None
        o = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


# ---------------------------------------------------------------------------
# DFlash2 (Qwen3.8-27B-DFlash2): two-tap dynamic causal conv + selector lattice
# ---------------------------------------------------------------------------


def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    """Two-tap grouped dynamic causal convolution.

    ``base`` has shape (kernel_size, hidden_size). Each of the ``kernel_size``
    taps applies ``base[offset] * x[t-offset] + dynamic[t, offset] * x[t-offset]``
    elementwise per group (a depthwise conv with a per-token dynamic kernel
    added on top of the static ``base_kernel``).
    """
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // int(group_size)
    blocks = hidden.reshape(batch, length, groups, int(group_size))
    dynamic = dynamic.reshape(batch, length, int(base.shape[0]), groups, 1)
    output = mx.zeros_like(blocks)
    for offset in range(int(base.shape[0])):
        if offset == 0:
            values = blocks
        else:
            values = mx.concatenate(
                (mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]),
                axis=1,
            )
        kernel = base[offset].reshape(1, 1, groups, int(group_size)).astype(hidden.dtype)
        output = output + kernel * values + dynamic[:, :, offset] * values
    return output.reshape(hidden.shape)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        groups = int(hidden_size) // self.group_size
        self.base_kernel = mx.zeros((2, self.kernel_size, int(hidden_size)))
        self.kernel_projection = nn.Linear(
            int(hidden_size),
            2 * self.kernel_size * groups,
            bias=False,
        )

    def prepare(self, hidden: mx.array) -> tuple:
        groups = int(hidden.shape[-1]) // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1],
            2,
            self.kernel_size,
            groups,
        )
        return (
            _grouped_dynamic_convolve(
                hidden,
                dynamic[..., 0, :, :],
                self.base_kernel[0],
                self.group_size,
            ),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden: mx.array, dynamic: mx.array) -> mx.array:
        return _grouped_dynamic_convolve(
            hidden,
            dynamic,
            self.base_kernel[1],
            self.group_size,
        )


class DFlash2DecoderLayer(nn.Module):
    """Qwen3.8 DFlash2 draft layer: qwen3 self-attn + two-tap conv gates."""

    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )

    def __call__(self, x, x_ctx, rope, cache):
        residual = x
        h, dyn = self.attention_conv.prepare(self.input_layernorm(x))
        h = residual + self.attention_conv.finish(
            self.self_attn(h, x_ctx, rope, cache), dyn
        )
        residual = h
        h, dyn = self.mlp_conv.prepare(self.post_attention_layernorm(h))
        return residual + self.mlp_conv.finish(self.mlp(h), dyn)


class CandidateSelector(nn.Module):
    """DFlash2 selector lattice head.

    ``S_t(a, b) = U_t(b) + <A(a) (.) H(h_t), B(b)>``
    Top-``selector_top_k`` candidates per position, then a greedy walk from
    the anchor token picking the most coherent path through the lattice.
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.top_k = int(config.selector_top_k)
        self.predecessor_codebook = nn.Embedding(
            config.vocab_size, int(config.selector_rank)
        )
        self.successor_codebook = nn.Embedding(
            config.vocab_size, int(config.selector_rank)
        )
        self.hidden_projection = nn.Linear(
            config.hidden_size, int(config.selector_rank), bias=False
        )

    def _edge_scores(self, predecessor_ids, successor_vectors, hidden):
        return mx.sum(
            self.predecessor_codebook(predecessor_ids)[:, :, None]
            * hidden[:, None, None]
            * successor_vectors[:, None],
            axis=-1,
        )

    def select(self, hidden, logits, anchor_ids, temperature: float = 0.0):
        """Return the greedy-walk block tokens, shape [B, block_len-1]."""
        candidates = mx.argpartition(logits, -self.top_k, axis=-1)[..., -self.top_k :]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        hidden_proj = self.hidden_projection(hidden)
        successors = self.successor_codebook(candidates)
        predecessor = anchor_ids.reshape(-1)
        path = []
        for position in range(int(hidden.shape[1])):
            edges = self._edge_scores(
                predecessor[:, None],
                successors[:, position],
                hidden_proj[:, position],
            )[:, 0]
            scores = unary[:, position] + edges
            if float(temperature) > 0:
                probs = mx.softmax(scores.astype(mx.float32) / float(temperature), axis=-1)
                selected = mx.random.categorical(mx.log(probs))
            else:
                selected = mx.argmax(scores, axis=-1)
            predecessor = mx.take_along_axis(
                candidates[:, position], selected[:, None], axis=-1
            )[:, 0]
            path.append(predecessor)
        return mx.stack(path, axis=1)


def _is_dflash2(config: DFlashConfig) -> bool:
    return int(getattr(config, "conv_group_size", 0) or 0) > 0


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not self.config.layer_types:
            self.config.layer_types = ["full_attention"] * self.config.num_hidden_layers
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        layer_cls = DFlash2DecoderLayer if _is_dflash2(config) else DFlashDecoderLayer
        self.layers = [layer_cls(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(config)
        self.embed_tokens = None
        self.embed_scale = 1.0
        self.lm_head = None
        if _is_dflash2(config):
            self.candidate_selector = CandidateSelector(config)
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model) -> "DFlashDraftModel":
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        else:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = getattr(
            self.embed_tokens, "embed_scale", getattr(inner, "embed_scale", 1.0)
        )
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self.embed_tokens.as_linear
        )
        return self

    def make_cache(self) -> List[KVCache]:
        window = getattr(self.config, "draft_window_size", None)
        if window is not None and int(window) > 0:
            return [
                BufferedRotatingKVCache(max_size=int(window), buffer_size=64)
                for _ in self.layers
            ]
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                if self.config.sliding_window is None:
                    raise ValueError(
                        "DFlash draft config must define sliding_window for sliding layers."
                    )
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    def reset(self, target_model) -> List[KVCache]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        return self.make_cache()

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List[KVCache],
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        mask_id = int(self.config.mask_token_id)
        if isinstance(last_bonus, int):
            block = mx.array(
                [[last_bonus] + [mask_id] * (block_size - 1)],
                dtype=token_dtype,
            )
        else:
            B = last_bonus.shape[0]
            masks = mx.full((B, block_size - 1), mask_id, dtype=token_dtype)
            block = mx.concatenate(
                [last_bonus[:, None].astype(token_dtype), masks], axis=1
            )
        draft_hidden = self._hidden(block, hidden, cache)
        draft_logits = self._logits(draft_hidden[:, 1:])
        if _is_dflash2(self.config):
            anchor_ids = block[:, 0].astype(mx.int32)
            return self.candidate_selector.select(
                draft_hidden[:, 1:], draft_logits, anchor_ids, temperature=0.0
            )
        return sampler(draft_logits)

    def _hidden(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: List[KVCache],
    ) -> mx.array:
        h = self._embed_input_tokens(inputs)
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        return self.norm(h)

    def _embed_input_tokens(self, inputs: mx.array) -> mx.array:
        return self.embed_tokens(inputs) * self.embed_scale

    def _logits(self, hidden: mx.array) -> mx.array:
        logits = self.lm_head(hidden)
        if self.config.final_logit_softcapping is not None:
            softcap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    def __call__(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: List[KVCache],
    ) -> mx.array:
        return self._logits(self._hidden(inputs, target_hidden, cache))

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[len("model.") :]
            if k == "candidate_selector.predecessor_codebook":
                k = "candidate_selector.predecessor_codebook.weight"
            elif k == "candidate_selector.successor_codebook":
                k = "candidate_selector.successor_codebook.weight"
            out[k] = v
        return out


DFlashKVCache = KVCache
