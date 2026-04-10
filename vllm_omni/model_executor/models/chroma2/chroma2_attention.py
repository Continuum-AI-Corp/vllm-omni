# SPDX-License-Identifier: Apache-2.0
"""
Chroma2 MergedAttention adapted for vLLM inference.

The original Chroma2MergedAttention concatenates self-attention KV and
cross-attention KV along the sequence dimension, then performs a single
attention computation. This module replicates that behavior exactly,
using flash_attn for efficiency instead of vLLM's PagedAttention.

Key differences from the HF implementation:
- Self-attention KV cache is managed as a pre-allocated tensor (not DynamicCache)
- Cross-attention KV is computed once at prefill and cached as a fixed tensor
- Uses flash_attn_func for the merged attention computation
- No attention weight output (inference only, no need for weight inspection)

Mathematical equivalence:
  Q = rope(q_norm(q_proj(decoder_hidden)))
  self_K = rope(k_norm(k_proj(decoder_hidden)))  # with RoPE
  self_V = v_proj(decoder_hidden)
  cross_K = k_norm(k_proj(encoder_hidden))        # no RoPE
  cross_V = v_proj(encoder_hidden)
  full_K = cat([self_K, cross_K], dim=seq)
  full_V = cat([self_V, cross_V], dim=seq)
  output = softmax(Q @ full_K^T / sqrt(d)) @ full_V
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chroma2RMSNorm(nn.Module):
    """RMSNorm matching the HF Chroma2 implementation (1+weight scaling)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K tensors.

    Matches the HF transformers implementation used by Chroma2.

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos: [seq_len, head_dim] or [1, 1, seq_len, head_dim]
        sin: [seq_len, head_dim] or [1, 1, seq_len, head_dim]
    """
    # Ensure cos/sin have proper shape for broadcasting
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class SelfKVCache:
    """Simple pre-allocated KV cache for self-attention.

    Not paged — just a contiguous tensor that grows with each decode step.
    For Chroma2's decoder scale (~24 layers, ~1000 tokens max), the memory
    overhead vs PagedAttention is negligible (~24MB per request).
    """

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Pre-allocate [num_layers, batch=1, num_kv_heads, max_seq_len, head_dim]
        # Batch dimension is handled externally; each request has its own cache.
        self.k_cache = torch.zeros(
            num_layers, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )
        self.v_cache = torch.zeros(
            num_layers, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )
        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new KV to cache and return full cached KV.

        Args:
            layer_idx: Which layer to update
            key_states: [num_kv_heads, new_seq_len, head_dim]
            value_states: [num_kv_heads, new_seq_len, head_dim]

        Returns:
            (cached_keys, cached_values) up to current seq_len + new_seq_len
        """
        new_len = key_states.shape[1]
        start = self.seq_len if layer_idx == 0 else self.seq_len  # all layers at same position
        end = start + new_len

        self.k_cache[layer_idx, :, start:end, :] = key_states
        self.v_cache[layer_idx, :, start:end, :] = value_states

        # Only increment seq_len once (after the last layer or first layer)
        # The caller must call advance_seq_len() after all layers are done.
        return (
            self.k_cache[layer_idx, :, :end, :],
            self.v_cache[layer_idx, :, :end, :],
        )

    def advance_seq_len(self, num_new_tokens: int):
        """Call after all layers have been updated for this step."""
        self.seq_len += num_new_tokens

    def reset(self):
        self.seq_len = 0


class Chroma2MergedAttentionVLLM(nn.Module):
    """Merged self + cross attention for Chroma2 decoder, adapted for vLLM.

    Replicates the exact computation of HF Chroma2MergedAttention:
    1. Q, self_K, self_V from decoder hidden states (with RoPE)
    2. cross_K, cross_V from encoder hidden states (no RoPE, only k_norm)
    3. full_K = cat([self_K, cross_K]), full_V = cat([self_V, cross_V])
    4. Single attention over full_K/full_V with causal mask on self part

    Uses F.scaled_dot_product_attention (which dispatches to flash_attn
    when available) instead of vLLM's PagedAttention.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = config.query_pre_attn_scalar ** -0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = Chroma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Chroma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

        # Cross-attention KV cache (computed once, reused across all decode steps)
        self._cross_k: Optional[torch.Tensor] = None  # [num_kv_heads, enc_seq, head_dim]
        self._cross_v: Optional[torch.Tensor] = None

    def set_cross_attention_cache(
        self,
        encoder_hidden_states: torch.Tensor,
    ):
        """Compute and cache cross-attention KV from encoder output.

        Called once during prefill. The cross KV does not change during
        decode steps.

        Args:
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
        """
        batch_size = encoder_hidden_states.shape[0]
        cross_k = self.k_proj(encoder_hidden_states)
        cross_v = self.v_proj(encoder_hidden_states)

        # Reshape to [batch, num_kv_heads, enc_seq_len, head_dim]
        cross_k = cross_k.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cross_v = cross_v.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply k_norm (no RoPE for cross-attention keys)
        cross_k = self.k_norm(cross_k)

        self._cross_k = cross_k
        self._cross_v = cross_v

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        self_kv_cache: Optional[SelfKVCache] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        merged_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass replicating HF Chroma2MergedAttention.

        Args:
            hidden_states: [batch, seq_len, hidden_size] - decoder input
            position_embeddings: (cos, sin) for RoPE
            self_kv_cache: Managed KV cache for self-attention
            encoder_hidden_states: [batch, enc_seq, hidden_size] - only at prefill
            merged_attention_mask: Optional mask for the merged attention

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # --- Q, K, V projections from decoder hidden states ---
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: [batch, seq_len, num_heads/kv_heads, head_dim] -> [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # --- Norms ---
        q = self.q_norm(q)
        k = self.k_norm(k)

        # --- RoPE on self-attention Q and K ---
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # --- Self-attention KV cache update ---
        if self_kv_cache is not None:
            # Remove batch dim for cache (cache is per-request)
            # TODO: handle batch > 1 properly in Phase 2
            k_for_cache = k.squeeze(0)  # [num_kv_heads, seq_len, head_dim]
            v_for_cache = v.squeeze(0)
            cached_k, cached_v = self_kv_cache.update(self.layer_idx, k_for_cache, v_for_cache)
            # Restore batch dim
            k = cached_k.unsqueeze(0)  # [1, num_kv_heads, total_seq, head_dim]
            v = cached_v.unsqueeze(0)

        # --- Cross-attention KV (computed once, then cached) ---
        if encoder_hidden_states is not None:
            self.set_cross_attention_cache(encoder_hidden_states)

        assert self._cross_k is not None, (
            "Cross-attention cache not initialized. "
            "Pass encoder_hidden_states on the first forward call."
        )

        # --- Merge self + cross KV ---
        # self_K: [batch, num_kv_heads, self_seq, head_dim]
        # cross_K: [batch, num_kv_heads, enc_seq, head_dim]
        full_k = torch.cat([k, self._cross_k], dim=2)
        full_v = torch.cat([v, self._cross_v], dim=2)

        # --- GQA: expand KV heads to match Q heads ---
        if self.num_kv_groups > 1:
            full_k = full_k.repeat_interleave(self.num_kv_groups, dim=1)
            full_v = full_v.repeat_interleave(self.num_kv_groups, dim=1)

        # --- Attention ---
        # Q: [batch, num_heads, q_len, head_dim]
        # full_K: [batch, num_heads, self_seq + enc_seq, head_dim]
        attn_output = F.scaled_dot_product_attention(
            q, full_k, full_v,
            attn_mask=merged_attention_mask,
            scale=self.scaling,
            dropout_p=0.0,  # no dropout at inference
        )
        # attn_output: [batch, num_heads, q_len, head_dim]

        # --- Output projection ---
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def clear_cache(self):
        """Clear cross-attention cache (call between requests)."""
        self._cross_k = None
        self._cross_v = None
