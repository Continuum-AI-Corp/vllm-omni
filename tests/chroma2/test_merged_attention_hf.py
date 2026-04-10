"""
Server-side test: compare vLLM MergedAttention against the REAL HF
Chroma2MergedAttention with EncoderDecoderCache, real RoPE, and
proper merged attention mask.

Requirements:
  - transformers >= 5.0.0rc0
  - chroma2/ accessible in PYTHONPATH

Run on server:
    cd /app  (or wherever chroma-backend is)
    PYTHONPATH=/app python vllm-omni/tests/chroma2/test_merged_attention_hf.py
"""

import sys
import os
import importlib.util
from unittest.mock import MagicMock

# Patch missing symbols in newer transformers before importing chroma2
# chroma2 was written for transformers 5.0.0rc0; some names moved in 5.5.0
import transformers.utils.generic as _generic_mod
if not hasattr(_generic_mod, "OutputRecorder"):
    _generic_mod.OutputRecorder = MagicMock

# Direct import of our vLLM attention module (avoids vllm_omni.__init__ deps)
_ATTN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "vllm_omni", "model_executor", "models", "chroma2", "chroma2_attention.py",
)
_spec = importlib.util.spec_from_file_location("chroma2_attention", _ATTN_PATH)
_attn_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_attn_mod)
Chroma2MergedAttentionVLLM = _attn_mod.Chroma2MergedAttentionVLLM
SelfKVCache = _attn_mod.SelfKVCache

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

# Import the REAL HF Chroma2 classes
from chroma2.modeling_chroma2 import (
    Chroma2MergedAttention,
    Chroma2RotaryEmbedding,
)
from chroma2.configuration_chroma2 import Chroma2DecoderConfig


def copy_weights(hf_attn: Chroma2MergedAttention, vllm_attn: Chroma2MergedAttentionVLLM):
    """Copy all weights from HF attention to vLLM attention."""
    vllm_attn.q_proj.load_state_dict(hf_attn.q_proj.state_dict())
    vllm_attn.k_proj.load_state_dict(hf_attn.k_proj.state_dict())
    vllm_attn.v_proj.load_state_dict(hf_attn.v_proj.state_dict())
    vllm_attn.o_proj.load_state_dict(hf_attn.o_proj.state_dict())
    vllm_attn.q_norm.load_state_dict(hf_attn.q_norm.state_dict())
    vllm_attn.k_norm.load_state_dict(hf_attn.k_norm.state_dict())


class VLLMConfig:
    """Minimal config for vLLM MergedAttention (no transformers dependency)."""
    hidden_size = 2048
    num_attention_heads = 16
    num_key_value_heads = 4
    head_dim = 64
    attention_bias = False
    rms_norm_eps = 1e-5
    query_pre_attn_scalar = 256
    attention_dropout = 0.0
    is_causal = True


def build_merged_attention_mask(decoder_seq_len: int, encoder_seq_len: int, device="cpu"):
    """Build the merged attention mask matching HF Chroma2's behavior.

    The merged mask has shape [1, 1, decoder_seq_len, decoder_seq_len + encoder_seq_len]

    - Self-attention part (left): causal mask (lower triangular)
    - Cross-attention part (right): all visible (no mask)
    """
    total_kv_len = decoder_seq_len + encoder_seq_len

    # Start with all -inf
    mask = torch.full((decoder_seq_len, total_kv_len), float("-inf"), device=device)

    # Self-attention: causal (lower triangular)
    for i in range(decoder_seq_len):
        mask[i, : i + 1] = 0.0

    # Cross-attention: all visible
    mask[:, decoder_seq_len:] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, dec_len, total_kv_len]


# =========================================================================
# Test 1: Prefill with real HF implementation, real RoPE, real mask
# =========================================================================

def test_prefill_with_real_hf():
    """Compare prefill output: real HF vs vLLM, with actual RoPE and mask."""
    print("[Test 1] Prefill with real HF + real RoPE + merged mask")

    torch.manual_seed(42)
    device = "cpu"

    # Create real HF config
    config = Chroma2DecoderConfig(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64,
        attention_bias=False,
        rms_norm_eps=1e-5,
        is_causal=True,
        audio_num_codebooks=8,
        num_hidden_layers=1,
    )
    # Force eager attention (no flash_attn on CPU)
    config._attn_implementation = "eager"

    # Create both models
    hf_attn = Chroma2MergedAttention(config, layer_idx=0).to(device).eval()
    vllm_attn = Chroma2MergedAttentionVLLM(VLLMConfig(), layer_idx=0).to(device).eval()
    copy_weights(hf_attn, vllm_attn)

    # Real RoPE
    rotary_emb = Chroma2RotaryEmbedding(config).to(device)

    B, dec_len, enc_len = 1, 10, 50
    dec_hidden = torch.randn(B, dec_len, config.hidden_size, device=device)
    enc_hidden = torch.randn(B, enc_len, config.hidden_size, device=device)

    # Real position embeddings from Chroma2RotaryEmbedding
    position_ids = torch.arange(dec_len, device=device).unsqueeze(0)
    cos, sin = rotary_emb(dec_hidden, position_ids, layer_type="full_attention")

    # Real merged attention mask
    mask = build_merged_attention_mask(dec_len, enc_len, device)

    # --- HF forward ---
    with torch.no_grad():
        hf_out, hf_self_w, hf_cross_w = hf_attn(
            hidden_states=dec_hidden,
            position_embeddings=(cos, sin),
            merged_attention_mask=mask,
            encoder_hidden_states=enc_hidden,
            past_key_values=None,
            cache_position=None,
        )

    # --- vLLM forward ---
    with torch.no_grad():
        vllm_out = vllm_attn(
            hidden_states=dec_hidden,
            position_embeddings=(cos, sin),
            self_kv_cache=None,
            encoder_hidden_states=enc_hidden,
            merged_attention_mask=mask,
        )

    max_diff = (hf_out - vllm_out).abs().max().item()
    mean_diff = (hf_out - vllm_out).abs().mean().item()
    print(f"  max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    print(f"  HF shape={hf_out.shape}, vLLM shape={vllm_out.shape}")
    assert max_diff < 1e-4, f"FAILED: max_diff={max_diff}"
    print("  PASSED\n")


# =========================================================================
# Test 2: Decode steps with real HF EncoderDecoderCache
# =========================================================================

def test_decode_with_real_hf_cache():
    """Compare decode steps: HF with EncoderDecoderCache vs vLLM with SelfKVCache."""
    print("[Test 2] Decode steps with real HF EncoderDecoderCache + real RoPE")

    torch.manual_seed(42)
    device = "cpu"

    config = Chroma2DecoderConfig(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64,
        attention_bias=False,
        rms_norm_eps=1e-5,
        is_causal=True,
        audio_num_codebooks=8,
    )
    config._attn_implementation = "eager"

    hf_attn = Chroma2MergedAttention(config, layer_idx=0).to(device).eval()
    vllm_attn = Chroma2MergedAttentionVLLM(VLLMConfig(), layer_idx=0).to(device).eval()
    copy_weights(hf_attn, vllm_attn)

    rotary_emb = Chroma2RotaryEmbedding(config).to(device)

    B, prefill_len, enc_len = 1, 10, 50
    decode_steps = 5

    enc_hidden = torch.randn(B, enc_len, config.hidden_size, device=device)
    prefill_hidden = torch.randn(B, prefill_len, config.hidden_size, device=device)

    # --- HF prefill with EncoderDecoderCache ---
    hf_cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
    position_ids = torch.arange(prefill_len, device=device).unsqueeze(0)
    cos, sin = rotary_emb(prefill_hidden, position_ids, layer_type="full_attention")
    cache_position = torch.arange(prefill_len, device=device)
    mask = build_merged_attention_mask(prefill_len, enc_len, device)

    with torch.no_grad():
        hf_out, _, _ = hf_attn(
            hidden_states=prefill_hidden,
            position_embeddings=(cos, sin),
            merged_attention_mask=mask,
            encoder_hidden_states=enc_hidden,
            past_key_values=hf_cache,
            cache_position=cache_position,
        )

    # --- vLLM prefill ---
    vllm_cache = SelfKVCache(
        num_layers=1,
        max_seq_len=prefill_len + decode_steps + 10,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        device=torch.device(device),
        dtype=torch.float32,
    )

    with torch.no_grad():
        vllm_out = vllm_attn(
            hidden_states=prefill_hidden,
            position_embeddings=(cos, sin),
            self_kv_cache=vllm_cache,
            encoder_hidden_states=enc_hidden,
            merged_attention_mask=mask,
        )
    vllm_cache.advance_seq_len(prefill_len)

    prefill_diff = (hf_out - vllm_out).abs().max().item()
    print(f"  Prefill: max_diff={prefill_diff:.2e}")
    assert prefill_diff < 1e-4, f"Prefill FAILED: {prefill_diff}"

    # --- Decode steps ---
    for step in range(decode_steps):
        pos = prefill_len + step
        step_hidden = torch.randn(B, 1, config.hidden_size, device=device)

        # Real RoPE for this position
        step_position_ids = torch.tensor([[pos]], device=device)
        cos_step, sin_step = rotary_emb(step_hidden, step_position_ids, layer_type="full_attention")
        cache_pos = torch.tensor([pos], device=device)

        # HF: decode mask (1 query attending to pos+1 self tokens + enc_len cross tokens)
        hf_mask = build_merged_attention_mask(pos + 1, enc_len, device)
        # Only the last row matters for single-token decode
        hf_mask = hf_mask[:, :, -1:, :]

        # vLLM: same mask shape for the merged attention
        # After cache update, self KV length = pos + 1, so total = pos + 1 + enc_len
        vllm_mask = build_merged_attention_mask(pos + 1, enc_len, device)
        vllm_mask = vllm_mask[:, :, -1:, :]

        with torch.no_grad():
            hf_out, _, _ = hf_attn(
                hidden_states=step_hidden,
                position_embeddings=(cos_step, sin_step),
                merged_attention_mask=hf_mask,
                encoder_hidden_states=enc_hidden,
                past_key_values=hf_cache,
                cache_position=cache_pos,
            )

            vllm_out = vllm_attn(
                hidden_states=step_hidden,
                position_embeddings=(cos_step, sin_step),
                self_kv_cache=vllm_cache,
                encoder_hidden_states=None,  # already cached from prefill
                merged_attention_mask=vllm_mask,
            )
        vllm_cache.advance_seq_len(1)

        diff = (hf_out - vllm_out).abs().max().item()
        print(f"  Step {step} (pos={pos}): max_diff={diff:.2e}")
        assert diff < 1e-4, f"Step {step} FAILED: {diff}"

    print("  PASSED\n")


# =========================================================================
# Test 3: Verify mask correctness (causal self + full cross)
# =========================================================================

def test_mask_structure():
    """Verify the merged attention mask has correct structure."""
    print("[Test 3] Merged attention mask structure")

    dec_len, enc_len = 5, 3
    mask = build_merged_attention_mask(dec_len, enc_len)

    # Expected: [5, 8] with causal on left 5 cols, all-zero on right 3 cols
    mask_2d = mask.squeeze()  # [5, 8]

    # Self-attention part: should be causal (lower triangular)
    self_part = mask_2d[:, :dec_len]
    for i in range(dec_len):
        for j in range(dec_len):
            if j <= i:
                assert self_part[i, j] == 0.0, f"Self mask[{i},{j}] should be 0 (visible)"
            else:
                assert self_part[i, j] == float("-inf"), f"Self mask[{i},{j}] should be -inf (masked)"

    # Cross-attention part: should be all visible
    cross_part = mask_2d[:, dec_len:]
    assert (cross_part == 0.0).all(), "Cross mask should be all 0 (fully visible)"

    print(f"  Mask shape: {mask_2d.shape}")
    print(f"  Self part (causal): correct")
    print(f"  Cross part (full): correct")
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Chroma2 MergedAttention: REAL HF vs vLLM")
    print("  (with EncoderDecoderCache, real RoPE, merged mask)")
    print("=" * 60)
    print()

    test_mask_structure()
    test_prefill_with_real_hf()
    test_decode_with_real_hf_cache()

    print("=" * 60)
    print("ALL TESTS PASSED — vLLM attention is equivalent to HF")
    print("=" * 60)
