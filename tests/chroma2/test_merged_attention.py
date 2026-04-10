"""
Test that Chroma2MergedAttentionVLLM produces identical output
to the original HF Chroma2MergedAttention logic.

Self-contained: does NOT import from chroma2.modeling_chroma2 to avoid
heavy transformers dependencies. Instead, reimplements the HF attention
logic inline using only PyTorch.

Run:
    cd e:/Projects/chroma-backend
    python vllm-omni/tests/chroma2/test_merged_attention.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Direct import of our module without triggering vllm_omni.__init__
# which pulls in heavy dependencies (aenum, vllm, etc.)
import importlib.util

_ATTENTION_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "vllm_omni", "model_executor", "models", "chroma2", "chroma2_attention.py",
)

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Inline reimplementation of HF Chroma2MergedAttention (reference)
# Only depends on PyTorch, no transformers needed.
# =========================================================================


class RefRMSNorm(nn.Module):
    """Matches chroma2 Chroma2RMSNorm: (1+weight) scaling."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def ref_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def ref_apply_rotary(q, k, cos, sin):
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (ref_rotate_half(q) * sin)
    k_embed = (k * cos) + (ref_rotate_half(k) * sin)
    return q_embed, k_embed


class RefMergedAttention(nn.Module):
    """Reference HF Chroma2MergedAttention, pure PyTorch."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, scaling, eps):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RefRMSNorm(head_dim, eps)
        self.k_norm = RefRMSNorm(head_dim, eps)

    def forward(self, hidden_states, encoder_hidden_states, cos, sin,
                self_k_cache=None, self_v_cache=None):
        """
        Returns: (output, updated_self_k_cache, updated_self_v_cache)
        """
        B, S, _ = hidden_states.shape
        E = encoder_hidden_states.shape[1]

        # Self-attention projections
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = ref_apply_rotary(q, k, cos, sin)

        # KV cache
        if self_k_cache is not None:
            k = torch.cat([self_k_cache, k], dim=2)
            v = torch.cat([self_v_cache, v], dim=2)

        # Cross-attention projections (same k_proj/v_proj, no RoPE)
        cross_k = self.k_proj(encoder_hidden_states).view(B, E, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cross_v = self.v_proj(encoder_hidden_states).view(B, E, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cross_k = self.k_norm(cross_k)

        # Merge
        full_k = torch.cat([k, cross_k], dim=2)
        full_v = torch.cat([v, cross_v], dim=2)

        # GQA expand KV to match Q heads
        num_groups = self.num_heads // self.num_kv_heads
        if num_groups > 1:
            full_k = full_k.repeat_interleave(num_groups, dim=1)
            full_v = full_v.repeat_interleave(num_groups, dim=1)

        # Attention (q already has num_heads from q_proj)
        out = F.scaled_dot_product_attention(q, full_k, full_v, scale=self.scaling)
        out = out.transpose(1, 2).contiguous().reshape(B, S, self.num_heads * self.head_dim)
        out = self.o_proj(out)

        return out, k, v  # return self-attention KV for caching


# =========================================================================
# Import vLLM attention (direct file import to avoid vllm_omni.__init__)
# =========================================================================

_spec = importlib.util.spec_from_file_location("chroma2_attention", _ATTENTION_MODULE_PATH)
_attn_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_attn_mod)

Chroma2MergedAttentionVLLM = _attn_mod.Chroma2MergedAttentionVLLM
SelfKVCache = _attn_mod.SelfKVCache


# =========================================================================
# Test config
# =========================================================================

HIDDEN_SIZE = 2048
NUM_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 64
SCALING = 256 ** -0.5
EPS = 1e-5


class MockConfig:
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_HEADS
    num_key_value_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM
    attention_bias = False
    rms_norm_eps = EPS
    query_pre_attn_scalar = 256
    attention_dropout = 0.0
    is_causal = True


def copy_weights(ref_attn, vllm_attn):
    vllm_attn.q_proj.load_state_dict(ref_attn.q_proj.state_dict())
    vllm_attn.k_proj.load_state_dict(ref_attn.k_proj.state_dict())
    vllm_attn.v_proj.load_state_dict(ref_attn.v_proj.state_dict())
    vllm_attn.o_proj.load_state_dict(ref_attn.o_proj.state_dict())
    vllm_attn.q_norm.load_state_dict(ref_attn.q_norm.state_dict())
    vllm_attn.k_norm.load_state_dict(ref_attn.k_norm.state_dict())


def make_cos_sin(seq_len, head_dim, device="cpu"):
    cos = torch.randn(1, 1, seq_len, head_dim, device=device)
    sin = torch.randn(1, 1, seq_len, head_dim, device=device)
    return cos, sin


# =========================================================================
# Tests
# =========================================================================


def test_prefill_equivalence():
    """Prefill: 10 decoder tokens + 50 encoder tokens, no cache."""
    print("[Test 1] Prefill equivalence (no cache)")

    torch.manual_seed(42)

    ref = RefMergedAttention(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, SCALING, EPS).eval()
    vllm = Chroma2MergedAttentionVLLM(MockConfig(), layer_idx=0).eval()
    copy_weights(ref, vllm)

    B, dec_len, enc_len = 1, 10, 50
    dec_hidden = torch.randn(B, dec_len, HIDDEN_SIZE)
    enc_hidden = torch.randn(B, enc_len, HIDDEN_SIZE)
    cos, sin = make_cos_sin(dec_len, HEAD_DIM)

    with torch.no_grad():
        ref_out, _, _ = ref(dec_hidden, enc_hidden, cos, sin)
        vllm_out = vllm(
            hidden_states=dec_hidden,
            position_embeddings=(cos, sin),
            self_kv_cache=None,
            encoder_hidden_states=enc_hidden,
        )

    max_diff = (ref_out - vllm_out).abs().max().item()
    mean_diff = (ref_out - vllm_out).abs().mean().item()
    print(f"  max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    assert max_diff < 1e-4, f"FAILED: max_diff={max_diff}"
    print("  PASSED\n")


def test_decode_step_equivalence():
    """Prefill 10 tokens, then 5 decode steps, compare each step."""
    print("[Test 2] Decode step equivalence (with KV cache)")

    torch.manual_seed(42)

    ref = RefMergedAttention(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, SCALING, EPS).eval()
    vllm = Chroma2MergedAttentionVLLM(MockConfig(), layer_idx=0).eval()
    copy_weights(ref, vllm)

    B, prefill_len, enc_len = 1, 10, 50
    decode_steps = 5

    enc_hidden = torch.randn(B, enc_len, HIDDEN_SIZE)
    prefill_hidden = torch.randn(B, prefill_len, HIDDEN_SIZE)
    cos_pre, sin_pre = make_cos_sin(prefill_len, HEAD_DIM)

    # --- Prefill ---
    vllm_cache = SelfKVCache(
        num_layers=1,
        max_seq_len=prefill_len + decode_steps + 10,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with torch.no_grad():
        ref_out, ref_k, ref_v = ref(prefill_hidden, enc_hidden, cos_pre, sin_pre)
        vllm_out = vllm(
            hidden_states=prefill_hidden,
            position_embeddings=(cos_pre, sin_pre),
            self_kv_cache=vllm_cache,
            encoder_hidden_states=enc_hidden,
        )
    vllm_cache.advance_seq_len(prefill_len)

    diff = (ref_out - vllm_out).abs().max().item()
    print(f"  Prefill: max_diff={diff:.2e}")
    assert diff < 1e-4, f"Prefill FAILED: {diff}"

    # --- Decode steps ---
    for step in range(decode_steps):
        step_hidden = torch.randn(B, 1, HIDDEN_SIZE)
        cos_step, sin_step = make_cos_sin(1, HEAD_DIM)

        with torch.no_grad():
            # Ref: pass full encoder each time (it recomputes cross KV each call)
            ref_out, ref_k, ref_v = ref(
                step_hidden, enc_hidden, cos_step, sin_step,
                self_k_cache=ref_k, self_v_cache=ref_v,
            )

            # vLLM: cross KV already cached, only pass new decoder hidden
            vllm_out = vllm(
                hidden_states=step_hidden,
                position_embeddings=(cos_step, sin_step),
                self_kv_cache=vllm_cache,
                encoder_hidden_states=None,  # cached
            )
        vllm_cache.advance_seq_len(1)

        diff = (ref_out - vllm_out).abs().max().item()
        print(f"  Step {step}: max_diff={diff:.2e}")
        assert diff < 1e-4, f"Step {step} FAILED: {diff}"

    print("  PASSED\n")


def test_batch_size_2():
    """Verify with batch_size=2 (different requests)."""
    print("[Test 3] Batch size 2")

    torch.manual_seed(123)

    ref = RefMergedAttention(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, SCALING, EPS).eval()
    vllm = Chroma2MergedAttentionVLLM(MockConfig(), layer_idx=0).eval()
    copy_weights(ref, vllm)

    B, dec_len, enc_len = 2, 8, 30
    dec_hidden = torch.randn(B, dec_len, HIDDEN_SIZE)
    enc_hidden = torch.randn(B, enc_len, HIDDEN_SIZE)
    cos, sin = make_cos_sin(dec_len, HEAD_DIM)

    with torch.no_grad():
        ref_out, _, _ = ref(dec_hidden, enc_hidden, cos, sin)
        vllm_out = vllm(
            hidden_states=dec_hidden,
            position_embeddings=(cos, sin),
            self_kv_cache=None,
            encoder_hidden_states=enc_hidden,
        )

    max_diff = (ref_out - vllm_out).abs().max().item()
    print(f"  max_diff={max_diff:.2e}")
    assert max_diff < 1e-4, f"FAILED: max_diff={max_diff}"
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Chroma2 MergedAttention: Reference vs vLLM")
    print("=" * 60)
    print()

    test_prefill_equivalence()
    test_decode_step_equivalence()
    test_batch_size_2()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
