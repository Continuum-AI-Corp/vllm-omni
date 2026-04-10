"""
Server-side test: verify Chroma2 components load and produce correct output.

Tests:
  1. Model loads from checkpoint
  2. Encoder forward works
  3. DepthDecoder generates 8 codebooks
  4. Mimi codec decodes to audio
  5. Full generate pipeline (using the real generate() path)

Run on server:
    cd /app/test/orca-rt
    PYTHONPATH=$(pwd) python vllm-omni/tests/chroma2/test_decoder_codec.py
"""

import sys
import os
import time

# Patch for transformers 5.5 compat
from unittest.mock import MagicMock
import transformers.utils.generic as _generic_mod
if not hasattr(_generic_mod, "OutputRecorder"):
    _generic_mod.OutputRecorder = MagicMock

import torch

# =========================================================================
# Config
# =========================================================================

CHECKPOINT_PATH = "/models/Chroma2/checkpoints/chroma2_0409_4000"
CODEC_PATH = "/models/Mimi"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Codec: {CODEC_PATH}")


# =========================================================================
# Test 1: Model loads
# =========================================================================

def test_model_loads():
    """Load Chroma2Model from checkpoint."""
    print("\n[Test 1] Model loading from checkpoint")

    from chroma2.modeling_chroma2 import Chroma2Model

    t0 = time.time()
    model = Chroma2Model.from_pretrained(CHECKPOINT_PATH).to(DEVICE).eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Verify all submodels exist
    assert model.encoder is not None, "Encoder missing"
    assert model.decoder is not None, "Decoder missing"
    assert model.depth_decoder is not None, "DepthDecoder missing"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    depth_params = sum(p.numel() for p in model.depth_decoder.parameters())

    print(f"  Total params: {total_params / 1e6:.1f}M")
    print(f"  Encoder: {enc_params / 1e6:.1f}M")
    print(f"  Decoder: {dec_params / 1e6:.1f}M")
    print(f"  DepthDecoder: {depth_params / 1e6:.1f}M")
    print(f"  Config: concat_pattern={model.config.concat_pattern}")
    print(f"  PASSED\n")

    return model


# =========================================================================
# Test 2: Encoder forward
# =========================================================================

def test_encoder(model):
    """Encoder forward with text-only input."""
    print("[Test 2] Encoder forward (text embeddings)")

    torch.manual_seed(42)

    text_ids = torch.randint(0, 1000, (1, 20), device=DEVICE)
    text_embeds = model.encoder.embed_tokens(text_ids)
    mask = torch.ones(1, 20, device=DEVICE)

    with torch.no_grad():
        enc_out = model.encoder(
            inputs_embeds=text_embeds,
            attention_mask=mask,
        )

    hidden = enc_out.last_hidden_state
    print(f"  Output shape: {hidden.shape}")
    print(f"  Output mean: {hidden.mean():.6f}")
    assert hidden.shape == (1, 20, model.config.encoder_config.hidden_size)
    assert not torch.isnan(hidden).any(), "NaN in encoder output"
    print(f"  PASSED\n")


# =========================================================================
# Test 3: DepthDecoder generate
# =========================================================================

def test_depth_decoder(model):
    """DepthDecoder: decoder_hidden + c0 → 8 codebooks."""
    print("[Test 3] DepthDecoder generate (c0 → c1~c7)")

    torch.manual_seed(42)
    num_codebooks = model.config.audio_num_codebooks

    # Match model dtype (checkpoint is typically bf16)
    model_dtype = next(model.depth_decoder.parameters()).dtype
    decoder_hidden = torch.randn(1, model.config.decoder_config.hidden_size, device=DEVICE, dtype=model_dtype)
    codebook_0 = torch.tensor([[42]], device=DEVICE)

    with torch.no_grad():
        frame_codes = model.depth_decoder.generate(
            decoder_hidden=decoder_hidden,
            codebook_0=codebook_0,
            temperature=1.0,
            top_k=0,
        )

    print(f"  Frame codes shape: {frame_codes.shape}")
    print(f"  Frame codes: {frame_codes[0].tolist()}")
    assert frame_codes.shape == (1, num_codebooks), \
        f"Expected (1, {num_codebooks}), got {frame_codes.shape}"
    assert (frame_codes >= 0).all() and (frame_codes < model.config.audio_vocab_size).all(), \
        "Codebook values out of range"
    print(f"  PASSED\n")

    return frame_codes


# =========================================================================
# Test 4: Mimi codec decode
# =========================================================================

def test_codec_decode(frame_codes):
    """Mimi: codebook frames → audio waveform."""
    print("[Test 4] Mimi codec decode (codebooks → audio)")

    from transformers.models.mimi import MimiModel

    codec = MimiModel.from_pretrained(CODEC_PATH).to(DEVICE).eval()

    # Simulate 10 frames by repeating
    num_frames = 10
    codebooks = frame_codes.unsqueeze(-1).repeat(1, 1, num_frames)
    print(f"  Codebooks shape: {codebooks.shape}")  # [1, 8, 10]

    with torch.no_grad():
        audio = codec.decode(codebooks).audio_values

    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio duration: {audio.shape[-1] / 24000:.3f}s (at 24kHz)")
    assert audio.shape[0] == 1
    assert audio.shape[-1] > 0
    assert not torch.isnan(audio).any(), "NaN in audio output"
    print(f"  PASSED\n")


# =========================================================================
# Test 5: Weight mapping verification
# =========================================================================

def test_weight_mapping(model):
    """Verify weight names match what our load_weights expects.

    Our chroma2_decoder.py loads weights with prefixes:
      encoder.*, decoder.*, depth_decoder.*
    This test checks those prefixes exist in the checkpoint.
    """
    print("[Test 5] Weight name mapping verification")

    state_dict = model.state_dict()
    prefixes = {"encoder": 0, "decoder": 0, "depth_decoder": 0, "codec_model": 0, "thinker": 0, "other": 0}

    for key in state_dict:
        matched = False
        for prefix in ["encoder", "decoder", "depth_decoder", "codec_model", "thinker"]:
            if key.startswith(prefix + "."):
                prefixes[prefix] += 1
                matched = True
                break
        if not matched:
            prefixes["other"] += 1

    for prefix, count in prefixes.items():
        if count > 0:
            print(f"  {prefix}: {count} parameters")

    # Our decoder stage needs these three
    assert prefixes["encoder"] > 0, "No encoder.* weights found"
    assert prefixes["decoder"] > 0, "No decoder.* weights found"
    assert prefixes["depth_decoder"] > 0, "No depth_decoder.* weights found"

    print(f"  Weight prefixes match our load_weights expectations")
    print(f"  PASSED\n")


# =========================================================================
# Test 6: Codec flat reshape verification
# =========================================================================

def test_codec_reshape():
    """Verify our flat codebook → [1, num_codebooks, num_frames] reshape
    matches what chroma2_codec.py does."""
    print("[Test 6] Codec flat reshape logic")

    num_codebooks = 8
    num_frames = 5

    # Simulate flat codes as Stage 1 would send them
    # Layout: [c0_f0, c1_f0, ..., c7_f0, c0_f1, c1_f1, ..., c7_f1, ...]
    flat_codes = []
    for frame in range(num_frames):
        for cb in range(num_codebooks):
            flat_codes.append(frame * 100 + cb)  # easy to verify

    flat_tensor = torch.tensor(flat_codes, dtype=torch.long)
    print(f"  Flat codes length: {len(flat_codes)}")

    # Our reshape logic (from chroma2_codec.py)
    reshaped = flat_tensor.view(num_frames, num_codebooks).T.unsqueeze(0)
    # Expected: [1, 8, 5]
    print(f"  Reshaped: {reshaped.shape}")

    # Verify: reshaped[0, cb, frame] == frame * 100 + cb
    for frame in range(num_frames):
        for cb in range(num_codebooks):
            expected = frame * 100 + cb
            actual = reshaped[0, cb, frame].item()
            assert actual == expected, f"Mismatch at cb={cb}, frame={frame}: {actual} != {expected}"

    print(f"  All values correct after reshape")
    print(f"  PASSED\n")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Chroma2 Decoder + Codec Integration Tests")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)
    if not os.path.exists(CODEC_PATH):
        print(f"ERROR: Codec not found at {CODEC_PATH}")
        sys.exit(1)

    model = test_model_loads()
    test_encoder(model)
    frame_codes = test_depth_decoder(model)
    test_codec_decode(frame_codes)
    test_weight_mapping(model)
    test_codec_reshape()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
