# SPDX-License-Identifier: Apache-2.0
"""
Chroma2 Decoder Stage (Stage 1) for vllm-omni.

This module contains Encoder + Decoder + DepthDecoder, all running
in the same stage process. The flow for each request:

  Prefill:
    1. Encoder processes reference audio + text → encoder_hidden_state
    2. Decoder processes initial tokens with cross-attention to encoder output

  Decode (per frame):
    1. Consume thinker token(s) from Stage 0 per concat_pattern
    2. Decoder forward → codebook_0 logits
    3. Sample codebook_0
    4. DepthDecoder generates codebook_1~7 (inline)
    5. Complete frame [c0~c7] feeds back as next decoder input

Since we use enforce_eager=true, we directly reuse the HF model classes
(Chroma2Encoder, Chroma2Decoder, Chroma2DepthDecoder) from the chroma2
package. The vLLM-specific adaptation is limited to:
  - forward/compute_logits/sample/load_weights interface
  - KV cache management for the decoder (via chroma2_attention.py)
  - decoder_preprocess for consuming thinker tokens

NOTE: This is the initial implementation that reuses HF classes directly.
A future optimization can replace the HF attention layers with our
Chroma2MergedAttentionVLLM for better batching performance.
"""

from collections.abc import Iterable
from functools import cached_property
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class Chroma2DecoderForGeneration(nn.Module):
    """Stage 1: Encoder + Decoder + DepthDecoder.

    Uses HF model classes directly (enforce_eager mode).
    """

    # Weight mapping from HF checkpoint to our module structure.
    # HF checkpoint has: encoder.*, decoder.*, depth_decoder.*
    # We load them under the same names, so no prefix remapping needed.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "encoder.": "encoder.",
            "decoder.": "decoder.",
            "depth_decoder.": "depth_decoder.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.prefix = prefix

        # Import HF classes (available because chroma2/ is in PYTHONPATH)
        from chroma2.modeling_chroma2 import (
            Chroma2Encoder,
            Chroma2Decoder,
            Chroma2DepthDecoder,
        )

        # Initialize submodels from HF classes
        self.encoder = Chroma2Encoder(config.encoder_config)
        self.decoder = Chroma2Decoder(config.decoder_config)
        self.depth_decoder = Chroma2DepthDecoder(config.depth_decoder_config)

        # Concat pattern for thinker token injection
        self.concat_pattern = config.concat_pattern  # e.g., ("T", "T", "T", "A", "A", "A", "A", "A", "A")

        # Special token IDs
        self.audio_token_id = config.audio_token_id
        self.codebook_eos_token_id = config.codebook_eos_token_id
        self.codebook_pad_token_id = config.codebook_pad_token_id

        # Cached encoder output (computed once per request at prefill)
        self._encoder_hidden_state: Optional[torch.Tensor] = None
        self._encoder_attention_mask: Optional[torch.Tensor] = None

    @cached_property
    def sampler(self):
        return Sampler()

    # ------------------------------------------------------------------
    # Forward: dispatches between prefill and decode
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass for the decoder stage.

        During prefill:
          - Runs encoder on reference audio/text (from additional_information)
          - Runs decoder on initial sequence

        During decode:
          - Decoder forward one step (thinker tokens already injected via preprocess)
          - Returns hidden states for logits computation

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Check if encoder output needs to be computed (prefill)
        additional_info = kwargs.get("additional_information", {})
        if additional_info and self._encoder_hidden_state is None:
            self._run_encoder(additional_info)

        # Decoder forward
        # For now, use HF decoder directly with its own cache management
        # TODO: Replace with MergedAttentionVLLM for batched inference
        decoder_output = self.decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            encoder_last_hidden_state=self._encoder_hidden_state,
            encoder_attention_mask=self._encoder_attention_mask,
            use_cache=True,
            past_key_values=kwargs.get("past_key_values"),
            cache_position=kwargs.get("cache_position"),
            attention_mask=kwargs.get("attention_mask"),
            input_features=kwargs.get("input_features"),
            feature_attention_mask=kwargs.get("feature_attention_mask"),
        )

        return decoder_output.last_hidden_state

    def _run_encoder(self, additional_info: dict):
        """Run encoder once and cache the output.

        Called during prefill when additional_information contains
        encoder inputs from Stage 0.
        """
        encoder_input_ids = additional_info.get("encoder_input_ids")
        encoder_input_features = additional_info.get("encoder_input_features")
        encoder_attention_mask = additional_info.get("encoder_attention_mask")
        encoder_feature_attention_mask = additional_info.get("encoder_feature_attention_mask")

        if encoder_input_ids is None:
            return

        device = next(self.encoder.parameters()).device

        # Move to device
        if isinstance(encoder_input_ids, torch.Tensor):
            encoder_input_ids = encoder_input_ids.to(device)
        if isinstance(encoder_input_features, torch.Tensor):
            encoder_input_features = encoder_input_features.to(device)
        if isinstance(encoder_attention_mask, torch.Tensor):
            encoder_attention_mask = encoder_attention_mask.to(device)
        if isinstance(encoder_feature_attention_mask, torch.Tensor):
            encoder_feature_attention_mask = encoder_feature_attention_mask.to(device)

        with torch.no_grad():
            encoder_output = self.encoder(
                input_ids=encoder_input_ids,
                input_features=encoder_input_features,
                attention_mask=encoder_attention_mask,
                feature_attention_mask=encoder_feature_attention_mask,
            )

        self._encoder_hidden_state = encoder_output.last_hidden_state
        self._encoder_attention_mask = encoder_output.attention_mask

        logger.info(
            f"Encoder output cached: shape={self._encoder_hidden_state.shape}"
        )

    # ------------------------------------------------------------------
    # DepthDecoder: generate codebook_1~7 from decoder hidden + c0
    # ------------------------------------------------------------------

    def generate_depth(
        self,
        decoder_hidden: torch.Tensor,
        codebook_0: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Run depth decoder to generate remaining codebooks for one frame.

        This is called inline after each decoder step that produces c0.
        Reuses the HF DepthDecoder.generate() directly.

        Args:
            decoder_hidden: [batch, hidden_size] - decoder last hidden state
            codebook_0: [batch, 1] - sampled first codebook token
            temperature: sampling temperature
            top_k: top-k filtering (0 = greedy)

        Returns:
            frame_codes: [batch, num_codebooks] - all 8 codebooks for this frame
        """
        return self.depth_decoder.generate(
            decoder_hidden=decoder_hidden,
            codebook_0=codebook_0,
            temperature=temperature,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Logits / sampling
    # ------------------------------------------------------------------

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Compute codebook_0 logits from decoder hidden states.

        Uses the decoder's lm_head (Chroma2LMHead).
        """
        if hidden_states is None:
            return None
        # Use decoder's lm_head for codebook_0 prediction
        logits = self.decoder.lm_head(hidden_states[:, -1:, :])
        return logits.squeeze(1)  # [batch, audio_vocab_size]

    def sample(self, logits, sampling_metadata):
        return self.sampler(logits, sampling_metadata)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load encoder + decoder + depth_decoder weights from HF checkpoint.

        Skips thinker and codec_model weights.
        """
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "codec_model."],
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Log summary
        total_bytes = 0
        for name, param in self.named_parameters():
            if param is not None:
                total_bytes += param.data.numel() * param.data.element_size()
        logger.info(
            f"[Chroma2Decoder] Loaded {len(loaded)} params, "
            f"total size: {total_bytes / (1024**2):.1f} MB"
        )
        return loaded

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_request_state(self):
        """Clear per-request cached state. Call between requests."""
        self._encoder_hidden_state = None
        self._encoder_attention_mask = None
