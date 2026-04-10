# SPDX-License-Identifier: Apache-2.0
"""
Chroma2 vllm-omni integration: top-level dispatcher.

3-stage pipeline:
  Stage 0 (thinker):  Qwen2.5-Omni text generation → stream token IDs
  Stage 1 (decoder):  Encoder + Decoder + DepthDecoder → stream codebook frames
  Stage 2 (codec):    Mimi codec decode → audio waveform

Usage:
  Set `model_stage` in vllm_config to one of: "thinker", "decoder", "codec"
"""

from collections.abc import Iterable
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class Chroma2ForConditionalGeneration(nn.Module, CustomProcessMixin):
    """Top-level Chroma2 model that dispatches to the active stage.

    Each stage runs as an independent vLLM engine process. This class
    is instantiated once per stage; only the submodel for that stage
    is loaded into memory.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.has_preprocess = False
        self.has_postprocess = False
        self.have_multimodal_outputs = False
        self.vllm_config = vllm_config

        config = vllm_config.model_config.hf_config
        self.config = config
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            self._init_thinker(vllm_config, config, prefix)
        elif self.model_stage == "decoder":
            self._init_decoder(vllm_config, config, prefix)
        elif self.model_stage == "codec":
            self._init_codec(vllm_config, config, prefix)
        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. "
                "Must be one of: 'thinker', 'decoder', 'codec'"
            )

    # ------------------------------------------------------------------
    # Stage initialization
    # ------------------------------------------------------------------

    def _init_thinker(self, vllm_config: VllmConfig, config, prefix: str):
        """Stage 0: Thinker (Qwen2.5-Omni text generation).

        Reuses vLLM's built-in Qwen2 support. The thinker is a standard
        autoregressive LM that generates text token IDs streamed to
        Stage 1 via async_chunk.
        """
        thinker_config = config.thinker_config
        self.thinker_config = thinker_config
        self.thinker = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "thinker"),
            hf_config=thinker_config,
            architectures=["Qwen2_5OmniThinkerModel"],
        )
        self.model = self.thinker
        self.decoder = None
        self.codec = None
        self.have_multimodal_outputs = True

    def _init_decoder(self, vllm_config: VllmConfig, config, prefix: str):
        """Stage 1: Encoder + Decoder + DepthDecoder.

        This is the core generation stage:
        - Encoder runs once during prefill to produce encoder_hidden_state
        - Decoder autoregressively generates codebook_0 tokens with
          cross-attention to encoder output
        - DepthDecoder generates codebook_1~7 per frame (inline)
        - Thinker token IDs consumed from Stage 0 via async_chunk
        """
        self.thinker = None
        self.codec = None
        self.has_preprocess = True
        self.set_custom_preprocess(self.decoder_preprocess)

        self.decoder = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "decoder"),
            hf_config=config,
            architectures=["Chroma2DecoderModel"],
        )
        self.model = self.decoder
        self.have_multimodal_outputs = True
        self.requires_raw_input_tokens = True

    def _init_codec(self, vllm_config: VllmConfig, config, prefix: str):
        """Stage 2: Mimi codec decode (codebook frames -> audio waveform)."""
        self.thinker = None
        self.decoder = None
        self.codec = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "codec"),
            hf_config=config,
            architectures=["Chroma2CodecModel"],
        )
        self.model = self.codec
        self.requires_raw_input_tokens = True
        self.have_multimodal_outputs = True

    # ------------------------------------------------------------------
    # Device utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        if self.model_stage == "thinker":
            return self._forward_thinker(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        elif self.model_stage == "decoder":
            return self._forward_decoder(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        elif self.model_stage == "codec":
            return self._forward_codec(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        raise RuntimeError(f"Unexpected model_stage: {self.model_stage}")

    def _forward_thinker(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
        """Thinker: standard AR text generation."""
        thinker_output = self.thinker(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return OmniOutput(
            text_hidden_states=thinker_output,
            multimodal_outputs=None,
        )

    def _forward_decoder(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
        """Decoder: encoder + decoder + depth_decoder generation."""
        hidden_states = self.decoder(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=None,
        )

    def _forward_codec(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
        """Codec: Mimi decode codebooks → audio waveform."""
        audio_output = self.codec(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audio_output},
        )

    # ------------------------------------------------------------------
    # Logits / sampling
    # ------------------------------------------------------------------

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        return self.model.compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        if hasattr(self.model, "sample"):
            return self.model.sample(logits, sampling_metadata)
        return self.sampler(logits, sampling_metadata)

    # ------------------------------------------------------------------
    # Decoder preprocess: consume thinker tokens from async_chunk
    # ------------------------------------------------------------------

    def decoder_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Consume streamed thinker token IDs and inject into decoder input.

        Called by vllm-omni's CustomProcessMixin before each decoder step.
        Retrieves thinker tokens from info_dict (populated by
        stage_input_processor) and embeds them according to concat_pattern.

        This is a skeleton — full implementation in Phase 1 Day 5-7.
        """
        update_dict: dict[str, Any] = {}
        # TODO: implement concat_pattern token injection
        # 1. Read thinker_token_ids from info_dict
        # 2. Based on current position in concat_pattern, decide if this step is "T" or "A"
        # 3. If "T": embed thinker token and return as input_embeds
        # 4. If "A": let normal audio token flow through
        return input_ids, input_embeds, update_dict

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.model_stage == "thinker":
            return self._load_thinker_weights(weights)
        elif self.model_stage == "decoder":
            return self._load_decoder_weights(weights)
        elif self.model_stage == "codec":
            return self._load_codec_weights(weights)
        raise RuntimeError(f"Unexpected model_stage: {self.model_stage}")

    def _load_thinker_weights(self, weights):
        """Load thinker weights, skipping encoder/decoder/depth_decoder/codec."""
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["encoder.", "decoder.", "depth_decoder.", "codec_model."],
        )
        return loader.load_weights(weights)

    def _load_decoder_weights(self, weights):
        """Load encoder + decoder + depth_decoder weights."""
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "codec_model."],
        )
        return loader.load_weights(weights)

    def _load_codec_weights(self, weights):
        """Load Mimi codec weights only."""
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "encoder.", "decoder.", "depth_decoder."],
        )
        return loader.load_weights(weights)

    # Pipeline parallel support
    @property
    def make_empty_intermediate_tensors(self):
        if hasattr(self.model, "make_empty_intermediate_tensors"):
            return self.model.make_empty_intermediate_tensors
        return lambda: None

    @make_empty_intermediate_tensors.setter
    def make_empty_intermediate_tensors(self, value):
        pass
