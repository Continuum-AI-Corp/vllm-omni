# SPDX-License-Identifier: Apache-2.0
"""
Chroma2 Codec Stage (Stage 2) for vllm-omni.

Receives codebook frames from Stage 1 (Decoder) and decodes them
to audio waveform using the Mimi codec model.

This is a non-autoregressive stage: it receives all codebook frames
at once (or in chunks via async_chunk) and produces audio output
in a single forward pass.
"""

from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class Chroma2CodecForGeneration(nn.Module):
    """Stage 2: Mimi codec decode (codebook frames → audio waveform).

    Uses the HF MimiModel directly for decoding.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "codec_model.": "codec_model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        # Import HF Mimi codec
        from transformers.models.mimi import MimiModel

        self.codec_model = MimiModel._from_config(config.codec_config)
        self.num_codebooks = config.audio_num_codebooks
        self.audio_frame_freq = getattr(config, "audio_frame_freq", 1920)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Decode codebook frames to audio waveform.

        Args:
            input_ids: Flat codebook codes from Stage 1.
                Shape: [num_frames * num_codebooks] (1D, flattened)
                Layout: [c0_f0, c1_f0, ..., c7_f0, c0_f1, c1_f1, ..., c7_f1, ...]

        Returns:
            audio: [1, audio_length] waveform tensor
        """
        device = next(self.codec_model.parameters()).device

        additional_info = kwargs.get("additional_information", {})

        # Determine codebook data source
        if additional_info and "codebook_codes" in additional_info:
            flat_codes = additional_info["codebook_codes"]
            if isinstance(flat_codes, list):
                flat_codes = torch.tensor(flat_codes, dtype=torch.long, device=device)
            else:
                flat_codes = flat_codes.to(device=device, dtype=torch.long)
        else:
            flat_codes = input_ids.to(device=device, dtype=torch.long)

        if flat_codes.numel() == 0:
            return torch.zeros(1, 0, device=device)

        # Reshape: [flat] → [1, num_codebooks, num_frames]
        num_codebooks = self.num_codebooks
        num_frames = flat_codes.shape[0] // num_codebooks

        if flat_codes.shape[0] % num_codebooks != 0:
            logger.warning(
                f"Codebook length {flat_codes.shape[0]} not divisible by "
                f"num_codebooks {num_codebooks}, truncating."
            )
            flat_codes = flat_codes[: num_frames * num_codebooks]

        # Reshape to [1, num_codebooks, num_frames] for Mimi decoder
        codebooks = flat_codes.view(num_frames, num_codebooks).T.unsqueeze(0)
        # codebooks shape: [1, num_codebooks, num_frames]

        # Mimi decode
        with torch.no_grad():
            audio_values = self.codec_model.decode(codebooks).audio_values
        # audio_values shape: [1, 1, audio_length]

        return audio_values.squeeze(1)  # [1, audio_length]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Codec stage doesn't produce logits (non-autoregressive)."""
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Mimi codec weights only."""
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "encoder.", "decoder.", "depth_decoder."],
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        total_bytes = 0
        for name, param in self.named_parameters():
            if param is not None:
                total_bytes += param.data.numel() * param.data.element_size()
        logger.info(
            f"[Chroma2Codec] Loaded {len(loaded)} params, "
            f"total size: {total_bytes / (1024**2):.1f} MB"
        )
        return loaded
