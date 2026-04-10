# SPDX-License-Identifier: Apache-2.0
"""
Stage input processors for Chroma2 pipeline.

Handles data conversion between stages:
  Stage 0 (Thinker) → Stage 1 (Decoder): thinker token IDs
  Stage 1 (Decoder) → Stage 2 (Codec):   codebook frames
"""

from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt


# =============================================================================
# Stage 0 → Stage 1: Thinker → Decoder (async_chunk mode)
# =============================================================================


def thinker_to_decoder_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Stream thinker token IDs to the decoder stage, chunk by chunk.

    Chroma2's thinker output is just token IDs (integers), not embedding
    vectors. This makes the transfer extremely lightweight compared to
    Qwen3-Omni which streams embedding tensors.

    chunk_id == 0 (first chunk):
        Send initial context: prompt info + first batch of thinker tokens.
    chunk_id > 0 (subsequent chunks):
        Send incremental thinker token IDs only.

    The decoder stage consumes these tokens according to concat_pattern:
    e.g., "T,T,T,A,A,A,A,A,A" means every 3 thinker tokens enable
    6 audio frames of decoder generation.
    """
    request_id = request.external_req_id
    chunk_id = transfer_manager.put_req_chunk.get(request_id, 0)

    if chunk_id == 0:
        # First chunk: include prompt context + all thinker tokens so far
        all_token_ids = list(request.all_token_ids)
        prompt_token_ids = list(request.prompt_token_ids)

        info = {
            "thinker_prompt_token_ids": prompt_token_ids,
            "thinker_all_token_ids": all_token_ids,
            "finished": torch.tensor(is_finished, dtype=torch.bool),
        }

        # Buffer until we have enough tokens or thinker finishes
        if not is_finished:
            if transfer_manager.request_payload.get(request_id) is None:
                transfer_manager.request_payload[request_id] = info
                return None
            else:
                # Merge with buffered payload
                prev = transfer_manager.request_payload.pop(request_id)
                info["thinker_all_token_ids"] = list(request.all_token_ids)

        return info

    else:
        # Subsequent chunks: only send new thinker output token IDs
        output_token_ids = list(request.output_token_ids)

        info = {
            "thinker_output_token_ids": output_token_ids,
            "finished": torch.tensor(is_finished, dtype=torch.bool),
        }
        return info


def thinker_to_decoder(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Non-streaming version: thinker fully completes, then decoder starts.

    Used when async_chunk is disabled. Collects all thinker output token IDs
    and packages them for the decoder stage.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    stage = stage_list[source_stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    thinker_outputs = stage.engine_outputs
    decoder_inputs: list[OmniTokensPrompt] = []

    for thinker_output in thinker_outputs:
        output = thinker_output.outputs[0]
        thinker_token_ids = list(output.token_ids)
        prompt_token_ids = list(thinker_output.prompt_token_ids)

        # Decoder receives dummy prompt tokens; actual thinker tokens
        # are passed via additional_information and consumed in preprocess.
        additional_information = {
            "thinker_prompt_token_ids": prompt_token_ids,
            "thinker_output_token_ids": thinker_token_ids,
        }

        decoder_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * 1,  # dummy placeholder
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return decoder_inputs


# =============================================================================
# Stage 1 → Stage 2: Decoder → Codec (async_chunk mode)
# =============================================================================


def decoder_to_codec_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Stream codebook frames from decoder to Mimi codec stage.

    Each decoder step produces one frame of 8 codebooks.
    We accumulate frames and send in chunks (aligned to codec_chunk_frames
    from connector config, default 6 frames per concat_pattern cycle).

    The codec stage receives flat codebook data:
      [c0_f0, c1_f0, ..., c7_f0, c0_f1, c1_f1, ..., c7_f1, ...]
    """
    if pooling_output is None:
        return None

    codebook_frame = pooling_output.get("codebook_frame")
    if codebook_frame is None:
        return None

    # codebook_frame shape: [num_codebooks] for one frame
    if isinstance(codebook_frame, torch.Tensor):
        codebook_frame = codebook_frame.cpu().to(torch.long)
        if codebook_frame.numel() == 0:
            return None
    else:
        return None

    # Read chunk config from connector
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 6))
    left_context = int(cfg.get("codec_left_context_frames", 6))

    # Accumulate frames
    request_id = request.external_req_id
    transfer_manager.code_prompt_token_ids[request_id].append(
        codebook_frame.tolist()
    )

    num_frames = len(transfer_manager.code_prompt_token_ids[request_id])

    # Only send when we have a full chunk, or when finished
    if num_frames % chunk_size != 0 and not is_finished:
        return None

    # Compute context window
    context_frames = num_frames % chunk_size if num_frames % chunk_size != 0 else chunk_size
    actual_left_context = max(0, min(num_frames - context_frames, left_context))
    end_idx = min(num_frames, actual_left_context + context_frames)

    # Flatten frames to 1D: [c0_f0, c1_f0, ..., c7_f0, c0_f1, ...]
    frames = transfer_manager.code_prompt_token_ids[request_id][-end_idx:]
    flat_codes = []
    for frame in frames:
        flat_codes.extend(frame)

    info = {
        "codebook_codes": flat_codes,
        "left_context_frames": actual_left_context,
        "num_codebooks": codebook_frame.shape[-1] if isinstance(codebook_frame, torch.Tensor) else 8,
        "finished": torch.tensor(is_finished, dtype=torch.bool),
    }
    return info


def decoder_to_codec(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Non-streaming version: decoder fully completes, then codec decodes."""
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    stage = stage_list[source_stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    decoder_outputs = stage.engine_outputs
    codec_inputs: list[OmniTokensPrompt] = []

    for decoder_output in decoder_outputs:
        output = decoder_output.outputs[0]

        # codebook_frames stored in multimodal_output by the decoder stage
        codebook_frames = output.multimodal_output.get("codebook_frames")
        if codebook_frames is None:
            continue

        # Flatten all frames: [num_frames, num_codebooks] → 1D list
        if isinstance(codebook_frames, torch.Tensor):
            flat_codes = codebook_frames.cpu().to(torch.long).reshape(-1).tolist()
        else:
            flat_codes = codebook_frames

        codec_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=flat_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return codec_inputs
