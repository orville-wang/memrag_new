#!/usr/bin/env python3
"""Minimal MemGen-style latent injector for frozen-policy experiments.

The official MemGen model consumes latent memory as input embeddings with shape
``[batch, latent_len, reasoner_hidden]``.  This module keeps that interface but
lets the experiment harness provide latents from an external compressor instead
of running the Weaver LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from memgen.model.weaver import MemGenWeaver
except Exception:  # pragma: no cover - import is available in the MemGen env.
    MemGenWeaver = nn.Module


@dataclass
class InjectionResult:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    latent_mask: torch.Tensor


class MemoryInjector(MemGenWeaver):
    """Drop-in prompt/inference augmenter backed by external latent tensors.

    The class intentionally does not call ``MemGenWeaver.__init__`` because it
    does not own a Weaver LLM.  It only implements the small augmentation
    contract used by MemGen: return latent hidden states plus mask/positions.
    """

    def __init__(self, hidden_size: int, prompt_latents_len: int, inference_latents_len: Optional[int] = None):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self._prompt_latents_len = int(prompt_latents_len)
        self._inference_latents_len = int(inference_latents_len if inference_latents_len is not None else prompt_latents_len)
        self._prompt_latents: Optional[torch.Tensor] = None
        self._inference_latents: Optional[torch.Tensor] = None

    @property
    def prompt_latents_num(self) -> int:
        return self._prompt_latents_len

    @property
    def inference_latents_num(self) -> int:
        return self._inference_latents_len

    @property
    def device(self):
        if self._prompt_latents is not None:
            return self._prompt_latents.device
        return torch.device("cpu")

    def set_prompt_latents(self, latents: torch.Tensor) -> None:
        self._prompt_latents = self._validate_latents(latents, self._prompt_latents_len)

    def set_inference_latents(self, latents: torch.Tensor) -> None:
        self._inference_latents = self._validate_latents(latents, self._inference_latents_len)

    def _validate_latents(self, latents: torch.Tensor, expected_len: int) -> torch.Tensor:
        if latents.dim() != 3:
            raise ValueError(f"latents must be [B, L, H], got {tuple(latents.shape)}")
        if latents.size(1) != expected_len:
            raise ValueError(f"expected latent len {expected_len}, got {latents.size(1)}")
        if latents.size(2) != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {latents.size(2)}")
        return latents

    def _augment(
        self,
        latents: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self._validate_latents(latents, latents.size(1))
        if latents.size(0) != attention_mask.size(0):
            raise ValueError("latent batch size must match attention_mask batch size")
        latents_mask = torch.ones(latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        last_position_ids = position_ids.max(dim=1)[0]
        relative_positions = torch.arange(latents.size(1), device=attention_mask.device)
        latents_position_ids = last_position_ids.unsqueeze(1) + relative_positions + 1
        return latents, latents_mask, latents_position_ids.long()

    def augment_prompt(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._prompt_latents is None:
            raise RuntimeError("prompt latents have not been set")
        latents = self._prompt_latents.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        return self._augment(latents, attention_mask, position_ids)

    def augment_inference(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self._inference_latents if self._inference_latents is not None else self._prompt_latents
        if latents is None:
            raise RuntimeError("inference latents have not been set")
        latents = latents.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        return self._augment(latents, attention_mask, position_ids)


def make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.cumsum(-1) - 1).clamp(min=0).long()


def inject_after_prompt(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    latents: Optional[torch.Tensor],
) -> InjectionResult:
    """Append latent memory after the prompt and return model-ready tensors."""

    if latents is None or latents.size(1) == 0:
        position_ids = make_position_ids(attention_mask)
        latent_mask = torch.zeros(attention_mask.shape, dtype=torch.bool, device=attention_mask.device)
        return InjectionResult(inputs_embeds, attention_mask, position_ids, latent_mask)

    latents = latents.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    if latents.dim() != 3 or latents.size(0) != inputs_embeds.size(0) or latents.size(2) != inputs_embeds.size(2):
        raise ValueError(
            "latents must be [B, L, H] matching prompt embeddings; "
            f"got {tuple(latents.shape)} vs {tuple(inputs_embeds.shape)}"
        )

    latent_attention = torch.ones(latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
    merged_embeds = torch.cat([inputs_embeds, latents], dim=1)
    merged_attention = torch.cat([attention_mask, latent_attention], dim=1)
    position_ids = make_position_ids(merged_attention)
    latent_mask = torch.cat(
        [
            torch.zeros(attention_mask.shape, dtype=torch.bool, device=attention_mask.device),
            torch.ones(latent_attention.shape, dtype=torch.bool, device=attention_mask.device),
        ],
        dim=1,
    )
    return InjectionResult(merged_embeds, merged_attention, position_ids, latent_mask)
