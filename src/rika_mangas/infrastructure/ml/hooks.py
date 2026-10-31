# infrastructure/ml/hooks.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


class ReferenceFeatureBank:
    def __init__(self) -> None:
        self.features: dict[str, torch.Tensor] = {}
        self._handles: list[RemovableHandle] = []

    def register(self, module: torch.nn.Module, name: str) -> None:
        # Pre-hook para capturar hidden_states antes de las proyecciones Q, K, V
        def pre_hook(_mod: torch.nn.Module, inp: tuple[Any, ...]) -> None:
            hidden_states = inp[0]
            self.features[name] = hidden_states.detach()

        handle = module.register_forward_pre_hook(pre_hook)
        self._handles.append(handle)

    def clear(self) -> None:
        self.features.clear()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.features.clear()


class MangaNinjaAttnProcessor2_0:
    def __init__(self, block_name: str, bank: ReferenceFeatureBank) -> None:
        self.block_name = block_name
        self.bank = bank
        self.ref_key = block_name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        ref_feat = self.bank.features.get(self.ref_key)

        if not is_cross_attention and ref_feat is not None:
            ref_feat = ref_feat.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if ref_feat.ndim == 4:
                ref_feat = ref_feat.view(ref_feat.shape[0], ref_feat.shape[1], -1).transpose(1, 2)

            # Ajustar batch_size de ref_feat al de hidden_states (CFG)
            if ref_feat.shape[0] != batch_size:
                ref_feat = ref_feat.expand(batch_size, -1, -1)

            key_target = attn.to_k(hidden_states)
            val_target = attn.to_v(hidden_states)

            key_ref = attn.to_k(ref_feat)
            val_ref = attn.to_v(ref_feat)

            key = torch.cat([key_target, key_ref], dim=1)
            value = torch.cat([val_target, val_ref], dim=1)
        else:
            ctx = encoder_hidden_states if is_cross_attention else hidden_states
            key = attn.to_k(ctx)
            value = attn.to_v(ctx)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor
