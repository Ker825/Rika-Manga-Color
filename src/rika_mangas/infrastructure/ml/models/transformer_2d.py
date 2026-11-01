from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput


@dataclass
class Transformer2DModelOutput(BaseOutput):
    sample: torch.Tensor
    ref_feature: torch.Tensor


class Transformer2DModel(ModelMixin, ConfigMixin):  # type: ignore[misc]
    """Transformer espacial 2D con bifurcación de características de referencia."""

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        norm_num_groups: int = 32,
    ) -> None:
        super().__init__()
        if in_channels is None:
            raise ValueError("in_channels debe ser especificado.")

        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        self.proj_in = nn.Conv2d(
            in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn="geglu",
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Conv2d(
            self.inner_dim, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        norm_states = self.norm(hidden_states)
        proj_states = self.proj_in(norm_states)
        inner_dim = proj_states.shape[1]

        flat_states = proj_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        ref_feature = flat_states.reshape(batch, height, width, inner_dim)

        for block in self.transformer_blocks:
            flat_states = block(
                flat_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                attention_mask=attention_mask,
            )

        out = (
            flat_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        out = self.proj_out(out) + residual

        if not return_dict:
            return out, ref_feature

        return Transformer2DModelOutput(sample=out, ref_feature=ref_feature)
