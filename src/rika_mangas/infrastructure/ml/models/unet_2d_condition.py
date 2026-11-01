from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.activations import get_activation

from rika_mangas.infrastructure.ml.models.refunet_2d_condition import (
    RefUNet2DConditionModel,
)


class UNet2DConditionModel(RefUNet2DConditionModel):
    """UNet de eliminación de ruido con proyección de salida reconstructiva."""

    def __init__(self, out_channels: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        first_stage_ch = self.config["block_out_channels"][0]
        self.conv_norm_out = nn.GroupNorm(
            num_channels=first_stage_ch, num_groups=32, eps=1e-5
        )
        self.conv_act = get_activation("silu")
        self.conv_out = nn.Conv2d(
            first_stage_ch, out_channels, kernel_size=3, padding=1
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals: tuple[torch.Tensor, ...] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        out = super().forward(
            sample,
            timestep,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            **kwargs,
        )[0]

        out = self.conv_norm_out(out)
        out = self.conv_act(out)
        out = self.conv_out(out)
        return (out,)
