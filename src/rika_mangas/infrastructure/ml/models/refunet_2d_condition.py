from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin

from rika_mangas.infrastructure.ml.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
)


class RefUNet2DConditionModel(ModelMixin, ConfigMixin):  # type: ignore[misc]
    """UNet de referencia para propagación de características hacia el lector."""

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        cross_attention_dim: int = 768,
        attention_head_dim: int = 8,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(
            block_out_channels[0],
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Down Blocks
        output_channel = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = out_ch
            is_final = i == len(block_out_channels) - 1

            down_block: nn.Module
            if i < 3:
                down_block = CrossAttnDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    num_attention_heads=attention_head_dim,
                    add_downsample=not is_final,
                )
            else:
                down_block = DownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block,
                    add_downsample=not is_final,
                )
            self.down_blocks.append(down_block)

        # Mid Block
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=attention_head_dim,
        )

        # Up Blocks
        rev_channels = list(reversed(block_out_channels))
        output_channel = rev_channels[0]
        for i, out_ch in enumerate(rev_channels):
            is_final = i == len(block_out_channels) - 1
            prev_out_ch = output_channel
            output_channel = out_ch
            input_channel = rev_channels[min(i + 1, len(block_out_channels) - 1)]

            up_block: nn.Module
            if i > 0:
                up_block = CrossAttnUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_out_ch,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block + 1,
                    cross_attention_dim=cross_attention_dim,
                    num_attention_heads=attention_head_dim,
                    add_upsample=not is_final,
                )
            else:
                up_block = UpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_out_ch,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block + 1,
                    add_upsample=not is_final,
                )
            self.up_blocks.append(up_block)
            prev_out_ch = output_channel

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals: tuple[torch.Tensor, ...] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        if not torch.is_tensor(timestep):
            t_tensor = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            t_tensor = timestep[None].to(sample.device)
        else:
            t_tensor = timestep

        t_emb = self.time_proj(t_tensor.expand(sample.shape[0]))
        emb = self.time_embedding(t_emb.to(dtype=sample.dtype))

        sample = self.conv_in(sample)
        down_res_samples: tuple[torch.Tensor, ...] = (sample,)

        # 1. Down Blocks
        for down_block in self.down_blocks:
            if getattr(down_block, "has_cross_attention", False):
                sample, res_samples = down_block(
                    sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = down_block(sample, temb=emb)
            down_res_samples = down_res_samples + res_samples

        # Inyección ControlNet: Down residuals a conexiones skip
        if down_block_additional_residuals is not None:
            down_res_samples = tuple(
                res + add_res
                for res, add_res in zip(
                    down_res_samples, down_block_additional_residuals, strict=False
                )
            )

        # 2. Mid Block
        sample = self.mid_block(
            sample, temb=emb, encoder_hidden_states=encoder_hidden_states
        )

        # Inyección ControlNet: Mid residual en cuello de botella (12x12)
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 3. Up Blocks
        for up_block in self.up_blocks:
            res_count = len(up_block.resnets)
            res_samples = down_res_samples[-res_count:]
            down_res_samples = down_res_samples[:-res_count]
            if getattr(up_block, "has_cross_attention", False):
                sample = up_block(
                    sample,
                    res_hidden_states_tuple=res_samples,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = up_block(
                    sample,
                    res_hidden_states_tuple=res_samples,
                    temb=emb,
                )

        return (sample,)
