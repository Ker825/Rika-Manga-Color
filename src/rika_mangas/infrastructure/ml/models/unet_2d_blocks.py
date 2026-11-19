from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D

from rika_mangas.infrastructure.ml.models.transformer_2d import Transformer2DModel


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_groups: int = 32,
        add_downsample: bool = True,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                )
                for i in range(num_layers)
            ]
        )
        self.downsamplers = (
            nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=1,
                        name="op",
                    )
                ]
            )
            if add_downsample
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        output_states: tuple[torch.Tensor, ...] = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_groups: int = 32,
        add_upsample: bool = True,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            res_skip = in_channels if (i == num_layers - 1) else out_channels
            r_in = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=r_in + res_skip,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                )
            )
        self.upsamplers = (
            nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                    )
                ]
            )
            if add_upsample
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
        temb: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        remaining_res = res_hidden_states_tuple
        for resnet in self.resnets:
            res_hidden = remaining_res[-1]
            remaining_res = remaining_res[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-5,
        resnet_groups: int = 32,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 768,
        add_downsample: bool = True,
    ) -> None:
        super().__init__()
        self.has_cross_attention = True
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                )
            )
            self.attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )

        self.downsamplers = (
            nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=1,
                        name="op",
                    )
                ]
            )
            if add_downsample
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        output_states: tuple[torch.Tensor, ...] = ()
        for resnet, attn in zip(self.resnets, self.attentions, strict=True):
            hidden_states = resnet(hidden_states, temb)
            hidden_states, _ = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                return_dict=False,
            )
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        resnet_eps: float = 1e-5,
        resnet_groups: int = 32,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 768,
        transformer_layers_per_block: int = 1,
    ) -> None:
        super().__init__()
        self.has_cross_attention = True
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                ),
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                ),
            ]
        )
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states, _ = self.attentions[0](
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            return_dict=False,
        )
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-5,
        resnet_groups: int = 32,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 768,
        add_upsample: bool = True,
    ) -> None:
        super().__init__()
        self.has_cross_attention = True
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(num_layers):
            res_skip = in_channels if (i == num_layers - 1) else out_channels
            r_in = prev_output_channel if i == 0 else out_channels

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=r_in + res_skip,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                )
            )
            self.attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )

        self.upsamplers = (
            nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                    )
                ]
            )
            if add_upsample
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
        temb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        remaining_res = res_hidden_states_tuple
        for resnet, attn in zip(self.resnets, self.attentions, strict=True):
            res_hidden = remaining_res[-1]
            remaining_res = remaining_res[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states, _ = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                return_dict=False,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
