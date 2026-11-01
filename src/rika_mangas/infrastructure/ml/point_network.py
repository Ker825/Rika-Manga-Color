from __future__ import annotations

import torch
import torch.nn as nn


class PointNet(nn.Module):
    """Red convolucional para codificar pistas espaciales en embeddings latentes."""

    def __init__(
        self,
        conditioning_channels: int = 1,
        out_channels: tuple[int, ...] = (320, 640, 1280, 1280),
        downsamples: tuple[int, ...] = (6, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        current_channels = conditioning_channels

        for out_channel, downsample in zip(out_channels, downsamples, strict=True):
            layers: list[nn.Module] = []
            for _ in range(downsample // 2):
                layers.append(
                    nn.Conv2d(
                        in_channels=current_channels,
                        out_channels=out_channel,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )
                layers.append(nn.SiLU())
                current_channels = out_channel

            self.blocks.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        embeddings: list[torch.Tensor] = []
        embedding = x
        for block in self.blocks:
            embedding = block(embedding)
            b, c, h, w = embedding.shape
            embeddings.append(embedding.view(b, c, h * w).transpose(1, 2))

        return embeddings
