from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from rika_mangas.infrastructure.security import safe_load_checkpoint


class ResidualBlock(nn.Module):
    """Bloque residual para el generador de extracción de lineart."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class Generator(nn.Module):
    """Red encoder-decoder con bloques residuales para detección de lineart."""

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 1,
        n_residual_blocks: int = 3,
        sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        model1: list[nn.Module] = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        self.model2 = nn.Sequential(
            *(ResidualBlock(in_features) for _ in range(n_residual_blocks))
        )

        model3: list[nn.Module] = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        model4: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
        ]
        if sigmoid:
            model4 += [nn.Sigmoid()]
        self.model4 = nn.Sequential(*model4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        return self.model4(out)


class BatchLineartDetector(nn.Module):
    """Detector neuronal de lineart compatible con los pesos sk_model.pth de MangaNinja."""

    def __init__(self, checkpoint_path: Path, expected_hash: str | None = None) -> None:
        super().__init__()
        state_dict = safe_load_checkpoint(checkpoint_path, expected_hash)
        model = Generator(3, 1, 3)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model

    def forward(
        self,
        input_image: torch.Tensor,
        mean: float = -1.0,
        std: float = 2.0,
    ) -> torch.Tensor:
        with torch.no_grad():
            image = (input_image - mean) / std
            line = self.model(image)
        return 1.0 - line
