import math
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rika_mangas.domain.ports.ports import IUpscalerEngine
from rika_mangas.infrastructure.security import safe_load_checkpoint


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 6,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        if scale != 4:
            raise ValueError(
                f"Escala no soportada: {scale}. Solo scale=4 está permitido."
            )
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *(RRDB(num_feat, num_grow_ch) for _ in range(num_block))
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class RealESRGANCPUAccumulator(IUpscalerEngine):
    def __init__(
        self, weights_path: Path, tile_size: int = 512, tile_pad: int = 10
    ) -> None:
        self.weights_path = weights_path
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.scale = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._model: RRDBNet | None = None

    def _ensure_model(self) -> RRDBNet:
        if self._model is not None:
            return self._model

        state = safe_load_checkpoint(self.weights_path)
        state_dict = state.get("params_ema", state.get("params", state))

        model = RRDBNet(num_block=6, scale=self.scale)
        model.load_state_dict(cast(dict[str, torch.Tensor], state_dict), strict=True)
        model.eval()

        self._model = (
            model.half().to(self.device)
            if self.device.type == "cuda"
            else model.to(self.device)
        )
        return self._model

    @torch.inference_mode()
    def upscale(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.size == 0:
            raise ValueError("Imagen vacía suministrada para escalado.")

        model = self._ensure_model()
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1]).astype(np.float32) / 255.0

        img_cpu = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
        batch, channel, height, width = img_cpu.shape
        out_h, out_w = height * self.scale, width * self.scale

        # Buffer alojado estrictamente en RAM de CPU
        output_cpu = torch.zeros(
            (batch, channel, out_h, out_w), dtype=torch.uint8, device="cpu"
        )

        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                x_start, y_start = x * self.tile_size, y * self.tile_size
                x_end, y_end = (
                    min(x_start + self.tile_size, width),
                    min(y_start + self.tile_size, height),
                )

                x_start_pad = max(x_start - self.tile_pad, 0)
                y_start_pad = max(y_start - self.tile_pad, 0)
                x_end_pad = min(x_end + self.tile_pad, width)
                y_end_pad = min(y_end + self.tile_pad, height)

                tile_gpu = img_cpu[
                    :, :, y_start_pad:y_end_pad, x_start_pad:x_end_pad
                ].to(self.device, dtype=self.dtype)
                out_tile = model(tile_gpu)

                ox_start = (x_start - x_start_pad) * self.scale
                oy_start = (y_start - y_start_pad) * self.scale
                ox_end = ox_start + (x_end - x_start) * self.scale
                oy_end = oy_start + (y_end - y_start) * self.scale

                dx_start, dy_start = x_start * self.scale, y_start * self.scale
                dx_end, dy_end = x_end * self.scale, y_end * self.scale

                out_uint8 = (
                    out_tile[:, :, oy_start:oy_end, ox_start:ox_end]
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .add(0.5)
                    .to(torch.uint8)
                    .cpu()
                )
                output_cpu[:, :, dy_start:dy_end, dx_start:dx_end] = out_uint8
                del tile_gpu, out_tile, out_uint8

        res = output_cpu.squeeze(0).permute(1, 2, 0).numpy()
        return cast(np.ndarray, np.ascontiguousarray(res[:, :, ::-1]))

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
