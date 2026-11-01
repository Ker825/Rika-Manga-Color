from __future__ import annotations

import torch

from rika_mangas.domain.ports.ports import HintPoint
from rika_mangas.infrastructure.ml.point_network import PointNet


class PointManager:
    """Responsable exclusivo de la rasterización y codificación multiescala de pistas espaciales."""

    @staticmethod
    def generate_point_mask(
        hints: list[HintPoint] | None,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
        radius: int = 2,
    ) -> torch.Tensor:
        """Genera un mapa espacial binario 2D con un radio de influencia para las pistas."""
        mask = torch.zeros((1, 1, height, width), dtype=dtype, device=device)
        if not hints:
            return mask

        for px, py, _ in hints:
            cx = max(0, min(int(px * width), width - 1))
            cy = max(0, min(int(py * height), height - 1))
            x0 = max(0, cx - radius)
            x1 = min(width, cx + radius + 1)
            y0 = max(0, cy - radius)
            y1 = min(height, cy + radius + 1)
            mask[0, 0, y0:y1, x0:x1] = 1.0

        return mask

    @staticmethod
    def encode_points(
        point_net: PointNet,
        point_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extrae la pirámide de embeddings multiescala mediante PointNet."""
        return point_net(point_mask)
