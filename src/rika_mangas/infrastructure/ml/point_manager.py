from __future__ import annotations

import cv2
import numpy as np
import torch

from rika_mangas.domain.ports.ports import HintPoint
from rika_mangas.infrastructure.ml.point_network import PointNet


def _to_pixel_coords(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Convierte coordenadas normalizadas [0, 1] o en píxeles a índices válidos."""
    cx = int(x * width) if 0.0 <= x <= 1.0 else int(x)
    cy = int(y * height) if 0.0 <= y <= 1.0 else int(y)
    return max(0, min(cx, width - 1)), max(0, min(cy, height - 1))


def _rgb_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    """Convierte una tupla RGB [0..255] al espacio CIELAB float32."""
    r, g, b = rgb
    bgr_px = np.array([[[b, g, r]]], dtype=np.uint8)
    return cv2.cvtColor(bgr_px, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]


class PointManager:
    """Responsable de la rasterización por pares e IDs y codificación con PointNet."""

    MAX_POINTS: int = 14

    @classmethod
    def generate_point_masks(
        cls,
        hints: list[HintPoint] | None,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
        ref_bgr: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Genera máscaras pareadas (ref, main) con IDs incrementales 1..N."""
        mask_ref = torch.zeros((1, 1, height, width), dtype=dtype, device=device)
        mask_main = torch.zeros((1, 1, height, width), dtype=dtype, device=device)
        if not hints:
            return mask_ref, mask_main

        ref_lab: np.ndarray | None = None
        if ref_bgr is not None and any(len(h) == 3 for h in hints[: cls.MAX_POINTS]):
            ref_resized = cv2.resize(
                ref_bgr, (width, height), interpolation=cv2.INTER_AREA
            )
            ref_lab = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

        for index, hint in enumerate(hints[: cls.MAX_POINTS]):
            val = float(index + 1)
            if len(hint) == 4:
                rx, ry, mx, my = hint
                cx_ref, cy_ref = _to_pixel_coords(rx, ry, width, height)
                cx_main, cy_main = _to_pixel_coords(mx, my, width, height)
                mask_ref[0, 0, cy_ref, cx_ref] = val
                mask_main[0, 0, cy_main, cx_main] = val
            else:
                mx, my, rgb = hint
                cx_main, cy_main = _to_pixel_coords(mx, my, width, height)
                mask_main[0, 0, cy_main, cx_main] = val
                if ref_lab is not None:
                    target_lab = _rgb_to_lab(rgb)
                    dist_sq = np.sum((ref_lab - target_lab) ** 2, axis=-1)
                    best_idx = int(np.argmin(dist_sq))
                    cy_ref, cx_ref = divmod(best_idx, width)
                    mask_ref[0, 0, cy_ref, cx_ref] = val

        return mask_ref, mask_main

    @staticmethod
    def generate_point_mask(
        hints: list[HintPoint] | None,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
        radius: int = 0,
    ) -> torch.Tensor:
        """Wrapper compatible que devuelve únicamente la máscara principal de puntos."""
        _, mask_main = PointManager.generate_point_masks(
            hints=hints,
            width=width,
            height=height,
            device=device,
            dtype=dtype,
        )
        return mask_main

    @staticmethod
    def encode_points(
        point_net: PointNet,
        point_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extrae la pirámide de embeddings multiescala mediante PointNet."""
        return point_net(point_mask)
