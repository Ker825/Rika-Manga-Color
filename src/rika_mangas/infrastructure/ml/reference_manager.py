from __future__ import annotations

import cv2
import numpy as np
import torch


class ReferenceManager:
    """Responsable exclusivo de la selección, normalización y extracción de características de referencia."""

    @staticmethod
    def calculate_chroma(img_bgr: np.ndarray) -> float:
        """Calcula la vivacidad de color (croma) de una imagen en el espacio CIELAB."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Distancia euclídea respecto al punto neutro (128, 128) en los canales A y B
        a_dev = lab[:, :, 1] - 128.0
        b_dev = lab[:, :, 2] - 128.0
        return float(np.sqrt(a_dev**2 + b_dev**2).mean())

    @classmethod
    def select_best_reference(
        cls, references: list[np.ndarray] | None, default_shape: tuple[int, int]
    ) -> np.ndarray:
        """Selecciona la referencia de mayor riqueza cromática o genera una base neutra si no hay."""
        if not references:
            h, w = default_shape
            return np.full((h, w, 3), 200, dtype=np.uint8)
        if len(references) == 1:
            return references[0]
        return max(references, key=cls.calculate_chroma)

    @staticmethod
    def prepare_reference_tensor(
        ref_bgr: np.ndarray,
        target_w: int,
        target_h: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Redimensiona y normaliza la referencia al rango [-1.0, 1.0] como tensor PyTorch."""
        ref_resized = cv2.resize(
            ref_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA
        )
        ref_rgb = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2RGB)
        ref_norm = (ref_rgb.astype(np.float32) / 255.0) * 2.0 - 1.0
        ref_tensor = (
            torch.from_numpy(ref_norm)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        return ref_rgb, ref_tensor
