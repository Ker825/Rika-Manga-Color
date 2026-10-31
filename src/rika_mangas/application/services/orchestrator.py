from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rika_mangas.config import settings
from rika_mangas.core.postprocessor import PostProcessor
from rika_mangas.domain.models.presets import get_style
from rika_mangas.domain.ports.ports import IColorizerEngine, IUpscalerEngine


class ColorizationOrchestrator:
    def __init__(
        self,
        colorizer: IColorizerEngine,
        upscaler: IUpscalerEngine | None = None,
    ) -> None:
        self._colorizer = colorizer
        self._upscaler = upscaler
        # Se desactivan máscaras destructivas y preservación neutra
        self._postprocessor = PostProcessor(
            style=get_style("neutral"),
            hard_masks=False,
            neutral_preservation=False,
            l_channel=True,
        )

    def process_page(
        self,
        job_id: str,
        page_idx: int,
        page_path: Path,
        reference_paths: list[Path],
        output_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        img_bgr = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Imposible leer la imagen en {page_path}")

        refs_bgr: list[np.ndarray] = []
        for ref_p in reference_paths:
            ref = cv2.imread(str(ref_p), cv2.IMREAD_COLOR)
            if ref is not None:
                refs_bgr.append(ref)

        try:
            colorized = self._colorizer.colorize_page(
                img_bgr,
                reference_images=refs_bgr,
                steps=settings.MANGANINJA_STEPS,
            )
        finally:
            self._colorizer.unload()

        # TEST DE AISLAMIENTO: Guardar salida directa de la IA
        cv2.imwrite(str(output_dir / "debug_raw_ia.png"), colorized)

        # Solo fusionar tinta negra original sobre el color
        processed = self._postprocessor.process(colorized, img_bgr)

        if self._upscaler is not None:
            try:
                processed = self._upscaler.upscale(processed)
            finally:
                self._upscaler.unload()

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"colored_{page_idx:04d}.png"
        if not cv2.imwrite(str(out_path), processed):
            raise OSError(f"Fallo al guardar la imagen procesada en {out_path}")

        return out_path
