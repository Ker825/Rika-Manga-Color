from pathlib import Path

import cv2

from rika_mangas.application.hints.guided_hints import DeterministicGuidedHints
from rika_mangas.core.postprocessor import PostProcessor
from rika_mangas.domain.models.presets import StylePreset, get_style


class ColorizationOrchestrator:
    def __init__(self, colorizer, upscaler=None, hints_detector=None) -> None:
        self._colorizer = colorizer
        self._upscaler = upscaler
        self._hints_detector = hints_detector or DeterministicGuidedHints()

    def process_page(
        self,
        job_id: str,
        page_idx: int,
        page_path: Path,
        reference_paths: list[Path],
        output_dir: Path,
        style: str | StylePreset = "neutral",
        fidelity: float | None = None,
    ) -> Path:
        img_bgr = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"No se pudo cargar la imagen: {page_path}")

        refs_bgr = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in reference_paths]
        refs_bgr = [r for r in refs_bgr if r is not None]
        primary_ref = refs_bgr[0] if refs_bgr else None

        raw_colorized = self._colorizer.colorize_page(
            img_bgr,
            reference_images=refs_bgr or None,
            hint_points=None,
        )
        if raw_colorized is None:
            raise ValueError(f"El colorizador no genero una imagen para: {page_path}")

        raw_path = output_dir / f"raw_{page_idx:04d}.png"
        cv2.imwrite(str(raw_path), raw_colorized)

        preset_obj = get_style(style) if isinstance(style, str) else style
        postprocessor = PostProcessor(style=preset_obj)
        processed = postprocessor.process(
            colorized=raw_colorized,
            original_gray=img_bgr,
            reference_bgr=primary_ref,
            fidelity=fidelity,
        )

        if self._upscaler is not None:
            processed = self._upscaler.upscale(processed)

        # La imagen original de inferencia debe ser .png
        out_path = output_dir / f"colored_{page_idx:04d}.png"
        cv2.imwrite(str(out_path), processed)
        return out_path

    def reprocess_page(
        self,
        page_idx: int,
        page_path: Path,
        output_dir: Path,
        primary_ref_path: Path | None,
        style: str | StylePreset = "neutral",
        fidelity: float | None = None,
    ) -> Path:
        """Aplica postprocesado en CPU sobre la salida cruda ya generada sin tocar la GPU."""
        raw_path = output_dir / f"raw_{page_idx:04d}.png"
        if not raw_path.is_file():
            raw_path = output_dir / f"raw_{page_idx:04d}.webp"
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"Salida cruda no disponible para pagina {page_idx}"
            )

        raw_colorized = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)

        img_bgr = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        if raw_colorized is None:
            raise ValueError(f"No se pudo cargar la salida cruda: {raw_path}")
        if img_bgr is None:
            raise ValueError(f"No se pudo cargar la imagen: {page_path}")
        primary_ref = (
            cv2.imread(str(primary_ref_path), cv2.IMREAD_COLOR)
            if primary_ref_path and primary_ref_path.is_file()
            else None
        )

        preset_obj = get_style(style) if isinstance(style, str) else style
        postprocessor = PostProcessor(style=preset_obj)
        processed = postprocessor.process(
            colorized=raw_colorized,
            original_gray=img_bgr,
            reference_bgr=primary_ref,
            fidelity=fidelity,
        )

        if self._upscaler is not None:
            processed = self._upscaler.upscale(processed)

        out_path = output_dir / f"colored_{page_idx:04d}.png"
        cv2.imwrite(str(out_path), processed)
        return out_path
