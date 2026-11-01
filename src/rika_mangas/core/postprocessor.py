import cv2
import numpy as np

from rika_mangas.core.masks import gutter_mask
from rika_mangas.domain.models.presets import StylePreset, get_style


class PostProcessor:
    def __init__(
        self,
        *,
        l_channel: bool = True,
        guided_filter: bool = True,
        neutral_preservation: bool = True,
        style: StylePreset | None = None,
    ) -> None:
        self.l_channel = l_channel
        self.guided_filter = guided_filter
        self.neutral_preservation = neutral_preservation
        self.style = style or get_style("neutral")

    def process(
        self,
        colorized: np.ndarray,
        original_gray: np.ndarray,
        reference_bgr: np.ndarray | None = None,
        fidelity: float | None = None,
    ) -> np.ndarray:
        gray = self._to_gray(original_gray)
        if gray.shape[:2] != colorized.shape[:2]:
            gray = cv2.resize(
                gray,
                (colorized.shape[1], colorized.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        # 1. Ponderación de fidelidad y estilo
        raw_fidelity = 1.0 if fidelity is None else float(fidelity)
        style_weight = float(np.clip(raw_fidelity, 0.0, 1.0))

        # 2. Gradación tonal y cromática en espacio LAB
        lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)

        # Saturación y tintes armónicos según el preset
        target_boost = self.style.saturation_boost
        boost = 1.0 + (target_boost - 1.0) * style_weight

        delta_a = a - 128.0
        delta_b = b - 128.0
        chroma_norm = np.clip(np.sqrt(delta_a**2 + delta_b**2) / 60.0, 0.15, 1.0)

        a_graded = (
            128.0 + delta_a * boost + (self.style.tint_a * style_weight * chroma_norm)
        )
        b_graded = (
            128.0 + delta_b * boost + (self.style.tint_b * style_weight * chroma_norm)
        )

        # 3. Curva Gamma y elevación de pedestal en luminosidad
        if abs(self.style.gamma - 1.0) > 0.005 or self.style.shadow_lift > 0.0:
            eff_gamma = 1.0 + (self.style.gamma - 1.0) * style_weight
            l_norm = np.clip(l / 255.0, 0.0, 1.0)
            l_curve = np.power(l_norm, eff_gamma) * 255.0
            eff_lift = self.style.shadow_lift * style_weight
            l = np.clip(l_curve + eff_lift * (1.0 - l_norm), 0.0, 255.0)

        # 4. Filtro guiado de crominancia con guía libre de ruido de tramas (screentones)
        if self.guided_filter:
            a_graded, b_graded = self._guided_chroma(gray, a_graded, b_graded)

        # 5. Preservación de trazos de contorno (excluye tramas de puntos y rellenos negros beta)
        black_thr = float(self.style.black_threshold)
        if black_thr > 0.0:
            ink_weight = self._stroke_ink_weight(gray, black_thr)
            target_l = np.minimum(l, gray.astype(np.float32))
            l = l * (1.0 - ink_weight) + target_l * ink_weight

        # 6. Preservación de neutros en márgenes exteriores reales con transición suave
        if self.neutral_preservation:
            gutters = gutter_mask(gray, white_thr=int(self.style.white_threshold))
            if np.any(gutters):
                gutter_alpha = cv2.GaussianBlur(
                    gutters.astype(np.float32), (5, 5), 0
                )
                a_graded = a_graded * (1.0 - gutter_alpha) + 128.0 * gutter_alpha
                b_graded = b_graded * (1.0 - gutter_alpha) + 128.0 * gutter_alpha
                l = l * (1.0 - gutter_alpha) + 255.0 * gutter_alpha

        # 7. Reconstrucción BGR final
        lab_final = np.stack(
            [
                np.clip(l, 0.0, 255.0),
                np.clip(a_graded, 0.0, 255.0),
                np.clip(b_graded, 0.0, 255.0),
            ],
            axis=-1,
        )
        return cv2.cvtColor(lab_final.astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _stroke_ink_weight(gray: np.ndarray, black_thr: float) -> np.ndarray:
        """Aísla líneas de contorno finas descartando tramas densas y rellenos beta."""
        dark = (gray <= black_thr).astype(np.uint8)
        if not np.any(dark):
            return np.zeros_like(gray, dtype=np.float32)

        # 1. Excluir rellenos negros beta y mallas densas de trama por densidad local
        local_density = cv2.boxFilter(
            dark.astype(np.float32), ddepth=-1, ksize=(11, 11)
        )
        thin_strokes = ((dark > 0) & (local_density <= 0.38)).astype(np.uint8)

        # 2. Excluir micro-puntos aislados de trama (screentones de área < 18px)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            thin_strokes, connectivity=8
        )
        if n_labels <= 1:
            return np.zeros_like(gray, dtype=np.float32)

        areas = stats[:, cv2.CC_STAT_AREA]
        valid_labels = areas >= 18
        valid_labels[0] = False
        stroke_mask = valid_labels[labels]

        weight = np.zeros_like(gray, dtype=np.float32)
        g_float = gray.astype(np.float32)
        weight[stroke_mask] = (
            np.clip((black_thr - g_float[stroke_mask]) / max(black_thr, 1.0), 0.0, 1.0)
            * 0.65
        )
        return weight

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _guided_chroma(
        self, gray: np.ndarray, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Suaviza A/B usando una guía de línea sin tramas de puntos."""
        radius = int(self.style.guided_filter_radius)
        ximgproc = getattr(cv2, "ximgproc", None)
        if radius <= 0 or ximgproc is None:
            return a, b
        # Cierre morfológico + filtro bilateral para eliminar micro-puntos de trama
        # en la guía pero conservar nítidos los contornos reales del dibujo.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        smooth_guide = cv2.bilateralFilter(
            closed, d=5, sigmaColor=35.0, sigmaSpace=3.0
        )
        guide = smooth_guide.astype(np.float32) / 255.0
        eps = float(self.style.guided_filter_eps) ** 2
        out: list[np.ndarray] = []
        for ch in (a, b):
            norm = (ch / 255.0).astype(np.float32)
            filtered = ximgproc.guidedFilter(guide, norm, radius, eps)
            out.append(np.asarray(filtered, dtype=np.float32) * 255.0)
        return out[0], out[1]
