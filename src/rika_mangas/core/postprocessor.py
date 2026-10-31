import cv2
import numpy as np

from rika_mangas.domain.models.presets import StylePreset, get_style

_GUIDED_FILTER_MAX_EDGE = 1024


class PostProcessor:
    def __init__(
        self,
        *,
        l_channel: bool = True,
        guided_filter: bool = True,
        neutral_preservation: bool = True,
        saturation_boost: float = 1.3,
        guided_filter_radius: int = 2,
        guided_filter_eps: float = 0.01,
        hard_masks: bool = True,
        style: StylePreset | None = None,
    ) -> None:
        self.l_channel = l_channel
        self.guided_filter = guided_filter
        self.neutral_preservation = neutral_preservation
        self.saturation_boost = saturation_boost
        self.guided_filter_radius = guided_filter_radius
        self.guided_filter_eps = guided_filter_eps
        self.hard_masks = hard_masks
        self.style = style or get_style("neutral")

    def process(self, colorized: np.ndarray, original_gray: np.ndarray) -> np.ndarray:
        gray = self._to_gray(original_gray)
        if gray.shape[:2] != colorized.shape[:2]:
            gray = cv2.resize(gray, (colorized.shape[1], colorized.shape[0]), interpolation=cv2.INTER_AREA)

        lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Suavizado de ruido
        lab[:, :, 1] = cv2.medianBlur(lab[:, :, 1].astype(np.uint8), 5)
        lab[:, :, 2] = cv2.medianBlur(lab[:, :, 2].astype(np.uint8), 5)

        # Inyectar multiplicador de saturación (centrado en 128)
        boost = self.style.saturation_boost if self.style else 1.5
        lab[:, :, 1] = np.clip(128.0 + (lab[:, :, 1] - 128.0) * boost, 0, 255)
        lab[:, :, 2] = np.clip(128.0 + (lab[:, :, 2] - 128.0) * boost, 0, 255)

        # Luminancia intacta
        lab[:, :, 0] = gray

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
