import cv2
import numpy as np

from rika_mangas.application.region_segmenter import segment_regions
from rika_mangas.domain.ports.ports import HintPoint

DEFAULT_PALETTE: dict[str, str] = {
    "skin": "#f0c8a0",
    "hair": "#6b4a32",
    "clothing_primary": "#5b7fa6",
    "clothing_secondary": "#a65e50",
    "clothing_accent": "#c9a227",
    "sky": "#a5c6e8",
    "background": "#d9cfc0",
}


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    s = hex_color.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


class DeterministicGuidedHints:
    def __init__(self, faces_detector=None) -> None:
        self._palette = DEFAULT_PALETTE
        self._faces_detector = faces_detector

    def generate_hints(self, image_bgr: np.ndarray) -> list[HintPoint]:
        H, W = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        seg = segment_regions(gray)
        if not seg.regions:
            return []

        faces = self._faces_detector.detect_faces(image_bgr) if self._faces_detector else []
        clothing_keys = ("clothing_primary", "clothing_secondary", "clothing_accent")
        cloth_idx = 0
        hints: list[HintPoint] = []

        for region in seg.regions:
            if region.mean_gray >= 238 or region.mean_gray < 70:
                continue

            rx, ry, rw, rh = region.bbox
            tag: str | None = None

            # Detección espacial por intersección de rostros
            for face in faces:
                fx, fy, fw, fh = face.x, face.y, face.w, face.h
                if not (rx + rw < fx or rx > fx + fw or ry + rh < fy or ry > fy + fh):
                    tag = "skin"
                    break
                if rx < fx + fw and rx + rw > fx and ry < fy and (fy - (ry + rh)) < int(fh * 0.4):
                    tag = "hair"
                    break
                if rx < fx + fw and rx + rw > fx and ry >= fy + fh and (ry - (fy + fh)) < int(fh * 1.5):
                    tag = clothing_keys[cloth_idx % len(clothing_keys)]
                    cloth_idx += 1
                    break

            if tag is None:
                if region.frac > 0.12 and ry < H * 0.25 and region.mean_gray > 190:
                    tag = "sky"
                elif region.frac > 0.15:
                    tag = "background"

            if tag and tag in self._palette:
                rgb = hex_to_rgb(self._palette[tag])
                px, py = region.interior_point
                v = int(gray[min(H - 1, max(0, py)), min(W - 1, max(0, px))])
                if 70 <= v < 238:
                    hints.append((px / W, py / H, rgb))

        return hints
