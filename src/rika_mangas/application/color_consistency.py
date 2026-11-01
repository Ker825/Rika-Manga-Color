import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from rika_mangas.domain.models.entities import CharacterSignature

# =====================================================================
# 1. CONSISTENCIA GLOBAL ENTRE PÁGINAS (REINHARD LAB TRANSFER)
# =====================================================================
logger = logging.getLogger("rika_mangas.color_consistency")


class PageChromaTransfer:
    """Alinea la distribución global de color entre páginas usando Reinhard LAB.

    Calcula la media y desviación estándar de los canales de crominancia (A y B)
    en una página ancla de referencia y ajusta suavemente las páginas siguientes
    sin modificar la luminancia (canal L).
    """

    def __init__(
        self, clip_ratio_min: float = 0.75, clip_ratio_max: float = 1.35
    ) -> None:
        self.clip_min = clip_ratio_min
        self.clip_max = clip_ratio_max
        self._ref_mean_a: float | None = None
        self._ref_std_a: float | None = None
        self._ref_mean_b: float | None = None
        self._ref_std_b: float | None = None

    @property
    def has_reference(self) -> bool:
        return self._ref_mean_a is not None

    @staticmethod
    def _get_tonal_mask(image_bgr: np.ndarray) -> np.ndarray:
        """Excluye tinta profunda y papel blanco para no desviar las estadísticas."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return (gray > 35) & (gray < 235)

    def set_reference(self, image_bgr: np.ndarray) -> None:
        """Extrae la firma estadística de la página de referencia."""
        mask = self._get_tonal_mask(image_bgr)
        if np.count_nonzero(mask) < 200:
            return

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_vals = lab[mask, 1]
        b_vals = lab[mask, 2]

        self._ref_mean_a = float(np.mean(a_vals))
        self._ref_std_a = float(np.std(a_vals)) or 1.0
        self._ref_mean_b = float(np.mean(b_vals))
        self._ref_std_b = float(np.std(b_vals)) or 1.0

    def apply(self, image_bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Aplica la transferencia cromática a una página secundaria."""
        if not self.has_reference or strength <= 0.0:
            return image_bgr

        mask = self._get_tonal_mask(image_bgr)
        if np.count_nonzero(mask) < 200:
            return image_bgr

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_vals = lab[mask, 1]
        b_vals = lab[mask, 2]

        src_mean_a = float(np.mean(a_vals))
        src_std_a = float(np.std(a_vals)) or 1.0
        src_mean_b = float(np.mean(b_vals))
        src_std_b = float(np.std(b_vals)) or 1.0

        # Escalar dispersión restringida para evitar monocromatismo excesivo
        a_scale = float(
            np.clip((self._ref_std_a or 1.0) / src_std_a, self.clip_min, self.clip_max)
        )
        a_shift = (self._ref_mean_a or 128.0) - src_mean_a * a_scale
        l_chan = lab[:, :, 0]
        tonal_weight = np.clip((l_chan - 25.0) / 25.0, 0.0, 1.0) * np.clip(
            (245.0 - l_chan) / 20.0, 0.0, 1.0
        )
        lab[:, :, 1] += (
            strength
            * tonal_weight
            * (lab[:, :, 1] * (a_scale - 1.0) + a_shift)
        )

        b_scale = float(
            np.clip((self._ref_std_b or 1.0) / src_std_b, self.clip_min, self.clip_max)
        )
        b_shift = (self._ref_mean_b or 128.0) - src_mean_b * b_scale
        lab[:, :, 2] += (
            strength
            * tonal_weight
            * (lab[:, :, 2] * (b_scale - 1.0) + b_shift)
        )

        np.clip(lab, 0.0, 255.0, out=lab)
        return cast(np.ndarray, cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR))


# =====================================================================
# 2. MEMORIA DE PERSONAJES DETERMINISTA (SIN DEPENDENCIAS EXTERNAS)
# =====================================================================


@dataclass(slots=True, frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int


class CharacterMemoryLAB:
    _detector: Any | None = None

    def __init__(self, match_threshold: float = 18.0) -> None:
        self.match_threshold = match_threshold
        self.characters: list[Any] = []

        # Importación diferida para evitar una dependencia circular con Settings.
        from rika_mangas.config import settings

        # 1. Buscar en rutas candidatas prioritarias
        candidates = [
            settings.WEIGHTS_DIR / "haarcascade_frontalface_default.xml",
        ]

        cv2_data = getattr(cv2, "data", None)
        if cv2_data is not None and hasattr(cv2_data, "haarcascades"):
            candidates.append(
                Path(cv2_data.haarcascades) / "haarcascade_frontalface_default.xml"
            )

        cascade_path: Path | None = next((p for p in candidates if p.is_file()), None)

        if cascade_path is None:
            logger.warning(
                "Clasificador 'haarcascade_frontalface_default.xml' no encontrado. "
                "La memoria de personajes continuará en modo pasivo (sin detección facial)."  # noqa: E501
            )
            return

        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if cascade_cls is not None:
            detector = cascade_cls(str(cascade_path))
            if not getattr(detector, "empty", lambda: True)():
                self._detector = detector
            else:
                logger.warning("Fallo al parsear el XML del clasificador Haar Cascade.")

    def detect_faces(
        self, image_bgr: np.ndarray, max_edge: int = 1280
    ) -> list[FaceBox]:
        """Detecta rostros escalando la imagen internamente para optimizar CPU."""
        detector = self._detector
        if detector is None:
            return []

        h, w = image_bgr.shape[:2]
        scale = 1.0
        work_img = image_bgr

        if max(h, w) > max_edge:
            scale = max_edge / max(h, w)
            work_img = cv2.resize(
                image_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)

        # Se elimina flags=cv2.CASCADE_SCALE_IMAGE (redundante en OpenCV 4+)
        boxes = detector.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(28, 28),
        )

        inv = 1.0 / scale
        results: list[FaceBox] = []

        # round() devuelve int directamente; se eliminan los int(...) redundantes
        for x, y, bw, bh in boxes:
            rx = round(x * inv)
            ry = round(y * inv)
            rw = min(w - rx, round(bw * inv))
            rh = min(h - ry, round(bh * inv))
            if rw > 10 and rh > 10:
                results.append(FaceBox(x=rx, y=ry, w=rw, h=rh))

        return results

    def _extract_signatures(
        self,
        image_bgr: np.ndarray,
        lab_img: np.ndarray,
        face: FaceBox,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        h, w = image_bgr.shape[:2]
        x, y, fw, fh = face.x, face.y, face.w, face.h

        # 1. Piel (dentro del bounding box del rostro)
        face_bgr = image_bgr[y : y + fh, x : x + fw]
        face_lab = lab_img[y : y + fh, x : x + fw]

        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)

        skin_lab: np.ndarray | None = None
        if np.count_nonzero(skin_mask) >= 16:
            skin_lab = np.mean(face_lab[skin_mask], axis=0).astype(np.float32)

        # 2. Cabello (franja superior)
        y_hair_1 = max(0, y - int(fh * 0.45))
        y_hair_2 = min(h, y + int(fh * 0.15))
        x_hair_1 = max(0, x - int(fw * 0.15))
        x_hair_2 = min(w, x + fw + int(fw * 0.15))

        hair_lab: np.ndarray | None = None
        if y_hair_2 > y_hair_1 and x_hair_2 > x_hair_1:
            hair_bgr = image_bgr[y_hair_1:y_hair_2, x_hair_1:x_hair_2]
            hair_lab_roi = lab_img[y_hair_1:y_hair_2, x_hair_1:x_hair_2]
            hair_gray = cv2.cvtColor(hair_bgr, cv2.COLOR_BGR2GRAY)
            hair_mask = (hair_gray > 35) & (hair_gray < 235)
            if np.count_nonzero(hair_mask) >= 16:
                hair_lab = np.mean(hair_lab_roi[hair_mask], axis=0).astype(np.float32)

        # 3. Ropa (torso inferior)
        y_cloth_1 = min(h, y + fh)
        y_cloth_2 = min(h, y + int(fh * 2.5))
        x_cloth_1 = max(0, x - int(fw * 0.25))
        x_cloth_2 = min(w, x + fw + int(fw * 0.25))

        cloth_lab: np.ndarray | None = None
        if y_cloth_2 > y_cloth_1 and x_cloth_2 > x_cloth_1:
            cloth_bgr = image_bgr[y_cloth_1:y_cloth_2, x_cloth_1:x_cloth_2]
            cloth_lab_roi = lab_img[y_cloth_1:y_cloth_2, x_cloth_1:x_cloth_2]
            cloth_gray = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2GRAY)
            cloth_mask = (cloth_gray > 30) & (cloth_gray < 240)
            if np.count_nonzero(cloth_mask) >= 16:
                cloth_lab = np.mean(cloth_lab_roi[cloth_mask], axis=0).astype(
                    np.float32
                )

        return skin_lab, hair_lab, cloth_lab

    def _match_or_create(
        self,
        skin_lab: np.ndarray,
        hair_lab: np.ndarray | None,
        clothes_lab: np.ndarray | None,
        page_num: int,
    ) -> CharacterSignature:
        best_char: CharacterSignature | None = None
        min_dist = float("inf")

        for char in self.characters:
            d_skin = float(np.linalg.norm(skin_lab[1:] - char.skin_lab[1:]))
            d_hair = (
                float(np.linalg.norm(hair_lab[1:] - char.hair_lab[1:]))
                if hair_lab is not None
                else 0.0
            )
            d_cloth = (
                float(np.linalg.norm(clothes_lab[1:] - char.clothes_lab[1:]))
                if clothes_lab is not None
                else 0.0
            )

            total_dist = 0.50 * d_skin + 0.30 * d_hair + 0.20 * d_cloth
            if total_dist < min_dist:
                min_dist = total_dist
                best_char = char

        if best_char is not None and min_dist < self.match_threshold:
            alpha = 1.0 / (best_char.n_observations + 1)
            best_char.skin_lab = (1.0 - alpha) * best_char.skin_lab + alpha * skin_lab
            if hair_lab is not None:
                best_char.hair_lab = (
                    1.0 - alpha
                ) * best_char.hair_lab + alpha * hair_lab
            if clothes_lab is not None:
                best_char.clothes_lab = (
                    1.0 - alpha
                ) * best_char.clothes_lab + alpha * clothes_lab
            best_char.n_observations += 1
            best_char.last_seen_page = page_num
            return best_char

        new_char = CharacterSignature(
            cid=len(self.characters) + 1,
            skin_lab=skin_lab.copy(),
            hair_lab=hair_lab.copy() if hair_lab is not None else skin_lab.copy(),
            clothes_lab=clothes_lab.copy()
            if clothes_lab is not None
            else skin_lab.copy(),
            n_observations=1,
            last_seen_page=page_num,
        )
        self.characters.append(new_char)
        return new_char

    @staticmethod
    def _blend_preserving_shading(
        lab_img: np.ndarray,
        box: tuple[int, int, int, int],
        target_lab: np.ndarray,
        strength: float,
    ) -> None:
        """Ajusta la crominancia media preservando gradientes y sombras originales."""
        x, y, w, h = box
        img_h, img_w = lab_img.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            return

        roi = lab_img[y1:y2, x1:x2]
        ch, cw = roi.shape[:2]

        # Máscara elíptica con desvanecimiento gaussiano
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (cw // 2, ch // 2),
            (int(cw * 0.45), int(ch * 0.50)),
            0,
            0,
            360,
            255,
            -1,
        )
        soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (15, 15), 0) * strength

        weights_sum = float(soft.sum())
        if weights_sum <= 1e-5:
            return

        norm_weights = soft / weights_sum
        for c in (1, 2):
            region_mean = float((roi[:, :, c] * norm_weights).sum())
            shift = (float(target_lab[c]) - region_mean) * soft
            roi[:, :, c] += shift

    def observe_and_apply(
        self, image_bgr: np.ndarray, page_num: int, strength: float = 0.55
    ) -> np.ndarray:
        """Detecta rostros, actualiza la memoria y retonifica suavemente."""
        faces = self.detect_faces(image_bgr)
        if not faces:
            return image_bgr

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        for face in faces:
            skin_lab, hair_lab, cloth_lab = self._extract_signatures(
                image_bgr, lab, face
            )
            if skin_lab is None:
                continue

            char = self._match_or_create(skin_lab, hair_lab, cloth_lab, page_num)

            # Re-tonificar piel
            self._blend_preserving_shading(
                lab, (face.x, face.y, face.w, face.h), char.skin_lab, strength
            )

            # Re-tonificar ropa (si hay firma válida)
            if cloth_lab is not None:
                cloth_box = (
                    max(0, face.x - int(face.w * 0.2)),
                    face.y + face.h,
                    int(face.w * 1.4),
                    int(face.h * 1.5),
                )
                self._blend_preserving_shading(
                    lab, cloth_box, char.clothes_lab, strength * 0.6
                )

        np.clip(lab, 0.0, 255.0, out=lab)
        return cast(np.ndarray, cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR))
