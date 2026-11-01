import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def bubble_mask(
    gray: np.ndarray, white_thr: int = 245, min_area_ratio: float = 0.002
) -> np.ndarray:
    g = _to_gray(gray)
    h, w = g.shape[:2]
    page_area = h * w
    min_area = max(256, int(page_area * min_area_ratio))

    _, white = cv2.threshold(g, white_thr, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n <= 1:
        return np.zeros_like(g, dtype=bool)

    mask = np.zeros_like(g, dtype=bool)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > (page_area * 0.25):
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]

        if x <= 2 or y <= 2 or (x + ww) >= (w - 2) or (y + hh) >= (h - 2):
            continue

        aspect = ww / float(hh)
        if aspect < 0.4 or aspect > 2.5:
            continue

        roi = g[y : y + hh, x : x + ww]
        roi_mask = labels[y : y + hh, x : x + ww] == i
        text_pixels = np.count_nonzero((roi < 80) & roi_mask)
        ink_ratio = text_pixels / float(area)

        if ink_ratio > 0.015:
            mask[y : y + hh, x : x + ww] |= roi_mask

    return mask


def gutter_mask(
    gray: np.ndarray, white_thr: int = 245, edge_band: float = 0.04
) -> np.ndarray:
    """Detecta únicamente márgenes exteriores continuos de papel blanco puro.

    Evita recortar viñetas a sangre (full-bleed) o tramas (screentones) que tocan
    los bordes de la página cerrando primero cualquier tinta/trama y exigiendo que
    las filas o columnas del margen sean >= 99.5% blancas desde el borde exterior.
    """
    g = _to_gray(gray)
    h, w = g.shape[:2]
    max_h = max(4, int(h * edge_band))
    max_w = max(4, int(w * edge_band))

    # Dilatar cualquier trazo o punto de trama (< white_thr) para no confundir
    # el fondo blanco entre puntos de screentone con un margen de página.
    ink = (g < white_thr).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    ink_closed = cv2.dilate(ink, kernel)
    pure_white = ink_closed == 0

    out = np.zeros((h, w), dtype=bool)
    min_white_ratio = 0.995
    min_margin_px = 4

    # Borde superior
    top = 0
    while top < max_h and float(np.mean(pure_white[top, :])) >= min_white_ratio:
        top += 1
    if top >= min_margin_px:
        out[:top, :] = True

    # Borde inferior
    bottom = 0
    while (
        bottom < max_h
        and float(np.mean(pure_white[h - 1 - bottom, :])) >= min_white_ratio
    ):
        bottom += 1
    if bottom >= min_margin_px:
        out[h - bottom :, :] = True

    # Borde izquierdo
    left = 0
    while left < max_w and float(np.mean(pure_white[:, left])) >= min_white_ratio:
        left += 1
    if left >= min_margin_px:
        out[:, :left] = True

    # Borde derecho
    right = 0
    while (
        right < max_w
        and float(np.mean(pure_white[:, w - 1 - right])) >= min_white_ratio
    ):
        right += 1
    if right >= min_margin_px:
        out[:, w - right :] = True

    return out


def combined_neutral_mask(
    gray: np.ndarray,
    *,
    bubble_white: int = 235,
    gutter_white: int = 245,
    blur: int = 3,
) -> np.ndarray:
    """Aísla exclusivamente globos de diálogo y márgenes reales para forzar blanco neutro."""
    g = _to_gray(gray)
    bubble = bubble_mask(g, white_thr=bubble_white)
    gutter = gutter_mask(g, white_thr=gutter_white)

    neutralize = bubble | gutter
    if not np.any(neutralize):
        return np.ones_like(g, dtype=np.float32)

    keep = (~neutralize).astype(np.float32)
    if blur > 0:
        keep = cv2.GaussianBlur(keep, (blur * 2 + 1, blur * 2 + 1), 0)
    return np.clip(keep, 0.0, 1.0)
