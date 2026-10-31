import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def lineart_mask(gray: np.ndarray, low: int = 60, dilate: int = 0) -> np.ndarray:
    g = _to_gray(gray)
    mask = g <= low
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask


def bubble_mask(gray: np.ndarray, white_thr: int = 245, min_area_ratio: float = 0.002) -> np.ndarray:
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

        # Ignorar áreas que tocan el borde exterior
        if x <= 2 or y <= 2 or (x + ww) >= (w - 2) or (y + hh) >= (h - 2):
            continue

        # Relación de aspecto: los bocadillos suelen ser ovalados o redondeados, no tiras delgadas
        aspect = ww / float(hh)
        if aspect < 0.4 or aspect > 2.5:
            continue

        # Un bocadillo real de manga contiene texto oscuro en su interior
        roi = g[y : y + hh, x : x + ww]
        roi_mask = labels[y : y + hh, x : x + ww] == i
        text_pixels = np.count_nonzero((roi < 80) & roi_mask)
        ink_ratio = text_pixels / float(area)

        # Si no contiene suficiente tinta interior, es piel o fondo, no un diálogo
        if ink_ratio > 0.015:
            mask |= labels == i

    return mask


def gutter_mask(gray: np.ndarray, white_thr: int = 240, edge_band: float = 0.04) -> np.ndarray:
    g = _to_gray(gray)
    h, w = g.shape[:2]
    band_h = max(8, int(h * edge_band))
    band_w = max(8, int(w * edge_band))

    out = np.zeros_like(g, dtype=bool)
    out[:band_h] |= g[:band_h] >= white_thr
    out[-band_h:] |= g[-band_h:] >= white_thr
    out[:, :band_w] |= g[:, :band_w] >= white_thr
    out[:, -band_w:] |= g[:, -band_w:] >= white_thr
    return out


def combined_neutral_mask(
    gray: np.ndarray,
    *,
    line_low: int = 60,
    bubble_white: int = 235,
    gutter_white: int = 240,
    line_dilate: int = 1,
    blur: int = 5,
) -> np.ndarray:
    g = _to_gray(gray)
    line = lineart_mask(g, low=line_low, dilate=line_dilate)
    bubble = bubble_mask(g, white_thr=bubble_white)
    gutter = gutter_mask(g, white_thr=gutter_white)

    neutralize = line | bubble | gutter
    keep = (~neutralize).astype(np.float32)

    if blur > 0:
        keep = cv2.GaussianBlur(keep, (blur * 2 + 1, blur * 2 + 1), 0)
    np.clip(keep, 0.0, 1.0, out=keep)
    return keep
