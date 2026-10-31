import cv2
import numpy as np

from rika_mangas.domain.models.entities import PageRegion, RegionBBox

_SEG_MAX_EDGE = 1400


class SegmentationResult:
    def __init__(self, regions: list[PageRegion], labels: np.ndarray, scale: float) -> None:
        self.regions = regions
        self.labels = labels
        self.scale = scale

    def interior_points(self, region: PageRegion, spacing: int = 110, max_points: int = 12) -> list[tuple[int, int]]:
        mask = (self.labels == region.label_id).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        step = max(24, int(spacing * self.scale))
        points: list[tuple[int, int]] = []
        h, w = dist.shape
        for yy in range(step // 2, h, step):
            for xx in range(step // 2, w, step):
                if dist[yy, xx] > 6:
                    points.append((int(xx / self.scale), int(yy / self.scale)))
                    if len(points) >= max_points:
                        return points
        return points


def segment_regions(
    gray: np.ndarray,
    *,
    line_low: int = 100,
    gap_close: int = 4,
    min_area_frac: float = 0.0008,
    max_regions: int = 28,
) -> SegmentationResult:
    gray_img = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if gray.ndim == 3 else gray

    h, w = gray_img.shape[:2]
    scale = 1.0
    g = gray_img
    if max(h, w) > _SEG_MAX_EDGE:
        scale = _SEG_MAX_EDGE / max(h, w)
        g = cv2.resize(gray_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gh, gw = g.shape[:2]
    ink: np.ndarray = (g < line_low).astype(np.uint8)
    if gap_close > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_close + 1, 2 * gap_close + 1))
        ink = cv2.dilate(ink, kernel)

    fillable = (ink == 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(fillable, connectivity=4)

    page_area = gh * gw
    min_area = max(64, int(page_area * min_area_frac))
    inv = 1.0 / scale

    regions: list[PageRegion] = []
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    for i in order:
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or len(regions) >= max_regions:
            break
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])

        crop_mask = (labels[y : y + hh, x : x + ww] == i).astype(np.uint8)
        dist = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 3)
        iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
        mean_gray = float(g[y : y + hh, x : x + ww][crop_mask.astype(bool)].mean())

        bbox = RegionBBox(x=int(x * inv), y=int(y * inv), w=int(ww * inv), h=int(hh * inv))
        regions.append(
            PageRegion(
                label_id=int(i),
                area=int(area * inv * inv),
                bbox=bbox,
                interior_point=(int((x + ix) * inv), int((y + iy) * inv)),
                mean_gray=mean_gray,
                frac=area / page_area,
            )
        )

    return SegmentationResult(regions, labels, scale)
