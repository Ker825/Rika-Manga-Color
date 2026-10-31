from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class RegionBBox(NamedTuple):
    x: int
    y: int
    w: int
    h: int


@dataclass(slots=True)
class CharacterSignature:
    cid: int
    skin_lab: np.ndarray
    hair_lab: np.ndarray
    clothes_lab: np.ndarray
    n_observations: int = 1
    last_seen_page: int = 0


@dataclass(frozen=True, slots=True)
class PageRegion:
    label_id: int
    area: int
    bbox: RegionBBox
    interior_point: tuple[int, int]
    mean_gray: float
    frac: float
