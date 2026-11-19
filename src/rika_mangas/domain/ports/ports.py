from typing import Protocol

import numpy as np

type HintPoint = (
    tuple[float, float, tuple[int, int, int]] | tuple[float, float, float, float]
)


class IColorizerEngine(Protocol):
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def colorize_page(
        self,
        image_bgr: np.ndarray,
        *,
        reference_images: list[np.ndarray] | None = None,
        size: int = 512,
        steps: int = 30,
        denoise_sigma: int = 0,
        tiled: bool = False,
        tile_size: int = 512,
        tile_overlap: int = 96,
        hint_points: list[HintPoint] | None = None,
    ) -> np.ndarray: ...


class IUpscalerEngine(Protocol):
    def upscale(self, image_bgr: np.ndarray) -> np.ndarray: ...
    def unload(self) -> None: ...
