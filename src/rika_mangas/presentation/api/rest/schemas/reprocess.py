from typing import Literal

from pydantic import BaseModel, Field

PresetType = Literal[
    "neutral",
    "faithful_1to1",
    "shonen",
    "vibrant",
    "webtoon_modern",
    "shojo_pastel",
    "seinen_dark",
    "retro_90s",
    "cyberpunk_neon",
]


class ReprocessPageRequest(BaseModel):
    style: PresetType = Field(
        default="neutral",
        description="Identificador del preset cromático y tonal a aplicar.",
    )
    fidelity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fidelidad cromática respecto a la referencia (0.0 = min, 1.0 = max).",
    )


class ReprocessPageResponse(BaseModel):
    status: str
    page_num: int
    style: str
    fidelity: float
    image_url: str
