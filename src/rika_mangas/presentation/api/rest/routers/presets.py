from typing import Any

from fastapi import APIRouter

from rika_mangas.domain.models.presets import STYLE_PRESETS

router = APIRouter(prefix="/presets", tags=["Presets"])


@router.get("")
async def list_presets() -> dict[str, Any]:
    return {
        "presets": [
            {"key": k, "label": v.label, "description": v.description}
            for k, v in STYLE_PRESETS.items()
        ]
    }
