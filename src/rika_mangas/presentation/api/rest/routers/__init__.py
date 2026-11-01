from fastapi import APIRouter

from .jobs import router as jobs_router
from .pages import router as pages_router
from .presets import router as presets_router
from .ws import router as ws_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(presets_router)
api_router.include_router(jobs_router)
api_router.include_router(pages_router)
api_router.include_router(ws_router)

__all__ = ["api_router"]
