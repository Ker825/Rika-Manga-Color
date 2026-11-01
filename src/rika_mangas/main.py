from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rika_mangas.config import settings
from rika_mangas.domain.models.presets import STYLE_PRESETS
from rika_mangas.presentation.api.rest.dependencies import JOB_STORAGE
from rika_mangas.presentation.api.rest.routers import api_router


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Rika Mangas Engine",
    version="0.1.0",
    lifespan=lifespan,
)

settings.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
settings.STATIC_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def index_view(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/viewer/{job_id}", response_class=HTMLResponse)
async def viewer_view(request: Request, job_id: str) -> HTMLResponse:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")

    job = JOB_STORAGE[job_id]
    total_pages = len(job["pages"])
    current_style = job.get("style", "neutral")
    current_fidelity = (
        job.get("fidelity", 1.0) if job.get("fidelity") is not None else 1.0
    )

    return templates.TemplateResponse(
        request=request,
        name="viewer.html",
        context={
            "job_id": job_id,
            "total_pages": total_pages,
            "current_style": current_style,
            "current_fidelity": current_fidelity,
            "presets": STYLE_PRESETS,
        },
    )
