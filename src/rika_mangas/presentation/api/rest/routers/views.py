from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from rika_mangas.domain.models.presets import STYLE_PRESETS
from rika_mangas.presentation.api.rest.dependencies import JOB_STORAGE

web_router = APIRouter(include_in_schema=False)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "static" / "templates"))


@web_router.get("/viewer/{job_id}", response_class=HTMLResponse)
async def viewer_page(request: Request, job_id: str):
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")

    job = JOB_STORAGE[job_id]
    total_pages = len(job["pages"])
    current_style = job.get("style", "neutral")
    current_fidelity = (
        job.get("fidelity", 1.0) if job.get("fidelity") is not None else 1.0
    )

    return templates.TemplateResponse(
        request,
        name="viewer.html",
        context={
            "request": request,
            "job_id": job_id,
            "total_pages": total_pages,
            "current_style": current_style,
            "current_fidelity": current_fidelity,
            "presets": STYLE_PRESETS,
        },
    )
