from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rika_mangas.config import settings
from rika_mangas.presentation.api.rest.routers.router import router as api_router


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
