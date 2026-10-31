import asyncio
import io
import shutil
import uuid
import zipfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from rika_mangas.application.gpu_worker import gpu_worker
from rika_mangas.application.services.orchestrator import ColorizationOrchestrator
from rika_mangas.config import settings
from rika_mangas.core.paint_bucket import flood_recolor
from rika_mangas.infrastructure.ml.manganinja import MangaNinjaStagedColorizer
from rika_mangas.infrastructure.ml.upscaler import RealESRGANCPUAccumulator
from rika_mangas.infrastructure.security import validate_and_decode_image
from rika_mangas.presentation.api.rest.schemas.touchup_request import TouchupRequest

router = APIRouter(prefix="/api/v1")

WS_CONNECTIONS: dict[str, list[WebSocket]] = {}
JOB_STORAGE: dict[str, dict[str, Any]] = {}
PAGES_FILE = File(...)
REFERENCES_FILE = File(default=[])

# Inyección de dependencias (DIP) cumpliendo IColorizerEngine e IUpscalerEngine
colorizer_engine = MangaNinjaStagedColorizer()
upscaler_engine = (
    RealESRGANCPUAccumulator(
        weights_path=settings.ESRGAN_WEIGHTS_PATH,
        tile_size=settings.ESRGAN_TILE_SIZE,
    )
    if settings.ESRGAN_WEIGHTS_PATH.is_file()
    else None
)

orchestrator = ColorizationOrchestrator(
    colorizer=colorizer_engine,
    upscaler=upscaler_engine,
)


async def notify_progress(job_id: str, data: dict[str, Any]) -> None:
    for ws in WS_CONNECTIONS.get(job_id, []):
        with suppress(Exception):
            await ws.send_json(data)


async def _path_is_file(path: Path) -> bool:
    return await asyncio.to_thread(path.is_file)


@router.post("/jobs")
async def create_job(
    pages: list[UploadFile] = PAGES_FILE,
    references: list[UploadFile] = REFERENCES_FILE,
) -> dict[str, Any]:
    if not pages:
        raise HTTPException(status_code=400, detail="No se suministraron paginas.")

    job_id = str(uuid.uuid4())
    job_input_dir = settings.UPLOAD_DIR / job_id
    job_output_dir = settings.OUTPUT_DIR / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    page_paths: list[Path] = []
    for idx, page_file in enumerate(pages):
        raw = await page_file.read()
        try:
            img = validate_and_decode_image(raw, settings.MAX_IMAGE_PIXELS, settings.MAX_DIMENSION)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Pagina {idx} invalida: {exc}") from exc
        dest = job_input_dir / f"page_{idx:04d}.png"
        cv2.imwrite(str(dest), img)
        page_paths.append(dest)

    ref_paths: list[Path] = []
    if references:
        for idx, ref_file in enumerate(references):
            raw = await ref_file.read()
            try:
                img = validate_and_decode_image(raw, settings.MAX_IMAGE_PIXELS, settings.MAX_DIMENSION)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Referencia {idx} invalida: {exc}") from exc
            dest = job_input_dir / f"ref_{idx:04d}.png"
            cv2.imwrite(str(dest), img)
            ref_paths.append(dest)

    JOB_STORAGE[job_id] = {
        "status": "ready",
        "pages": page_paths,
        "references": ref_paths,
        "colorized": {},
        "input_dir": job_input_dir,
        "output_dir": job_output_dir,
    }
    return {"job_id": job_id, "total_pages": len(page_paths)}


@router.post("/jobs/{job_id}/start")
async def start_job(job_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")

    job = JOB_STORAGE[job_id]
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="El trabajo ya esta en ejecucion")

    job["status"] = "processing"

    async def _run_job() -> None:
        total = len(job["pages"])
        for idx in range(total):
            await notify_progress(job_id, {"page": idx, "total": total, "status": "processing"})
            page_p = job["pages"][idx]
            refs = job["references"]
            out_dir = job["output_dir"]

            out_file = await gpu_worker.submit(
                orchestrator.process_page,
                job_id,
                idx,
                page_p,
                refs,
                out_dir,
            )
            job["colorized"][idx] = out_file
            await notify_progress(job_id, {"page": idx, "total": total, "status": "page_done"})

        job["status"] = "completed"
        await notify_progress(job_id, {"status": "done"})

    background_tasks.add_task(_run_job)
    return {"status": "started", "job_id": job_id}


@router.websocket("/jobs/{job_id}/ws")
async def job_websocket(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    if job_id not in WS_CONNECTIONS:
        WS_CONNECTIONS[job_id] = []
    WS_CONNECTIONS[job_id].append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        WS_CONNECTIONS[job_id].remove(websocket)


@router.post("/jobs/{job_id}/pages/{page_num}/touchup")
async def touchup_page_endpoint(job_id: str, page_num: int, req: TouchupRequest) -> dict[str, Any]:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    job = JOB_STORAGE[job_id]
    file_path: Path | None = job["colorized"].get(page_num)
    if not file_path or not await _path_is_file(file_path):
        raise HTTPException(status_code=404, detail="La pagina aun no ha sido coloreada")

    img_bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="Error al leer la pagina a modificar")

    prev_path = file_path.with_suffix(".prev.png")
    await asyncio.to_thread(shutil.copy2, file_path, prev_path)

    recolored, _ = flood_recolor(img_bgr, req.x, req.y, req.hex_color, tolerance=req.tolerance, feather=req.feather)
    cv2.imwrite(str(file_path), recolored)

    return {"status": "ok", "can_undo": True}


@router.post("/jobs/{job_id}/pages/{page_num}/undo")
async def undo_page_endpoint(job_id: str, page_num: int) -> dict[str, Any]:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    file_path: Path | None = JOB_STORAGE[job_id]["colorized"].get(page_num)
    if not file_path or not await _path_is_file(file_path):
        raise HTTPException(status_code=404, detail="Pagina no encontrada")

    prev_path = file_path.with_suffix(".prev.png")
    if not await _path_is_file(prev_path):
        raise HTTPException(status_code=400, detail="No hay acciones previas para deshacer")

    tmp_path = file_path.with_suffix(".swap.png")
    await asyncio.to_thread(file_path.replace, tmp_path)
    await asyncio.to_thread(prev_path.replace, file_path)
    await asyncio.to_thread(tmp_path.replace, prev_path)

    return {"status": "ok", "undone": True}


@router.get("/jobs/{job_id}/download")
async def download_archive(job_id: str) -> StreamingResponse:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")

    job = JOB_STORAGE[job_id]
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx in sorted(job["colorized"].keys()):
            p: Path = job["colorized"][idx]
            if await _path_is_file(p):
                zf.write(p, arcname=f"colored_{idx:04d}.png")

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=colorcomic_{job_id}.zip"},
    )


@router.get("/jobs/{job_id}/pages/{page_num}/image")
async def get_page_image(job_id: str, page_num: int, original: bool = False) -> FileResponse:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    job = JOB_STORAGE[job_id]

    if original:
        if page_num < 0 or page_num >= len(job["pages"]):
            raise HTTPException(status_code=404, detail="Pagina original no encontrada")
        path = job["pages"][page_num]
    else:
        path = job["colorized"].get(page_num)
        if not path or not path.is_file():
            path = job["pages"][page_num]

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Archivo de imagen no encontrado")

    return FileResponse(str(path), media_type="image/png")
