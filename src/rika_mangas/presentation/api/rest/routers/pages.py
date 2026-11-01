import asyncio
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from rika_mangas.core.paint_bucket import flood_recolor
from rika_mangas.presentation.api.rest.dependencies import (
    JOB_STORAGE,
    orchestrator,
    path_is_file,
    save_job_state,
)
from rika_mangas.presentation.api.rest.schemas.reprocess import (
    ReprocessPageRequest,
    ReprocessPageResponse,
)
from rika_mangas.presentation.api.rest.schemas.touchup_request import TouchupRequest

router = APIRouter(prefix="/jobs/{job_id}/pages", tags=["Pages"])


@router.post("/{page_num}/touchup")
async def touchup_page(
    job_id: str, page_num: int, req: TouchupRequest
) -> dict[str, Any]:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    job = JOB_STORAGE[job_id]
    file_path: Path | None = job["colorized"].get(page_num)
    if not file_path or not await path_is_file(file_path):
        raise HTTPException(
            status_code=404, detail="La pagina aun no ha sido coloreada"
        )

    img_bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(
            status_code=500, detail="Error al leer la pagina a modificar"
        )

    # Historial en RAM en búfer WebP (sin duplicar archivos en disco)
    success, encoded_bytes = cv2.imencode(
        ".webp", img_bgr, [cv2.IMWRITE_WEBP_QUALITY, 95]
    )
    if success:
        history_stack: list[bytes] = job.setdefault(f"history_{page_num}", [])
        history_stack.append(encoded_bytes.tobytes())
        if len(history_stack) > 15:
            history_stack.pop(0)

    recolored, _ = flood_recolor(
        img_bgr,
        req.x,
        req.y,
        req.hex_color,
        tolerance=req.tolerance,
        feather=req.feather,
    )

    # La versión original se mantiene en .png; las modificaciones se guardan en .webp
    target_path = file_path.with_suffix(".webp")
    cv2.imwrite(str(target_path), recolored, [cv2.IMWRITE_WEBP_QUALITY, 95])
    job["colorized"][page_num] = target_path
    save_job_state(job_id)

    return {"status": "ok", "can_undo": True}


@router.post("/{page_num}/undo")
async def undo_page(job_id: str, page_num: int) -> dict[str, Any]:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    job = JOB_STORAGE[job_id]
    file_path: Path | None = job["colorized"].get(page_num)
    history_stack: list[bytes] = job.get(f"history_{page_num}", [])

    if not file_path or not history_stack:
        raise HTTPException(
            status_code=400, detail="No hay acciones previas para deshacer"
        )

    import numpy as np

    last_step_bytes = history_stack.pop()
    nparr = np.frombuffer(last_step_bytes, np.uint8)
    restored_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if restored_bgr is None:
        raise HTTPException(
            status_code=500, detail="Error al decodificar estado previo de imagen"
        )

    # Si se desvanecieron todos los retoques, revertir al .png original si existe
    if len(history_stack) == 0:
        orig_png = file_path.with_suffix(".png")
        if orig_png.is_file():
            if file_path.suffix.lower() == ".webp" and file_path.is_file():
                file_path.unlink(missing_ok=True)
            job["colorized"][page_num] = orig_png
            save_job_state(job_id)
            return {"status": "ok", "can_undo": False}

    target_path = file_path.with_suffix(".webp")
    cv2.imwrite(str(target_path), restored_bgr, [cv2.IMWRITE_WEBP_QUALITY, 95])
    job["colorized"][page_num] = target_path
    save_job_state(job_id)

    return {"status": "ok", "can_undo": len(history_stack) > 0}


@router.get("/{page_num}/image")
async def get_page_image(
    job_id: str, page_num: int, original: bool = False
) -> FileResponse:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    job = JOB_STORAGE[job_id]
    path = (
        job["pages"][page_num]
        if original
        else job["colorized"].get(page_num, job["pages"][page_num])
    )

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Archivo de imagen no encontrado")

    media_type = "image/webp" if path.suffix.lower() == ".webp" else "image/png"
    return FileResponse(
        str(path),
        media_type=media_type,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.post("/{page_num}/reprocess", response_model=ReprocessPageResponse)
async def reprocess_page_style(
    job_id: str,
    page_num: int,
    req: ReprocessPageRequest,
) -> ReprocessPageResponse:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    job = JOB_STORAGE[job_id]

    if page_num < 0 or page_num >= len(job["pages"]):
        raise HTTPException(status_code=404, detail="Indice de pagina fuera de rango")

    page_path = job["pages"][page_num]
    output_dir = job["output_dir"]
    primary_ref = job["references"][0] if job["references"] else None

    try:
        out_file = await asyncio.to_thread(
            orchestrator.reprocess_page,
            page_num,
            page_path,
            output_dir,
            primary_ref,
            req.style,
            req.fidelity,
        )
        job["colorized"][page_num] = out_file
        job["style"] = req.style
        job["fidelity"] = req.fidelity
        save_job_state(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"La pagina aun no tiene inferencia base completada: {exc}",
        ) from exc

    return ReprocessPageResponse(
        status="ok",
        page_num=page_num,
        style=req.style,
        fidelity=req.fidelity,
        image_url=f"/api/v1/jobs/{job_id}/pages/{page_num}/image",
    )
