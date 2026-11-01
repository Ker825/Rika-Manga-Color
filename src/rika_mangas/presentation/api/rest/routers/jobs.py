import asyncio
import io
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Annotated, Any, Literal

import cv2
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from rika_mangas.application.gpu_worker import gpu_worker
from rika_mangas.config import settings
from rika_mangas.domain.models.presets import STYLE_PRESETS
from rika_mangas.infrastructure.security import validate_and_decode_image
from rika_mangas.presentation.api.rest.dependencies import (
    JOB_STORAGE,
    WS_CONNECTIONS,
    notify_progress,
    orchestrator,
    path_is_file,
    save_job_state,
)

StyleKey = Literal[
    "neutral",
    "shonen",
    "vibrant",
    "summer_glow",
    "webtoon_modern",
    "shojo_pastel",
    "seinen_dark",
    "retro_90s",
    "cyberpunk_neon",
    "faithful_1to1",
]

router = APIRouter(prefix="/jobs", tags=["Jobs"])

PAGES_FILE = File(...)
REFERENCES_FILE = File(default=[])


@router.post("")
async def create_job(
    pages: list[UploadFile] = PAGES_FILE,
    references: list[UploadFile] = REFERENCES_FILE,
    style: Annotated[StyleKey, Form()] = "neutral",
    fidelity: Annotated[
        float | None,
        Form(
            ge=0.0,
            le=1.0,
            description="Intensidad de copia cromática de la referencia (0.0 = nulo, 1.0 = total).",
        ),
    ] = None,
) -> dict[str, Any]:
    if not pages:
        raise HTTPException(status_code=400, detail="No se suministraron paginas.")

    if fidelity is not None:
        fidelity = max(0.0, min(1.0, fidelity))

    if style not in STYLE_PRESETS:
        style = "neutral"

    job_id = str(uuid.uuid4())
    job_input_dir = settings.UPLOAD_DIR / job_id
    job_output_dir = settings.OUTPUT_DIR / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    page_paths: list[Path] = []
    for idx, page_file in enumerate(pages):
        raw = await page_file.read()
        try:
            img = validate_and_decode_image(
                raw, settings.MAX_IMAGE_PIXELS, settings.MAX_DIMENSION
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"Pagina {idx} invalida: {exc}"
            ) from exc
        dest = job_input_dir / f"page_{idx:04d}.png"
        cv2.imwrite(str(dest), img)
        page_paths.append(dest)

    ref_paths: list[Path] = []
    if references:
        for idx, ref_file in enumerate(references):
            raw = await ref_file.read()
            try:
                img = validate_and_decode_image(
                    raw, settings.MAX_IMAGE_PIXELS, settings.MAX_DIMENSION
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail=f"Referencia {idx} invalida: {exc}"
                ) from exc
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
        "style": style,
        "fidelity": fidelity,
    }
    save_job_state(job_id)

    return {
        "job_id": job_id,
        "total_pages": len(page_paths),
        "style": style,
        "fidelity": fidelity,
    }


@router.post("/{job_id}/start")
async def start_job(job_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")

    job = JOB_STORAGE[job_id]
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="El trabajo ya esta en ejecucion")

    job["status"] = "processing"
    save_job_state(job_id)

    async def _run_job() -> None:
        total = len(job["pages"])
        for idx in range(total):
            await notify_progress(
                job_id, {"page": idx, "total": total, "status": "processing"}
            )

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
                job.get("style", "neutral"),
                job.get("fidelity"),
            )
            job["colorized"][idx] = out_file
            save_job_state(job_id)

            await notify_progress(
                job_id, {"page": idx, "total": total, "status": "page_done"}
            )

        job["status"] = "completed"
        save_job_state(job_id)
        await notify_progress(job_id, {"status": "done"})

    background_tasks.add_task(_run_job)
    return {"status": "started", "job_id": job_id}


@router.get("/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    job = JOB_STORAGE[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "total_pages": len(job["pages"]),
        "completed_pages": len(job["colorized"]),
        "style": job.get("style", "neutral"),
        "fidelity": job.get("fidelity"),
    }


@router.get("/{job_id}/download")
async def download_archive(job_id: str) -> StreamingResponse:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")

    job = JOB_STORAGE[job_id]
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx in sorted(job["colorized"].keys()):
            p: Path = job["colorized"][idx]
            if await path_is_file(p):
                zf.write(p, arcname=f"colored_{idx:04d}.png")

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=colorcomic_{job_id}.zip"
        },
    )


@router.delete("/{job_id}")
async def delete_job(job_id: str) -> dict[str, Any]:
    if job_id not in JOB_STORAGE:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")

    job = JOB_STORAGE.pop(job_id)
    if job_id in WS_CONNECTIONS:
        WS_CONNECTIONS.pop(job_id)

    await asyncio.to_thread(shutil.rmtree, job["input_dir"], ignore_errors=True)
    await asyncio.to_thread(shutil.rmtree, job["output_dir"], ignore_errors=True)

    return {"status": "deleted"}
