import asyncio
import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from rika_mangas.application.services.orchestrator import ColorizationOrchestrator
from rika_mangas.config import settings
from rika_mangas.infrastructure.ml.manganinja import MangaNinjaStagedColorizer
from rika_mangas.infrastructure.ml.upscaler import RealESRGANCPUAccumulator

WS_CONNECTIONS: dict[str, list[WebSocket]] = {}
JOB_STORAGE: dict[str, dict[str, Any]] = {}

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

logger = logging.getLogger(__name__)


async def notify_progress(job_id: str, data: dict[str, Any]) -> None:
    for ws in list(WS_CONNECTIONS.get(job_id, [])):
        with suppress(Exception):
            await ws.send_json(data)


async def path_is_file(path: Path) -> bool:
    """Comprueba la existencia del archivo de forma no bloqueante."""
    return await asyncio.to_thread(path.is_file)


def save_job_state(job_id: str) -> None:
    job = JOB_STORAGE.get(job_id)
    if not job:
        return
    state_file = job["output_dir"] / "state.json"
    data = {
        "status": job["status"],
        "pages": [str(p) for p in job["pages"]],
        "references": [str(p) for p in job["references"]],
        "colorized": {str(k): str(v) for k, v in job["colorized"].items()},
        "input_dir": str(job["input_dir"]),
        "output_dir": str(job["output_dir"]),
        "style": job.get("style", "neutral"),
        "fidelity": job.get("fidelity"),
    }
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def restore_jobs_from_disk() -> None:
    if not settings.OUTPUT_DIR.is_dir():
        return
    for job_dir in settings.OUTPUT_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        state_file = job_dir / "state.json"
        if state_file.is_file():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                JOB_STORAGE[job_dir.name] = {
                    "status": data["status"],
                    "pages": [Path(p) for p in data["pages"]],
                    "references": [Path(p) for p in data["references"]],
                    "colorized": {
                        int(k): Path(v) for k, v in data["colorized"].items()
                    },
                    "input_dir": Path(data["input_dir"]),
                    "output_dir": Path(data["output_dir"]),
                    "style": data.get("style", "neutral"),
                    "fidelity": data.get("fidelity"),
                }
            except Exception:
                logger.warning(
                    "Fallo al restaurar estado de %s", state_file, exc_info=True
                )
                continue


restore_jobs_from_disk()
