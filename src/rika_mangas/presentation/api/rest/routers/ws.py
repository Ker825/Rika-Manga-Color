from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from rika_mangas.presentation.api.rest.dependencies import WS_CONNECTIONS

router = APIRouter(prefix="/jobs", tags=["WebSockets"])


@router.websocket("/{job_id}/ws")
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
