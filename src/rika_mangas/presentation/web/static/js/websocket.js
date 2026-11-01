// src/rika_mangas/presentation/web/static/js/websocket.js

/**
 * Inicializa la conexión WebSocket para monitorear el progreso del job.
 * @param {string} jobId
 * @param {Record<string, Function>} callbacks
 * @returns {WebSocket & { ready: Promise<void> }}
 */
export function initJobWebSocket(jobId, callbacks) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/api/v1/jobs/${jobId}/ws`);

    let resolveReady;
    ws.ready = new Promise((resolve) => {
        resolveReady = resolve;
        setTimeout(resolve, 2500); // Timeout defensivo
    });

    ws.onopen = () => {
        if (resolveReady) resolveReady();
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (callbacks[data.status]) {
                callbacks[data.status](data);
            }
        } catch (err) {
            console.error("WebSocket parse error:", err);
        }
    };

    ws.onerror = (err) => {
        if (resolveReady) resolveReady();
        if (callbacks.error) {
            callbacks.error(err);
        }
    };

    ws.onclose = () => {
        if (callbacks.close) {
            callbacks.close();
        }
    };

    return ws;
}
