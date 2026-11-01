// src/rika_mangas/presentation/web/static/js/canvas.js
import { state } from "./state.js";

const canvas = document.getElementById("view-canvas");
const ctx = canvas.getContext("2d");
const viewportRes = document.getElementById("viewport-resolution");
const viewportCoords = document.getElementById("viewport-coords");
const btnCompareOriginal = document.getElementById("btn-compare-original");
const activeImg = new Image();

activeImg.crossOrigin = "anonymous";

activeImg.onload = () => {
    canvas.width = activeImg.naturalWidth;
    canvas.height = activeImg.naturalHeight;
    const modeLabel = state.showingOriginal ? "Original B&N" : `Pág ${state.activePageIndex + 1}`;
    viewportRes.textContent = `${activeImg.naturalWidth} × ${activeImg.naturalHeight} px · ${modeLabel}`;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(activeImg, 0, 0);

    canvas.classList.remove("is-loading");
    canvas.classList.add("loaded");

    if (btnCompareOriginal) {
        btnCompareOriginal.hidden = false;
    }
};

activeImg.onerror = () => {
    canvas.classList.remove("is-loading");
    viewportRes.textContent = "Error al cargar lienzo";
};

export function renderPage(pageIdx, isOriginal = false) {
    if (!state.currentJobId) return;
    state.showingOriginal = isOriginal;
    canvas.classList.remove("loaded");
    canvas.classList.add("is-loading");

    if (btnCompareOriginal) {
        btnCompareOriginal.classList.toggle("is-active", isOriginal);
        btnCompareOriginal.textContent = isOriginal ? "Ver Color" : "Comparar B&N";
    }

    activeImg.src = `/api/v1/jobs/${state.currentJobId}/pages/${pageIdx}/image?original=${isOriginal}&t=${Date.now()}`;
}

export function getCanvasCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: Math.round((e.clientX - rect.left) * scaleX),
        y: Math.round((e.clientY - rect.top) * scaleY)
    };
}

let ticking = false;
canvas.addEventListener("mousemove", (e) => {
    if (!ticking) {
        requestAnimationFrame(() => {
            if (canvas.width > 0 && canvas.height > 0) {
                const { x, y } = getCanvasCoordinates(e);
                viewportCoords.textContent = `X: ${x} | Y: ${y}`;
            }
            ticking = false;
        });
        ticking = true;
    }
});
