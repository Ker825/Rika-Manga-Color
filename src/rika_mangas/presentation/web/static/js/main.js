// src/rika_mangas/presentation/web/static/js/main.js
import { state } from "./state.js";
import { fetchPresets, createJob, startJob, touchupPage, undoTouchup, reprocessPage } from "./api.js";
import { initJobWebSocket } from "./websocket.js";
import { setupDropzone } from "./dropzone.js";
import { renderPage, getCanvasCoordinates } from "./canvas.js";
import { showToast } from "./toast.js";

document.addEventListener("DOMContentLoaded", async () => {
    let activeWs = null;
    let reprocessDebounceTimer = null;

    // Elementos DOM
    const zonePages = document.getElementById("zone-pages");
    const zoneRefs = document.getElementById("zone-refs");
    const inputPages = document.getElementById("input-pages");
    const inputRefs = document.getElementById("input-refs");
    const labelPages = document.getElementById("label-pages");
    const labelRefs = document.getElementById("label-refs");
    const styleSelector = document.getElementById("style-selector");
    const btnUpload = document.getElementById("btn-upload");
    const fidelitySlider = document.getElementById("fidelity-slider");
    const fidelityVal = document.getElementById("fidelity-val");

    const cardExec = document.getElementById("card-execution");
    const cardTouchup = document.getElementById("card-touchup");
    const btnStart = document.getElementById("btn-start");
    const btnDownload = document.getElementById("btn-download");
    const btnReset = document.getElementById("btn-reset");

    const progressContainer = document.getElementById("progress-container");
    const progressTrack = document.getElementById("progress-track");
    const progressText = document.getElementById("progress-text");
    const pageSelector = document.getElementById("page-selector");
    const statusText = document.getElementById("engine-status-text");
    const statusDot = document.getElementById("status-dot");

    // Controles en vivo en tarjeta 3
    const tuneStyleSelector = document.getElementById("tune-style-selector");
    const tuneFidelitySlider = document.getElementById("tune-fidelity-slider");
    const tuneFidelityVal = document.getElementById("tune-fidelity-val");

    const canvas = document.getElementById("view-canvas");
    const colorPicker = document.getElementById("color-picker");
    const colorHex = document.getElementById("color-hex");
    const toleranceSlider = document.getElementById("tolerance-slider");
    const toleranceVal = document.getElementById("tolerance-val");
    const btnUndo = document.getElementById("btn-undo");
    const viewportRes = document.getElementById("viewport-resolution");
    const viewportCoords = document.getElementById("viewport-coords");
    const btnToggleSidebar = document.getElementById("btn-toggle-sidebar");
    const btnCompareOriginal = document.getElementById("btn-compare-original");
    const workspaceGrid = document.querySelector(".workspace-grid");

    // Sincronización visual del riel activo de los sliders
    function syncSliderFill(slider) {
        if (!slider) return;
        const min = parseFloat(slider.min) || 0;
        const max = parseFloat(slider.max) || 100;
        const val = parseFloat(slider.value) || 0;
        const pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
        slider.style.setProperty("--slider-pct", `${pct}%`);
    }

    [fidelitySlider, tuneFidelitySlider, toleranceSlider].forEach(syncSliderFill);

    if (fidelitySlider && fidelityVal) {
        fidelitySlider.addEventListener("input", (e) => {
            fidelityVal.textContent = parseFloat(e.target.value).toFixed(2);
            syncSliderFill(fidelitySlider);
        });
    }

    if (tuneFidelitySlider && tuneFidelityVal) {
        tuneFidelitySlider.addEventListener("input", (e) => {
            tuneFidelityVal.textContent = parseFloat(e.target.value).toFixed(2);
            syncSliderFill(tuneFidelitySlider);
            triggerDynamicReprocess();
        });
    }

    if (toleranceSlider && toleranceVal) {
        toleranceSlider.addEventListener("input", (e) => {
            toleranceVal.textContent = e.target.value;
            syncSliderFill(toleranceSlider);
        });
    }

    if (tuneStyleSelector) {
        tuneStyleSelector.addEventListener("change", () => {
            triggerDynamicReprocess();
        });
    }

    function triggerDynamicReprocess() {
        if (!state.currentJobId) return;
        clearTimeout(reprocessDebounceTimer);
        reprocessDebounceTimer = setTimeout(async () => {
            statusText.textContent = "Ajustando estilo...";
            statusDot.className = "status-dot working";
            try {
                await reprocessPage(
                    state.currentJobId,
                    state.activePageIndex,
                    tuneStyleSelector.value,
                    tuneFidelitySlider.value
                );
                renderPage(state.activePageIndex, false);
                statusText.textContent = "Listo";
                statusDot.className = "status-dot active";
            } catch (err) {
                statusText.textContent = "Error al ajustar";
                statusDot.className = "status-dot error";
                showToast("Fallo al ajustar el estilo en vivo.", "error");
                console.error(err);
            }
        }, 140);
    }

    if (btnToggleSidebar && workspaceGrid) {
        const toggleSidebar = () => {
            workspaceGrid.classList.toggle("sidebar-closed");
            window.dispatchEvent(new Event("resize"));
        };
        btnToggleSidebar.addEventListener("click", toggleSidebar);
        document.addEventListener("keydown", (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
                e.preventDefault();
                toggleSidebar();
            }
        });
    }

    function addUniqueFiles(targetArray, newFiles) {
        const existing = new Set(targetArray.map((f) => `${f.name}-${f.size}`));
        newFiles.forEach((file) => {
            if (file.type.startsWith("image/") && !existing.has(`${file.name}-${file.size}`)) {
                targetArray.push(file);
                existing.add(`${file.name}-${file.size}`);
            }
        });
    }

    function resetWorkspace() {
        if (activeWs) {
            activeWs.close();
            activeWs = null;
        }

        state.currentJobId = null;
        state.totalPages = 0;
        state.activePageIndex = 0;
        state.selectedPages = [];
        state.selectedRefs = [];
        state.showingOriginal = false;
        state.isProcessingTouchup = false;

        labelPages.textContent = "Hacer clic o arrastrar páginas (PNG / JPG)";
        labelRefs.textContent = "Hacer clic o arrastrar referencias";
        if (zonePages) zonePages.classList.remove("has-files");
        if (zoneRefs) zoneRefs.classList.remove("has-files");
        if (inputPages) inputPages.value = "";
        if (inputRefs) inputRefs.value = "";

        btnUpload.hidden = false;
        btnUpload.disabled = false;
        btnUpload.textContent = "Cargar Lote";

        btnStart.disabled = false;
        cardExec.hidden = true;
        cardTouchup.hidden = true;
        btnDownload.hidden = true;
        if (btnReset) btnReset.hidden = true;
        if (btnCompareOriginal) btnCompareOriginal.hidden = true;

        progressTrack.style.width = "0%";
        progressContainer.setAttribute("aria-valuenow", "0");
        progressText.textContent = "En cola...";
        statusText.textContent = "Listo";
        statusDot.className = "status-dot";

        canvas.width = 0;
        canvas.height = 0;
        canvas.classList.remove("loaded", "is-loading");
        viewportRes.textContent = "0 × 0 px";
        viewportCoords.textContent = "X: 0 | Y: 0";
    }

    // 1. Cargar Presets de forma segura (sin innerHTML)
    try {
        const data = await fetchPresets();
        if (data.presets && data.presets.length > 0) {
            const createOptions = () =>
                data.presets.map((p) => {
                    const opt = new Option(p.label, p.key);
                    if (p.description) opt.title = p.description;
                    return opt;
                });
            if (styleSelector) styleSelector.replaceChildren(...createOptions());
            if (tuneStyleSelector) tuneStyleSelector.replaceChildren(...createOptions());
        }
    } catch {
        const fallback = () => [new Option("Neutral (Por defecto)", "neutral")];
        if (styleSelector) styleSelector.replaceChildren(...fallback());
        if (tuneStyleSelector) tuneStyleSelector.replaceChildren(...fallback());
    }

    // 2. Dropzones
    setupDropzone(zonePages, inputPages, (files) => {
        addUniqueFiles(state.selectedPages, files);
        labelPages.textContent = `${state.selectedPages.length} página(s) lista(s)`;
    });

    setupDropzone(zoneRefs, inputRefs, (files) => {
        addUniqueFiles(state.selectedRefs, files);
        labelRefs.textContent = state.selectedRefs.length
            ? `${state.selectedRefs.length} ref(s) lista(s)`
            : "Hacer clic o arrastrar referencias";
    });

    colorPicker.addEventListener("input", (e) => {
        colorHex.textContent = e.target.value.toUpperCase();
    });

    // 3. Subir Lote
    btnUpload.addEventListener("click", async () => {
        if (!state.selectedPages.length) {
            showToast("Debes seleccionar al menos una página en blanco y negro.", "error");
            return;
        }

        const selectedStyle = styleSelector ? styleSelector.value : "neutral";
        btnUpload.disabled = true;
        btnUpload.textContent = "Cargando lote...";

        try {
            const fidelity =
                state.selectedRefs.length > 0 && fidelitySlider ? parseFloat(fidelitySlider.value) : null;

            const data = await createJob(state.selectedPages, state.selectedRefs, selectedStyle, fidelity);
            state.currentJobId = data.job_id;
            state.totalPages = data.total_pages;

            statusText.textContent = `Lote Listo (${state.totalPages} págs)`;
            statusDot.className = "status-dot active";
            cardExec.hidden = false;
            btnUpload.hidden = true;

            const pageOptions = Array.from(
                { length: state.totalPages },
                (_, i) => new Option(`Página ${i + 1}`, `${i}`)
            );
            pageSelector.replaceChildren(...pageOptions);

            if (tuneStyleSelector) tuneStyleSelector.value = selectedStyle;
            if (tuneFidelitySlider && fidelity !== null) {
                tuneFidelitySlider.value = fidelity;
                tuneFidelityVal.textContent = Number(fidelity).toFixed(2);
                syncSliderFill(tuneFidelitySlider);
            }

            renderPage(0, true);
            showToast(`Lote preparado: ${state.totalPages} páginas.`, "success");
        } catch (err) {
            showToast(err.message || "Error al procesar el lote", "error");
            btnUpload.disabled = false;
            btnUpload.textContent = "Cargar Lote";
        }
    });

    // 4. Inferencia y WebSocket Seguro
    btnStart.addEventListener("click", async () => {
        btnStart.disabled = true;
        statusText.textContent = "Inferencia GPU";
        statusDot.className = "status-dot working";

        activeWs = initJobWebSocket(state.currentJobId, {
            processing: (data) => {
                progressText.textContent = `Página ${data.page + 1} de ${data.total}`;
                const pct = (data.page / data.total) * 100;
                progressTrack.style.width = `${pct}%`;
                progressContainer.setAttribute("aria-valuenow", `${Math.round(pct)}`);
            },
            page_done: (data) => {
                const pct = ((data.page + 1) / data.total) * 100;
                progressTrack.style.width = `${pct}%`;
                progressContainer.setAttribute("aria-valuenow", `${Math.round(pct)}`);
                if (data.page === state.activePageIndex) {
                    renderPage(state.activePageIndex, false);
                }
            },
            done: () => {
                progressText.textContent = "Completado";
                progressTrack.style.width = "100%";
                progressContainer.setAttribute("aria-valuenow", "100");
                statusText.textContent = "Completado";
                statusDot.className = "status-dot active";
                btnDownload.href = `/api/v1/jobs/${state.currentJobId}/download`;
                btnDownload.hidden = false;
                if (btnReset) btnReset.hidden = false;
                cardTouchup.hidden = false;
                renderPage(state.activePageIndex, false);
                showToast("Inferencia finalizada con éxito.", "success");
            },
            error: () => {
                statusText.textContent = "Canal interrumpido";
                statusDot.className = "status-dot error";
                showToast("Se interrumpió el WebSocket de monitoreo.", "error");
            }
        });

        try {
            await activeWs.ready;
            await startJob(state.currentJobId);
        } catch {
            statusText.textContent = "Error al iniciar";
            statusDot.className = "status-dot error";
            btnStart.disabled = false;
            showToast("No se pudo iniciar la inferencia en backend.", "error");
        }
    });

    pageSelector.addEventListener("change", (e) => {
        state.activePageIndex = parseInt(e.target.value, 10);
        renderPage(state.activePageIndex, false);
    });

    // 5. Retoque Interactivo (Paint Bucket con bloqueo de clics concurrentes)
    canvas.addEventListener("click", async (e) => {
        if (cardTouchup.hidden || state.isProcessingTouchup || state.showingOriginal) return;

        const coords = getCanvasCoordinates(e);
        statusText.textContent = `Retocando (${coords.x}, ${coords.y})...`;
        statusDot.className = "status-dot working";
        state.isProcessingTouchup = true;

        try {
            const resp = await touchupPage(state.currentJobId, state.activePageIndex, {
                x: coords.x,
                y: coords.y,
                hex_color: colorPicker.value,
                tolerance: parseInt(toleranceSlider.value, 10),
                feather: 3
            });

            if (resp.ok) {
                renderPage(state.activePageIndex, false);
                statusText.textContent = "Color aplicado";
                statusDot.className = "status-dot active";
            } else {
                statusText.textContent = "Fallo al retocar";
                statusDot.className = "status-dot";
                showToast("No se pudo aplicar el retoque.", "error");
            }
        } catch {
            statusText.textContent = "Error al retocar";
            statusDot.className = "status-dot error";
            showToast("Error de conexión al retocar punto.", "error");
        } finally {
            state.isProcessingTouchup = false;
        }
    });

    btnUndo.addEventListener("click", async () => {
        if (state.isProcessingTouchup) return;
        state.isProcessingTouchup = true;
        statusText.textContent = "Deshaciendo...";
        statusDot.className = "status-dot working";

        try {
            const resp = await undoTouchup(state.currentJobId, state.activePageIndex);
            if (resp.ok) {
                renderPage(state.activePageIndex, false);
                statusText.textContent = "Paso deshecho";
                statusDot.className = "status-dot active";
            } else {
                statusText.textContent = "Sin historial";
                statusDot.className = "status-dot";
                showToast("No hay más estados previos para esta página.", "info");
            }
        } catch {
            statusText.textContent = "Error al deshacer";
            statusDot.className = "status-dot error";
            showToast("Error al procesar deshacer retoque.", "error");
        } finally {
            state.isProcessingTouchup = false;
        }
    });

    // 6. Comparador A/B (Original B&N vs Color)
    if (btnCompareOriginal) {
        btnCompareOriginal.addEventListener("click", () => {
            if (!state.currentJobId) return;
            renderPage(state.activePageIndex, !state.showingOriginal);
        });
    }

    document.addEventListener("keydown", (e) => {
        if (e.code === "Space" && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
            e.preventDefault();
            if (state.currentJobId && btnCompareOriginal && !btnCompareOriginal.hidden) {
                renderPage(state.activePageIndex, !state.showingOriginal);
            }
        }
    });

    // 7. Reset
    if (btnReset) {
        btnReset.addEventListener("click", resetWorkspace);
    }
});
