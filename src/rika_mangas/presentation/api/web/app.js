document.addEventListener("DOMContentLoaded", () => {
    let currentJobId = null;
    let totalPages = 0;
    let activePageIndex = 0;
    let wsConnection = null;

    let selectedPages = [];
    let selectedRefs = [];

    const zonePages = document.getElementById("zone-pages");
    const zoneRefs = document.getElementById("zone-refs");
    const inputPages = document.getElementById("input-pages");
    const inputRefs = document.getElementById("input-refs");
    const labelPages = document.getElementById("label-pages");
    const labelRefs = document.getElementById("label-refs");
    const btnUpload = document.getElementById("btn-upload");

    const cardExec = document.getElementById("card-execution");
    const cardTouchup = document.getElementById("card-touchup");
    const btnStart = document.getElementById("btn-start");
    const btnDownload = document.getElementById("btn-download");
    const progressTrack = document.getElementById("progress-track");
    const progressText = document.getElementById("progress-text");
    const pageSelector = document.getElementById("page-selector");
    const statusText = document.getElementById("engine-status-text");
    const statusDot = document.getElementById("status-dot");
    const canvas = document.getElementById("view-canvas");
    const ctx = canvas.getContext("2d");
    const colorPicker = document.getElementById("color-picker");
    const colorHex = document.getElementById("color-hex");
    const toleranceSlider = document.getElementById("tolerance-slider");
    const toleranceVal = document.getElementById("tolerance-val");
    const btnUndo = document.getElementById("btn-undo");
    const viewportRes = document.getElementById("viewport-resolution");
    const viewportCoords = document.getElementById("viewport-coords");

    // ── Enlazar Zonas de Carga (Click & Drag-and-Drop) ──
    function setupDropzone(zone, input, onFilesSelected) {
        zone.addEventListener("click", () => input.click());

        zone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                input.click();
            }
        });

        ["dragenter", "dragover"].forEach((evtName) => {
            zone.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((evtName) => {
            zone.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.remove("dragover");
            });
        });

        zone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length) {
                onFilesSelected(Array.from(dt.files));
            }
        });

        input.addEventListener("change", () => {
            if (input.files && input.files.length) {
                onFilesSelected(Array.from(input.files));
            }
        });
    }

    setupDropzone(zonePages, inputPages, (files) => {
        const valid = files.filter((f) => f.type.startsWith("image/"));
        if (!valid.length) {
            alert("Selecciona archivos de imagen válidos (PNG o JPEG).");
            return;
        }
        selectedPages = valid;
        labelPages.textContent = `${selectedPages.length} página(s) seleccionada(s)`;
    });

    setupDropzone(zoneRefs, inputRefs, (files) => {
        const valid = files.filter((f) => f.type.startsWith("image/"));
        selectedRefs = valid;
        labelRefs.textContent = selectedRefs.length
            ? `${selectedRefs.length} referencia(s) lista(s)`
            : "Hacer clic o arrastrar referencias";
    });

    colorPicker.addEventListener("input", (e) => {
        colorHex.textContent = e.target.value.toUpperCase();
    });

    toleranceSlider.addEventListener("input", (e) => {
        toleranceVal.textContent = e.target.value;
    });

    // ── 1. Cargar Lote ──
    btnUpload.addEventListener("click", async () => {
        if (!selectedPages.length) {
            alert("Debes seleccionar al menos una página en blanco y negro.");
            return;
        }

        const formData = new FormData();
        selectedPages.forEach((file) => formData.append("pages", file));
        selectedRefs.forEach((file) => formData.append("references", file));

        btnUpload.disabled = true;
        btnUpload.textContent = "Cargando lote...";

        try {
            const resp = await fetch("/api/v1/jobs", { method: "POST", body: formData });
            if (!resp.ok) {
                const errDetail = await resp.json().catch(() => ({}));
                throw new Error(errDetail.detail || "Fallo en la validación del servidor.");
            }
            const data = await resp.json();

            currentJobId = data.job_id;
            totalPages = data.total_pages;

            statusText.textContent = `Lote Listo (${totalPages} págs)`;
            statusDot.className = "status-dot active";
            cardExec.style.display = "flex";
            btnUpload.style.display = "none";

            populatePageSelector(totalPages);
            loadPageToCanvas(0, true);
        } catch (err) {
            alert(err.message);
            btnUpload.disabled = false;
            btnUpload.textContent = "Cargar Lote";
        }
    });

    // ── 2. Inferencia y WebSockets ──
    btnStart.addEventListener("click", async () => {
        btnStart.disabled = true;
        statusText.textContent = "Inferencia GPU en curso";
        statusDot.className = "status-dot working";

        connectWebSocket(currentJobId);
        await fetch(`/api/v1/jobs/${currentJobId}/start`, { method: "POST" });
    });

    function connectWebSocket(jobId) {
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        wsConnection = new WebSocket(`${proto}//${location.host}/api/v1/jobs/${jobId}/ws`);

        wsConnection.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.status === "processing") {
                progressText.textContent = `Página ${data.page + 1} de ${data.total}`;
                progressTrack.style.width = `${(data.page / data.total) * 100}%`;
            } else if (data.status === "page_done") {
                progressTrack.style.width = `${((data.page + 1) / data.total) * 100}%`;
                if (data.page === activePageIndex) {
                    loadPageToCanvas(activePageIndex, false);
                }
            } else if (data.status === "done") {
                progressText.textContent = "Proceso finalizado";
                progressTrack.style.width = "100%";
                statusText.textContent = "Completado";
                statusDot.className = "status-dot active";
                btnDownload.href = `/api/v1/jobs/${jobId}/download`;
                btnDownload.style.display = "flex";
                cardTouchup.style.display = "flex";
                loadPageToCanvas(activePageIndex, false);
            }
        };
    }

    function populatePageSelector(count) {
        pageSelector.innerHTML = "";
        for (let i = 0; i < count; i++) {
            const opt = document.createElement("option");
            opt.value = i;
            opt.textContent = `Página ${i + 1}`;
            pageSelector.appendChild(opt);
        }
    }

    pageSelector.addEventListener("change", (e) => {
        activePageIndex = parseInt(e.target.value, 10);
        loadPageToCanvas(activePageIndex, false);
    });

    const activeImg = new Image();
    function loadPageToCanvas(pageIdx, isOriginal) {
        activeImg.crossOrigin = "anonymous";
        activeImg.src = `/api/v1/jobs/${currentJobId}/pages/${pageIdx}/image?original=${isOriginal}&t=${Date.now()}`;
        activeImg.onload = () => {
            canvas.width = activeImg.naturalWidth;
            canvas.height = activeImg.naturalHeight;
            viewportRes.textContent = `${activeImg.naturalWidth} × ${activeImg.naturalHeight} px`;
            ctx.drawImage(activeImg, 0, 0);
        };
    }

    canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = Math.round((e.clientX - rect.left) * scaleX);
        const y = Math.round((e.clientY - rect.top) * scaleY);
        viewportCoords.textContent = `X: ${x} | Y: ${y}`;
    });

    // ── 3. Retoque LAB (Bote de pintura) ──
    canvas.addEventListener("click", async (e) => {
        if (cardTouchup.style.display === "none") return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = Math.round((e.clientX - rect.left) * scaleX);
        const y = Math.round((e.clientY - rect.top) * scaleY);

        statusText.textContent = `Retocando (${x}, ${y})...`;

        try {
            const resp = await fetch(`/api/v1/jobs/${currentJobId}/pages/${activePageIndex}/touchup`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    x: x,
                    y: y,
                    hex_color: colorPicker.value,
                    tolerance: parseInt(toleranceSlider.value, 10),
                    feather: 3
                })
            });

            if (resp.ok) {
                loadPageToCanvas(activePageIndex, false);
                statusText.textContent = "Color aplicado";
            }
        } catch {
            statusText.textContent = "Error al retocar";
        }
    });

    // ── 4. Deshacer ──
    btnUndo.addEventListener("click", async () => {
        const resp = await fetch(`/api/v1/jobs/${currentJobId}/pages/${activePageIndex}/undo`, {
            method: "POST"
        });
        if (resp.ok) {
            loadPageToCanvas(activePageIndex, false);
            statusText.textContent = "Paso deshecho";
        } else {
            alert("No hay más estados previos para esta página.");
        }
    });
});