// src/rika_mangas/presentation/web/static/js/dropzone.js

/**
 * Configura una zona de arrastre y soltado de archivos accesible y sin parpadeos.
 * @param {HTMLElement} zone
 * @param {HTMLInputElement} input
 * @param {(files: File[]) => void} onFilesSelected
 */
export function setupDropzone(zone, input, onFilesSelected) {
    if (!zone || !input) return;

    zone.addEventListener("click", () => input.click());

    zone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            input.click();
        }
    });

    zone.addEventListener("dragenter", (e) => {
        e.preventDefault();
        zone.classList.add("dragover");
    });

    zone.addEventListener("dragover", (e) => {
        e.preventDefault();
        zone.classList.add("dragover");
    });

    zone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        if (zone.contains(e.relatedTarget)) return;
        zone.classList.remove("dragover");
    });

    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("dragover");
        if (e.dataTransfer?.files?.length) {
            const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
            if (files.length > 0) {
                zone.classList.add("has-files");
                onFilesSelected(files);
            }
        }
    });

    input.addEventListener("change", () => {
        if (input.files?.length) {
            const files = Array.from(input.files).filter((f) => f.type.startsWith("image/"));
            if (files.length > 0) {
                zone.classList.add("has-files");
                onFilesSelected(files);
            }
            input.value = ""; // Permite seleccionar los mismos archivos o sumar más
        }
    });
}
