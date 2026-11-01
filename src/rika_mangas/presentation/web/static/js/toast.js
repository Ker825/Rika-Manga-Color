// src/rika_mangas/presentation/web/static/js/toast.js
const TOAST_DURATION_MS = 4200;

/**
 * Muestra una notificación no bloqueante tipo píldora HUD (estilo macOS).
 * @param {string} message - Texto del mensaje
 * @param {"info" | "error" | "success"} type - Tipo de notificación
 */
export function showToast(message, type = "info") {
    const container = document.getElementById("toast-region");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast-pill toast-${type}`;
    toast.setAttribute("role", type === "error" ? "alert" : "status");

    const dot = document.createElement("span");
    dot.className = "toast-dot";

    const text = document.createElement("span");
    text.className = "toast-msg";
    text.textContent = message;

    toast.append(dot, text);
    container.appendChild(toast);

    requestAnimationFrame(() => toast.classList.add("is-visible"));

    setTimeout(() => {
        toast.classList.remove("is-visible");
        toast.addEventListener("transitionend", () => toast.remove(), { once: true });
    }, TOAST_DURATION_MS);
}
