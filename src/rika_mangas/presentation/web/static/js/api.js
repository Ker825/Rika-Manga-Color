export async function fetchPresets() {
    const res = await fetch("/api/v1/presets");
    if (!res.ok) throw new Error("Error al obtener presets");
    return res.json();
}

export async function createJob(pages, references, style, fidelity = null) {
    const formData = new FormData();
    pages.forEach((p) => formData.append("pages", p));
    references.forEach((r) => formData.append("references", r));
    formData.append("style", style);
    if (fidelity !== null) {
        formData.append("fidelity", fidelity);
    }

    const res = await fetch("/api/v1/jobs", { method: "POST", body: formData });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Error en validación");
    }
    return res.json();
}

export async function startJob(jobId) {
    return fetch(`/api/v1/jobs/${jobId}/start`, { method: "POST" });
}

export async function touchupPage(jobId, pageNum, payload) {
    return fetch(`/api/v1/jobs/${jobId}/pages/${pageNum}/touchup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
}

export async function undoTouchup(jobId, pageNum) {
    return fetch(`/api/v1/jobs/${jobId}/pages/${pageNum}/undo`, { method: "POST" });
}

export async function reprocessPage(jobId, pageNum, style, fidelity) {
    const res = await fetch(`/api/v1/jobs/${jobId}/pages/${pageNum}/reprocess`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            style: style,
            fidelity: parseFloat(fidelity)
        })
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Error al reprocesar preset");
    }
    return res.json();
}