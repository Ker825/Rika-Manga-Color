from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

base_weights = Path("models/weights")
base_weights.mkdir(parents=True, exist_ok=True)

# 1. SD 1.5: Solo scheduler, VAE y el config de UNet
sd15_dir = base_weights / "sd15"
print("[*] Descargando componentes minimos de SD 1.5...")
hf_hub_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    filename="scheduler/scheduler_config.json",
    local_dir=sd15_dir,
)
hf_hub_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    filename="unet/config.json",
    local_dir=sd15_dir,
)
snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir=sd15_dir,
    allow_patterns=["vae/*.json", "vae/*.safetensors"],
)

# 2. ControlNet Lineart: Solo config.json
cnet_dir = base_weights / "controlnet_lineart"
print("[*] Descargando configuracion de ControlNet Lineart...")
hf_hub_download(
    repo_id="lllyasviel/control_v11p_sd15_lineart",
    filename="config.json",
    local_dir=cnet_dir,
)

# 3. CLIP Vision (large-patch14)
clip_dir = base_weights / "clip"
print("[*] Descargando CLIP Vision...")
snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir=clip_dir,
    allow_patterns=["*.json", "*.safetensors"],
)

# 4. Annotators (Lineart sk_model.pth)
annotators_dir = base_weights / "annotators"
annotators_dir.mkdir(parents=True, exist_ok=True)
print("[*] Descargando Lineart Annotator (sk_model.pth)...")
hf_hub_download(
    repo_id="lllyasviel/Annotators",
    filename="sk_model.pth",
    local_dir=annotators_dir,
)

print("\n[OK] Componentes base instalados correctamente.")
