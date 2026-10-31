import argparse
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

BASE_WEIGHTS = Path("models/weights")


def download_manganinja() -> None:
    target_dir = BASE_WEIGHTS / "manganinja"
    target_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "Johanan0528/MangaNinjia"
    files = [
        "denoising_unet.pth",
        "reference_unet.pth",
        "controlnet.pth",
        "point_net.pth",
    ]

    print(f"[*] Descargando pesos de MangaNinja ({repo_id})...")
    for filename in files:
        dest = target_dir / filename
        if dest.is_file():
            print(f"  [OK] {filename} ya existe.")
            continue
        print(f"  -> Descargando {filename}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
        )
    print("[*] Pesos de MangaNinja descargados en:", target_dir)


def download_base_components() -> None:
    # 1. SD 1.5 (VAE, Scheduler, UNet base)
    sd15_dir = BASE_WEIGHTS / "sd15"
    if not (sd15_dir / "unet").exists() or not (sd15_dir / "vae").exists():
        print("[*] Descargando componentes base de SD 1.5...")
        snapshot_download(
            repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
            local_dir=sd15_dir,
            allow_patterns=["unet/*", "vae/*", "scheduler/*"],
        )

    # 2. CLIP Vision (image_encoder y processor)
    clip_dir = BASE_WEIGHTS / "clip"
    if not clip_dir.exists():
        print("[*] Descargando CLIP Vision (large-patch14)...")
        snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=clip_dir,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"],
        )

    # 3. ControlNet Lineart
    cnet_dir = BASE_WEIGHTS / "controlnet_lineart"
    if not cnet_dir.exists():
        print("[*] Descargando ControlNet Lineart...")
        snapshot_download(
            repo_id="lllyasviel/control_v11p_sd15_lineart",
            local_dir=cnet_dir,
            allow_patterns=["*.json", "*.safetensors", "*.bin"],
        )

    # 4. Real-ESRGAN Anime 6B
    esrgan_path = BASE_WEIGHTS / "RealESRGAN_x4plus_anime_6B.pth"
    if not esrgan_path.is_file():
        print("[*] Descargando RealESRGAN_x4plus_anime_6B.pth...")
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        urllib.request.urlretrieve(url, esrgan_path)
        print("  [OK] Real-ESRGAN listo.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga de pesos para Rika Mangas")
    parser.add_argument("--only-manganinja", action="store_true", help="Descarga únicamente los checkpoints de MangaNinja")
    args = parser.parse_args()

    BASE_WEIGHTS.mkdir(parents=True, exist_ok=True)
    download_manganinja()

    if not args.only_manganinja:
        download_base_components()

    print("\n[V] Descarga completada exitosamente.")


if __name__ == "__main__":
    main()
