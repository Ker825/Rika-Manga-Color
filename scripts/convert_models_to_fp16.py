#!/usr/bin/env python3
"""
Script de conversión de modelos a FP16 (Half Precision) para ahorro de espacio en frío.

Convierte los checkpoints de FP32 a FP16 en disco:
  - MangaNinja: controlnet, denoising_unet, reference_unet, point_net, uncond_embed
  - CLIP Vision: model.safetensors
  - Real-ESRGAN: RealESRGAN_x4plus_anime_6B.pth
  - SD 1.5 VAE: reemplazo seguro con safetensors FP16 y remoción de archivos binarios redundantes.
"""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rika_mangas.config import settings


def format_size(num_bytes: int) -> str:
    """Retorna un tamaño en formato legible (B, KB, MB, GB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def convert_pth_dict_to_fp16(src_path: Path, top_level_key: str | None = None) -> tuple[int, int]:
    """Convierte un checkpoint .pth de PyTorch a FP16 de manera atómica."""
    if not src_path.is_file():
        print(f"  [-] Archivo no encontrado: {src_path}")
        return 0, 0

    size_before = src_path.stat().st_size
    print(f"[*] Procesando {src_path.name} ({format_size(size_before)})...")

    data: Any = torch.load(str(src_path), map_location="cpu", weights_only=True)

    if top_level_key:
        inner_dict = data[top_level_key]
        fp16_dict = {
            k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in inner_dict.items()
        }
        data[top_level_key] = fp16_dict
    elif isinstance(data, dict):
        data = {
            k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in data.items()
        }
    elif isinstance(data, torch.Tensor):
        if data.is_floating_point():
            data = data.half()
    else:
        print(f"  [-] Formato no reconocido para {src_path}")
        return size_before, size_before

    tmp_path = src_path.with_suffix(".tmp")
    torch.save(data, tmp_path)
    del data
    gc.collect()

    os.replace(tmp_path, src_path)
    size_after = src_path.stat().st_size
    diff = size_before - size_after
    print(f"  [OK] {src_path.name}: {format_size(size_before)} -> {format_size(size_after)} (Ahorro: {format_size(diff)})")
    return size_before, size_after


def convert_clip_safetensors_to_fp16(clip_dir: Path) -> tuple[int, int]:
    """Convierte el archivo model.safetensors de CLIP a FP16 conservando tensores enteros."""
    src_path = clip_dir / "model.safetensors"
    if not src_path.is_file():
        print(f"  [-] CLIP model.safetensors no encontrado en {clip_dir}")
        return 0, 0

    size_before = src_path.stat().st_size
    print(f"[*] Procesando CLIP model.safetensors ({format_size(size_before)})...")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(src_path), framework="pt") as f:
        meta = f.metadata() or {"format": "pt"}
        for k in f.keys():
            t = f.get_tensor(k)
            tensors[k] = t.half() if t.is_floating_point() else t

    tmp_path = src_path.with_suffix(".tmp")
    save_file(tensors, str(tmp_path), metadata=meta)
    del tensors
    gc.collect()

    os.replace(tmp_path, src_path)
    size_after = src_path.stat().st_size
    diff = size_before - size_after
    print(f"  [OK] CLIP model.safetensors: {format_size(size_before)} -> {format_size(size_after)} (Ahorro: {format_size(diff)})")
    return size_before, size_after


def optimize_sd15_vae(sd15_dir: Path) -> tuple[int, int]:
    """Optimiza el VAE de SD 1.5 consolidando en safetensors FP16 y eliminando .bin duplicados."""
    vae_dir = sd15_dir / "vae"
    if not vae_dir.is_dir():
        print(f"  [-] Directorio VAE no encontrado en {vae_dir}")
        return 0, 0

    size_before = sum(f.stat().st_size for f in vae_dir.glob("*") if f.is_file())
    print(f"[*] Optimizando VAE SD 1.5 ({format_size(size_before)} en total)...")

    fp16_safetensors = vae_dir / "diffusion_pytorch_model.fp16.safetensors"
    std_safetensors = vae_dir / "diffusion_pytorch_model.safetensors"

    # Si existe la versión fp16.safetensors, actualizar diffusion_pytorch_model.safetensors con ella
    if fp16_safetensors.is_file():
        shutil.copyfile(fp16_safetensors, std_safetensors)
        print("  [OK] diffusion_pytorch_model.safetensors actualizado a FP16.")

    # Remover archivos .bin legacy y pesados
    redundant_files = [
        vae_dir / "diffusion_pytorch_model.bin",
        vae_dir / "diffusion_pytorch_model.fp16.bin",
    ]
    for rf in redundant_files:
        if rf.is_file():
            rf_size = rf.stat().st_size
            rf.unlink()
            print(f"  [Removido] Archivo redundante: {rf.name} ({format_size(rf_size)})")

    size_after = sum(f.stat().st_size for f in vae_dir.glob("*") if f.is_file())
    diff = size_before - size_after
    print(f"  [OK] VAE SD 1.5 optimizado: {format_size(size_before)} -> {format_size(size_after)} (Ahorro: {format_size(diff)})")
    return size_before, size_after


def main() -> None:
    print("=" * 65)
    print("  CONVERSIÓN DE MODELOS A FP16 (AHORRO EN FRÍO) - RIKA MANGA")
    print("=" * 65)

    total_before = 0
    total_after = 0

    # 1. MangaNinja Checkpoints
    manganinja_dir = settings.MANGANINJA_WEIGHTS_DIR
    manganinja_models = [
        "controlnet.pth",
        "denoising_unet.pth",
        "reference_unet.pth",
        "point_net.pth",
    ]
    for m in manganinja_models:
        b, a = convert_pth_dict_to_fp16(manganinja_dir / m)
        total_before += b
        total_after += a

    # uncond_embed.pt
    b, a = convert_pth_dict_to_fp16(manganinja_dir / "uncond_embed.pt")
    total_before += b
    total_after += a

    # 2. Real-ESRGAN
    b, a = convert_pth_dict_to_fp16(settings.ESRGAN_WEIGHTS_PATH, top_level_key="params_ema")
    total_before += b
    total_after += a

    # 3. CLIP Vision safetensors
    b, a = convert_clip_safetensors_to_fp16(settings.CLIP_LOCAL_DIR)
    total_before += b
    total_after += a

    # 4. SD 1.5 VAE
    b, a = optimize_sd15_vae(settings.SD_15_LOCAL_DIR)
    total_before += b
    total_after += a

    total_saved = total_before - total_after
    pct_saved = (total_saved / total_before * 100) if total_before > 0 else 0.0

    print("=" * 65)
    print(f"Espacio total antes:    {format_size(total_before)}")
    print(f"Espacio total después:  {format_size(total_after)}")
    print(f"Espacio total ahorrado: {format_size(total_saved)} ({pct_saved:.1f}% de reducción)")
    print("=" * 65)
    print("[V] Conversión completada exitosamente.")


if __name__ == "__main__":
    main()
