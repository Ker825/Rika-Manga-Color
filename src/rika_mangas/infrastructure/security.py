import hashlib
import hmac
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


def verify_file_sha256(file_path: Path, expected_hash: str | None = None) -> None:
    if not file_path.is_file():
        raise FileNotFoundError(f"Weight file missing: {file_path}")
    if expected_hash is not None:
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if not hmac.compare_digest(digest, expected_hash.lower()):
            raise PermissionError(f"Integrity check failed: SHA-256 mismatch for {file_path}")


def safe_load_checkpoint(file_path: Path, expected_hash: str | None = None) -> dict[str, Any]:
    verify_file_sha256(file_path, expected_hash)
    data: Any = torch.load(str(file_path), map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise TypeError(f"Invalid checkpoint content in {file_path}: expected dict, got {type(data)}")
    return data


def validate_and_decode_image(
    raw_bytes: bytes,
    max_pixels: int = 20_000_000,
    max_dim: int = 4096,
) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = max_pixels
    with Image.open(io.BytesIO(raw_bytes)) as img:
        fmt = img.format
        if fmt not in ("PNG", "JPEG", "WEBP"):
            raise ValueError(f"Unsupported image format: {fmt}")
        width, height = img.size
        if width > max_dim or height > max_dim or (width * height) > max_pixels:
            raise ValueError(f"Image dimensions ({width}x{height}) exceed safety limits")
        rgb_img = img.convert("RGB")
        arr = np.asarray(rgb_img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
