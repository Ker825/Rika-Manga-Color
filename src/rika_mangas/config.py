import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Bloqueo estricto de red y telemetría antes de importar librerías de difusión
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RIKA_", extra="ignore")
    BASE_DIR: Path = Path(__file__).resolve().parent
    PROJECT_ROOT: Path = BASE_DIR.parents[1]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    TORCH_DEVICE: str = "cuda"
    WEIGHTS_DIR: Path = PROJECT_ROOT / "models" / "weights"
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    OUTPUT_DIR: Path = STORAGE_DIR / "output"
    SD_15_LOCAL_DIR: Path = WEIGHTS_DIR / "sd15"
    CLIP_LOCAL_DIR: Path = WEIGHTS_DIR / "clip"
    CONTROLNET_LINEART_LOCAL_DIR: Path = WEIGHTS_DIR / "controlnet_lineart"
    MANGANINJA_WEIGHTS_DIR: Path = WEIGHTS_DIR / "manganinja"
    ESRGAN_WEIGHTS_PATH: Path = WEIGHTS_DIR / "RealESRGAN_x4plus_anime_6B.pth"

    TEMPLATES_DIR: Path = BASE_DIR / "presentation" / "api" / "web" / "templates"
    STATIC_DIR: Path = BASE_DIR / "presentation" / "api" / "web"

    MAX_IMAGE_PIXELS: int = 20_000_000
    MAX_DIMENSION: int = 4096
    MANGANINJA_STEPS: int = 16
    MANGANINJA_TILE_SIZE: int = 512
    CHARACTER_BLEND_STRENGTH: float = 0.55
    ESRGAN_TILE_SIZE: int = 512

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


settings = Settings()
