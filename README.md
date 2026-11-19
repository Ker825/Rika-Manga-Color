# Rika-Manga-Color

Motor de colorización offline de manga y cómic guiada por referencia, basado en **MangaNinja** y optimizado mediante gestión de memoria por etapas (**Staged VRAM**) para ejecutarse en GPUs de **6GB de VRAM**.

---

## Características principales

- **Paridad de inferencia con MangaNinja:**
  - Extractor neuronal de lineart `BatchLineartDetector` (`sk_model.pth`) para eliminar screentones y rellenos negros antes de ControlNet.
  - `DDIMScheduler` nativo (30 pasos por defecto), `resnet_eps=1e-5` en bloques UNet y decodificación VAE sin recorte destructivo de latentes.
  - Codificación CLIP preservando el *aspect ratio* original de la referencia.
  - `PointNet` con máscaras pareadas e IDs incrementales (`1..N`) y CFG de 3 vías (`guidance_scale_ref=9.0`, `guidance_scale_point=15.0`).
- **Presupuesto < 6GB VRAM (Staged Offloading):**
  - Carga y descarga secuencial GPU/CPU por etapa (`BatchLineartDetector`, `AutoencoderKL`, `CLIPVisionModelWithProjection`, `PointNet`, `RefUNet2DConditionModel`, `ControlNetModel`, `UNet2DConditionModel`).
  - VAE con `enable_tiling()` y `enable_slicing()`, inferencia en FP16 bajo CUDA y un único vector `uncond_embed.pt` precalculado (sin cargar `CLIPTextModel` en runtime).
- **Modo offline estricto:**
  - Bloqueo de red y telemetría (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `DIFFUSERS_OFFLINE=1`) en [config.py](src/rika_mangas/config.py) y validación de checkpoints con `weights_only=True`.
- **Aplicación web completa:**
  - API REST + WebSockets en FastAPI con cola serializada de GPU (`GPUJobWorker`) y persistencia de trabajos en `storage/`.
  - Postprocesado determinista en espacio CIELAB con presets de estilo (`neutral`, `faithful_1to1`, `shonen`, `vibrant`, `summer_glow`, `webtoon_modern`, `shojo_pastel`, `seinen_dark`, `retro_90s`, `cyberpunk_neon`) y reprocesamiento instantáneo en CPU sin tocar la GPU.
  - Retoque interactivo *flood-fill* con historial *undo* y reescalado 4x por tiles con Real-ESRGAN Anime 6B.

---

## Requisitos

- Python `>= 3.12, < 3.14`
- [`uv`](https://docs.astral.sh/uv/) para gestión de dependencias y entorno virtual
- GPU NVIDIA con soporte CUDA (recomendado 6GB+ VRAM; soporta CPU como respaldo)

---

## Instalación

```bash
git clone <url-del-repositorio>
cd Rika-Manga-Color
uv sync
```

---

## Descarga y preparación de pesos

Descarga los checkpoints de MangaNinja, componentes base de SD 1.5, CLIP Vision, ControlNet Lineart, el annotator `sk_model.pth` y Real-ESRGAN:

```bash
uv run python scripts/download_models.py
```

Opcionalmente, para reducir espacio en disco convirtiendo los pesos a FP16:

```bash
uv run python scripts/download_models.py --convert-fp16
```

Si necesitas regenerar el embedding incondicional (`models/weights/manganinja/uncond_embed.pt`):

```bash
uv run python scripts/generate_uncond.py
```

Estructura esperada bajo `models/weights/`:

```text
models/weights/
├── annotators/
│   └── sk_model.pth
├── clip/
├── controlnet_lineart/
│   └── config.json
├── manganinja/
│   ├── controlnet.pth
│   ├── denoising_unet.pth
│   ├── point_net.pth
│   ├── reference_unet.pth
│   └── uncond_embed.pt
├── sd15/
│   ├── scheduler/scheduler_config.json
│   ├── unet/config.json
│   └── vae/
└── RealESRGAN_x4plus_anime_6B.pth
```

---

## Ejecución

Inicia el servidor FastAPI con Uvicorn:

```bash
uv run uvicorn rika_mangas.main:app --host 0.0.0.0 --port 8000
```

Abre `http://localhost:8000` en tu navegador.

### Configuración por variables de entorno (`RIKA_*`)

Todas las opciones en `Settings` ([src/rika_mangas/config.py](src/rika_mangas/config.py)) pueden sobreescribirse con el prefijo `RIKA_`, por ejemplo:

- `RIKA_TORCH_DEVICE=cuda` (`cuda` o `cpu`)
- `RIKA_MANGANINJA_STEPS=30`
- `RIKA_MANGANINJA_TILE_SIZE=512`
- `RIKA_ESRGAN_TILE_SIZE=512`

---

## Estructura del proyecto

```text
Rika-Manga-Color/
├── docs/                  # Documentación viva (PRD, Architecture, Design, Rules, Tasks)
├── models/weights/        # Pesos locales offline
├── scripts/               # Descarga, generación de uncond_embed y conversión FP16
├── storage/               # Subidas y salidas por job_id
└── src/rika_mangas/
    ├── config.py          # Configuración offline y rutas
    ├── main.py            # Aplicación FastAPI y vistas HTML
    ├── domain/            # Entidades, presets y puertos (IColorizerEngine, IUpscalerEngine)
    ├── core/              # Máscaras, PostProcessor LAB y flood_recolor
    ├── application/       # ColorizationOrchestrator y GPUJobWorker
    ├── infrastructure/    # Carga segura y motor MangaNinja + Lineart + Real-ESRGAN
    └── presentation/      # Routers REST/WS, esquemas Pydantic y UI web
```

---

## Documentación y créditos

- Documentación técnica del proyecto en [`docs/`](docs/ARCHITECTURE.md).
- Basado en el trabajo oficial **MangaNinja: Line Art Colorization with Precise Reference Following** ([arXiv:2501.08332](https://arxiv.org/abs/2501.08332)).
- Licencia del proyecto: MIT (`pyproject.toml`).
