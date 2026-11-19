from __future__ import annotations

import gc
import logging
from contextlib import suppress
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
)
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from rika_mangas.config import settings
from rika_mangas.domain.ports.ports import HintPoint, IColorizerEngine
from rika_mangas.infrastructure.ml.hooks import ReferenceAttentionControl
from rika_mangas.infrastructure.ml.lineart_detector import BatchLineartDetector
from rika_mangas.infrastructure.ml.models.refunet_2d_condition import (
    RefUNet2DConditionModel,
)
from rika_mangas.infrastructure.ml.models.unet_2d_condition import (
    UNet2DConditionModel,
)
from rika_mangas.infrastructure.ml.point_manager import PointManager
from rika_mangas.infrastructure.ml.point_network import PointNet
from rika_mangas.infrastructure.ml.reference_manager import ReferenceManager
from rika_mangas.infrastructure.ml.weight_loader import load_model_weights

logger = logging.getLogger("rika_mangas.manganinja")


def _move_to(
    module: Any,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Traslada submódulos a GPU/CPU de manera segura sorteando decoradores dinámicos."""
    if module is None:
        return None
    if device is not None and dtype is not None:
        return nn.Module.to(module, device, dtype=dtype)
    if device is not None:
        return nn.Module.to(module, device)
    if dtype is not None:
        return nn.Module.to(module, dtype=dtype)
    return module


class MangaNinjaStagedColorizer(IColorizerEngine):
    """Motor de colorización guiada MangaNinja optimizado para GPUs de 6GB VRAM.

    Aplica una arquitectura desacoplada SOLID:
    - ReferenceManager para selección y preparación de referencia cromática.
    - PointManager para rasterización y embedding multiescala de pistas espaciales.
    - ReferenceAttentionControl para mutual self-attention con desacoplamiento K/V.
    - Gestión de memoria por etapas (Staged VRAM) para mantener la memoria activa bajo 6GB.
    """

    RGB_LATENT_SCALE_FACTOR: float = 0.18215

    device: torch.device
    dtype: torch.dtype
    _loaded: bool
    scheduler: Any | None
    vae: Any | None
    image_encoder: Any | None
    clip_proc: CLIPImageProcessor | None
    denoising_unet: UNet2DConditionModel | None
    reference_unet: RefUNet2DConditionModel | None
    controlnet: Any | None
    point_net: PointNet | None
    lineart_detector: BatchLineartDetector | None
    uncond_embed: torch.Tensor | None

    def __init__(self) -> None:
        target_device = getattr(settings, "TORCH_DEVICE", "cuda")
        self.device = torch.device(
            target_device if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._loaded = False

        self.scheduler = None
        self.vae = None
        self.image_encoder = None
        self.clip_proc = None
        self.denoising_unet = None
        self.reference_unet = None
        self.controlnet = None
        self.point_net = None
        self.lineart_detector = None
        self.uncond_embed = None

    def load(self) -> None:
        """Carga y valida todos los pesos requeridos para el pipeline offline."""
        if self._loaded:
            return

        logger.info("Cargando componentes nativos de MangaNinja...")

        # 1. Scheduler DDIM nativo de MangaNinja
        sched_raw: Any = DDIMScheduler.from_pretrained(
            str(settings.SD_15_LOCAL_DIR),
            subfolder="scheduler",
            local_files_only=True,
        )
        self.scheduler = sched_raw[0] if isinstance(sched_raw, tuple) else sched_raw

        # 2. VAE con soporte de tiling para baja memoria
        vae_model = AutoencoderKL.from_pretrained(
            str(settings.SD_15_LOCAL_DIR),
            subfolder="vae",
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        vae_any: Any = vae_model
        vae_any.enable_tiling()
        vae_any.enable_slicing()
        self.vae = vae_any

        # 3. CLIP Image Encoder y Preprocesador
        clip_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(settings.CLIP_LOCAL_DIR),
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        self.image_encoder = clip_encoder
        self.clip_proc = CLIPImageProcessor.from_pretrained(
            str(settings.CLIP_LOCAL_DIR),
            local_files_only=True,
        )

        # 4. Vector estático incondicional precalculado
        self.uncond_embed = torch.load(
            settings.MANGANINJA_WEIGHTS_DIR / "uncond_embed.pt",
            map_location="cpu",
            weights_only=True,
        ).to(dtype=self.dtype)

        # 5. UNet Denoising
        denoising = UNet2DConditionModel(
            in_channels=4,
            out_channels=4,
            block_out_channels=(320, 640, 1280, 1280),
            layers_per_block=2,
            cross_attention_dim=768,
            attention_head_dim=8,
        )
        load_model_weights(
            denoising,
            settings.MANGANINJA_WEIGHTS_DIR / "denoising_unet.pth",
            ignore_output_layers=False,
        )
        _move_to(denoising, dtype=self.dtype)
        self.denoising_unet = denoising

        # 6. UNet Reference
        reference = RefUNet2DConditionModel(
            in_channels=4,
            block_out_channels=(320, 640, 1280, 1280),
            layers_per_block=2,
            cross_attention_dim=768,
            attention_head_dim=8,
        )
        load_model_weights(
            reference,
            settings.MANGANINJA_WEIGHTS_DIR / "reference_unet.pth",
            ignore_output_layers=True,
        )
        _move_to(reference, dtype=self.dtype)
        self.reference_unet = reference

        # 7. ControlNet (Lineart)
        ctrl_config = ControlNetModel.load_config(
            str(settings.CONTROLNET_LINEART_LOCAL_DIR),
            local_files_only=True,
        )
        ctrl_model = ControlNetModel.from_config(ctrl_config)
        load_model_weights(
            ctrl_model,
            settings.MANGANINJA_WEIGHTS_DIR / "controlnet.pth",
            ignore_output_layers=False,
        )
        ctrl_any: Any = ctrl_model
        _move_to(ctrl_any, dtype=self.dtype)
        self.controlnet = ctrl_any

        # 8. PointNet
        point_net = PointNet(
            conditioning_channels=1,
            out_channels=(320, 640, 1280, 1280),
            downsamples=(6, 2, 2, 2),
        )
        load_model_weights(
            point_net,
            settings.MANGANINJA_WEIGHTS_DIR / "point_net.pth",
            ignore_output_layers=False,
        )
        _move_to(point_net, dtype=self.dtype)
        self.point_net = point_net

        # 9. Detector neuronal de lineart (sk_model.pth)
        if settings.LINEART_ANNOTATOR_PATH.is_file():
            detector = BatchLineartDetector(settings.LINEART_ANNOTATOR_PATH)
            _move_to(detector, "cpu", dtype=torch.float32)
            self.lineart_detector = detector
        else:
            logger.warning(
                "Annotator sk_model.pth no encontrado en %s; se usará fallback OpenCV.",
                settings.LINEART_ANNOTATOR_PATH,
            )
            self.lineart_detector = None

        self._loaded = True
        logger.info("MangaNinja inicializado correctamente.")

    def _prepare_canvas(
        self, image_bgr: np.ndarray, size: int
    ) -> tuple[int, int, int, int, np.ndarray]:
        """Calcula dimensiones escaladas a múltiplos exactos de 64 sin bandas de padding."""
        orig_h, orig_w = image_bgr.shape[:2]
        scale = min(size / max(orig_h, orig_w), 1.0)
        pad_w = max(64, round(orig_w * scale / 64) * 64)
        pad_h = max(64, round(orig_h * scale / 64) * 64)

        img_resized = cv2.resize(
            image_bgr, (pad_w, pad_h), interpolation=cv2.INTER_AREA
        )
        return orig_w, orig_h, pad_w, pad_h, img_resized

    def _prepare_lineart_condition(
        self,
        img_padded: np.ndarray,
        denoise_sigma: int = 0,
        is_lineart: bool = False,
    ) -> torch.Tensor:
        """Extrae y normaliza el mapa de lineart según el protocolo nativo de MangaNinja."""
        if not is_lineart and self.lineart_detector is not None:
            rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb_norm = (rgb / 255.0) * 2.0 - 1.0
            rgb_tensor = (
                torch.from_numpy(rgb_norm)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=self.device, dtype=torch.float32)
            )
            _move_to(self.lineart_detector, self.device, dtype=torch.float32)
            edge = self.lineart_detector(rgb_tensor)
            _move_to(self.lineart_detector, "cpu")
            edge[edge <= 0.24] = 0.0
            return (edge.repeat(1, 3, 1, 1) * 2.0 - 1.0).to(
                device=self.device, dtype=self.dtype
            )

        gray_u8 = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)
        if denoise_sigma > 0 and not is_lineart:
            gray_u8 = cv2.bilateralFilter(
                gray_u8, d=5, sigmaColor=float(denoise_sigma), sigmaSpace=3.0
            )
        # ControlNet lineart requiere líneas blancas (1.0) sobre fondo negro (0.0).
        lines = 1.0 - (gray_u8.astype(np.float32) / 255.0)
        edge = (
            torch.from_numpy(lines)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
        )
        edge[edge <= 0.24] = 0.0
        return edge.repeat(1, 3, 1, 1) * 2.0 - 1.0

    def colorize_page(
        self,
        image_bgr: np.ndarray,
        *,
        reference_images: list[np.ndarray] | None = None,
        size: int = settings.MANGANINJA_TILE_SIZE,
        steps: int = settings.MANGANINJA_STEPS,
        denoise_sigma: int = 0,
        is_lineart: bool = False,
        tiled: bool = False,
        tile_size: int = settings.MANGANINJA_TILE_SIZE,
        tile_overlap: int = 96,
        hint_points: list[HintPoint] | None = None,
        guidance_scale_ref: float = 9.0,
        guidance_scale_point: float = 15.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Ejecuta el ciclo de colorización completa por difusión con mutual attention."""
        with torch.inference_mode():
            self.load()

            if (
                self.scheduler is None
                or self.vae is None
                or self.image_encoder is None
                or self.clip_proc is None
                or self.denoising_unet is None
                or self.reference_unet is None
                or self.controlnet is None
                or self.point_net is None
                or self.uncond_embed is None
            ):
                raise RuntimeError("Componentes de MangaNinja no inicializados.")

            orig_w, orig_h, work_w, work_h, img_padded = self._prepare_canvas(
                image_bgr, size
            )
            pad_h, pad_w = img_padded.shape[:2]

            # 1. Selección y acondicionamiento de la imagen de referencia
            ref_bgr = ReferenceManager.select_best_reference(
                reference_images, (pad_h, pad_w)
            )
            ref_rgb, ref_tensor = ReferenceManager.prepare_reference_tensor(
                ref_bgr, pad_w, pad_h, device=self.device, dtype=self.dtype
            )

            # 2. Codificación VAE y CLIP de la referencia (VRAM Staged)
            _move_to(self.vae, self.device)
            vae_out: Any = self.vae.encode(ref_tensor)
            dist = (vae_out[0] if isinstance(vae_out, tuple) else vae_out).latent_dist
            ref_latents = dist.sample() * self.RGB_LATENT_SCALE_FACTOR
            _move_to(self.vae, "cpu")

            _move_to(self.image_encoder, self.device)
            clip_inputs: Any = self.clip_proc(images=ref_rgb, return_tensors="pt")
            pixel_vals = cast(torch.Tensor, clip_inputs["pixel_values"]).to(
                self.device, dtype=self.dtype
            )
            img_out: Any = self.image_encoder(pixel_values=pixel_vals)
            ref_embeds = img_out.image_embeds.unsqueeze(1)
            _move_to(self.image_encoder, "cpu")

            uncond_embeds = self.uncond_embed.to(device=self.device, dtype=self.dtype)

            # 3. Codificación espacial PointNet y CFG de 3 vías (idéntico a MangaNinjiaPipeline)
            num_cfg = 3
            point_mask_ref, point_mask_main = PointManager.generate_point_masks(
                hint_points,
                pad_w,
                pad_h,
                self.device,
                self.dtype,
                ref_bgr=ref_bgr,
            )
            _move_to(self.point_net, self.device)
            point_main_embs = PointManager.encode_points(
                self.point_net, point_mask_main
            )
            point_ref_embs = PointManager.encode_points(
                self.point_net, point_mask_ref
            )
            _move_to(self.point_net, "cpu")

            cfg_encoder_hidden_states = torch.cat(
                [uncond_embeds, ref_embeds, ref_embeds], dim=0
            )

            # 4. Configuración de hooks de Mutual Self-Attention
            writer: ReferenceAttentionControl | None = None
            reader: ReferenceAttentionControl | None = None

            try:
                writer = ReferenceAttentionControl(
                    self.reference_unet,
                    mode="write",
                    do_classifier_free_guidance=False,
                    fusion_blocks="full",
                )
                reader = ReferenceAttentionControl(
                    self.denoising_unet,
                    mode="read",
                    do_classifier_free_guidance=True,
                    fusion_blocks="full",
                )

                # 5. Paso 0: Extracción de características de referencia en Reference UNet
                _move_to(self.reference_unet, self.device)
                zero_timestep = torch.zeros((1,), dtype=torch.long, device=self.device)
                self.reference_unet(
                    ref_latents,
                    zero_timestep,
                    encoder_hidden_states=ref_embeds,
                    return_dict=False,
                )
                # Copiar banco del escritor al lector e inyectar embeddings espaciales
                reader.update(
                    writer,
                    point_embedding_ref=point_ref_embs,
                    point_embedding_main=point_main_embs,
                    dtype=self.dtype,
                )
                # Liberar Reference UNet de VRAM para mantener presupuesto de 6GB
                _move_to(self.reference_unet, "cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 6. Preparación de lineart y latentes de partida
                edge_cond = self._prepare_lineart_condition(
                    img_padded, denoise_sigma=denoise_sigma, is_lineart=is_lineart
                )
                cfg_cond_tensor = edge_cond.repeat(num_cfg, 1, 1, 1)

                latents = torch.randn(
                    (1, 4, pad_h // 8, pad_w // 8),
                    device=self.device,
                    dtype=self.dtype,
                )
                self.scheduler.set_timesteps(steps, device=self.device)
                latents = latents * self.scheduler.init_noise_sigma

                # 7. Ciclo de difusión (Denoising Loop)
                _move_to(self.controlnet, self.device)
                _move_to(self.denoising_unet, self.device)

                for t in self.scheduler.timesteps:
                    latent_model_input = latents.repeat(num_cfg, 1, 1, 1)
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    ctrl_out: Any = self.controlnet(
                        latent_model_input,
                        timestep=t,
                        encoder_hidden_states=cfg_encoder_hidden_states,
                        controlnet_cond=cfg_cond_tensor,
                        conditioning_scale=1.0,
                        return_dict=False,
                    )
                    down_res, mid_res = ctrl_out[0], ctrl_out[1]

                    unet_out: Any = self.denoising_unet(
                        latent_model_input,
                        timestep=t,
                        encoder_hidden_states=cfg_encoder_hidden_states,
                        down_block_additional_residuals=down_res,
                        mid_block_additional_residual=mid_res,
                        return_dict=False,
                    )
                    pred_sample = unet_out[0]

                    # 8. Predicción de ruido guiada por clasificador (CFG 3 vías)
                    noise_pred_uncond, noise_pred_ref, noise_pred_point = (
                        pred_sample.chunk(3)
                    )
                    noise_pred_1 = noise_pred_uncond + guidance_scale_ref * (
                        noise_pred_ref - noise_pred_uncond
                    )
                    noise_pred_2 = noise_pred_ref + guidance_scale_point * (
                        noise_pred_point - noise_pred_ref
                    )
                    noise_pred = (noise_pred_1 + noise_pred_2) / 2.0

                    step_res: Any = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )
                    latents = (
                        step_res[0]
                        if isinstance(step_res, tuple)
                        else step_res.prev_sample
                    )

                # 9. Decodificación final de latentes con VAE
                _move_to(self.controlnet, "cpu")
                _move_to(self.denoising_unet, "cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                _move_to(self.vae, self.device)
                latents_scaled = latents / self.RGB_LATENT_SCALE_FACTOR
                dec_out: Any = self.vae.decode(
                    latents_scaled,
                    return_dict=False,
                )
                decoded = (
                    dec_out[0]
                    if isinstance(dec_out, tuple)
                    else getattr(dec_out, "sample", dec_out)
                )

                out_img = (
                    ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5)
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
                out_bgr = cv2.cvtColor(
                    out_img.round().clamp(0, 255).to(torch.uint8).cpu().numpy(),
                    cv2.COLOR_RGB2BGR,
                )

                # Deshacer padding y ajustar a la resolución nativa
                unpadded = out_bgr[:work_h, :work_w]
                return cv2.resize(
                    unpadded, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
                )

            finally:
                if reader is not None:
                    reader.remove_hooks()
                if writer is not None:
                    writer.remove_hooks()

                for component in (
                    self.denoising_unet,
                    self.reference_unet,
                    self.controlnet,
                    self.vae,
                    self.point_net,
                    self.image_encoder,
                    self.lineart_detector,
                ):
                    if component is not None:
                        with suppress(Exception):
                            _move_to(component, "cpu")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def unload(self) -> None:
        """Libera todos los módulos y memoria de GPU/CPU."""
        if self._loaded:
            self.vae = None
            self.image_encoder = None
            self.clip_proc = None
            self.denoising_unet = None
            self.reference_unet = None
            self.controlnet = None
            self.point_net = None
            self.lineart_detector = None
            self.scheduler = None
            self.uncond_embed = None
            self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
