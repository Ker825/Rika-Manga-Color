import gc
import logging
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from rika_mangas.config import settings
from rika_mangas.domain.ports.ports import HintPoint, IColorizerEngine
from rika_mangas.infrastructure.ml.hooks import MangaNinjaAttnProcessor2_0, ReferenceFeatureBank
from rika_mangas.infrastructure.security import safe_load_checkpoint

logger = logging.getLogger("rika_mangas.manganinja")


class MangaNinjaStagedColorizer(IColorizerEngine):
    scheduler: Any
    vae: Any
    image_encoder: Any
    clip_proc: Any
    denoising_unet: Any
    reference_unet: Any
    controlnet: Any

    def __init__(self) -> None:
        target_device = getattr(settings, "TORCH_DEVICE", "cuda")
        self.device = torch.device(target_device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.bank = ReferenceFeatureBank()
        self._loaded = False
        self.scheduler = None
        self.vae = None
        self.image_encoder = None
        self.clip_proc = None
        self.denoising_unet = None
        self.reference_unet = None
        self.controlnet = None

    def load(self) -> None:
        if self._loaded:
            return

        logger.info("Inicializando componentes base de MangaNinja...")

        sched_raw: Any = DPMSolverMultistepScheduler.from_pretrained(
            str(settings.SD_15_LOCAL_DIR),
            subfolder="scheduler",
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
            local_files_only=True,
        )
        self.scheduler = sched_raw[0] if isinstance(sched_raw, tuple) else sched_raw

        vae_model = AutoencoderKL.from_pretrained(
            str(settings.SD_15_LOCAL_DIR),
            subfolder="vae",
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        self.vae = vae_model
        self.vae.enable_tiling()
        self.vae.enable_slicing()

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(settings.CLIP_LOCAL_DIR),
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        self.clip_proc = CLIPImageProcessor.from_pretrained(
            str(settings.CLIP_LOCAL_DIR),
            local_files_only=True,
        )

        unet_config = UNet2DConditionModel.load_config(
            str(settings.SD_15_LOCAL_DIR),
            subfolder="unet",
            local_files_only=True,
        )
        self.denoising_unet = UNet2DConditionModel.from_config(unet_config).to(dtype=self.dtype)
        denoise_w = safe_load_checkpoint(settings.MANGANINJA_WEIGHTS_DIR / "denoising_unet.pth")
        self.denoising_unet.load_state_dict(denoise_w, strict=False)

        self.reference_unet = UNet2DConditionModel.from_config(unet_config).to(dtype=self.dtype)
        ref_w = safe_load_checkpoint(settings.MANGANINJA_WEIGHTS_DIR / "reference_unet.pth")
        self.reference_unet.load_state_dict(ref_w, strict=False)

        ctrl_config = ControlNetModel.load_config(
            str(settings.CONTROLNET_LINEART_LOCAL_DIR),
            local_files_only=True,
        )
        self.controlnet = ControlNetModel.from_config(ctrl_config).to(dtype=self.dtype)
        ctrl_w = safe_load_checkpoint(settings.MANGANINJA_WEIGHTS_DIR / "controlnet.pth")
        self.controlnet.load_state_dict(ctrl_w, strict=False)

        for name, module in self.denoising_unet.named_modules():
            if module.__class__.__name__ == "Attention" and name.endswith(".attn1"):
                set_p = getattr(module, "set_processor", None)
                if callable(set_p):
                    set_p(MangaNinjaAttnProcessor2_0(name, self.bank))

        self._loaded = True
        logger.info("Modelos cargados correctamente.")

    @staticmethod
    def _calculate_chroma(img_bgr: np.ndarray) -> float:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a = lab[:, :, 1] - 128.0
        b = lab[:, :, 2] - 128.0
        return float(np.sqrt(a * a + b * b).mean())

    def _select_reference(self, refs: list[np.ndarray]) -> np.ndarray:
        if not refs:
            raise ValueError("No se proporcionaron imagenes de referencia")
        if len(refs) == 1:
            return refs[0]
        return max(refs, key=self._calculate_chroma)

    @torch.inference_mode()
    def colorize_page(
        self,
        image_bgr: np.ndarray,
        *,
        reference_images: list[np.ndarray] | None = None,
        size: int = 768,
        steps: int = 16,
        denoise_sigma: int = 15,
        tiled: bool = False,
        tile_size: int = 768,
        tile_overlap: int = 96,
        hint_points: list[HintPoint] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        self.load()

        orig_h, orig_w = image_bgr.shape[:2]

        # 1. Escalar manteniendo aspect ratio al tamaño de trabajo (lado mayor <= size)
        scale = min(size / max(orig_h, orig_w), 1.0)
        work_w = int(orig_w * scale)
        work_h = int(orig_h * scale)

        # Alinear a múltiplos de 64 para VAE / UNet
        pad_w = max(64, (work_w + 63) // 64 * 64)
        pad_h = max(64, (work_h + 63) // 64 * 64)

        img_resized = cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h - work_h, 0, pad_w - work_w, cv2.BORDER_REFLECT_101)
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

        if reference_images:
            ref_bgr = self._select_reference(reference_images)
            ref_resized = cv2.resize(ref_bgr, (pad_w, pad_h), interpolation=cv2.INTER_AREA)
        else:
            # Si no hay referencia, usar lienzo neutro balanceado (evita colapso a sepia)
            ref_resized = np.full((pad_h, pad_w, 3), 128, dtype=np.uint8)

        ref_rgb = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2RGB)

        try:
            # SUB-ETAPA A: Codificación VAE / CLIP
            self.vae.to(self.device)
            nn.Module.to(self.image_encoder, self.device)

            ref_tensor = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype) / 127.5 - 1.0
            vae_cfg = self.vae.config
            scale_fac = float(
                vae_cfg.get("scaling_factor", 0.18215) if isinstance(vae_cfg, dict) else getattr(vae_cfg, "scaling_factor", 0.18215)
            )

            vae_out: Any = self.vae.encode(ref_tensor)
            dist = getattr(vae_out, "latent_dist", vae_out[0] if isinstance(vae_out, tuple) else vae_out)
            ref_latents = dist.sample() * scale_fac

            clip_inputs: Any = self.clip_proc(images=ref_rgb, return_tensors="pt")
            pixel_vals = cast(torch.Tensor, clip_inputs["pixel_values"]).to(self.device, dtype=self.dtype)
            img_out: Any = self.image_encoder(pixel_values=pixel_vals)
            image_embeds = img_out.image_embeds.unsqueeze(1)

            self.vae.to("cpu")
            nn.Module.to(self.image_encoder, "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # SUB-ETAPA B: Extracción de Características de Referencia
            self.bank.remove_hooks()
            for name, module in self.reference_unet.named_modules():
                if module.__class__.__name__ == "Attention" and name.endswith(".attn1"):
                    self.bank.register(module, name)

            self.reference_unet.to(self.device)
            zero_t = torch.zeros(1, device=self.device, dtype=torch.long)
            self.reference_unet(
                ref_latents.to(self.device),
                timestep=zero_t,
                encoder_hidden_states=image_embeds.to(self.device),
                return_dict=False,
            )
            self.reference_unet.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # SUB-ETAPA C: Bucle de Denoising con CFG
            self.controlnet.to(self.device)
            self.denoising_unet.to(self.device)

            # 1. Lineart invertido (Fondo=0, Línea=1) para ControlNet
            gray = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)
            lineart_inv = np.clip((255.0 - cv2.bilateralFilter(gray, 5, 50, 50)) / 255.0, 0.0, 1.0)
            cond_np = np.repeat(lineart_inv[:, :, np.newaxis], 3, axis=2)
            cond_tensor = torch.from_numpy(cond_np).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)

            # 2. CFG: Duplicar tensores (uncond, cond)
            cond_tensor = torch.cat([cond_tensor, cond_tensor], dim=0)
            uncond_embeds = torch.zeros_like(image_embeds)
            image_embeds_cfg = torch.cat([uncond_embeds, image_embeds], dim=0)

            latents_shape = (1, 4, pad_h // 8, pad_w // 8)
            latents = torch.randn(latents_shape, device=self.device, dtype=self.dtype)

            self.scheduler.set_timesteps(steps, device=self.device)
            latents = latents * self.scheduler.init_noise_sigma
            guidance_scale = 2.0

            for t in self.scheduler.timesteps:
                # Expandir latentes para batch=2
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                ctrl_out: Any = self.controlnet(
                    latent_model_input,
                    timestep=t,
                    encoder_hidden_states=image_embeds_cfg,
                    controlnet_cond=cond_tensor,
                    conditioning_scale=0.35,  # Bajado para priorizar inyección de color
                    return_dict=False,
                )

                unet_out: Any = self.denoising_unet(
                    latent_model_input,
                    timestep=t,
                    encoder_hidden_states=image_embeds_cfg,
                    down_block_additional_residuals=ctrl_out[0],
                    mid_block_additional_residual=ctrl_out[1],
                    return_dict=False,
                )

                # 3. Calcular CFG
                noise_pred_uncond, noise_pred_text = unet_out[0].chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # SUB-ETAPA D: Decodificación VAE Final
            self.vae.to(self.device)
            latents_scaled = latents / scale_fac
            dec_out: Any = self.vae.decode(latents_scaled, return_dict=False)
            decoded = dec_out[0] if isinstance(dec_out, tuple) else getattr(dec_out, "sample", dec_out)
            self.vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            out_img = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).squeeze(0).permute(1, 2, 0)
            out_np = out_img.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            colored_work = out_bgr[:work_h, :work_w]
            return cv2.resize(colored_work, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        finally:
            self.bank.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload(self) -> None:
        self.bank.remove_hooks()
        if self._loaded:
            self.vae = None
            self.image_encoder = None
            self.clip_proc = None
            self.denoising_unet = None
            self.reference_unet = None
            self.controlnet = None
            self.scheduler = None
            self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
