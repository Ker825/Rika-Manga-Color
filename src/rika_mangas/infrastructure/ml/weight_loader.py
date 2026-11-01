from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rika_mangas.infrastructure.security import safe_load_checkpoint

logger = logging.getLogger("rika_mangas.weight_loader")


def load_model_weights(
    model: nn.Module,
    checkpoint_path: Path,
    expected_hash: str | None = None,
    ignore_output_layers: bool = False,
) -> None:
    """Carga y mapea pesos de checkpoint adaptando la topología de la UNet."""
    state_dict: dict[str, Any] = safe_load_checkpoint(checkpoint_path, expected_hash)

    # Extraer el diccionario si viene empaquetado
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    sanitized_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        # Remover prefijos de envoltura comunes
        clean_key = key
        for prefix in ("module.", "unet.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]

        # Descartar capas reconstructivas si el modelo actúa como extractor de referencia
        if ignore_output_layers and any(
            clean_key.startswith(layer) for layer in ("conv_norm_out.", "conv_out.")
        ):
            continue

        sanitized_dict[clean_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(sanitized_dict, strict=False)

    # Validar que no falten capas estructurales internas críticas
    critical_missing = [
        k
        for k in missing_keys
        if not (ignore_output_layers and ("conv_norm_out" in k or "conv_out" in k))
    ]
    if critical_missing:
        raise ValueError(
            f"Faltan {len(critical_missing)} claves criticas al cargar {checkpoint_path.name}: "
            f"{critical_missing[:5]}..."
        )

    if unexpected_keys:
        logger.debug(
            "Claves descartadas al cargar %s: %s",
            checkpoint_path.name,
            unexpected_keys[:5],
        )
