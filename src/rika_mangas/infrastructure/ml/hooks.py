from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock
from einops import rearrange

from rika_mangas.infrastructure.ml.models.attention_processor import AttnProcessor2_0

logger = logging.getLogger("rika_mangas.hooks")


def _torch_dfs(model: Any) -> list[nn.Module]:
    """Recorrido DFS recursivo seguro para recopilar submódulos PyTorch."""
    if not isinstance(model, nn.Module):
        return []
    result: list[nn.Module] = [model]
    for child in model.children():
        result.extend(_torch_dfs(child))
    return result


def _get_block_channel_dim(block: BasicTransformerBlock) -> int:
    """Extrae la dimensión del canal del bloque transformer para ordenamiento consistente."""
    norm = getattr(block, "norm1", None)
    if norm is not None:
        shape = getattr(norm, "normalized_shape", None)
        if isinstance(shape, (tuple, list)) and len(shape) > 0:
            return int(shape[0])
        if isinstance(shape, int):
            return shape
    dim = getattr(block, "dim", 0)
    return int(dim) if isinstance(dim, int) else 0


def _filter_matrices_by_size(
    matrix_list: list[torch.Tensor] | None, reference_matrix: torch.Tensor
) -> list[torch.Tensor]:
    """Filtra embeddings espaciales cuya resolución y canales coinciden con el bloque transformer."""
    if not matrix_list:
        return []
    ref_shape = reference_matrix.shape[-2:]
    return [m for m in matrix_list if m.shape[-2:] == ref_shape]


class ReferenceAttentionControl:
    """Controlador de Mutual Self-Attention con soporte para Point Hints y desacoplamiento K/V."""

    def __init__(
        self,
        unet: nn.Module,
        mode: str = "write",
        do_classifier_free_guidance: bool = False,
        fusion_blocks: str = "full",
    ) -> None:
        if mode not in ("read", "write"):
            raise ValueError("mode debe ser 'read' o 'write'")
        if fusion_blocks not in ("midup", "full"):
            raise ValueError("fusion_blocks debe ser 'midup' o 'full'")

        self.unet = unet
        self.mode = mode
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.fusion_blocks = fusion_blocks
        self._original_forwards: dict[int, Callable[..., Any]] = {}
        self._original_processors: dict[int, Any] = {}

        self.register_reference_hooks()

    def register_reference_hooks(self) -> None:
        """Registra los hooks de intercepción en los bloques transformer."""
        if self.fusion_blocks == "midup":
            mid = getattr(self.unet, "mid_block", None)
            up = getattr(self.unet, "up_blocks", None)
            raw_modules = _torch_dfs(mid) + _torch_dfs(up)
        else:
            raw_modules = _torch_dfs(self.unet)

        attn_modules = [m for m in raw_modules if isinstance(m, BasicTransformerBlock)]
        attn_modules = sorted(attn_modules, key=_get_block_channel_dim, reverse=True)

        mode = self.mode
        do_cfg = self.do_classifier_free_guidance

        def hacked_forward(
            module: BasicTransformerBlock,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            timestep: torch.LongTensor | None = None,
            cross_attention_kwargs: dict[str, Any] | None = None,
            class_labels: torch.LongTensor | None = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            norm_hidden_states = module.norm1(hidden_states)
            attn_kwargs = cross_attention_kwargs or {}

            if mode == "write":
                bank = cast(list[torch.Tensor], getattr(module, "bank", []))
                bank.append(norm_hidden_states.clone())
                attn_output = module.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=(
                        encoder_hidden_states if module.only_cross_attention else None
                    ),
                    attention_mask=attention_mask,
                    **attn_kwargs,
                )
                hidden_states = attn_output + hidden_states

            elif mode == "read":
                read_bank = cast(list[torch.Tensor], getattr(module, "bank", []))
                if read_bank:
                    cur_bs = norm_hidden_states.shape[0]
                    first_ref = read_bank[0]
                    # Expandir referencia para asegurar coincidencia de longitud con el batch actual
                    if first_ref.shape[0] == 1:
                        ref_feat = first_ref.expand(cur_bs, -1, -1)
                    elif first_ref.shape[0] >= cur_bs:
                        ref_feat = first_ref[:cur_bs]
                    else:
                        ref_feat = first_ref.repeat(cur_bs, 1, 1)[:cur_bs]

                    if do_cfg and cur_bs >= 3:
                        hidden_states_c = hidden_states.clone()

                        # Incondicional (Solo self-attention sobre sí mismo)
                        mask_0 = (
                            attention_mask[0:1] if attention_mask is not None else None
                        )
                        hidden_states_c[0:1] = (
                            module.attn1(
                                norm_hidden_states[0:1],
                                encoder_hidden_states=norm_hidden_states[0:1],
                                attention_mask=mask_0,
                                **attn_kwargs,
                            )
                            + hidden_states[0:1]
                        )

                        # Referencia (Mutual attention con la referencia, sin hints de puntos)
                        modify_k_ref = torch.cat(
                            [norm_hidden_states[1:2], ref_feat[1:2]], dim=1
                        )
                        mask_1 = (
                            attention_mask[1:2] if attention_mask is not None else None
                        )
                        hidden_states_c[1:2] = (
                            module.attn1(
                                norm_hidden_states[1:2],
                                encoder_hidden_states=modify_k_ref,
                                attention_mask=mask_1,
                                **attn_kwargs,
                            )
                            + hidden_states[1:2]
                        )

                        # Point Hints (Mutual attention inyectando embeddings espaciales en Q y K)
                        p_main = cast(
                            list[torch.Tensor],
                            getattr(module, "point_bank_main", []),
                        )
                        p_ref = cast(
                            list[torch.Tensor],
                            getattr(module, "point_bank_ref", []),
                        )
                        mask_2 = (
                            attention_mask[2:3] if attention_mask is not None else None
                        )

                        if p_main and p_ref:
                            main_emb = p_main[0]
                            ref_emb = p_ref[0]
                            if main_emb.shape[0] != 1:
                                main_emb = main_emb[0:1]
                            if ref_emb.shape[0] != 1:
                                ref_emb = ref_emb[0:1]

                            q_pts = norm_hidden_states[2:3] + main_emb
                            k_pts = torch.cat([q_pts, ref_feat[2:3] + ref_emb], dim=1)
                            v_pts = torch.cat(
                                [norm_hidden_states[2:3], ref_feat[2:3]], dim=1
                            )

                            hidden_states_c[2:3] = (
                                module.attn1(
                                    q_pts,
                                    encoder_hidden_states=k_pts,
                                    encoder_hidden_states_v=v_pts,
                                    attention_mask=mask_2,
                                    **attn_kwargs,
                                )
                                + hidden_states[2:3]
                            )
                        else:
                            hidden_states_c[2:3] = (
                                module.attn1(
                                    norm_hidden_states[2:3],
                                    encoder_hidden_states=modify_k_ref.clone(),
                                    attention_mask=mask_2,
                                    **attn_kwargs,
                                )
                                + hidden_states[2:3]
                            )

                        hidden_states = hidden_states_c

                    elif do_cfg and cur_bs == 2:
                        hidden_states_c = hidden_states.clone()

                        # Incondicional
                        mask_0 = (
                            attention_mask[0:1] if attention_mask is not None else None
                        )
                        hidden_states_c[0:1] = (
                            module.attn1(
                                norm_hidden_states[0:1],
                                encoder_hidden_states=norm_hidden_states[0:1],
                                attention_mask=mask_0,
                                **attn_kwargs,
                            )
                            + hidden_states[0:1]
                        )

                        # Referencia
                        modify_k_ref = torch.cat(
                            [norm_hidden_states[1:2], ref_feat[1:2]], dim=1
                        )
                        mask_1 = (
                            attention_mask[1:2] if attention_mask is not None else None
                        )
                        hidden_states_c[1:2] = (
                            module.attn1(
                                norm_hidden_states[1:2],
                                encoder_hidden_states=modify_k_ref,
                                attention_mask=mask_1,
                                **attn_kwargs,
                            )
                            + hidden_states[1:2]
                        )
                        hidden_states = hidden_states_c

                    else:
                        # Modo directo sin CFG o batch simple
                        p_main = cast(
                            list[torch.Tensor],
                            getattr(module, "point_bank_main", []),
                        )
                        p_ref = cast(
                            list[torch.Tensor],
                            getattr(module, "point_bank_ref", []),
                        )
                        if p_main and p_ref:
                            main_emb = p_main[0].expand(cur_bs, -1, -1)
                            ref_emb = p_ref[0].expand(cur_bs, -1, -1)
                            q_pts = norm_hidden_states + main_emb
                            k_pts = torch.cat([q_pts, ref_feat + ref_emb], dim=1)
                            v_pts = torch.cat([norm_hidden_states, ref_feat], dim=1)
                            hidden_states = (
                                module.attn1(
                                    q_pts,
                                    encoder_hidden_states=k_pts,
                                    encoder_hidden_states_v=v_pts,
                                    attention_mask=attention_mask,
                                    **attn_kwargs,
                                )
                                + hidden_states
                            )
                        else:
                            modify_k_ref = torch.cat(
                                [norm_hidden_states, ref_feat], dim=1
                            )
                            hidden_states = (
                                module.attn1(
                                    norm_hidden_states,
                                    encoder_hidden_states=modify_k_ref,
                                    attention_mask=attention_mask,
                                    **attn_kwargs,
                                )
                                + hidden_states
                            )
                else:
                    hidden_states = (
                        module.attn1(
                            norm_hidden_states,
                            attention_mask=attention_mask,
                            **attn_kwargs,
                        )
                        + hidden_states
                    )

            if module.attn2 is not None:
                attn_output = module.attn2(
                    module.norm2(hidden_states),
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **attn_kwargs,
                )
                hidden_states = attn_output + hidden_states

            return module.ff(module.norm3(hidden_states)) + hidden_states

        for module in attn_modules:
            mod_id = id(module)
            if mod_id not in self._original_forwards:
                self._original_forwards[mod_id] = module.forward
            if hasattr(module, "attn1") and hasattr(module.attn1, "processor"):
                if mod_id not in self._original_processors:
                    self._original_processors[mod_id] = module.attn1.processor
                setattr(module.attn1, "processor", AttnProcessor2_0())

            setattr(module, "bank", [])
            setattr(module, "point_bank_ref", [])
            setattr(module, "point_bank_main", [])
            object.__setattr__(
                module, "forward", hacked_forward.__get__(module, BasicTransformerBlock)
            )

    def update(
        self,
        writer: ReferenceAttentionControl,
        point_embedding_ref: list[torch.Tensor] | None = None,
        point_embedding_main: list[torch.Tensor] | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Copia el banco de características del escritor al lector y enlaza embeddings de puntos."""
        reader_modules = [
            m
            for m in (
                _torch_dfs(getattr(self.unet, "mid_block", None))
                + _torch_dfs(getattr(self.unet, "up_blocks", None))
                if self.fusion_blocks == "midup"
                else _torch_dfs(self.unet)
            )
            if isinstance(m, BasicTransformerBlock)
        ]
        writer_modules = [
            m
            for m in (
                _torch_dfs(getattr(writer.unet, "mid_block", None))
                + _torch_dfs(getattr(writer.unet, "up_blocks", None))
                if writer.fusion_blocks == "midup"
                else _torch_dfs(writer.unet)
            )
            if isinstance(m, BasicTransformerBlock)
        ]

        reader_modules = sorted(
            reader_modules, key=_get_block_channel_dim, reverse=True
        )
        writer_modules = sorted(
            writer_modules, key=_get_block_channel_dim, reverse=True
        )

        if len(reader_modules) != len(writer_modules):
            raise ValueError(
                f"Discrepancia en cantidad de bloques de atencion: "
                f"Reader={len(reader_modules)}, Writer={len(writer_modules)}"
            )

        for r, w in zip(reader_modules, writer_modules, strict=True):
            source_bank = cast(list[torch.Tensor], getattr(w, "bank", []))
            cloned_bank = [v.clone().to(dtype) for v in source_bank]
            setattr(r, "bank", cloned_bank)

            if (
                point_embedding_main is not None
                and point_embedding_ref is not None
                and cloned_bank
            ):
                ref_mat = cloned_bank[0]
                setattr(
                    r,
                    "point_bank_ref",
                    _filter_matrices_by_size(point_embedding_ref, ref_mat),
                )
                setattr(
                    r,
                    "point_bank_main",
                    _filter_matrices_by_size(point_embedding_main, ref_mat),
                )

    def clear(self) -> None:
        """Limpia los buffers de memoria de atención sin desregistrar los hooks."""
        modules = [
            m for m in _torch_dfs(self.unet) if isinstance(m, BasicTransformerBlock)
        ]
        for m in modules:
            bank = getattr(m, "bank", None)
            if isinstance(bank, list):
                bank.clear()
            p_ref = getattr(m, "point_bank_ref", None)
            if isinstance(p_ref, list):
                p_ref.clear()
            p_main = getattr(m, "point_bank_main", None)
            if isinstance(p_main, list):
                p_main.clear()

    def remove_hooks(self) -> None:
        """Restaura los métodos forward y procesadores originales para evitar fugas."""
        self.clear()
        modules = [
            m for m in _torch_dfs(self.unet) if isinstance(m, BasicTransformerBlock)
        ]
        for m in modules:
            mod_id = id(m)
            if mod_id in self._original_forwards:
                object.__setattr__(m, "forward", self._original_forwards[mod_id])
            if (
                mod_id in self._original_processors
                and hasattr(m, "attn1")
                and hasattr(m.attn1, "processor")
            ):
                m.attn1.processor = self._original_processors[mod_id]

        self._original_forwards.clear()
        self._original_processors.clear()
