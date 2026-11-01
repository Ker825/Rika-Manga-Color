from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Capa de atención multi-cabezal con soporte para proyección desacoplada K/V."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
    ) -> None:
        super().__init__()
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.heads = heads
        self.scale = dim_head**-0.5
        self.upcast_attention = upcast_attention

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        self.to_out = nn.ModuleList(
            [
                nn.Linear(self.inner_dim, query_dim, bias=True),
                nn.Dropout(dropout),
            ]
        )
        self.processor: AttnProcessor2_0 = AttnProcessor2_0()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_v: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_v=encoder_hidden_states_v,
            attention_mask=attention_mask,
            **kwargs,
        )


class AttnProcessor2_0:
    """Procesador SDPA nativo optimizado con desacoplamiento de tensor V."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_v: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        enc_states = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        key = attn.to_k(enc_states)

        enc_states_v = (
            encoder_hidden_states_v
            if encoder_hidden_states_v is not None
            else enc_states
        )
        value = attn.to_v(enc_states_v)

        head_dim = attn.inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        resolved_mask = attention_mask
        if resolved_mask is not None and resolved_mask.ndim == 2:
            resolved_mask = resolved_mask.unsqueeze(1).unsqueeze(2)

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=resolved_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out = (
            out.transpose(1, 2).reshape(batch_size, -1, attn.inner_dim).to(query.dtype)
        )

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out
