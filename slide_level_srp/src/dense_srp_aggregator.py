"""Dense-MHSA TransMIL-style aggregator for retained-patch experiments.

The classes below keep the TransMIL scaffold and Gated SRP post-attention
formula while replacing Nyström attention with exact softmax attention on a
deterministically retained patch subset.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .gate_signed import TokenHeadGate, collect_gate_module_ids
from slide_level.src.aggregator import DropPath, Mlp
from slide_level.src.ppeg import PPEG
from .srp_attention import (
    _gate_num_token_features,
    _make_token_diag,
    streaming_neighborhood_mean,
)


_CHECKPOINT_MODES = ("whole_block", "per_module", "off")


class DenseSRPAttention(nn.Module):
    """Exact full self-attention with an optional Gated SRP correction.

    ``use_srp=False`` is a true dense MHSA baseline and consumes only token
    features.  ``use_srp=True`` applies the post-attention signed
    Gated SRP update to real patch rows while leaving the CLS row and square
    padding rows unchanged. The correction formula matches the primary method:

    ``z = y - beta_eff * <y, r_hat> * r_hat``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_srp: bool = False,
        gate_active: bool = True,
        delta_scale: float = 2.0,
        gate_hidden_dim: int = 16,
        detach_gate_inputs: bool = True,
        gate_output_init: str = "zero",
        gate_output_init_scale: float = 1.0,
        gate_init_beta0: float = 0.0,
        gate_activation: str = "tanh",
        gate_activation_temperature: float = 1.0,
        gate_factorization: str = "full",
        gate_delta_mode: str = "fixed",
        gate_count_features: str = "legacy",
        retain_gate_beta_for_loss: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_srp = bool(use_srp)
        self.gate_active = bool(gate_active and use_srp)
        self.detach_gate_inputs = bool(detach_gate_inputs)
        self.gate_count_features = gate_count_features
        self.retain_gate_beta_for_loss = bool(retain_gate_beta_for_loss)
        self._last_gate_stats: dict | None = None
        self._last_gate_beta_eff_for_loss: Optional[torch.Tensor] = None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.gate_active:
            # Match the Nyström Gated SRP gate surface exactly.
            # Save/restore RNG so constructing the optional gate cannot shift
            # common backbone initialization relative to the dense-MHSA
            # baseline under the same global seed.
            rng_state_cpu = torch.get_rng_state()
            try:
                self.gate = TokenHeadGate(
                    num_token_features=_gate_num_token_features(
                        gate_count_features,
                        include_y_norm_mean=False,
                    ),
                    num_head_features=3,
                    hidden_dim=gate_hidden_dim,
                    num_heads=num_heads,
                    delta_scale=delta_scale,
                    output_init=gate_output_init,
                    output_init_scale=gate_output_init_scale,
                    init_beta0=gate_init_beta0,
                    activation=gate_activation,
                    activation_temperature=gate_activation_temperature,
                    factorization=gate_factorization,
                    delta_mode=gate_delta_mode,
                )
            finally:
                torch.set_rng_state(rng_state_cpu)
        else:
            self.gate = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        n_real: int,
        neighbor_index: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        h_local: Optional[torch.Tensor] = None,
        neighbor_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(
            bsz, seq_len, 3, self.num_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        y = attn @ v

        if self.gate_active:
            if neighbor_index is None or neighbor_mask is None or h_local is None:
                raise ValueError("dense_mhsa_srp requires neighbor_index, neighbor_mask, and h_local")
            if h_local.shape != (bsz, n_real):
                raise ValueError(f"h_local shape {tuple(h_local.shape)} != ({bsz}, {n_real})")
            y_patch = y[:, :, 1 : 1 + n_real, :]
            v_patch = v[:, :, 1 : 1 + n_real, :]
            # Use the exact streaming local mean implementation.  This avoids
            # materializing the old (B,H,N,K,D) neighbor stack while preserving
            # the method definition of r as a local value-vector mean.
            _r_det, r_hat_det, cnt = streaming_neighborhood_mean(
                v_patch.detach(), neighbor_index, neighbor_mask, neighbor_weight,
            )
            dot_yr = (y_patch * r_hat_det).sum(dim=-1, keepdim=True)
            y_norms = y_patch.norm(dim=-1)
            cnt_bn = cnt.squeeze(-1).squeeze(1).to(dtype=y.dtype)
            token_diag = _make_token_diag(
                h_local=h_local.to(y.dtype),
                cnt_bn=cnt_bn,
                max_neighbors=int(neighbor_index.shape[-1]),
                mode=self.gate_count_features,
            )
            cos_yr = dot_yr.squeeze(-1) / (y_norms + 1e-12)
            head_diag = torch.stack(
                [cos_yr, cos_yr.abs(), torch.log1p(y_norms)],
                dim=-1,
            )
            if self.detach_gate_inputs:
                token_diag = token_diag.detach()
                head_diag = head_diag.detach()
            beta_eff = self.gate(token_diag, head_diag)  # type: ignore[operator]
            if self.retain_gate_beta_for_loss:
                self._last_gate_beta_eff_for_loss = beta_eff
            z_patch = y_patch - beta_eff * dot_yr * r_hat_det
            y = y.clone()
            y[:, :, 1 : 1 + n_real, :] = z_patch
            with torch.no_grad():
                self._last_gate_stats = {
                    "beta_eff": beta_eff.detach(),
                    "delta_eff": self.gate.current_delta().detach(),  # type: ignore[union-attr]
                    "delta_mode": getattr(self.gate, "delta_mode", "fixed"),
                    "cos_yr": cos_yr.detach(),
                    "y_norms": y_norms.detach(),
                    "h_local": h_local.detach(),
                    "neighbour_count": cnt_bn.detach(),
                }
        else:
            self._last_gate_stats = None

        out = y.transpose(1, 2).reshape(bsz, seq_len, dim)
        out = self.proj(out)
        return self.proj_drop(out)


class DenseSRPBlock(nn.Module):
    """Pre-norm Transformer block used by the dense-attention backend."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        drop_path: float,
        checkpoint_mode: str,
        **attn_kwargs,
    ) -> None:
        super().__init__()
        if checkpoint_mode not in _CHECKPOINT_MODES:
            raise ValueError(f"checkpoint_mode must be one of {_CHECKPOINT_MODES}, got {checkpoint_mode!r}")
        self.checkpoint_mode = checkpoint_mode
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DenseSRPAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **attn_kwargs,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_dim=dim, hidden_dim=int(dim * mlp_ratio), drop=proj_drop)
        self.drop_path = DropPath(drop_path)

    def _forward_inner(
        self,
        x: torch.Tensor,
        n_real: int,
        neighbor_index: Optional[torch.Tensor],
        neighbor_mask: Optional[torch.Tensor],
        h_local: Optional[torch.Tensor],
        neighbor_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(
            self.norm1(x),
            n_real=n_real,
            neighbor_index=neighbor_index,
            neighbor_mask=neighbor_mask,
            h_local=h_local,
            neighbor_weight=neighbor_weight,
        ))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(
        self,
        x: torch.Tensor,
        *,
        n_real: int,
        neighbor_index: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        h_local: Optional[torch.Tensor] = None,
        neighbor_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.checkpoint_mode == "off" or not self.training:
            return self._forward_inner(x, n_real, neighbor_index, neighbor_mask, h_local, neighbor_weight)
        if self.checkpoint_mode == "whole_block":
            return cp.checkpoint(
                lambda t: self._forward_inner(
                    t, n_real, neighbor_index, neighbor_mask, h_local, neighbor_weight,
                ),
                x,
                use_reentrant=False,
            )
        attn_out = cp.checkpoint(
            lambda t: self.attn(
                self.norm1(t),
                n_real=n_real,
                neighbor_index=neighbor_index,
                neighbor_mask=neighbor_mask,
                h_local=h_local,
                neighbor_weight=neighbor_weight,
            ),
            x,
            use_reentrant=False,
        )
        x = x + self.drop_path(attn_out)
        mlp_out = cp.checkpoint(lambda t: self.mlp(self.norm2(t)), x, use_reentrant=False)
        return x + self.drop_path(mlp_out)


class DenseAttentionSRPAggregator(nn.Module):
    """TransMIL-style exact-attention aggregator for capped WSI bags."""

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        num_classes: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        checkpoint_mode: str = "whole_block",
        use_srp: bool = False,
        delta_scale: float = 2.0,
        gate_hidden_dim: int = 16,
        detach_gate_inputs: bool = True,
        gate_output_init: str = "zero",
        gate_output_init_scale: float = 1.0,
        gate_init_beta0: float = 0.0,
        gate_activation: str = "tanh",
        gate_activation_temperature: float = 1.0,
        gate_factorization: str = "full",
        gate_delta_mode: str = "fixed",
        gate_count_features: str = "legacy",
        retain_gate_beta_for_loss: bool = False,
        use_ppeg: bool = True,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_srp = bool(use_srp)
        self.use_ppeg = bool(use_ppeg)
        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist() if depth > 0 else []
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Match the post-attention placement: the final block is
            # left inactive because its patch-row update is not consumed by the
            # CLS-only slide head.
            gate_active = self.use_srp and i < max(depth - 1, 0)
            self.blocks.append(DenseSRPBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                drop_path=dpr[i],
                checkpoint_mode=checkpoint_mode,
                use_srp=self.use_srp,
                gate_active=gate_active,
                delta_scale=delta_scale,
                gate_hidden_dim=gate_hidden_dim,
                detach_gate_inputs=detach_gate_inputs,
                gate_output_init=gate_output_init,
                gate_output_init_scale=gate_output_init_scale,
                gate_init_beta0=gate_init_beta0,
                gate_activation=gate_activation,
                gate_activation_temperature=gate_activation_temperature,
                gate_factorization=gate_factorization,
                gate_delta_mode=gate_delta_mode,
                gate_count_features=gate_count_features,
                retain_gate_beta_for_loss=retain_gate_beta_for_loss,
            ))
        self.ppeg = PPEG(dim=embed_dim) if self.use_ppeg else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        method_module_ids = collect_gate_module_ids(self)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if id(module) in method_module_ids:
                continue
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, TokenHeadGate):
                module.reset_output_path()

    def forward(
        self,
        features: torch.Tensor,
        neighbor_index: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        h_morph: Optional[torch.Tensor] = None,
        h_local: Optional[torch.Tensor] = None,
        neighbor_weight: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        del h_morph  # Dense signed-gated SRP uses the local-similarity feature.
        bsz, n_real, _ = features.shape
        x = self.in_proj(features)
        h = w = int(math.ceil(math.sqrt(n_real)))
        hw = h * w
        add = hw - n_real
        if add > 0:
            x = torch.cat([x, x[:, :add, :]], dim=1)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                n_real=n_real,
                neighbor_index=neighbor_index,
                neighbor_mask=neighbor_mask,
                h_local=h_local,
                neighbor_weight=neighbor_weight,
            )
            if i == 0 and self.use_ppeg:
                x = self.ppeg(x, h, w)
        x = self.norm(x)
        return self.head(x[:, 0])
