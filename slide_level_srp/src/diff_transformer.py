"""Diff Transformer comparator for slide-level MIL.

This module implements the Differential Transformer attention mechanism from
Ye et al. (ICLR 2025) inside the same TransMIL-style scaffold used by the
project's baseline, XSA, and gated-SRP runs.

Implementation note:
    The original Diff Transformer computes two full softmax attention maps and
    subtracts them.  Full N x N attention is not safe for WSI bags with tens of
    thousands of patch tokens, so this comparator applies the same differential
    operation to two Nyström attention approximations.  This keeps the paper's
    Q1/K1 versus Q2/K2 denoising mechanism, lambda parameterization, per-head
    RMSNorm, and half-head compute convention while matching the memory budget
    of the rest of this project.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from slide_level.src.aggregator import DropPath, Mlp
from slide_level.src.ppeg import PPEG
from slide_level.src.nystrom_xsa import moore_penrose_iter_inv


_CHECKPOINT_MODES = ("whole_block", "per_module", "off")


def lambda_init_fn(depth: int) -> float:
    """Official Diff Transformer depth-dependent lambda initialization."""
    return float(0.8 - 0.6 * math.exp(-0.3 * depth))


def _pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, int]:
    """Pad a (..., N, D) tensor at the end so N is divisible by m."""
    n_tokens = x.shape[-2]
    remainder = n_tokens % m
    if remainder == 0:
        return x, 0
    pad = m - remainder
    return F.pad(x, (0, 0, 0, pad)), pad


class RMSNorm(nn.Module):
    """Minimal RMSNorm used for Diff Transformer's per-head normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize only over the per-head channel dimension.  This mirrors the
        # paper's GroupNorm/RMSNorm intent: heads keep independent statistics
        # after the differential subtraction, which can otherwise yield much
        # more diverse head scales than ordinary attention.
        inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x.float() * inv_rms
        return out.to(dtype=x.dtype) * self.weight.to(device=x.device, dtype=x.dtype)


def nystrom_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_landmarks: int,
    pinv_iterations: int,
    attn_drop: nn.Module,
) -> torch.Tensor:
    """Nyström approximation of one softmax(QK^T)V map.

    Shapes:
      q, k: (B, H, N, D)
      v:    (B, H, N, Dv)

    This mirrors ``slide_level.src.nystrom_xsa.NystromXSAAttention`` so the
    comparator differs from the baseline by the Diff Transformer attention
    mechanism, not by a separate sequence-scaling implementation.
    """
    bsz, n_heads, n_tokens, _ = q.shape
    m = int(num_landmarks)
    if m <= 0:
        raise ValueError(f"num_landmarks must be positive, got {num_landmarks}")

    q_pad, pad = _pad_to_multiple(q, m)
    k_pad, _ = _pad_to_multiple(k, m)
    v_pad, _ = _pad_to_multiple(v, m)
    n_padded = q_pad.shape[-2]
    seg = n_padded // m

    if pad > 0:
        mask = torch.ones(bsz, 1, n_padded, 1, device=q.device, dtype=torch.bool)
        mask[..., n_tokens:, :] = False
    else:
        mask = None

    def seg_mean(t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(bsz, n_heads, m, seg, t.shape[-1])
        if mask is None:
            return t.mean(dim=3)
        mseg = mask.reshape(bsz, 1, m, seg, 1).to(t.dtype)
        summed = (t * mseg).sum(dim=3)
        count = mseg.sum(dim=3).clamp(min=1e-6)
        return summed / count

    q_tilde = seg_mean(q_pad)
    k_tilde = seg_mean(k_pad)

    factor_logits = q_pad @ k_tilde.transpose(-2, -1)
    landmark_logits = q_tilde @ k_tilde.transpose(-2, -1)
    value_logits = q_tilde @ k_pad.transpose(-2, -1)
    if pad > 0:
        value_mask = mask.reshape(bsz, 1, 1, n_padded)
        value_logits = value_logits.masked_fill(~value_mask, float("-inf"))

    factor = attn_drop(factor_logits.softmax(dim=-1))
    landmark = landmark_logits.softmax(dim=-1)
    value = attn_drop(value_logits.softmax(dim=-1))

    landmark_inv = moore_penrose_iter_inv(landmark, iters=pinv_iterations)
    value_v = value @ v_pad
    y_pad = factor @ (landmark_inv @ value_v)
    return y_pad[..., :n_tokens, :]


def nystrom_attention_with_cls_patch_row(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_landmarks: int,
    pinv_iterations: int,
    attn_drop: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Nyström attention output plus the real CLS-to-patch row.

    Diff Transformer's faithful heatmap score is the signed differential row
    ``A1_cls,patch - lambda * A2_cls,patch``.  Computing that row requires the
    branch-level CLS-to-patch attention before multiplying by V.  This helper
    mirrors :func:`nystrom_attention` but returns only that single row, avoiding
    the full WSI-scale ``N x N`` attention matrix.
    """
    bsz, n_heads, n_tokens, _ = q.shape
    m = int(num_landmarks)
    if m <= 0:
        raise ValueError(f"num_landmarks must be positive, got {num_landmarks}")

    q_pad, pad = _pad_to_multiple(q, m)
    k_pad, _ = _pad_to_multiple(k, m)
    v_pad, _ = _pad_to_multiple(v, m)
    n_padded = q_pad.shape[-2]
    seg = n_padded // m

    if pad > 0:
        mask = torch.ones(bsz, 1, n_padded, 1, device=q.device, dtype=torch.bool)
        mask[..., n_tokens:, :] = False
    else:
        mask = None

    def seg_mean(t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(bsz, n_heads, m, seg, t.shape[-1])
        if mask is None:
            return t.mean(dim=3)
        mseg = mask.reshape(bsz, 1, m, seg, 1).to(t.dtype)
        summed = (t * mseg).sum(dim=3)
        count = mseg.sum(dim=3).clamp(min=1e-6)
        return summed / count

    q_tilde = seg_mean(q_pad)
    k_tilde = seg_mean(k_pad)

    factor_logits = q_pad @ k_tilde.transpose(-2, -1)
    landmark_logits = q_tilde @ k_tilde.transpose(-2, -1)
    value_logits = q_tilde @ k_pad.transpose(-2, -1)
    if pad > 0:
        value_mask = mask.reshape(bsz, 1, 1, n_padded)
        value_logits = value_logits.masked_fill(~value_mask, float("-inf"))

    factor = attn_drop(factor_logits.softmax(dim=-1))
    landmark = landmark_logits.softmax(dim=-1)
    value = attn_drop(value_logits.softmax(dim=-1))

    landmark_inv = moore_penrose_iter_inv(landmark, iters=pinv_iterations)
    value_v = value @ v_pad
    y_pad = factor @ (landmark_inv @ value_v)

    # CLS is always token 0 in the TransMIL-style bags. Patch columns are
    # 1..N-1; padded columns are excluded by slicing before returning.
    cls_row = torch.einsum(
        "bhm,bhmn->bhn",
        factor[:, :, 0, :],
        landmark_inv @ value,
    )
    cls_patch_row = cls_row[:, :, 1:n_tokens].detach()
    return y_pad[..., :n_tokens, :], cls_patch_row


class NystromDifferentialAttention(nn.Module):
    """Multi-head differential attention with Nyström softmax maps.

    ``baseline_num_heads`` is the ordinary Transformer head count used by the
    baseline run.  Diff Transformer uses half that number of differential
    heads while each head carries two Q/K branches and a 2d value vector, which
    preserves the embedding width and tracks the official compute convention.
    """

    def __init__(
        self,
        dim: int,
        *,
        depth_index: int,
        baseline_num_heads: int = 6,
        num_landmarks: int = 64,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pinv_iterations: int = 6,
    ) -> None:
        super().__init__()
        if baseline_num_heads % 2 != 0:
            raise ValueError(
                "Diff Transformer needs an even baseline head count so it "
                f"can use half as many differential heads; got {baseline_num_heads}"
            )
        self.dim = int(dim)
        self.baseline_num_heads = int(baseline_num_heads)
        self.num_heads = self.baseline_num_heads // 2
        if self.num_heads <= 0 or dim % (2 * self.num_heads) != 0:
            raise ValueError(
                f"dim={dim} must be divisible by 2 * diff_heads={2 * self.num_heads}"
            )
        self.head_dim = dim // (2 * self.num_heads)
        self.scale = self.head_dim ** -0.5
        self.num_landmarks = int(num_landmarks)
        self.pinv_iterations = int(pinv_iterations)
        self.lambda_init = lambda_init_fn(depth_index)
        self._capture_stats = False
        self.last_stats: dict | None = None

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Official parameterization keeps lambda vectors shared across heads in
        # a layer.  The exponentials start close to one another, so lambda_full
        # starts near lambda_init and then learns the relative subtraction
        # strength.
        self.lambda_q1 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32))
        self.lambda_k1 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32))
        self.lambda_q2 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32))
        self.lambda_k2 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32))
        for param in (self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2):
            nn.init.normal_(param, mean=0.0, std=0.1)
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

    def lambda_full(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        return (lambda_1 - lambda_2 + self.lambda_init).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, dim = x.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(bsz, n_tokens, h, 2, d).permute(0, 2, 3, 1, 4)
        k = self.k_proj(x).reshape(bsz, n_tokens, h, 2, d).permute(0, 2, 3, 1, 4)
        v = self.v_proj(x).reshape(bsz, n_tokens, h, 2 * d).permute(0, 2, 1, 3)
        q1 = q[:, :, 0] * self.scale
        q2 = q[:, :, 1] * self.scale
        k1 = k[:, :, 0]
        k2 = k[:, :, 1]

        if self._capture_stats:
            y1, cls_patch_1 = nystrom_attention_with_cls_patch_row(
                q1,
                k1,
                v,
                num_landmarks=self.num_landmarks,
                pinv_iterations=self.pinv_iterations,
                attn_drop=self.attn_drop,
            )
            y2, cls_patch_2 = nystrom_attention_with_cls_patch_row(
                q2,
                k2,
                v,
                num_landmarks=self.num_landmarks,
                pinv_iterations=self.pinv_iterations,
                attn_drop=self.attn_drop,
            )
        else:
            y1 = nystrom_attention(
                q1,
                k1,
                v,
                num_landmarks=self.num_landmarks,
                pinv_iterations=self.pinv_iterations,
                attn_drop=self.attn_drop,
            )
            y2 = nystrom_attention(
                q2,
                k2,
                v,
                num_landmarks=self.num_landmarks,
                pinv_iterations=self.pinv_iterations,
                attn_drop=self.attn_drop,
            )
            cls_patch_1 = cls_patch_2 = None

        lambda_full = self.lambda_full(dtype=y1.dtype, device=y1.device)
        diff = y1 - lambda_full * y2
        if self._capture_stats:
            # This signed row is the actual differential attention coefficient
            # multiplying V before RMSNorm/output projection.  It is not a
            # probability distribution and can be negative, so exporters should
            # use a diverging visualization for the main Diff heatmap.
            assert cls_patch_1 is not None and cls_patch_2 is not None
            diff_cls_patch_attn = cls_patch_1 - lambda_full * cls_patch_2
            self.last_stats = {
                "cls_patch_attn_signed": diff_cls_patch_attn.detach(),
                "cls_patch_attn_branch1": cls_patch_1.detach(),
                "cls_patch_attn_branch2": cls_patch_2.detach(),
                "lambda_full": lambda_full.detach().reshape(1),
                "num_cls_tokens": 1,
            }
        diff = self.subln(diff)
        # This fixed multiplier is part of the official stabilization recipe.
        diff = diff * (1.0 - self.lambda_init)
        diff = diff.transpose(1, 2).reshape(bsz, n_tokens, dim)
        out = self.out_proj(diff)
        return self.proj_drop(out)


class DiffBlock(nn.Module):
    """Pre-norm TransMIL block using differential attention."""

    def __init__(
        self,
        dim: int,
        *,
        depth_index: int,
        baseline_num_heads: int,
        num_landmarks: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pinv_iterations: int = 6,
        drop_path: float = 0.0,
        checkpoint_mode: str = "whole_block",
    ) -> None:
        super().__init__()
        if checkpoint_mode not in _CHECKPOINT_MODES:
            raise ValueError(f"checkpoint_mode must be one of {_CHECKPOINT_MODES}")
        self.checkpoint_mode = checkpoint_mode
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NystromDifferentialAttention(
            dim=dim,
            depth_index=depth_index,
            baseline_num_heads=baseline_num_heads,
            num_landmarks=num_landmarks,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            pinv_iterations=pinv_iterations,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_dim=dim, hidden_dim=int(dim * mlp_ratio), drop=proj_drop)
        self.drop_path = DropPath(drop_path)

    def _attn_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm1(x))

    def _mlp_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm2(x))

    def _forward_inner(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self._attn_branch(x))
        x = x + self.drop_path(self._mlp_branch(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.checkpoint_mode
        if mode == "off" or not self.training:
            return self._forward_inner(x)
        if mode == "whole_block":
            return cp.checkpoint(self._forward_inner, x, use_reentrant=False)
        attn_out = cp.checkpoint(self._attn_branch, x, use_reentrant=False)
        x = x + self.drop_path(attn_out)
        mlp_out = cp.checkpoint(self._mlp_branch, x, use_reentrant=False)
        return x + self.drop_path(mlp_out)


class NystromDiffTransformerAggregator(nn.Module):
    """TransMIL-style slide aggregator with Diff Transformer attention."""

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        num_landmarks: int = 64,
        num_classes: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        pinv_iterations: int = 6,
        checkpoint_mode: str = "whole_block",
        use_ppeg: bool = True,
    ) -> None:
        super().__init__()
        if checkpoint_mode not in _CHECKPOINT_MODES:
            raise ValueError(f"checkpoint_mode must be one of {_CHECKPOINT_MODES}")
        if num_heads % 2 != 0:
            raise ValueError(
                f"Diff Transformer requires even --num_heads; got {num_heads}"
            )
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.diff_num_heads = num_heads // 2
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.num_cls_tokens = 1
        self.checkpoint_mode = checkpoint_mode
        self.use_ppeg = bool(use_ppeg)

        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist() if depth > 0 else []
        self.blocks = nn.ModuleList(
            [
                DiffBlock(
                    dim=embed_dim,
                    depth_index=i,
                    baseline_num_heads=num_heads,
                    num_landmarks=num_landmarks,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    pinv_iterations=pinv_iterations,
                    drop_path=dpr[i],
                    checkpoint_mode=checkpoint_mode,
                )
                for i in range(depth)
            ]
        )
        self.ppeg = PPEG(dim=embed_dim) if self.use_ppeg else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Match the project-wide TransMIL initialization convention."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        diff_lambda_param_ids = {
            id(param)
            for module in self.modules()
            if isinstance(module, NystromDifferentialAttention)
            for param in (module.lambda_q1, module.lambda_k1, module.lambda_q2, module.lambda_k2)
        }
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # The global init above intentionally leaves official lambda vectors
        # alone: they are not Linear/Norm weights, and their N(0, 0.1) init was
        # done inside NystromDifferentialAttention.  This assertion protects
        # that convention from future broad initializer changes.
        current_lambda_param_ids = {
            id(param)
            for module in self.modules()
            if isinstance(module, NystromDifferentialAttention)
            for param in (module.lambda_q1, module.lambda_k1, module.lambda_q2, module.lambda_k2)
        }
        if current_lambda_param_ids != diff_lambda_param_ids:
            raise RuntimeError("Diff Transformer lambda parameter registration changed during init")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, _ = features.shape
        x = self.in_proj(features)

        h = w = int(math.ceil(math.sqrt(n_tokens)))
        add = h * w - n_tokens
        if add > 0:
            x = torch.cat([x, x[:, :add, :]], dim=1)

        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.blocks[0](x)
        if self.use_ppeg:
            x = self.ppeg(x, h, w)
        for block in self.blocks[1:]:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])
