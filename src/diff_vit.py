"""Patch-level Diff Transformer ViT for ADP comparator runs.

This model keeps ADP's raw-RGB patch pipeline intact while replacing the
ordinary ViT attention blocks with full differential-attention blocks. It is
intentionally separate from :mod:`src.vit` so baseline/XSA/SRP ADP jobs keep
their existing class and initialization path.

Why ADP uses full attention here:
    The slide-level Diff Transformer comparator uses a Nyström approximation
    because WSI bags can contain tens of thousands of tokens. ADP is a raw-RGB
    patch task with a small 17x17 token grid, so full softmax differential
    attention is both feasible and closer to the original Diff Transformer
    implementation. The only intentional adaptation is that ADP/ViT attention
    is bidirectional and non-rotary, matching the rest of this project's patch
    ViT comparators rather than the language-model causal setting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from slide_level_srp.src.diff_transformer import RMSNorm, lambda_init_fn

from .vit import DropPath, Mlp, PatchEmbed


_CHECKPOINT_MODES = ("whole_block", "per_module", "off")


class FullDifferentialAttention(nn.Module):
    """Full bidirectional differential attention for ADP-scale token grids.

    The parameterization mirrors the official Diff Transformer MHA path:
    ``num_heads`` is interpreted as the baseline Transformer's head count, while
    the Diff Transformer internally uses half as many differential heads. Each
    differential head owns two Q/K branches and a value vector of width ``2d``;
    after subtracting the two softmax maps with the learned lambda coefficient,
    the output width is therefore still the original embedding dimension.
    """

    def __init__(
        self,
        dim: int,
        *,
        depth_index: int,
        baseline_num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if baseline_num_heads % 2 != 0:
            raise ValueError(
                "Diff Transformer needs an even baseline head count so it can "
                f"use half as many differential heads; got {baseline_num_heads}"
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
        self.lambda_init = lambda_init_fn(depth_index)

        # Keep separate Q/K/V projections, as in the official implementation.
        # qkv_bias stays configurable for experiments, but ADP reported
        # Diff Transformer uses the official bias-free default.
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Official lambda parameterization. The parameters are layer-shared
        # across heads, initialized near zero, and combined as
        # exp(q1*k1) - exp(q2*k2) + lambda_init.
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

    def _softmax_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # ADP ViT is bidirectional. We intentionally do not apply the causal mask
        # used by the language-model reference implementation.
        dropout_p = self.attn_drop.p if self.training else 0.0
        try:
            # This is exact full softmax attention, not Nyström. SDPA lets PyTorch
            # choose a memory-efficient kernel so the ADP comparator does not need
            # to materialize two B x H x N x N attention matrices per block.
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                is_causal=False,
                scale=self.scale,
            )
        except TypeError:
            # Compatibility fallback for older Torch builds without the `scale`
            # keyword. It is mathematically identical but less memory efficient.
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_logits = torch.nan_to_num(attn_logits)
            attn = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
            attn = self.attn_drop(attn)
            return torch.matmul(attn, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, dim = x.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(bsz, n_tokens, h, 2, d).permute(0, 2, 3, 1, 4)
        k = self.k_proj(x).reshape(bsz, n_tokens, h, 2, d).permute(0, 2, 3, 1, 4)
        v = self.v_proj(x).reshape(bsz, n_tokens, h, 2 * d).permute(0, 2, 1, 3)
        q1 = q[:, :, 0]
        q2 = q[:, :, 1]
        k1 = k[:, :, 0]
        k2 = k[:, :, 1]

        y1 = self._softmax_attention(q1, k1, v)
        y2 = self._softmax_attention(q2, k2, v)
        diff = y1 - self.lambda_full(dtype=y1.dtype, device=y1.device) * y2
        diff = self.subln(diff)
        # Official stabilization multiplier after the per-head RMSNorm.
        diff = diff * (1.0 - self.lambda_init)
        diff = diff.transpose(1, 2).reshape(bsz, n_tokens, dim)
        out = self.out_proj(diff)
        return self.proj_drop(out)


class DiffBlock(nn.Module):
    """Pre-norm ViT block using full differential attention."""

    def __init__(
        self,
        dim: int,
        *,
        depth_index: int,
        baseline_num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        checkpoint_mode: str = "off",
    ) -> None:
        super().__init__()
        if checkpoint_mode not in _CHECKPOINT_MODES:
            raise ValueError(f"checkpoint_mode must be one of {_CHECKPOINT_MODES}")
        self.checkpoint_mode = checkpoint_mode
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FullDifferentialAttention(
            dim=dim,
            depth_index=depth_index,
            baseline_num_heads=baseline_num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
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


class DiffTransformerViT(nn.Module):
    """ViT-style image classifier whose token mixer is Diff Transformer.

    ADP is patch-level raw RGB, not a slide-level feature-bag task.  Reusing
    the slide aggregator directly would incorrectly treat a 272x272 image as a
    WSI bag and would add TransMIL PPEG behavior.  This class instead mirrors
    ``ViT``'s patch embedding, CLS token, optional absolute position embedding,
    final norm, and classifier head, while using ``DiffBlock`` for each
    transformer layer.
    """

    def __init__(
        self,
        img_size: int = 272,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 9,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_landmarks: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        pinv_iterations: int = 6,
        checkpoint_mode: str = "off",
        use_abs_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError(
                "DiffTransformerViT requires an even --num_heads because "
                f"Differential Attention uses half as many heads; got {num_heads}."
            )
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        # Kept for CLI/artifact compatibility with the WSI Diff Transformer
        # comparator. ADP uses full attention, so this value is not consumed.
        self.num_landmarks = int(num_landmarks)
        self.use_abs_pos_embed = bool(use_abs_pos_embed)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist() if depth > 0 else []
        self.blocks = nn.ModuleList(
            [
                DiffBlock(
                    dim=embed_dim,
                    depth_index=i,
                    baseline_num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    drop_path=dpr[i],
                    checkpoint_mode=checkpoint_mode,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Match the existing ViT initialization without touching lambda init."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        if self.use_abs_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x[:, 0])
