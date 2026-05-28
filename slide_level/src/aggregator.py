"""
NystromXSAggregator: TransMIL-style MIL aggregator with Nystrom-XSA attention.

Architecture (the released protocol):
    Input (B, N, 1024) UNI v1 features
      |
      v
    [Linear(1024 -> 384)]                     input projection
      |
      v
    [pad N to next perfect square via feature self-repeat]
      |
      v
    [prepend learned CLS token]               -> (B, 1 + H*W, 384)
      |
      v
    TransLayer 1   (Nystrom-XSA + MLP, pre-norm, drop-path)
      |
      v
    PPEG (conv-based positional, once after block 1, TransMIL-faithful)
      |
      v
    TransLayer 2
    TransLayer 3
    TransLayer 4
      |
      v
    LayerNorm -> CLS[0] -> Linear(384 -> K)   K = num_classes logits
                                              (K=4 in the main sweep;
                                              K=2 for ad-hoc binary)

Each TransLayer is a pre-norm transformer block:
    x' = x + DropPath(Attn(LN(x)))
    x'' = x' + DropPath(MLP(LN(x')))

Activation checkpointing (by design):
    Three modes selected by `checkpoint_mode`:
      "whole_block" (default): one checkpoint boundary per TransLayer
          that wraps (Attn + residual + MLP + residual). Peak memory
          during backward ≈ one TransLayer's activations.
      "per_module": two boundaries per TransLayer (Attn submodule,
          MLP submodule separately). Lower peak; ~15-20% more recompute.
      "off": no checkpointing. For speed benchmarking / short slides.

Drop-path (by design):
    Linearly scaled 0 -> drop_path_rate across the `depth` layers,
    standard DeiT/TransMIL recipe. Per-sample mask; no effect in eval.

Diagnostic capture (via src/diagnostics.py):
    Each TransLayer is a `Block` that, under `_capture_cls_pipeline`,
    records CLS before-attn / after-attn / after-block for the
    per-layer CLS cosine evolution plots. The underlying Nystrom-XSA
    attention records its own stats under `_capture_stats`. The
    diagnostic module toggles both.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .nystrom_xsa import NystromXSAAttention
from .ppeg import PPEG


_CHECKPOINT_MODES = ("whole_block", "per_module", "off")


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic depth, ported verbatim from stage 1 src/vit.py.

    Per-sample (not per-token): one Bernoulli draw per batch row,
    surviving rows rescaled by 1/keep_prob. Identity when drop_prob == 0
    or in eval mode.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """nn.Module wrapper around drop_path with a fixed drop probability."""
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"


class Mlp(nn.Module):
    """Two-layer MLP with GELU, same as stage 1 src/vit.py."""
    def __init__(self, in_dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    """
    Pre-norm Nystrom-XSA TransLayer.

    Named `Block` (not `TransLayer`) to keep the diagnostics module's
    class-name lookup (_BLOCK_CLS_NAME = "Block") unchanged between
    stage 1 and stage 2.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_landmarks: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        alpha_cls_mode: str = "zero",
        alpha_patch_mode: str = "zero",
        alpha_init: float = 1.0,
        num_cls_tokens: int = 1,
        pinv_iterations: int = 6,
        drop_path: float = 0.0,
        checkpoint_mode: str = "whole_block",
    ) -> None:
        super().__init__()
        assert checkpoint_mode in _CHECKPOINT_MODES
        self.checkpoint_mode = checkpoint_mode

        self.norm1 = nn.LayerNorm(dim)
        self.attn = NystromXSAAttention(
            dim=dim, num_heads=num_heads, num_landmarks=num_landmarks,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
            alpha_cls_mode=alpha_cls_mode,
            alpha_patch_mode=alpha_patch_mode,
            alpha_init=alpha_init,
            num_cls_tokens=num_cls_tokens,
            pinv_iterations=pinv_iterations,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_dim=dim, hidden_dim=int(dim * mlp_ratio), drop=proj_drop)
        self.drop_path = DropPath(drop_path)

        # CLS trajectory capture hooks (same interface as stage 1).
        self._capture_cls_pipeline = False
        self.last_cls_states: dict | None = None

    # --- submodule forward helpers used by per-module checkpointing -----
    def _attn_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Attention residual branch output (pre drop-path, pre residual)."""
        return self.attn(self.norm1(x))

    def _mlp_branch(self, x: torch.Tensor) -> torch.Tensor:
        """MLP residual branch output (pre drop-path, pre residual)."""
        return self.mlp(self.norm2(x))

    def _forward_inner(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unchecked forward path (whole block). Used both as the direct
        forward in checkpoint_mode="off" and as the callable for
        checkpoint_mode="whole_block".
        """
        attn_out = self._attn_branch(x)
        x_after_attn = x + self.drop_path(attn_out)
        mlp_out = self._mlp_branch(x_after_attn)
        x_out = x_after_attn + self.drop_path(mlp_out)

        # Stats capture is intentionally inside _forward_inner so both
        # non-checkpointed and whole-block checkpoint paths hit it.
        # Per-module checkpoint mode runs the attention submodule inside
        # a checkpointed call; that submodule's internal `_capture_stats`
        # still runs but the stats dict is stashed on self.attn.last_stats
        # as usual.
        if self._capture_cls_pipeline:
            self.last_cls_states = {
                "cls_before_attn": x[:, 0].detach(),
                "cls_after_attn": x_after_attn[:, 0].detach(),
                "cls_after_block": x_out[:, 0].detach(),
            }
        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.checkpoint_mode
        if mode == "off" or not self.training:
            # Inference path: checkpointing would only hurt eval latency.
            return self._forward_inner(x)

        if mode == "whole_block":
            # One boundary per layer; recompute = full layer on backward.
            # use_reentrant=False is the modern path and is required for
            # gradients through nn.Parameter inside the checkpointed
            # function (otherwise needs input-grad tricks).
            return cp.checkpoint(self._forward_inner, x, use_reentrant=False)

        # mode == "per_module": two boundaries per layer.
        attn_out = cp.checkpoint(self._attn_branch, x, use_reentrant=False)
        x_after_attn = x + self.drop_path(attn_out)
        mlp_out = cp.checkpoint(self._mlp_branch, x_after_attn, use_reentrant=False)
        x_out = x_after_attn + self.drop_path(mlp_out)
        if self._capture_cls_pipeline:
            self.last_cls_states = {
                "cls_before_attn": x[:, 0].detach(),
                "cls_after_attn": x_after_attn[:, 0].detach(),
                "cls_after_block": x_out[:, 0].detach(),
            }
        return x_out


class NystromXSAggregator(nn.Module):
    """
    Full TransMIL-style slide-level aggregator with Nystrom + XSA.

    Defaults track the released protocol
    """
    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        num_landmarks: int = 64,
        num_classes: int = 4,                # K-class head; 4 in the main sweep
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        alpha_cls_mode: str = "zero",
        alpha_patch_mode: str = "zero",
        alpha_init: float = 1.0,
        drop_path_rate: float = 0.1,
        pinv_iterations: int = 6,
        checkpoint_mode: str = "whole_block",
    ) -> None:
        super().__init__()
        assert checkpoint_mode in _CHECKPOINT_MODES
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.num_cls_tokens = 1
        self.checkpoint_mode = checkpoint_mode

        # Input projection 1024 -> 384.
        self.in_proj = nn.Linear(in_dim, embed_dim)

        # Learned CLS token.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Drop-path schedule: linear 0 -> drop_path_rate across `depth`.
        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist() if depth > 0 else []

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                num_landmarks=num_landmarks,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                alpha_cls_mode=alpha_cls_mode,
                alpha_patch_mode=alpha_patch_mode,
                alpha_init=alpha_init,
                num_cls_tokens=self.num_cls_tokens,
                pinv_iterations=pinv_iterations,
                drop_path=dpr[i],
                checkpoint_mode=checkpoint_mode,
            )
            for i in range(depth)
        ])

        # PPEG after block 1 (the released protocol).
        self.ppeg = PPEG(dim=embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        # K-class head: `num_classes` logits. For K=4 the training path
        # uses cross-entropy; for K=2 it uses CE over 2 logits (mathematically
        # equivalent to BCE on 1 logit + softmax-renormalization). Raw
        # logits are returned for numerical stability; softmax/argmax
        # happens in train.py's eval path.
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Truncated-normal for learnable params; zeros for biases."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # Depthwise convs in PPEG: use trunc_normal like a 1x1 linear
                # would get. Bias zero.
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, N, in_dim) raw UNI v1 features for a single slide
        (B should be 1 at train time per the released protocol).

        Returns: (B, num_classes) logit tensor.

        The internal sequence length grows from N to 1 + H*W where
        H = W = ceil(sqrt(N)) so the patch grid is square for PPEG.
        The extra HW - N padding rows are filled by replicating the
        first HW-N features (TransMIL's approach); all downstream
        attention/PPEG sees those rows as real tokens.
        """
        B, N, _ = features.shape
        x = self.in_proj(features)                   # (B, N, embed_dim)

        # Square-pad for PPEG. H*W >= N.
        H = W = int(math.ceil(math.sqrt(N)))
        add = H * W - N
        if add > 0:
            # TransMIL replicates the first `add` features rather than
            # zero-padding. This keeps the "shape" of the value
            # distribution unchanged -- zero features would be
            # out-of-distribution for the learned Q/K/V weights.
            x = torch.cat([x, x[:, :add, :]], dim=1)  # (B, HW, embed_dim)

        # Prepend CLS.
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                # (B, 1+HW, embed_dim)

        # Block 1 -> PPEG -> Blocks 2..depth (TransMIL-faithful placement).
        x = self.blocks[0](x)
        x = self.ppeg(x, H, W)
        for blk in self.blocks[1:]:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                             # (B, embed_dim)
        return self.head(cls_out)                     # (B, num_classes)
