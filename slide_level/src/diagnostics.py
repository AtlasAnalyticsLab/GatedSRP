"""
Per-layer CLS and Nystrom-XSA attention diagnostics for the slide-level PoC.

Ported from stage-1 src/diagnostics.py. The semantics, per-example
averaging scheme, and W&B-facing output dict are identical; only two
things change at stage 2:

  (1) `cls_self_attn` is read directly from `attn.last_stats["cls_self_attn"]`
      (precomputed inside Nystrom-XSA) instead of being extracted from an
      attn_probs tensor. Nystrom never materializes the full attention
      matrix, so precomputing just the (CLS, CLS) entry in the attention
      module is the only tractable option at slide scale (DESIGN.md §4.4).

  (2) The class-name string used for attention-module discovery is
      "NystromXSAAttention" instead of "XSAAttention".

Everything else -- StatsAccumulator, extract_alpha_values, autocast_ctx,
the per-example vs. per-token-per-example counting -- is byte-for-byte
the same logic.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


# Class-name-based discovery (stage 1's convention; avoids circular imports).
_XSA_ATTN_CLS_NAME = "NystromXSAAttention"
_BLOCK_CLS_NAME = "Block"


def set_capture_mode(model: torch.nn.Module, enable: bool) -> None:
    """
    Toggle both capture flags on every relevant submodule
    (NystromXSAAttention and Block). On disable, also clear stashed
    buffers to free GPU memory.
    """
    for m in model.modules():
        cname = type(m).__name__
        if cname == _XSA_ATTN_CLS_NAME and hasattr(m, "_capture_stats"):
            m._capture_stats = enable
            if not enable:
                m.last_stats = None
        elif cname == _BLOCK_CLS_NAME and hasattr(m, "_capture_cls_pipeline"):
            m._capture_cls_pipeline = enable
            if not enable:
                m.last_cls_states = None


# (sum_tensor, count_int) pairs accumulated separately; the final
# metric is sum / count, yielding per-example (or per-token-per-example)
# means invariant to batch-size variation.
_Sample = Tuple[torch.Tensor, int]


def extract_batch_stats(model: torch.nn.Module) -> Dict[str, _Sample]:
    """
    Pull per-layer stats from the most recent forward pass, returning
    (sum, count) pairs for later averaging.

    Requires set_capture_mode(model, True) and one prior forward pass.

    Differences vs. stage 1:
      - `cls_self_attn` is read from last_stats["cls_self_attn"] (shape
        (B, H)) rather than indexing attn_probs[:, :, 0, 0].
      - `attn_probs` is never present in last_stats (Nystrom does not
        materialize it).

    Counting semantics match stage 1:
      - CLS-only metrics use count = B (one value per image, at position 0)
      - Patch metrics use count = B * num_patches (per-token-per-image)
    """
    cls_attn_sums: list[torch.Tensor] = []
    cos_beforeafter_attn_sums: list[torch.Tensor] = []
    cos_beforeafter_block_sums: list[torch.Tensor] = []
    cos_yv_cls_pre_sums: list[torch.Tensor] = []
    cos_yv_cls_post_sums: list[torch.Tensor] = []
    cos_yv_patch_pre_sums: list[torch.Tensor] = []
    cos_yv_patch_post_sums: list[torch.Tensor] = []
    y_norm_cls_sums: list[torch.Tensor] = []
    v_norm_cls_sums: list[torch.Tensor] = []
    z_norm_cls_sums: list[torch.Tensor] = []
    y_norm_patch_sums: list[torch.Tensor] = []
    v_norm_patch_sums: list[torch.Tensor] = []
    z_norm_patch_sums: list[torch.Tensor] = []
    z_over_y_cls_sums: list[torch.Tensor] = []
    z_over_y_patch_sums: list[torch.Tensor] = []

    count_per_image = None
    count_cls = None
    count_patch = None

    for blk in model.blocks:
        assert blk.attn.last_stats is not None, (
            "No attention stats captured. Did you call "
            "set_capture_mode(model, True) and run a forward pass?"
        )
        assert blk.last_cls_states is not None, "No CLS states captured."

        cos_yv_pre = blk.attn.last_stats["cos_yv_pre"].float()    # (B, H, N)
        cos_zv_post = blk.attn.last_stats["cos_zv_post"].float()  # (B, H, N)
        cls_self_attn_bh = blk.attn.last_stats["cls_self_attn"].float()  # (B, H)
        n_cls = int(blk.attn.last_stats["num_cls_tokens"])

        B, H, N = cos_yv_pre.shape
        if count_per_image is None:
            count_per_image = B
            count_cls = B * n_cls
            count_patch = B * (N - n_cls)

        # CLS self-attention: (B, H) -> (H,) via batch sum.
        cls_attn_sums.append(cls_self_attn_bh.sum(dim=0))

        before = blk.last_cls_states["cls_before_attn"].float()
        after_attn = blk.last_cls_states["cls_after_attn"].float()
        after_block = blk.last_cls_states["cls_after_block"].float()
        cos_beforeafter_attn_sums.append(
            F.cosine_similarity(before, after_attn, dim=-1).sum()
        )
        cos_beforeafter_block_sums.append(
            F.cosine_similarity(before, after_block, dim=-1).sum()
        )

        # Role-split cos_yv sums.
        pre_cls = cos_yv_pre[:, :, :n_cls].sum(dim=(0, 2))
        post_cls = cos_zv_post[:, :, :n_cls].sum(dim=(0, 2))
        pre_pat = cos_yv_pre[:, :, n_cls:].sum(dim=(0, 2))
        post_pat = cos_zv_post[:, :, n_cls:].sum(dim=(0, 2))
        cos_yv_cls_pre_sums.append(pre_cls)
        cos_yv_cls_post_sums.append(post_cls)
        cos_yv_patch_pre_sums.append(pre_pat)
        cos_yv_patch_post_sums.append(post_pat)

        # Role-split norm sums (scale diagnostics).
        y_norm = blk.attn.last_stats["y_norm"].float()
        v_norm = blk.attn.last_stats["v_norm"].float()
        z_norm = blk.attn.last_stats["z_norm"].float()
        y_norm_cls_sums.append(y_norm[:, :, :n_cls].sum(dim=(0, 2)))
        v_norm_cls_sums.append(v_norm[:, :, :n_cls].sum(dim=(0, 2)))
        z_norm_cls_sums.append(z_norm[:, :, :n_cls].sum(dim=(0, 2)))
        y_norm_patch_sums.append(y_norm[:, :, n_cls:].sum(dim=(0, 2)))
        v_norm_patch_sums.append(v_norm[:, :, n_cls:].sum(dim=(0, 2)))
        z_norm_patch_sums.append(z_norm[:, :, n_cls:].sum(dim=(0, 2)))

        # Per-token preserved-fraction (mean-of-ratios of ||z||/||y||).
        z_over_y_full = z_norm / (y_norm + 1e-8)
        z_over_y_cls_sums.append(z_over_y_full[:, :, :n_cls].sum(dim=(0, 2)))
        z_over_y_patch_sums.append(z_over_y_full[:, :, n_cls:].sum(dim=(0, 2)))

    out: Dict[str, _Sample] = {
        "cls_self_attn":              (torch.stack(cls_attn_sums, dim=0),          count_per_image),
        "cos_before_after_attn":      (torch.stack(cos_beforeafter_attn_sums, 0),  count_per_image),
        "cos_before_after_block":     (torch.stack(cos_beforeafter_block_sums, 0), count_per_image),
        "cos_yv_cls_pre":             (torch.stack(cos_yv_cls_pre_sums, 0),        count_cls),
        "cos_yv_cls_post":            (torch.stack(cos_yv_cls_post_sums, 0),       count_cls),
        "y_norm_cls":                 (torch.stack(y_norm_cls_sums, 0),            count_cls),
        "v_norm_cls":                 (torch.stack(v_norm_cls_sums, 0),            count_cls),
        "z_norm_cls":                 (torch.stack(z_norm_cls_sums, 0),            count_cls),
        "y_norm_patch":               (torch.stack(y_norm_patch_sums, 0),          count_patch),
        "v_norm_patch":               (torch.stack(v_norm_patch_sums, 0),          count_patch),
        "z_norm_patch":               (torch.stack(z_norm_patch_sums, 0),          count_patch),
        "z_over_y_cls":               (torch.stack(z_over_y_cls_sums, 0),          count_cls),
        "z_over_y_patch":             (torch.stack(z_over_y_patch_sums, 0),        count_patch),
        "cos_yv_patch_pre":           (torch.stack(cos_yv_patch_pre_sums, 0),      count_patch),
        "cos_yv_patch_post":          (torch.stack(cos_yv_patch_post_sums, 0),     count_patch),
    }
    return out


class StatsAccumulator:
    """
    Streaming per-example-weighted accumulator.

    update() takes {metric_name: (sum_tensor, count_int)} pairs;
    result() returns {metric_name: sum/count} so the final means are
    per-example (or per-token-per-example) invariant to batch size
    variation -- critical here because slide batches are always 1 and
    the total "example count" accumulates straightforwardly.

    Same implementation as stage 1.
    """
    def __init__(self) -> None:
        self._sums: Dict[str, torch.Tensor] = {}
        self._counts: Dict[str, int] = {}

    def update(self, batch_stats: Dict[str, _Sample]) -> None:
        for k, (s, c) in batch_stats.items():
            if k not in self._sums:
                self._sums[k] = s.clone().float()
                self._counts[k] = int(c)
            else:
                self._sums[k] = self._sums[k] + s.float()
                self._counts[k] = self._counts[k] + int(c)

    def result(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, s in self._sums.items():
            c = max(1, self._counts[k])
            out[k] = s / c
        return out


def autocast_ctx(device: torch.device, dtype: Optional[torch.dtype]):
    """CUDA-only autocast context; nullcontext elsewhere."""
    import contextlib
    if dtype is not None and device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def extract_alpha_values(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Snapshot current alpha_cls and alpha_patch from every Block.
    Same API as stage 1.

    Returns:
      {"alpha_cls":   (depth, num_heads) tensor,
       "alpha_patch": (depth, num_heads) tensor}
    """
    alpha_cls_rows = []
    alpha_patch_rows = []
    for blk in model.blocks:
        alpha_cls_rows.append(blk.attn.alpha_cls.detach().clone())
        alpha_patch_rows.append(blk.attn.alpha_patch.detach().clone())
    return {
        "alpha_cls": torch.stack(alpha_cls_rows, dim=0),
        "alpha_patch": torch.stack(alpha_patch_rows, dim=0),
    }
