"""
Per-layer CLS and attention-mechanism diagnostics for patch-level ViT runs.

This module provides:

  1. cos(y_i, v_i) diagnostics -- the paper's central "attention similarity
     bias". Captured per token, split by role (CLS vs patch), averaged per
     example (see comment 2 fix below), per head per layer:
       - cos_yv_cls_pre, cos_yv_cls_post      shape (depth, num_heads)
       - cos_yv_patch_pre, cos_yv_patch_post  shape (depth, num_heads)

  2. CLS-only diagnostics:
       - cls_self_attn[layer, head]  = a_{cls,cls}, the attention probability
                                        CLS assigns to itself
       - cos_before_after_attn[layer] = cos(CLS_in, CLS_after_attn_sublayer)
       - cos_before_after_block[layer]= cos(CLS_in, CLS_after_full_block)

  3. Alpha snapshots (data-independent; read directly from module state):
       - alpha_cls[layer, head], alpha_patch[layer, head]

Averaging semantics
-------------------
extract_batch_stats returns, per metric, a pair (sum_tensor, count_int),
where `sum_tensor` is the SUM of that metric over its sample units in the
current batch, and `count_int` is the number of sample units that
contributed to the sum. Different metrics use different sample units:

  - cls_self_attn, cos_before_after_attn, cos_before_after_block,
    cos_yv_cls_pre, cos_yv_cls_post, y/v/z_norm_cls, z_over_y_cls:
                                            unit = image (count = B)
  - cos_yv_patch_pre, cos_yv_patch_post,
    y/v/z_norm_patch, z_over_y_patch:       unit = patch-token-image pair
                                            (count = B * num_patches)

StatsAccumulator then accumulates `sums` and `counts` independently and
divides at the end. This makes the reported means PER-EXAMPLE (or
per-token-per-example), which is correct when the last batch is smaller
than the others -- as happens at test time: 7180 images / batch=512 leaves
a tail of 12 images whose contribution should be weighted 12/7180, not 1/15.

Capture is opt-in (set_capture_mode). Usage pattern:

    set_capture_mode(model, True)
    acc = StatsAccumulator()
    for imgs, _ in loader:
        _ = model(imgs)
        acc.update(extract_batch_stats(model))
    set_capture_mode(model, False)
    stats = acc.result()  # dict of per-example-averaged tensors
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

# Lazy string-based type-checking to avoid circular imports.
# PatchSRPAttention shares this diagnostics harness because it populates the
# same .last_stats keys expected by the accumulator.
_XSA_ATTN_CLS_NAMES = ("XSAAttention", "PatchSRPAttention")
_XSA_ATTN_CLS_NAME = "XSAAttention"   # retained for back-compat
_BLOCK_CLS_NAME = "Block"


def set_capture_mode(model: torch.nn.Module, enable: bool) -> None:
    """
    Toggle the two capture flags on every relevant submodule (any of the
    attention classes listed in _XSA_ATTN_CLS_NAMES, plus Block).
    On disable, also clear any stashed buffers to free memory.
    """
    for m in model.modules():
        cname = type(m).__name__
        if cname in _XSA_ATTN_CLS_NAMES and hasattr(m, "_capture_stats"):
            m._capture_stats = enable
            if not enable:
                m.last_stats = None
        elif cname == _BLOCK_CLS_NAME and hasattr(m, "_capture_cls_pipeline"):
            m._capture_cls_pipeline = enable
            if not enable:
                m.last_cls_states = None


# A "sample" is a pair (sum_tensor, count_int). Sums and counts are
# accumulated separately across batches; the final metric is sum / count.
_Sample = Tuple[torch.Tensor, int]


def extract_batch_stats(model: torch.nn.Module) -> Dict[str, _Sample]:
    """
    Pull per-layer stats from the most recent forward pass, returning
    (sum, count) pairs so the caller can accumulate across batches and
    divide by the true number of sample units at the end.

    Requires set_capture_mode(model, True) and one prior forward pass.
    """
    # Collectors, one per metric. Each list entry corresponds to one block.
    cls_attn_sums: list[torch.Tensor] = []      # per-batch sum (H,), count=B
    cos_beforeafter_attn_sums: list[torch.Tensor] = []  # scalar per layer, count=B
    cos_beforeafter_block_sums: list[torch.Tensor] = []
    cos_yv_cls_pre_sums: list[torch.Tensor] = []        # (H,), count=B*num_cls
    cos_yv_cls_post_sums: list[torch.Tensor] = []
    cos_yv_patch_pre_sums: list[torch.Tensor] = []      # (H,), count=B*num_patch
    cos_yv_patch_post_sums: list[torch.Tensor] = []
    # Magnitude (L2-norm) diagnostics. Per-head norms of the attention
    # output y, the value vector v, and the XSA-projected output z,
    # reported as role-wise means. These are **scale diagnostics** only
    # -- do not interpret mean(z_norm)/mean(y_norm) as "fraction of
    # attention magnitude preserved", since mean-of-ratios != ratio-of-means.
    y_norm_cls_sums: list[torch.Tensor] = []
    v_norm_cls_sums: list[torch.Tensor] = []
    z_norm_cls_sums: list[torch.Tensor] = []
    y_norm_patch_sums: list[torch.Tensor] = []
    v_norm_patch_sums: list[torch.Tensor] = []
    z_norm_patch_sums: list[torch.Tensor] = []
    # Proper per-token preserved-fraction metric: compute ||z||/||y||
    # per (batch, head, token) BEFORE averaging, so the role-wise mean
    # is a true mean-of-ratios. Under alpha=1 this equals sqrt(1 - cos^2(y,v))
    # per token, and the mean-of-ratios is the right population-level
    # summary of "on average, how much attention magnitude did XSA preserve?"
    z_over_y_cls_sums: list[torch.Tensor] = []
    z_over_y_patch_sums: list[torch.Tensor] = []

    # Counts are batch-specific; all layers share the same batch so we
    # compute them once from the first block's tensors.
    count_per_image = None     # B
    count_cls = None           # B * num_cls_tokens
    count_patch = None         # B * num_patch_tokens

    for bi, blk in enumerate(model.blocks):
        assert blk.attn.last_stats is not None, (
            "No attention stats captured. Did you call set_capture_mode(model, True) "
            "and run a forward pass?"
        )
        assert blk.last_cls_states is not None, "No CLS states captured."

        # attn_probs: (B, H, N, N) -- after softmax, pre-dropout.
        attn_probs = blk.attn.last_stats["attn_probs"].float()
        cos_yv_pre = blk.attn.last_stats["cos_yv_pre"].float()   # (B, H, N)
        cos_zv_post = blk.attn.last_stats["cos_zv_post"].float() # (B, H, N)
        n_cls = int(blk.attn.last_stats["num_cls_tokens"])

        B, H, N, _ = attn_probs.shape
        if count_per_image is None:
            count_per_image = B
            count_cls = B * n_cls
            count_patch = B * (N - n_cls)

        # --- CLS-only metrics: a_{cls,cls} per head, summed over batch. ---
        # attn_probs[:, :, 0, 0] is (B, H); we sum over B to get (H,).
        # For num_cls_tokens > 1 we would sum the CLS block diagonal; the
        # current ADP model uses exactly one CLS token, so we index position 0.
        cls_attn_sums.append(attn_probs[:, :, 0, 0].sum(dim=0))

        before = blk.last_cls_states["cls_before_attn"].float()       # (B, C)
        after_attn = blk.last_cls_states["cls_after_attn"].float()    # (B, C)
        after_block = blk.last_cls_states["cls_after_block"].float()  # (B, C)
        # Per-image cosine along the embedding axis -> (B,), then sum over batch.
        cos_beforeafter_attn_sums.append(
            F.cosine_similarity(before, after_attn, dim=-1).sum()
        )
        cos_beforeafter_block_sums.append(
            F.cosine_similarity(before, after_block, dim=-1).sum()
        )

        # --- cos(y, v) role-split sums ------------------------------------
        # cos_yv_pre, cos_yv_post are (B, H, N). Split along N into CLS rows
        # (first n_cls) and patch rows (remaining N - n_cls), then sum over
        # batch and tokens for each group -> (H,).
        pre_cls = cos_yv_pre[:, :, :n_cls].sum(dim=(0, 2))        # (H,)
        post_cls = cos_zv_post[:, :, :n_cls].sum(dim=(0, 2))      # (H,)
        pre_pat = cos_yv_pre[:, :, n_cls:].sum(dim=(0, 2))        # (H,)
        post_pat = cos_zv_post[:, :, n_cls:].sum(dim=(0, 2))      # (H,)
        cos_yv_cls_pre_sums.append(pre_cls)
        cos_yv_cls_post_sums.append(post_cls)
        cos_yv_patch_pre_sums.append(pre_pat)
        cos_yv_patch_post_sums.append(post_pat)

        # --- norm role-split sums (scale diagnostics only) ----------------
        # Same split as cos_yv, applied to the per-head L2 norms of y, v, z.
        y_norm = blk.attn.last_stats["y_norm"].float()   # (B, H, N)
        v_norm = blk.attn.last_stats["v_norm"].float()
        z_norm = blk.attn.last_stats["z_norm"].float()
        y_norm_cls_sums.append(y_norm[:, :, :n_cls].sum(dim=(0, 2)))
        v_norm_cls_sums.append(v_norm[:, :, :n_cls].sum(dim=(0, 2)))
        z_norm_cls_sums.append(z_norm[:, :, :n_cls].sum(dim=(0, 2)))
        y_norm_patch_sums.append(y_norm[:, :, n_cls:].sum(dim=(0, 2)))
        v_norm_patch_sums.append(v_norm[:, :, n_cls:].sum(dim=(0, 2)))
        z_norm_patch_sums.append(z_norm[:, :, n_cls:].sum(dim=(0, 2)))

        # --- per-token preserved-fraction ratio ---------------------------
        # Compute ||z||/||y|| per token FIRST, then aggregate. This is the
        # mean-of-ratios statistic that cannot be recovered from separate
        # means of ||y|| and ||z||. Epsilon guards
        # against zero-norm y (degenerate; in practice y norms are well
        # above 1 by late-layer scales).
        z_over_y_full = z_norm / (y_norm + 1e-8)          # (B, H, N)
        z_over_y_cls_sums.append(z_over_y_full[:, :, :n_cls].sum(dim=(0, 2)))
        z_over_y_patch_sums.append(z_over_y_full[:, :, n_cls:].sum(dim=(0, 2)))

    # Stack into (depth, ...) tensors, then bundle with their counts.
    out: Dict[str, _Sample] = {
        "cls_self_attn":              (torch.stack(cls_attn_sums, dim=0),          count_per_image),
        "cos_before_after_attn":      (torch.stack(cos_beforeafter_attn_sums, 0),  count_per_image),
        "cos_before_after_block":     (torch.stack(cos_beforeafter_block_sums, 0), count_per_image),
        "cos_yv_cls_pre":             (torch.stack(cos_yv_cls_pre_sums, 0),        count_cls),
        "cos_yv_cls_post":            (torch.stack(cos_yv_cls_post_sums, 0),       count_cls),
        # Norms split by role. Counts match the cos_yv_* counts above
        # (one value per token per example).
        "y_norm_cls":                 (torch.stack(y_norm_cls_sums, 0),            count_cls),
        "v_norm_cls":                 (torch.stack(v_norm_cls_sums, 0),            count_cls),
        "z_norm_cls":                 (torch.stack(z_norm_cls_sums, 0),            count_cls),
        "y_norm_patch":               (torch.stack(y_norm_patch_sums, 0),          count_patch),
        "v_norm_patch":               (torch.stack(v_norm_patch_sums, 0),          count_patch),
        "z_norm_patch":               (torch.stack(z_norm_patch_sums, 0),          count_patch),
        # Proper per-token preserved-fraction: mean-of-ratios of ||z||/||y||.
        "z_over_y_cls":               (torch.stack(z_over_y_cls_sums, 0),          count_cls),
        "z_over_y_patch":             (torch.stack(z_over_y_patch_sums, 0),        count_patch),
        "cos_yv_patch_pre":           (torch.stack(cos_yv_patch_pre_sums, 0),      count_patch),
        "cos_yv_patch_post":          (torch.stack(cos_yv_patch_post_sums, 0),     count_patch),
    }
    return out


class StatsAccumulator:
    """
    Streaming per-example-weighted accumulator.

    update() takes a dict of (sum_tensor, count_int) pairs, accumulates sums
    and counts separately, and result() returns the per-example means
    (sum / count) for each metric.

    This is the correct per-example mean regardless of batch size variation,
    including the tail batch at the end of the test loader. Tail batches no
    longer get disproportionate weight.
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
    """CUDA-only autocast context; nullcontext on CPU or when dtype is None."""
    import contextlib
    if dtype is not None and device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def run_capture_eval(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Diagnostic-only eval: runs forward passes with capture enabled and
    returns per-example-averaged stats. Does not compute predictions or
    loss; use the inline path in train.py if you need both metrics and
    diagnostics in one loop.
    """
    model.eval()
    set_capture_mode(model, True)
    acc = StatsAccumulator()

    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break
                imgs = batch[0].to(device, non_blocking=True)
                with autocast_ctx(device, autocast_dtype):
                    _ = model(imgs)
                acc.update(extract_batch_stats(model))
    finally:
        set_capture_mode(model, False)

    return acc.result()


def extract_alpha_values(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Snapshot the current alpha_cls and alpha_patch tensors from every block.

    Returns:
        {"alpha_cls":   (depth, num_heads) tensor,
         "alpha_patch": (depth, num_heads) tensor}

    These come directly from module state (Parameter or buffer), so this
    function does not need a forward pass.
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


def extract_beta_values(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Snapshot the current beta_patch tensor from every SRP block.

    Returns:
        {"beta_patch": (depth, num_heads) tensor}

    PatchSRPAttention has a single per-head `beta_patch` parameter/buffer;
    there is no beta_cls because CLS is structurally untouched by SRP
    This is the SRP counterpart of extract_alpha_values.
    """
    rows = []
    for blk in model.blocks:
        rows.append(blk.attn.beta_patch.detach().clone())
    return {"beta_patch": torch.stack(rows, dim=0)}
