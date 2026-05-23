"""
Diagnostics for SRP-enabled training runs.

Scope: per-example-weighted accumulation of the NystromSRPAttention.last_stats
dict across a pass (val-sample or full-test-fold), producing role-split
means ({cls, patch}) per (layer, head). Same averaging discipline as
slide_level/src/diagnostics.py — counts are kept separate from sums and
the final means divide at .result() time.

Key differences from stage-2 diagnostics:
  * SRP-specific stats (cos(y, r), cos(z, r), h^V, h^morph, ρ, cos(v, r))
    are accumulated.
  * Class-name lookups discover NystromSRPAttention / SRPBlock instead
    of NystromXSAAttention / Block.
  * ρ, cos(v_j, r_j), cos(v'_j, r_j) are captured ONLY under srp_mode=pre_v;
    the accumulator gracefully skips None fields.

The set_capture_mode / StatsAccumulator / autocast_ctx helpers have the
same signatures as stage 2, so training-loop code is drop-in.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


_SRP_ATTN_CLS_NAME = "NystromSRPAttention"
_SRP_BLOCK_CLS_NAME = "SRPBlock"


def set_capture_mode(model: torch.nn.Module, enable: bool) -> None:
    """Toggle capture flags on every NystromSRPAttention + SRPBlock."""
    for m in model.modules():
        cname = type(m).__name__
        if cname == _SRP_ATTN_CLS_NAME and hasattr(m, "_capture_stats"):
            m._capture_stats = enable
            if not enable:
                m.last_stats = None
        elif cname == _SRP_BLOCK_CLS_NAME and hasattr(m, "_capture_cls_pipeline"):
            m._capture_cls_pipeline = enable
            if not enable:
                m.last_cls_states = None


# (sum_tensor, count_int) pair — same convention as stage 2.
_Sample = Tuple[torch.Tensor, int]


def extract_batch_stats(model: torch.nn.Module) -> Dict[str, _Sample]:
    """
    Pull per-layer stats from the most recent forward and return
    (sum, count) pairs ready for streaming averaging.

    For every Block we stack per-layer tensors so the output's first
    axis is the layer index. This matches stage-2's layout and lets
    downstream W&B tables be written per (layer, head).

    Keys emitted (exact list):
      # Stage-2-compatible role-split stats.
      cls_self_attn              (D, H)           count = B
      cos_before_after_attn      (D,)             count = B
      cos_before_after_block     (D,)             count = B
      cos_yv_cls_pre/post        (D, H)           count = B * n_cls
      cos_yv_patch_pre/post      (D, H)           count = B * N_real
      y/v/z_norm_cls             (D, H)           count = B * n_cls
      y/v/z_norm_patch           (D, H)           count = B * N_real
      z_over_y_cls/patch         (D, H)           count = B * (n_cls|N_real)

      # SRP-specific (§8.2.A, B, D).
      cos_yr_patch_pre/post      (D, H)           count = B * N_real
      h_V_patch                  (D, H)           count = B * N_real
      cos_vr_patch_pre/post      (D, H)           count = B * N_real  (pre_v only)
      rho_patch                  (D, H)           count = B * N_real  (pre_v only)
      h_morph_patch              (D,)             count = B * N_real  (gated only)

    Patch-slice counts use N_real — pad duplicates are NOT included
    (proposal §12.2 rule 4). n_cls = num_cls_tokens.
    """
    out: Dict[str, _Sample] = {}

    # We'll accumulate per-layer tensors in these lists, then stack.
    lists: Dict[str, list] = {}

    def push(key: str, tensor: torch.Tensor) -> None:
        lists.setdefault(key, []).append(tensor)

    # Optional lists (pre_v only). Track whether each key should be
    # emitted based on whether the first block produced a non-None value.
    optional_present: Dict[str, bool] = {
        "cos_vr_patch_pre": False,
        "cos_vr_patch_post": False,
        "rho_patch": False,
        "h_morph_patch": False,
        "bar_rho_cls": False,
    }

    B_cls = None
    count_per_image = None
    count_cls = None
    count_patch = None

    for blk in model.blocks:
        stats = blk.attn.last_stats
        assert stats is not None, (
            "No attention stats captured. Did you call set_capture_mode("
            "model, True) and run a forward pass?"
        )
        assert blk.last_cls_states is not None, "No CLS states captured."

        n_cls = int(stats["num_cls_tokens"])
        N_real = int(stats["N_real"])

        # cls_self_attn: (B, H). Sum over batch -> (H,).
        cls_self_attn_bh = stats["cls_self_attn"].float()
        if B_cls is None:
            B_cls = cls_self_attn_bh.shape[0]
            count_per_image = B_cls
            count_cls = B_cls * n_cls
            count_patch = B_cls * N_real
        push("cls_self_attn", cls_self_attn_bh.sum(dim=0))      # (H,)

        # CLS evolution cosines (scalar per layer per example).
        before = blk.last_cls_states["cls_before_attn"].float()
        after_attn = blk.last_cls_states["cls_after_attn"].float()
        after_block = blk.last_cls_states["cls_after_block"].float()
        push("cos_before_after_attn",
             F.cosine_similarity(before, after_attn, dim=-1).sum())
        push("cos_before_after_block",
             F.cosine_similarity(before, after_block, dim=-1).sum())

        # Stage-2 cos_yv, role-split.
        cos_yv_cls_pre = stats["cos_yv_cls_pre"].float()        # (B, H, n_cls)
        cos_yv_cls_post = stats["cos_yv_cls_post"].float()
        cos_yv_patch_pre = stats["cos_yv_patch_pre"].float()    # (B, H, N_real)
        cos_yv_patch_post = stats["cos_yv_patch_post"].float()
        push("cos_yv_cls_pre",   cos_yv_cls_pre.sum(dim=(0, 2)))
        push("cos_yv_cls_post",  cos_yv_cls_post.sum(dim=(0, 2)))
        push("cos_yv_patch_pre",  cos_yv_patch_pre.sum(dim=(0, 2)))
        push("cos_yv_patch_post", cos_yv_patch_post.sum(dim=(0, 2)))

        # Role-split norms.
        y_norm = stats["y_norm"].float()        # (B, H, L)
        v_norm = stats["v_norm"].float()
        z_norm = stats["z_norm"].float()
        # Slice roles: CLS is positions [0 : n_cls]; patch is positions
        # [n_cls : n_cls + N_real]. Pad duplicates (positions after) are
        # NOT included — proposal §12.2 rule 4.
        push("y_norm_cls",   y_norm[:, :, :n_cls].sum(dim=(0, 2)))
        push("v_norm_cls",   v_norm[:, :, :n_cls].sum(dim=(0, 2)))
        push("z_norm_cls",   z_norm[:, :, :n_cls].sum(dim=(0, 2)))
        push("y_norm_patch", y_norm[:, :, n_cls : n_cls + N_real].sum(dim=(0, 2)))
        push("v_norm_patch", v_norm[:, :, n_cls : n_cls + N_real].sum(dim=(0, 2)))
        push("z_norm_patch", z_norm[:, :, n_cls : n_cls + N_real].sum(dim=(0, 2)))

        # z_over_y preserved-fraction (per-token mean-of-ratios).
        z_over_y_full = z_norm / (y_norm + 1e-8)
        push("z_over_y_cls",   z_over_y_full[:, :, :n_cls].sum(dim=(0, 2)))
        push("z_over_y_patch", z_over_y_full[:, :, n_cls : n_cls + N_real].sum(dim=(0, 2)))

        # SRP-specific (§8.2.A, B).
        cos_yr_pre = stats["cos_yr_patch_pre"].float()
        cos_zr_post = stats["cos_zr_patch_post"].float()
        h_V = stats["h_V_patch"].float()
        push("cos_yr_patch_pre",  cos_yr_pre.sum(dim=(0, 2)))
        push("cos_yr_patch_post", cos_zr_post.sum(dim=(0, 2)))
        push("h_V_patch",         h_V.sum(dim=(0, 2)))

        # §8.2.E placement signature (per-slide, so per-example count).
        cos_y_cls_rbar = stats["cos_y_cls_rbar"].float()        # (B, H)
        push("cos_y_cls_rbar", cos_y_cls_rbar.sum(dim=0))

        # Optional pre_v-only diagnostics.
        for key in ("cos_vr_patch_pre", "cos_vr_patch_post", "rho_patch"):
            val = stats.get(key)
            if val is not None:
                optional_present[key] = True
                push(key, val.float().sum(dim=(0, 2)))

        # §8.2.D3 attention-weighted retention (pre_v only; per-example).
        bar_rho_cls = stats.get("bar_rho_cls")
        if bar_rho_cls is not None:
            optional_present["bar_rho_cls"] = True
            push("bar_rho_cls", bar_rho_cls.float().sum(dim=0))

        # Optional h_morph (per-slide, shared across heads).
        h_m = stats.get("h_morph_patch")
        if h_m is not None:
            optional_present["h_morph_patch"] = True
            # Shape (B, N_real); sum to a scalar per layer (scalar keyed
            # as the layer's aggregate, paired with count_patch).
            push("h_morph_patch", h_m.float().sum())

    # Stack per-layer and emit with count metadata.
    def emit(key: str, count: int) -> None:
        out[key] = (torch.stack(lists[key], dim=0), count)

    emit("cls_self_attn",            count_per_image)
    emit("cos_before_after_attn",    count_per_image)
    emit("cos_before_after_block",   count_per_image)
    emit("cos_yv_cls_pre",           count_cls)
    emit("cos_yv_cls_post",          count_cls)
    emit("cos_yv_patch_pre",         count_patch)
    emit("cos_yv_patch_post",        count_patch)
    emit("y_norm_cls",               count_cls)
    emit("v_norm_cls",               count_cls)
    emit("z_norm_cls",               count_cls)
    emit("y_norm_patch",             count_patch)
    emit("v_norm_patch",             count_patch)
    emit("z_norm_patch",             count_patch)
    emit("z_over_y_cls",             count_cls)
    emit("z_over_y_patch",           count_patch)
    emit("cos_yr_patch_pre",         count_patch)
    emit("cos_yr_patch_post",        count_patch)
    emit("h_V_patch",                count_patch)
    emit("cos_y_cls_rbar",           count_per_image)
    if optional_present["cos_vr_patch_pre"]:
        emit("cos_vr_patch_pre",     count_patch)
    if optional_present["cos_vr_patch_post"]:
        emit("cos_vr_patch_post",    count_patch)
    if optional_present["rho_patch"]:
        emit("rho_patch",            count_patch)
    if optional_present["bar_rho_cls"]:
        emit("bar_rho_cls",          count_per_image)
    if optional_present["h_morph_patch"]:
        emit("h_morph_patch",        count_patch)

    return out


class StatsAccumulator:
    """
    Streaming per-example-weighted accumulator (same as stage 2).
    update() takes {name: (sum_tensor, count_int)}; result() returns
    {name: sum/count}.
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
    """CUDA-only autocast context; nullcontext elsewhere. Same as stage 2."""
    import contextlib
    if dtype is not None and device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def extract_beta_values(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Snapshot beta_patch from every SRPBlock.

    Returns:
      {"beta_patch": (depth, num_heads) tensor}
    """
    rows = []
    for blk in model.blocks:
        rows.append(blk.attn.beta_patch.detach().clone())
    return {
        "beta_patch": torch.stack(rows, dim=0),   # (D, H)
    }


def extract_layerscale_values(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Snapshot CaiT-style LayerScale vectors from every SRPBlock.

    Returns an empty dict when LayerScale is disabled. Disabled runs register no
    gamma parameters by design, so old checkpoints and queue-launched commands
    retain their historical parameter surface.
    """
    attn_rows = []
    mlp_rows = []
    for blk in getattr(model, "blocks", []):
        gamma_attn = getattr(blk, "gamma_attn", None)
        gamma_mlp = getattr(blk, "gamma_mlp", None)
        if gamma_attn is None or gamma_mlp is None:
            continue
        # Clone detached tensors so artifact creation cannot keep graph edges
        # alive after test evaluation.
        attn_rows.append(gamma_attn.detach().clone())
        mlp_rows.append(gamma_mlp.detach().clone())
    if not attn_rows:
        return {}
    return {
        "layerscale_attn": torch.stack(attn_rows, dim=0),  # (D, embed_dim)
        "layerscale_mlp": torch.stack(mlp_rows, dim=0),    # (D, embed_dim)
    }


def extract_per_slide_diagnostics(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Pull per-slide scalar summaries from the most recent forward pass.

    Designed to be called ONCE per test slide (B=1) so the caller can
    stream per-slide records to disk without collapsing distributional
    information across slides. Returns, per layer:

      cos_y_cls_rbar:  (D, H)   -- §8.2.E placement signature
      mean_h_V:        (D, H)   -- V-space neighborhood coherence
      bar_rho_cls:     (D, H)   -- §8.2.D3 attention-weighted retention
                                   (only populated under pre_v)
      mean_cos_yr_pre: (D, H)   -- cos(y, r̂) before projection (patch avg)
      mean_cos_yr_post:(D, H)   -- cos(z, r̂) after projection (patch avg)
      mean_rho:        (D, H)   -- per-token ρ_j averaged (pre_v only)

    Requires capture mode active and a fresh forward pass. The function
    is cheap — it only reads tensors already stashed on each block's
    attn.last_stats.
    """
    cos_y_cls_rbar_layers = []
    mean_h_V_layers = []
    mean_cos_yr_pre_layers = []
    mean_cos_yr_post_layers = []
    bar_rho_cls_layers = []      # may stay empty if pre_v not active
    mean_rho_layers = []          # may stay empty

    for blk in model.blocks:
        stats = blk.attn.last_stats
        assert stats is not None, "per-slide extraction requires capture + forward"

        # cos_y_cls_rbar is (B, H); we're at B=1.
        cos_y_cls_rbar_layers.append(stats["cos_y_cls_rbar"][0].float())   # (H,)
        # h_V_patch is (B, H, N_real). Mean over N axis (skip pad — the
        # slice handed in already excluded pads via N_real).
        mean_h_V_layers.append(stats["h_V_patch"].float().mean(dim=(0, 2)))
        # cos_yr_patch_pre/post are (B, H, N_real).
        mean_cos_yr_pre_layers.append(
            stats["cos_yr_patch_pre"].float().mean(dim=(0, 2))
        )
        mean_cos_yr_post_layers.append(
            stats["cos_zr_patch_post"].float().mean(dim=(0, 2))
        )
        # Optional pre_v fields.
        if stats.get("bar_rho_cls") is not None:
            bar_rho_cls_layers.append(stats["bar_rho_cls"][0].float())
        if stats.get("rho_patch") is not None:
            mean_rho_layers.append(
                stats["rho_patch"].float().mean(dim=(0, 2))
            )

    out: Dict[str, torch.Tensor] = {
        "cos_y_cls_rbar":    torch.stack(cos_y_cls_rbar_layers, dim=0),   # (D, H)
        "mean_h_V":          torch.stack(mean_h_V_layers, dim=0),
        "mean_cos_yr_pre":   torch.stack(mean_cos_yr_pre_layers, dim=0),
        "mean_cos_yr_post":  torch.stack(mean_cos_yr_post_layers, dim=0),
    }
    if bar_rho_cls_layers:
        out["bar_rho_cls"] = torch.stack(bar_rho_cls_layers, dim=0)
    if mean_rho_layers:
        out["mean_rho"]    = torch.stack(mean_rho_layers, dim=0)
    return out


def compute_z_over_y_by_h_morph_quartile(
    model: torch.nn.Module,
    h_morph_slide: torch.Tensor,          # (N_real,) float
) -> "torch.Tensor":
    """
    For the most recent forward pass, bin real patches by their
    h_morph quartile (per-slide quantiles) and compute the mean
    z_over_y within each bin, per layer per head.

    Returns a tensor of shape (4, D, H) where index [q, l, h] is the
    mean z_over_y among patches whose h_morph falls in quartile q
    (0 = lowest, 3 = highest), at layer l head h.

    Bins are computed per-slide (not globally), which is what we want:
    the §8.2.C diagnostic asks whether z_over_y decreases with LOCAL
    homogeneity relative to the slide's own distribution. Global
    quartiles would mix slide-intrinsic heterogeneity with cross-slide
    heterogeneity.
    """
    import numpy as np
    h_np = h_morph_slide.detach().cpu().numpy()
    if h_np.size == 0:
        return torch.zeros(4, len(model.blocks), 1)
    # Per-slide quartile boundaries from empirical quantiles. np.digitize
    # returns 0 for values <= first bin edge, ..., 3 for > last.
    edges = np.quantile(h_np, [0.25, 0.5, 0.75])
    bins = np.digitize(h_np, edges)          # (N_real,) in {0..3}

    per_layer_per_quartile = []
    for blk in model.blocks:
        stats = blk.attn.last_stats
        assert stats is not None, "per-slide quartile computation requires capture"
        # Per-token z_over_y at real-patch rows only.
        y_norm = stats["y_norm"].float()                  # (B, H, L)
        z_norm = stats["z_norm"].float()
        n_cls = int(stats["num_cls_tokens"])
        N_real = int(stats["N_real"])
        z_over_y = (
            z_norm[:, :, n_cls : n_cls + N_real]
            / (y_norm[:, :, n_cls : n_cls + N_real] + 1e-8)
        ).squeeze(0)                                      # (H, N_real)
        z_over_y_np = z_over_y.cpu().numpy()
        H_heads = z_over_y_np.shape[0]
        # Per-quartile mean per head. Fill NaN for empty bins (tiny slides).
        per_q = np.zeros((4, H_heads), dtype=np.float32)
        for q in range(4):
            mask_q = (bins == q)
            if mask_q.any():
                per_q[q] = z_over_y_np[:, mask_q].mean(axis=1)
            else:
                per_q[q] = float("nan")
        per_layer_per_quartile.append(per_q)
    # Stack -> (D, 4, H), transpose to (4, D, H).
    stacked = np.stack(per_layer_per_quartile, axis=0)    # (D, 4, H)
    return torch.from_numpy(stacked.transpose(1, 0, 2)).contiguous()
