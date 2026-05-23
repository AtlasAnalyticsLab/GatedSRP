"""
Single-run training entry for the slide-level SRP PoC (stage 3).

Adapted from slide_level/train.py with minimal surgical changes per
proposal §12.6:

  (1) `--ablation` accepts the stage-3 names:
        baseline | xsa_all_ref | srp_patch_hard | srp_patch_learn
        | srp_patch_preV | srp_patch_gated

  (2) The dataset path now uses SRPSlideFeatureDataset which adds
      neighbor_index, neighbor_mask, and h_morph to every batch.
      Also writes `mean_h_morph` to predictions.csv (the slide-intrinsic
      covariate for the §13.4.3a homogeneity regression).

  (3) The model is chosen by ablation backend:
        xsa_all_ref                  -> stage-2 NystromXSAggregator
                                        (the ONLY xsa-backend run; kept
                                        for stage-2 continuity at α_*=1)
        baseline, srp_patch_*        -> NystromSRPAggregator (stage-3),
                                        baseline with beta_patch_mode=
                                        "zero" (numerically equivalent
                                        to stage-2 α_*=0 per unit tests)
      The SRPSlideFeatureDataset collate always emits SRP tensors; the
      XSA-backend model simply ignores them because it doesn't take
      those positional arguments — we dispatch in `_model_forward`.

  (4) Beta / alpha trajectory logging: we log alpha_patch/alpha_cls for
      the xsa-backend ablation and beta_patch for every SRP-backend
      ablation (including baseline, whose beta is fixed at 0), into
      separately-keyed W&B metrics (alpha_step/* vs beta_step/*).

  (5) At test time, test_artifacts.npz captures alphas OR betas
      depending on which backend was used, under distinct keys.
      SRP-backend runs additionally write per_slide_diagnostics.npz
      with §8.2.D/E per-slide summaries (placement signature,
      attention-weighted retention under pre_v, z_over_y by
      h^morph quartile).

Reuses stage-2 helpers (fold assignment, autocast_ctx, compute_metrics,
cosine_warmup_lr) via direct import.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

import wandb

# Ensure the repository root is importable so `slide_level.*` works from
# direct script execution as well as `python -m` invocation.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Stage-2 imports — used for the xsa_all_ref ablation ONLY (baseline
# now routes through the SRP backend with beta_patch=0, which is
# numerically equivalent to stage-2 α=0 per tests/test_srp_attention::test_I).
from slide_level.src.aggregator import NystromXSAggregator
from slide_level.src.diagnostics import (
    StatsAccumulator as XSAStatsAccumulator,
    autocast_ctx,
    extract_alpha_values,
    extract_batch_stats as extract_batch_stats_xsa,
    set_capture_mode as set_capture_mode_xsa,
)

# Stage-3 (SRP) imports.
from slide_level_srp.src.diff_transformer import NystromDiffTransformerAggregator
from slide_level_srp.src.srp_aggregator import NystromSRPAggregator
from slide_level_srp.src.srp_diagnostics import (
    StatsAccumulator as SRPStatsAccumulator,
    compute_z_over_y_by_h_morph_quartile,
    extract_batch_stats as extract_batch_stats_srp,
    extract_beta_values,
    extract_layerscale_values,
    extract_per_slide_diagnostics,
    set_capture_mode as set_capture_mode_srp,
)
from slide_level_srp.data_ext import (
    build_srp_loaders_for_fold,
    build_fold_assignments,
    enumerate_slides,
)


# --- Method-parameter classifier (Phase-A.9 review fixes F2 + F1-3rd) ---
# Single source of truth for "what counts as a method-specific parameter
# for this ablation". Used by:
#   - --freeze_others (the conditional-optimum probe)
#   - method-parameter counts / names in the startup log
#   - optimizer weight-decay grouping
#   - --ab_lr_mult LR scaling (Phase-A.9 third-review fix F1)
#   - W&B / artifact logging
#
# For non-signed-gated ablations, the method scalars are alpha/beta. For
# `srp_patch_signed_gated`, the actual learnables live under `gate.*`
# (e.g. blocks.{i}.attn.gate.token_mlp_*.weight, gate.head_weight,
# gate.head_bias, gate.layer_head_bias) and `beta_patch` becomes a
# non-trainable buffer. Pre-fix, freeze_others on a signed-gated run
# would freeze every gate parameter → zero method params trainable.
_SIGNED_GATE_ABLATIONS = {
    "srp_patch_signed_gated",
    "srp_patch_signed_gated_pre_q",
    "srp_patch_signed_gated_pre_k",
    "srp_patch_signed_gated_pre_v",
}
_LEARNED_R_SIGNED_GATE_ABLATIONS = {"srp_patch_signed_gated_learned_r"}
_RCD_ABLATIONS = {"srp_patch_rcd", "srp_patch_rcd_learned_r"}
_MLP_CONTROL_ABLATIONS = {"srp_patch_mlp_control"}
_SIGNED_GATE_SRP_MODES = {
    "post_agg_signed_gated",
    "post_agg_signed_gated_learned_r",
    "pre_q_signed_gated",
    "pre_k_signed_gated",
    "pre_v_signed_gated",
}
_GATE_L2_ABLATIONS = _SIGNED_GATE_ABLATIONS | _LEARNED_R_SIGNED_GATE_ABLATIONS
_METHOD_SURFACE_ABLATIONS = (
    _SIGNED_GATE_ABLATIONS
    | _LEARNED_R_SIGNED_GATE_ABLATIONS
    | _RCD_ABLATIONS
    | _MLP_CONTROL_ABLATIONS
)


def is_method_param(name: str, ablation: str) -> bool:
    """Return True iff `name` is a method-specific learnable parameter
    for the given ablation."""
    if ablation in _SIGNED_GATE_ABLATIONS:
        # Match per-block gate sub-modules; e.g.
        # `blocks.0.attn.gate.head_weight`,
        # `blocks.2.attn.gate.token_mlp_hidden.bias`.
        return ".gate." in name or name.startswith("gate.")
    if ablation in _LEARNED_R_SIGNED_GATE_ABLATIONS:
        # Standalone Method 2.4 is the original signed-gate surface plus a
        # learned local-r scorer.  It deliberately excludes rcd_recomposer.*
        # so freeze/method-only probes cannot accidentally train Method 2.1.
        return (
            ".gate." in name
            or name.startswith("gate.")
            or ".context_scorer." in name
            or name.startswith("context_scorer.")
        )
    if ablation in _RCD_ABLATIONS:
        # Method 2.1 lives under rcd_recomposer.  Method 2.4 adds the
        # context_scorer surface.  beta_patch is a non-trainable zero
        # buffer for these ablations and should not be treated as the
        # learnable method mechanism.
        return (
            ".rcd_recomposer." in name
            or name.startswith("rcd_recomposer.")
            or ".context_scorer." in name
            or name.startswith("context_scorer.")
        )
    if ablation in _MLP_CONTROL_ABLATIONS:
        # Paper-ready mechanism-vs-capacity control: the only method surface
        # is the no-geometry plain adapter.  beta_patch is a frozen zero
        # buffer and should not be counted as a trainable SRP mechanism.
        return ".mlp_control." in name or name.startswith("mlp_control.")
    # All other ablations: classical alpha/beta scalars.
    return ("alpha_cls" in name or "alpha_patch" in name or
            "beta_patch" in name)


def is_method_no_decay(name: str, ablation: str) -> bool:
    """Return True iff `name` is a method-specific param that should
    receive `weight_decay=0`. This is a SUBSET of `is_method_param` —
    every method no-decay param is a method param, but not vice versa.

    For non-signed-gated ablations: all alpha/beta scalars are no-decay.
    For signed-gated: only gate **biases** (`*.layer_head_bias`,
    `*.head_bias`, gate-internal `*.bias`) are no-decay; gate **weights**
    receive the normal weight_decay (proposal §6.4 specifies that biases
    initialized at zero must not be pulled back to zero by AdamW decay,
    but weights have no such constraint).
    """
    if ablation in _SIGNED_GATE_ABLATIONS:
        return (
            ".gate." in name
            and (name.endswith(".layer_head_bias")
                 or name.endswith(".head_bias")
                 or name.endswith(".bias"))
        )
    if ablation in _LEARNED_R_SIGNED_GATE_ABLATIONS:
        return (
            is_method_param(name, ablation)
            and (name.endswith(".layer_head_bias")
                 or name.endswith(".head_bias")
                 or name.endswith(".bias"))
        )
    if ablation in _RCD_ABLATIONS:
        return (
            is_method_param(name, ablation)
            and (name.endswith(".bias") or name.endswith("_diag_delta"))
        )
    if ablation in _MLP_CONTROL_ABLATIONS:
        return is_method_param(name, ablation) and name.endswith(".bias")
    return ("alpha_cls" in name or "alpha_patch" in name or
            "beta_patch" in name)


# --- Ablation spec ------------------------------------------------------

# Map --ablation to (backend, kwargs for the model-specific constructor).
# backend == "xsa":  use stage-2 NystromXSAggregator, forwarded(features).
# backend == "diff": use Diff Transformer comparator, forwarded(features).
# backend == "srp":  use stage-3 NystromSRPAggregator, forwarded(features,
#                    neighbor_index, neighbor_mask, h_morph=h_morph).
_ABLATIONS = {
    # baseline now runs through the SRP backend with beta_patch=zero so
    # SRP-specific diagnostics (cos(y, r), h^V, cos(y_cls, r̄)) are
    # captured against the same reference point the SRP variants are
    # compared with. Unit + aggregator tests (tests/test_srp_attention
    # test_I, tests/test_srp_aggregator test_B) confirm that beta=0 SRP
    # is numerically equivalent to alpha=0 XSA, so the stage-2 baseline
    # comparison is still clean.
    "baseline": {
        "backend": "srp",
        "beta_patch_mode": "zero", "srp_mode": "post_agg",
    },
    # xsa_all_ref keeps the stage-2 backend because it tests the original
    # XSA formulation (self-direction projection, not SRP).
    "xsa_all_ref": {
        "backend": "xsa",
        "alpha_cls_mode": "one", "alpha_patch_mode": "one",
    },
    # Diff Transformer comparator.  The implementation keeps the official
    # differential-attention operation (two Q/K maps, lambda subtraction,
    # per-head RMSNorm, half-head convention) but uses the same Nyström
    # factorization as the rest of this WSI codebase so it remains memory-safe
    # for slide bags with very large token counts.
    "diff_transformer": {
        "backend": "diff",
    },
    "srp_patch_hard": {
        "backend": "srp",
        "beta_patch_mode": "one", "srp_mode": "post_agg",
    },
    "srp_patch_learn": {
        "backend": "srp",
        "beta_patch_mode": "learn", "srp_mode": "post_agg",
    },
    "srp_patch_preV": {
        "backend": "srp",
        "beta_patch_mode": "learn", "srp_mode": "pre_v",
    },
    "srp_patch_gated": {
        "backend": "srp",
        "beta_patch_mode": "learn", "srp_mode": "post_agg_gated",
    },
    # --- Phase-1.5: β-init sweep (proposal §5.2, §Phase-1.5 RESULTS §10) ---
    # Goal: disambiguate whether β = init is a local minimum (β drifts back
    # to ≈1 from any start) or the β gradient is genuinely too weak
    # (β stays near its init regardless). All three share β_mode=learn
    # and srp_mode=post_agg so the only axis varied is the init. Each
    # ablation pins beta_init via the spec, so the launcher doesn't need
    # to thread --beta_init on the CLI.
    "srp_patch_learn_init0": {
        "backend": "srp",
        "beta_patch_mode": "learn", "srp_mode": "post_agg",
        "beta_init": 0.0,
    },
    "srp_patch_learn_init05": {
        "backend": "srp",
        "beta_patch_mode": "learn", "srp_mode": "post_agg",
        "beta_init": 0.5,
    },
    "srp_patch_learn_init2": {
        "backend": "srp",
        "beta_patch_mode": "learn", "srp_mode": "post_agg",
        "beta_init": 2.0,
    },
    # --- Phase-1.5 β-grid: fixed β at values not already covered by
    # Phase-1 baselines/hard. The β-init sweep showed β stays at init
    # throughout training (drift < 0.003), so "fixed β" and "learnable
    # β at init=X" are empirically interchangeable; we use "fixed"
    # for the gridsearch to make intent explicit and eliminate the
    # small AdamW-state perturbation caused by declaring β as a
    # trainable parameter (see RESULTS.md §β-init).
    "srp_patch_fixed_beta0": {
        "backend": "srp",
        "beta_patch_mode": "fixed", "srp_mode": "post_agg",
        "beta_init": 0.0,
    },
    "srp_patch_fixed_beta05": {
        "backend": "srp",
        "beta_patch_mode": "fixed", "srp_mode": "post_agg",
        "beta_init": 0.5,
    },
    "srp_patch_fixed_beta1": {
        "backend": "srp",
        "beta_patch_mode": "fixed", "srp_mode": "post_agg",
        "beta_init": 1.0,
    },
    "srp_patch_fixed_beta15": {
        "backend": "srp",
        "beta_patch_mode": "fixed", "srp_mode": "post_agg",
        "beta_init": 1.5,
    },
    "srp_patch_fixed_beta2": {
        "backend": "srp",
        "beta_patch_mode": "fixed", "srp_mode": "post_agg",
        "beta_init": 2.0,
    },
    "srp_patch_fixed_beta25": {
        "backend": "srp",
        "beta_patch_mode": "fixed", "srp_mode": "post_agg",
        "beta_init": 2.5,
    },
    "srp_patch_fixed_betaneg1": {
        "backend": "srp",
        "beta_patch_mode": "fixed", "srp_mode": "post_agg",
        "beta_init": -1.0,
    },
    # Gated variant with β_base = 2 (instead of Phase-1's β_base = 1).
    # Effective β_i = 2 · clamp(h_morph_i, 0, 1). On average,
    # mean(h_morph) ≈ 0.61, so effective β ≈ 1.22 — lies in the
    # "between full projection and full reflection" regime. Tests
    # whether the reflection mechanism (β=2 winner from β-init) stacks
    # with the gate mechanism (gated winner from Phase-1). [Phase-1.5+
    # result: does NOT stack, p=0.49 vs β=2 fixed; see RESULTS.md §14.4.]
    "srp_patch_gated_beta2": {
        "backend": "srp",
        "beta_patch_mode": "fixed", "srp_mode": "post_agg_gated",
        "beta_init": 2.0,
    },
    # Pre-aggregation reflection (Phase-1.5+ final follow-up).
    # v'_j = v_j - 2·(v_j·r̂_j)·r̂_j applied BEFORE Nyström aggregation.
    # β̃=1 pre_v was catastrophic in Phase-1 (F1=0.49) because it
    # projected away ~50% of each v_j's magnitude (ρ=0.48), destroying
    # signal alongside redundancy. β̃=2 (reflection) is norm-preserving
    # (ρ=1.0 by construction) — tests whether reflection rescues pre-V
    # the way it rescued post-agg SRP, or whether pre-V fails for a
    # different reason (e.g., CLS's aggregation can't recover even
    # when per-patch magnitudes are preserved).
    "srp_patch_preV_beta2": {
        "backend": "srp",
        "beta_patch_mode": "fixed", "srp_mode": "pre_v",
        "beta_init": 2.0,
    },
    # --- Phase-A signed-gate (LEARNED_GATE_SRP_PROPOSAL.md §2). The
    # `beta_patch_mode='signed_gated'` + `srp_mode='post_agg_signed_gated'`
    # pair is mandatory; the aggregator asserts this. delta_scale is a
    # CLI flag (--delta_scale) so the same ablation can be run at
    # δ=1 (signed-projection range) or δ=2 (full reflection range).
    "srp_patch_signed_gated": {
        "backend": "srp",
        "beta_patch_mode": "signed_gated",
        "srp_mode": "post_agg_signed_gated",
    },
    # --- Pre-attention signed-gated SRP placement ablations. These keep
    # the same TokenHeadGate surface as srp_patch_signed_gated but move the
    # signed SRP operator before attention: after Q, after K, or after V.
    # The aggregator keeps final-block gates active only for pre-K/pre-V,
    # because those streams have a same-block path into the CLS readout.
    "srp_patch_signed_gated_pre_q": {
        "backend": "srp",
        "beta_patch_mode": "signed_gated",
        "srp_mode": "pre_q_signed_gated",
    },
    "srp_patch_signed_gated_pre_k": {
        "backend": "srp",
        "beta_patch_mode": "signed_gated",
        "srp_mode": "pre_k_signed_gated",
    },
    "srp_patch_signed_gated_pre_v": {
        "backend": "srp",
        "beta_patch_mode": "signed_gated",
        "srp_mode": "pre_v_signed_gated",
    },
    # --- Standalone Method 2.4: learned local redundancy direction on the
    # original signed-gate intervention.  This is intentionally NOT RCD:
    # z_i = y_i - beta_eff_i * (y_i^T r_hat_i^theta) r_hat_i^theta.
    # It lets Phase14b isolate whether learning r_hat helps without changing
    # the signed-gate formula that Phase10/11/12 already evaluated.
    "srp_patch_signed_gated_learned_r": {
        "backend": "srp",
        "beta_patch_mode": "signed_gated",
        "srp_mode": "post_agg_signed_gated_learned_r",
    },
    # --- Method 2.1: corrected residual/common decomposition (RCD).
    # beta_patch is explicitly disabled so the intervention is isolated to
    # non-shared common/residual branch maps.  The RCD output branch is
    # identity-safe at init, so this starts from the no-gate baseline.
    "srp_patch_rcd": {
        "backend": "srp",
        "beta_patch_mode": "zero",
        "srp_mode": "post_agg_rcd",
    },
    # --- Method 2.4 integrated with Method 2.1.  The local context
    # direction remains neighbourhood-local but learns additive logits over
    # the existing neighbour weights; zero init recovers the fixed-neighbour
    # RCD direction exactly.
    "srp_patch_rcd_learned_r": {
        "backend": "srp",
        "beta_patch_mode": "zero",
        "srp_mode": "post_agg_rcd_learned_r",
    },
    # --- Paper-ready capacity control: parameter-matched plain adapter.
    # z_i = y_i + Adapter(y_i), with Adapter's output path zero-initialized
    # and its bottleneck chosen to approximately match the current gate's
    # parameter count.  Unlike signed-gated SRP/RCD, this branch never receives
    # neighbour vectors, r_hat, h_local, or h_morph for the model output.
    "srp_patch_mlp_control": {
        "backend": "srp",
        "beta_patch_mode": "zero",
        "srp_mode": "post_agg_mlp_control",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(
    step: int, warmup_steps: int, total_steps: int,
    base_lr: float, min_lr: float = 0.0,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def compute_metrics(y_true, y_pred, y_prob, num_classes: int) -> dict:
    """Classification metrics (accuracy, macro P/R/F1, macro OvR AUC)."""
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="macro", zero_division=0,
        labels=list(range(num_classes)),
    )
    metrics = {
        "acc": float(acc), "precision": float(p),
        "recall": float(r), "f1": float(f1),
    }
    # Renormalize under bf16 to keep sklearn's row-sum check happy.
    row_sums = y_prob.sum(axis=-1, keepdims=True)
    y_prob = y_prob / np.maximum(row_sums, 1e-12)
    present = np.unique(y_true)
    if present.size < num_classes:
        metrics["auc"] = float("nan")
    elif num_classes == 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    else:
        metrics["auc"] = float(roc_auc_score(
            y_true, y_prob,
            multi_class="ovr", average="macro",
            labels=list(range(num_classes)),
        ))
    return metrics


# --- Forward dispatch ---------------------------------------------------

def _compute_h_local_torch(
    feats: torch.Tensor,                # (B, N, D) raw input features
    nbi: torch.Tensor,                  # (B, N, K) int64 neighbour indices, -1 invalid
    nbm: torch.Tensor,                  # (B, N, K) bool neighbour mask
) -> torch.Tensor:
    """
    Per-patch local homogeneity = mean cos(x_i, x_j) over valid
    neighbours j ∈ N(i). Returned shape (B, N), dtype matching `feats`.

    Mirrors `src/data_panda.py::_compute_h_local` but operates on
    PyTorch tensors and supports the (B, N, K) variable-length CAM17
    neighbour graph. K is 8 for the default 3x3 window, 24 for 5x5,
    48 for 7x7, etc. Computed once per forward (no caching) — cheap
    relative to the Nyström attention path.

    Used as a per-token diagnostic input to the signed gate
    (LEARNED_GATE_SRP_PROPOSAL.md §6.2). h_local is intrinsic to the
    raw input features and shared across all aggregator blocks.
    """
    eps = 1e-12
    # Normalise once; cosine = dot of normalised vectors.
    f_norm = feats / (feats.norm(dim=-1, keepdim=True) + eps)
    # Clamp -1 indices to 0 for safe gather (masked out below).
    safe_idx = nbi.clamp(min=0)                                   # (B, N, K)
    B, N, _ = feats.shape
    # Match the actual neighbour slot count instead of assuming 3x3/8 slots.
    # Larger neighbour-window ablations (5x5/7x7) produce more slots and must
    # share the same h_local path as the default signed-gate baseline.
    K = int(safe_idx.shape[-1])
    batch_idx = torch.arange(B, device=feats.device).view(B, 1, 1).expand(B, N, K)
    nbr = f_norm[batch_idx, safe_idx, :]                          # (B, N, K, D)
    cos = (nbr * f_norm[:, :, None, :]).sum(dim=-1)               # (B, N, K)
    cos = cos * nbm.to(cos.dtype)
    cnt = nbm.sum(dim=-1).to(cos.dtype).clamp(min=1.0)            # (B, N)
    return cos.sum(dim=-1) / cnt                                  # (B, N)


def _model_forward(model, batch, backend: str, device, ablation_spec=None):
    """
    Run the model forward for one collated batch.

    backend == "xsa"  -> stage-2 model; only `features` consumed.
    backend == "diff" -> Diff Transformer comparator; only `features`
                         consumed.
    backend == "srp" -> stage-3 model; neighbor_index, neighbor_mask, and
                        h_morph threaded through. Under
                        any signed-gated SRP mode, or
                        'post_agg_rcd_learned_r', `h_local` is also computed
                        on-the-fly from the input features and the neighbour
                        graph.

    For efficiency the SRP tensors are moved to device only when backend
    == "srp"; otherwise they are left on CPU in the batch dict.

    `ablation_spec` is the entry from `_ABLATIONS[args.ablation]`. We
    only need it to detect whether the SRP mode requires h_local; if
    None (legacy callers) we behave as if signed-gated is off.
    """
    feats = batch["features"].to(device, non_blocking=True)
    if backend in {"xsa", "diff"}:
        return model(feats)
    # SRP path.
    neighbor_index = batch["neighbor_index"].to(device, non_blocking=True)
    neighbor_mask  = batch["neighbor_mask"].to(device, non_blocking=True)
    neighbor_weight = batch.get("neighbor_weight")
    if neighbor_weight is not None:
        neighbor_weight = neighbor_weight.to(device, non_blocking=True)
    h_morph        = batch["h_morph"].to(device, non_blocking=True)
    needs_h_local = bool(
        ablation_spec is not None
        and (
            ablation_spec.get("srp_mode") in _SIGNED_GATE_SRP_MODES
            or ablation_spec.get("srp_mode") == "post_agg_rcd_learned_r"
        )
    )
    if needs_h_local:
        h_local = _compute_h_local_torch(feats, neighbor_index, neighbor_mask)
        return model(
            feats, neighbor_index, neighbor_mask,
            h_morph=h_morph, h_local=h_local,
            neighbor_weight=neighbor_weight,
        )
    return model(
        feats, neighbor_index, neighbor_mask,
        h_morph=h_morph,
        neighbor_weight=neighbor_weight,
    )


def run_eval(
    model, loader, device, *, num_classes: int,
    backend: str,
    capture_stats: bool, max_stats_batches,
    autocast_dtype=torch.bfloat16,
    collect_per_slide: bool = False,
    collect_per_slide_diagnostics: bool = False,
    ablation_spec: dict | None = None,
    collect_gate_stats: bool = False,
):
    """
    Eval loop for K-class classification. Unified across backends; the
    capture / stats-accumulator machinery is selected via `backend`.

    If `collect_per_slide_diagnostics` is True (test time only), also
    compute the per-slide SRP diagnostic bundle that §8.2.D/E and
    §13.4.3a's homogeneity regression need — cos(y_cls, r̄) per (L, H),
    bar_rho_cls per (L, H), z_over_y by h^morph quartile per (L, H),
    mean_h_morph, mean_h_V. XSA-backend runs get zeros for the SRP
    fields, keeping the output schema uniform. The per-slide pack is
    returned as the fourth element.
    """
    model.eval()
    if backend == "xsa":
        set_capture_mode_fn = set_capture_mode_xsa
        extract_stats_fn = extract_batch_stats_xsa
        Acc = XSAStatsAccumulator
    elif backend == "srp":
        set_capture_mode_fn = set_capture_mode_srp
        extract_stats_fn = extract_batch_stats_srp
        Acc = SRPStatsAccumulator
    else:
        # Diff Transformer has no XSA/SRP diagnostic extractor.  Keep metrics
        # and logits collection active, but skip attention-diagnostic capture so
        # the comparator can run without pretending to emit incompatible stats.
        capture_stats = False
        set_capture_mode_fn = None
        extract_stats_fn = None
        Acc = None

    if capture_stats:
        assert set_capture_mode_fn is not None
        set_capture_mode_fn(model, True)

    # Phase-A.9 calibration-reframe (CALIBRATION_REFRAME_2026-04-28.md):
    # Accumulate raw logits in addition to softmax probs so test_artifacts.npz
    # can carry pre-softmax outputs for downstream ECE / Brier / temperature-
    # scaling analysis.
    all_labels, all_probs, all_preds, all_logits = [], [], [], []
    loss_sum = 0.0
    n = 0
    stats_acc = Acc() if capture_stats and Acc is not None else None
    stats_done = 0
    capture_active = capture_stats
    per_slide: list[dict] = []
    # Per-slide diagnostic records: one dict per test slide.
    per_slide_diag: list[dict] = []
    # Per-example signed-gate accumulator. No-op for non-signed-gated
    # runs because the model's _last_gate_stats stays None on those
    # forwards.
    if collect_gate_stats:
        from slide_level_srp.src.gate_signed import GateStatsAccumulator
        gate_acc = GateStatsAccumulator()
    else:
        gate_acc = None

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)
            with autocast_ctx(device, autocast_dtype):
                logits = _model_forward(
                    model, batch, backend, device,
                    ablation_spec=ablation_spec,
                )
                loss_per_slide = F.cross_entropy(
                    logits.float(), labels, reduction="sum",
                )
            if gate_acc is not None:
                gate_acc.update(model)
            prob = F.softmax(logits.float(), dim=-1)
            pred = prob.argmax(dim=-1)

            all_probs.append(prob.cpu())
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.float().detach().cpu())
            loss_sum += float(loss_per_slide.item())
            n += labels.numel()

            # Per-slide covariate for the homogeneity regression
            # (§13.4.3a). Computed from the raw UNI-derived h^morph,
            # slide-intrinsic regardless of ablation.
            mean_h_morph = float(batch["h_morph"].mean().item()) if "h_morph" in batch else float("nan")

            if collect_per_slide:
                logits_np = logits.float().squeeze(0).cpu().numpy()
                prob_np = prob.squeeze(0).cpu().numpy()
                per_slide.append({
                    "slide_id": batch["slide_id"][0],
                    "patient_id": batch["patient_id"][0],
                    "center": int(batch["center"][0]),
                    "N_tokens": int(batch["n_tokens"][0]),
                    "y_true": int(labels.item()),
                    "y_pred_class": int(pred.item()),
                    "y_logits": logits_np.tolist(),
                    "y_probs": prob_np.tolist(),
                    "per_slide_loss": float(loss_per_slide.item()),
                    "mean_h_morph": mean_h_morph,
                })

            if capture_active:
                assert stats_acc is not None
                assert extract_stats_fn is not None
                assert set_capture_mode_fn is not None
                stats_acc.update(extract_stats_fn(model))
                stats_done += 1
                if max_stats_batches is not None and stats_done >= max_stats_batches:
                    set_capture_mode_fn(model, False)
                    capture_active = False

            # Per-slide SRP diagnostic bundle (test-time only). Captured
            # BEFORE capture_active could flip off — the check uses the
            # local `capture_active` variable's *previous* state this
            # iteration, which is fine because we only care that capture
            # is still on right now.
            if collect_per_slide_diagnostics and backend == "srp":
                psd = extract_per_slide_diagnostics(model)
                # z_over_y by h^morph quartile: cheap per-slide bin.
                z_by_q = compute_z_over_y_by_h_morph_quartile(
                    model, batch["h_morph"].squeeze(0),
                )
                record = {
                    "slide_id": batch["slide_id"][0],
                    "patient_id": batch["patient_id"][0],
                    "center": int(batch["center"][0]),
                    "N_tokens": int(batch["n_tokens"][0]),
                    "y_true": int(labels.item()),
                    "y_pred_class": int(pred.item()),
                    "mean_h_morph": mean_h_morph,
                    "cos_y_cls_rbar": psd["cos_y_cls_rbar"].cpu().numpy(),
                    "mean_h_V":       psd["mean_h_V"].cpu().numpy(),
                    "mean_cos_yr_pre":  psd["mean_cos_yr_pre"].cpu().numpy(),
                    "mean_cos_yr_post": psd["mean_cos_yr_post"].cpu().numpy(),
                    "z_over_y_by_h_morph_quartile": z_by_q.cpu().numpy(),  # (4, D, H)
                }
                if "bar_rho_cls" in psd:
                    record["bar_rho_cls"] = psd["bar_rho_cls"].cpu().numpy()
                if "mean_rho" in psd:
                    record["mean_rho"] = psd["mean_rho"].cpu().numpy()
                per_slide_diag.append(record)

    if capture_active:
        assert set_capture_mode_fn is not None
        set_capture_mode_fn(model, False)

    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_logits = torch.cat(all_logits).numpy() if all_logits else None
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    metrics["loss"] = loss_sum / max(1, n)
    cls_stats = stats_acc.result() if (capture_stats and stats_done > 0) else None
    gate_stats = gate_acc.finalize() if gate_acc is not None else {}
    # Bundle (y_true, y_logits) into the metrics dict for the caller to
    # save into test_artifacts.npz under top-level npz keys. Keeping them
    # off the metrics dict (which is JSON-serialised) preserves the
    # existing W&B / log path; the caller pulls them off explicitly.
    eval_arrays = {"y_true": y_true, "y_logits": y_logits}
    return metrics, cls_stats, per_slide, per_slide_diag, gate_stats, eval_arrays


# --- W&B scalar logging for alphas / betas ------------------------------

def log_alpha_scalars(alphas, prefix: str, step: int) -> None:
    """Emit per-(role, layer, head) + role-wise-mean scalars for alpha."""
    payload = {}
    for role, arr in alphas.items():
        arr_np = arr.detach().cpu().numpy()
        depth, n_heads = arr_np.shape
        for li in range(depth):
            for hi in range(n_heads):
                payload[f"{prefix}/{role}_L{li}_H{hi}"] = float(arr_np[li, hi])
        payload[f"{prefix}/{role}_mean"] = float(arr_np.mean())
    wandb.log(payload, step=step)


def log_beta_scalars(betas, prefix: str, step: int) -> None:
    """Emit per-(layer, head) + mean scalars for beta_patch."""
    payload = {}
    for role, arr in betas.items():
        arr_np = arr.detach().cpu().numpy()
        depth, n_heads = arr_np.shape
        for li in range(depth):
            for hi in range(n_heads):
                payload[f"{prefix}/{role}_L{li}_H{hi}"] = float(arr_np[li, hi])
        payload[f"{prefix}/{role}_mean"] = float(arr_np.mean())
    wandb.log(payload, step=step)


# --- CLI ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--wandb_project", type=str, default="GatedSRP_slide")
    p.add_argument("--wandb_mode", type=str, default="disabled",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--out_dir", type=str, default="./runs")

    p.add_argument("--ablation", type=str, required=True,
                   choices=list(_ABLATIONS.keys()),
                   help="One of: " + ", ".join(_ABLATIONS.keys()))
    p.add_argument("--beta_init", type=float, default=1.0,
                   help="Init value for learnable beta_patch (SRP backends only).")
    # Signed-gate parameters (LEARNED_GATE_SRP_PROPOSAL.md §2.1, §2.2).
    # Only consumed under signed-gated SRP ablations; ignored otherwise.
    # Default δ=2 covers identity / anti-SRP / projection / reflection in
    # [-2, +2]; pass --delta_scale 1.0 for signed-projection-only variants.
    p.add_argument("--delta_scale", type=float, default=2.0,
                   help="Signed-gate range bound: β_eff = δ · tanh(raw). "
                        "Default 2.0 covers reflection. "
                        "Used with signed-gated SRP ablations.")
    p.add_argument("--gate_hidden_dim", type=int, default=16,
                   help="Token-MLP hidden width for the signed gate.")
    p.add_argument("--gate_output_init", type=str, default="zero",
                   choices=["zero", "tiny_normal", "xavier_uniform",
                            "kaiming_uniform", "orthogonal", "constant_beta"])
    p.add_argument("--gate_output_init_scale", type=float, default=1.0)
    p.add_argument("--gate_init_beta0", type=float, default=0.0)
    p.add_argument("--gate_activation", type=str, default="tanh",
                   choices=["tanh", "scaled_sigmoid", "sigmoid01",
                            "softsign", "hardtanh", "atan"])
    p.add_argument("--gate_activation_temperature", type=float, default=1.0)
    p.add_argument("--gate_factorization", type=str, default="full",
                   choices=["full", "token_only", "head_only", "no_bias"],
                   help="Signed-gate factorization ablation. full uses token "
                        "MLP + head diagnostics + biases; token_only drops "
                        "head diagnostics; head_only drops token diagnostics; "
                        "no_bias keeps token/head terms but removes head and "
                        "layer-head biases.")
    p.add_argument("--gate_count_features", type=str, default="legacy",
                   choices=["legacy", "rawlog", "normlog", "none"])
    p.add_argument("--gate_l2_reg", type=float, default=0.0,
                   help="Optional signed-gate containment loss. When >0, "
                        "adds gate_l2_reg * mean(beta_eff^2) to the "
                        "training objective for signed-gated SRP runs.")
    p.add_argument("--rcd_adapter_kind", type=str, default="lowrank",
                   choices=["lowrank", "diag"],
                   help="Method 2.1 branch map type for RCD ablations. "
                        "lowrank is the primary design; diag is a lower-"
                        "capacity control.")
    p.add_argument("--rcd_rank", type=int, default=16,
                   help="Low-rank bottleneck for Method 2.1 RCD branch maps.")
    p.add_argument("--learned_r_hidden_dim", type=int, default=16,
                   help="Hidden width for the Method 2.4 local context "
                        "direction scorer.")
    p.add_argument("--srp_freeze_epochs", type=int, default=0,
                   help="Initial epochs with SRP/gate/RCD method parameters "
                        "frozen. Intended for two-stage method-surface runs.")
    p.add_argument("--stage2_epochs", type=int, default=0,
                   help="Extra epochs after the base --epochs budget. At the "
                        "stage-2 boundary the best stage-1 checkpoint is "
                        "reloaded, trainability is reset according to "
                        "--stage2_mode, and the optimizer is rebuilt.")
    p.add_argument("--stage2_mode", type=str, default="joint",
                   choices=["joint", "srp_only"],
                   help="Stage-2 trainability policy. joint trains all "
                        "parameters; srp_only trains only SRP/gate method "
                        "parameters.")
    p.add_argument("--stage2_lr_mult", type=float, default=1.0,
                   help="Multiplier applied to the base LR schedule during "
                        "stage 2. Primary protocol keeps this at 1.0.")
    p.add_argument("--no_detach_gate_inputs", action="store_true",
                   help="Disable proposal §6.3 detach convention — let "
                        "gradients flow through gate diagnostic inputs "
                        "(cos_yr / y_norms) into y. Default is the "
                        "spec-compliant detached regime; this flag "
                        "enables the empirically better-performing "
                        "'live' regime per §6.3.1.")
    p.add_argument("--no_ppeg", action="store_true",
                   help="Phase-A.9 ablation (REPORT.md §17.2): replace "
                        "PPEG with nn.Identity to test whether the "
                        "δ=2 reflection regime's productivity on "
                        "TransMIL is PPEG-mediated. Default is the "
                        "TransMIL-faithful PPEG-on path.")
    p.add_argument("--pos_mode", type=str, default="original",
                   choices=["original", "none"],
                   help="TransMIL positional policy. original=PPEG on; none=PPEG off.")
    p.add_argument("--neighbor_window", type=int, default=3,
                   help="Odd spatial neighbor window size; 3, 5, or 7 in this study.")
    p.add_argument("--neighbor_shell", type=str, default="cumulative",
                   choices=["cumulative", "ring"])
    p.add_argument("--neighbor_source", type=str, default="real",
                   choices=["real", "shuffled"])
    p.add_argument("--neighbor_shuffle_seed", type=int, default=0)
    p.add_argument("--neighbor_weighting", type=str, default="uniform",
                   choices=["uniform", "gaussian", "inverse_distance"])
    p.add_argument("--neighbor_weight_sigma", type=float, default=1.0)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--checkpoint_mode", type=str, default="whole_block",
                   choices=["whole_block", "per_module", "off"])
    p.add_argument("--layerscale_init", type=float, default=0.0,
                   help="CaiT LayerScale init for attention/MLP residual "
                        "branch vectors. 0.0 disables LayerScale and "
                        "preserves the historical model parameter surface; "
                        "0.1 is CaiT's shallow-depth setting.")
    p.add_argument("--ln_specialization", type=str, default="shared",
                   choices=["shared", "cls_patch"],
                   help="[CLS]/patch LayerNorm specialization. shared keeps "
                        "plain nn.LayerNorm and historical checkpoint keys; "
                        "cls_patch uses separate affine parameters for CLS "
                        "and patch tokens.")
    p.add_argument("--ln_specialization_scope", type=str, default="block",
                   choices=["block", "block_final"],
                   help="Where to apply specialized LN. block specializes "
                        "SRPBlock norm1/norm2 only; block_final also "
                        "specializes the final pre-head norm.")

    p.add_argument("--split_mode", type=str, default="cv_fold",
                   choices=["cv_fold", "global_seed_holdout"],
                   help="cv_fold keeps the historical 5-fold protocol and "
                        "requires --fold. global_seed_holdout builds one "
                        "train/val/test split directly from --global_seed "
                        "and deliberately rejects --fold so corrected reruns "
                        "cannot silently reuse a fixed fold.")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--fold_seed", type=int, default=0)
    p.add_argument("--global_seed", type=int, default=None,
                   help="Single seed controlling the corrected holdout split, "
                        "training RNG, and neighbor-shuffle RNG. Required for "
                        "--split_mode global_seed_holdout.")
    p.add_argument("--val_patients_per_fold", type=int, default=10)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--base_lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--train_cap", type=int, default=None)
    p.add_argument("--val_cap", type=int, default=None)
    p.add_argument("--test_cap", type=int, default=None)

    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--num_landmarks", type=int, default=64)
    p.add_argument("--pinv_iterations", type=int, default=6)
    # --- Dataset dispatch (cam17 = CAM17 4-class default; cam16 =
    # binary tumor/normal using the configured CAM16 feature root).
    p.add_argument("--dataset", type=str, default="cam17",
                   choices=["cam17", "cam17_univ2", "cam16", "cam16_univ2", "kgh", "bracs"],
                   help="Dataset adapter to use. cam17 = original "
                        "Stage-3 4-class slide classification; cam16 = "
                        "binary tumor/normal CAM16 on ViT-B/16 features; "
                        "cam17_univ2 = CAM17 4-class on 1536-d UNI-v2; "
                        "cam16_univ2 = binary CAM16 on 1536-d UNI-v2; "
                        "kgh = KGH UNI-v2 biopsy classification; "
                        "bracs = BRACS WSI-level 7-class UNI-v2 classification.")
    p.add_argument("--kgh_true_4class", action="store_true",
                   help="Keep KGH as the intended disease-only 4-output task. "
                        "Without this opt-in, legacy KGH runs preserve the "
                        "historical 5-output compatibility path so old "
                        "phase00b artifacts remain reproducible.")
    p.add_argument("--in_dim", type=int, default=1024,
                   help="Per-patch feature dim. legacy CAM17 UNI-v1 = 1024; "
                        "CAM17 UNI-v2 = 1536; "
                        "CAM16 ViT-B/16 = 768; BRACS/KGH/CAM16 UNI-v2 = 1536. Must match the features "
                        "stored in the h5 files.")
    p.add_argument("--feature_root", type=str, default=None,
                   help="Dataset-specific feature root override for CAM16 UNI-v2/KGH/BRACS.")
    p.add_argument("--feature_key", type=str, default=None,
                   help="H5 feature key override, e.g. features/uni_v2.")

    p.add_argument("--val_stats_batches", type=int, default=10)

    # --- Probe arguments (added for the "is α/β stuck or co-adapting?" probes)
    # These are additive / backward-compatible: defaults reproduce the original
    # behavior. Used by:
    #   Probe 1 ("can higher LR move β?"): --ab_lr_mult > 1, no checkpoint, normal training.
    #   Probe 2 ("what β does the loss want when surroundings are pinned?"):
    #       --init_from_checkpoint <best.pt>  --freeze_others  --epochs 5
    p.add_argument("--ab_lr_mult", type=float, default=1.0,
                   help="LR multiplier applied ONLY to the alpha/beta param group "
                        "(ab_nodecay). Default 1.0 reproduces existing behavior. "
                        "Use >1 to ask whether the optimizer would move α/β further "
                        "if it had more bandwidth on those scalars.")
    p.add_argument("--init_from_checkpoint", type=str, default=None,
                   help="Path to a best.pt produced by a prior run. The model "
                        "state_dict is loaded after construction and before training. "
                        "Architecture/ablation must match the checkpoint's run.")
    p.add_argument("--freeze_others", action="store_true",
                   help="Freeze every parameter whose name does NOT contain "
                        "the active method surface. With this flag + "
                        "--init_from_checkpoint, only alpha/beta, gate, or RCD "
                        "parameters are optimized from the trained surrounding "
                        "weights.")

    args = p.parse_args()
    # Guard the accumulation protocol before any derived scheduler math uses
    # this value.  The training loop already handles partial final windows
    # correctly; this prevents invalid manual/env overrides from creating
    # incomparable new runs or division-by-zero failures.
    if args.grad_accum <= 0:
        raise SystemExit(
            f"[parse_args] --grad_accum must be > 0, got {args.grad_accum}"
        )
    if args.split_mode == "cv_fold":
        if args.fold is None:
            raise SystemExit(
                "[parse_args] --split_mode cv_fold requires --fold. "
                "Use --split_mode global_seed_holdout --global_seed SEED "
                "for the corrected random-split reruns."
            )
    elif args.split_mode == "global_seed_holdout":
        if args.fold is not None:
            raise SystemExit(
                "[parse_args] --split_mode global_seed_holdout must not pass "
                "--fold; the train/val/test holdout is generated directly "
                "from --global_seed."
            )
        if args.global_seed is None:
            raise SystemExit(
                "[parse_args] --split_mode global_seed_holdout requires "
                "--global_seed."
            )
        if args.seed != args.global_seed:
            raise SystemExit(
                "[parse_args] corrected reruns use one global seed: "
                f"--seed ({args.seed}) must equal --global_seed "
                f"({args.global_seed})."
            )
        if args.neighbor_shuffle_seed != args.global_seed:
            raise SystemExit(
                "[parse_args] corrected reruns use one global seed: "
                f"--neighbor_shuffle_seed ({args.neighbor_shuffle_seed}) "
                f"must equal --global_seed ({args.global_seed})."
            )
        if args.fold_seed != 0:
            raise SystemExit(
                "[parse_args] --fold_seed is ignored by "
                "global_seed_holdout and must be left at 0 so manifests "
                "cannot imply a fixed-fold protocol."
            )
    else:
        raise SystemExit(f"[parse_args] unknown --split_mode {args.split_mode!r}")
    if args.neighbor_window < 3 or args.neighbor_window % 2 != 1:
        raise SystemExit(
            f"[parse_args] --neighbor_window must be odd and >= 3, got "
            f"{args.neighbor_window}"
        )
    if args.gate_l2_reg < 0.0:
        raise SystemExit(
            f"[parse_args] --gate_l2_reg must be >= 0, got {args.gate_l2_reg}"
        )
    if args.rcd_rank <= 0:
        raise SystemExit(
            f"[parse_args] --rcd_rank must be > 0, got {args.rcd_rank}"
        )
    if args.learned_r_hidden_dim <= 0:
        raise SystemExit(
            "[parse_args] --learned_r_hidden_dim must be > 0, got "
            f"{args.learned_r_hidden_dim}"
        )
    if args.srp_freeze_epochs < 0:
        raise SystemExit(
            f"[parse_args] --srp_freeze_epochs must be >= 0, got {args.srp_freeze_epochs}"
        )
    if args.stage2_epochs < 0:
        raise SystemExit(
            f"[parse_args] --stage2_epochs must be >= 0, got {args.stage2_epochs}"
        )
    if args.stage2_lr_mult <= 0.0:
        raise SystemExit(
            f"[parse_args] --stage2_lr_mult must be > 0, got {args.stage2_lr_mult}"
        )
    if args.layerscale_init < 0.0:
        raise SystemExit(
            "[parse_args] --layerscale_init must be >= 0, got "
            f"{args.layerscale_init}"
        )
    uses_gate_protocol = (
        args.gate_l2_reg > 0.0
        or args.srp_freeze_epochs > 0
        or args.stage2_epochs > 0
    )
    if args.gate_l2_reg > 0.0 and args.ablation not in _GATE_L2_ABLATIONS:
        raise SystemExit(
            "[parse_args] --gate_l2_reg is valid only for "
            "signed-gated SRP ablations: "
            + ", ".join(sorted(_GATE_L2_ABLATIONS))
        )
    if uses_gate_protocol and args.ablation not in _METHOD_SURFACE_ABLATIONS:
        raise SystemExit(
            "[parse_args] two-stage controls are valid only for explicit "
            "method-surface ablations: "
            + ", ".join(sorted(_METHOD_SURFACE_ABLATIONS))
        )
    if args.srp_freeze_epochs > args.epochs:
        raise SystemExit(
            f"[parse_args] --srp_freeze_epochs ({args.srp_freeze_epochs}) "
            f"cannot exceed --epochs ({args.epochs})."
        )
    if args.stage2_epochs > 0 and args.srp_freeze_epochs <= 0:
        raise SystemExit(
            "[parse_args] --stage2_epochs requires --srp_freeze_epochs > 0 "
            "so the run has an explicit frozen stage-1 checkpoint."
        )
    if args.freeze_others and (args.srp_freeze_epochs > 0 or args.stage2_epochs > 0):
        raise SystemExit(
            "[parse_args] --freeze_others is a probe mode and cannot be "
            "combined with the two-stage protocol."
        )
    if args.pos_mode == "none":
        args.no_ppeg = True
    if args.no_ppeg:
        args.pos_mode = "none"
    if args.dataset == "cam17_univ2":
        if args.in_dim == 1024:
            args.in_dim = 1536
    if args.dataset == "cam16_univ2":
        if args.in_dim == 1024:
            args.in_dim = 1536
        if args.num_classes == 4:
            args.num_classes = 2
    if args.dataset == "kgh":
        if args.in_dim == 1024:
            args.in_dim = 1536
        # KGH phase00b historically rewrote 4 requested disease classes to a
        # 5-output head.  Keep that behavior for old commands, but provide an
        # explicit opt-in for the corrected disease-only reruns so new fold4
        # stability checks do not repeat the absent-class artifact.
        if args.num_classes == 4 and not args.kgh_true_4class:
            args.num_classes = 5
    if args.dataset == "bracs":
        if args.in_dim == 1024:
            args.in_dim = 1536
        if args.num_classes == 4:
            args.num_classes = 7
    return args


def _build_model(args, backend: str, spec: dict, device):
    """Instantiate the correct aggregator for this ablation.

    β_init resolution order (SRP backend only):
      1. spec["beta_init"] if present in _ABLATIONS — pins β_init for
         the β-init sweep ablations (srp_patch_learn_init0/05/2) so the
         launcher doesn't need to thread --beta_init on the CLI.
      2. args.beta_init CLI flag (default 1.0) — used by the primary
         sweep's learnable ablations (srp_patch_learn, srp_patch_preV,
         srp_patch_gated), where the init is always 1.0 per proposal §5.2.
    """
    if backend == "xsa":
        if args.ln_specialization != "shared":
            raise ValueError(
                "--ln_specialization cls_patch is implemented for the "
                "SRP/Nystrom CAM path only; xsa_all_ref still uses the "
                "stage-2 aggregator without specialized LayerNorm."
            )
        if args.layerscale_init > 0.0:
            raise ValueError(
                "--layerscale_init is implemented for the SRP/Nystrom "
                "CAM path only; xsa_all_ref still uses the stage-2 "
                "aggregator without LayerScale."
            )
        model = NystromXSAggregator(
            in_dim=args.in_dim, embed_dim=args.embed_dim, depth=args.depth,
            num_heads=args.num_heads, num_landmarks=args.num_landmarks,
            num_classes=args.num_classes,
            alpha_cls_mode=spec["alpha_cls_mode"],
            alpha_patch_mode=spec["alpha_patch_mode"],
            alpha_init=1.0,
            drop_path_rate=args.drop_path,
            pinv_iterations=args.pinv_iterations,
            checkpoint_mode=args.checkpoint_mode,
        ).to(device)
    elif backend == "diff":
        if args.ln_specialization != "shared":
            raise ValueError(
                "--ln_specialization cls_patch is implemented for the "
                "SRP/Nystrom CAM path only; diff_transformer uses the "
                "Diff Transformer comparator without specialized LayerNorm."
            )
        if args.layerscale_init > 0.0:
            raise ValueError(
                "--layerscale_init is implemented for the SRP/Nystrom "
                "CAM path only; diff_transformer uses the Diff Transformer "
                "comparator without LayerScale."
            )
        model = NystromDiffTransformerAggregator(
            in_dim=args.in_dim, embed_dim=args.embed_dim, depth=args.depth,
            num_heads=args.num_heads, num_landmarks=args.num_landmarks,
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
            pinv_iterations=args.pinv_iterations,
            checkpoint_mode=args.checkpoint_mode,
            use_ppeg=not args.no_ppeg,
        ).to(device)
    else:
        beta_init = float(spec.get("beta_init", args.beta_init))
        model = NystromSRPAggregator(
            in_dim=args.in_dim, embed_dim=args.embed_dim, depth=args.depth,
            num_heads=args.num_heads, num_landmarks=args.num_landmarks,
            num_classes=args.num_classes,
            beta_patch_mode=spec["beta_patch_mode"],
            beta_init=beta_init,
            srp_mode=spec["srp_mode"],
            drop_path_rate=args.drop_path,
            pinv_iterations=args.pinv_iterations,
            checkpoint_mode=args.checkpoint_mode,
            layerscale_init=args.layerscale_init,
            ln_specialization=args.ln_specialization,
            ln_specialization_scope=args.ln_specialization_scope,
            # Signed-gate parameters; consumed only when the selected SRP
            # mode is signed-gated. Default δ=2.0 covers identity /
            # anti-SRP / projection / reflection.
            delta_scale=args.delta_scale,
            gate_hidden_dim=args.gate_hidden_dim,
            detach_gate_inputs=not args.no_detach_gate_inputs,
            gate_output_init=args.gate_output_init,
            gate_output_init_scale=args.gate_output_init_scale,
            gate_init_beta0=args.gate_init_beta0,
            gate_activation=args.gate_activation,
            gate_activation_temperature=args.gate_activation_temperature,
            gate_factorization=getattr(args, "gate_factorization", "full"),
            gate_count_features=args.gate_count_features,
            rcd_adapter_kind=args.rcd_adapter_kind,
            rcd_rank=args.rcd_rank,
            learned_r_hidden_dim=args.learned_r_hidden_dim,
            use_ppeg=not args.no_ppeg,
        ).to(device)
    return model


def _checked_fold_assignment(fold_assignments, fold: int):
    """Return the requested fold or fail with a CLI-stable error.

    `assert` statements are skipped under `python -O`; launch scripts
    should still fail clearly if a manifest or manual command asks for
    an out-of-range fold.
    """
    if not 0 <= fold < len(fold_assignments):
        raise SystemExit(
            f"fold out of range: {fold}; expected 0..{len(fold_assignments) - 1}"
        )
    return fold_assignments[fold]


def _bounded_fraction_count(n_items: int, frac: float, min_remaining: int) -> int:
    """Return a rounded fractional count while keeping a usable remainder."""
    if n_items <= min_remaining or frac <= 0.0:
        return 0
    # Each non-empty stratum should contribute to the requested split when
    # possible; otherwise rare classes/centers can disappear from val/test.
    n_take = max(1, int(round(float(frac) * n_items)))
    return min(n_take, n_items - min_remaining)


def _global_seed_stratified_assignment(
    records,
    *,
    global_seed: int,
    group_key,
    unit_key,
    test_frac: float = 0.20,
    val_frac: float = 0.10,
):
    """Build one train/val/test split from a single global seed.

    The unit key defines the indivisible split entity: CAM17 uses patients
    so slides from the same patient never cross splits, while CAM16/KGH use
    slide IDs because each slide is its own patient.  The group key defines
    the stratification axis (CAM17 center, CAM16/KGH class label).
    """
    unit_to_group: dict[str, object] = {}
    for r in records:
        unit = str(unit_key(r))
        group = group_key(r)
        if unit in unit_to_group and unit_to_group[unit] != group:
            raise RuntimeError(
                "global_seed_holdout cannot stratify a split unit that spans "
                f"multiple groups: unit={unit!r} groups="
                f"{unit_to_group[unit]!r}/{group!r}"
            )
        unit_to_group[unit] = group

    by_group: dict[object, list[str]] = defaultdict(list)
    for unit, group in sorted(unit_to_group.items(), key=lambda item: item[0]):
        by_group[group].append(unit)

    rng = np.random.default_rng(int(global_seed))
    train_units: list[str] = []
    val_units: list[str] = []
    test_units: list[str] = []
    group_debug: dict[str, dict[str, int]] = {}

    for group, units in sorted(by_group.items(), key=lambda item: str(item[0])):
        shuffled = list(units)
        rng.shuffle(shuffled)

        # Hold out test first (20% of the whole stratum), then val as 10% of
        # the original stratum from the remaining pool.  This yields roughly
        # 70/10/20 train/val/test while keeping all arms for a seed paired.
        n_test = _bounded_fraction_count(len(shuffled), test_frac, min_remaining=2)
        remaining = shuffled[n_test:]
        n_val = _bounded_fraction_count(len(remaining), val_frac, min_remaining=1)

        test_part = shuffled[:n_test]
        val_part = remaining[:n_val]
        train_part = remaining[n_val:]
        if not train_part:
            raise RuntimeError(
                f"global_seed_holdout produced an empty train stratum for "
                f"group={group!r}; records={len(shuffled)}"
            )

        train_units.extend(train_part)
        val_units.extend(val_part)
        test_units.extend(test_part)
        group_debug[str(group)] = {
            "train": len(train_part),
            "val": len(val_part),
            "test": len(test_part),
            "total": len(shuffled),
        }

    return SimpleNamespace(
        train_patients=sorted(train_units),
        val_patients=sorted(val_units),
        test_patients=sorted(test_units),
        group_debug=group_debug,
    )


def _records_for_split(records, split_ids, unit_key):
    keep = {str(x) for x in split_ids}
    return [r for r in records if str(unit_key(r)) in keep]


def _counter_by_attr(records, attr: str) -> dict[str, int]:
    counts = Counter(str(getattr(r, attr)) for r in records if hasattr(r, attr))
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _build_split_metadata(
    *,
    args,
    records,
    assignment,
    unit_key,
    stratification: str,
) -> dict:
    """Persist enough split detail to audit seed pairing after the run."""
    split_to_ids = {
        "train": list(assignment.train_patients),
        "val": list(assignment.val_patients),
        "test": list(assignment.test_patients),
    }
    metadata = {
        "dataset": args.dataset,
        "split_mode": args.split_mode,
        "fold": args.fold,
        "fold_seed": args.fold_seed,
        "global_seed": args.global_seed,
        "train_seed": args.seed,
        "neighbor_shuffle_seed": args.neighbor_shuffle_seed,
        "unit_key": "patient_id" if args.dataset in {"cam17", "cam17_univ2", "bracs"} else "slide_id",
        "stratification": stratification,
        "split_ids": split_to_ids,
        "unit_counts": {name: len(ids) for name, ids in split_to_ids.items()},
    }
    per_split_records = {
        name: _records_for_split(records, ids, unit_key)
        for name, ids in split_to_ids.items()
    }
    metadata["counts"] = {
        name: len(split_records)
        for name, split_records in per_split_records.items()
    }
    metadata["slide_counts"] = metadata["counts"]
    metadata["class_counts"] = {
        name: _counter_by_attr(split_records, "label")
        for name, split_records in per_split_records.items()
    }
    metadata["center_counts"] = {
        name: _counter_by_attr(split_records, "center")
        for name, split_records in per_split_records.items()
    }
    if any(hasattr(r, "stage") for r in records):
        metadata["stage_counts"] = {
            name: _counter_by_attr(split_records, "stage")
            for name, split_records in per_split_records.items()
        }
    if hasattr(assignment, "group_debug"):
        metadata["stratum_unit_counts"] = assignment.group_debug
    return metadata


def _numeric_fold_id_for_artifacts(fold: int | None) -> int:
    """Return an integer fold id for array artifacts.

    Global-seed holdout runs intentionally do not have a CV fold.  NPZ
    diagnostics still need a numeric array so downstream NumPy code can load
    one stable dtype; `-1` is reserved here to mean "not a fold-based split".
    """
    return -1 if fold is None else int(fold)


def _stage_trainability_mode_name(mode: str) -> str:
    names = {
        "all": "all parameters trainable",
        "freeze_method": "SRP/gate/RCD method parameters frozen",
        "method_only": "only SRP/gate/RCD method parameters trainable",
    }
    if mode not in names:
        raise ValueError(f"unknown trainability mode {mode!r}")
    return names[mode]


def apply_trainability_mode(model, args, mode: str) -> dict[str, int]:
    """Set `requires_grad` for one training stage.

    The two-stage protocol depends on a precise separation between the
    method surface (`gate.*` for signed-gated SRP) and the ordinary
    backbone/head/PPEG parameters.  Reusing `is_method_param` here keeps
    freeze masks, optimizer groups, and audit counts aligned.
    """
    if mode not in {"all", "freeze_method", "method_only"}:
        raise ValueError(f"unknown trainability mode {mode!r}")
    counts = {
        "method_trainable": 0,
        "method_frozen": 0,
        "nonmethod_trainable": 0,
        "nonmethod_frozen": 0,
    }
    for name, param in model.named_parameters():
        is_method = is_method_param(name, args.ablation)
        if mode == "all":
            trainable = True
        elif mode == "freeze_method":
            trainable = not is_method
        else:  # method_only
            trainable = is_method
        param.requires_grad = trainable
        key = (
            "method_trainable" if is_method and trainable else
            "method_frozen" if is_method else
            "nonmethod_trainable" if trainable else
            "nonmethod_frozen"
        )
        counts[key] += param.numel()
    return counts


def build_adamw_optimizer(model, args) -> torch.optim.Optimizer:
    """Build the three-group AdamW optimizer used by the slide trainer."""
    backbone_decay, method_decay, method_nodecay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_method_no_decay(name, args.ablation):
            method_nodecay.append(param)
        elif is_method_param(name, args.ablation):
            method_decay.append(param)
        else:
            backbone_decay.append(param)

    param_groups = []
    if backbone_decay:
        param_groups.append({
            "params": backbone_decay,
            "weight_decay": args.weight_decay,
            "group_name": "decay",
        })
    if method_decay:
        param_groups.append({
            "params": method_decay,
            "weight_decay": args.weight_decay,
            "group_name": "method_decay",
        })
    if method_nodecay:
        # Group name kept as `ab_nodecay` for backwards compatibility
        # with existing optimizer-state readers and training logs.
        param_groups.append({
            "params": method_nodecay,
            "weight_decay": 0.0,
            "group_name": "ab_nodecay",
        })
    if not param_groups:
        raise RuntimeError(
            f"[{args.run_name}] no trainable parameters under current "
            "trainability mode."
        )
    return torch.optim.AdamW(
        param_groups,
        lr=args.base_lr,
        betas=(0.9, 0.999),
    )


def method_parameter_summary(model, args) -> tuple[list[str], int, int, int]:
    """Return method names, method count, trainable count, and total count."""
    method_names = [
        name for name, param in model.named_parameters()
        if param.requires_grad and is_method_param(name, args.ablation)
    ]
    n_method = sum(
        param.numel() for name, param in model.named_parameters()
        if param.requires_grad and is_method_param(name, args.ablation)
    )
    n_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    n_total = sum(param.numel() for param in model.parameters())
    return method_names, n_method, n_trainable, n_total


def _feature_family_from_key(feature_key: str) -> str:
    """Map an H5 feature key to the AtlasPatch directory family name."""
    key = feature_key.strip("/")
    if key.startswith("features/"):
        return key.split("/", 1)[1]
    return Path(key).name


def collect_gate_beta_l2(model, device) -> tuple[torch.Tensor, int]:
    """Collect differentiable `mean(beta_eff^2)` from active signed gates.

    Each signed-gate attention layer writes a non-detached beta cache during
    forward.  The regularizer averages per-block means so slides with very
    large token counts do not dominate the penalty solely because they have
    more `(head, token)` entries.
    """
    values: list[torch.Tensor] = []
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        beta_eff = getattr(attn, "_last_gate_beta_eff_for_loss", None)
        if beta_eff is None:
            continue
        values.append(beta_eff.float().pow(2).mean())
    if not values:
        return torch.zeros((), dtype=torch.float32, device=device), 0
    return torch.stack(values).mean(), len(values)


def main() -> None:
    args = parse_args()
    spec = _ABLATIONS[args.ablation]
    backend = spec["backend"]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # β_init visibility: log the RESOLVED value (spec wins over CLI arg),
    # not the CLI default, so "srp_patch_learn_init05" obviously shows
    # β_init=0.5 in the run header and W&B config even if the caller
    # didn't pass --beta_init 0.5.
    resolved_beta_init = float(spec.get("beta_init", args.beta_init)) if backend == "srp" else None
    print(f"[{args.run_name}] device={device} backend={backend} ablation={args.ablation}"
          + (f" beta_init={resolved_beta_init}" if resolved_beta_init is not None else "")
          + (f" layerscale_init={args.layerscale_init:g}" if backend == "srp" else "")
          + (f" ln_specialization={args.ln_specialization}"
             f" ln_scope={args.ln_specialization_scope}" if backend == "srp" else ""))
    if device.type == "cuda":
        print(f"[{args.run_name}] gpu={torch.cuda.get_device_name(0)}")

    # --- Data + fold ------------------------------------------------------
    # Dispatch on --dataset. Default = cam17 (the original Stage-3
    # path). cam16 = binary tumor/normal CAM16 with ViT-B/16 features;
    # see slide_level_srp/data_cam16.py for the adapter and env vars.
    split_metadata = None
    if args.dataset == "cam17":
        records = enumerate_slides(num_classes=args.num_classes)
        unit_key = lambda r: r.patient_id
        stratification = "center"
        if args.split_mode == "cv_fold":
            fold_assignments = build_fold_assignments(
                records,
                n_folds=5,
                val_patients_per_fold=args.val_patients_per_fold,
                fold_seed=args.fold_seed,
            )
            fa = _checked_fold_assignment(fold_assignments, args.fold)
        else:
            fa = _global_seed_stratified_assignment(
                records,
                global_seed=args.global_seed,
                group_key=lambda r: r.center,
                unit_key=unit_key,
                test_frac=0.20,
                val_frac=0.10,
            )
        train_loader, val_loader, test_loader = build_srp_loaders_for_fold(
            records, fa, num_workers=args.num_workers,
            train_cap=args.train_cap,
            val_cap=args.val_cap,
            test_cap=args.test_cap,
            neighbor_radius=(args.neighbor_window - 1) // 2,
            neighbor_shell=args.neighbor_shell,
            neighbor_source=args.neighbor_source,
            neighbor_shuffle_seed=args.neighbor_shuffle_seed,
            neighbor_weighting=args.neighbor_weighting,
            neighbor_weight_sigma=args.neighbor_weight_sigma,
        )
        split_metadata = _build_split_metadata(
            args=args, records=records, assignment=fa,
            unit_key=unit_key, stratification=stratification,
        )
    elif args.dataset == "cam17_univ2":
        from slide_level_srp.data_cam17_univ2 import (
            CAM17_UNIV2_FEATURE_KEY,
            CAM17_UNIV2_ROOT,
            enumerate_cam17_univ2_slides,
            build_fold_assignments as build_cam17_univ2_fold_assignments,
            build_cam17_univ2_loaders_for_fold,
        )
        feature_root = args.feature_root or CAM17_UNIV2_ROOT
        feature_key = args.feature_key or CAM17_UNIV2_FEATURE_KEY
        records = enumerate_cam17_univ2_slides(
            feature_root=feature_root,
            num_classes=args.num_classes,
        )
        unit_key = lambda r: r.patient_id
        stratification = "center"
        if args.split_mode == "cv_fold":
            fold_assignments = build_cam17_univ2_fold_assignments(
                records,
                n_folds=5,
                val_patients_per_fold=args.val_patients_per_fold,
                fold_seed=args.fold_seed,
            )
            fa = _checked_fold_assignment(fold_assignments, args.fold)
        else:
            fa = _global_seed_stratified_assignment(
                records,
                global_seed=args.global_seed,
                group_key=lambda r: r.center,
                unit_key=unit_key,
                test_frac=0.20,
                val_frac=0.10,
            )
        train_loader, val_loader, test_loader = build_cam17_univ2_loaders_for_fold(
            records, fa, num_workers=args.num_workers,
            subsample_cap=args.train_cap,
            train_cap=args.train_cap,
            val_cap=args.val_cap,
            test_cap=args.test_cap,
            feature_key=feature_key,
            expected_dim=args.in_dim,
            neighbor_radius=(args.neighbor_window - 1) // 2,
            neighbor_shell=args.neighbor_shell,
            neighbor_source=args.neighbor_source,
            neighbor_shuffle_seed=args.neighbor_shuffle_seed,
            neighbor_weighting=args.neighbor_weighting,
            neighbor_weight_sigma=args.neighbor_weight_sigma,
        )
        split_metadata = _build_split_metadata(
            args=args, records=records, assignment=fa,
            unit_key=unit_key, stratification=stratification,
        )
    elif args.dataset == "cam16":
        from slide_level_srp.data_cam16 import (
            enumerate_cam16_slides,
            build_cam16_fold_assignments,
            build_cam16_loaders_for_fold,
        )
        records = enumerate_cam16_slides()
        unit_key = lambda r: r.slide_id
        stratification = "label"
        if args.split_mode == "cv_fold":
            fold_assignments = build_cam16_fold_assignments(
                records, n_folds=5, fold_seed=args.fold_seed,
            )
            fa = _checked_fold_assignment(fold_assignments, args.fold)
        else:
            fa = _global_seed_stratified_assignment(
                records,
                global_seed=args.global_seed,
                group_key=lambda r: r.label,
                unit_key=unit_key,
                test_frac=0.20,
                val_frac=0.10,
            )
        # Phase-A.9 review fix F8: pass all three caps explicitly. Pre-fix
        # only `train_cap` was plumbed (as `subsample_cap` for all splits),
        # silently ignoring `--val_cap` / `--test_cap`. Now each split
        # honours its own cap; defaults to `subsample_cap` when unset.
        train_loader, val_loader, test_loader = build_cam16_loaders_for_fold(
            records, fa, num_workers=args.num_workers,
            subsample_cap=args.train_cap,
            train_cap=args.train_cap,
            val_cap=args.val_cap,
            test_cap=args.test_cap,
            neighbor_radius=(args.neighbor_window - 1) // 2,
            neighbor_shell=args.neighbor_shell,
            neighbor_source=args.neighbor_source,
            neighbor_shuffle_seed=args.neighbor_shuffle_seed,
            neighbor_weighting=args.neighbor_weighting,
            neighbor_weight_sigma=args.neighbor_weight_sigma,
        )
        split_metadata = _build_split_metadata(
            args=args, records=records, assignment=fa,
            unit_key=unit_key, stratification=stratification,
        )
    elif args.dataset == "cam16_univ2":
        from slide_level_srp.data_cam16_univ2 import (
            CAM16_UNIV2_FEATURE_KEY,
            CAM16_UNIV2_ROOT,
            enumerate_cam16_univ2_slides,
            build_cam16_fold_assignments,
            build_cam16_univ2_loaders_for_fold,
        )
        feature_root = args.feature_root or CAM16_UNIV2_ROOT
        feature_key = args.feature_key or CAM16_UNIV2_FEATURE_KEY
        records = enumerate_cam16_univ2_slides(
            feature_root,
            feature_family=_feature_family_from_key(feature_key),
        )
        unit_key = lambda r: r.slide_id
        stratification = "label"
        if args.split_mode == "cv_fold":
            fold_assignments = build_cam16_fold_assignments(
                records, n_folds=5, fold_seed=args.fold_seed,
            )
            fa = _checked_fold_assignment(fold_assignments, args.fold)
        else:
            fa = _global_seed_stratified_assignment(
                records,
                global_seed=args.global_seed,
                group_key=lambda r: r.label,
                unit_key=unit_key,
                test_frac=0.20,
                val_frac=0.10,
            )
        train_loader, val_loader, test_loader = build_cam16_univ2_loaders_for_fold(
            records, fa, num_workers=args.num_workers,
            subsample_cap=args.train_cap,
            train_cap=args.train_cap,
            val_cap=args.val_cap,
            test_cap=args.test_cap,
            feature_key=feature_key,
            expected_dim=args.in_dim,
            neighbor_radius=(args.neighbor_window - 1) // 2,
            neighbor_shell=args.neighbor_shell,
            neighbor_source=args.neighbor_source,
            neighbor_shuffle_seed=args.neighbor_shuffle_seed,
            neighbor_weighting=args.neighbor_weighting,
            neighbor_weight_sigma=args.neighbor_weight_sigma,
        )
        split_metadata = _build_split_metadata(
            args=args, records=records, assignment=fa,
            unit_key=unit_key, stratification=stratification,
        )
    elif args.dataset == "kgh":
        from slide_level_srp.data_kgh import (
            KGH_FEATURE_KEY,
            KGH_FEATURE_ROOT,
            enumerate_kgh_slides,
            build_kgh_fold_assignments,
            build_kgh_loaders_for_fold,
        )
        feature_root = args.feature_root or KGH_FEATURE_ROOT
        feature_key = args.feature_key or KGH_FEATURE_KEY
        records = enumerate_kgh_slides(feature_root=feature_root)
        unit_key = lambda r: r.slide_id
        stratification = "label"
        if args.split_mode == "cv_fold":
            fold_assignments = build_kgh_fold_assignments(
                records, n_folds=5, fold_seed=args.fold_seed,
            )
            fa = _checked_fold_assignment(fold_assignments, args.fold)
        else:
            fa = _global_seed_stratified_assignment(
                records,
                global_seed=args.global_seed,
                group_key=lambda r: r.label,
                unit_key=unit_key,
                test_frac=0.20,
                val_frac=0.10,
            )
        train_loader, val_loader, test_loader = build_kgh_loaders_for_fold(
            records, fa, num_workers=args.num_workers,
            subsample_cap=args.train_cap,
            train_cap=args.train_cap,
            val_cap=args.val_cap,
            test_cap=args.test_cap,
            feature_key=feature_key,
            expected_dim=args.in_dim,
            neighbor_radius=(args.neighbor_window - 1) // 2,
            neighbor_shell=args.neighbor_shell,
            neighbor_source=args.neighbor_source,
            neighbor_shuffle_seed=args.neighbor_shuffle_seed,
            neighbor_weighting=args.neighbor_weighting,
            neighbor_weight_sigma=args.neighbor_weight_sigma,
        )
        split_metadata = _build_split_metadata(
            args=args, records=records, assignment=fa,
            unit_key=unit_key, stratification=stratification,
        )
    elif args.dataset == "bracs":
        from slide_level_srp.data_bracs import (
            BRACS_FEATURE_KEY,
            BRACS_FEATURE_ROOT,
            enumerate_bracs_slides,
            build_bracs_fold_assignments,
            build_bracs_global_seed_assignment,
            build_bracs_loaders_for_fold,
        )
        feature_root = args.feature_root or BRACS_FEATURE_ROOT
        feature_key = args.feature_key or BRACS_FEATURE_KEY
        records = enumerate_bracs_slides(feature_root=feature_root)
        unit_key = lambda r: r.patient_id
        stratification = "patient_label_vector"
        if args.split_mode == "cv_fold":
            fold_assignments = build_bracs_fold_assignments(
                records, n_folds=5, fold_seed=args.fold_seed,
            )
            fa = _checked_fold_assignment(fold_assignments, args.fold)
        else:
            # BRACS patients can contribute slides from multiple diagnostic
            # classes.  The shared global-seed splitter requires one stratum per
            # split unit, so use the BRACS adapter's multi-label patient splitter
            # to keep patient isolation without throwing away label balance.
            fa = build_bracs_global_seed_assignment(
                records,
                global_seed=args.global_seed,
                test_frac=0.20,
                val_frac=0.10,
            )
        train_loader, val_loader, test_loader = build_bracs_loaders_for_fold(
            records, fa, num_workers=args.num_workers,
            subsample_cap=args.train_cap,
            train_cap=args.train_cap,
            val_cap=args.val_cap,
            test_cap=args.test_cap,
            feature_key=feature_key,
            expected_dim=args.in_dim,
            neighbor_radius=(args.neighbor_window - 1) // 2,
            neighbor_shell=args.neighbor_shell,
            neighbor_source=args.neighbor_source,
            neighbor_shuffle_seed=args.neighbor_shuffle_seed,
            neighbor_weighting=args.neighbor_weighting,
            neighbor_weight_sigma=args.neighbor_weight_sigma,
        )
        split_metadata = _build_split_metadata(
            args=args, records=records, assignment=fa,
            unit_key=unit_key, stratification=stratification,
        )
    else:
        raise ValueError(f"unknown --dataset: {args.dataset}")
    split_label = (
        f"fold={args.fold}"
        if args.split_mode == "cv_fold"
        else f"global_seed={args.global_seed}"
    )
    print(f"[{args.run_name}] split_mode={args.split_mode} {split_label} "
          f"train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}  "
          f"caps(train,val,test)=({args.train_cap},{args.val_cap},{args.test_cap})")

    # --- Model ------------------------------------------------------------
    model = _build_model(args, backend, spec, device)

    # --- Probe-mode hooks (additive; default-behavior preserved) ---------
    # 1. Optionally warm-start from a previously-saved best.pt.  Architecture
    #    must match (same ablation), so we use strict load. weights_only=False
    #    matches train.py's existing torch.save format which stores a dict
    #    {"epoch", "model", "args", "val_metrics"}.
    if getattr(args, "init_from_checkpoint", None):
        ckpt = torch.load(args.init_from_checkpoint,
                          map_location=device, weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=True)
        print(f"[{args.run_name}] loaded init from {args.init_from_checkpoint}  "
              f"missing={len(missing) if hasattr(missing, '__len__') else missing}  "
              f"unexpected={len(unexpected) if hasattr(unexpected, '__len__') else unexpected}")

    # 2. Optionally freeze all parameters EXCEPT the α/β scalars. Used for
    #    the conditional-optimum probe: with surrounding weights pinned at
    #    their trained values, α/β have no co-adaptation partner, so wherever
    #    they land is their loss-conditional optimum.
    # Phase-A.9 review fix F2: route `freeze_others` and method-param
    # accounting through `is_method_param(name, args.ablation)` so that
    # `srp_patch_signed_gated` correctly identifies `gate.*` as the
    # method's learnable surface. Pre-fix, the predicate was
    # alpha/beta-only and would freeze the entire gate or undercount
    # signed-gated method params.
    if getattr(args, "freeze_others", False):
        n_frozen, n_unfrozen = 0, 0
        for name, p in model.named_parameters():
            if is_method_param(name, args.ablation):
                p.requires_grad = True
                n_unfrozen += p.numel()
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        which = (
            "mlp_control.*" if args.ablation in _MLP_CONTROL_ABLATIONS
            else "gate.*" if args.ablation in _GATE_L2_ABLATIONS
            else "α/β"
        )
        print(f"[{args.run_name}] freeze_others ON: "
              f"trainable_params={n_unfrozen} (only {which}); "
              f"frozen_params={n_frozen}")

    stage_protocol_enabled = args.srp_freeze_epochs > 0 or args.stage2_epochs > 0
    if stage_protocol_enabled:
        counts = apply_trainability_mode(model, args, "freeze_method")
        print(
            f"[{args.run_name}] stage1 trainability: "
            f"{_stage_trainability_mode_name('freeze_method')} "
            f"counts={counts}"
        )

    # Collect the names of the method-specific learnable params for
    # inspection + no-decay param-group separation below. Uses the
    # ablation-aware predicate (F2 fix).
    ab_param_names, n_ab, n_params_trainable, n_params_total = method_parameter_summary(model, args)
    method_label = (
        "gate_context_params" if args.ablation in _LEARNED_R_SIGNED_GATE_ABLATIONS
        else "gate_params" if args.ablation in _SIGNED_GATE_ABLATIONS
        else "rcd_params" if args.ablation in _RCD_ABLATIONS
        else "mlp_control_params" if args.ablation in _MLP_CONTROL_ABLATIONS
        else "diff_params" if backend == "diff"
        else "ab_params"
    )
    # Truncate name list when it gets long (signed_gated has up to 21).
    name_str = ('+'.join(ab_param_names[:5])
                + (f" + {len(ab_param_names) - 5} more"
                   if len(ab_param_names) > 5 else ""))
    print(f"[{args.run_name}] params_total={n_params_total:,} "
          f"trainable={n_params_trainable:,} {method_label}={n_ab} "
          f"({name_str if ab_param_names else 'none'})")

    # --- Optimizer --------------------------------------------------------
    # alpha/beta parameters get weight_decay=0 so decoupled AdamW does not
    # mechanically pull them toward 0. Signed-gate biases share that no-decay
    # treatment, while gate weights remain in a decayed method group.  The
    # helper is reused at the stage-2 boundary after trainability changes.
    optimizer = build_adamw_optimizer(model, args)
    steps_per_epoch = (len(train_loader.dataset) + args.grad_accum - 1) // args.grad_accum
    base_total_steps = args.epochs * steps_per_epoch
    total_train_epochs = args.epochs + args.stage2_epochs
    total_steps = total_train_epochs * steps_per_epoch
    warmup_steps = max(1, int(base_total_steps * args.warmup_ratio))
    stage_total_steps = base_total_steps
    stage_warmup_steps = warmup_steps
    stage_step = 0
    stage_name = "stage1_frozen" if stage_protocol_enabled else "main"
    print(
        f"[{args.run_name}] total_opt_steps={total_steps} "
        f"base_steps={base_total_steps} stage2_steps={args.stage2_epochs * steps_per_epoch} "
        f"warmup={warmup_steps}"
    )

    # --- W&B --------------------------------------------------------------
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    split_metadata_path = out_dir / "split_metadata.json"
    split_metadata_path.write_text(
        json.dumps(split_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        dir=str(out_dir),
        config={
            **vars(args),
            "backend": backend,
            "ablation_spec": spec,
            "resolved_beta_init": resolved_beta_init,
            "n_params_total": n_params_total,
            "n_params_trainable": n_params_trainable,
            "n_ab_params": n_ab,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "ab_param_names": ab_param_names,
            "train_slides": len(train_loader.dataset),
            "val_slides": len(val_loader.dataset),
            "test_slides": len(test_loader.dataset),
            "split_metadata": split_metadata,
        },
        tags=[
            f"ablation-{args.ablation}",
            f"backend-{backend}",
            f"split-{args.split_mode}",
            (
                f"fold-{args.fold}"
                if args.split_mode == "cv_fold"
                else f"global-seed-{args.global_seed}"
            ),
            f"seed-{args.seed}",
            f"dp-{args.drop_path:g}",
            f"ckpt-{args.checkpoint_mode}",
        ],
    )

    best_val_f1 = -1.0
    best_ckpt = out_dir / "best.pt"
    stage1_best_ckpt = out_dir / "stage1_best.pt"
    stage1_best_val_f1 = float("nan")
    stage2_best_val_f1 = float("nan")
    global_step = 0
    gate_l2_missing_steps = 0
    train_history: list[dict[str, object]] = []

    # --- Step-level alpha/beta trajectory logging ------------------------
    trajectory_log_every = 25
    has_learnable_ab = n_ab > 0 and any(
        p.requires_grad for n, p in model.named_parameters()
        if "alpha_cls" in n or "alpha_patch" in n or "beta_patch" in n
    )
    # Phase-A.9 fourth-review fix F6: signed-gated runs have no α/β
    # scalars, so the legacy `has_learnable_ab` branch is silent. Detect
    # signed_gated mode separately and log a small gate-step summary
    # alongside the trajectory cadence.
    has_signed_gate = args.ablation in _GATE_L2_ABLATIONS
    ab_history: list[dict] = []

    # --- Training loop ---------------------------------------------------
    for epoch in range(total_train_epochs):
        # Stage-2 starts after the ordinary --epochs budget.  Reloading the
        # best stage-1 checkpoint prevents a noisy final stage-1 epoch from
        # defining the fine-tune start point, and resetting `best_val_f1`
        # ensures final testing evaluates the best stage-2 checkpoint rather
        # than silently falling back to the frozen-gate baseline.
        if stage_protocol_enabled and args.stage2_epochs > 0 and epoch == args.epochs:
            stage1_best_val_f1 = float(best_val_f1)
            if best_ckpt.exists():
                shutil.copy2(best_ckpt, stage1_best_ckpt)
                ckpt = torch.load(stage1_best_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"])
                print(
                    f"[{args.run_name}] stage2 reload: "
                    f"{stage1_best_ckpt} val_f1={stage1_best_val_f1:.4f}"
                )
            else:
                raise RuntimeError(
                    f"[{args.run_name}] stage1 completed without {best_ckpt}; "
                    "cannot start two-stage fine-tuning."
                )
            stage_mode = "method_only" if args.stage2_mode == "srp_only" else "all"
            counts = apply_trainability_mode(model, args, stage_mode)
            optimizer = build_adamw_optimizer(model, args)
            best_val_f1 = -1.0
            stage_step = 0
            stage_total_steps = args.stage2_epochs * steps_per_epoch
            stage_warmup_steps = max(1, int(stage_total_steps * args.warmup_ratio))
            stage_name = f"stage2_{args.stage2_mode}"
            names, n_method_stage, n_trainable_stage, n_total_stage = method_parameter_summary(model, args)
            print(
                f"[{args.run_name}] stage2 trainability: "
                f"{_stage_trainability_mode_name(stage_mode)} counts={counts} "
                f"trainable={n_trainable_stage:,}/{n_total_stage:,} "
                f"method_trainable={n_method_stage} "
                f"names={'+'.join(names[:5]) if names else 'none'}"
            )

        model.train()
        t0 = time.time()
        epoch_loss_sum = 0.0
        epoch_objective_sum = 0.0
        epoch_gate_l2_sum = 0.0
        epoch_n = 0
        epoch_labels = []
        epoch_probs = []

        optimizer.zero_grad(set_to_none=True)
        slides_in_accum = 0

        pbar = tqdm(
            train_loader,
            desc=f"[{args.run_name}] {stage_name} ep {epoch+1}/{total_train_epochs}",
            leave=False,
            mininterval=1.0,
        )
        for bi, batch in enumerate(pbar):
            labels = batch["label"].to(device, non_blocking=True)

            stage_base_lr = args.base_lr * (
                args.stage2_lr_mult if stage_name.startswith("stage2") else 1.0
            )
            lr_now = cosine_warmup_lr(
                stage_step, stage_warmup_steps, stage_total_steps, stage_base_lr, min_lr=0.0,
            )
            # Phase-A.9 third-review fix F1: --ab_lr_mult now scales
            # BOTH method groups (gate weights + gate biases under
            # signed_gated; α/β scalars under legacy ablations). Pre-fix,
            # only ab_nodecay (gate biases) was scaled — meaning the
            # lrmult* runs in the existing PHASE_A6/A7 results actually
            # scaled gate biases only. With mult=1.0 this is a no-op and
            # bit-exactly reproduces the original training trajectory.
            method_groups = ("ab_nodecay", "method_decay")
            for pg in optimizer.param_groups:
                if pg.get("group_name") in method_groups and args.ab_lr_mult != 1.0:
                    pg["lr"] = lr_now * args.ab_lr_mult
                else:
                    pg["lr"] = lr_now

            # Phase-A.9 review fix F4: divide by the **actual** size of the
            # current accumulation window, not by `args.grad_accum`. Without
            # this, the final partial window of an epoch (with size
            # `len(loader) % grad_accum`) had its gradients underscaled —
            # e.g. with grad_accum=16 and 398 train slides, the last
            # update saw 14 slides at 14/16 = 87.5 % effective weight.
            # Window for batch index `bi` = batches in [window_start, ...,
            # min(window_start+grad_accum, len(loader)) - 1].
            window_start = (bi // args.grad_accum) * args.grad_accum
            window_size  = min(args.grad_accum, len(train_loader) - window_start)

            with autocast_ctx(device, torch.bfloat16):
                logits = _model_forward(
                    model, batch, backend, device,
                    ablation_spec=spec,
                )
                ce_loss = F.cross_entropy(logits.float(), labels)
                gate_l2, gate_l2_blocks = (
                    collect_gate_beta_l2(model, device)
                    if args.gate_l2_reg > 0.0
                    else (torch.zeros((), dtype=torch.float32, device=device), 0)
                )
                if args.gate_l2_reg > 0.0 and gate_l2_blocks == 0:
                    gate_l2_missing_steps += 1
                objective = ce_loss + (args.gate_l2_reg * gate_l2)
                loss = objective / window_size

            loss.backward()

            epoch_loss_sum += float(ce_loss.detach().item()) * labels.numel()
            epoch_objective_sum += float(objective.detach().item()) * labels.numel()
            epoch_gate_l2_sum += float(gate_l2.detach().item()) * labels.numel()
            epoch_n += labels.numel()
            epoch_labels.append(labels.detach().cpu())
            epoch_probs.append(F.softmax(logits.float(), dim=-1).detach().cpu())
            slides_in_accum += 1

            is_last = (bi == len(train_loader) - 1)
            if slides_in_accum >= args.grad_accum or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                slides_in_accum = 0

                if global_step % 50 == 0:
                    # F4 fix: undo the per-window scaling consistently.
                    payload = {
                        "train/loss_step": float(ce_loss.detach().item()),
                        "train/objective_step": float(objective.detach().item()),
                        "train/lr": lr_now,
                    }
                    if args.gate_l2_reg > 0.0:
                        payload["train/gate_l2_step"] = float(gate_l2.detach().item())
                        payload["train/gate_l2_blocks"] = gate_l2_blocks
                    wandb.log(payload, step=global_step)
                if has_learnable_ab and (global_step % trajectory_log_every == 0):
                    if backend == "xsa":
                        a = extract_alpha_values(model)
                        ab_history.append({
                            "step": int(global_step),
                            "alpha_cls":   a["alpha_cls"].cpu().numpy().copy(),
                            "alpha_patch": a["alpha_patch"].cpu().numpy().copy(),
                        })
                        wandb.log(
                            {"alpha_step/alpha_cls_mean":   float(a["alpha_cls"].mean()),
                             "alpha_step/alpha_patch_mean": float(a["alpha_patch"].mean())},
                            step=global_step,
                        )
                    else:
                        b = extract_beta_values(model)
                        ab_history.append({
                            "step": int(global_step),
                            "beta_patch": b["beta_patch"].cpu().numpy().copy(),
                        })
                        wandb.log(
                            {"beta_step/beta_patch_mean": float(b["beta_patch"].mean())},
                            step=global_step,
                        )
                # Phase-A.9 fourth-review fix F6: signed-gate trajectory
                # snapshot. Reads `_last_gate_stats` populated by the most
                # recent forward and logs per-block β_eff means / sign-bin
                # fractions. No-op when the gate is not active.
                if has_signed_gate and (global_step % trajectory_log_every == 0):
                    from slide_level_srp.src.gate_signed import signed_gate_step_summary
                    gate_payload = signed_gate_step_summary(model)
                    if gate_payload:
                        wandb.log(gate_payload, step=global_step)
                global_step += 1
                stage_step += 1

        # End-of-epoch train metrics.
        train_y = torch.cat(epoch_labels).numpy()
        train_prob = torch.cat(epoch_probs).numpy()
        train_pred = train_prob.argmax(axis=-1)
        train_metrics = compute_metrics(
            train_y, train_pred, train_prob, num_classes=args.num_classes,
        )
        train_metrics["loss"] = epoch_loss_sum / max(1, epoch_n)
        train_metrics["objective"] = epoch_objective_sum / max(1, epoch_n)
        if args.gate_l2_reg > 0.0:
            train_metrics["gate_l2"] = epoch_gate_l2_sum / max(1, epoch_n)
            train_metrics["gate_l2_penalty"] = args.gate_l2_reg * train_metrics["gate_l2"]

        val_metrics, val_cls_stats, _, _, _, _ = run_eval(
            model, val_loader, device,
            num_classes=args.num_classes,
            backend=backend,
            capture_stats=True, max_stats_batches=args.val_stats_batches,
            autocast_dtype=torch.bfloat16,
            ablation_spec=spec,
        )

        # Snapshot alphas or betas for W&B logging.
        if backend == "xsa":
            current_ab = extract_alpha_values(model)
            log_alpha_scalars(current_ab, prefix="alpha", step=global_step)
        elif backend == "srp":
            current_ab = extract_beta_values(model)
            log_beta_scalars(current_ab, prefix="beta", step=global_step)
        else:
            current_ab = {}

        wandb.log({f"train/{k}": v for k, v in train_metrics.items()}, step=global_step)
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

        # Val CLS diagnostic scalars (role-split cos / norm / z_over_y).
        # We log only the scalar means here — the per-(layer, head) tables
        # are written to test_artifacts.npz at final test time.
        if val_cls_stats is not None:
            payload = {}
            for k, v in val_cls_stats.items():
                v_np = v.detach().cpu().numpy()
                payload[f"val_cls/{k}_mean"] = float(v_np.mean())
            wandb.log(payload, step=global_step)

        dt = time.time() - t0
        print(
            f"[{args.run_name}] {stage_name} ep{epoch+1}: "
            f"train[loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.4f} "
            f"auc={train_metrics.get('auc', float('nan')):.4f}"
            + (
                f" gate_l2={train_metrics['gate_l2']:.6f}"
                if args.gate_l2_reg > 0.0 else ""
            )
            + "] "
            f"| val[loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.4f} "
            f"auc={val_metrics.get('auc', float('nan')):.4f}] "
            f"| {dt:.1f}s"
        )
        train_history.append({
            "epoch": epoch + 1,
            "stage": stage_name,
            "train": train_metrics,
            "val": val_metrics,
            "seconds": dt,
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {"epoch": epoch + 1, "model": model.state_dict(),
                 "args": vars(args), "val_metrics": val_metrics,
                 "stage": stage_name},
                best_ckpt,
            )

    if stage_protocol_enabled and args.stage2_epochs == 0:
        stage1_best_val_f1 = float(best_val_f1)
    if stage_protocol_enabled and args.stage2_epochs > 0:
        stage2_best_val_f1 = float(best_val_f1)

    # --- Final test -----------------------------------------------------
    print(f"[{args.run_name}] reloading best ckpt (val_f1={best_val_f1:.4f})")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    test_metrics, test_cls_stats, per_slide, per_slide_diag, test_gate_stats, test_eval_arrays = run_eval(
        model, test_loader, device,
        num_classes=args.num_classes,
        backend=backend,
        capture_stats=True, max_stats_batches=None,
        autocast_dtype=torch.bfloat16,
        collect_per_slide=True,
        collect_per_slide_diagnostics=True,
        ablation_spec=spec,
        collect_gate_stats=True,
    )
    print(
        f"[{args.run_name}] TEST: loss={test_metrics['loss']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"auc={test_metrics.get('auc', float('nan')):.4f} "
        f"(n_slides={len(per_slide)})"
    )

    # Snapshot final alpha / beta for the NPZ.
    if backend == "xsa":
        final_ab = extract_alpha_values(model)
        final_layerscale = {}
    elif backend == "srp":
        final_ab = extract_beta_values(model)
        final_layerscale = extract_layerscale_values(model)
    else:
        final_ab = {}
        final_layerscale = {}

    wandb.log({f"test/{k}": v for k, v in test_metrics.items()}, step=global_step)
    if final_layerscale:
        ls_payload = {}
        for role, arr in final_layerscale.items():
            arr_np = arr.detach().cpu().numpy()
            for li, row in enumerate(arr_np):
                # Mean/std/min/max are enough to detect whether a block stayed
                # near the CaiT initialization or moved sharply during training
                # without emitting one scalar per channel to W&B.
                prefix = f"test_layerscale/{role}_L{li}"
                ls_payload[f"{prefix}_mean"] = float(row.mean())
                ls_payload[f"{prefix}_std"] = float(row.std())
                ls_payload[f"{prefix}_min"] = float(row.min())
                ls_payload[f"{prefix}_max"] = float(row.max())
        wandb.log(ls_payload, step=global_step)
    if test_cls_stats is not None:
        # Per-(layer, head) W&B tables for the key diagnostics.
        cols_by_head = None
        payload = {}
        for k, v in test_cls_stats.items():
            v_np = v.detach().cpu().numpy()
            if v_np.ndim == 2:
                depth, n_heads = v_np.shape
                if cols_by_head is None:
                    cols_by_head = ["layer"] + [f"head_{h}" for h in range(n_heads)]
                payload[f"test_cls/table_{k}"] = wandb.Table(
                    columns=cols_by_head,
                    data=[[li, *v_np[li].tolist()] for li in range(depth)],
                )
            elif v_np.ndim == 1:
                depth = v_np.shape[0]
                payload[f"test_cls/table_{k}"] = wandb.Table(
                    columns=["layer", "value"],
                    data=[[li, float(v_np[li])] for li in range(depth)],
                )
        # Alpha / beta tables.
        for role, arr in final_ab.items():
            arr_np = arr.cpu().numpy()
            depth, n_heads = arr_np.shape
            cols = ["layer"] + [f"head_{h}" for h in range(n_heads)]
            payload[f"test_ab/table_{role}"] = wandb.Table(
                columns=cols, data=[[li, *arr_np[li].tolist()] for li in range(depth)],
            )
        wandb.log(payload, step=global_step)

    # --- Per-slide CSV --------------------------------------------------
    # Adds mean_h_morph to the stage-2 schema — this is the slide-intrinsic
    # covariate the homogeneity regression (§13.4.3a) consumes. Under the
    # XSA backend (xsa_all_ref ablation only) we still compute it from the
    # SRP data path since the batch dict always contains h_morph; it's a
    # slide-intrinsic quantity independent of model state.
    K = args.num_classes
    base_cols = [
        "slide_id", "patient_id", "fold", "center", "N_tokens",
        "ablation", "y_true", "y_pred_class",
    ]
    logit_cols = [f"y_pred_logit_{k}" for k in range(K)]
    prob_cols = [f"y_pred_prob_{k}" for k in range(K)]
    fieldnames = base_cols + logit_cols + prob_cols + ["per_slide_loss", "mean_h_morph"]
    fold_for_csv = "" if args.fold is None else args.fold
    fold_for_artifacts = _numeric_fold_id_for_artifacts(args.fold)
    global_seed_for_artifacts = -1 if args.global_seed is None else int(args.global_seed)

    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_slide:
            record = {
                "slide_id": row["slide_id"],
                "patient_id": row["patient_id"],
                # Foldless global-seed runs keep this CSV cell blank.  The
                # paired-result collector gets the seed from the run name, and
                # fixed-fold CV runs keep their historical integer value.
                "fold": fold_for_csv,
                "center": row["center"],
                "N_tokens": row["N_tokens"],
                "ablation": args.ablation,
                "y_true": row["y_true"],
                "y_pred_class": row["y_pred_class"],
                "per_slide_loss": f"{row['per_slide_loss']:.6g}",
                "mean_h_morph": f"{row['mean_h_morph']:.6g}",
            }
            for k in range(K):
                record[f"y_pred_logit_{k}"] = f"{row['y_logits'][k]:.6g}"
                record[f"y_pred_prob_{k}"] = f"{row['y_probs'][k]:.6g}"
            writer.writerow(record)
    print(f"[{args.run_name}] wrote {csv_path}  ({len(per_slide)} rows, K={K})")

    # --- Per-slide SRP diagnostics (supplementary, analyze.py §8.2.D/E) ---
    # Only populated for SRP-backend ablations; for xsa_all_ref this list
    # is empty and we skip writing the npz so downstream code can
    # distinguish "SRP diagnostics unavailable" from "empty".
    if per_slide_diag:
        slide_ids = np.array([r["slide_id"] for r in per_slide_diag], dtype=object)
        patient_ids = np.array([r["patient_id"] for r in per_slide_diag], dtype=object)
        centers = np.array([r["center"] for r in per_slide_diag], dtype=np.int64)
        n_tokens = np.array([r["N_tokens"] for r in per_slide_diag], dtype=np.int64)
        y_true_arr = np.array([r["y_true"] for r in per_slide_diag], dtype=np.int64)
        y_pred_arr = np.array([r["y_pred_class"] for r in per_slide_diag], dtype=np.int64)
        mean_h_morph_arr = np.array([r["mean_h_morph"] for r in per_slide_diag], dtype=np.float32)

        def _stack(key):
            return np.stack([r[key] for r in per_slide_diag], axis=0)

        psd_payload = {
            "slide_id":       slide_ids,
            "patient_id":     patient_ids,
            "center":         centers,
            "N_tokens":       n_tokens,
            # NPZ diagnostics use numeric arrays.  `-1` means the split was
            # generated by `global_seed_holdout` and therefore has no CV fold.
            "fold":           np.full(len(per_slide_diag), fold_for_artifacts, dtype=np.int64),
            "global_seed":    np.full(len(per_slide_diag), global_seed_for_artifacts, dtype=np.int64),
            "split_mode":     np.array([args.split_mode] * len(per_slide_diag), dtype=object),
            "y_true":         y_true_arr,
            "y_pred_class":   y_pred_arr,
            "mean_h_morph":   mean_h_morph_arr,
            "cos_y_cls_rbar":    _stack("cos_y_cls_rbar"),      # (S, D, H)
            "mean_h_V":          _stack("mean_h_V"),             # (S, D, H)
            "mean_cos_yr_pre":   _stack("mean_cos_yr_pre"),      # (S, D, H)
            "mean_cos_yr_post":  _stack("mean_cos_yr_post"),     # (S, D, H)
            "z_over_y_by_h_morph_quartile":  _stack("z_over_y_by_h_morph_quartile"),  # (S, 4, D, H)
            "ablation":       np.array([args.ablation] * len(per_slide_diag), dtype=object),
        }
        # pre_v-only fields (present only when srp_mode=pre_v under this
        # ablation). We check the first record for key presence; SRP
        # mode is constant across a run.
        if "bar_rho_cls" in per_slide_diag[0]:
            psd_payload["bar_rho_cls"] = _stack("bar_rho_cls")    # (S, D, H)
        if "mean_rho" in per_slide_diag[0]:
            psd_payload["mean_rho"]    = _stack("mean_rho")       # (S, D, H)
        np.savez(out_dir / "per_slide_diagnostics.npz", **psd_payload)
        print(f"[{args.run_name}] wrote per_slide_diagnostics.npz "
              f"({len(per_slide_diag)} slides)")

    # --- Artifact npz ----------------------------------------------------
    npz_payload = {
        "test_metrics": json.dumps(test_metrics),
        "best_val_f1": best_val_f1,
        "stage1_best_val_f1": stage1_best_val_f1,
        "stage2_best_val_f1": stage2_best_val_f1,
        "gate_l2_reg": float(args.gate_l2_reg),
        "gate_l2_missing_steps": int(gate_l2_missing_steps),
        "srp_freeze_epochs": int(args.srp_freeze_epochs),
        "stage2_epochs": int(args.stage2_epochs),
        "stage2_mode": args.stage2_mode,
        "stage2_lr_mult": float(args.stage2_lr_mult),
        "train_history": json.dumps(train_history),
        "backend": backend,
        "ablation": args.ablation,
        # Store architecture-level switches inside the artifact, not only in
        # the run name/W&B config, so offline analysis can recover the exact
        # factorial arm from `test_artifacts.npz` alone.
        "layerscale_init": float(args.layerscale_init),
        "ln_specialization": args.ln_specialization,
        "ln_specialization_scope": args.ln_specialization_scope,
    }
    # Phase-A.9 calibration-reframe instrumentation
    # ([CALIBRATION_REFRAME_2026-04-28.md](analysis_phaseA/CALIBRATION_REFRAME_2026-04-28.md)).
    # Persist (y_true, y_logits) so downstream ECE / Brier / NLL /
    # temperature-scaling analyses don't need a re-evaluation.
    if test_eval_arrays.get("y_true") is not None:
        npz_payload["test_y"] = test_eval_arrays["y_true"]
    if test_eval_arrays.get("y_logits") is not None:
        npz_payload["test_logits"] = test_eval_arrays["y_logits"]
    for role, arr in final_ab.items():
        npz_payload[role] = arr.cpu().numpy()
    for role, arr in final_layerscale.items():
        npz_payload[role] = arr.cpu().numpy()
    if test_cls_stats is not None:
        for k, v in test_cls_stats.items():
            npz_payload[k] = v.cpu().numpy()
    if len(ab_history) > 0:
        npz_payload["ab_history_steps"] = np.array(
            [h["step"] for h in ab_history], dtype=np.int64,
        )
        if backend == "xsa":
            npz_payload["alpha_history_cls"] = np.stack(
                [h["alpha_cls"] for h in ab_history], axis=0,
            )
            npz_payload["alpha_history_patch"] = np.stack(
                [h["alpha_patch"] for h in ab_history], axis=0,
            )
        else:
            npz_payload["beta_history_patch"] = np.stack(
                [h["beta_patch"] for h in ab_history], axis=0,
            )
    # Per-example signed-gate stats (proposal §2.6). Empty dict for
    # non-signed-gated runs; under signed_gated the keys are the
    # gate_block{i}_{stat} entries that downstream analysis relies on.
    for k, v in test_gate_stats.items():
        npz_payload[f"gate_{k}"] = v
    np.savez(out_dir / "test_artifacts.npz", **npz_payload)

    wandb.finish()
    print(f"[{args.run_name}] done")


if __name__ == "__main__":
    main()
