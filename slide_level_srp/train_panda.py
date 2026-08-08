"""PANDA single-run training entry point.

The PANDA task comparison uses the TransMIL-style `--arch transmil` path so NA,
XSA, Diff, and Gated SRP share the same slide-level scaffold. The
attention-operator comparison also exposes `--arch vit4`, a dense set-style
ViT over native-length PANDA UNI-v2 slide features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, confusion_matrix,
)
from tqdm import tqdm

import wandb

from src.data_panda import (
    build_panda_folds, build_panda_global_seed_splits, build_panda_loaders,
    enumerate_slides, N_ISUP, PANDA_H5_DIR, UNI_DIM,
)
from src.vit_panda import PandaSlideViT
from slide_level_srp.src.diff_transformer import DiffBlock, RMSNorm
from slide_level_srp.src.mil_baselines import dsmil_dual_stream_cross_entropy
from slide_level_srp.src.runtime_profile import RuntimeProfiler


# --- Helpers ------------------------------------------------------------
_SIGNED_GATE_MODES = {
    "srp_signed_gated",
    "srp_signed_gated_pre_q",
    "srp_signed_gated_pre_k",
}
_FIXED_BETA_MODES = {"srp_beta2", "srp_fixed_beta"}
_LEARNED_R_SIGNED_GATE_MODES = {"srp_signed_gated_learned_r"}
_RCD_MODES = {"srp_rcd", "srp_rcd_learned_r"}
_CAPACITY_CONTROL_MODES = {"srp_mlp_control"}
_MIL_BASELINE_MODES = {"abmil", "dsmil", "official_transmil"}
_DENSE_MODES = {"dense_mhsa", "dense_mhsa_srp"}
_DENSE_SRP_MODES = {"dense_mhsa_srp"}
_OFFICIAL_ARCH_MODES = {
    "official_span_baseline",
    "official_span_srp",
    "official_longnet_baseline",
    "official_longnet_srp",
}
_OFFICIAL_ARCH_SRP_MODES = {"official_span_srp", "official_longnet_srp"}
_METHOD_SURFACE_MODES = (
    _SIGNED_GATE_MODES
    | _LEARNED_R_SIGNED_GATE_MODES
    | _RCD_MODES
    | _CAPACITY_CONTROL_MODES
    | _DENSE_SRP_MODES
    | _OFFICIAL_ARCH_SRP_MODES
)
_TRANSMIL_NEIGHBOR_SRP_MODES = (
    {"baseline"}
    | _FIXED_BETA_MODES
    | _SIGNED_GATE_MODES
    | _LEARNED_R_SIGNED_GATE_MODES
    | _RCD_MODES
    | _CAPACITY_CONTROL_MODES
    | _DENSE_SRP_MODES
)


def transmil_srp_mode_for_panda_mode(mode: str) -> str:
    """Map PANDA mode names onto the shared TransMIL SRP aggregator names."""
    mapping = {
        "baseline": "post_agg",
        "srp_beta2": "post_agg",
        # Audit-friendly alias for fixed beta values other than 2.0.  The
        # existing srp_beta2 path already accepted --beta <value>; the new
        # name avoids implying beta=2 in fixed-beta variant manifests.
        "srp_fixed_beta": "post_agg",
        "srp_signed_gated": "post_agg_signed_gated",
        "srp_signed_gated_pre_q": "pre_q_signed_gated",
        "srp_signed_gated_pre_k": "pre_k_signed_gated",
        "srp_signed_gated_learned_r": "post_agg_signed_gated_learned_r",
        "srp_rcd": "post_agg_rcd",
        "srp_rcd_learned_r": "post_agg_rcd_learned_r",
        "srp_mlp_control": "post_agg_mlp_control",
    }
    if mode not in mapping:
        raise ValueError(f"PANDA TransMIL has no SRP-mode mapping for mode={mode!r}")
    return mapping[mode]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int,
                     base_lr: float, min_lr: float = 0.0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def compute_panda_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          y_prob: np.ndarray) -> dict:
    """
    PANDA metric battery:
      kappa_quad   — Kaggle's official metric. Penalises 0↔5 confusion
                     much more than 0↔1.
      kappa_lin    — linear-weighted variant for context.
      acc          — top-1 accuracy.
      balanced_acc — average per-class recall (insensitive to class freq).
      macro_f1     — macro F1 across 6 ISUP classes.
      macro_auc    — macro one-vs-rest ROC AUC across 6 ISUP classes.
      binary_f1    — any-cancer F1 (ISUP > 0).
      binary_auc   — same, ROC AUC under one-vs-rest on the
                     `1 - prob[ISUP=0]` score.
    """
    from sklearn.metrics import roc_auc_score
    out = {
        "kappa_quad":    float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "kappa_lin":     float(cohen_kappa_score(y_true, y_pred, weights="linear")),
        "acc":           float(accuracy_score(y_true, y_pred)),
        "balanced_acc":  float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1":      float(f1_score(y_true, y_pred, average="macro",
                                        zero_division=0,
                                        labels=list(range(N_ISUP)))),
        "binary_f1":     float(f1_score((y_true > 0).astype(int),
                                        (y_pred > 0).astype(int),
                                        zero_division=0)),
    }
    # Binary AUC: probability of ANY cancer = 1 - P(ISUP=0).
    try:
        bin_score = 1.0 - y_prob[:, 0]
        out["binary_auc"] = float(roc_auc_score((y_true > 0).astype(int),
                                                 bin_score))
    except Exception:
        out["binary_auc"] = float("nan")
    # global-seed tables use a common F1 / ACC / AUC surface across datasets.
    # PANDA is multiclass ordinal, so use macro OvR AUC for the unified AUC
    # column while retaining Kaggle kappa and binary AUC as additional metrics.
    try:
        out["macro_auc"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=list(range(N_ISUP)),
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        out["macro_auc"] = float("nan")
    out["f1"] = out["macro_f1"]
    out["auc"] = out["macro_auc"]
    return out


# --- Model / forward dispatch -------------------------------------------

class PandaDiffTransformerClassifier(torch.nn.Module):
    """Set-style PANDA classifier with Diff Transformer blocks.

    PANDA batches are padded variable-length bags.  The project Diff
    Transformer slide aggregator intentionally ignores masks because the
    TransMIL path is single-bag oriented.  This wrapper trims each PANDA sample
    by its real-token mask before applying the same differential-attention
    block, so pad rows cannot absorb attention mass or change gradients.
    """

    def __init__(
        self,
        in_dim: int = UNI_DIM,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        num_landmarks: int = 64,
        num_classes: int = N_ISUP,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        pinv_iterations: int = 6,
    ) -> None:
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError(
                "PANDA Diff Transformer requires an even --num_heads; "
                f"got {num_heads}."
            )
        self.in_proj = torch.nn.Linear(in_dim, embed_dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist() if depth > 0 else []
        self.blocks = torch.nn.ModuleList(
            [
                DiffBlock(
                    dim=embed_dim,
                    depth_index=i,
                    baseline_num_heads=num_heads,
                    num_landmarks=num_landmarks,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=False,
                    drop_path=dpr[i],
                    pinv_iterations=pinv_iterations,
                    checkpoint_mode="off",
                )
                for i in range(depth)
            ]
        )
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = torch.nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                torch.nn.init.ones_(module.weight)

    def _forward_one(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[0] <= 0:
            raise ValueError("PANDA Diff Transformer received an empty slide bag.")
        x = self.in_proj(features.unsqueeze(0))
        cls = self.cls_token.expand(1, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = []
        for row_features, row_mask in zip(features, mask):
            logits.append(self._forward_one(row_features[row_mask.bool()]))
        return torch.cat(logits, dim=0)


def build_model(args) -> torch.nn.Module:
    # Default in_dim=1536 preserves UNI-v2.  Encoder-transfer variants set
    # this explicitly so the model projection and the H5 feature validation
    # share one reported dimensionality.
    in_dim = int(getattr(args, "in_dim", UNI_DIM))
    if args.mode == "diff_transformer" and args.arch in ("vit4", "vit12"):
        if args.arch not in ("vit4", "vit12"):
            raise ValueError("PANDA Diff Transformer supports --arch vit4 or vit12.")
        depth = 4 if args.arch == "vit4" else 12
        return PandaDiffTransformerClassifier(
            in_dim=in_dim,
            embed_dim=args.embed_dim,
            depth=depth,
            num_heads=args.num_heads,
            num_landmarks=args.num_landmarks,
            num_classes=N_ISUP,
            drop_path_rate=args.drop_path,
        )
    if args.arch in ("vit4", "vit12"):
        depth = 4 if args.arch == "vit4" else 12
        # Validation: programmatic callers that
        # bypass parse_args() (tests, future launchers, notebooks) used
        # to silently get their `srp_r_target` coerced to "knn8" under
        # signed-gated mode. That hides experiment-config errors. Now
        # raise explicitly — parse_args() already validates the same
        # combination at CLI level, so this is the matching API guard.
        r_target_eff = args.srp_r_target
        if (
            args.mode in _SIGNED_GATE_MODES
            or args.mode in _LEARNED_R_SIGNED_GATE_MODES
            or args.mode == "srp_rcd_learned_r"
        ) and r_target_eff != "knn8":
            raise ValueError(
                f"build_model: mode={args.mode!r} requires "
                f"r_target='knn8'; got r_target='{r_target_eff}'. "
                f"Set args.srp_r_target='knn8' or use --srp_r_target knn8."
            )
        return PandaSlideViT(
            in_dim=in_dim, embed_dim=args.embed_dim, depth=depth,
            num_heads=args.num_heads, num_classes=N_ISUP,
            drop_path_rate=args.drop_path,
            mode=args.mode, beta=args.beta,
            r_target=r_target_eff,
            # Signed-gate parameters; consumed only under signed-gate modes.
            # Default delta_scale=2.0
            # matches the signed-gate design
            delta_scale=args.delta_scale,
            gate_hidden_dim=args.gate_hidden_dim,
            detach_gate_inputs=not args.no_detach_gate_inputs,
            gate_output_init=args.gate_output_init,
            gate_output_init_scale=args.gate_output_init_scale,
            gate_init_beta0=args.gate_init_beta0,
            gate_activation=args.gate_activation,
            gate_activation_temperature=args.gate_activation_temperature,
            gate_count_features=args.gate_count_features,
            rcd_adapter_kind=args.rcd_adapter_kind,
            rcd_rank=args.rcd_rank,
            learned_r_hidden_dim=args.learned_r_hidden_dim,
            pos_mode=args.pos_mode,
            coord_pos_dim=args.coord_pos_dim,
            coord_norm=args.coord_norm,
        )
    elif args.arch == "transmil":
        from slide_level.src.aggregator import NystromXSAggregator
        from slide_level_srp.src.dense_srp_aggregator import DenseAttentionSRPAggregator
        from slide_level_srp.src.diff_transformer import NystromDiffTransformerAggregator
        from slide_level_srp.src.mil_baselines import build_mil_baseline_aggregator
        from slide_level_srp.src.official_architectures import (
            OfficialGigaPathLongNetAggregator,
            OfficialSPANAggregator,
        )
        from slide_level_srp.src.srp_aggregator import NystromSRPAggregator

        if args.mode in _MIL_BASELINE_MODES:
            return build_mil_baseline_aggregator(
                kind=args.mode,
                in_dim=in_dim,
                num_classes=N_ISUP,
            )

        if args.mode == "xsa_all_hard":
            # Hard XSA is the original XSA comparator: fixed alpha buffers change
            # the forward projection but intentionally add no trainable surface.
            return NystromXSAggregator(
                in_dim=in_dim, embed_dim=args.embed_dim, depth=4,
                num_heads=args.num_heads, num_landmarks=args.num_landmarks,
                num_classes=N_ISUP,
                alpha_cls_mode="one", alpha_patch_mode="one",
                alpha_init=1.0,
                drop_path_rate=args.drop_path,
                pinv_iterations=6, checkpoint_mode="off",
            )

        if args.mode == "nystrom_na":
            return NystromXSAggregator(
                in_dim=in_dim, embed_dim=args.embed_dim, depth=4,
                num_heads=args.num_heads, num_landmarks=args.num_landmarks,
                num_classes=N_ISUP,
                alpha_cls_mode="zero", alpha_patch_mode="zero",
                alpha_init=1.0,
                drop_path_rate=args.drop_path,
                pinv_iterations=6, checkpoint_mode="off",
            )

        if args.mode == "diff_transformer":
            # This is the WSI-scale Diff Transformer comparator used by the
            # slide-level trainer: differential attention inside the same
            # Nystrom/PPEG scaffold, rather than the set-style PANDA ViT wrapper.
            return NystromDiffTransformerAggregator(
                in_dim=in_dim, embed_dim=args.embed_dim, depth=4,
                num_heads=args.num_heads, num_landmarks=args.num_landmarks,
                num_classes=N_ISUP,
                drop_path_rate=args.drop_path,
                pinv_iterations=6, checkpoint_mode="off",
            )

        if args.mode in _DENSE_MODES:
            return DenseAttentionSRPAggregator(
                in_dim=in_dim,
                embed_dim=args.embed_dim,
                depth=4,
                num_heads=args.num_heads,
                num_classes=N_ISUP,
                drop_path_rate=args.drop_path,
                checkpoint_mode="off",
                use_srp=args.mode == "dense_mhsa_srp",
                delta_scale=args.delta_scale,
                gate_hidden_dim=args.gate_hidden_dim,
                detach_gate_inputs=not args.no_detach_gate_inputs,
                gate_output_init=args.gate_output_init,
                gate_output_init_scale=args.gate_output_init_scale,
                gate_init_beta0=args.gate_init_beta0,
                gate_activation=args.gate_activation,
                gate_activation_temperature=args.gate_activation_temperature,
                gate_factorization=args.gate_factorization,
                gate_delta_mode=args.gate_delta_mode,
                gate_count_features=args.gate_count_features,
                retain_gate_beta_for_loss=args.gate_l2_reg > 0.0,
            )

        if args.mode in _OFFICIAL_ARCH_MODES:
            if args.mode.startswith("official_span"):
                return OfficialSPANAggregator(
                    in_dim=in_dim,
                    num_classes=N_ISUP,
                    target="panda",
                    use_srp=args.mode.endswith("_srp"),
                    gate_hidden_dim=args.gate_hidden_dim,
                    delta_scale=args.delta_scale,
                )
            return OfficialGigaPathLongNetAggregator(
                in_dim=in_dim,
                embed_dim=args.embed_dim,
                depth=args.official_longnet_depth,
                num_classes=N_ISUP,
                use_srp=args.mode.endswith("_srp"),
                drop_path_rate=args.drop_path,
                gate_hidden_dim=args.gate_hidden_dim,
                delta_scale=args.delta_scale,
            )

        if args.mode in _TRANSMIL_NEIGHBOR_SRP_MODES:
            if args.mode == "baseline":
                beta_patch_mode = "zero"
                beta_init = 0.0
            elif args.mode in _FIXED_BETA_MODES:
                beta_patch_mode = "fixed"
                beta_init = float(args.beta)
            elif args.mode in _RCD_MODES:
                # RCD modes disable the scalar beta path and route the method
                # through the learned recomposer, matching the shared SRP module
                # contract.
                beta_patch_mode = "zero"
                beta_init = 0.0
            elif args.mode in _CAPACITY_CONTROL_MODES:
                # Matched-capacity variant: add the same kind of learned
                # adapter surface without spatial SRP geometry.  The shared
                # SRP module requires beta disabled for this mode by design.
                beta_patch_mode = "zero"
                beta_init = 0.0
            else:
                beta_patch_mode = "signed_gated"
                beta_init = 0.0
            return NystromSRPAggregator(
                in_dim=in_dim, embed_dim=args.embed_dim, depth=4,
                num_heads=args.num_heads, num_landmarks=args.num_landmarks,
                num_classes=N_ISUP,
                beta_patch_mode=beta_patch_mode,
                beta_init=beta_init,
                srp_mode=transmil_srp_mode_for_panda_mode(args.mode),
                drop_path_rate=args.drop_path,
                pinv_iterations=6,
                checkpoint_mode="off",
                ln_specialization=getattr(args, "ln_specialization", "shared"),
                ln_specialization_scope=getattr(args, "ln_specialization_scope", "block"),
                delta_scale=args.delta_scale,
                gate_hidden_dim=args.gate_hidden_dim,
                detach_gate_inputs=not args.no_detach_gate_inputs,
                gate_output_init=args.gate_output_init,
                gate_output_init_scale=args.gate_output_init_scale,
                gate_init_beta0=args.gate_init_beta0,
                gate_activation=args.gate_activation,
                gate_activation_temperature=args.gate_activation_temperature,
                gate_factorization=args.gate_factorization,
                gate_delta_mode=args.gate_delta_mode,
                gate_count_features=args.gate_count_features,
                srp_context_impl=args.srp_context_impl,
                srp_correction_chunk_size=args.srp_correction_chunk_size,
                retain_gate_beta_for_loss=args.gate_l2_reg > 0.0,
                rcd_adapter_kind=args.rcd_adapter_kind,
                rcd_rank=args.rcd_rank,
                learned_r_hidden_dim=args.learned_r_hidden_dim,
            )

        raise ValueError(
            f"PANDA TransMIL does not support mode={args.mode!r}. "
            "Supported comparison methods are baseline, xsa_all_hard, "
            "diff_transformer, and srp_signed_gated."
        )
    else:
        raise ValueError(f"unknown --arch={args.arch}")


def gate_diagnostics(model, prefix: str = "gate") -> dict:
    """
    Walk the gate-active blocks of a PandaSlideViT and summarise the
    most recently-cached `_last_gate_stats` from each. Returns a flat
    dict suitable for `wandb.log(...)`.

    Stats computed per block (by design):
      mean / std of beta_eff
      fraction of beta_eff in {neg, near_zero, projection, reflection}
      mean of cos(y, r_hat) — alignment of attention output with the
        local mean direction; gate decisions depend on this signal
      Pearson correlation between beta_eff (mean across heads) and
        h_local — a non-zero correlation says the gate is using the
        homogeneity input.

    The bins are:
      neg         : beta_eff <= -0.5
      near_zero   : -0.5 < beta_eff < 0.5
      projection  : 0.5 <= beta_eff <= 1.5  (≈ projection regime)
      reflection  : beta_eff > 1.5          (≈ reflection regime)

    This must be called immediately after a forward pass (the
    `_last_gate_stats` cache is overwritten each forward). It returns
    an empty dict when no gate is active anywhere in the model — the
    caller can blindly merge it into a wandb.log payload.
    """
    out: dict[str, float] = {}
    if not hasattr(model, "blocks"):
        return out
    for i, blk in enumerate(model.blocks):
        if not hasattr(blk, "attn"):
            continue
        stats = getattr(blk.attn, "_last_gate_stats", None)
        if stats is None:
            continue
        beta_eff = stats["beta_eff"].float()                # (B, H, N, 1)
        beta_flat = beta_eff.flatten()
        n_total = beta_flat.numel()
        if n_total == 0:
            # Last-batch W&B diagnostics should never invalidate a completed
            # training/eval pass. Mirror GateStatsAccumulator's empty-example
            # policy: keep the keys present, but mark values as NaN.
            out[f"{prefix}/block{i}/beta_eff_mean"] = float("nan")
            out[f"{prefix}/block{i}/beta_eff_std"] = float("nan")
            out[f"{prefix}/block{i}/frac_neg"] = float("nan")
            out[f"{prefix}/block{i}/frac_near_zero"] = float("nan")
            out[f"{prefix}/block{i}/frac_projection"] = float("nan")
            out[f"{prefix}/block{i}/frac_reflection"] = float("nan")
            if "cos_yr" in stats:
                out[f"{prefix}/block{i}/mean_cos_yr"] = float("nan")
            if "h_local" in stats:
                out[f"{prefix}/block{i}/corr_beta_h_local"] = float("nan")
            continue
        out[f"{prefix}/block{i}/beta_eff_mean"] = float(beta_flat.mean().item())
        out[f"{prefix}/block{i}/beta_eff_std"] = float(beta_flat.std().item())
        out[f"{prefix}/block{i}/frac_neg"] = float(
            (beta_flat <= -0.5).sum().item() / n_total)
        out[f"{prefix}/block{i}/frac_near_zero"] = float(
            ((beta_flat > -0.5) & (beta_flat < 0.5)).sum().item() / n_total)
        out[f"{prefix}/block{i}/frac_projection"] = float(
            ((beta_flat >= 0.5) & (beta_flat <= 1.5)).sum().item() / n_total)
        out[f"{prefix}/block{i}/frac_reflection"] = float(
            (beta_flat > 1.5).sum().item() / n_total)
        # Mean cos(y, r_hat) — a contextual signal for the gate.
        if "cos_yr" in stats:
            out[f"{prefix}/block{i}/mean_cos_yr"] = float(
                stats["cos_yr"].float().mean().item())
        # Correlation gate ↔ h_local. beta_eff is (B, H, N, 1); reduce
        # over the head axis (per-token mean) and align with h_local
        # (B, N) before computing pearson.
        if "h_local" in stats:
            hl = stats["h_local"].float().flatten()         # (B*N,)
            be = beta_eff.mean(dim=1).squeeze(-1).flatten()  # (B*N,)
            if hl.numel() > 1 and be.std() > 1e-8 and hl.std() > 1e-8:
                # Pearson correlation; avoid call to torch.corrcoef
                # for back-compat with older torch versions.
                hl_c = hl - hl.mean()
                be_c = be - be.mean()
                corr = float(
                    (hl_c * be_c).sum().item()
                    / (hl_c.norm().item() * be_c.norm().item() + 1e-12)
                )
                out[f"{prefix}/block{i}/corr_beta_h_local"] = corr
    return out


def collect_gate_beta_l2(model, device: torch.device) -> tuple[torch.Tensor, int]:
    """Collect differentiable `mean(beta_eff^2)` from PANDA signed gates.

    PandaAttention stores a live beta cache only for gate-active blocks during
    the current forward pass.  We average block means rather than summing over
    every token/head entry so unusually long PANDA slides do not dominate the
    regularizer just because they contain more patches.
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


def is_method_param(name: str, mode: str) -> bool:
    """Return True for PANDA XSA/SRP parameters controlled by the intervention.

    two-stage needs the same method/non-method split as the slide trainer.  For
    signed-gated PANDA, the learned intervention surface is every parameter in
    each `*.gate.*` module; for legacy hard/scalar modes, it is the alpha/beta
    scalar surface.  Keeping this as a helper prevents the freeze mask and
    optimizer grouping from drifting apart.
    """
    if mode in _SIGNED_GATE_MODES:
        return ".gate." in name or name.startswith("gate.")
    if mode in _LEARNED_R_SIGNED_GATE_MODES:
        # Standalone Method 2.4 keeps the signed gate and adds a learned
        # local-r scorer.  It must not include rcd_recomposer parameters,
        # otherwise method-only/freeze probes would silently become Method 2.1.
        return (
            ".gate." in name
            or name.startswith("gate.")
            or ".context_scorer." in name
            or name.startswith("context_scorer.")
        )
    if mode in _RCD_MODES:
        return (
            ".rcd_recomposer." in name
            or name.startswith("rcd_recomposer.")
            or ".context_scorer." in name
            or name.startswith("context_scorer.")
        )
    if mode in _CAPACITY_CONTROL_MODES:
        # The capacity-control branch is the method surface for that variant:
        # it should be counted/frozen independently from the shared TransMIL
        # backbone during any future two-stage probe.
        return ".mlp_control." in name or name.startswith("mlp_control.")
    if mode in _DENSE_SRP_MODES:
        return ".gate." in name or name.startswith("gate.")
    if mode in _OFFICIAL_ARCH_SRP_MODES:
        return ".srp_modules." in name or name.startswith("srp_modules.")
    return ("alpha_cls" in name or "alpha_patch" in name or "beta_patch" in name)


def is_method_no_decay(name: str, mode: str) -> bool:
    """Subset of method params that should be protected from AdamW decay.

    Gate weights still receive normal decay; only gate biases are no-decay so
    zero/near-zero identity initialization is not mechanically pulled back by
    AdamW after the gate starts moving.  This mirrors the existing gate-containment
    grouping and the CAM16/CAM17/KGH slide trainer.
    """
    if mode in _SIGNED_GATE_MODES:
        return (
            ".gate." in name
            and (name.endswith(".layer_head_bias")
                 or name.endswith(".head_bias")
                 or name.endswith(".bias"))
        )
    if mode in _LEARNED_R_SIGNED_GATE_MODES:
        return (
            is_method_param(name, mode)
            and (name.endswith(".layer_head_bias")
                 or name.endswith(".head_bias")
                 or name.endswith(".bias"))
        )
    if mode in _RCD_MODES:
        return (
            is_method_param(name, mode)
            and (name.endswith(".bias") or name.endswith("_diag_delta"))
        )
    if mode in _CAPACITY_CONTROL_MODES:
        return is_method_param(name, mode) and name.endswith(".bias")
    if mode in (_DENSE_SRP_MODES | _OFFICIAL_ARCH_SRP_MODES):
        return is_method_param(name, mode) and name.endswith(".bias")
    return ("alpha_cls" in name or "alpha_patch" in name or "beta_patch" in name)


def _stage_trainability_mode_name(mode: str) -> str:
    names = {
        "all": "all parameters trainable",
        "freeze_method": "PANDA SRP/gate/RCD method parameters frozen",
        "method_only": "only PANDA SRP/gate/RCD method parameters trainable",
    }
    if mode not in names:
        raise ValueError(f"unknown trainability mode {mode!r}")
    return names[mode]


def apply_trainability_mode(model, args, mode: str) -> dict[str, int]:
    """Set `requires_grad` for one PANDA two-stage training stage.

    Stage 1 freezes the signed-gate/SRP intervention so the ordinary PANDA
    backbone/head can converge as a standard-attention reference.  Stage 2 then
    either trains only the intervention surface (`srp_only`) or jointly trains
    everything (`joint`) from the best Stage-1 checkpoint.
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
        method = is_method_param(name, args.mode)
        if mode == "all":
            trainable = True
        elif mode == "freeze_method":
            trainable = not method
        else:
            trainable = method
        param.requires_grad = trainable
        key = (
            "method_trainable" if method and trainable else
            "method_frozen" if method else
            "nonmethod_trainable" if trainable else
            "nonmethod_frozen"
        )
        counts[key] += param.numel()
    return counts


def build_adamw_optimizer(model, args) -> torch.optim.Optimizer:
    """Build PANDA AdamW groups after any two-stage freeze/unfreeze change."""
    backbone_decay, method_decay, method_nodecay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_method_no_decay(name, args.mode):
            method_nodecay.append(param)
        elif is_method_param(name, args.mode):
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
        param_groups.append({
            "params": method_nodecay,
            "weight_decay": 0.0,
            "group_name": "ab_nodecay",
        })
    if not param_groups:
        raise RuntimeError(
            f"[{args.run_name}] no trainable PANDA parameters under current "
            "two-stage trainability mode."
        )
    return torch.optim.AdamW(param_groups, lr=args.base_lr, betas=(0.9, 0.999))


def method_parameter_summary(model, args) -> tuple[list[str], int, int, int]:
    """Return method names, method count, trainable count, and total count."""
    names = [
        name for name, param in model.named_parameters()
        if param.requires_grad and is_method_param(name, args.mode)
    ]
    n_method = sum(
        param.numel() for name, param in model.named_parameters()
        if param.requires_grad and is_method_param(name, args.mode)
    )
    n_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    n_total = sum(param.numel() for param in model.parameters())
    return names, n_method, n_trainable, n_total


def model_forward(model, batch, device, arch: str,
                  srp_r_target: str = "slide_mean",
                  mode: str = "baseline") -> torch.Tensor:
    feats = batch["features"].to(device, non_blocking=True)            # (B, N_max, 1536)
    if mode == "diff_transformer" and arch in ("vit4", "vit12"):
        mask = batch["mask"].to(device, non_blocking=True)
        return model(feats, mask)
    if arch in ("vit4", "vit12"):
        mask = batch["mask"].to(device, non_blocking=True)              # (B, N_max)
        # Only ship neighbor tensors when the model actually needs them.
        # Sending them unconditionally would add a small CPU→GPU copy on
        # every step for the slide_mean / baseline / xsa_all_hard paths.
        # signed-gated always uses knn8 r̂, so it requires the neighbour
        # graph + h_local — handled by the same branch.
        if (
            srp_r_target == "knn8"
            or mode in _SIGNED_GATE_MODES
            or mode in _LEARNED_R_SIGNED_GATE_MODES
            or mode == "srp_rcd_learned_r"
        ):
            nbi = batch["neighbor_index"].to(device, non_blocking=True)  # (B, N_max, 8)
            nbm = batch["neighbor_mask"].to(device, non_blocking=True)   # (B, N_max, 8)
            nbw = batch.get("neighbor_weight")
            nbw = nbw.to(device, non_blocking=True) if nbw is not None else None
            coords = batch.get("coords")
            coords = coords.to(device, non_blocking=True) if coords is not None else None
            if (
                mode in _SIGNED_GATE_MODES
                or mode in _LEARNED_R_SIGNED_GATE_MODES
                or mode == "srp_rcd_learned_r"
            ):
                hloc = batch["h_local"].to(device, non_blocking=True)   # (B, N_max)
                return model(feats, mask,
                             neighbor_index=nbi, neighbor_mask=nbm,
                             neighbor_weight=nbw, h_local=hloc,
                             coords=coords)
            return model(
                feats, mask,
                neighbor_index=nbi, neighbor_mask=nbm,
                neighbor_weight=nbw, coords=coords,
            )
        coords = batch.get("coords")
        coords = coords.to(device, non_blocking=True) if coords is not None else None
        return model(feats, mask, coords=coords)
    if arch == "transmil":
        if mode in _OFFICIAL_ARCH_MODES:
            nbi = batch["neighbor_index"].to(device, non_blocking=True)
            nbm = batch["neighbor_mask"].to(device, non_blocking=True)
            nbw = batch.get("neighbor_weight")
            nbw = nbw.to(device, non_blocking=True) if nbw is not None else None
            coords = batch["coords"].to(device, non_blocking=True)
            return model(
                feats,
                neighbor_index=nbi,
                neighbor_mask=nbm,
                neighbor_weight=nbw,
                coords=coords,
            )
        if mode in _TRANSMIL_NEIGHBOR_SRP_MODES:
            nbi = batch["neighbor_index"].to(device, non_blocking=True)
            nbm = batch["neighbor_mask"].to(device, non_blocking=True)
            nbw = batch.get("neighbor_weight")
            nbw = nbw.to(device, non_blocking=True) if nbw is not None else None
            if mode in (
                _SIGNED_GATE_MODES
                | _LEARNED_R_SIGNED_GATE_MODES
                | {"srp_rcd_learned_r"}
            ):
                hloc = batch["h_local"].to(device, non_blocking=True)
                return model(
                    feats,
                    nbi,
                    nbm,
                    h_local=hloc,
                    neighbor_weight=nbw,
                )
            return model(feats, nbi, nbm, neighbor_weight=nbw)

        # XSA and Diff Transformer TransMIL comparators do not consume the
        # spatial neighbor graph. PANDA collates one native-length slide per
        # batch, so no mask stripping is needed before the square-pad/PPEG path.
        return model(feats)
    raise ValueError(f"unknown PANDA architecture for forward: {arch!r}")


# --- Eval loop ----------------------------------------------------------

def evaluate(model, loader, device, arch: str,
             srp_r_target: str = "slide_mean",
             mode: str = "baseline",
             autocast_dtype=torch.bfloat16,
             collect_gate_stats: bool = False) -> tuple:
    """Returns (metrics, per_slide rows[, gate_stats]).

    If `collect_gate_stats` is True, also returns a dict of per-example
    signed-gate summaries (β_eff mean / std / sign-bin fractions /
    h_local correlation per block per slide). Empty dict when the
    model is not in a signed-gate mode. Backwards-compatible with the
    2-tuple form by default — callers that don't care about gate
    stats keep their existing call signature.
    """
    from slide_level_srp.src.gate_signed import GateStatsAccumulator
    gate_acc = GateStatsAccumulator() if collect_gate_stats else None
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    per_slide: list[dict] = []
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=autocast_dtype):
                logits = model_forward(model, batch, device, arch,
                                       srp_r_target=srp_r_target,
                                       mode=mode)
                loss = F.cross_entropy(logits.float(), labels, reduction="sum")
            # Snapshot per-example gate stats from the just-completed
            # forward. Idempotent for non-gate modes — the accumulator
            # silently no-ops when blocks have no _last_gate_stats.
            if gate_acc is not None:
                gate_acc.update(model)
            prob = F.softmax(logits.float(), dim=-1)
            pred = prob.argmax(dim=-1)
            all_y.append(labels.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_prob.append(prob.cpu().numpy())
            loss_sum += float(loss.item())
            n += int(labels.numel())

            for i in range(labels.size(0)):
                per_slide.append({
                    "image_id": batch["image_id"][i],
                    "data_provider": batch["data_provider"][i],
                    "n_real": int(batch["n_real"][i]),
                    "y_true": int(labels[i].item()),
                    "y_pred": int(pred[i].item()),
                    "y_logits": logits[i].float().cpu().numpy().tolist(),
                    "y_probs": prob[i].cpu().numpy().tolist(),
                })

    y_true = np.concatenate(all_y); y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob, axis=0)
    metrics = compute_panda_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = loss_sum / max(1, n)
    if gate_acc is not None:
        return metrics, per_slide, gate_acc.finalize()
    return metrics, per_slide


# --- CLI ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--wandb_project", type=str, default="GatedSRP_panda")
    p.add_argument("--wandb_mode", type=str, default="disabled",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--out_dir", type=str, default="./runs_panda")

    p.add_argument("--arch", type=str, default="vit4",
                   choices=["vit4", "vit12", "transmil"])
    p.add_argument("--mode", type=str, default="baseline",
                   choices=["baseline", "nystrom_na", "xsa_all_hard", "srp_beta2",
                            "srp_fixed_beta", "srp_mlp_control",
                            "diff_transformer",
                            "abmil", "dsmil", "official_transmil",
                            "dense_mhsa", "dense_mhsa_srp",
                            "official_span_baseline", "official_span_srp",
                            "official_longnet_baseline", "official_longnet_srp",
                            "srp_signed_gated", "srp_signed_gated_pre_q",
                            "srp_signed_gated_pre_k",
                            "srp_signed_gated_learned_r",
                            "srp_rcd",
                            "srp_rcd_learned_r"],
                   help="Intervention mode. 'srp_signed_gated' is the "
                        "post-attention learned-gate variant; RCD modes are "
                        "Method 2.1 and Method 2.1+2.4; "
                        "'srp_signed_gated_pre_q' applies the same signed "
                        "gate to patch Q before QK; "
                        "'srp_signed_gated_pre_k' applies it to patch K "
                        "before QK; "
                        "'srp_signed_gated_learned_r' isolates Method 2.4 "
                        "on the original signed-gate formula; "
                        "'srp_fixed_beta' is the clear fixed-beta "
                        "alias for srp_beta2 --beta <value>; "
                        "'srp_mlp_control' is the matched-capacity no-SRP "
                        "geometry control.")
    p.add_argument("--beta", type=float, default=2.0,
                   help="SRP β (used when mode=srp_beta2 or "
                        "mode=srp_fixed_beta; ignored under srp_signed_gated "
                        "where β_eff is learned).")
    p.add_argument("--feature_root", type=str, default=None,
                   help="Directory containing PANDA H5 files.")
    p.add_argument("--feature_key", type=str, default="features/uni_v2",
                   help="H5 dataset key for patch embeddings. Defaults to "
                        "AtlasPatch UNI-v2 features.")
    p.add_argument("--in_dim", type=int, default=UNI_DIM,
                   help="Feature dimension expected at --feature_key and "
                        "used by the model input projection.")
    p.add_argument("--delta_scale", type=float, default=2.0,
                   help="Range bound for the signed gate (by design): "
                        "β_eff = δ · tanh(raw_logit) ∈ [-δ, +δ]. "
                        "δ=2 covers identity, anti-SRP, full projection, "
                        "and reflection regimes. δ=1 restricts to the "
                        "signed-projection range [-1, +1] (no reflection).")
    p.add_argument("--gate_hidden_dim", type=int, default=16,
                   help="Hidden width of the gate's token MLP. Cheap; "
                        "16 matches the reference protocol and is used in tests.")
    p.add_argument("--gate_output_init", type=str, default="zero",
                   choices=["zero", "tiny_normal", "xavier_uniform",
                            "kaiming_uniform", "orthogonal", "constant_beta"],
                   help="Output-path initialization for TokenHeadGate.")
    p.add_argument("--gate_output_init_scale", type=float, default=1.0,
                   help="Scale/std for non-zero gate output init arms.")
    p.add_argument("--gate_init_beta0", type=float, default=0.0,
                   help="Initial beta for --gate_output_init constant_beta.")
    p.add_argument("--gate_activation", type=str, default="tanh",
                   choices=["tanh", "scaled_sigmoid", "softsign",
                            "hardtanh", "atan", "sigmoid01"],
                   help="Bounded signed activation mapping raw gate logits to [-1,1].")
    p.add_argument("--gate_activation_temperature", type=float, default=1.0,
                   help="Temperature applied as raw/temperature before gate activation.")
    p.add_argument(
        "--gate_delta_mode",
        default="fixed",
        choices=["fixed", "direct_beta_softclip"],
    )
    p.add_argument("--gate_factorization", type=str, default="full",
                   choices=["full", "token_only", "head_only", "no_bias"],
                   help="Token/head factorization for the signed-gate surface. "
                        "This is consumed by the TransMIL SRP aggregator and "
                        "kept as a no-op-compatible default for existing PANDA "
                        "ViT runs.")
    p.add_argument("--gate_count_features", type=str, default="legacy",
                   choices=["legacy", "rawlog", "normlog", "none"],
                   help="Count-feature channels supplied to the signed gate.")
    p.add_argument("--gate_l2_reg", type=float, default=0.0,
                   help="Optional signed-gate containment loss. "
                        "When >0, adds gate_l2_reg * mean(beta_eff^2) "
                        "to PANDA signed-gate training objectives.")
    p.add_argument("--rcd_adapter_kind", type=str, default="lowrank",
                   choices=["lowrank", "diag"],
                   help="Method 2.1 branch map type for PANDA RCD modes.")
    p.add_argument("--rcd_rank", type=int, default=16,
                   help="Low-rank bottleneck for PANDA Method 2.1 RCD branch maps.")
    p.add_argument("--learned_r_hidden_dim", type=int, default=16,
                   help="Hidden width for PANDA Method 2.4 local context scorer.")
    p.add_argument("--srp_freeze_epochs", type=int, default=0,
                   help="Enable two-stage training by freezing PANDA "
                        "SRP/gate/RCD method parameters during the base stage. "
                        "Use the same value as --epochs for the reference "
                        "15+5 protocol.")
    p.add_argument("--stage2_epochs", type=int, default=0,
                   help="Extra two-stage epochs after the frozen-method base "
                        "stage. At the boundary, the best base checkpoint is "
                        "reloaded, trainability follows --stage2_mode, and "
                        "the optimizer is rebuilt.")
    p.add_argument("--stage2_mode", type=str, default="joint",
                   choices=["joint", "srp_only"],
                   help="Second-stage trainability: joint trains all "
                        "parameters; srp_only trains only PANDA SRP/gate/RCD "
                        "method parameters.")
    p.add_argument("--stage2_lr_mult", type=float, default=1.0,
                   help="Learning-rate multiplier applied only during "
                        "the second stage.")
    p.add_argument("--no_detach_gate_inputs", action="store_true",
                   help="Disable the default detach convention: let "
                        "gradients flow through gate diagnostic inputs "
                        "(cos_yr / y_norms / log_norm_y_mean) into y. "
                        "Default is the detached regime; this flag enables "
                        "the live-input variant.")
    p.add_argument("--srp_r_target", type=str, default="slide_mean",
                       choices=["slide_mean", "knn8"],
                       help="SRP r̂ projection target. 'slide_mean' uses the "
                            "slide-wide mean of v; 'knn8' uses the "
                            "8-neighbour grid-local mean of v from the H5 "
                            "/coords field and mirrors slide-level SRP. "
                            "Meaningful for srp_beta2/srp_fixed_beta and srp_rcd; required for "
                        "srp_signed_gated, srp_signed_gated_pre_q, "
                        "srp_signed_gated_pre_k, "
                        "srp_signed_gated_learned_r, and srp_rcd_learned_r.")
    p.add_argument("--neighbor_window", type=int, default=3,
                   help="Odd spatial neighbor window size; 3, 5, or 7 in this study.")
    p.add_argument("--neighbor_shell", type=str, default="cumulative",
                   choices=["cumulative", "ring"],
                   help="Use all offsets inside the window or only the outer ring.")
    p.add_argument("--neighbor_source", type=str, default="real",
                   choices=["real", "shuffled", "nearest_retained"],
                   help="Use spatial neighbors or same-slide shuffled neighbor identities.")
    p.add_argument("--neighbor_shuffle_seed", type=int, default=0)
    p.add_argument("--neighbor_weighting", type=str, default="uniform",
                   choices=["uniform", "gaussian", "inverse_distance"])
    p.add_argument("--neighbor_weight_sigma", type=float, default=1.0)
    p.add_argument(
        "--subsample_mode",
        default="coord_uniform",
        choices=["coord_uniform", "random_retained"],
    )
    p.add_argument("--subsample_seed", type=int, default=None)
    p.add_argument(
        "--srp_context_impl",
        default="streaming_mean",
        choices=["streaming_mean", "stacked"],
    )
    p.add_argument("--srp_correction_chunk_size", type=int, default=32768)
    p.add_argument("--pos_mode", type=str, default="none",
                   choices=["none", "coord_mlp"],
                   help="PANDA positional policy. Default none preserves vit4.")
    p.add_argument("--coord_pos_dim", type=int, default=64)
    p.add_argument("--coord_norm", type=str, default="slide_minmax",
                   choices=["slide_minmax"])
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--num_landmarks", type=int, default=64,
                   help="Nyström landmarks (transmil only).")
    p.add_argument("--official_longnet_depth", type=int, default=12)
    p.add_argument("--ln_specialization", type=str, default="shared",
                   choices=["shared", "cls_patch"],
                   help="LayerNorm specialization for the TransMIL SRP "
                        "path. Default shared preserves existing PANDA runs.")
    p.add_argument("--ln_specialization_scope", type=str, default="block",
                   choices=["block", "block_final"],
                   help="Whether class/patch-specialized LayerNorm applies "
                        "inside SRP blocks only or also to the final norm.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    # CAM17-style protocol: BS=1 + grad_accum gives effective batch = grad_accum.
    # Native-length per slide; no n_max truncation. n_max is preserved as an
    # optional anomalous-slide safety ceiling (None = no cap).
    p.add_argument("--grad_accum", type=int, default=16,
                   help="Gradient accumulation steps. Effective batch = grad_accum.")
    p.add_argument("--n_max", type=int, default=None,
                   help="Optional safety ceiling on patches/slide. None=no cap. "
                        "PANDA p100=2686 fits comfortably at depth=4.")
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--base_lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--profile_runtime",
        action="store_true",
        help="Write synchronized phase throughput and peak CUDA memory.",
    )

    p.add_argument("--split_mode", type=str, default="cv_fold",
                   choices=["cv_fold", "global_seed_holdout"],
                   help="cv_fold keeps the historical PANDA generated-fold "
                        "protocol. global_seed_holdout builds one "
                        "train/val/test split directly from --global_seed "
                        "and rejects --fold so corrected reruns cannot "
                        "accidentally reuse a fixed fold.")
    p.add_argument("--global_seed", type=int, default=None,
                   help="Single seed controlling PANDA global-seed holdout, "
                        "training RNG, and neighbor-shuffle RNG. Required "
                        "with --split_mode global_seed_holdout.")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=None,
                   help="CV fold ID used as held-out TEST. Remaining folds "
                        "split between TRAIN and inner-VAL per "
                        "--inner_val_frac. Defaults to 0 for legacy "
                        "cv_fold commands; must be omitted under "
                        "global_seed_holdout.")
    p.add_argument("--fold_seed", type=int, default=0)
    # Validation: PANDA used to use the held-out fold for
    # both best-epoch selection AND final reporting (no inner val). That
    # makes evaluation metrics optimistically biased on absolute scale
    # (paired-Δ vs same-fold baseline is unaffected). The fix carves
    # an inner-val split out of the training pool for selection; the
    # held-out fold becomes the actual untouched test.
    p.add_argument("--inner_val_frac", type=float, default=0.1,
                   help="Fraction of (train_records ∪ rest of folds) to "
                        "carve out as inner validation for best-epoch "
                        "selection. 0.0 reproduces the legacy bug where "
                        "the held-out fold serves as both selection and "
                        "test; default 0.1 enables proper 3-way split.")

    args = p.parse_args()

    # Validate the effective-batch protocol before split construction and
    # scheduler setup.  The loop still scales partial accumulation windows by
    # their actual size; this guard only rejects invalid new launches.
    if args.grad_accum <= 0:
        raise SystemExit(
            f"[parse_args] --grad_accum must be > 0, got {args.grad_accum}."
        )
    if args.in_dim <= 0:
        raise SystemExit(
            f"[parse_args] --in_dim must be > 0, got {args.in_dim}."
        )
    if not args.feature_key:
        raise SystemExit("[parse_args] --feature_key must be a non-empty H5 key.")
    if args.mode in {"srp_fixed_beta", "srp_mlp_control"} and args.arch != "transmil":
        raise SystemExit(
            f"[parse_args] --mode {args.mode} is defined for "
            "--arch transmil in the PANDA variant matrix. "
            "Use srp_beta2 for the legacy ViT fixed-beta path."
        )
    transmil_only_modes = (
        _MIL_BASELINE_MODES
        | _DENSE_MODES
        | _OFFICIAL_ARCH_MODES
        | {"nystrom_na"}
    )
    if args.mode in transmil_only_modes and args.arch != "transmil":
        raise SystemExit(
            f"[parse_args] --mode {args.mode} requires --arch transmil."
        )
    if (
        args.ln_specialization != "shared"
        and not (args.arch == "transmil" and args.mode in _TRANSMIL_NEIGHBOR_SRP_MODES)
    ):
        raise SystemExit(
            "[parse_args] --ln_specialization cls_patch is implemented for "
            "the PANDA TransMIL SRP path only. Use --mode srp_signed_gated "
            "or another TransMIL SRP mode."
        )

    # F1 check: refuse silently-buggy 0.0 unless explicitly requested.
    if args.inner_val_frac < 0.0 or args.inner_val_frac >= 1.0:
        raise SystemExit(
            f"[parse_args] --inner_val_frac must be in [0.0, 1.0); "
            f"got {args.inner_val_frac}. Use 0.0 only to reproduce the "
            f"pre-F1-fix selection-bias for back-compat with already-"
            f"completed cubes; new runs should use 0.1 (default)."
        )
    if args.n_folds <= 1:
        raise SystemExit(
            f"[parse_args] --n_folds must be > 1 for cv_fold mode, "
            f"got {args.n_folds}."
        )
    if args.split_mode == "cv_fold":
        # Preserve the historical PANDA CLI behavior for old launchers and
        # smoke tests: omitting --fold means fold 0.  New corrected reruns use
        # --split_mode global_seed_holdout, where --fold is rejected below.
        if args.fold is None:
            args.fold = 0
        if args.fold < 0 or args.fold >= args.n_folds:
            raise SystemExit(
                f"[parse_args] --fold must be in [0, {args.n_folds - 1}], "
                f"got {args.fold}."
            )
    elif args.split_mode == "global_seed_holdout":
        if args.fold is not None:
            raise SystemExit(
                "[parse_args] --split_mode global_seed_holdout must not pass "
                "--fold; PANDA train/val/test membership is generated from "
                "--global_seed."
            )
        if args.global_seed is None:
            raise SystemExit(
                "[parse_args] --split_mode global_seed_holdout requires "
                "--global_seed."
            )
        if args.seed != args.global_seed:
            raise SystemExit(
                "[parse_args] corrected PANDA reruns use one global seed: "
                f"--seed ({args.seed}) must equal --global_seed "
                f"({args.global_seed})."
            )
        if args.neighbor_shuffle_seed != args.global_seed:
            raise SystemExit(
                "[parse_args] corrected PANDA reruns use one global seed: "
                f"--neighbor_shuffle_seed ({args.neighbor_shuffle_seed}) "
                f"must equal --global_seed ({args.global_seed})."
            )
        if args.fold_seed != 0:
            raise SystemExit(
                "[parse_args] --fold_seed is ignored by PANDA "
                "global_seed_holdout and must be left at 0 so manifests "
                "cannot imply a fixed-fold protocol."
            )
    else:
        raise SystemExit(f"[parse_args] unknown --split_mode {args.split_mode!r}")
    if args.neighbor_window < 3 or args.neighbor_window % 2 != 1:
        raise SystemExit(
            f"[parse_args] --neighbor_window must be an odd integer >= 3, "
            f"got {args.neighbor_window}."
        )
    if args.subsample_seed is None:
        args.subsample_seed = (
            args.global_seed if args.global_seed is not None else args.seed
        )
    if args.srp_correction_chunk_size < 0:
        raise SystemExit(
            "[parse_args] --srp_correction_chunk_size must be non-negative."
        )
    if args.official_longnet_depth <= 0:
        raise SystemExit("[parse_args] --official_longnet_depth must be positive.")
    if args.neighbor_source == "nearest_retained":
        if args.n_max is None or args.n_max > 4096:
            raise SystemExit(
                "[parse_args] nearest_retained requires --n_max no larger than 4096."
            )
        if args.subsample_mode != "random_retained":
            raise SystemExit(
                "[parse_args] nearest_retained requires "
                "--subsample_mode random_retained."
            )
    if args.gate_l2_reg < 0.0:
        raise SystemExit(
            f"[parse_args] --gate_l2_reg must be >= 0, got {args.gate_l2_reg}."
        )
    if args.gate_l2_reg > 0.0 and args.mode not in _SIGNED_GATE_MODES:
        raise SystemExit(
            "[parse_args] --gate_l2_reg is valid only with "
            "signed-gated PANDA modes, where beta_eff exists."
        )
    if args.rcd_rank <= 0:
        raise SystemExit(
            f"[parse_args] --rcd_rank must be > 0, got {args.rcd_rank}."
        )
    if args.learned_r_hidden_dim <= 0:
        raise SystemExit(
            "[parse_args] --learned_r_hidden_dim must be > 0, got "
            f"{args.learned_r_hidden_dim}."
        )
    if args.srp_freeze_epochs < 0:
        raise SystemExit(
            f"[parse_args] --srp_freeze_epochs must be >= 0, got "
            f"{args.srp_freeze_epochs}."
        )
    if args.stage2_epochs < 0:
        raise SystemExit(
            f"[parse_args] --stage2_epochs must be >= 0, got "
            f"{args.stage2_epochs}."
        )
    if args.stage2_lr_mult <= 0.0:
        raise SystemExit(
            f"[parse_args] --stage2_lr_mult must be > 0, got "
            f"{args.stage2_lr_mult}."
        )
    if args.srp_freeze_epochs > args.epochs:
        raise SystemExit(
            f"[parse_args] --srp_freeze_epochs ({args.srp_freeze_epochs}) "
            f"cannot exceed --epochs ({args.epochs})."
        )
    if args.stage2_epochs > 0 and args.srp_freeze_epochs <= 0:
        raise SystemExit(
            "[parse_args] --stage2_epochs requires --srp_freeze_epochs > 0 "
            "so the stage-1 method-freeze point is explicit."
        )
    if (
        (args.srp_freeze_epochs > 0 or args.stage2_epochs > 0)
        and args.mode not in _METHOD_SURFACE_MODES
    ):
        raise SystemExit(
                "[parse_args] two-stage PANDA training is currently "
            "defined only for explicit method-surface modes: "
            + ", ".join(sorted(_METHOD_SURFACE_MODES))
        )
    # --- Argument cross-validation ----------------------------------
    # The signed-gate path is hard-pinned to the knn8 r̂ family at the
    # PandaAttention layer (a slide-wide r̂ would degenerate under
    # per-token gating; see src/vit_panda.py). The CLI default for
        # --srp_r_target is "slide_mean" by default, so a user
    # running `--mode srp_signed_gated` without explicitly passing
    # `--srp_r_target knn8` would fail at model construction with a
    # cryptic assertion. Catch it here with a clear message so launch
    # scripts surface the mismatch immediately.
    if (
        args.mode in _SIGNED_GATE_MODES
        or args.mode in _LEARNED_R_SIGNED_GATE_MODES
        or args.mode == "srp_rcd_learned_r"
    ) and args.srp_r_target != "knn8":
        raise SystemExit(
            f"[parse_args] --mode {args.mode} requires "
            f"--srp_r_target knn8 (got '{args.srp_r_target}'). The "
            f"token-local learned method path is knn8-only by construction. "
            f"If you want a slide-mean fixed-r run, use `--mode srp_fixed_beta` "
            f"(or legacy `--mode srp_beta2`) or `--mode srp_rcd` with "
            f"`--srp_r_target slide_mean`."
        )

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_label = (
        f"fold={args.fold}"
        if args.split_mode == "cv_fold"
        else f"global_seed={args.global_seed}"
    )
    print(f"[{args.run_name}] device={device} arch={args.arch} mode={args.mode} "
          f"seed={args.seed} split_mode={args.split_mode} {split_label}")

    # --- Data + splits -------------------------------------------------
    # PANDA supports two protocols:
    #   1. cv_fold: historical provider/ISUP stratified generated folds, with
    #      an inner validation split carved from the non-test folds.
    #   2. global_seed_holdout: corrected global-seed protocol matching the slide
    #      global-seed holdout idea; no fold id enters the command or split.
    feature_root = Path(args.feature_root) if args.feature_root else PANDA_H5_DIR
    print(
        f"[{args.run_name}] features root={feature_root} "
        f"key={args.feature_key} in_dim={args.in_dim}"
    )
    records = enumerate_slides(
        h5_dir=feature_root,
        feature_key=args.feature_key,
        feature_dim=args.in_dim,
    )
    split_metadata: dict = {
        "dataset": "panda",
        "split_mode": args.split_mode,
        "seed": int(args.seed),
        "neighbor_shuffle_seed": int(args.neighbor_shuffle_seed),
        "feature_root": str(feature_root),
        "feature_key": args.feature_key,
        "in_dim": int(args.in_dim),
    }

    if args.split_mode == "global_seed_holdout":
        gs_split = build_panda_global_seed_splits(
            records,
            global_seed=int(args.global_seed),
            test_frac=0.20,
            val_frac=0.10,
        )
        train_idx = list(gs_split.train_idx)
        val_idx = list(gs_split.val_idx)
        test_idx = list(gs_split.test_idx)
        train_loader, val_loader, test_loader = build_panda_loaders(
            records, train_idx, val_idx, test_idx,
            num_workers=args.num_workers,
            safety_cap=args.n_max,
            neighbor_radius=(args.neighbor_window - 1) // 2,
            neighbor_shell=args.neighbor_shell,
            neighbor_source=args.neighbor_source,
            neighbor_shuffle_seed=args.neighbor_shuffle_seed,
            neighbor_weighting=args.neighbor_weighting,
            neighbor_weight_sigma=args.neighbor_weight_sigma,
            feature_key=args.feature_key,
            feature_dim=args.in_dim,
            subsample_mode=args.subsample_mode,
            subsample_seed=args.subsample_seed,
        )
        split_metadata.update({
            "global_seed": int(args.global_seed),
            "fold": None,
            "fold_seed": 0,
            "unit_key": "image_id",
            "stratification": "data_provider|isup_grade",
            "split_indices": {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            },
            "counts": {
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
            "stratum_counts": gs_split.stratum_counts,
        })
        print(f"[{args.run_name}] global-seed 3-way split: "
              f"train_slides={len(train_loader.dataset)} "
              f"val_slides={len(val_loader.dataset)} "
              f"test_slides={len(test_loader.dataset)} "
              f"(global_seed={args.global_seed})")
    else:
        # Validation: 3-way split. The held-out outer fold is
        # the untouched TEST set; an inner-val split is carved out of the
        # remaining 4 folds for best-epoch selection. Previously, the held-out
        # fold did both jobs, optimistically biasing the reported metric.
        folds = build_panda_folds(records, n_folds=args.n_folds, fold_seed=args.fold_seed)
        test_idx = folds[args.fold]
        pool_idx = [i for f, fold_list in enumerate(folds) if f != args.fold for i in fold_list]

        if args.inner_val_frac > 0.0:
            # Validation: stratify the inner-val split
            # by (data_provider, isup_grade) so model-selection is not
            # confounded by provider/site or rare-grade drift.
            inner_rng = np.random.default_rng(seed=args.fold_seed * 1000 + args.fold + 1)
            # Bucket pool indices by (provider, ISUP).
            from collections import defaultdict
            strata = defaultdict(list)
            for idx in pool_idx:
                r = records[idx]
                strata[(r.data_provider, r.isup_grade)].append(idx)
            val_idx, train_idx = [], []
            for stratum_key, stratum_idxs in strata.items():
                shuf = list(stratum_idxs)
                inner_rng.shuffle(shuf)
                n_val_s = max(1, int(round(len(shuf) * args.inner_val_frac)))
                # Edge case: if stratum has <= 1 slide, give it to train
                # because we cannot simultaneously train and validate on it.
                if len(shuf) <= 1:
                    train_idx.extend(shuf)
                    continue
                val_idx.extend(shuf[:n_val_s])
                train_idx.extend(shuf[n_val_s:])
            # Shuffle the concatenated train/val ids so DataLoader's own
            # shuffle=True starts from a seed-specific but stratified order.
            inner_rng.shuffle(train_idx)
            inner_rng.shuffle(val_idx)
            train_loader, val_loader, test_loader = build_panda_loaders(
                records, train_idx, val_idx, test_idx,
                num_workers=args.num_workers,
                safety_cap=args.n_max,
                neighbor_radius=(args.neighbor_window - 1) // 2,
                neighbor_shell=args.neighbor_shell,
                neighbor_source=args.neighbor_source,
                neighbor_shuffle_seed=args.neighbor_shuffle_seed,
                neighbor_weighting=args.neighbor_weighting,
                neighbor_weight_sigma=args.neighbor_weight_sigma,
                feature_key=args.feature_key,
                feature_dim=args.in_dim,
                subsample_mode=args.subsample_mode,
                subsample_seed=args.subsample_seed,
            )
            print(f"[{args.run_name}] 3-way split (F1 fix): "
                  f"train_slides={len(train_loader.dataset)} "
                  f"inner_val_slides={len(val_loader.dataset)} "
                  f"test_slides={len(test_loader.dataset)} "
                  f"(inner_val_frac={args.inner_val_frac})")
        else:
            # Legacy bug-compatible path. val_loader == test_loader.
            train_idx = list(pool_idx)
            val_idx = list(test_idx)
            train_loader, val_loader = build_panda_loaders(
                records, train_idx, val_idx,
                num_workers=args.num_workers,
                safety_cap=args.n_max,
                neighbor_radius=(args.neighbor_window - 1) // 2,
                neighbor_shell=args.neighbor_shell,
                neighbor_source=args.neighbor_source,
                neighbor_shuffle_seed=args.neighbor_shuffle_seed,
                neighbor_weighting=args.neighbor_weighting,
                neighbor_weight_sigma=args.neighbor_weight_sigma,
                feature_key=args.feature_key,
                feature_dim=args.in_dim,
                subsample_mode=args.subsample_mode,
                subsample_seed=args.subsample_seed,
            )
            test_loader = val_loader  # explicit alias — both point at same data
            print(f"[{args.run_name}] LEGACY split (selection==test, F1 bug): "
                  f"train_slides={len(train_loader.dataset)} "
                  f"val=test_slides={len(val_loader.dataset)}")

        split_metadata.update({
            "global_seed": None,
            "fold": int(args.fold),
            "fold_seed": int(args.fold_seed),
            "n_folds": int(args.n_folds),
            "unit_key": "image_id",
            "stratification": "data_provider|isup_grade",
            "split_indices": {
                "train": list(train_idx),
                "val": list(val_idx),
                "test": list(test_idx),
            },
            "counts": {
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
        })

    # --- Model + optimizer --------------------------------------------
    model = build_model(args).to(device)
    stage_protocol_enabled = args.srp_freeze_epochs > 0 or args.stage2_epochs > 0
    if stage_protocol_enabled:
        counts = apply_trainability_mode(model, args, "freeze_method")
        print(
            f"[{args.run_name}] stage1 trainability: "
            f"{_stage_trainability_mode_name('freeze_method')} counts={counts}"
        )

    method_names, n_method, n_trainable, n_params = method_parameter_summary(model, args)
    name_paudit = (
        "+".join(method_names[:5])
        + (f" + {len(method_names) - 5} more" if len(method_names) > 5 else "")
    )
    print(
        f"[{args.run_name}] params_total={n_params:,} trainable={n_trainable:,} "
        f"method_params={n_method} ({name_paudit if method_names else 'none'})"
    )

    optimizer = build_adamw_optimizer(model, args)

    # One optimizer step per `grad_accum` slides (BS=1 + accumulation).
        # Mirrors the slide-level and slide-level SRP optimization protocol.
    n_train_slides = len(train_loader.dataset)
    steps_per_epoch = (n_train_slides + args.grad_accum - 1) // args.grad_accum
    base_total_steps = args.epochs * steps_per_epoch
    total_train_epochs = args.epochs + args.stage2_epochs
    total_steps = total_train_epochs * steps_per_epoch
    warmup_steps = max(1, int(base_total_steps * args.warmup_ratio))
    stage_total_steps = base_total_steps
    stage_warmup_steps = warmup_steps
    stage_step = 0
    stage_name = "stage1_frozen" if stage_protocol_enabled else "main"
    print(f"[{args.run_name}] grad_accum={args.grad_accum}  "
          f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  "
          f"base_steps={base_total_steps}  "
          f"stage2_steps={args.stage2_epochs * steps_per_epoch}  "
          f"warmup={warmup_steps}")

    # --- W&B -----------------------------------------------------------
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = RuntimeProfiler(enabled=args.profile_runtime, device=device)
    wandb.init(
        project=args.wandb_project, name=args.run_name, mode=args.wandb_mode,
        dir=str(out_dir),
        config={
            **vars(args),
            "n_params_total": n_params, "n_params_trainable": n_trainable,
            "n_ab_params": n_method,
            "total_steps": total_steps, "warmup_steps": warmup_steps,
            "method_param_names": method_names,
            "n_train_slides": len(train_loader.dataset),
            "n_val_slides": len(val_loader.dataset),
        },
        tags=[f"arch-{args.arch}", f"mode-{args.mode}",
              f"split-{args.split_mode}", split_label.replace("=", "-"),
              f"seed-{args.seed}",
              f"dp-{args.drop_path:g}"],
    )

    best_val_kappa = -1.0
    best_ckpt = out_dir / "best.pt"
    stage1_best_ckpt = out_dir / "stage1_best.pt"
    stage1_best_val_kappa = float("nan")
    stage2_best_val_kappa = float("nan")
    global_step = 0
    gate_l2_missing_steps = 0

    # --- Training loop (BS=1 + grad_accum) ------------------------------
    for epoch in range(total_train_epochs):
            # Two-stage training starts the second stage after the ordinary
            # `--epochs` budget. We
        # restart from the best frozen-method checkpoint, not the final epoch,
        # because the final frozen checkpoint may already be past the best
        # model-selection point.
        if stage_protocol_enabled and args.stage2_epochs > 0 and epoch == args.epochs:
            stage1_best_val_kappa = float(best_val_kappa)
            if not best_ckpt.exists():
                raise RuntimeError(
                    f"[{args.run_name}] stage1 completed without {best_ckpt}; "
                    "cannot start PANDA two-stage fine-tuning."
                )
            shutil.copy2(best_ckpt, stage1_best_ckpt)
            ckpt = torch.load(stage1_best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            print(
                f"[{args.run_name}] stage2 reload: {stage1_best_ckpt} "
                f"val_kappa={stage1_best_val_kappa:.4f}"
            )
            stage_mode = "method_only" if args.stage2_mode == "srp_only" else "all"
            counts = apply_trainability_mode(model, args, stage_mode)
            optimizer = build_adamw_optimizer(model, args)
            best_val_kappa = -1.0
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
        loss_sum = 0.0
        objective_sum = 0.0
        gate_l2_sum = 0.0
        n_train = 0
        ep_y, ep_pred, ep_prob = [], [], []

        optimizer.zero_grad(set_to_none=True)
        slides_in_accum = 0

        pbar = tqdm(train_loader,
                    desc=f"[{args.run_name}] {stage_name} ep {epoch+1}/{total_train_epochs}",
                    leave=False, mininterval=1.0)
        runtime.start("train")
        for bi, batch in enumerate(pbar):
            labels = batch["label"].to(device, non_blocking=True)

            # Per-optimizer-step LR update: recompute on step boundaries
            # only (i.e., when slides_in_accum == 0).
            stage_base_lr = args.base_lr * (
                args.stage2_lr_mult if stage_name.startswith("stage2") else 1.0
            )
            lr_now = cosine_warmup_lr(
                stage_step, stage_warmup_steps, stage_total_steps,
                stage_base_lr, min_lr=0.0,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            # Validation: divide by the **actual** size of the
            # current accumulation window, not by `args.grad_accum`, so the
            # final partial window of an epoch is not silently underweighted.
            # See slide_level_srp/train.py for the matching fix.
            window_start = (bi // args.grad_accum) * args.grad_accum
            window_size  = min(args.grad_accum, len(train_loader) - window_start)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model_forward(model, batch, device, args.arch,
                                       srp_r_target=args.srp_r_target,
                                       mode=args.mode)
                if args.mode == "dsmil":
                    ce_loss = dsmil_dual_stream_cross_entropy(
                        model, logits.float(), labels,
                    )
                else:
                    ce_loss = F.cross_entropy(logits.float(), labels)
                # gate-containment mirrors slide_level_srp/train.py: penalize the
                # realized signed-gate beta surface, not raw logits.  A missing
                # cache means the command is misconfigured or the gate path did
                # not execute; count it for artifact audit instead of hiding it.
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

            with torch.no_grad():
                prob = F.softmax(logits.float(), dim=-1)
            ep_y.append(labels.cpu().numpy())
            ep_pred.append(prob.argmax(dim=-1).cpu().numpy())
            ep_prob.append(prob.cpu().numpy())
            loss_sum += float(ce_loss.detach().item()) * labels.numel()
            objective_sum += float(objective.detach().item()) * labels.numel()
            gate_l2_sum += float(gate_l2.detach().item()) * labels.numel()
            n_train += int(labels.numel())
            slides_in_accum += 1

            is_last = (bi == len(train_loader) - 1)
            if slides_in_accum >= args.grad_accum or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                slides_in_accum = 0

                if global_step % 50 == 0:
                    # Validation: undo per-window scaling
                    # consistently (matches the F4 fix above). With
                    # `args.grad_accum`, the final partial window's W&B
                    # step loss was inflated by grad_accum / window_size.
                    payload = {
                        "train/loss_step": float(ce_loss.detach().item()),
                        "train/objective_step": float(objective.detach().item()),
                        "train/lr": lr_now,
                    }
                    if args.gate_l2_reg > 0.0:
                        payload["train/gate_l2_step"] = float(gate_l2.detach().item())
                        payload["train/gate_l2_blocks"] = gate_l2_blocks
                    wandb.log(payload, step=global_step)
                # Validation: per-step gate trajectory
                # for signed-gated runs. Cadence (every 25 steps) matches
                # the CAM17/CAM16 trainer for cross-task comparability.
                # No-op when the gate is not active.
                if args.mode in (_SIGNED_GATE_MODES | _LEARNED_R_SIGNED_GATE_MODES) and global_step % 25 == 0:
                    from slide_level_srp.src.gate_signed import signed_gate_step_summary
                    gate_payload = signed_gate_step_summary(model)
                    if gate_payload:
                        wandb.log(gate_payload, step=global_step)
                stage_step += 1
                global_step += 1
        runtime.stop(n_slides=len(train_loader.dataset))

        train_y = np.concatenate(ep_y)
        train_pred = np.concatenate(ep_pred)
        train_prob = np.concatenate(ep_prob)
        train_metrics = compute_panda_metrics(train_y, train_pred, train_prob)
        train_metrics["loss"] = loss_sum / max(1, n_train)
        train_metrics["objective"] = objective_sum / max(1, n_train)
        if args.gate_l2_reg > 0.0:
            train_metrics["gate_l2"] = gate_l2_sum / max(1, n_train)
            train_metrics["gate_l2_penalty"] = args.gate_l2_reg * train_metrics["gate_l2"]

        # --- Val + best-checkpoint -----------------------------------
        runtime.start("validation")
        val_metrics, _ = evaluate(model, val_loader, device, args.arch,
                                  srp_r_target=args.srp_r_target,
                                  mode=args.mode)
        runtime.stop(n_slides=len(val_loader.dataset))

        wandb.log({f"train/{k}": v for k, v in train_metrics.items()}, step=global_step)
        wandb.log({f"val/{k}":   v for k, v in val_metrics.items()},   step=global_step)
        # Gate-distribution diagnostics: only fires when the gate is
        # active. The values come from whichever batch was last
        # forwarded inside evaluate(); for eval-set-wide aggregates,
        # the final-eval npz (test_gate_stats) carries the proper
        # accumulator output. Validation: prefix renamed
        # to `gate_last_batch/` so the W&B dashboard is self-describing
        # (these are last-batch snapshots, not validation aggregates).
        gate_log = gate_diagnostics(model, prefix="gate_last_batch")
        if gate_log:
            wandb.log(gate_log, step=global_step)
        dt = time.time() - t0
        print(
            f"[{args.run_name}] ep{epoch+1}: "
            f"train[loss={train_metrics['loss']:.4f} κ={train_metrics['kappa_quad']:.4f} "
            f"f1={train_metrics['macro_f1']:.4f}] "
            f"val[loss={val_metrics['loss']:.4f} κ={val_metrics['kappa_quad']:.4f} "
            f"f1={val_metrics['macro_f1']:.4f} acc={val_metrics['acc']:.4f}] "
            f"| {dt:.1f}s"
        )

        if val_metrics["kappa_quad"] > best_val_kappa:
            best_val_kappa = val_metrics["kappa_quad"]
            torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                        "args": vars(args), "val_metrics": val_metrics,
                        "stage": stage_name}, best_ckpt)

    if stage_protocol_enabled and args.stage2_epochs == 0:
        stage1_best_val_kappa = float(best_val_kappa)
    if stage_protocol_enabled and args.stage2_epochs > 0:
        stage2_best_val_kappa = float(best_val_kappa)

    # --- Final test on the held-out outer fold ------------------------
    # Validation: report on `test_loader` (the untouched
    # held-out fold), not on `val_loader` (which under the proper 3-way
    # split is the inner-val pool used only for best-epoch selection).
    # Under legacy --inner_val_frac=0, `test_loader is val_loader` and
    # behavior matches the previous code.
    print(f"[{args.run_name}] reloading best (val_kappa={best_val_kappa:.4f})")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    runtime.start("test")
    final_metrics, per_slide, final_gate_stats = evaluate(
        model, test_loader, device, args.arch,
        srp_r_target=args.srp_r_target,
        mode=args.mode,
        collect_gate_stats=True,
    )
    runtime.stop(n_slides=len(test_loader.dataset))
    runtime.write(out_dir / "runtime_profile.json")
    print(
        f"[{args.run_name}] FINAL test: κ_quad={final_metrics['kappa_quad']:.4f} "
        f"κ_lin={final_metrics['kappa_lin']:.4f} "
        f"acc={final_metrics['acc']:.4f} bal_acc={final_metrics['balanced_acc']:.4f} "
        f"macro_F1={final_metrics['macro_f1']:.4f} "
        f"binary_F1={final_metrics['binary_f1']:.4f} "
        f"binary_AUC={final_metrics['binary_auc']:.4f}"
    )
    wandb.log({f"test/{k}": v for k, v in final_metrics.items()}, step=global_step)

    # --- Per-slide CSV --------------------------------------------------
    # Keep the historical column name `fold` for downstream readers, but under
    # global_seed_holdout store an explicit split label instead of pretending a
    # generated fold id was used.
    prediction_split_value = (
        str(args.fold)
        if args.split_mode == "cv_fold"
        else f"global_seed_{args.global_seed}"
    )
    csv_path = out_dir / "predictions.csv"
    base_cols = ["image_id", "data_provider", "fold", "n_real",
                 "arch", "mode", "y_true", "y_pred"]
    logit_cols = [f"y_pred_logit_{k}" for k in range(N_ISUP)]
    prob_cols = [f"y_pred_prob_{k}" for k in range(N_ISUP)]
    fields = base_cols + logit_cols + prob_cols
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_slide:
            row = {
                "image_id": r["image_id"],
                "data_provider": r["data_provider"],
                "fold": prediction_split_value,
                "n_real": r["n_real"],
                "arch": args.arch,
                "mode": args.mode,
                "y_true": r["y_true"],
                "y_pred": r["y_pred"],
            }
            for k in range(N_ISUP):
                row[f"y_pred_logit_{k}"] = f"{r['y_logits'][k]:.6g}"
                row[f"y_pred_prob_{k}"] = f"{r['y_probs'][k]:.6g}"
            w.writerow(row)
    print(f"[{args.run_name}] wrote {csv_path} ({len(per_slide)} slides)")

    # --- Artifact npz ---------------------------------------------------
    # Validation: `per_slide` now comes from the held-out
    # outer fold (test_loader), not the inner-val pool. Save canonically
    # as `test_y` / `test_logits` etc. The legacy `val_*` aliases are
    # retained so older readouts continue to work; under
    # `--inner_val_frac > 0` they point at the same held-out-fold data
    # because we evaluate test_loader once at the end. (Older scripts
    # treating these as "val" are still semantically off, but bit-
    # compatible with prior PANDA artifacts.)
    test_y = np.array([r["y_true"] for r in per_slide], dtype=np.int64)
    test_logits = np.stack([np.asarray(r["y_logits"]) for r in per_slide]).astype(np.float32)
    test_n_real = np.array([r["n_real"] for r in per_slide], dtype=np.int64)
    test_provider = np.array([r["data_provider"] for r in per_slide], dtype=object)

    npz = {
        "test_metrics": json.dumps(final_metrics),
        "best_val_kappa_quad": best_val_kappa,
        "stage1_best_val_kappa_quad": stage1_best_val_kappa,
        "stage2_best_val_kappa_quad": stage2_best_val_kappa,
        "arch": args.arch, "mode": args.mode,
        "split_mode": args.split_mode,
        "feature_root": str(feature_root),
        "feature_key": args.feature_key,
        "in_dim": int(args.in_dim),
        # Store -1 for fold/global_seed when not applicable so downstream npz
        # readers do not need object arrays for None.
        "fold": int(args.fold) if args.fold is not None else -1,
        "fold_seed": int(args.fold_seed),
        "global_seed": int(args.global_seed) if args.global_seed is not None else -1,
        "seed": args.seed,
        "inner_val_frac": float(args.inner_val_frac),
        "split_metadata": json.dumps(split_metadata),
        "gate_l2_reg": float(args.gate_l2_reg),
        "gate_l2_missing_steps": int(gate_l2_missing_steps),
        "srp_freeze_epochs": int(args.srp_freeze_epochs),
        "stage2_epochs": int(args.stage2_epochs),
        "stage2_mode": args.stage2_mode,
        "stage2_lr_mult": float(args.stage2_lr_mult),
        # Canonical names: held-out outer fold = the actual test set.
        "test_y": test_y, "test_logits": test_logits,
        "test_n_real": test_n_real, "test_provider": test_provider,
    }
    # Persist canonical split indices for every modern 3-way protocol.  Keep
    # the old inner_* aliases for cv_fold so legacy analysis can still audit
    # the validation inner validation fix.
    npz["train_idx"] = np.array(train_idx, dtype=np.int64)
    npz["val_idx"]   = np.array(val_idx, dtype=np.int64)
    npz["test_idx"]  = np.array(test_idx, dtype=np.int64)
    if args.split_mode == "cv_fold" and args.inner_val_frac > 0.0:
        npz["inner_train_idx"] = np.array(train_idx, dtype=np.int64)
        npz["inner_val_idx"]   = np.array(val_idx, dtype=np.int64)
    # F8 fix: legacy `val_*` aliases are misleading under the proper
    # 3-way split (`val_*` traditionally means inner-val, but pre-F4-fix
    # PANDA pre-existing readouts treat `val_*` as the held-out
    # outer fold). Keep aliases ONLY in the legacy --inner_val_frac=0
    # regime — there `test_loader is val_loader` so they really are
    # interchangeable. Under proper split, drop the aliases; new
    # readouts must use `test_*`.
    if args.inner_val_frac == 0.0:
        npz["val_y"] = test_y
        npz["val_logits"] = test_logits
        npz["val_n_real"] = test_n_real
        npz["val_provider"] = test_provider
    # Preserve XSA/SRP scalars where applicable.
    for n, p in model.named_parameters():
        if "alpha_cls" in n or "alpha_patch" in n or "beta_patch" in n:
            npz[n.replace(".", "_")] = p.detach().cpu().numpy()
    # Per-slide signed-gate stats. Empty dict for non-signed-gated runs;
    # we prefix with "gate_" to keep the npz namespace organised so a
    # downstream reader can `[k for k in d if k.startswith("gate_")]`.
    # Per `slide_level_srp.src.gate_signed.GateStatsAccumulator.finalize`,
    # keys are
    # "block{i}_{stat}" — under the prefix they become
    # "gate_block{i}_{stat}".
    for k, v in final_gate_stats.items():
        npz[f"gate_{k}"] = v
    np.savez(out_dir / "test_artifacts.npz", **npz)
    wandb.finish()
    print(f"[{args.run_name}] done")


if __name__ == "__main__":
    main()
