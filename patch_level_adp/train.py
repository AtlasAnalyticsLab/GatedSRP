"""Patch-level ADP training for the attention-operator comparison.

ADP is a multi-label raw-RGB patch task. The comparison uses a ViT-S/16 model
over 272 x 272 patches, producing a 17 x 17 patch-token grid plus a CLS token.
The baseline arm uses standard MHSA; the Gated SRP arm uses the same model
dimensions and swaps in the patch-level signed SRP backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

import wandb

from src.data_adp import (
    build_global_seed_split_indices,
    build_loaders,
    build_loaders_from_indices,
    get_label_columns,
    label_level,
)
from src.diagnostics import (
    StatsAccumulator,
    autocast_ctx,
    extract_alpha_values,
    extract_batch_stats,
    extract_beta_values,
    set_capture_mode,
)
from src.diff_vit import DiffTransformerViT
from src.vit import ViT


def set_seed(seed: int) -> None:
    """Match train.py's seeding policy: Python+NumPy+PyTorch (CUDA included)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int,
                     base_lr: float, min_lr: float = 0.0) -> float:
    """Linear warmup → cosine decay. Identical to train.py."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# --- Multi-label metrics -------------------------------------------------
def compute_multilabel_map(y_true: np.ndarray, y_score: np.ndarray,
                           label_cols: list[str]) -> dict:
    """
    Multi-label mean Average Precision and per-level breakdowns.

    y_true: (N, n_labels) float in {0, 1}
    y_score: (N, n_labels) float, raw logits or sigmoid probabilities
             — sklearn's average_precision_score is invariant to monotone
             transforms on the score, so logits work directly.

    Returns {"map_macro": ..., "ap_per_label": dict, "map_L1": ..., ...}.
    For labels with zero positives in y_true the per-label AP is set to
    NaN and excluded from any mean. (Should not happen post-pruning, but
    defensive against accidental empty val/test cells.)
    """
    n_labels = y_true.shape[1]
    aps = np.full(n_labels, np.nan, dtype=np.float64)
    for j in range(n_labels):
        if y_true[:, j].sum() == 0:
            continue   # AP undefined when no positives
        aps[j] = average_precision_score(y_true[:, j], y_score[:, j])

    valid = ~np.isnan(aps)
    map_macro = float(np.nanmean(aps)) if valid.any() else float("nan")

    out: dict = {"map_macro": map_macro,
                 "ap_per_label": {label_cols[j]: float(aps[j]) for j in range(n_labels)}}
    # Per-level mAP: average APs over labels grouped by hierarchy depth.
    for lvl in (1, 2, 3):
        idxs = [j for j, L in enumerate(label_cols) if label_level(L) == lvl]
        if not idxs:
            out[f"map_L{lvl}"] = float("nan"); continue
        sel = aps[idxs]
        out[f"map_L{lvl}"] = float(np.nanmean(sel)) if (~np.isnan(sel)).any() else float("nan")
    # This is evaluation-split support-weighted mAP, not train-support
    # weighted. The per-label support comes from whichever split is being
    # scored, which keeps the summary tied to the actual validation/test pool.
    pos_counts = y_true.sum(axis=0)
    if pos_counts.sum() > 0:
        # Drop NaN APs to avoid contaminating the weighted mean.
        good = valid & (pos_counts > 0)
        out["map_weighted"] = float(np.average(aps[good], weights=pos_counts[good]))
    else:
        out["map_weighted"] = float("nan")
    # ADP is multilabel, so use the standard sigmoid>=0.5 decision rule
    # (equivalently raw logit >= 0) and report macro-over-label summaries.
    y_true_i = y_true.astype(np.int64)
    y_pred_i = (y_score >= 0.0).astype(np.int64)
    macro_f1 = float(
        f1_score(y_true_i, y_pred_i, average="macro", zero_division=0)
    )
    label_acc = (y_pred_i == y_true_i).mean(axis=0)
    acc = float(np.nanmean(label_acc)) if label_acc.size else float("nan")
    aucs = np.full(n_labels, np.nan, dtype=np.float64)
    for j in range(n_labels):
        # ROC AUC is undefined when the evaluation split has one class only.
        # This can happen under global-seed splits for rare ADP labels; those
        # labels are excluded from the macro mean rather than forcing 0.5.
        if np.unique(y_true_i[:, j]).size < 2:
            continue
        aucs[j] = roc_auc_score(y_true_i[:, j], y_score[:, j])
    macro_auc = float(np.nanmean(aucs)) if (~np.isnan(aucs)).any() else float("nan")
    out["f1"] = macro_f1
    out["macro_f1"] = macro_f1
    out["acc"] = acc
    out["auc"] = macro_auc
    out["macro_auc"] = macro_auc
    out["auc_per_label"] = {
        label_cols[j]: float(aucs[j]) for j in range(n_labels)
    }
    return out


# --- W&B helpers --------------------------------------------------------
def log_alpha_or_beta_scalars(snap: dict, prefix: str, step: int) -> None:
    """
    Emit one W&B scalar per (role, layer, head) for whichever ab params
    are present (alpha for XSA path, beta for SRP path). Mirror of
    train.py::log_alpha_scalars but agnostic to dict keys.
    """
    payload = {}
    for role, arr in snap.items():
        arr_np = arr.detach().cpu().numpy()  # (D, H)
        depth, n_heads = arr_np.shape
        for li in range(depth):
            for hi in range(n_heads):
                payload[f"{prefix}/{role}_L{li}_H{hi}"] = float(arr_np[li, hi])
        payload[f"{prefix}/{role}_mean"] = float(arr_np.mean())
    wandb.log(payload, step=step)


def log_val_diagnostics(stats: dict, step: int) -> None:
    """Push the same per-(layer, head) diagnostics that train.py logs."""
    payload = {}
    cls_self = stats["cls_self_attn"].cpu().numpy()
    depth, n_heads = cls_self.shape
    for li in range(depth):
        for hi in range(n_heads):
            payload[f"val_diag/cls_self_attn_L{li}_H{hi}"] = float(cls_self[li, hi])
    payload["val_diag/cls_self_attn_mean"] = float(cls_self.mean())

    # cos(y, v) and friends — same canonical naming as train.py.
    for role in ("cls", "patch"):
        for stage in ("pre", "post"):
            arr = stats[f"cos_yv_{role}_{stage}"].cpu().numpy()
            for li in range(depth):
                for hi in range(n_heads):
                    payload[f"val_diag/cos_yv_{role}_{stage}_L{li}_H{hi}"] = float(arr[li, hi])
            payload[f"val_diag/cos_yv_{role}_{stage}_mean"] = float(arr.mean())
    for role in ("cls", "patch"):
        arr = stats[f"z_over_y_{role}"].cpu().numpy()
        for li in range(depth):
            for hi in range(n_heads):
                payload[f"val_diag/z_over_y_{role}_L{li}_H{hi}"] = float(arr[li, hi])
        payload[f"val_diag/z_over_y_{role}_mean"] = float(arr.mean())
    wandb.log(payload, step=step)


def supports_xsa_srp_diagnostics(model) -> bool:
    """
    Return True only for ADP backends that expose the XSA/SRP diagnostic hooks.

    Diff Transformer is a separate comparator with paired Q/K attention maps,
    learned differential lambda parameters, and no alpha/beta projection
    buffers. Treating it as "non-SRP == XSA" makes evaluation crash and would
    also imply diagnostics that the method does not define. Metrics, logits, and
    checkpoints remain comparable; only method-specific attention diagnostics
    are disabled for unsupported blocks.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return False
    for blk in blocks:
        attn = getattr(blk, "attn", None)
        # extract_batch_stats needs attention.last_stats and block-level
        # last_cls_states populated by set_capture_mode after the forward pass.
        # Standard/XSA and SRP blocks provide these fields; DiffBlock does not.
        if attn is None or not hasattr(attn, "last_stats"):
            return False
        if not hasattr(blk, "last_cls_states"):
            return False
    return True


# --- Eval loop -----------------------------------------------------------
def evaluate(
    model, loader, device, n_labels: int,
    *, capture_stats: bool, max_stats_batches: int | None,
    autocast_dtype=torch.bfloat16,
    collect_gate_stats: bool = False,
):
    """
    Multi-label eval. Returns
      (metrics_dict, optional cls_stats, y_true, y_score, gate_stats)

    gate_stats is an empty dict for non-signed-gated runs; under the
    signed-gate mode it carries per-example summaries of beta_eff
    (see gate_signed.GateStatsAccumulator). Backwards-compatible
    callers can simply discard the trailing element.
    """
    model.eval()
    if capture_stats and not supports_xsa_srp_diagnostics(model):
        # Keep evaluation robust for comparator backends. This guard prevents an
        # incompatible diagnostics request from mutating the method definition or
        # failing after metrics have already been computed.
        capture_stats = False
    if capture_stats:
        set_capture_mode(model, True)
    if collect_gate_stats:
        from slide_level_srp.src.gate_signed import GateStatsAccumulator
        gate_acc = GateStatsAccumulator()
    else:
        gate_acc = None

    all_y, all_p = [], []
    loss_sum = 0.0
    n_obs = 0
    stats_acc = StatsAccumulator() if capture_stats else None
    capture_active = capture_stats
    stats_done = 0

    with torch.no_grad():
        for bi, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_ctx(device, autocast_dtype):
                logits = model(imgs)
                # `reduction="sum"` → loss_sum is sum over (B × n_labels)
                # contributions, so per-example mean-loss = loss_sum / (n_obs).
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels,
                                                          reduction="sum")
            if gate_acc is not None:
                gate_acc.update(model)
            all_y.append(labels.cpu().numpy())
            all_p.append(logits.float().cpu().numpy())
            loss_sum += float(loss.item())
            # n_obs counts (B*n_labels) so that the reported mean loss is the
            # standard per-element BCE — comparable across runs even if
            # n_labels changes.
            n_obs += labels.numel()
            if capture_active:
                stats_acc.update(extract_batch_stats(model))
                stats_done += 1
                if max_stats_batches is not None and stats_done >= max_stats_batches:
                    set_capture_mode(model, False)
                    capture_active = False
    if capture_active:
        set_capture_mode(model, False)

    y_true = np.concatenate(all_y, axis=0)
    y_score = np.concatenate(all_p, axis=0)
    metrics = compute_multilabel_map(y_true, y_score, label_cols=loader.dataset.label_cols)
    metrics["loss"] = loss_sum / max(1, n_obs)

    cls_stats = stats_acc.result() if (capture_stats and stats_done > 0) else None
    gate_stats = gate_acc.finalize() if gate_acc is not None else {}
    return metrics, cls_stats, y_true, y_score, gate_stats


# --- CLI ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--wandb_project", type=str, default="GatedSRP_adp")
    p.add_argument("--wandb_mode", type=str, default="disabled",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--out_dir", type=str, default="./runs_adp")
    p.add_argument("--split_mode", type=str, default="official",
                   choices=["official", "global_seed_holdout"],
                   help="ADP split source. 'official' uses Release1's "
                        "train/valid/test npy files. 'global_seed_holdout' "
                        "builds a new patch-level train/val/test split from "
                        "one global seed.")
    p.add_argument("--global_seed", type=int, default=None,
                   help="Single seed controlling split membership and all "
                        "randomness under --split_mode global_seed_holdout.")

    # XSA path (identical to train.py)
    p.add_argument("--alpha_cls_mode", type=str, default="zero",
                   choices=["zero", "one", "learn"])
    p.add_argument("--alpha_patch_mode", type=str, default="zero",
                   choices=["zero", "one", "learn"])
    p.add_argument("--alpha_init", type=float, default=1.0)
    p.add_argument("--mask_diagonal", action="store_true")
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--diff_transformer", action="store_true",
                   help="Use the Diff Transformer token mixer for the ADP "
                        "raw-RGB ViT comparator. This is mutually exclusive "
                        "with XSA/SRP intervention flags.")
    # SRP path (identical to train.py)
    p.add_argument("--use_srp", action="store_true")
    p.add_argument("--beta_patch_mode", type=str, default="fixed",
                   choices=["zero", "one", "learn", "fixed", "signed_gated"],
                   help="'signed_gated' enables the learned local gate. Requires "
                        "--use_srp; --beta_init is ignored under this mode.")
    p.add_argument("--beta_init", type=float, default=2.0)
    # Signed-gate parameters; consumed only under --beta_patch_mode signed_gated.
    p.add_argument("--srp_gate_placement", type=str, default="post_agg",
                   choices=["post_agg", "pre_q", "pre_k"],
                   help="Signed-gated SRP placement for ADP/NCT ViT. "
                        "post_agg is the default placement; pre_q applies the "
                        "same signed projection to Q before QK attention; "
                        "pre_k applies it to K before QK attention.")
    p.add_argument("--delta_scale", type=float, default=2.0,
                   help="Signed-gate range bound: β_eff = δ · tanh(raw). "
                        "Default 2.0 covers identity / anti-SRP / projection "
                        "/ reflection. δ=1 restricts to signed-projection.")
    p.add_argument("--gate_hidden_dim", type=int, default=16,
                   help="Token-MLP hidden width for the signed gate.")
    p.add_argument("--gate_output_init", type=str, default="zero",
                   choices=["zero", "tiny_normal", "xavier_uniform",
                            "kaiming_uniform", "orthogonal", "constant_beta"])
    p.add_argument("--gate_output_init_scale", type=float, default=1.0)
    p.add_argument("--gate_init_beta0", type=float, default=0.0)
    p.add_argument("--gate_activation", type=str, default="tanh",
                   choices=["tanh", "scaled_sigmoid", "softsign",
                            "hardtanh", "atan"])
    p.add_argument("--gate_activation_temperature", type=float, default=1.0)
    p.add_argument("--gate_count_features", type=str, default="legacy",
                   choices=["legacy", "rawlog", "normlog", "none"])
    p.add_argument("--no_detach_gate_inputs", action="store_true",
                   help="Let gradients flow back through gate diagnostic "
                        "inputs into the patch stream. By default these "
                        "diagnostic inputs are detached.")
    # Probe-style flag (carried over from slide-level): scale ab group LR.
    p.add_argument("--ab_lr_mult", type=float, default=1.0)
    p.add_argument("--neighbor_window", type=int, default=3)
    p.add_argument("--neighbor_shell", type=str, default="cumulative",
                   choices=["cumulative", "ring"])
    p.add_argument("--neighbor_source", type=str, default="real",
                   choices=["real", "shuffled"])
    p.add_argument("--neighbor_shuffle_seed", type=int, default=0)
    p.add_argument("--neighbor_weighting", type=str, default="uniform",
                   choices=["uniform", "gaussian", "inverse_distance"])
    p.add_argument("--neighbor_weight_sigma", type=float, default=1.0)
    p.add_argument("--pos_mode", type=str, default="original",
                   choices=["original", "none"],
                   help="ADP positional policy. original keeps ViT abs pos; none skips it.")

    # Training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--base_lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=8)

    # Diagnostics
    p.add_argument("--val_stats_batches", type=int, default=2)

    # Architecture (rarely needs changing)
    p.add_argument("--image_size", type=int, default=272,
                   help="Native ADP resolution; ViT runs on (image_size/16)² patch tokens.")
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--num_landmarks", type=int, default=64,
                   help="Compatibility knob for shared Diff Transformer code. "
                        "ADP Diff Transformer uses full attention, so this is "
                        "ignored by patch_level_adp.train.")

    args = p.parse_args()

    # --- Argument cross-validation ----------------------------------
    # The signed-gate path is implemented inside PatchSRPAttention
    # (the SRP backend). Without --use_srp the model instantiates the
    # XSAAttention path, which has no signed-gate code. A user passing
    # `--beta_patch_mode signed_gated` without `--use_srp` would
    # silently train the XSA backend instead of the gate, with no
    # visible error, so catch that incompatible flag combination here.
    if args.beta_patch_mode == "signed_gated" and not args.use_srp:
        raise SystemExit(
            "[parse_args] --beta_patch_mode signed_gated requires "
            "--use_srp. The signed-gate code lives in "
            "PatchSRPAttention; without --use_srp the XSAAttention "
            "path is selected and the gate is never instantiated. "
            "Re-run with `--use_srp --beta_patch_mode signed_gated`."
        )
    if args.diff_transformer:
        if args.use_srp:
            raise SystemExit(
                "[parse_args] --diff_transformer is a separate comparator "
                "backend and cannot be combined with --use_srp."
            )
        if args.alpha_cls_mode != "zero" or args.alpha_patch_mode != "zero":
            raise SystemExit(
                "[parse_args] --diff_transformer cannot be combined with "
                "XSA alpha modes. Use alpha_cls_mode=zero and "
                "alpha_patch_mode=zero for the Diff Transformer arm."
            )
        if args.mask_diagonal:
            raise SystemExit(
                "[parse_args] --diff_transformer cannot be combined with "
                "--mask_diagonal; diagonal masking is an XSA/SA variant."
            )
    if args.srp_gate_placement in {"pre_q", "pre_k"} and (
        not args.use_srp or args.beta_patch_mode != "signed_gated"
    ):
        raise SystemExit(
            f"[parse_args] --srp_gate_placement {args.srp_gate_placement} "
            "requires `--use_srp --beta_patch_mode signed_gated`. "
            "Pre-attention placement is a "
            "signed-gated SRP placement, not a legacy fixed-beta mode."
        )
    if args.split_mode == "official":
        if args.global_seed is not None:
            raise SystemExit(
                "[parse_args] --global_seed is valid only with "
                "--split_mode global_seed_holdout. The official ADP split "
                "is fixed by Release1 npy files."
            )
    else:
        if args.global_seed is None:
            raise SystemExit(
                "[parse_args] --split_mode global_seed_holdout requires "
                "--global_seed."
            )
        if args.seed != args.global_seed:
            raise SystemExit(
                "[parse_args] global-seed holdout requires "
                f"--seed == --global_seed; got seed={args.seed}, "
                f"global_seed={args.global_seed}."
            )
        if args.neighbor_shuffle_seed != args.global_seed:
            raise SystemExit(
                "[parse_args] global-seed holdout requires "
                "--neighbor_shuffle_seed == --global_seed so shuffled "
                "neighbor controls share the same seed; got "
                f"neighbor_shuffle_seed={args.neighbor_shuffle_seed}, "
                f"global_seed={args.global_seed}."
            )
    if args.neighbor_window < 3 or args.neighbor_window % 2 != 1:
        raise SystemExit(
            f"[parse_args] --neighbor_window must be odd and >= 3, got "
            f"{args.neighbor_window}"
        )

    return args


def _build_split_metadata(args, train_loader, val_loader, test_loader) -> dict:
    """
    Persist enough split information to verify cross-arm split invariants.

    The full row-id arrays are intentionally included: they are small for ADP
    and allow users to verify that all arms for a global seed used the same
    train/val/test membership, with no hidden fold reuse.
    """
    split_arrays = {
        "train": np.asarray(train_loader.dataset.row_indices, dtype=np.int64),
        "val": np.asarray(val_loader.dataset.row_indices, dtype=np.int64),
        "test": np.asarray(test_loader.dataset.row_indices, dtype=np.int64),
    }
    return {
        "split_mode": args.split_mode,
        "global_seed": args.global_seed,
        "seed": args.seed,
        "neighbor_shuffle_seed": args.neighbor_shuffle_seed,
        "unit": "adp_csv_patch_row",
        "counts": {k: int(v.size) for k, v in split_arrays.items()},
        "indices": {k: v.tolist() for k, v in split_arrays.items()},
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.run_name}] device={device}")

    # --- Data --------------------------------------------------------
    label_cols = get_label_columns()
    n_labels = len(label_cols)
    if args.split_mode == "official":
        train_loader, val_loader, test_loader, _ = build_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size, label_cols=label_cols,
        )
    else:
        split_idx = build_global_seed_split_indices(
            label_cols=label_cols,
            global_seed=args.global_seed,
        )
        train_loader, val_loader, test_loader, _ = build_loaders_from_indices(
            train_idx=split_idx["train"],
            val_idx=split_idx["val"],
            test_idx=split_idx["test"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            label_cols=label_cols,
        )
    split_metadata = _build_split_metadata(args, train_loader, val_loader, test_loader)
    print(f"[{args.run_name}] train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)} "
          f"n_labels={n_labels} split_mode={args.split_mode} "
          f"global_seed={args.global_seed}")

    # --- Model -------------------------------------------------------
    # Built with the same XSA + SRP plumbing as train.py — args.use_srp
    # toggles the backend. img_size=272 → 17×17 patch grid carried through
    # to PatchSRPAttention's neighbor index automatically.
    if args.diff_transformer:
        model = DiffTransformerViT(
            img_size=args.image_size, patch_size=args.patch_size, in_chans=3,
            num_classes=n_labels,
            embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
            num_landmarks=args.num_landmarks,
            mlp_ratio=4.0, qkv_bias=False,
            drop_path_rate=args.drop_path,
            checkpoint_mode="off",
            use_abs_pos_embed=(args.pos_mode == "original"),
        ).to(device)
    else:
        model = ViT(
            img_size=args.image_size, patch_size=args.patch_size, in_chans=3,
            num_classes=n_labels,
            embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
            mlp_ratio=4.0, qkv_bias=True,
            alpha_cls_mode=args.alpha_cls_mode,
            alpha_patch_mode=args.alpha_patch_mode,
            alpha_init=args.alpha_init,
            mask_diagonal=args.mask_diagonal,
            drop_path_rate=args.drop_path,
            use_srp=args.use_srp,
            beta_patch_mode=args.beta_patch_mode,
            beta_init=args.beta_init,
            # Signed-gate parameters; consumed only under
            # use_srp=True + beta_patch_mode='signed_gated'.
            delta_scale=args.delta_scale,
            gate_hidden_dim=args.gate_hidden_dim,
            detach_gate_inputs=not args.no_detach_gate_inputs,
            neighbor_radius=(args.neighbor_window - 1) // 2,
            neighbor_shell=args.neighbor_shell,
            neighbor_source=args.neighbor_source,
            neighbor_shuffle_seed=args.neighbor_shuffle_seed,
            neighbor_weighting=args.neighbor_weighting,
            neighbor_weight_sigma=args.neighbor_weight_sigma,
            gate_output_init=args.gate_output_init,
            gate_output_init_scale=args.gate_output_init_scale,
            gate_init_beta0=args.gate_init_beta0,
            gate_activation=args.gate_activation,
            gate_activation_temperature=args.gate_activation_temperature,
            gate_count_features=args.gate_count_features,
            srp_gate_placement=args.srp_gate_placement,
            use_abs_pos_embed=(args.pos_mode == "original"),
        ).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Under signed_gated, the trainable method
    # params live under `gate.*`, not under the legacy α/β scalars. The old
    # accounting reported `ab_params=0` for signed-gated runs even though
    # the gate has thousands of trainable params. We compute this *before*
    # the optimizer groups for the early banner, then recompute below from
    # the canonical group lists for W&B.
    if args.use_srp and args.beta_patch_mode == "signed_gated":
        n_ab = sum(p.numel() for n, p in model.named_parameters()
                   if p.requires_grad and (".gate." in n or n.startswith("gate.")))
    else:
        n_ab = sum(p.numel() for n, p in model.named_parameters()
                   if p.requires_grad and ("alpha_cls" in n or "alpha_patch" in n
                                            or "beta_patch" in n))
    backend_name = "diff_transformer" if args.diff_transformer else ("srp" if args.use_srp else "xsa")
    method_diag_supported = supports_xsa_srp_diagnostics(model)
    print(f"[{args.run_name}] params_total={n_total:,} trainable={n_trainable:,} "
          f"ab_params={n_ab} backend={backend_name}")

    def snapshot_method_projection_params() -> dict:
        """
        Return the projection parameters defined by the selected method.

        XSA exposes alpha tensors and SRP exposes beta tensors. Diff Transformer
        follows the original differential-attention design instead, so it has
        lambda parameters inside its attention blocks rather than alpha/beta
        SRP geometry. Returning an empty dict keeps artifact schemas valid
        without pretending those projection parameters exist.
        """
        if args.diff_transformer:
            return {}
        if args.use_srp:
            return extract_beta_values(model)
        return extract_alpha_values(model)

    # --- Optimizer ----------------------------------------------------
    # Three optimizer groups let `--ab_lr_mult` scale the full gate
    # (weights + biases), while preserving zero weight decay for gate biases.
    #
    # Groups (signed_gated mode):
    #   backbone_decay  : non-gate params, normal weight_decay
    #   method_decay    : gate weights, normal weight_decay
    #   method_nodecay  : gate biases, weight_decay=0
    #
    # Groups (legacy α/β variants):
    #   backbone_decay  : everything not α/β, normal weight_decay
    #   method_nodecay  : α/β scalars, weight_decay=0
    #   (method_decay is empty in legacy modes)
    def _is_legacy_no_decay(n: str) -> bool:
        return ("alpha_cls" in n or "alpha_patch" in n or "beta_patch" in n)

    def _is_gate_no_decay(n: str) -> bool:
        return (
            "gate." in n
            and (n.endswith(".layer_head_bias")
                 or n.endswith(".head_bias")
                 or n.endswith(".bias"))
        )

    def _is_method_param(n: str) -> bool:
        # Under signed_gated, all gate params are method params.
        if args.beta_patch_mode == "signed_gated":
            return ".gate." in n or n.startswith("gate.")
        # Otherwise method params == legacy α/β scalars.
        return _is_legacy_no_decay(n)

    eff_lr = args.base_lr * (args.batch_size / 256.0)   # linear scale rule on BS=256
    backbone_decay, method_decay, method_nodecay = [], [], []
    method_decay_names, method_nodecay_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        is_no_decay = _is_legacy_no_decay(name) or _is_gate_no_decay(name)
        if is_no_decay:
            method_nodecay.append(p); method_nodecay_names.append(name)
        elif _is_method_param(name):
            method_decay.append(p); method_decay_names.append(name)
        else:
            backbone_decay.append(p)

    # `ab_param_names` retained for back-compat with the W&B config dump
    # below; under the 3-group split it represents both method groups.
    ab_param_names = method_decay_names + method_nodecay_names
    ab_params = method_decay + method_nodecay  # for legacy logging
    # Recompute n_ab_params from the canonical
    # group lists so signed-gated runs report the gate's full param count
    # (the early banner above only handles the common case).
    n_ab_params = sum(p.numel() for p in ab_params)

    param_groups = []
    if backbone_decay:
        param_groups.append({"params": backbone_decay, "weight_decay": args.weight_decay,
                             "group_name": "decay"})
    if method_decay:
        param_groups.append({"params": method_decay, "weight_decay": args.weight_decay,
                             "group_name": "method_decay"})
    if method_nodecay:
        param_groups.append({"params": method_nodecay, "weight_decay": 0.0,
                             "group_name": "ab_nodecay"})
    optimizer = torch.optim.AdamW(param_groups, lr=eff_lr, betas=(0.9, 0.999))

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)

    # --- W&B ----------------------------------------------------------
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_metadata.json").write_text(
        json.dumps(split_metadata, indent=2),
        encoding="utf-8",
    )
    wandb.init(
        project=args.wandb_project, name=args.run_name, mode=args.wandb_mode,
        dir=str(out_dir),
        config={
            **vars(args),
            "n_labels": n_labels, "label_cols": label_cols,
            "split_metadata": split_metadata,
            "n_params_total": n_total, "n_params_trainable": n_trainable,
            "n_ab_params": n_ab_params,
            "effective_lr": eff_lr,
            "total_steps": total_steps, "warmup_steps": warmup_steps,
            "ab_param_names": ab_param_names,
            # Report all three group sizes explicitly. The legacy keys decay_param_count /
            # nodecay_param_count are kept for back-compat but now reflect
            # the new split: "decay" = backbone_decay (non-method weighted),
            # "nodecay" = method_nodecay (legacy α/β + gate biases).
            "backbone_decay_tensor_count": len(backbone_decay),
            "method_decay_tensor_count": len(method_decay),
            "method_nodecay_tensor_count": len(method_nodecay),
            "decay_param_count": len(backbone_decay),
            "nodecay_param_count": len(method_nodecay),
            "backend": backend_name,
        },
        tags=[f"seed-{args.seed}", "adp",
              f"backend-{backend_name}",
              f"dp-{args.drop_path:g}"],
    )

    best_val_map = -1.0
    best_ckpt = out_dir / "best.pt"
    global_step = 0
    has_learnable_alpha = (not args.use_srp) and (
        args.alpha_cls_mode == "learn" or args.alpha_patch_mode == "learn")
    has_learnable_beta = args.use_srp and (args.beta_patch_mode == "learn")
    # Signed-gated runs have no α/β
    # scalars; surface a per-step gate trajectory summary instead.
    has_signed_gate = args.use_srp and (args.beta_patch_mode == "signed_gated")
    alpha_history: list[dict] = []
    beta_history: list[dict] = []
    log_every = 25

    # --- Train loop ---------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss_sum = 0.0; epoch_n = 0
        ep_y, ep_p = [], []

        pbar = tqdm(train_loader, desc=f"[{args.run_name}] ep {epoch+1}/{args.epochs}",
                    leave=False, mininterval=1.0)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr_now = cosine_warmup_lr(global_step, warmup_steps, total_steps, eff_lr)
            # Scale both method groups.
            method_groups = ("ab_nodecay", "method_decay")
            for pg in optimizer.param_groups:
                if pg.get("group_name") in method_groups and args.ab_lr_mult != 1.0:
                    pg["lr"] = lr_now * args.ab_lr_mult
                else:
                    pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(device, torch.bfloat16):
                logits = model(imgs)
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_sum += float(loss.item()) * labels.numel()
            epoch_n += labels.numel()
            ep_y.append(labels.detach().cpu().numpy())
            ep_p.append(logits.float().detach().cpu().numpy())

            if global_step % 50 == 0:
                wandb.log({"train/loss_step": float(loss.item()),
                           "train/lr": lr_now}, step=global_step)
            if has_learnable_alpha and global_step % log_every == 0:
                a = extract_alpha_values(model)
                alpha_history.append({"step": int(global_step),
                                      "alpha_cls": a["alpha_cls"].cpu().numpy().copy(),
                                      "alpha_patch": a["alpha_patch"].cpu().numpy().copy()})
                wandb.log({"alpha_step/alpha_cls_mean": float(a["alpha_cls"].mean()),
                           "alpha_step/alpha_patch_mean": float(a["alpha_patch"].mean())},
                          step=global_step)
            elif has_learnable_beta and global_step % log_every == 0:
                b = extract_beta_values(model)
                beta_history.append({"step": int(global_step),
                                     "beta_patch": b["beta_patch"].cpu().numpy().copy()})
                wandb.log({"beta_step/beta_patch_mean": float(b["beta_patch"].mean())},
                          step=global_step)
            # Signed-gate trajectory.
            # Per-block β_eff means / sign-bin fractions, no-op when the
            # gate is not active.
            if has_signed_gate and global_step % log_every == 0:
                from slide_level_srp.src.gate_signed import signed_gate_step_summary
                gate_payload = signed_gate_step_summary(model)
                if gate_payload:
                    wandb.log(gate_payload, step=global_step)
            global_step += 1

        # --- end of epoch ---
        train_y = np.concatenate(ep_y, axis=0)
        train_p = np.concatenate(ep_p, axis=0)
        train_metrics = compute_multilabel_map(train_y, train_p, label_cols)
        train_metrics["loss"] = epoch_loss_sum / max(1, epoch_n)

        val_metrics, val_diag, _, _, _ = evaluate(
            model, val_loader, device, n_labels,
            capture_stats=method_diag_supported, max_stats_batches=args.val_stats_batches,
        )
        ab_snap = snapshot_method_projection_params()

        wandb.log({"train/loss": train_metrics["loss"],
                   "train/map_macro": train_metrics["map_macro"],
                   "train/f1": train_metrics["f1"],
                   "train/acc": train_metrics["acc"],
                   "train/auc": train_metrics["auc"],
                   "train/map_L1": train_metrics["map_L1"],
                   "train/map_L2": train_metrics["map_L2"],
                   "train/map_L3": train_metrics["map_L3"]}, step=global_step)
        wandb.log({"val/loss": val_metrics["loss"],
                   "val/map_macro": val_metrics["map_macro"],
                   "val/f1": val_metrics["f1"],
                   "val/acc": val_metrics["acc"],
                   "val/auc": val_metrics["auc"],
                   "val/map_L1": val_metrics["map_L1"],
                   "val/map_L2": val_metrics["map_L2"],
                   "val/map_L3": val_metrics["map_L3"],
                   "val/map_weighted": val_metrics["map_weighted"]}, step=global_step)
        if val_diag is not None:
            log_val_diagnostics(val_diag, step=global_step)
        if ab_snap:
            log_alpha_or_beta_scalars(ab_snap, prefix="ab", step=global_step)

        dt = time.time() - t0
        print(f"[{args.run_name}] ep{epoch+1}: "
              f"train[loss={train_metrics['loss']:.4f} mAP={train_metrics['map_macro']:.4f}] "
              f"val[loss={val_metrics['loss']:.4f} mAP={val_metrics['map_macro']:.4f} "
              f"f1={val_metrics['f1']:.3f} acc={val_metrics['acc']:.3f} "
              f"auc={val_metrics['auc']:.3f} "
              f"L1={val_metrics['map_L1']:.3f} L2={val_metrics['map_L2']:.3f} "
              f"L3={val_metrics['map_L3']:.3f}] | {dt:.1f}s")

        # Best by val macro-mAP.
        if val_metrics["map_macro"] > best_val_map:
            best_val_map = val_metrics["map_macro"]
            torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                        "args": vars(args), "val_metrics": val_metrics},
                       best_ckpt)

    # --- Final test pass on best checkpoint ---
    print(f"[{args.run_name}] reloading best (val_mAP={best_val_map:.4f})")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    test_metrics, test_diag, test_y, test_p, test_gate_stats = evaluate(
        model, test_loader, device, n_labels,
        capture_stats=method_diag_supported, max_stats_batches=None,
        collect_gate_stats=has_signed_gate,
    )
    print(f"[{args.run_name}] TEST mAP={test_metrics['map_macro']:.4f}  "
          f"F1={test_metrics['f1']:.4f}  ACC={test_metrics['acc']:.4f}  "
          f"AUC={test_metrics['auc']:.4f}  "
          f"L1={test_metrics['map_L1']:.3f}  L2={test_metrics['map_L2']:.3f}  "
          f"L3={test_metrics['map_L3']:.3f}  loss={test_metrics['loss']:.4f}")

    final_ab = snapshot_method_projection_params()

    wandb.log({"test/loss": test_metrics["loss"],
               "test/map_macro": test_metrics["map_macro"],
               "test/f1": test_metrics["f1"],
               "test/acc": test_metrics["acc"],
               "test/auc": test_metrics["auc"],
               "test/map_L1": test_metrics["map_L1"],
               "test/map_L2": test_metrics["map_L2"],
               "test/map_L3": test_metrics["map_L3"],
               "test/map_weighted": test_metrics["map_weighted"]}, step=global_step)

    # NPZ payload — one row per (label, AP) for downstream analysis,
    # plus the diagnostics tensors.
    npz_payload = {
        "test_metrics": json.dumps({k: v for k, v in test_metrics.items()
                                    if k not in ("ap_per_label", "auc_per_label")}),
        "ap_per_label_keys": np.array(list(test_metrics["ap_per_label"].keys())),
        "ap_per_label_values": np.array([test_metrics["ap_per_label"][k]
                                         for k in test_metrics["ap_per_label"]],
                                        dtype=np.float64),
        "auc_per_label_keys": np.array(list(test_metrics["auc_per_label"].keys())),
        "auc_per_label_values": np.array([test_metrics["auc_per_label"][k]
                                          for k in test_metrics["auc_per_label"]],
                                         dtype=np.float64),
        "best_val_map_macro": best_val_map,
        "label_cols": np.array(label_cols),
        "backend": backend_name,
        "split_mode": args.split_mode,
        "global_seed": -1 if args.global_seed is None else int(args.global_seed),
        "split_metadata": json.dumps(split_metadata),
        "train_indices": np.asarray(split_metadata["indices"]["train"], dtype=np.int64),
        "val_indices": np.asarray(split_metadata["indices"]["val"], dtype=np.int64),
        "test_indices": np.asarray(split_metadata["indices"]["test"], dtype=np.int64),
        "test_y": test_y.astype(np.uint8),       # ground truth multi-hot
        "test_logits": test_p.astype(np.float32) # raw model logits, for re-analysis
    }
    for role, arr in final_ab.items():
        npz_payload[role] = arr.cpu().numpy()
    if test_diag is not None:
        for k, v in test_diag.items():
            npz_payload[k] = v.cpu().numpy()
    if alpha_history:
        npz_payload["alpha_history_steps"] = np.array([h["step"] for h in alpha_history], dtype=np.int64)
        npz_payload["alpha_history_cls"] = np.stack([h["alpha_cls"] for h in alpha_history], axis=0)
        npz_payload["alpha_history_patch"] = np.stack([h["alpha_patch"] for h in alpha_history], axis=0)
    if beta_history:
        npz_payload["beta_history_steps"] = np.array([h["step"] for h in beta_history], dtype=np.int64)
        npz_payload["beta_history_patch"] = np.stack([h["beta_patch"] for h in beta_history], axis=0)
    # Per-example signed-gate stats. Empty dict for
    # non-signed-gated runs; under signed_gated the keys are the
    # gate_block{i}_{stat} entries that downstream analysis relies on.
    for k, v in test_gate_stats.items():
        npz_payload[f"gate_{k}"] = v
    np.savez(out_dir / "test_artifacts.npz", **npz_payload)

    wandb.finish()
    print(f"[{args.run_name}] done")


if __name__ == "__main__":
    main()
