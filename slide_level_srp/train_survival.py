"""TCGA-to-Atlas survival trainer for reported gated-SRP experiments.

This is an additive entrypoint rather than a mode inside ``train.py``.  The
classification path stays stable while survival gets its own case-level split
logic, discrete-time NLL, C-index model selection, and per-case prediction
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

import wandb

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slide_level.src.diagnostics import autocast_ctx
from slide_level_srp.data_tcga_survival import (
    TCGA_ATLAS_FEATURE_ROOT,
    TCGA_FEATURE_KEY,
    TCGA_SURVIVAL_LABEL_CSV,
    build_tcga_survival_fold_assignments,
    build_tcga_survival_loaders_for_fold,
    build_tcga_survival_seed_holdout_assignment,
    enumerate_tcga_survival_slides,
    filter_tcga_survival_records,
    make_time_bin_edges,
    survival_split_metadata,
    tcga_survival_inventory,
)
from slide_level_srp.train import (
    _ABLATIONS,
    _build_model,
    _checked_fold_assignment,
    _model_forward,
    build_adamw_optimizer,
    cosine_warmup_lr,
    method_parameter_summary,
    set_seed,
)


def survival_nll_loss(
    logits: torch.Tensor,
    event: torch.Tensor,
    time_bin: torch.Tensor,
) -> torch.Tensor:
    """Discrete-time survival negative log-likelihood.

    For an observed event in bin ``y``, the likelihood is survival through
    bins ``< y`` and event hazard at ``y``.  For a censored case in bin ``y``,
    the likelihood is survival through bins ``<= y``.  This is the CLAM /
    2DMamba-style objective used in the approved survival plan.
    """
    hazards = torch.sigmoid(logits.float()).clamp(min=1e-7, max=1.0 - 1e-7)
    event = event.float().view(-1)
    time_bin = time_bin.long().view(-1).clamp(min=0, max=hazards.shape[1] - 1)
    bins = torch.arange(hazards.shape[1], device=hazards.device).view(1, -1)
    y = time_bin.view(-1, 1)
    log_h = torch.log(hazards)
    log_1mh = torch.log1p(-hazards)
    event_survival = (log_1mh * (bins < y).to(log_1mh.dtype)).sum(dim=1)
    event_hazard = log_h.gather(1, y).squeeze(1)
    censor_survival = (log_1mh * (bins <= y).to(log_1mh.dtype)).sum(dim=1)
    per_item = -event * (event_survival + event_hazard) - (1.0 - event) * censor_survival
    return per_item.mean()


def risk_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Higher risk means earlier event for Harrell C-index."""
    hazards = torch.sigmoid(logits.float()).clamp(min=1e-7, max=1.0 - 1e-7)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return -survival.sum(dim=1)


def harrell_c_index(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
) -> float:
    """Harrell concordance index with 0.5 credit for tied risk."""
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    events = np.asarray(events, dtype=np.int64).reshape(-1)
    risks = np.asarray(risks, dtype=np.float64).reshape(-1)
    comparable = 0
    concordant = 0.0
    n = times.shape[0]
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if times[i] >= times[j]:
                continue
            comparable += 1
            if risks[i] > risks[j]:
                concordant += 1.0
            elif risks[i] == risks[j]:
                concordant += 0.5
    if comparable == 0:
        return float("nan")
    return float(concordant / comparable)


def aggregate_case_predictions(slide_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in slide_rows:
        grouped[row["case_id"]].append(row)
    out: list[dict] = []
    for case_id, rows in sorted(grouped.items()):
        first = rows[0]
        risk = float(np.mean([float(r["risk"]) for r in rows]))
        out.append({
            "case_id": case_id,
            "cohort": first["cohort"],
            "endpoint": first["endpoint"],
            "event": int(first["event"]),
            "time_days": float(first["time_days"]),
            "time_bin": int(first["time_bin"]),
            "risk": risk,
            "n_slides": len(rows),
            "slide_ids": ";".join(r["slide_id"] for r in rows),
        })
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_survival(
    model,
    loader,
    *,
    device,
    backend: str,
    ablation_spec: dict,
    autocast_dtype=torch.bfloat16,
) -> tuple[dict, list[dict], list[dict]]:
    model.eval()
    slide_rows: list[dict] = []
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            event = batch["event"].to(device, non_blocking=True)
            time_bin = batch["time_bin"].to(device, non_blocking=True)
            with autocast_ctx(device, autocast_dtype):
                logits = _model_forward(
                    model, batch, backend, device,
                    ablation_spec=ablation_spec,
                )
                loss = survival_nll_loss(logits, event, time_bin)
            risk = risk_from_logits(logits)
            hazards = torch.sigmoid(logits.float()).squeeze(0).detach().cpu().numpy()
            slide_rows.append({
                "cohort": batch["cohort"][0],
                "endpoint": batch["endpoint"][0],
                "case_id": batch["case_id"][0],
                "slide_id": batch["slide_id"][0],
                "event": int(event.item()),
                "time_days": float(batch["time_days"].item()),
                "time_bin": int(time_bin.item()),
                "risk": float(risk.item()),
                "hazards": json.dumps([float(x) for x in hazards.tolist()]),
                "n_tokens": int(batch["n_tokens"][0]),
            })
            loss_sum += float(loss.item())
            n += 1
    case_rows = aggregate_case_predictions(slide_rows)
    slide_c = harrell_c_index(
        np.asarray([row["time_days"] for row in slide_rows]),
        np.asarray([row["event"] for row in slide_rows]),
        np.asarray([row["risk"] for row in slide_rows]),
    )
    case_c = harrell_c_index(
        np.asarray([row["time_days"] for row in case_rows]),
        np.asarray([row["event"] for row in case_rows]),
        np.asarray([row["risk"] for row in case_rows]),
    )
    metrics = {
        "loss": float(loss_sum / max(1, n)),
        "slide_c_index": float(slide_c),
        "case_c_index": float(case_c),
        "n_slides": int(len(slide_rows)),
        "n_cases": int(len(case_rows)),
        "n_events": int(sum(row["event"] for row in case_rows)),
    }
    metrics["n_censored"] = int(metrics["n_cases"] - metrics["n_events"])
    return metrics, slide_rows, case_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--out_dir", default="runs/tcga_survival")
    parser.add_argument("--wandb_project", default="GatedSRP_tcga_survival")
    parser.add_argument("--wandb_mode", default="disabled", choices=["online", "offline", "disabled"])

    parser.add_argument("--cohort", required=True, help="KIRC, KIRP, LUAD, STAD, or UCEC")
    parser.add_argument("--endpoint", default="OS")
    parser.add_argument("--label_csv", default=TCGA_SURVIVAL_LABEL_CSV)
    parser.add_argument("--feature_root", default=TCGA_ATLAS_FEATURE_ROOT)
    parser.add_argument("--feature_key", default=TCGA_FEATURE_KEY)
    parser.add_argument("--split_mode", default="case_level_5fold",
                        choices=["case_level_5fold", "global_seed_holdout"])
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--fold_seed", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_bins", type=int, default=4)
    parser.add_argument("--drop_nonpositive_time", action="store_true", default=True)
    parser.add_argument("--keep_nonpositive_time", action="store_false", dest="drop_nonpositive_time")

    parser.add_argument("--ablation", required=True, choices=list(_ABLATIONS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_cap", type=int, default=None)
    parser.add_argument("--val_cap", type=int, default=None)
    parser.add_argument("--test_cap", type=int, default=None)

    parser.add_argument("--in_dim", type=int, default=1536)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--num_landmarks", type=int, default=64)
    parser.add_argument("--pinv_iterations", type=int, default=6)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--checkpoint_mode", default="whole_block", choices=["whole_block", "per_module", "off"])
    parser.add_argument("--layerscale_init", type=float, default=0.0)
    parser.add_argument("--ln_specialization", default="shared", choices=["shared", "cls_patch"])
    parser.add_argument("--ln_specialization_scope", default="block", choices=["block", "block_final"])
    parser.add_argument("--pos_mode", default="original", choices=["original", "none"])
    parser.add_argument("--no_ppeg", action="store_true")

    parser.add_argument("--beta_init", type=float, default=1.0)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    parser.add_argument("--gate_output_init", default="zero",
                        choices=["zero", "tiny_normal", "xavier_uniform", "kaiming_uniform", "orthogonal", "constant_beta"])
    parser.add_argument("--gate_output_init_scale", type=float, default=1.0)
    parser.add_argument("--gate_init_beta0", type=float, default=0.0)
    parser.add_argument("--gate_activation", default="tanh",
                        choices=["tanh", "scaled_sigmoid", "sigmoid01", "softsign", "hardtanh", "atan"])
    parser.add_argument("--gate_activation_temperature", type=float, default=1.0)
    parser.add_argument("--gate_factorization", default="full",
                        choices=["full", "token_only", "head_only", "no_bias"])
    parser.add_argument("--gate_count_features", default="legacy",
                        choices=["legacy", "rawlog", "normlog", "none"])
    parser.add_argument("--no_detach_gate_inputs", action="store_true")
    parser.add_argument("--gate_l2_reg", type=float, default=0.0)
    parser.add_argument("--rcd_adapter_kind", default="lowrank", choices=["lowrank", "diag"])
    parser.add_argument("--rcd_rank", type=int, default=16)
    parser.add_argument("--learned_r_hidden_dim", type=int, default=16)
    parser.add_argument("--ab_lr_mult", type=float, default=1.0)

    parser.add_argument("--neighbor_window", type=int, default=3)
    parser.add_argument("--neighbor_shell", default="cumulative", choices=["cumulative", "ring"])
    parser.add_argument("--neighbor_source", default="real", choices=["real", "shuffled"])
    parser.add_argument("--neighbor_shuffle_seed", type=int, default=None)
    parser.add_argument("--neighbor_weighting", default="uniform", choices=["uniform", "gaussian", "inverse_distance"])
    parser.add_argument("--neighbor_weight_sigma", type=float, default=1.0)
    args = parser.parse_args()

    if args.grad_accum <= 0:
        raise SystemExit(f"--grad_accum must be > 0, got {args.grad_accum}")
    if args.split_mode == "case_level_5fold":
        if args.fold is None:
            raise SystemExit("--fold is required when --split_mode case_level_5fold")
    else:
        # For the reported sweep the global seed is the single source of
        # split randomness.  Falling back to --seed keeps ad-hoc runs concise
        # while the generated manifests still write --global_seed explicitly.
        if args.global_seed is None:
            args.global_seed = args.seed
        if args.fold is None:
            args.fold = -1
    if args.neighbor_window < 3 or args.neighbor_window % 2 != 1:
        raise SystemExit(f"--neighbor_window must be odd and >=3, got {args.neighbor_window}")
    if args.neighbor_shuffle_seed is None:
        args.neighbor_shuffle_seed = args.seed
    if args.pos_mode == "none":
        args.no_ppeg = True
    if args.n_bins < 2:
        raise SystemExit(f"--n_bins must be >=2, got {args.n_bins}")
    args.num_classes = args.n_bins
    return args


def main() -> None:
    args = parse_args()
    spec = _ABLATIONS[args.ablation]
    backend = spec["backend"]
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.split_mode == "case_level_5fold":
        split_label = f"fold={args.fold}/{args.n_folds}"
    else:
        split_label = f"global_seed={args.global_seed} outer_parts={args.n_folds}"
    print(
        f"[{args.run_name}] TCGA survival cohort={args.cohort} endpoint={args.endpoint} "
        f"{split_label} ablation={args.ablation} device={device}"
    )
    if device.type == "cuda":
        print(f"[{args.run_name}] gpu={torch.cuda.get_device_name(0)}")

    records = enumerate_tcga_survival_slides(
        cohort=args.cohort,
        endpoint=args.endpoint,
        label_csv=args.label_csv,
        feature_root=args.feature_root,
        feature_key=args.feature_key,
        drop_nonpositive_time=args.drop_nonpositive_time,
        require_h5=True,
    )
    if args.split_mode == "case_level_5fold":
        fold_assignments = build_tcga_survival_fold_assignments(
            records,
            n_folds=args.n_folds,
            fold_seed=args.fold_seed,
        )
        fold = _checked_fold_assignment(fold_assignments, args.fold)
    else:
        fold = build_tcga_survival_seed_holdout_assignment(
            records,
            global_seed=int(args.global_seed),
            n_outer_parts=args.n_folds,
        )
    train_records = filter_tcga_survival_records(records, fold.train_cases)
    bin_edges = make_time_bin_edges(train_records, n_bins=args.n_bins)
    train_loader, val_loader, test_loader = build_tcga_survival_loaders_for_fold(
        records,
        fold,
        bin_edges=bin_edges,
        num_workers=args.num_workers,
        train_cap=args.train_cap,
        val_cap=args.val_cap,
        test_cap=args.test_cap,
        feature_key=args.feature_key,
        expected_dim=args.in_dim,
        neighbor_radius=(args.neighbor_window - 1) // 2,
        neighbor_shell=args.neighbor_shell,
        neighbor_source=args.neighbor_source,
        neighbor_shuffle_seed=args.neighbor_shuffle_seed,
        neighbor_weighting=args.neighbor_weighting,
        neighbor_weight_sigma=args.neighbor_weight_sigma,
    )
    split_meta = survival_split_metadata(
        records,
        fold,
        bin_edges=bin_edges,
        n_bins=args.n_bins,
    )
    if args.split_mode == "case_level_5fold":
        split_meta["fold_seed"] = args.fold_seed
    else:
        split_meta["outer_parts"] = args.n_folds
    split_meta["inventory"] = tcga_survival_inventory(records)
    print(
        f"[{args.run_name}] train={len(train_loader.dataset)} slides "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)} "
        f"bin_edges={','.join(f'{x:.1f}' for x in bin_edges)}"
    )

    model = _build_model(args, backend, spec, device)
    names, n_method, n_trainable, n_total = method_parameter_summary(model, args)
    print(
        f"[{args.run_name}] params_total={n_total:,} trainable={n_trainable:,} "
        f"method_params={n_method} ({'+'.join(names[:5]) if names else 'none'})"
    )
    optimizer = build_adamw_optimizer(model, args)
    steps_per_epoch = max(1, math.ceil(len(train_loader.dataset) / args.grad_accum))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_metadata.json").write_text(
        json.dumps(split_meta, indent=2, sort_keys=True),
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
            "split_metadata": split_meta,
            "n_params_total": n_total,
            "n_params_trainable": n_trainable,
            "n_method_params": n_method,
        },
        tags=[
            "tcga-survival",
            f"cohort-{args.cohort}",
            f"endpoint-{args.endpoint}",
            f"split-{args.split_mode}",
            f"fold-{args.fold}",
            f"global-seed-{args.global_seed}",
            f"ablation-{args.ablation}",
        ],
    )

    best_val_c = -float("inf")
    best_ckpt = out_dir / "best.pt"
    history: list[dict] = []
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_seen = 0
        optimizer.zero_grad(set_to_none=True)
        slides_in_accum = 0
        pbar = tqdm(
            train_loader,
            desc=f"[{args.run_name}] ep {epoch + 1}/{args.epochs}",
            leave=False,
            mininterval=1.0,
        )
        for batch_idx, batch in enumerate(pbar):
            event = batch["event"].to(device, non_blocking=True)
            time_bin = batch["time_bin"].to(device, non_blocking=True)
            lr_now = cosine_warmup_lr(global_step, warmup_steps, total_steps, args.base_lr, min_lr=0.0)
            for group in optimizer.param_groups:
                if group.get("group_name") in {"ab_nodecay", "method_decay"} and args.ab_lr_mult != 1.0:
                    group["lr"] = lr_now * args.ab_lr_mult
                else:
                    group["lr"] = lr_now

            window_start = (batch_idx // args.grad_accum) * args.grad_accum
            window_size = min(args.grad_accum, len(train_loader) - window_start)
            with autocast_ctx(device, torch.bfloat16):
                logits = _model_forward(
                    model,
                    batch,
                    backend,
                    device,
                    ablation_spec=spec,
                )
                raw_loss = survival_nll_loss(logits, event, time_bin)
                loss = raw_loss / window_size
            loss.backward()
            loss_sum += float(raw_loss.detach().item())
            n_seen += 1
            slides_in_accum += 1

            is_last = batch_idx == len(train_loader) - 1
            if slides_in_accum >= args.grad_accum or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                slides_in_accum = 0
                if global_step % 50 == 0:
                    wandb.log({"train/loss_step": float(raw_loss.item()), "train/lr": lr_now}, step=global_step)
                global_step += 1

        train_metrics = {"loss": float(loss_sum / max(1, n_seen))}
        val_metrics, _, _ = evaluate_survival(
            model,
            val_loader,
            device=device,
            backend=backend,
            ablation_spec=spec,
        )
        dt = time.time() - t0
        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "seconds": dt,
        })
        wandb.log({f"train/{k}": v for k, v in train_metrics.items()}, step=global_step)
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
        print(
            f"[{args.run_name}] ep{epoch + 1}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_c={val_metrics['case_c_index']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} {dt:.1f}s"
        )
        val_c = val_metrics["case_c_index"]
        val_for_selection = val_c if not np.isnan(val_c) else -val_metrics["loss"]
        if val_for_selection > best_val_c:
            best_val_c = val_for_selection
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                    "split_metadata": split_meta,
                },
                best_ckpt,
            )

    if not best_ckpt.exists():
        raise RuntimeError(f"[{args.run_name}] training ended without a best checkpoint")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_metrics, val_slide_rows, val_case_rows = evaluate_survival(
        model,
        val_loader,
        device=device,
        backend=backend,
        ablation_spec=spec,
    )
    test_metrics, test_slide_rows, test_case_rows = evaluate_survival(
        model,
        test_loader,
        device=device,
        backend=backend,
        ablation_spec=spec,
    )
    metrics = {
        "best_epoch": int(ckpt["epoch"]),
        "val": val_metrics,
        "test": test_metrics,
        "cohort": args.cohort,
        "endpoint": args.endpoint,
        "split_mode": args.split_mode,
        "fold": args.fold,
        "global_seed": args.global_seed,
        "ablation": args.ablation,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _write_csv(out_dir / "predictions_val_slide.csv", val_slide_rows)
    _write_csv(out_dir / "predictions_val_case.csv", val_case_rows)
    _write_csv(out_dir / "predictions_test_slide.csv", test_slide_rows)
    _write_csv(out_dir / "predictions_test_case.csv", test_case_rows)
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()}, step=global_step)
    wandb.finish()
    print(
        f"[{args.run_name}] TEST case_c={test_metrics['case_c_index']:.4f} "
        f"slide_c={test_metrics['slide_c_index']:.4f} "
        f"n_cases={test_metrics['n_cases']}"
    )


if __name__ == "__main__":
    main()
