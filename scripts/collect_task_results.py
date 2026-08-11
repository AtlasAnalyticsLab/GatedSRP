#!/usr/bin/env python3
"""Collect rerun metrics into reference TSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


CLASSIFICATION_METRICS = {
    "cam16": ["f1", "acc", "auc"],
    "cam17": ["f1", "acc", "auc"],
    "kgh": ["f1", "acc", "auc"],
    "panda": ["kappa_quad", "f1", "acc", "auc"],
    "bracs": ["f1", "acc", "auc"],
}

TASK_SELECTOR_FIELDS = (
    "dataset",
    "cohort",
    "method",
    "seed",
    "experiment",
    "endpoint",
    "method_label",
    "delta_scale",
    "gate_hidden_dim",
    "selection_status",
    "run_name",
)


def non_empty_selector(value: str) -> str:
    """Reject empty selectors before result collection starts."""
    if not value:
        raise argparse.ArgumentTypeError("selector values cannot be empty")
    return value


def row_matches(row: dict[str, str], filters: dict[str, str]) -> bool:
    """Select only rows that match every requested manifest field exactly."""
    return all(row.get(key) == value for key, value in filters.items())


def selected_filters(args: argparse.Namespace) -> dict[str, str]:
    """Build the shared task selector map from parsed named options."""
    return {
        field: value
        for field in TASK_SELECTOR_FIELDS
        if (value := getattr(args, field)) is not None
    }


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def npz_json_metric(path: Path) -> dict[str, float]:
    """Load the JSON test_metrics payload saved by classification trainers."""
    data = np.load(path, allow_pickle=True)
    raw = data["test_metrics"]
    text = raw.item() if hasattr(raw, "item") else str(raw)
    return json.loads(text)


def fmt(values: list[float]) -> str:
    if not values:
        return ""
    arr = np.asarray(values, dtype=np.float64)
    # Partial smoke reruns often contain one seed. Use a zero sample spread
    # there instead of emitting NumPy's ddof=1 NaN, while preserving the
    # sample-standard-deviation convention for the full five-seed tables.
    spread = 0.0 if arr.size == 1 else float(arr.std(ddof=1))
    return f"{arr.mean():.4f} +/- {spread:.4f} ({arr.size}/5)"


def collect_classification(
    manifest: Path,
    out_root: Path,
    strict: bool,
    *,
    filters: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    per_run: list[dict] = []
    missing: list[str] = []
    for row in read_tsv(manifest):
        # Apply selectors before strict artifact checks so one selected launcher
        # row can be validated without requiring the rest of the run matrix.
        if not row_matches(row, filters or {}):
            continue
        artifact = out_root / row["run_name"] / "test_artifacts.npz"
        if not artifact.exists():
            missing.append(row["run_name"])
            continue
        metrics = npz_json_metric(artifact)
        rec = {
            "dataset": row["dataset"],
            "method": row["method"],
            "method_label": row["method_label"],
            "seed": int(row["seed"]),
            "run_name": row["run_name"],
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                rec[key] = float(value)
        per_run.append(rec)

    if missing and strict:
        raise SystemExit(f"missing classification artifacts: {', '.join(missing[:10])}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in per_run:
        grouped[(rec["dataset"], rec["method"])].append(rec)

    summary: list[dict] = []
    for (dataset, method), rows in sorted(grouped.items()):
        out = {
            "dataset": dataset,
            "method": method,
            "method_label": rows[0]["method_label"],
            "n": len(rows),
        }
        for metric in CLASSIFICATION_METRICS.get(dataset, []):
            values = [float(r[metric]) for r in rows if metric in r]
            out[metric] = fmt(values)
        summary.append(out)
    return per_run, summary


def collect_survival(
    manifest: Path,
    out_root: Path,
    strict: bool,
    *,
    filters: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    per_run: list[dict] = []
    missing: list[str] = []
    for row in read_tsv(manifest):
        if not row_matches(row, filters or {}):
            continue
        path = out_root / row["run_name"] / "metrics.json"
        if not path.exists():
            missing.append(row["run_name"])
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rec = {
            "cohort": row["cohort"],
            "method": row["method"],
            "method_label": row["method_label"],
            "seed": int(row["seed"]),
            "run_name": row["run_name"],
            "best_epoch": metrics.get("best_epoch", ""),
            "val_case_c_index": metrics["val"]["case_c_index"],
            "test_case_c_index": metrics["test"]["case_c_index"],
            "test_slide_c_index": metrics["test"]["slide_c_index"],
            "test_n_cases": metrics["test"]["n_cases"],
            "test_n_events": metrics["test"]["n_events"],
        }
        per_run.append(rec)

    if missing and strict:
        raise SystemExit(f"missing TCGA survival artifacts: {', '.join(missing[:10])}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in per_run:
        grouped[(rec["cohort"], rec["method"])].append(rec)

    summary: list[dict] = []
    for (cohort, method), rows in sorted(grouped.items()):
        summary.append({
            "cohort": cohort,
            "method": method,
            "method_label": rows[0]["method_label"],
            "n": len(rows),
            "test_case_c_index": fmt([float(r["test_case_c_index"]) for r in rows]),
            "val_case_c_index": fmt([float(r["val_case_c_index"]) for r in rows]),
        })
    return per_run, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classification-manifest",
        type=Path,
        default=Path("configs/classification_tasks.tsv"),
    )
    parser.add_argument(
        "--survival-manifest",
        type=Path,
        default=Path("configs/survival_tasks.tsv"),
    )
    parser.add_argument(
        "--classification-runs",
        type=Path,
        default=Path(
            os.environ.get("GATEDSRP_CLASSIFICATION_OUT", "runs/classification")
        ),
    )
    parser.add_argument(
        "--survival-runs",
        type=Path,
        default=Path(
            os.environ.get("GATEDSRP_TCGA_SURVIVAL_OUT", "runs/survival_tasks")
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/rerun"))
    parser.add_argument("--strict", action="store_true", help="Fail if any manifest artifact is missing.")
    for field in TASK_SELECTOR_FIELDS:
        parser.add_argument(
            f"--{field.replace('_', '-')}",
            dest=field,
            type=non_empty_selector,
            metavar="VALUE",
            help=f"Collect rows whose {field} column exactly matches VALUE.",
        )
    args = parser.parse_args()
    filters = selected_filters(args)

    class_per, class_summary = collect_classification(
        args.classification_manifest,
        args.classification_runs,
        args.strict,
        filters=filters,
    )
    surv_per, surv_summary = collect_survival(
        args.survival_manifest,
        args.survival_runs,
        args.strict,
        filters=filters,
    )
    # A misspelled selector value would otherwise produce empty TSV files while
    # returning success. In strict mode, require at least one validated run.
    if args.strict and not class_per and not surv_per:
        raise SystemExit("no task runs matched the requested selectors")

    class_fields = sorted({key for row in class_per for key in row})
    surv_fields = sorted({key for row in surv_per for key in row})
    write_tsv(args.out_dir / "classification_per_seed.tsv", class_per, class_fields)
    write_tsv(
        args.out_dir / "classification_summary.tsv",
        class_summary,
        sorted({key for row in class_summary for key in row}),
    )
    write_tsv(args.out_dir / "survival_per_seed.tsv", surv_per, surv_fields)
    write_tsv(
        args.out_dir / "survival_summary.tsv",
        surv_summary,
        sorted({key for row in surv_summary for key in row}),
    )

    print(f"classification: collected {len(class_per)} runs into {args.out_dir}")
    print(f"tcga survival: collected {len(surv_per)} runs into {args.out_dir}")


if __name__ == "__main__":
    main()
