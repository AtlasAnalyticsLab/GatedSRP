#!/usr/bin/env python3
"""Collect seed-level metrics from a typed-comparison manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SURVIVAL_DATASETS = {"KIRC", "KIRP", "LUAD", "STAD", "UCEC"}
MANIFEST_METADATA = {"command", "run_name", "seed", "access"}
RUNTIME_FIELDS = (
    "peak_reserved_gib",
    "train_wsi_per_second",
    "test_wsi_per_second",
)


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a command manifest while preserving its public grouping columns."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"run_name", "seed", "command"}
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return fields, rows


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def is_survival(row: dict[str, str]) -> bool:
    """Infer the artifact schema from explicit task or TCGA cohort metadata."""
    if row.get("task"):
        return row["task"] == "survival"
    dataset = row.get("cohort") or row.get("dataset", "")
    return dataset.upper().removeprefix("TCGA-") in SURVIVAL_DATASETS


def _finite_numeric_metrics(payload: dict[str, object]) -> dict[str, float]:
    """Keep scalar test metrics while excluding labels and nested payloads."""
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        # JSON booleans are integers in Python, but averaging a status flag as
        # an evaluation metric would be misleading.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            metrics[key] = numeric
    return metrics


def load_classification_metrics(run_dir: Path) -> dict[str, float]:
    """Read every scalar test metric embedded by a classification trainer."""
    artifact = run_dir / "test_artifacts.npz"
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    payload = np.load(artifact, allow_pickle=True)["test_metrics"]
    text = payload.item() if hasattr(payload, "item") else str(payload)
    return _finite_numeric_metrics(json.loads(text))


def primary_classification_metric(
    metrics: dict[str, float],
    dataset: str,
) -> tuple[str, float]:
    """Select the headline metric while retaining all metrics separately."""
    dataset_key = dataset.strip().upper()
    if dataset_key.startswith("PANDA"):
        metric = "kappa_quad"
    elif dataset_key == "ADP":
        metric = "map_macro"
    else:
        metric = "f1"
    if metric not in metrics:
        raise KeyError(f"classification artifact has no {metric!r} metric")
    return metric, float(metrics[metric])


def load_survival_metrics(run_dir: Path) -> dict[str, float]:
    """Read every scalar test metric produced by a survival trainer."""
    artifact = run_dir / "metrics.json"
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    return _finite_numeric_metrics(payload["test"])


def primary_survival_metric(metrics: dict[str, float]) -> tuple[str, float]:
    """Select the case-level C-index used by the released survival tables."""
    if "case_c_index" not in metrics:
        raise KeyError("survival artifact has no 'case_c_index' metric")
    return "case_c_index", float(metrics["case_c_index"])


def load_runtime(run_dir: Path, *, required: bool) -> dict[str, object]:
    """Read profiler fields, requiring the artifact for profiler manifest rows."""
    path = run_dir / "runtime_profile.json"
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    phases = payload.get("phases", {})
    memory = payload.get("memory", {})
    peak_reserved_mb = memory.get("peak_reserved_mb")
    return {
        "peak_reserved_gib": (
            float(peak_reserved_mb) / 1024.0
            if peak_reserved_mb is not None
            else ""
        ),
        "train_wsi_per_second": phases.get("train", {}).get("slides_per_second_mean", ""),
        "test_wsi_per_second": phases.get("test", {}).get("slides_per_second_mean", ""),
    }


def sample_std(values: list[float]) -> float:
    """Use the same sample-standard-deviation convention as bundled tables."""
    return 0.0 if len(values) == 1 else float(np.std(values, ddof=1))


def _runtime_mean(rows: list[dict[str, object]], field: str) -> object:
    """Average a profiler field while preserving blank unavailable phases."""
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return float(np.mean(values)) if values else ""


def summarize_runtime(
    per_run: list[dict[str, object]],
    group_fields: list[str],
) -> list[dict[str, object]]:
    """Build per-setting profiles and the cross-cohort TCGA resource mean."""
    runtime_rows = [
        row for row in per_run if any(row.get(field, "") != "" for field in RUNTIME_FIELDS)
    ]
    if not runtime_rows:
        return []

    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in runtime_rows:
        key = tuple(str(row.get(field, "")) for field in group_fields)
        grouped[key].append(row)

    summary: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        item: dict[str, object] = dict(zip(group_fields, key))
        item.update({field: _runtime_mean(rows, field) for field in RUNTIME_FIELDS})
        item["runs"] = len(rows)
        summary.append(item)

    # The resource table compares a single PANDA profile with the arithmetic
    # mean of the five public TCGA cohorts. Derive that aggregate directly from
    # the collected rows so no manual spreadsheet step is required.
    if "dataset" in group_fields:
        non_dataset_fields = [field for field in group_fields if field != "dataset"]
        tcga_groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
        for row in runtime_rows:
            dataset = str(row.get("dataset", "")).upper().removeprefix("TCGA-")
            if dataset in SURVIVAL_DATASETS:
                key = tuple(str(row.get(field, "")) for field in non_dataset_fields)
                tcga_groups[key].append(row)
        for key, rows in sorted(tcga_groups.items()):
            cohorts = {
                str(row.get("dataset", "")).upper().removeprefix("TCGA-")
                for row in rows
            }
            if cohorts != SURVIVAL_DATASETS:
                # A partial smoke run should stay useful without being labeled
                # as the complete five-cohort resource mean.
                continue
            item = dict(zip(non_dataset_fields, key))
            item["dataset"] = "TCGA-5 mean"
            item.update({field: _runtime_mean(rows, field) for field in RUNTIME_FIELDS})
            item["runs"] = len(rows)
            summary.append(item)

    return summary


def summarize_metrics(
    per_metric: list[dict[str, object]],
    group_fields: list[str],
) -> list[dict[str, object]]:
    """Aggregate every numeric test metric across seeds in long form."""
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in per_metric:
        key = tuple(str(row.get(field, "")) for field in group_fields)
        grouped[key].append(row)

    summary: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        values = [float(row["value"]) for row in rows]
        item: dict[str, object] = dict(zip(group_fields, key))
        item.update({
            "mean": float(np.mean(values)),
            "std": sample_std(values),
            "runs": len(values),
        })
        summary.append(item)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Output root used by the selected manifest commands.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to results/rerun/<manifest stem>.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any selected run artifact is missing.",
    )
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="Skip manifest rows marked access=restricted.",
    )
    args = parser.parse_args()

    manifest_fields, manifest_rows = read_manifest(args.manifest)
    if args.public_only:
        # Missing access defaults to public for compatibility with custom
        # manifests that predate the explicit access column.
        manifest_rows = [
            row for row in manifest_rows if row.get("access", "public") == "public"
        ]
    group_fields = [field for field in manifest_fields if field not in MANIFEST_METADATA]
    out_dir = args.out_dir or Path("results/rerun") / args.manifest.stem
    per_run: list[dict[str, object]] = []
    per_metric: list[dict[str, object]] = []
    missing: list[str] = []

    for row in manifest_rows:
        run_dir = args.run_root / row["run_name"]
        try:
            if is_survival(row):
                metrics = load_survival_metrics(run_dir)
                metric, value = primary_survival_metric(metrics)
            else:
                dataset = row.get("dataset", "")
                metrics = load_classification_metrics(run_dir)
                metric, value = primary_classification_metric(metrics, dataset)
            # A profiler row is incomplete without its runtime artifact. Keep
            # that contract inside the same missing-artifact path as metrics,
            # so non-strict smoke collection skips partial runs and strict
            # collection reports them together.
            runtime = load_runtime(
                run_dir,
                required="--profile_runtime" in row.get("command", ""),
            )
        except (FileNotFoundError, KeyError, ValueError):
            missing.append(row["run_name"])
            continue

        record: dict[str, object] = {
            field: row[field] for field in group_fields if row.get(field, "") != ""
        }
        record.update({
            "seed": int(row["seed"]),
            "run_name": row["run_name"],
            "primary_metric": metric,
            "value": value,
        })
        record.update(runtime)
        per_run.append(record)
        for metric_name, metric_value in sorted(metrics.items()):
            metric_record: dict[str, object] = {
                field: row[field]
                for field in group_fields
                if row.get(field, "") != ""
            }
            metric_record.update({
                "seed": int(row["seed"]),
                "run_name": row["run_name"],
                "metric": metric_name,
                "value": metric_value,
            })
            per_metric.append(metric_record)

    if missing and args.strict:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"missing {len(missing)} run artifacts; first entries: {preview}")

    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in per_run:
        key = tuple(str(row.get(field, "")) for field in group_fields)
        grouped[key].append(row)

    summary: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        values = [float(row["value"]) for row in rows]
        item: dict[str, object] = dict(zip(group_fields, key))
        item.update({
            "primary_metric": rows[0]["primary_metric"],
            "mean": float(np.mean(values)),
            "std": sample_std(values),
            "runs": len(values),
        })
        summary.append(item)

    metric_group_fields = [field for field in group_fields if field != "metric"]
    metric_group_fields.append("metric")
    metric_summary = summarize_metrics(per_metric, metric_group_fields)

    per_fields = [*group_fields, "seed", "run_name", "primary_metric", "value"]
    per_fields.extend(field for field in RUNTIME_FIELDS if any(field in row for row in per_run))
    summary_fields = [*group_fields, "primary_metric", "mean", "std", "runs"]
    write_tsv(out_dir / "per_seed.tsv", per_run, per_fields)
    write_tsv(out_dir / "summary.tsv", summary, summary_fields)
    write_tsv(
        out_dir / "all_metrics_per_seed.tsv",
        per_metric,
        [*metric_group_fields[:-1], "seed", "run_name", "metric", "value"],
    )
    write_tsv(
        out_dir / "all_metrics_summary.tsv",
        metric_summary,
        [*metric_group_fields, "mean", "std", "runs"],
    )

    runtime_summary = summarize_runtime(per_run, group_fields)
    if runtime_summary:
        runtime_fields = [*group_fields, *RUNTIME_FIELDS, "runs"]
        write_tsv(out_dir / "runtime.tsv", runtime_summary, runtime_fields)

    print(f"collected {len(per_run)} runs from {args.manifest}")
    if missing:
        print(f"skipped {len(missing)} missing runs (use --strict to require all rows)")
    print(
        "wrote primary and all-metric seed/summary tables under "
        f"{out_dir}"
    )
    if runtime_summary:
        print(f"wrote {out_dir / 'runtime.tsv'}")


if __name__ == "__main__":
    main()
