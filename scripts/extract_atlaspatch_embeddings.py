#!/usr/bin/env python
"""Dataset-aware AtlasPatch extraction launcher for GatedSRP.

The trainers expect AtlasPatch-style H5 files with row-aligned ``coords`` and
``features/<encoder>`` datasets.  This script keeps the AtlasPatch call itself
simple, but routes each dataset to the directory layout used by the released
manifests so users do not have to reverse-engineer H5 locations.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import subprocess
from pathlib import Path


SLIDE_EXTENSIONS = (
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".bif",
    ".dcm",
    ".vms",
    ".vmu",
)
KGH_CLASSES = ("CP_HP", "CP_SSL", "CP_TA", "CP_TVA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate GatedSRP-compatible H5 patch embeddings with AtlasPatch.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["generic", "camelyon16", "camelyon17", "kgh", "panda", "bracs", "tcga"],
        help="Dataset layout preset. Use 'generic' for a flat WSI directory.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Raw WSI directory or a single WSI file. This can be any local/server path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Dataset feature root. The script appends dataset-specific subfolders when needed.",
    )
    parser.add_argument(
        "--encoder",
        default="uni_v2",
        help="AtlasPatch patch feature extractor name. The reported main runs use uni_v2.",
    )
    parser.add_argument("--cohort", choices=["KIRC", "KIRP", "LUAD", "STAD", "UCEC"])
    parser.add_argument("--label-csv", type=Path, help="Optional label manifest used to stage selected slides.")
    parser.add_argument("--cam16-normal-dir", type=Path, help="Existing CAMELYON16 normal-slide directory.")
    parser.add_argument("--cam16-tumor-dir", type=Path, help="Existing CAMELYON16 tumor-slide directory.")
    parser.add_argument("--mpp-csv", type=Path, help="Optional AtlasPatch MPP CSV.")
    parser.add_argument("--feature-plugin", action="append", default=[], help="Optional AtlasPatch feature plugin path.")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--target-mag", type=int, default=20)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--feature-device", default=os.environ.get("FEATURE_DEVICE"))
    parser.add_argument("--feature-batch-size", type=int, default=int(os.environ.get("FEATURE_BATCH_SIZE", "24")))
    parser.add_argument("--feature-num-workers", type=int, default=int(os.environ.get("FEATURE_NUM_WORKERS", "2")))
    parser.add_argument("--patch-workers", type=int, default=int(os.environ.get("PATCH_WORKERS", "2")))
    parser.add_argument("--seg-batch-size", type=int, default=None)
    parser.add_argument("--max-open-slides", type=int, default=None)
    parser.add_argument("--feature-precision", default=os.environ.get("FEATURE_PRECISION", "float16"))
    parser.add_argument("--force", action="store_true", help="Pass AtlasPatch --force and rebuild existing H5 files.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching AtlasPatch.")
    return parser.parse_args()


def _single_encoder(args: argparse.Namespace) -> str:
    """Return the single encoder used for directory-family presets."""
    encoders = [item.strip() for item in args.encoder.replace(",", " ").split() if item.strip()]
    if len(encoders) != 1:
        raise SystemExit(
            "Dataset presets require exactly one --encoder so the output folder "
            f"can be named unambiguously; got {args.encoder!r}."
        )
    return encoders[0]


def _stage_kgh_inputs(raw_root: Path, label_csv: Path | None, staging_dir: Path) -> Path:
    """Build a flat symlink directory for the KGH slides used by the task."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    if label_csv is None:
        for source_split in ("train", "test"):
            for class_name in KGH_CLASSES:
                class_dir = raw_root / source_split / class_name
                if not class_dir.is_dir():
                    continue
                for source in sorted(class_dir.iterdir()):
                    if source.is_file() and source.suffix.lower() in SLIDE_EXTENSIONS:
                        target = staging_dir / source.name
                        if not target.exists():
                            target.symlink_to(source)
        return staging_dir

    with label_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"slide_id", "source_split", "class_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"KGH label CSV missing required columns: {sorted(missing)}")
        for row in reader:
            slide_id = row["slide_id"].strip()
            class_name = row["class_name"].strip()
            source_split = row["source_split"].strip()
            source = _find_slide(raw_root / source_split / class_name, slide_id)
            target = staging_dir / source.name
            if not target.exists():
                target.symlink_to(source)
    return staging_dir


def _find_slide(directory: Path, slide_id: str) -> Path:
    """Find a slide by stem and supported WSI extension."""
    for suffix in SLIDE_EXTENSIONS:
        candidate = directory / f"{slide_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find slide {slide_id!r} under {directory}")


def _jobs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    """Return ``(input_path, atlaspatch_output_root)`` jobs for the preset."""
    if args.dataset == "camelyon16":
        encoder = _single_encoder(args)
        geom = f"{args.target_mag}x_{args.patch_size}"
        normal_dir = args.cam16_normal_dir or args.input / "normal"
        tumor_dir = args.cam16_tumor_dir or args.input / "tumor"
        return [
            (normal_dir, args.output / "normal" / encoder / geom),
            (tumor_dir, args.output / "tumor" / encoder / geom),
        ]
    if args.dataset == "tcga":
        if not args.cohort:
            raise SystemExit("--dataset tcga requires --cohort")
        encoder = _single_encoder(args)
        geom = f"{args.target_mag}x_{args.patch_size}"
        return [(args.input, args.output / f"A-TCGA-{args.cohort}" / "40x" / encoder / geom)]
    if args.dataset == "kgh":
        label_csv = args.label_csv
        staging = args.output / ".atlaspatch_inputs" / "kgh"
        if args.dry_run:
            return [(staging, args.output)]
        return [(_stage_kgh_inputs(args.input, label_csv, staging), args.output)]
    return [(args.input, args.output)]


def _atlaspatch_command(args: argparse.Namespace, input_path: Path, output_path: Path) -> list[str]:
    """Build the AtlasPatch CLI command for one extraction job."""
    cmd = [
        "atlaspatch",
        "process",
        str(input_path),
        "--output",
        str(output_path),
        "--patch-size",
        str(args.patch_size),
        "--target-mag",
        str(args.target_mag),
        "--feature-extractors",
        args.encoder,
        "--device",
        args.device,
        "--feature-device",
        args.feature_device or args.device,
        "--feature-batch-size",
        str(args.feature_batch_size),
        "--feature-num-workers",
        str(args.feature_num_workers),
        "--patch-workers",
        str(args.patch_workers),
        "--feature-precision",
        args.feature_precision,
    ]
    for plugin in args.feature_plugin:
        cmd.extend(["--feature-plugin", plugin])
    if args.seg_batch_size is not None:
        cmd.extend(["--seg-batch-size", str(args.seg_batch_size)])
    if args.max_open_slides is not None:
        cmd.extend(["--max-open-slides", str(args.max_open_slides)])
    if args.mpp_csv is not None:
        cmd.extend(["--mpp-csv", str(args.mpp_csv)])
    if args.force:
        cmd.append("--force")
    return cmd


def main() -> None:
    args = parse_args()
    if shutil.which("atlaspatch") is None and not args.dry_run:
        raise SystemExit(
            "atlaspatch CLI not found. Install it with "
            "`python -m pip install atlas-patch` and install SAM2 as required by AtlasPatch."
        )

    for input_path, output_path in _jobs(args):
        if not input_path.exists() and not args.dry_run:
            raise SystemExit(f"missing input path: {input_path}")
        if not args.dry_run:
            output_path.mkdir(parents=True, exist_ok=True)
        cmd = _atlaspatch_command(args, input_path, output_path)
        print(shlex.join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
