#!/usr/bin/env python
"""Arrange gdc-client TCGA slide downloads into cohort directories."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create data/raw/tcga-to-atlas/A-TCGA-{cohort}/<slide>.svs links "
            "from a gdc-client download directory."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/tcga-to-atlas/gdc_slide_metadata_tcga_os.tsv"),
        help="Metadata TSV written by prepare_tcga_gdc_manifest.py.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/raw/tcga-to-atlas/gdc-client-downloads"),
        help="Directory passed to gdc-client with -d/--dir.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/raw/tcga-to-atlas"),
        help="Root that will contain A-TCGA-{cohort}/ slide folders.",
    )
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _find_downloaded_slide(download_dir: Path, file_id: str, filename: str) -> Path:
    """Find a gdc-client output file by the common UUID folder or fallback scan."""
    direct = download_dir / file_id / filename
    if direct.exists():
        return direct
    matches = list(download_dir.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"downloaded slide not found: {filename}")
    raise RuntimeError(f"multiple downloaded slides named {filename}: {matches[:5]}")


def _replace_existing(path: Path, overwrite: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"target exists: {path}; use --overwrite to replace")
    if path.is_dir() and not path.is_symlink():
        raise IsADirectoryError(f"refusing to replace directory target: {path}")
    path.unlink()


def main() -> None:
    args = parse_args()
    linked = 0
    with args.metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cohort", "filename", "gdc_file_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"metadata missing required columns: {sorted(missing)}")
        for row in reader:
            cohort = row["cohort"].strip()
            filename = row["filename"].strip()
            file_id = row["gdc_file_id"].strip()
            source = _find_downloaded_slide(args.download_dir, file_id, filename).resolve()
            target_dir = args.out_root / f"A-TCGA-{cohort}"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / filename
            _replace_existing(target, args.overwrite)
            if args.mode == "copy":
                shutil.copy2(source, target)
            else:
                # Symlinks keep the AtlasPatch-facing layout without duplicating
                # the roughly 2 TB TCGA download tree on local storage.
                os.symlink(source, target)
            linked += 1
    print(f"{args.mode}ed {linked} TCGA slides into {args.out_root}")


if __name__ == "__main__":
    main()
