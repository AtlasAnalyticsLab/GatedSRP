#!/usr/bin/env python
"""Stage already-downloaded TCGA SVS files for GatedSRP extraction."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable


TCGA_COHORTS = ("KIRC", "KIRP", "LUAD", "STAD", "UCEC")
SLIDE_EXTENSIONS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find TCGA slides that already exist on local/server storage and "
            "arrange them as A-TCGA-{cohort}/<slide>.svs for AtlasPatch."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        required=True,
        help="Existing TCGA slide file or directory. Repeat for multiple storage roots.",
    )
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=Path("data/labels/tcga_survival/all_matched_survival_labels_long.csv"),
    )
    parser.add_argument("--endpoint", default="OS")
    parser.add_argument(
        "--cohort",
        action="append",
        choices=TCGA_COHORTS,
        help="Repeat to stage selected cohorts. Defaults to all five TCGA cohorts.",
    )
    parser.add_argument(
        "--include-nonpositive-time",
        action="store_true",
        help="Keep rows flagged with non-positive survival time. Default matches training and drops them.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/raw/tcga-to-atlas"),
        help="Root that will contain A-TCGA-{cohort}/ slide folders.",
    )
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing", action="store_true", help="Write available links even if some slides are absent.")
    parser.add_argument("--follow-symlinks", action="store_true", help="Follow symlinked directories while scanning sources.")
    parser.add_argument(
        "--duplicate-policy",
        choices=["error", "first"],
        default="error",
        help="How to handle multiple source files with the same requested filename.",
    )
    parser.add_argument(
        "--missing-out",
        type=Path,
        default=Path("data/raw/tcga-to-atlas/tcga_existing_missing.tsv"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug helper: stage only the first N selected slides.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def canonical_cohort(value: str) -> str:
    text = value.strip().upper().replace("-", "_")
    if text.startswith("TCGA_"):
        text = text[5:]
    if text not in TCGA_COHORTS:
        raise ValueError(f"unsupported TCGA cohort: {value!r}")
    return text


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    cohorts = set(args.cohort or TCGA_COHORTS)
    endpoint = args.endpoint.strip().upper()
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with args.label_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cohort = canonical_cohort(row.get("cohort", ""))
            if cohort not in cohorts or row.get("endpoint", "").strip().upper() != endpoint:
                continue
            if not args.include_nonpositive_time and row.get("has_nonpositive_time", "").strip().lower() == "true":
                continue
            filename = Path(row.get("filename", "")).name
            if not filename or filename in seen:
                continue
            # Match the survival loader: one slide file per selected OS row,
            # with non-positive time rows removed unless explicitly requested.
            seen.add(filename)
            out = dict(row)
            out["cohort_short"] = cohort
            out["filename"] = filename
            rows.append(out)
    rows.sort(key=lambda r: (r["cohort_short"], r["filename"]))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no TCGA label rows selected")
    return rows


def iter_slide_files(sources: Iterable[Path], follow_symlinks: bool) -> Iterable[Path]:
    for source in sources:
        if source.is_file():
            if source.suffix.lower() in SLIDE_EXTENSIONS:
                yield source
            continue
        if not source.is_dir():
            raise FileNotFoundError(f"source path does not exist: {source}")
        for root, _, files in os.walk(source, followlinks=follow_symlinks):
            root_path = Path(root)
            for filename in files:
                path = root_path / filename
                if path.suffix.lower() in SLIDE_EXTENSIONS:
                    yield path


def build_filename_index(sources: list[Path], follow_symlinks: bool) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in iter_slide_files(sources, follow_symlinks):
        index.setdefault(path.name, []).append(path)
    return index


def replace_existing(path: Path, overwrite: bool, source: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    try:
        if path.resolve() == source.resolve():
            return True
    except FileNotFoundError:
        pass
    if not overwrite:
        raise FileExistsError(f"target exists: {path}; use --overwrite to replace")
    if path.is_dir() and not path.is_symlink():
        raise IsADirectoryError(f"refusing to replace directory target: {path}")
    path.unlink()
    return False


def write_missing(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cohort", "filename"], delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({"cohort": row["cohort_short"], "filename": row["filename"]})


def main() -> None:
    args = parse_args()
    rows = selected_rows(args)
    counts = Counter(row["cohort_short"] for row in rows)
    print(f"selected {len(rows)} TCGA {args.endpoint.upper()} slide files: {dict(sorted(counts.items()))}")

    index = build_filename_index(args.source, args.follow_symlinks)
    staged = 0
    already_ready = 0
    missing: list[dict[str, str]] = []
    duplicates: list[str] = []

    for row in rows:
        filename = row["filename"]
        matches = index.get(filename, [])
        if not matches:
            missing.append(row)
            continue
        if len(matches) > 1 and args.duplicate_policy == "error":
            duplicates.append(f"{filename}: {matches[:5]}")
            continue
        source = matches[0].resolve()
        target_dir = args.out_root / f"A-TCGA-{row['cohort_short']}"
        target = target_dir / filename
        if args.dry_run:
            print(f"{source} -> {target}")
            staged += 1
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        if replace_existing(target, args.overwrite, source):
            already_ready += 1
            continue
        if args.mode == "copy":
            shutil.copy2(source, target)
        else:
            # Symlinks avoid duplicating multi-terabyte TCGA slide collections
            # when the files already live on shared server storage.
            target.symlink_to(source)
        staged += 1

    if duplicates:
        raise SystemExit("duplicate source files found:\n" + "\n".join(duplicates[:20]))

    write_missing(args.missing_out, missing)
    print(f"indexed {len(index)} slide-like files from {len(args.source)} source root(s)")
    print(f"{args.mode}ed {staged} TCGA slides into {args.out_root}; already_ready={already_ready}")
    if missing:
        print(f"missing {len(missing)} requested slides; see {args.missing_out}")
        if not args.allow_missing:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
