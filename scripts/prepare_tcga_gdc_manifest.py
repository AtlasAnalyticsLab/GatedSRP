#!/usr/bin/env python
"""Create a GDC download manifest for the TCGA slides used by GatedSRP."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable


GDC_API = "https://api.gdc.cancer.gov"
TCGA_COHORTS = ("KIRC", "KIRP", "LUAD", "STAD", "UCEC")
FIELDS = (
    "file_id",
    "file_name",
    "md5sum",
    "file_size",
    "state",
    "data_type",
    "data_format",
    "data_category",
    "access",
    "cases.submitter_id",
    "cases.project.project_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query the GDC files API for the TCGA SVS slide filenames listed in "
            "data/labels/tcga_survival/all_matched_survival_labels_long.csv and "
            "write a gdc-client manifest."
        )
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
        help="Repeat to select cohorts. Defaults to all five released TCGA cohorts.",
    )
    parser.add_argument(
        "--include-nonpositive-time",
        action="store_true",
        help="Keep rows flagged with non-positive survival time. Default matches the trainer and drops them.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/raw/tcga-to-atlas/gdc_manifest_tcga_os.tsv"),
        help="Output GDC manifest path for gdc-client.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("data/raw/tcga-to-atlas/gdc_slide_metadata_tcga_os.tsv"),
        help="Output slide-to-GDC metadata table used by the organizer script.",
    )
    parser.add_argument(
        "--missing-out",
        type=Path,
        default=Path("data/raw/tcga-to-atlas/gdc_missing_tcga_os.tsv"),
        help="Output table for requested filenames not found through the GDC API.",
    )
    parser.add_argument("--api", default=GDC_API)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional seconds between API chunks.")
    parser.add_argument("--limit", type=int, default=None, help="Debug helper: query only the first N slides.")
    parser.add_argument("--allow-missing", action="store_true", help="Write outputs even if some slides are missing.")
    parser.add_argument("--summary-only", action="store_true", help="Print selected slide counts without calling GDC.")
    return parser.parse_args()


def canonical_cohort(value: str) -> str:
    text = value.strip().upper().replace("-", "_")
    if text.startswith("TCGA_"):
        text = text[5:]
    if text not in TCGA_COHORTS:
        raise ValueError(f"unsupported TCGA cohort: {value!r}")
    return text


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    """Read label rows that match the trainer's survival inclusion rule."""
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
            # The survival CSV is a label table, not a download manifest. Keep
            # one manifest row per SVS filename so multi-row label extensions do
            # not make gdc-client download the same slide twice.
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


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("--chunk-size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def gdc_query(api: str, filenames: list[str], projects: list[str]) -> list[dict]:
    """Query one batch of filenames from the GDC files endpoint."""
    # The complete TCGA OS cohort contains long SVS filenames. Sending a large
    # batch through query-string GET can exceed common URL limits, so the same
    # public GDC Files API request is submitted as form-encoded POST data.
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "file_name", "value": filenames}},
            {"op": "in", "content": {"field": "cases.project.project_id", "value": projects}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
        ],
    }
    params = urllib.parse.urlencode(
        {
            "filters": json.dumps(filters),
            "fields": ",".join(FIELDS),
            "format": "JSON",
            "size": str(max(1, len(filenames) * 2)),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{api.rstrip('/')}/files",
        data=params,
        headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return payload.get("data", {}).get("hits", [])


def resolve_gdc_files(args: argparse.Namespace, rows: list[dict[str, str]]) -> tuple[dict[str, dict], list[str]]:
    """Map requested filenames to exactly one GDC file record each."""
    requested = [row["filename"] for row in rows]
    projects = sorted({f"TCGA-{row['cohort_short']}" for row in rows})
    hits_by_name: dict[str, list[dict]] = {}
    for batch in chunks(requested, args.chunk_size):
        for hit in gdc_query(args.api, batch, projects):
            hits_by_name.setdefault(hit["file_name"], []).append(hit)
        if args.sleep:
            time.sleep(args.sleep)

    resolved: dict[str, dict] = {}
    missing: list[str] = []
    duplicate_messages: list[str] = []
    for filename in requested:
        hits = hits_by_name.get(filename, [])
        if len(hits) == 1:
            resolved[filename] = hits[0]
        elif not hits:
            missing.append(filename)
        else:
            duplicate_messages.append(f"{filename}: {len(hits)} GDC hits")

    if duplicate_messages:
        # A duplicate hit means the script cannot prove which GDC UUID matches
        # the reported slide. Fail closed instead of silently choosing one.
        raise SystemExit("duplicate GDC records found:\n" + "\n".join(duplicate_messages[:20]))
    return resolved, missing


def _case_project(hit: dict) -> str:
    cases = hit.get("cases") or []
    if not cases:
        return ""
    return (cases[0].get("project") or {}).get("project_id", "")


def _case_submitter(hit: dict) -> str:
    cases = hit.get("cases") or []
    if not cases:
        return ""
    return cases[0].get("submitter_id", "")


def write_outputs(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    resolved: dict[str, dict],
    missing: list[str],
) -> None:
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.missing_out.parent.mkdir(parents=True, exist_ok=True)

    with args.manifest_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "filename", "md5", "size", "state"], delimiter="\t")
        writer.writeheader()
        for filename in sorted(resolved):
            hit = resolved[filename]
            writer.writerow(
                {
                    "id": hit["file_id"],
                    "filename": hit["file_name"],
                    "md5": hit.get("md5sum", ""),
                    "size": hit.get("file_size", ""),
                    "state": hit.get("state", ""),
                }
            )

    row_by_filename = {row["filename"]: row for row in rows}
    metadata_fields = [
        "cohort",
        "endpoint",
        "case_barcode",
        "filename",
        "gdc_file_id",
        "gdc_project",
        "gdc_case_submitter_id",
        "access",
        "state",
        "md5sum",
        "file_size",
        "data_type",
        "data_format",
    ]
    with args.metadata_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields, delimiter="\t")
        writer.writeheader()
        for filename in sorted(resolved):
            row = row_by_filename[filename]
            hit = resolved[filename]
            writer.writerow(
                {
                    "cohort": row["cohort_short"],
                    "endpoint": row.get("endpoint", ""),
                    "case_barcode": row.get("case_barcode", ""),
                    "filename": filename,
                    "gdc_file_id": hit["file_id"],
                    "gdc_project": _case_project(hit),
                    "gdc_case_submitter_id": _case_submitter(hit),
                    "access": hit.get("access", ""),
                    "state": hit.get("state", ""),
                    "md5sum": hit.get("md5sum", ""),
                    "file_size": hit.get("file_size", ""),
                    "data_type": hit.get("data_type", ""),
                    "data_format": hit.get("data_format", ""),
                }
            )

    with args.missing_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename"], delimiter="\t")
        writer.writeheader()
        for filename in missing:
            writer.writerow({"filename": filename})


def main() -> None:
    args = parse_args()
    rows = selected_rows(args)
    counts = Counter(row["cohort_short"] for row in rows)
    print(f"selected {len(rows)} TCGA {args.endpoint.upper()} slide files: {dict(sorted(counts.items()))}")
    if args.summary_only:
        return

    resolved, missing = resolve_gdc_files(args, rows)
    if missing and not args.allow_missing:
        print(f"missing {len(missing)} requested files; see {args.missing_out}", file=sys.stderr)
        write_outputs(args, rows, resolved, missing)
        raise SystemExit(2)
    write_outputs(args, rows, resolved, missing)
    total_bytes = sum(int(hit.get("file_size") or 0) for hit in resolved.values())
    total_gib = total_bytes / (1024**3)
    print(f"wrote manifest: {args.manifest_out} ({len(resolved)} files)")
    print(f"total download size: {total_gib:.2f} GiB")
    print(f"wrote metadata: {args.metadata_out}")
    if missing:
        print(f"wrote missing list: {args.missing_out} ({len(missing)} files)")


if __name__ == "__main__":
    main()
