#!/usr/bin/env python3
"""Run released command manifests row-by-row.

The manifests in ``configs/`` keep every evaluation run as an explicit shell
command. This wrapper adds named row selection and fail-fast execution while
leaving the actual trainer arguments visible and editable in the TSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


SELECTOR_FIELDS = (
    "dataset",
    "cohort",
    "method",
    "seed",
    "experiment",
    "task",
    "architecture",
    "variant",
    "encoder",
    "window",
    "parameterization",
    "arm",
    "endpoint",
    "method_label",
    "delta_scale",
    "gate_hidden_dim",
    "profile_epochs",
    "selection_status",
    "run_name",
)


def non_empty_selector(value: str) -> str:
    """Reject empty named selectors that would otherwise match nothing."""
    if not value:
        raise argparse.ArgumentTypeError("selector values cannot be empty")
    return value


def add_selector_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose manifest columns as conventional named command-line options."""
    for field in SELECTOR_FIELDS:
        option = f"--{field.replace('_', '-')}"
        parser.add_argument(
            option,
            dest=field,
            type=non_empty_selector,
            metavar="VALUE",
            help=f"Select rows whose {field} column exactly matches VALUE.",
        )


def selected_filters(args: argparse.Namespace) -> dict[str, str]:
    """Return only named selectors that the caller supplied."""
    return {
        field: value
        for field in SELECTOR_FIELDS
        if (value := getattr(args, field)) is not None
    }


def row_matches(row: dict[str, str], filters: dict[str, str]) -> bool:
    """Return True when all requested selectors match this TSV row."""
    return all(row.get(field) == value for field, value in filters.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="TSV file with a command column.")
    add_selector_arguments(parser)
    parser.add_argument("--limit", type=int, default=None,
                        help="Run at most this many matching rows.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selected commands without executing them.")
    parser.add_argument("--start-at", default=None,
                        help="Skip matching rows until this run_name is reached.")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")

    filters = selected_filters(args)
    rows: list[dict[str, str]] = []
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if "command" not in fieldnames:
            raise SystemExit(f"{args.manifest} is missing a command column")
        unavailable = sorted(set(filters).difference(fieldnames))
        if unavailable:
            options = ", ".join(f"--{field.replace('_', '-')}" for field in unavailable)
            raise SystemExit(
                f"{args.manifest} does not support selector(s): {options}"
            )
        for row in reader:
            if row_matches(row, filters):
                rows.append(row)

    if args.start_at is not None:
        for idx, row in enumerate(rows):
            if row.get("run_name") == args.start_at:
                rows = rows[idx:]
                break
        else:
            raise SystemExit(f"--start-at run_name not found after selection: {args.start_at}")

    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        print("No manifest rows selected.")
        return

    for idx, row in enumerate(rows, start=1):
        run_name = row.get("run_name", f"row_{idx}")
        command = row["command"]
        print(f"[{idx}/{len(rows)}] {run_name}")
        print(command)
        if args.dry_run:
            continue
        # Run through bash so manifest expressions such as ${VAR:?message}
        # fail early with an actionable missing-environment-variable message.
        completed = subprocess.run(
            ["bash", "-lc", f"set -euo pipefail\n{command}"],
            cwd=Path.cwd(),
            env=os.environ.copy(),
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(f"{run_name} failed with exit code {completed.returncode}")


if __name__ == "__main__":
    main()
