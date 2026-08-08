#!/usr/bin/env python3
"""Run public command manifests row-by-row.

The manifests in ``configs/`` keep every evaluation run as an explicit shell
command.  This wrapper adds light filtering and fail-fast execution while
leaving the actual trainer arguments visible and editable in the TSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


def parse_key_value(text: str) -> tuple[str, str]:
    """Parse ``column=value`` filters used to select manifest rows."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected column=value, got {text!r}")
    key, value = text.split("=", 1)
    if not key or not value:
        raise argparse.ArgumentTypeError(f"expected non-empty column=value, got {text!r}")
    return key, value


def row_matches(row: dict[str, str], filters: list[tuple[str, str]]) -> bool:
    """Return True when all requested filters match this TSV row."""
    for key, value in filters:
        if row.get(key) != value:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="TSV file with a command column.")
    parser.add_argument("--where", action="append", type=parse_key_value, default=[],
                        help="Filter rows by exact column match, e.g. --where dataset=cam16.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run at most this many matching rows.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selected commands without executing them.")
    parser.add_argument("--start-at", default=None,
                        help="Skip matching rows until this run_name is reached.")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")

    rows: list[dict[str, str]] = []
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "command" not in (reader.fieldnames or []):
            raise SystemExit(f"{args.manifest} is missing a command column")
        for row in reader:
            if row_matches(row, args.where):
                rows.append(row)

    if args.start_at is not None:
        for idx, row in enumerate(rows):
            if row.get("run_name") == args.start_at:
                rows = rows[idx:]
                break
        else:
            raise SystemExit(f"--start-at run_name not found after filtering: {args.start_at}")

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
