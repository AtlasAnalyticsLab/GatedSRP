#!/usr/bin/env python3
"""Validate AtlasPatch-style H5 embedding files before launching training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py


def validate_file(path: Path, feature_key: str, expected_dim: int | None) -> tuple[bool, str, int, int | None]:
    """Check only the fields consumed by the trainers."""
    try:
        with h5py.File(path, "r") as h5:
            if feature_key not in h5:
                return False, f"missing feature key {feature_key}", 0, None
            if "coords" not in h5:
                return False, "missing coords", 0, None
            feats = h5[feature_key]
            coords = h5["coords"]
            if len(feats.shape) != 2:
                return False, f"feature dataset is not 2D: shape={tuple(feats.shape)}", 0, None
            if coords.shape[0] != feats.shape[0]:
                return False, f"coords/features row mismatch {coords.shape[0]} != {feats.shape[0]}", int(feats.shape[0]), int(feats.shape[1])
            if feats.shape[0] <= 0:
                return False, "empty feature bag", 0, int(feats.shape[1])
            if expected_dim is not None and feats.shape[1] != expected_dim:
                return False, f"expected dim {expected_dim}, got {feats.shape[1]}", int(feats.shape[0]), int(feats.shape[1])
            return True, "ok", int(feats.shape[0]), int(feats.shape[1])
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", 0, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing H5 files.")
    parser.add_argument("--feature-key", required=True, help="H5 dataset key, e.g. features/uni_v2.")
    parser.add_argument("--expected-dim", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--bad-tsv", default=None)
    parser.add_argument("--summary-tsv", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    paths = sorted(root.glob("**/*.h5" if args.recursive else "*.h5"))
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit(f"no .h5 files found under {root}")

    bad_rows: list[dict[str, object]] = []
    n_ok = 0
    total_tokens = 0
    dims: set[int] = set()
    for path in paths:
        ok, reason, n_tokens, dim = validate_file(path, args.feature_key, args.expected_dim)
        if ok:
            n_ok += 1
            total_tokens += n_tokens
            if dim is not None:
                dims.add(dim)
        else:
            bad_rows.append({"path": str(path), "reason": reason, "n_tokens": n_tokens, "dim": "" if dim is None else dim})

    summary = {
        "root": str(root),
        "feature_key": args.feature_key,
        "expected_dim": "" if args.expected_dim is None else args.expected_dim,
        "scanned": len(paths),
        "ok": n_ok,
        "bad": len(bad_rows),
        "total_tokens_ok": total_tokens,
        "dims_ok": ",".join(str(x) for x in sorted(dims)),
    }

    if args.bad_tsv:
        out = Path(args.bad_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "reason", "n_tokens", "dim"], delimiter="\t")
            writer.writeheader()
            writer.writerows(bad_rows)
    if args.summary_tsv:
        out = Path(args.summary_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary), delimiter="\t")
            writer.writeheader()
            writer.writerow(summary)

    print(
        f"scanned={summary['scanned']} ok={summary['ok']} bad={summary['bad']} "
        f"dims={summary['dims_ok']} total_tokens_ok={summary['total_tokens_ok']}"
    )
    if bad_rows:
        raise SystemExit("embedding validation failed")


if __name__ == "__main__":
    main()
