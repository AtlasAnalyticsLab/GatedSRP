"""CAMELYON16 UNI-v2 slide adapter for the SRP TransMIL trainer.

This is intentionally separate from :mod:`data_cam16`, which is the
historical 768-d ViT-B/16 adapter. The released runs use 1536-d UNI-v2
embeddings under ``data/features/camelyon16`` by default, or the path
provided in ``CAM16_UNIV2_ROOT``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from slide_level_srp.data_ext import build_neighbor_graph, compute_h_morph
from slide_level_srp.data_cam16 import (
    Cam16FoldAssignment,
    build_cam16_fold_assignments,
    filter_records_cam16,
)


CAM16_UNIV2_ROOT = os.environ.get(
    "CAM16_UNIV2_ROOT",
    "data/features/camelyon16",
)
CAM16_LABEL_CSV = os.environ.get("CAM16_LABEL_CSV", "data/labels/camelyon16/slides.csv")
CAM16_UNIV2_FEATURE_KEY = os.environ.get("CAM16_UNIV2_FEATURE_KEY", "features/uni_v2")
CAM16_UNIV2_DIM = 1536
_N_MAX_SAFETY_CAM16_UNIV2 = 16384


@dataclass(frozen=True)
class Cam16UniV2SlideRecord:
    slide_id: str
    patient_id: str
    center: int
    label: int
    h5_path: str


def enumerate_cam16_univ2_slides(
    feature_root: str = CAM16_UNIV2_ROOT,
    *,
    require_both_classes: bool = True,
    feature_family: str = "uni_v2",
    label_csv: str = CAM16_LABEL_CSV,
) -> List[Cam16UniV2SlideRecord]:
    """Enumerate labeled normal/tumor H5 files for one feature family.

    The release path reads ``data/labels/camelyon16/slides.csv`` so the binary
    labels are explicit.  Feature H5s remain organized by class directory to
    match the AtlasPatch extraction script.  When the label CSV is absent, the
    function falls back to the historical class-directory inventory.
    """
    root = Path(feature_root)
    family = feature_family.strip("/")
    label_path = Path(label_csv)
    if label_path.exists():
        return _enumerate_cam16_univ2_from_label_csv(
            label_path=label_path,
            feature_root=root,
            feature_family=family,
            require_both_classes=require_both_classes,
        )

    out: List[Cam16UniV2SlideRecord] = []
    class_counts: dict[str, int] = {}
    for label_int, subdir in ((1, "tumor"), (0, "normal")):
        h5_dir = root / subdir / family / "20x_256" / "patches"
        if not h5_dir.is_dir():
            if require_both_classes:
                raise FileNotFoundError(f"missing CAM16 {family} class dir: {h5_dir}")
            class_counts[subdir] = 0
            continue
        paths = sorted(h5_dir.glob("*.h5"))
        class_counts[subdir] = len(paths)
        for p in paths:
            slide_id = p.stem
            out.append(Cam16UniV2SlideRecord(
                slide_id=slide_id,
                patient_id=slide_id,
                center=0,
                label=label_int,
                h5_path=str(p),
            ))
    if require_both_classes and any(class_counts.get(c, 0) == 0 for c in ("tumor", "normal")):
        raise RuntimeError(
            f"CAM16 {family} inventory incomplete: "
            f"tumor={class_counts.get('tumor', 0)} normal={class_counts.get('normal', 0)}. "
            "Do not launch CAM16 variants until both labeled classes are present."
        )
    out.sort(key=lambda r: r.slide_id)
    return out


def _enumerate_cam16_univ2_from_label_csv(
    *,
    label_path: Path,
    feature_root: Path,
    feature_family: str,
    require_both_classes: bool,
) -> List[Cam16UniV2SlideRecord]:
    """Read the released CAMELYON16 label manifest and join it to class H5s."""
    out: list[Cam16UniV2SlideRecord] = []
    class_counts: dict[str, int] = {"tumor": 0, "normal": 0}
    seen: set[str] = set()
    class_to_label = {"normal": 0, "tumor": 1}
    with label_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"slide_id", "class_name", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"CAM16 label CSV missing required columns: {sorted(missing)}")
        for row in reader:
            slide_id = row.get("slide_id", "").strip()
            class_name = row.get("class_name", "").strip().lower()
            if not slide_id:
                continue
            if slide_id in seen:
                raise RuntimeError(f"duplicate CAM16 slide_id in label CSV: {slide_id}")
            seen.add(slide_id)
            if class_name not in class_to_label:
                raise RuntimeError(
                    f"CAM16 slide {slide_id} has unsupported class {class_name!r}; "
                    "expected 'normal' or 'tumor'"
                )
            label = int(row.get("label", class_to_label[class_name]))
            if label != class_to_label[class_name]:
                raise RuntimeError(
                    f"CAM16 slide {slide_id} label={label} disagrees with "
                    f"class_name={class_name!r}"
                )
            h5_path = (
                feature_root
                / class_name
                / feature_family
                / "20x_256"
                / "patches"
                / f"{slide_id}.h5"
            )
            if not h5_path.exists():
                continue
            class_counts[class_name] += 1
            out.append(Cam16UniV2SlideRecord(
                slide_id=slide_id,
                patient_id=slide_id,
                center=0,
                label=label,
                h5_path=str(h5_path),
            ))
    if require_both_classes and any(class_counts.get(c, 0) == 0 for c in ("tumor", "normal")):
        raise RuntimeError(
            f"CAM16 {feature_family} inventory incomplete after label join: "
            f"tumor={class_counts.get('tumor', 0)} normal={class_counts.get('normal', 0)}"
        )
    out.sort(key=lambda r: r.slide_id)
    return out


class Cam16UniV2SlideDataset(Dataset):
    """Emit the same batch schema as CAM17 SRP datasets, using UNI-v2 H5s."""

    def __init__(
        self,
        records: Sequence[Cam16UniV2SlideRecord],
        subsample_cap: Optional[int] = _N_MAX_SAFETY_CAM16_UNIV2,
        feature_key: str = CAM16_UNIV2_FEATURE_KEY,
        expected_dim: int = CAM16_UNIV2_DIM,
        neighbor_radius: int = 1,
        neighbor_shell: str = "cumulative",
        neighbor_source: str = "real",
        neighbor_shuffle_seed: int = 0,
        neighbor_weighting: str = "uniform",
        neighbor_weight_sigma: float = 1.0,
    ) -> None:
        self.records = list(records)
        self.subsample_cap = subsample_cap
        self.feature_key = feature_key
        self.expected_dim = int(expected_dim)
        self.neighbor_radius = int(neighbor_radius)
        self.neighbor_shell = neighbor_shell
        self.neighbor_source = neighbor_source
        self.neighbor_shuffle_seed = int(neighbor_shuffle_seed)
        self.neighbor_weighting = neighbor_weighting
        self.neighbor_weight_sigma = float(neighbor_weight_sigma)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        import h5py

        r = self.records[idx]
        with h5py.File(r.h5_path, "r") as f:
            feats = np.asarray(f[self.feature_key][:], dtype=np.float32)
            coords = np.asarray(f["coords"][:], dtype=np.int64)
            stride = int(f.attrs.get("patch_size_level0", 512))
        if feats.ndim != 2 or feats.shape[1] != self.expected_dim:
            got = feats.shape[1] if feats.ndim == 2 else "non-2D"
            raise ValueError(
                f"{r.h5_path}: expected {self.expected_dim}-d features at "
                f"{self.feature_key}, got {got}"
            )
        if self.subsample_cap is not None and feats.shape[0] > self.subsample_cap:
            # Deterministic coordinate-spaced cap keeps train/eval views
            # identical and avoids injecting augmentation into variants.
            order = np.lexsort((coords[:, 1], coords[:, 0]))
            keep = np.linspace(0, feats.shape[0] - 1, num=self.subsample_cap, dtype=np.int64)
            keep_idx = order[keep]
            feats = feats[keep_idx]
            coords = coords[keep_idx]

        coords_2d = coords[:, :2].astype(np.int64)
        nbi, nbm, nbw = build_neighbor_graph(
            coords_2d,
            stride=stride,
            radius=self.neighbor_radius,
            shell=self.neighbor_shell,
            source=self.neighbor_source,
            shuffle_seed=self.neighbor_shuffle_seed + idx,
            weighting=self.neighbor_weighting,
            weight_sigma=self.neighbor_weight_sigma,
        )
        h_morph = compute_h_morph(feats, nbi, nbm)
        return {
            "features": torch.from_numpy(feats),
            "coords": torch.from_numpy(coords_2d),
            "neighbor_index": torch.from_numpy(nbi),
            "neighbor_mask": torch.from_numpy(nbm),
            "neighbor_weight": torch.from_numpy(nbw),
            "h_morph": torch.from_numpy(h_morph),
            "label": torch.tensor(r.label, dtype=torch.int64),
            "slide_id": r.slide_id,
            "patient_id": r.patient_id,
            "center": r.center,
            "n_tokens": int(feats.shape[0]),
        }


def cam16_univ2_slide_collate(batch: List[Dict]) -> Dict:
    if len(batch) != 1:
        raise ValueError(f"cam16_univ2_slide_collate expects batch_size=1, got {len(batch)}")
    b = batch[0]
    return {
        "features": b["features"].unsqueeze(0),
        "coords": b["coords"].unsqueeze(0),
        "neighbor_index": b["neighbor_index"].unsqueeze(0),
        "neighbor_mask": b["neighbor_mask"].unsqueeze(0),
        "neighbor_weight": b["neighbor_weight"].unsqueeze(0),
        "h_morph": b["h_morph"].unsqueeze(0),
        "label": b["label"].unsqueeze(0),
        "slide_id": [b["slide_id"]],
        "patient_id": [b["patient_id"]],
        "center": [b["center"]],
        "n_tokens": [b["n_tokens"]],
    }


def build_cam16_univ2_loaders_for_fold(
    records: Sequence[Cam16UniV2SlideRecord],
    fold: Cam16FoldAssignment,
    num_workers: int = 2,
    subsample_cap: Optional[int] = _N_MAX_SAFETY_CAM16_UNIV2,
    train_cap: Optional[int] = None,
    val_cap: Optional[int] = None,
    test_cap: Optional[int] = None,
    feature_key: str = CAM16_UNIV2_FEATURE_KEY,
    expected_dim: int = CAM16_UNIV2_DIM,
    neighbor_radius: int = 1,
    neighbor_shell: str = "cumulative",
    neighbor_source: str = "real",
    neighbor_shuffle_seed: int = 0,
    neighbor_weighting: str = "uniform",
    neighbor_weight_sigma: float = 1.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    eff_train_cap = train_cap if train_cap is not None else subsample_cap
    eff_val_cap = val_cap if val_cap is not None else subsample_cap
    eff_test_cap = test_cap if test_cap is not None else subsample_cap
    ds_kwargs = dict(
        feature_key=feature_key,
        expected_dim=expected_dim,
        neighbor_radius=neighbor_radius,
        neighbor_shell=neighbor_shell,
        neighbor_source=neighbor_source,
        neighbor_shuffle_seed=neighbor_shuffle_seed,
        neighbor_weighting=neighbor_weighting,
        neighbor_weight_sigma=neighbor_weight_sigma,
    )
    train_ds = Cam16UniV2SlideDataset(
        filter_records_cam16(records, fold.train_patients),
        subsample_cap=eff_train_cap,
        **ds_kwargs,
    )
    val_ds = Cam16UniV2SlideDataset(
        filter_records_cam16(records, fold.val_patients),
        subsample_cap=eff_val_cap,
        **ds_kwargs,
    )
    test_ds = Cam16UniV2SlideDataset(
        filter_records_cam16(records, fold.test_patients),
        subsample_cap=eff_test_cap,
        **ds_kwargs,
    )
    common = dict(num_workers=num_workers, collate_fn=cam16_univ2_slide_collate, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False, **common),
        DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False, **common),
    )


__all__ = [
    "CAM16_UNIV2_ROOT",
    "CAM16_UNIV2_FEATURE_KEY",
    "CAM16_UNIV2_DIM",
    "Cam16UniV2SlideRecord",
    "enumerate_cam16_univ2_slides",
    "build_cam16_fold_assignments",
    "build_cam16_univ2_loaders_for_fold",
]
