"""CAMELYON17 UNI-v2 slide adapter for reproducibility SRP runs.

The historical CAM17 path in :mod:`slide_level.src.data` points at UNI-v1
1024-d features under ``20x_256px_0px_overlap/features_uni_v1``.  The
reported classification runs use the consistent UNI-v2, 20x, 256-pixel embedding
family once extraction has completed.  This adapter keeps CAM17's official
patient/group fold logic and label mapping, but requires callers to opt into a
separate dataset name so no run can silently mix feature families.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from slide_level.src.data import (
    FoldAssignment,
    N_MAX_SAFETY,
    SlideRecord,
    build_fold_assignments,
    enumerate_slides,
    filter_records,
)
from slide_level_srp.data_ext import build_neighbor_graph, compute_h_morph


CAM17_UNIV2_ROOT = os.environ.get(
    "CAM17_UNIV2_ROOT",
    "data/features/camelyon17/patches",
)
CAM17_STAGES_CSV = os.environ.get("CAM17_STAGES_CSV", "data/labels/camelyon17/stages.csv")
CAM17_UNIV2_FEATURE_KEY = os.environ.get("CAM17_UNIV2_FEATURE_KEY", "features/uni_v2")
CAM17_UNIV2_DIM = 1536


def enumerate_cam17_univ2_slides(
    feature_root: str = CAM17_UNIV2_ROOT,
    *,
    num_classes: int = 4,
    csv_path: str = CAM17_STAGES_CSV,
) -> List[SlideRecord]:
    """Join CAM17 labels with the reported UNI-v2 feature inventory.

    We intentionally reuse CAM17's existing ``enumerate_slides`` helper.  That
    keeps the official label mapping and the documented missing-slide behavior
    identical to the legacy adapter while changing only the H5 root.
    """
    return enumerate_slides(
        csv_path=csv_path,
        feature_root=feature_root,
        num_classes=num_classes,
    )


class Cam17UniV2SlideDataset(Dataset):
    """Emit one CAM17 UNI-v2 WSI bag in the schema expected by ``train.py``."""

    def __init__(
        self,
        records: Sequence[SlideRecord],
        subsample_cap: Optional[int] = N_MAX_SAFETY,
        feature_key: str = CAM17_UNIV2_FEATURE_KEY,
        expected_dim: int = CAM17_UNIV2_DIM,
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

        record = self.records[idx]
        with h5py.File(record.h5_path, "r") as h5:
            if self.feature_key not in h5:
                raise KeyError(f"{record.h5_path}: missing feature key {self.feature_key}")
            feats = np.asarray(h5[self.feature_key][:], dtype=np.float32)
            coords = np.asarray(h5["coords"][:], dtype=np.int64)
            stride = int(h5.attrs.get("patch_size_level0", 512))

        if feats.ndim != 2 or feats.shape[1] != self.expected_dim:
            got = feats.shape[1] if feats.ndim == 2 else "non-2D"
            raise ValueError(
                f"{record.h5_path}: expected {self.expected_dim}-d features at "
                f"{self.feature_key}, got {got}"
            )
        if coords.ndim != 2 or coords.shape[0] != feats.shape[0] or coords.shape[1] < 2:
            raise ValueError(
                f"{record.h5_path}: coords shape {coords.shape} is incompatible with "
                f"features shape {feats.shape}"
            )
        if feats.shape[0] <= 0:
            raise ValueError(f"{record.h5_path}: empty CAM17 UNI-v2 feature bag")

        if self.subsample_cap is not None and feats.shape[0] > self.subsample_cap:
            # Deterministic coordinate-spaced safety cap: keeps all variant
            # arms paired and avoids introducing a stochastic patch sampler.
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
            "label": torch.tensor(record.label, dtype=torch.int64),
            "slide_id": record.slide_id,
            "patient_id": record.patient_id,
            "center": record.center,
            "stage": record.stage,
            "n_tokens": int(feats.shape[0]),
        }


def cam17_univ2_slide_collate(batch: List[Dict]) -> Dict:
    if len(batch) != 1:
        raise ValueError(f"cam17_univ2_slide_collate expects batch_size=1, got {len(batch)}")
    item = batch[0]
    return {
        "features": item["features"].unsqueeze(0),
        "coords": item["coords"].unsqueeze(0),
        "neighbor_index": item["neighbor_index"].unsqueeze(0),
        "neighbor_mask": item["neighbor_mask"].unsqueeze(0),
        "neighbor_weight": item["neighbor_weight"].unsqueeze(0),
        "h_morph": item["h_morph"].unsqueeze(0),
        "label": item["label"].unsqueeze(0),
        "slide_id": [item["slide_id"]],
        "patient_id": [item["patient_id"]],
        "center": [item["center"]],
        "stage": [item["stage"]],
        "n_tokens": [item["n_tokens"]],
    }


def build_cam17_univ2_loaders_for_fold(
    records: Sequence[SlideRecord],
    fold: FoldAssignment,
    num_workers: int = 2,
    subsample_cap: Optional[int] = N_MAX_SAFETY,
    train_cap: Optional[int] = None,
    val_cap: Optional[int] = None,
    test_cap: Optional[int] = None,
    feature_key: str = CAM17_UNIV2_FEATURE_KEY,
    expected_dim: int = CAM17_UNIV2_DIM,
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
    train_ds = Cam17UniV2SlideDataset(
        filter_records(records, fold.train_patients),
        subsample_cap=eff_train_cap,
        **ds_kwargs,
    )
    val_ds = Cam17UniV2SlideDataset(
        filter_records(records, fold.val_patients),
        subsample_cap=eff_val_cap,
        **ds_kwargs,
    )
    test_ds = Cam17UniV2SlideDataset(
        filter_records(records, fold.test_patients),
        subsample_cap=eff_test_cap,
        **ds_kwargs,
    )
    common = dict(
        num_workers=num_workers,
        collate_fn=cam17_univ2_slide_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return (
        DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False, **common),
        DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False, **common),
    )


__all__ = [
    "CAM17_UNIV2_ROOT",
    "CAM17_UNIV2_FEATURE_KEY",
    "CAM17_UNIV2_DIM",
    "enumerate_cam17_univ2_slides",
    "build_fold_assignments",
    "build_cam17_univ2_loaders_for_fold",
]
