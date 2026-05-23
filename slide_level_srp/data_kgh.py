"""KGH UNI-v2 slide adapter for the SRP TransMIL trainer.

KGH feature H5s are flat under ``KGH_FEATURE_ROOT`` while class labels
come from the raw-slide tree under ``KGH_RAW_ROOT/{train,test}/<class>``.
This adapter joins those inventories by slide stem and then builds
stratified slide-level folds, matching the CAM16/PANDA/CAM17 ablation
protocol.

The current KGH ablation task is disease subtype classification only:
``CP_HP``, ``CP_SSL``, ``CP_TA``, and ``CP_TVA``.  ``Normal`` slides may be
present in the raw tree and embedding folder, but they are intentionally
excluded so the trainer sees a stable 4-class target.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from slide_level_srp.data_ext import build_neighbor_graph, compute_h_morph


KGH_FEATURE_ROOT = os.environ.get(
    "KGH_FEATURE_ROOT",
    "data/features/kgh/patches",
)
KGH_RAW_ROOT = os.environ.get("KGH_RAW_ROOT", "data/raw/kgh")
KGH_FEATURE_KEY = os.environ.get("KGH_FEATURE_KEY", "features/uni_v2")
KGH_DIM = 1536
KGH_CLASSES = ["CP_HP", "CP_SSL", "CP_TA", "CP_TVA"]
_N_MAX_SAFETY_KGH = 32768


@dataclass(frozen=True)
class KGHSlideRecord:
    slide_id: str
    patient_id: str
    center: int
    label: int
    h5_path: str


@dataclass(frozen=True)
class KGHFoldAssignment:
    train_patients: List[str]
    val_patients: List[str]
    test_patients: List[str]


def enumerate_kgh_slides(
    feature_root: str = KGH_FEATURE_ROOT,
    raw_root: str = KGH_RAW_ROOT,
) -> List[KGHSlideRecord]:
    """Join raw KGH class folders with extracted UNI-v2 feature H5s."""
    h5_root = Path(feature_root)
    raw = Path(raw_root)
    if not h5_root.is_dir():
        raise FileNotFoundError(f"missing KGH feature dir: {h5_root}")
    out: List[KGHSlideRecord] = []
    for label, cls in enumerate(KGH_CLASSES):
        for split in ("train", "test"):
            cls_dir = raw / split / cls
            if not cls_dir.is_dir():
                continue
            for p in sorted(cls_dir.glob("*.tif")):
                slide_id = p.stem
                h5_path = h5_root / f"{slide_id}.h5"
                if not h5_path.exists():
                    continue
                out.append(KGHSlideRecord(
                    slide_id=slide_id,
                    patient_id=slide_id,
                    center=0,
                    label=label,
                    h5_path=str(h5_path),
                ))
    if not out:
        raise RuntimeError(
            f"no KGH slides found by joining raw_root={raw} with feature_root={h5_root}"
        )
    out.sort(key=lambda r: r.slide_id)
    return out


def build_kgh_fold_assignments(
    records: Sequence[KGHSlideRecord],
    n_folds: int = 5,
    val_frac: float = 0.10,
    fold_seed: int = 0,
) -> List[KGHFoldAssignment]:
    """Stratified slide-level folds by KGH class label."""
    rng = np.random.default_rng(fold_seed)
    by_class: Dict[int, List[KGHSlideRecord]] = {i: [] for i in range(len(KGH_CLASSES))}
    for r in records:
        by_class[r.label].append(r)
    for label, rows in by_class.items():
        if len(rows) < n_folds:
            raise RuntimeError(
                f"KGH class {KGH_CLASSES[label]} has only {len(rows)} slides; "
                f"cannot build {n_folds} folds."
            )
        rng.shuffle(rows)
    fold_test: List[List[KGHSlideRecord]] = [[] for _ in range(n_folds)]
    for rows in by_class.values():
        for i, r in enumerate(rows):
            fold_test[i % n_folds].append(r)

    out: List[KGHFoldAssignment] = []
    for k in range(n_folds):
        test = fold_test[k]
        train_pool = [r for j, fold_rows in enumerate(fold_test) if j != k for r in fold_rows]
        train_by_class: Dict[int, List[KGHSlideRecord]] = {i: [] for i in range(len(KGH_CLASSES))}
        for r in train_pool:
            train_by_class[r.label].append(r)
        val: List[KGHSlideRecord] = []
        train: List[KGHSlideRecord] = []
        for rows in train_by_class.values():
            rng.shuffle(rows)
            n_val = max(1, int(round(val_frac * len(rows))))
            val.extend(rows[:n_val])
            train.extend(rows[n_val:])
        out.append(KGHFoldAssignment(
            train_patients=[r.slide_id for r in train],
            val_patients=[r.slide_id for r in val],
            test_patients=[r.slide_id for r in test],
        ))
    return out


def _filter_records(records: Sequence[KGHSlideRecord], slide_ids: Sequence[str]) -> List[KGHSlideRecord]:
    keep = set(slide_ids)
    return [r for r in records if r.slide_id in keep]


class KGHSlideDataset(Dataset):
    def __init__(
        self,
        records: Sequence[KGHSlideRecord],
        subsample_cap: Optional[int] = _N_MAX_SAFETY_KGH,
        feature_key: str = KGH_FEATURE_KEY,
        expected_dim: int = KGH_DIM,
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
            stride = int(f.attrs.get("patch_size_level0", 256))
        if feats.ndim != 2 or feats.shape[1] != self.expected_dim:
            got = feats.shape[1] if feats.ndim == 2 else "non-2D"
            raise ValueError(
                f"{r.h5_path}: expected {self.expected_dim}-d features at "
                f"{self.feature_key}, got {got}"
            )
        if self.subsample_cap is not None and feats.shape[0] > self.subsample_cap:
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


def kgh_slide_collate(batch: List[Dict]) -> Dict:
    if len(batch) != 1:
        raise ValueError(f"kgh_slide_collate expects batch_size=1, got {len(batch)}")
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


def build_kgh_loaders_for_fold(
    records: Sequence[KGHSlideRecord],
    fold: KGHFoldAssignment,
    num_workers: int = 2,
    subsample_cap: Optional[int] = _N_MAX_SAFETY_KGH,
    train_cap: Optional[int] = None,
    val_cap: Optional[int] = None,
    test_cap: Optional[int] = None,
    feature_key: str = KGH_FEATURE_KEY,
    expected_dim: int = KGH_DIM,
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
    train_ds = KGHSlideDataset(_filter_records(records, fold.train_patients), eff_train_cap, **ds_kwargs)
    val_ds = KGHSlideDataset(_filter_records(records, fold.val_patients), eff_val_cap, **ds_kwargs)
    test_ds = KGHSlideDataset(_filter_records(records, fold.test_patients), eff_test_cap, **ds_kwargs)
    common = dict(num_workers=num_workers, collate_fn=kgh_slide_collate, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False, **common),
        DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False, **common),
    )


__all__ = [
    "KGH_CLASSES",
    "KGH_DIM",
    "enumerate_kgh_slides",
    "build_kgh_fold_assignments",
    "build_kgh_loaders_for_fold",
]
