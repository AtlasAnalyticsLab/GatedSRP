"""
CAM16 dataset adapter for the slide-level SRP pipeline.

Mirrors the API of `slide_level/src/data.py` + `data_ext.py` so the
existing TransMIL aggregator + SRP attention can train on CAMELYON16
binary slide classification (normal vs tumor) with minimal trainer
changes.

Differences vs. the CAM17 path:

  - **Embeddings**: ViT-B/16 (768-dim), stored under
    `CAM16_FEATURE_ROOT` (default: `data/features/camelyon16_vit_b16`)
    inside the group `features/vit_b_16`. The CAM17 pipeline uses
    `features` (UNI-v2, 1024-dim).
  - **Coords**: 5-D (X, Y, RW, RH, LV) per the embeddings' passport
    format. We use the first two columns (level-0 pixel X, Y) for
    neighbour-graph reconstruction.
  - **Patch stride at level 0**: 448 pixels (from each h5's
    `patch_size_level0` attribute; CAM16 patches are 224 px at 20×
    extracted from a 448×448 region at 40×). Compare with CAM17's
    PATCH_STRIDE_L0 = 512.
  - **Labels**: binary {0: normal, 1: tumor}. Derived from the
    enclosing directory: `tumor/patches/*` → 1, `normal/patches/*` → 0.
  - **No patient grouping**: each slide is its own "patient" in
    CAM16. Fold assignment is **per-slide stratified** by the binary
    label (270 slides → 5 stratified folds).

The adapter reuses `build_neighbor_index` and `compute_h_morph` from
`data_ext.py` because the spatial neighbour computation only needs
(coords, stride) — independent of feature backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Reuse the SRP-side helpers verbatim — the only thing that changes
# for CAM16 is what `coords[:, :2]` and `stride` are at the call site.
from slide_level_srp.data_ext import (
    build_neighbor_graph,
    build_neighbor_index,
    compute_h_morph,
)


# CAM16 layout on disk.
_CAM16_FEATURE_ROOT = os.environ.get("CAM16_FEATURE_ROOT", "data/features/camelyon16_vit_b16")
_CAM16_FEATURE_KEY = os.environ.get("CAM16_FEATURE_KEY", "features/vit_b_16")     # 768-dim per patch
# Level-0 patch stride for CAM16 embeddings (from `patch_size_level0`
# attr in each h5; 448 px = 224 px × 2× resize from 40× → 20×).
PATCH_STRIDE_L0_CAM16 = 448

# Safety ceiling on per-slide patch count, matching stage-2's
# N_MAX_SAFETY. CAM16 slides go up to ~14k patches at full grid.
_N_MAX_SAFETY_CAM16 = 16384


@dataclass(frozen=True)
class Cam16SlideRecord:
    """Per-slide metadata. Field names match `SlideRecord` from
    `slide_level/src/data.py` so downstream code that reads
    `r.slide_id`, `r.label`, `r.h5_path` works without changes.
    `patient_id` and `center` are populated with the slide_id and 0
    respectively (CAM16 has no patient or center grouping).
    """
    slide_id: str
    patient_id: str
    center: int
    label: int
    h5_path: str


def enumerate_cam16_slides(
    feature_root: str = _CAM16_FEATURE_ROOT,
) -> List[Cam16SlideRecord]:
    """Walk the {tumor, normal} directories and emit one record per h5.

    Returns slides sorted by slide_id for deterministic iteration.
    """
    out: List[Cam16SlideRecord] = []
    for label_int, subdir in ((1, "tumor"), (0, "normal")):
        h5_dir = Path(feature_root) / subdir / "patches"
        if not h5_dir.is_dir():
            raise FileNotFoundError(f"missing: {h5_dir}")
        for p in sorted(h5_dir.glob("*.h5")):
            slide_id = p.stem
            out.append(Cam16SlideRecord(
                slide_id=slide_id,
                patient_id=slide_id,        # one slide = one patient
                center=0,                   # no center info
                label=label_int,
                h5_path=str(p),
            ))
    out.sort(key=lambda r: r.slide_id)
    return out


@dataclass(frozen=True)
class Cam16FoldAssignment:
    """Per-fold (train, val, test) slide-id partition.

    For naming compatibility with stage-2's FoldAssignment we expose
    `train_patients`, `val_patients`, `test_patients` — for CAM16
    these are slide-id lists rather than patient-id lists, since each
    slide is its own patient.
    """
    train_patients: List[str]
    val_patients: List[str]
    test_patients: List[str]


def build_cam16_fold_assignments(
    records: Sequence[Cam16SlideRecord],
    n_folds: int = 5,
    val_frac: float = 0.10,
    fold_seed: int = 0,
) -> List[Cam16FoldAssignment]:
    """Per-slide stratified k-fold split.

    Stratification is by the binary label (tumor vs normal). A
    deterministic per-class shuffle followed by interleaved fold
    assignment guarantees each fold has roughly the same class ratio.

    The val set is a 10 % carve-out of the train set within each
    fold (also stratified). Carving val out of train (rather than
    test) keeps test sizes uniform and matches stage-2 convention.
    """
    rng = np.random.default_rng(fold_seed)

    # Per-class shuffles, then stride-based fold assignment.
    by_class: Dict[int, List[Cam16SlideRecord]] = {0: [], 1: []}
    for r in records:
        by_class[r.label].append(r)
    for lst in by_class.values():
        rng.shuffle(lst)

    fold_test: List[List[Cam16SlideRecord]] = [[] for _ in range(n_folds)]
    for lst in by_class.values():
        for i, r in enumerate(lst):
            fold_test[i % n_folds].append(r)

    out: List[Cam16FoldAssignment] = []
    for k in range(n_folds):
        test_recs = fold_test[k]
        # Train pool = all other folds' records.
        train_pool: List[Cam16SlideRecord] = []
        for j in range(n_folds):
            if j != k:
                train_pool.extend(fold_test[j])
        # Within-pool shuffle for val carve-out (still stratified
        # per-class so val class balance is preserved).
        train_by_class: Dict[int, List[Cam16SlideRecord]] = {0: [], 1: []}
        for r in train_pool:
            train_by_class[r.label].append(r)
        for lst in train_by_class.values():
            rng.shuffle(lst)
        val_recs: List[Cam16SlideRecord] = []
        train_recs: List[Cam16SlideRecord] = []
        for cls_recs in train_by_class.values():
            n_val = max(1, int(round(val_frac * len(cls_recs))))
            val_recs.extend(cls_recs[:n_val])
            train_recs.extend(cls_recs[n_val:])

        out.append(Cam16FoldAssignment(
            train_patients=[r.slide_id for r in train_recs],
            val_patients=[r.slide_id for r in val_recs],
            test_patients=[r.slide_id for r in test_recs],
        ))
    return out


def filter_records_cam16(
    records: Sequence[Cam16SlideRecord],
    slide_ids: Sequence[str],
) -> List[Cam16SlideRecord]:
    """Return the subset of records whose slide_id is in slide_ids."""
    keep = set(slide_ids)
    return [r for r in records if r.slide_id in keep]


def _deterministic_subsample_cam16(
    feats: np.ndarray, coords: np.ndarray, cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample a slide that exceeds N_MAX_SAFETY. Deterministic by
    feature-row count + cap; matches the shared loader semantics so a given
    slide always yields the same subsampled rows."""
    if feats.shape[0] <= cap:
        return feats, coords
    # Deterministic shuffle via a slide-content-derived seed.
    seed = int(np.uint32(np.frombuffer(
        feats[0].tobytes(), dtype=np.uint32,
    )[0])) ^ int(cap)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(feats.shape[0])[:cap]
    idx = np.sort(idx)
    return feats[idx], coords[idx]


class Cam16SlideDataset(Dataset):
    """Mirror of `SRPSlideFeatureDataset` for CAM16. Returns the same
    dict shape so the trainer's forward / collate / loss code works
    unchanged.
    """

    def __init__(
        self,
        records: Sequence[Cam16SlideRecord],
        subsample_cap: Optional[int] = _N_MAX_SAFETY_CAM16,
        neighbor_radius: int = 1,
        neighbor_shell: str = "cumulative",
        neighbor_source: str = "real",
        neighbor_shuffle_seed: int = 0,
        neighbor_weighting: str = "uniform",
        neighbor_weight_sigma: float = 1.0,
    ) -> None:
        self.records = list(records)
        self.subsample_cap = subsample_cap
        self.neighbor_radius = int(neighbor_radius)
        self.neighbor_shell = neighbor_shell
        self.neighbor_source = neighbor_source
        self.neighbor_shuffle_seed = int(neighbor_shuffle_seed)
        self.neighbor_weighting = neighbor_weighting
        self.neighbor_weight_sigma = float(neighbor_weight_sigma)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        import h5py
        with h5py.File(r.h5_path, "r") as f:
            # CAM16 features live at /features/vit_b_16. coords are
            # 5-D; keep all 5 columns for diagnostic, but only the
            # first 2 (X, Y level-0 pixel) drive the neighbour graph.
            feats = np.asarray(f[_CAM16_FEATURE_KEY][:], dtype=np.float32)
            coords = np.asarray(f["coords"][:], dtype=np.int64)
            # Use stride from h5 attrs if available; fall back to the
            # module-level default. Both paths converge to 448 today.
            stride = int(f.attrs.get("patch_size_level0", PATCH_STRIDE_L0_CAM16))

        if self.subsample_cap is not None and feats.shape[0] > self.subsample_cap:
            feats, coords = _deterministic_subsample_cam16(
                feats, coords, self.subsample_cap,
            )

        # Restrict coords to (X, Y) for the neighbour-graph; the SRP
        # neighbour code expects shape (N, 2). The other 3 columns
        # (RW, RH, LV) are not used.
        coords_2d = coords[:, :2].astype(np.int64)
        neighbor_index, neighbor_mask, neighbor_weight = build_neighbor_graph(
            coords_2d, stride=stride,
            radius=self.neighbor_radius,
            shell=self.neighbor_shell,
            source=self.neighbor_source,
            shuffle_seed=self.neighbor_shuffle_seed + idx,
            weighting=self.neighbor_weighting,
            weight_sigma=self.neighbor_weight_sigma,
        )
        h_morph = compute_h_morph(feats, neighbor_index, neighbor_mask)

        return {
            "features":       torch.from_numpy(feats),            # (N, 768)
            "coords":         torch.from_numpy(coords_2d),        # (N, 2)
            "neighbor_index": torch.from_numpy(neighbor_index),   # (N, 8)
            "neighbor_mask":  torch.from_numpy(neighbor_mask),    # (N, 8)
            "neighbor_weight": torch.from_numpy(neighbor_weight),  # (N, K)
            "h_morph":        torch.from_numpy(h_morph),          # (N,)
            "label":          torch.tensor(r.label, dtype=torch.int64),
            "slide_id":       r.slide_id,
            "patient_id":     r.patient_id,
            "center":         r.center,
            "n_tokens":       int(feats.shape[0]),
        }


def cam16_slide_collate(batch: List[Dict]) -> Dict:
    """batch_size=1 only, mirroring stage-2 / data_ext convention.

    Validation: runtime collate contract → ValueError.
    """
    if len(batch) != 1:
        raise ValueError(
            f"cam16_slide_collate expects batch_size=1 (got {len(batch)})"
        )
    b = batch[0]
    return {
        "features":       b["features"].unsqueeze(0),
        "coords":         b["coords"].unsqueeze(0),
        "neighbor_index": b["neighbor_index"].unsqueeze(0),
        "neighbor_mask":  b["neighbor_mask"].unsqueeze(0),
        "neighbor_weight": b["neighbor_weight"].unsqueeze(0),
        "h_morph":        b["h_morph"].unsqueeze(0),
        "label":          b["label"].unsqueeze(0),
        "slide_id":       [b["slide_id"]],
        "patient_id":     [b["patient_id"]],
        "center":         [b["center"]],
        "n_tokens":       [b["n_tokens"]],
    }


def build_cam16_loaders_for_fold(
    records: Sequence[Cam16SlideRecord],
    fold: Cam16FoldAssignment,
    num_workers: int = 2,
    subsample_cap: Optional[int] = _N_MAX_SAFETY_CAM16,
    train_cap: Optional[int] = None,
    val_cap: Optional[int] = None,
    test_cap: Optional[int] = None,
    neighbor_radius: int = 1,
    neighbor_shell: str = "cumulative",
    neighbor_source: str = "real",
    neighbor_shuffle_seed: int = 0,
    neighbor_weighting: str = "uniform",
    neighbor_weight_sigma: float = 1.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train, val, test) DataLoaders for one CAM16 fold.

    Validation: per-split caps are now first-class so the
    trainer's `--train_cap` / `--val_cap` / `--test_cap` flags actually
    take effect. Each per-split cap defaults to `subsample_cap`, which
    was the previous (uniform) behaviour.
    """
    eff_train_cap = train_cap if train_cap is not None else subsample_cap
    eff_val_cap   = val_cap   if val_cap   is not None else subsample_cap
    eff_test_cap  = test_cap  if test_cap  is not None else subsample_cap

    train_ds = Cam16SlideDataset(
        filter_records_cam16(records, fold.train_patients),
        subsample_cap=eff_train_cap,
        neighbor_radius=neighbor_radius,
        neighbor_shell=neighbor_shell,
        neighbor_source=neighbor_source,
        neighbor_shuffle_seed=neighbor_shuffle_seed,
        neighbor_weighting=neighbor_weighting,
        neighbor_weight_sigma=neighbor_weight_sigma,
    )
    val_ds = Cam16SlideDataset(
        filter_records_cam16(records, fold.val_patients),
        subsample_cap=eff_val_cap,
        neighbor_radius=neighbor_radius,
        neighbor_shell=neighbor_shell,
        neighbor_source=neighbor_source,
        neighbor_shuffle_seed=neighbor_shuffle_seed,
        neighbor_weighting=neighbor_weighting,
        neighbor_weight_sigma=neighbor_weight_sigma,
    )
    test_ds = Cam16SlideDataset(
        filter_records_cam16(records, fold.test_patients),
        subsample_cap=eff_test_cap,
        neighbor_radius=neighbor_radius,
        neighbor_shell=neighbor_shell,
        neighbor_source=neighbor_source,
        neighbor_shuffle_seed=neighbor_shuffle_seed,
        neighbor_weighting=neighbor_weighting,
        neighbor_weight_sigma=neighbor_weight_sigma,
    )

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=num_workers, collate_fn=cam16_slide_collate,
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=cam16_slide_collate,
        pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=cam16_slide_collate,
        pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, test_loader
