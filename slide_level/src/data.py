"""
Slide-level dataset + fold assignment for CAMELYON17 (stage 2 of the XSA PoC).

Data layout on disk:
  data/labels/camelyon17/stages.csv
      Mixed patient- and slide-level rows. We only consume slide rows
      (patient_XXX_node_Y.tif) and drop the patient-level rows
      (patient_XXX.zip) used for pN staging.
      Columns: (patient, stage, center). For slide rows, `patient` is
      `patient_XXX_node_Y.tif`, `stage` in {negative, itc, micro, macro},
      `center` in {0, 1, 2, 3, 4}.

  data/features/camelyon17_uni_v1/
      499 h5 files, one per slide. Filename pattern
      `patient_XXX_node_Y.h5`. The CSV lists 500 slides; one is missing
      features and gets silently skipped (see README inspection in
      DESIGN.md §3.1).

Each h5 file has:
  /features  (N, 1024) fp32   -- UNI v1 ViT-L/14 pooled patch features
  /coords    (N, 2)   int64   -- level-0 pixel coordinates at 40x,
                                 stride 512 per patch

Task in this stage (§3.3):
  Default is 4-class: {negative: 0, itc: 1, micro: 2, macro: 3}.
  Binary (y = 0 if stage == "negative" else 1) is still supported via
  num_classes=2 for backwards comparability with pilot runs.

  Why 4-class is the main task: binary at UNI v1 feature quality
  saturates too quickly -- macro metastases are trivially detectable
  and dominate the positive class, leaving no room for ablation
  separation. The 4-way split reintroduces the detection difficulty
  (itc is tiny tumor regions; micro is moderate; macro is obvious).

Split design (§3.4):
  5-fold patient-grouped, center-stratified CV. fold_seed=0 fixed.
  For each outer fold k (= test fold), the remaining 80 patients are
  deterministically split into 70 train + 10 val. val_patients_per_fold
  defaults to 10 so the val draw is exactly 2 patients per center on
  CAM17's 5-center grid. Fold assignments are identical across
  ablations -- this is essential for the per-slide paired comparison in
  §8.3.

Sequence-length handling (§6.3):
  Training, val, and test are all uncapped by default. A hard safety
  ceiling `N_max_safety = 49152` triggers deterministic coord-sorted
  subsampling only for anomalous slides (no current slide exceeds it).
  The tiered fallback is implemented at the training loop / config
  level, not here -- this module exposes the raw slide and a subsample
  hook so the caller can truncate if needed.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# --- constants and filename parsing --------------------------------------

# Canonical stage-to-label mappings. §3.3 of DESIGN.md.
# 4-class is the default; binary is kept for ad-hoc comparisons only.
_NEGATIVE_STAGE = "negative"
_VALID_STAGES = {"negative", "itc", "micro", "macro"}
_STAGE_TO_LABEL_4CLASS = {"negative": 0, "itc": 1, "micro": 2, "macro": 3}
CLASS_NAMES_4CLASS = ["negative", "itc", "micro", "macro"]


def _stage_to_label(stage: str, num_classes: int) -> int:
    """
    Map a CAMELYON17 stage string to an integer class index.

    num_classes=2: binary {negative -> 0, else -> 1}
    num_classes=4: {negative: 0, itc: 1, micro: 2, macro: 3}
    """
    if num_classes == 2:
        return 0 if stage == _NEGATIVE_STAGE else 1
    if num_classes == 4:
        return _STAGE_TO_LABEL_4CLASS[stage]
    raise ValueError(f"num_classes must be 2 or 4, got {num_classes}")

# Slide rows in stages.csv look like "patient_017_node_2.tif". Patient
# rows (skipped here) look like "patient_017.zip".
_SLIDE_ROW_RE = re.compile(r"^(patient_\d{3})_node_(\d)\.tif$")

# Safety ceiling: hard upper bound on the per-slide token count. No
# current slide exceeds p100 = 37666, so this is defense against data
# drift only. See §6.3.
N_MAX_SAFETY = 49152


@dataclass(frozen=True)
class SlideRecord:
    """One slide's metadata + integer class label.

    `label` is derived from `stage` at enumeration time via the
    `num_classes` argument to `enumerate_slides` (see that docstring):
      num_classes=4 -> label in {0: negative, 1: itc, 2: micro, 3: macro}
      num_classes=2 -> label in {0: negative, 1: any tumor}
    """
    slide_id: str      # e.g. "patient_017_node_2"
    patient_id: str    # e.g. "patient_017"
    center: int        # 0..4
    stage: str         # "negative" | "itc" | "micro" | "macro"
    label: int         # see class docstring; depends on num_classes at enumeration
    h5_path: str       # absolute path to the feature h5

    @property
    def fold_will_be_set_externally(self) -> None:
        # Fold is not stored on the record; it is assigned by
        # build_fold_assignments() and carried in a separate mapping so
        # SlideRecord stays a pure metadata object.
        return None


# --- stages.csv parsing ---------------------------------------------------

def _read_stages_csv(csv_path: str) -> Dict[str, Tuple[str, int]]:
    """
    Read stages.csv and return {slide_id -> (stage, center)} for slide rows
    only. Patient-level rows (patient_XXX.zip) are silently dropped.

    Why slide_id and not the raw `patient_XXX_node_Y.tif`: we strip the
    .tif extension so it matches the h5 filename base and the fold-split
    code can key on a single canonical string.
    """
    out: Dict[str, Tuple[str, int]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["patient"]
            m = _SLIDE_ROW_RE.match(name)
            if m is None:
                # patient-level row (pN stage); we don't use these.
                continue
            stage = row["stage"].strip()
            if stage not in _VALID_STAGES:
                raise ValueError(f"Unexpected stage '{stage}' for slide {name}")
            center = int(row["center"])
            slide_id = name[:-4]  # strip ".tif"
            out[slide_id] = (stage, center)
    return out


def enumerate_slides(
    csv_path: str = os.environ.get("CAM17_STAGES_CSV", "data/labels/camelyon17/stages.csv"),
    feature_root: str = os.environ.get(
        "CAM17_FEATURE_ROOT",
        "data/features/camelyon17_uni_v1",
    ),
    num_classes: int = 4,
) -> List[SlideRecord]:
    """
    Join stages.csv with the on-disk h5 inventory. Slides listed in the
    CSV but missing an h5 are silently dropped (matches DESIGN.md §3.1:
    500 listed, 499 with features; one patient_027_node_1 is missing).

    num_classes=4 (default) -> labels in {0: negative, 1: itc, 2: micro, 3: macro}.
    num_classes=2 -> binary {0: negative, 1: any positive}.

    Returns records sorted by slide_id for deterministic iteration.
    """
    stage_map = _read_stages_csv(csv_path)

    available_h5 = {
        p.stem: str(p)
        for p in Path(feature_root).glob("*.h5")
    }

    records: List[SlideRecord] = []
    for slide_id in sorted(stage_map.keys()):
        if slide_id not in available_h5:
            # Missing features: skip silently (documented behavior).
            continue
        stage, center = stage_map[slide_id]
        m = _SLIDE_ROW_RE.match(slide_id + ".tif")
        patient_id = m.group(1)
        label = _stage_to_label(stage, num_classes)
        records.append(SlideRecord(
            slide_id=slide_id,
            patient_id=patient_id,
            center=center,
            stage=stage,
            label=label,
            h5_path=available_h5[slide_id],
        ))
    return records


# --- fold assignment ------------------------------------------------------

def _center_stratified_patient_partition(
    patients: Sequence[Tuple[str, int]],  # (patient_id, center)
    n_folds: int,
    seed: int,
) -> List[List[str]]:
    """
    Partition patients into n_folds folds so each fold contains an
    equal share of patients from each center.

    For CAMELYON17's exact 5-center, 20-patients-per-center grid and
    n_folds=5, this yields exactly 4 patients per (center, fold). For
    other configurations we distribute the within-center patients
    round-robin over folds after a seeded shuffle.

    The shuffle is the *only* source of randomness; given the same seed
    the same assignment is produced across invocations.
    """
    rng = np.random.default_rng(seed)
    by_center: Dict[int, List[str]] = {}
    for pid, c in patients:
        by_center.setdefault(c, []).append(pid)

    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for c in sorted(by_center.keys()):
        pats = sorted(by_center[c])  # sort first for determinism
        rng.shuffle(pats)
        for i, pid in enumerate(pats):
            # Round-robin within a center; offset by center index so
            # any remainder spreads across different folds for different
            # centers, not piling up on fold 0 for every center.
            folds[(i + c) % n_folds].append(pid)
    # Sort each fold's patients for stable output.
    return [sorted(f) for f in folds]


@dataclass(frozen=True)
class FoldAssignment:
    """
    Precomputed fold + train/val/test split. Identical across ablations.

    Attributes:
      fold_id:         which outer fold is the test fold (0..n_folds-1)
      train_patients:  list of patient_ids in train
      val_patients:    list of patient_ids in val
      test_patients:   list of patient_ids in test (this fold)
      all_folds:       the full n_folds patient partition (needed for
                       sanity checks and for patient_id -> fold lookup)
    """
    fold_id: int
    train_patients: List[str]
    val_patients: List[str]
    test_patients: List[str]
    all_folds: List[List[str]]


def _select_val_patients(
    pool: Sequence[Tuple[str, int]],
    val_n: int,
    seed: int,
) -> List[str]:
    """
    From `pool` (list of (patient_id, center)), deterministically pick
    exactly `val_n` patients center-stratified. If `val_n` is not
    divisible by the number of centers, the extra allocations are
    distributed round-robin over centers, with the starting center
    chosen by the seed so different folds do not systematically pull
    extras from the same center.

    Guarantees len(output) == val_n.
    """
    rng = np.random.default_rng(seed)
    by_center: Dict[int, List[str]] = {}
    for p, c in pool:
        by_center.setdefault(c, []).append(p)
    centers = sorted(by_center.keys())
    n_c = len(centers)
    # Phase-A.9 fourth-review fix F4: split-budget validation must raise
    # ValueError so it survives `python -O`. Otherwise an over-large val_n
    # would silently slice past the pool and yield a misshaped split.
    pool_size = sum(len(v) for v in by_center.values())
    if val_n > pool_size:
        raise ValueError(f"val_n={val_n} exceeds pool size {pool_size}")
    # Even base per center; distribute the remainder round-robin.
    base = val_n // n_c
    remainder = val_n - base * n_c
    per_center = [base] * n_c
    # Choose which centers get the +1 by seeded rotation so different
    # folds pull extras from different centers instead of always center 0.
    rotate = int(rng.integers(0, n_c))
    for i in range(remainder):
        per_center[(rotate + i) % n_c] += 1

    val: List[str] = []
    for c_idx, c in enumerate(centers):
        # Re-shuffle within each center with a center-dependent sub-seed
        # so val selection is stable but isn't anti-correlated with the
        # outer patient-partition shuffle.
        sub_rng = np.random.default_rng(seed * 100 + c)
        pats = sorted(by_center[c])
        sub_rng.shuffle(pats)
        val.extend(pats[:per_center[c_idx]])
    return sorted(val)


def build_fold_assignments(
    records: Sequence[SlideRecord],
    n_folds: int = 5,
    val_patients_per_fold: int = 10,
    fold_seed: int = 0,
) -> List[FoldAssignment]:
    """
    Build the n_folds patient-grouped, center-stratified CV assignments.

    Deterministic in fold_seed. At fold k:
      test  = patients in fold k (~20 patients, ~100 slides for CAM17/5-fold)
      val   = val_patients_per_fold patients, center-stratified, drawn
              from the non-test pool using a fold-dependent sub-seed
      train = non-test, non-val patients (~70 patients, ~350 slides)

    val_patients_per_fold default of 10 gives an exact 2-per-center draw
    for CAM17's 5-center grid, which is both clean and large enough
    (~50 slides) for reliable early-stopping / best-val-F1 selection.

    Returns a list of length n_folds.
    """
    # (patient_id, center) pairs, deduplicated and sorted.
    patient_center: Dict[str, int] = {}
    for r in records:
        if r.patient_id in patient_center:
            # Phase-A.9 fourth-review fix F4: data-integrity check must
            # raise ValueError so it survives `python -O`. A patient with
            # inconsistent centers indicates corrupted metadata that
            # would silently produce wrong stratification.
            if patient_center[r.patient_id] != r.center:
                raise ValueError(
                    f"Patient {r.patient_id} has inconsistent center assignments "
                    f"across slides: {patient_center[r.patient_id]} vs {r.center}"
                )
        else:
            patient_center[r.patient_id] = r.center
    patients = sorted(patient_center.items())  # deterministic order

    # Outer partition: which patients go in each test fold.
    all_folds = _center_stratified_patient_partition(
        patients, n_folds=n_folds, seed=fold_seed,
    )
    # Sanity: every patient assigned exactly once.
    flat = [p for f in all_folds for p in f]
    # Phase-A.9 fourth-review fix F4: clean-cover invariant must raise
    # ValueError (RuntimeError-class) so it survives `python -O`. A
    # broken cover means leakage between folds — silent dataset bug.
    if not (len(flat) == len(patient_center) == len(set(flat))):
        raise RuntimeError(
            "Internal error: fold partition is not a clean cover"
        )

    assignments: List[FoldAssignment] = []
    for k in range(n_folds):
        test_set = set(all_folds[k])
        # Train+val pool = patients not in test fold, with their centers.
        pool = [(p, patient_center[p]) for p in patient_center if p not in test_set]
        # Fold-dependent sub-seed so each fold has its own val draw but
        # everything is reproducible from fold_seed alone.
        val = _select_val_patients(pool, val_patients_per_fold,
                                   seed=fold_seed * 1000 + k + 1)
        val_set = set(val)
        train = sorted(p for (p, _) in pool if p not in val_set)
        assignments.append(FoldAssignment(
            fold_id=k,
            train_patients=train,
            val_patients=sorted(val),
            test_patients=sorted(all_folds[k]),
            all_folds=all_folds,
        ))
    return assignments


def filter_records(
    records: Sequence[SlideRecord],
    patients: Sequence[str],
) -> List[SlideRecord]:
    """Return the subset of `records` whose patient_id is in `patients`."""
    allow = set(patients)
    return [r for r in records if r.patient_id in allow]


# --- torch Dataset --------------------------------------------------------

class SlideFeatureDataset(Dataset):
    """
    PyTorch dataset that lazily loads (features, coords, label, metadata)
    from h5 per-slide. Features are read fresh on every __getitem__ so
    memory stays bounded at one slide at a time; with 500 slides and up
    to 37k tokens at 1024 dim fp32, the total is ~6.7M * 1024 * 4 ≈
    27 GB which is the full feature directory on disk.

    Subsample policy:
      Only invoked when N > subsample_cap (default N_MAX_SAFETY).
      Deterministic coord-sorted subsample: sort by (coord_y, coord_x)
      and take evenly-spaced indices so the retained patches cover the
      slide uniformly, not randomly. Applied identically at train, val,
      and test time so train-time randomness doesn't contaminate eval.

    Returns dict with:
      features:   (N, 1024) fp32 tensor
      coords:     (N, 2) int64 tensor (level-0 pixel coords)
      label:      scalar int64 tensor (0 or 1)
      slide_id:   str
      patient_id: str
      center:     int
      n_tokens:   int  (after any subsample)
    """
    def __init__(
        self,
        records: Sequence[SlideRecord],
        subsample_cap: Optional[int] = N_MAX_SAFETY,
    ) -> None:
        self.records = list(records)
        self.subsample_cap = subsample_cap

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        with h5py.File(r.h5_path, "r") as f:
            feats = np.asarray(f["features"][:], dtype=np.float32)  # (N, 1024)
            coords = np.asarray(f["coords"][:], dtype=np.int64)     # (N, 2)

        if self.subsample_cap is not None and feats.shape[0] > self.subsample_cap:
            feats, coords = _deterministic_subsample(feats, coords, self.subsample_cap)

        return {
            "features": torch.from_numpy(feats),
            "coords": torch.from_numpy(coords),
            "label": torch.tensor(r.label, dtype=torch.int64),
            "slide_id": r.slide_id,
            "patient_id": r.patient_id,
            "center": r.center,
            "n_tokens": int(feats.shape[0]),
        }


def _deterministic_subsample(
    feats: np.ndarray, coords: np.ndarray, cap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coord-sorted deterministic subsample. Sort by (y, x) so patches are
    in a stable spatial order, then take cap evenly-spaced indices. No
    RNG -- same output every call for a given (feats, coords).
    """
    # Phase-A.9 third-review fix F6: runtime data contract → ValueError.
    if feats.shape[0] != coords.shape[0]:
        raise ValueError(
            f"feats and coords disagree on N: {feats.shape[0]} vs {coords.shape[0]}"
        )
    N = feats.shape[0]
    order = np.lexsort((coords[:, 1], coords[:, 0]))  # (y, x) as (secondary, primary)
    # evenly-spaced N -> cap indices (inclusive endpoints).
    keep = np.linspace(0, N - 1, num=cap, dtype=np.int64)
    keep_sorted = order[keep]
    return feats[keep_sorted], coords[keep_sorted]


# --- collate + loaders ----------------------------------------------------

def slide_collate(batch: List[Dict]) -> Dict:
    """
    Collate function for batch_size=1 slide batches.

    Slides have variable N, so we do NOT stack `features` / `coords`;
    instead we return them as a length-1 list so the training loop can
    treat them per-slide. Scalars are stacked normally.

    This assumes batch_size=1; multi-slide batching would require padding
    + masks, which we explicitly avoid (§6.2).
    """
    # Phase-A.9 third-review fix F6: runtime collate contract → ValueError.
    if len(batch) != 1:
        raise ValueError(
            f"slide_collate expects batch_size=1 (got {len(batch)}); "
            "see DESIGN.md §6.2 for rationale."
        )
    b = batch[0]
    return {
        "features": b["features"].unsqueeze(0),   # (1, N, 1024)
        "coords": b["coords"].unsqueeze(0),       # (1, N, 2)
        "label": b["label"].unsqueeze(0),         # (1,)
        "slide_id": [b["slide_id"]],
        "patient_id": [b["patient_id"]],
        "center": [b["center"]],
        "n_tokens": [b["n_tokens"]],
    }


def _effective_cap(cap: Optional[int]) -> int:
    """
    Combine a caller-supplied cap with the hard safety ceiling
    N_MAX_SAFETY. None means "no per-split cap"; the safety ceiling still
    applies as an anti-pathology guard. Any explicit cap that exceeds
    N_MAX_SAFETY is silently clamped to the safety ceiling.
    """
    if cap is None:
        return N_MAX_SAFETY
    return min(int(cap), N_MAX_SAFETY)


def build_loaders_for_fold(
    records: Sequence[SlideRecord],
    fold: FoldAssignment,
    num_workers: int = 2,
    train_cap: Optional[int] = None,
    val_cap: Optional[int] = None,
    test_cap: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build (train, val, test) DataLoaders for a specific CV fold.

    Per-split caps (DESIGN.md §6.3 tiered fallback):
      - Primary tier:     all caps = None -> N_MAX_SAFETY hard ceiling only
      - Fallback 2:       train_cap = 32768,  val/test = None (uncapped)
      - Fallback 3:       train_cap = val_cap = 16384, test_cap = None
    (Fallback 1 is a checkpointing change, not a cap change; see §6.3.)

    None means "no per-split cap beyond the safety ceiling". Any explicit
    cap is combined with N_MAX_SAFETY via min() so the ceiling is never
    exceeded regardless of caller input.

    batch_size is hard-coded to 1; the training loop handles effective
    batching via gradient accumulation (see DESIGN.md §6.2).
    """
    tr_cap = _effective_cap(train_cap)
    va_cap = _effective_cap(val_cap)
    te_cap = _effective_cap(test_cap)

    train_ds = SlideFeatureDataset(
        filter_records(records, fold.train_patients), subsample_cap=tr_cap,
    )
    val_ds = SlideFeatureDataset(
        filter_records(records, fold.val_patients), subsample_cap=va_cap,
    )
    test_ds = SlideFeatureDataset(
        filter_records(records, fold.test_patients), subsample_cap=te_cap,
    )

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=num_workers, collate_fn=slide_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=slide_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=slide_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader
