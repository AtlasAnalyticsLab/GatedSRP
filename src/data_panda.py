"""
PANDA slide-level data loader (default: UNI-v2 features, 20x / 256-px patches).

Source layout:
  data/labels/panda/train.csv
       columns: image_id, data_provider, isup_grade, gleason_score
       10,616 rows.
  data/features/panda/patches/<image_id>.h5
       /coords     (N, 5) int32  — patch coordinates + level info
       /features/uni_v2  (N, 1536) float32 — UNI-v2 embeddings
       /passports  (group; metadata, not used)

Returned per slide (always at NATIVE patch count — no caps, no
subsampling). The trainer / eval loop runs at batch_size=1 with
gradient accumulation (matching the slide_level / slide_level_srp
protocol that established this for CAMELYON17), so the dataset emits
variable-length items and the collate is trivial.

  features        (N, 1536) float32
  mask            (N,) bool         — all True (no padding at the
                                      dataset level; preserved for
                                      forward-signature compatibility
                                      with the original PANDA model)
  neighbor_index  (N, 8) int64      — 8-neighbour graph; indices in
                                      [0, N), or −1 at slide-edge slots
  neighbor_mask   (N, 8) bool       — True at valid neighbour slots
  label           int64             — ISUP grade in {0..5}
  n_real          int               — N (carried for record-keeping)
  image_id        str

Stratified-by-(provider × isup_grade) splits are produced via
build_panda_folds(); fold assignment is deterministic in fold_seed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PANDA_ROOT = Path(os.environ.get("PANDA_ROOT", "data/raw/panda"))
PANDA_LABEL_CSV = Path(os.environ.get(
    "PANDA_CSV",
    os.environ.get("PANDA_LABEL_CSV", "data/labels/panda/train.csv"),
))
PANDA_CSV = PANDA_LABEL_CSV
PANDA_H5_DIR = Path(os.environ.get(
    "PANDA_FEATURE_ROOT",
    "data/features/panda/patches",
))
N_ISUP = 6
UNI_DIM = 1536

# AtlasPatch's PANDA extraction is 20x / 256-px / no-overlap. The H5
# /coords field stores (x, y, patch_size_x, patch_size_y, level) with
# level=0 px coordinates. Adjacent patches differ by exactly 256 px on
# each axis. Verified on representative slides 2026-04-25.
PANDA_PATCH_STRIDE_L0 = 256

# 8-neighbour graph builder is shared with slide-level SRP CAMELYON17 SRP. Both
# stages' H5s store level-0 pixel coords with a fixed stride, so the
# only PANDA-specific bit is the stride value (256 vs CAM17's 512) and
# the column slice (PANDA's coords are (N, 5); we keep the first two as
# (x, y)). Cross-validated bit-identical on real PANDA slides 2026-04-25.
from slide_level_srp.data_ext import (                  # noqa: E402
    build_neighbor_graph as _build_neighbor_graph_stage3,
    build_neighbor_index as _build_neighbor_index_stage3,
)


def build_panda_neighbor_index(
    coords: np.ndarray,                # (N, ≥2) PANDA H5 /coords (col 0=x, 1=y at level-0 px)
    stride: int = PANDA_PATCH_STRIDE_L0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin PANDA-specific wrapper around the slide-level SRP builder. Slices the
    first two columns of the (N, 5) PANDA H5 coords, then delegates.
    Returns (neighbor_index: (N, 8) int64, neighbor_mask: (N, 8) bool).
    """
    # Validation: data-shape contract must raise
    # ValueError so it survives `python -O`. A wrongly-shaped coords
    # array would silently produce garbage neighbours otherwise.
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"coords must be a 2-D array with >=2 columns, got shape "
            f"{tuple(coords.shape)}"
        )
    return _build_neighbor_index_stage3(
        coords[:, :2].astype(np.int64), stride=stride,
    )


def build_panda_neighbor_graph(
    coords: np.ndarray,
    stride: int = PANDA_PATCH_STRIDE_L0,
    radius: int = 1,
    shell: str = "cumulative",
    source: str = "real",
    shuffle_seed: int = 0,
    weighting: str = "uniform",
    weight_sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PANDA wrapper returning neighbor index, mask, and per-slot weights."""
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"coords must be a 2-D array with >=2 columns, got shape "
            f"{tuple(coords.shape)}"
        )
    return _build_neighbor_graph_stage3(
        coords[:, :2].astype(np.int64),
        stride=stride,
        radius=radius,
        shell=shell,
        source=source,
        shuffle_seed=shuffle_seed,
        weighting=weighting,
        weight_sigma=weight_sigma,
    )


def _compute_h_local(
    feats: np.ndarray,                       # (N, D) float32 patch features
    nbi: np.ndarray,                         # (N, 8) int64 neighbour indices
    nbm: np.ndarray,                         # (N, 8) bool neighbour mask
) -> np.ndarray:
    """
    Per-patch local homogeneity = mean cosine similarity between the
    patch's raw feature vector and its valid 8-neighbours' feature
    vectors. Returned shape (N,), dtype float32.

    h_local serves as a per-token diagnostic input to the signed
    learned gate (the signed-gate design). It is intrinsic
    to the frozen patch embeddings, so it is computed once per slide
    here rather than at every transformer block.

    Boundary patches with fewer than 8 valid neighbours: average over
    valid neighbours only. Patches with zero valid neighbours
    (pathological — should not happen for any real PANDA slide where
    every patch has at least one orthogonal neighbour) get h_local = 0
    via the count==0 → safe-divide-by-1 path.
    """
    n = feats.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    feats_n = feats / norms                                # (N, D)
    # Gather neighbour normalized features. Invalid slots gather row 0
    # but get masked out in the next step.
    safe_idx = np.where(nbm, nbi, 0)
    nbr_n = feats_n[safe_idx]                              # (N, 8, D)
    cos = (nbr_n * feats_n[:, None, :]).sum(axis=2)        # (N, 8)
    cos = cos * nbm                                        # zero-out invalid
    valid_count = nbm.sum(axis=1).astype(np.float32)       # (N,)
    h = np.where(
        valid_count > 0,
        cos.sum(axis=1) / np.maximum(valid_count, 1.0),
        0.0,
    )
    return h.astype(np.float32)


@dataclass(frozen=True)
class SlideRecord:
    image_id: str
    data_provider: str
    isup_grade: int
    h5_path: str


@dataclass(frozen=True)
class PandaGlobalSeedSplit:
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]
    stratum_counts: Dict[str, Dict[str, int]]


def enumerate_slides(
    csv_path: Path | str = PANDA_CSV,
    h5_dir: Path | str = PANDA_H5_DIR,
    *,
    validate_h5: bool = True,
    min_patches: int = 1,
    feature_key: str = "features/uni_v2",
    feature_dim: int = UNI_DIM,
) -> List[SlideRecord]:
    """
    Read train.csv and return valid SlideRecord entries.

    The validation pass intentionally happens before fold construction:
    a slide with zero extracted patches cannot provide WSI evidence and
    can crash coordinate-position runs that need a per-slide coordinate
    min/max. Dropping invalid H5s here keeps train/val/test folds
    deterministic and prevents one bad slide from failing only when a
    worker happens to sample it mid-run.
    """
    df = pd.read_csv(csv_path)
    h5_dir = Path(h5_dir)
    out: List[SlideRecord] = []
    missing = 0
    invalid: Counter[str] = Counter()
    invalid_examples: Dict[str, str] = {}
    for _, row in df.iterrows():
        p = h5_dir / f"{row['image_id']}.h5"
        if not p.exists():
            missing += 1
            continue
        if validate_h5:
            ok, reason = _validate_panda_h5_metadata(
                p,
                min_patches=min_patches,
                feature_key=feature_key,
                feature_dim=feature_dim,
            )
            if not ok:
                invalid[reason] += 1
                invalid_examples.setdefault(reason, p.name)
                continue
        out.append(SlideRecord(
            image_id=row["image_id"],
            data_provider=row["data_provider"],
            isup_grade=int(row["isup_grade"]),
            h5_path=str(p),
        ))
    if missing > 0 or invalid:
        invalid_total = sum(invalid.values())
        invalid_detail = ", ".join(
            f"{reason}={count}" for reason, count in sorted(invalid.items())
        ) or "none"
        example_detail = ", ".join(
            f"{reason}:{name}" for reason, name in sorted(invalid_examples.items())
        ) or "none"
        print(
            "[data_panda] dropped slides before fold split: "
            f"missing_h5={missing}, invalid_h5={invalid_total} "
            f"({invalid_detail}); examples=({example_detail}); "
            f"using {len(out)}"
        )
    return out


def _validate_panda_h5_metadata(
    h5_path: Path,
    *,
    min_patches: int = 1,
    feature_key: str = "features/uni_v2",
    feature_dim: int = UNI_DIM,
) -> Tuple[bool, str]:
    """
    Validate only H5 metadata, not the full feature arrays.

    Opening each file once during enumeration is cheaper than discovering
    a malformed slide inside a DataLoader worker after an experiment has
    already consumed GPU time. The checks mirror PandaSlideDataset's
    runtime contract so both entry points reject the same bad inputs.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            # Encoder-selection ablations reuse the same PANDA split/loader
            # contract with alternate AtlasPatch feature groups.  Keep the
            # default as UNI-v2 so historical runs are bit-for-bit configured
            # the same, but validate the caller-selected key before fold split.
            if feature_key not in f:
                return False, f"missing_{feature_key.replace('/', '_')}"
            if "coords" not in f:
                return False, "missing_coords"
            feat_shape = tuple(f[feature_key].shape)
            coord_shape = tuple(f["coords"].shape)
    except OSError:
        return False, "h5_read_error"

    if len(feat_shape) != 2:
        return False, "bad_feature_rank"
    if feat_shape[1] != int(feature_dim):
        return False, "bad_feature_dim"
    if len(coord_shape) != 2:
        return False, "bad_coord_rank"
    if coord_shape[1] < 2:
        return False, "bad_coord_dim"
    if feat_shape[0] != coord_shape[0]:
        return False, "feature_coord_length_mismatch"
    if feat_shape[0] < min_patches:
        return False, "zero_patches" if feat_shape[0] == 0 else "too_few_patches"
    return True, "ok"


def build_panda_folds(
    records: Sequence[SlideRecord],
    n_folds: int = 5,
    fold_seed: int = 0,
) -> List[List[int]]:
    """
    Stratified k-fold by (provider, isup_grade) joint. Returns
    a list of length n_folds, each containing record indices.

    The two providers (karolinska, radboud) score very differently
    (reported configuration), so naive random splits leak provider as a label
    predictor. Here we stratify on the joint to neutralize that.

    Determinism: fold assignment depends only on (record order, fold_seed)
    after a single seeded shuffle within each provider × isup bucket.
    """
    rng = np.random.default_rng(fold_seed)
    by_bucket: Dict[Tuple[str, int], List[int]] = {}
    for i, r in enumerate(records):
        key = (r.data_provider, r.isup_grade)
        by_bucket.setdefault(key, []).append(i)

    folds: List[List[int]] = [[] for _ in range(n_folds)]
    # Round-robin within each bucket (with a seeded offset per bucket so
    # any remainder spreads across folds rather than piling on fold 0).
    for bi, key in enumerate(sorted(by_bucket.keys())):
        idxs = by_bucket[key].copy()
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[(j + bi) % n_folds].append(idx)
    # Stable per-fold ordering for reproducibility.
    return [sorted(f) for f in folds]


def _bounded_fraction_count(n_items: int, frac: float, min_remaining: int) -> int:
    """Return a rounded split count while keeping enough samples behind."""
    if n_items <= min_remaining or frac <= 0.0:
        return 0
    # Keep each non-empty PANDA provider/ISUP stratum represented in val/test
    # when possible, but never consume the whole stratum because training still
    # needs examples for every provider/grade combination.
    n_take = max(1, int(round(float(frac) * n_items)))
    return min(n_take, n_items - min_remaining)


def build_panda_global_seed_splits(
    records: Sequence[SlideRecord],
    *,
    global_seed: int,
    test_frac: float = 0.20,
    val_frac: float = 0.10,
) -> PandaGlobalSeedSplit:
    """
    Build one PANDA train/val/test split directly from `global_seed`.

    This is the PANDA analogue of slide_level_srp's global-seed holdout:
    a single seed controls the split membership, training RNG, and neighbour
    shuffle RNG.  The split unit is the slide (`image_id`) and stratification
    is the existing PANDA joint bucket `(data_provider, isup_grade)` so the
    held-out test set does not drift heavily by site or disease grade.
    """
    if not 0.0 <= test_frac < 1.0:
        raise ValueError(f"test_frac must be in [0, 1), got {test_frac}")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if test_frac + val_frac >= 1.0:
        raise ValueError(
            "PANDA global-seed split requires test_frac + val_frac < 1; "
            f"got {test_frac + val_frac}"
        )

    by_bucket: Dict[Tuple[str, int], List[int]] = {}
    for i, r in enumerate(records):
        by_bucket.setdefault((r.data_provider, int(r.isup_grade)), []).append(i)

    rng = np.random.default_rng(int(global_seed))
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    stratum_counts: Dict[str, Dict[str, int]] = {}

    for provider, grade in sorted(by_bucket.keys(), key=lambda key: (key[0], key[1])):
        shuffled = list(by_bucket[(provider, grade)])
        rng.shuffle(shuffled)

        # Match the slide-level implementation: hold out test first as about
        # 20% of the stratum, then validation as about 10% of the original
        # stratum from the remaining pool.  This yields roughly 70/10/20 while
        # keeping all arms for a given global seed paired.
        n_test = _bounded_fraction_count(len(shuffled), test_frac, min_remaining=2)
        remaining = shuffled[n_test:]
        n_val = _bounded_fraction_count(len(remaining), val_frac, min_remaining=1)

        test_part = shuffled[:n_test]
        val_part = remaining[:n_val]
        train_part = remaining[n_val:]
        if not train_part:
            raise ValueError(
                "PANDA global-seed holdout produced an empty train stratum "
                f"for provider={provider!r}, isup_grade={grade!r}; "
                f"records={len(shuffled)}"
            )

        train_idx.extend(train_part)
        val_idx.extend(val_part)
        test_idx.extend(test_part)
        stratum_counts[f"{provider}|isup{grade}"] = {
            "train": len(train_part),
            "val": len(val_part),
            "test": len(test_part),
            "total": len(shuffled),
        }

    return PandaGlobalSeedSplit(
        train_idx=sorted(train_idx),
        val_idx=sorted(val_idx),
        test_idx=sorted(test_idx),
        stratum_counts=stratum_counts,
    )


# --- Dataset --------------------------------------------------------------

class PandaSlideDataset(Dataset):
    """
    Native-length variable-N slide dataset. No caps, no subsampling, no
    train/eval policy difference. Used with batch_size=1 + gradient
    accumulation (mirroring the CAMELYON17 / slide-level SRP protocol), so the
    dataset returns each slide at its actual patch count and the
    trainer handles effective-batch-size via grad_accum.

    The `train` flag is preserved on the constructor as a no-op for
    API compatibility with prior callers; randomness lives in the
    trainer (e.g. data shuffle order) rather than here.
    """
    def __init__(
        self,
        records: Sequence[SlideRecord],
        n_max: int | None = None,        # legacy arg, accepted for back-compat
        train: bool = True,
        seed: int | None = None,
        neighbor_radius: int = 1,
        neighbor_shell: str = "cumulative",
        neighbor_source: str = "real",
        neighbor_shuffle_seed: int = 0,
        neighbor_weighting: str = "uniform",
        neighbor_weight_sigma: float = 1.0,
        feature_key: str = "features/uni_v2",
        feature_dim: int = UNI_DIM,
    ) -> None:
        self.records = list(records)
        self.train = train
        # Defaults preserve the original UNI-v2 PANDA path.  These attributes
        # make encoder-transfer ablations explicit in the run config instead
        # of hard-coding feature groups in the dataset implementation.
        self.feature_key = feature_key
        self.feature_dim = int(feature_dim)
        self.neighbor_radius = int(neighbor_radius)
        self.neighbor_shell = neighbor_shell
        self.neighbor_source = neighbor_source
        self.neighbor_shuffle_seed = int(neighbor_shuffle_seed)
        self.neighbor_weighting = neighbor_weighting
        self.neighbor_weight_sigma = float(neighbor_weight_sigma)
        # Optional safety cap. Any explicit positive cap is honoured and
        # reported so the user knows truncation is active. PANDA's p100 at
        # native length is 2686, so caps below that do truncate by request.
        if n_max is not None and n_max > 0:
            self._safety_cap = n_max
            if n_max < 4096:
                import warnings
                warnings.warn(
                    f"PandaSlideDataset: applying n_max={n_max} below the "
                    f"4096-slide safety threshold; this WILL truncate slides "
                    f"with n_real > {n_max} (PANDA p100=2686 at native "
                    f"length). Pre-validation, caps < 4096 were silently "
                    f"ignored — the new behaviour honours the cap exactly.",
                    UserWarning,
                )
        else:
            self._safety_cap = None
        # Per-worker RNG kept for compatibility (no longer used here).
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        with h5py.File(r.h5_path, "r") as f:
            if self.feature_key not in f:
                raise KeyError(
                    f"PANDA slide {r.image_id} is missing feature key "
                    f"{self.feature_key!r} in {r.h5_path}."
                )
            feats = np.asarray(f[self.feature_key][:], dtype=np.float32)   # (N, D)
            coords = np.asarray(f["coords"][:], dtype=np.int64)            # (N, 5)

        # Keep this runtime guard even though enumerate_slides() filters
        # invalid H5s by default. Tests and legacy tools can construct
        # SlideRecord lists directly; failing here with the offending
        # image_id is much easier to debug than a later attention crash.
        if feats.ndim != 2 or feats.shape[1] != self.feature_dim:
            raise ValueError(
                f"PANDA slide {r.image_id} has invalid {self.feature_key} "
                f"shape {tuple(feats.shape)}; expected (N, {self.feature_dim})."
            )
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"PANDA slide {r.image_id} has invalid coords shape "
                f"{tuple(coords.shape)}; expected (N, >=2)."
            )
        if coords.shape[0] != feats.shape[0]:
            raise ValueError(
                f"PANDA slide {r.image_id} has feature/coord length mismatch: "
                f"features={feats.shape[0]}, coords={coords.shape[0]}."
            )
        if feats.shape[0] == 0:
            raise ValueError(
                f"PANDA slide {r.image_id} has zero extracted patches. "
                "Use enumerate_slides(validate_h5=True) so zero-patch H5s "
                "are removed before fold construction."
            )

        n_real = feats.shape[0]

        # Optional safety cap for anomalous slides only. Deterministic
        # coord-sorted subsample so train and eval pass the same set
        # under the cap (no augmentation jitter).
        if self._safety_cap is not None and n_real > self._safety_cap:
            order = np.lexsort((coords[:, 1], coords[:, 0]))
            keep = np.linspace(0, n_real - 1, num=self._safety_cap, dtype=np.int64)
            keep_idx = order[keep]
            feats = feats[keep_idx]
            coords = coords[keep_idx]
            n_real = feats.shape[0]

        # 8-neighbour graph at native length. Indices in [0, n_real).
        nbi, nbm, nbw = build_panda_neighbor_graph(
            coords,
            radius=self.neighbor_radius,
            shell=self.neighbor_shell,
            source=self.neighbor_source,
            shuffle_seed=self.neighbor_shuffle_seed + idx,
            weighting=self.neighbor_weighting,
            weight_sigma=self.neighbor_weight_sigma,
        )

        # mask is all-True at the dataset level; the collate stacks it
        # along the batch axis. Preserved in the dict so downstream code
        # that reads `mask` (e.g. PandaSlideViT.forward) doesn't need to
        # branch on whether it exists.
        mask_all = np.ones((n_real,), dtype=np.bool_)

        # h_local: per-patch local homogeneity, mean cosine similarity
        # with valid 8-neighbours. Used as a per-token gate input under
        # the signed-gate SRP variant (the signed-gate design
        # by design). Computed once per slide here; cheap (≈ 6 M flops on a
        # median ~500-patch slide) and freezes a stable signal that
        # downstream blocks all consume. h_local is intrinsic to the
        # raw input features — it does not depend on layer or training
        # state, so precomputing it is correct and avoids re-doing it
        # at every block forward.
        h_local = _compute_h_local(feats, nbi, nbm)                         # (n_real,)

        return {
            "features":       torch.from_numpy(feats),         # (n_real, D)
            "mask":           torch.from_numpy(mask_all),      # (n_real,) all True
            "coords":         torch.from_numpy(coords[:, :2].astype(np.float32)),
            "neighbor_index": torch.from_numpy(nbi),           # (n_real, 8)
            "neighbor_mask":  torch.from_numpy(nbm),           # (n_real, 8)
            "neighbor_weight": torch.from_numpy(nbw),           # (n_real, K)
            "h_local":        torch.from_numpy(h_local),       # (n_real,) float32
            "label":          torch.tensor(r.isup_grade, dtype=torch.int64),
            "n_real":         int(n_real),
            "image_id":       r.image_id,
            "data_provider":  r.data_provider,
        }


def panda_collate(batch: List[Dict]) -> Dict:
    """
    BS=1 collate: each batch is exactly one slide at its native length.
    Effective batching is achieved via gradient accumulation in the
    trainer (mirroring slide_level / slide_level_srp). Adding the
    leading batch dim to every tensor keeps the model's existing
    (B, N, …) signature working unchanged.

    Asserts batch_size == 1; multi-slide batches would require padding
    to max-in-batch which we deliberately avoid (DESIGN: train and eval
    at native N for parity with CAM17 protocol).
    """
    # Validation: runtime data-contract validation should
    # be a real exception (kept under `python -O`), not an assert.
    if len(batch) != 1:
        raise ValueError(
            f"panda_collate expects batch_size=1 (got {len(batch)}); use "
            "gradient accumulation for effective batch size > 1."
        )
    b = batch[0]
    return {
        "features":       b["features"].unsqueeze(0),         # (1, N, D)
        "mask":           b["mask"].unsqueeze(0),             # (1, N)
        "coords":         b["coords"].unsqueeze(0),           # (1, N, 2)
        "neighbor_index": b["neighbor_index"].unsqueeze(0),   # (1, N, 8)
        "neighbor_mask":  b["neighbor_mask"].unsqueeze(0),    # (1, N, 8)
        "neighbor_weight": b["neighbor_weight"].unsqueeze(0),  # (1, N, K)
        "h_local":        b["h_local"].unsqueeze(0),          # (1, N) float32
        "label":          b["label"].unsqueeze(0),            # (1,)
        "n_real":         [b["n_real"]],
        "image_id":       [b["image_id"]],
        "data_provider":  [b["data_provider"]],
    }


def build_panda_loaders(
    records: Sequence[SlideRecord],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int] | None = None,
    *,
    num_workers: int = 4,
    safety_cap: int | None = None,
    # Legacy kwargs are kept for back-compat with older callers. `batch_size`
    # is ignored because the BS=1 + grad_accum protocol forces one slide per
    # step. `n_max` maps to `safety_cap` when `safety_cap` is None, preserving
    # the intended explicit-token-cap behavior.
    batch_size: int = 1,
    n_max: int | None = None,
    neighbor_radius: int = 1,
    neighbor_shell: str = "cumulative",
    neighbor_source: str = "real",
    neighbor_shuffle_seed: int = 0,
    neighbor_weighting: str = "uniform",
    neighbor_weight_sigma: float = 1.0,
    feature_key: str = "features/uni_v2",
    feature_dim: int = UNI_DIM,
) -> Tuple[DataLoader, ...]:
    """
    Build PANDA DataLoaders.

    Validation: when `test_idx` is provided, returns a
    3-tuple (train, val, test) with all three loaders. The trainer
    selects `best.pt` on `val_loader` and reports final metrics on the
    untouched `test_loader`. Backward-compat: when `test_idx is None`,
    returns the legacy 2-tuple (train, val) — used by the stress tools
    and old callers; the trainer no longer relies on this branch.

    All loaders run at batch_size=1 with native-length per-slide
    tensors; effective batching is the trainer's responsibility via
    gradient accumulation.
    """
    # F5 fix: honour the legacy `n_max` kwarg as a fallback when no
    # explicit `safety_cap` was given. Previously this kwarg was accepted
    # but silently ignored.
    effective_cap = safety_cap if safety_cap is not None else n_max
    train_records = [records[i] for i in train_idx]
    val_records   = [records[i] for i in val_idx]
    ds_kwargs = dict(
        n_max=effective_cap,
        neighbor_radius=neighbor_radius,
        neighbor_shell=neighbor_shell,
        neighbor_source=neighbor_source,
        neighbor_shuffle_seed=neighbor_shuffle_seed,
        neighbor_weighting=neighbor_weighting,
        neighbor_weight_sigma=neighbor_weight_sigma,
        feature_key=feature_key,
        feature_dim=feature_dim,
    )
    train_ds = PandaSlideDataset(train_records, train=True, **ds_kwargs)
    val_ds   = PandaSlideDataset(val_records, train=False, **ds_kwargs)

    common = dict(num_workers=num_workers,
                  pin_memory=True,
                  persistent_workers=(num_workers > 0))
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=panda_collate, drop_last=False, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=panda_collate, drop_last=False, **common,
    )

    if test_idx is None:
        return train_loader, val_loader

    test_records = [records[i] for i in test_idx]
    test_ds = PandaSlideDataset(test_records, train=False, **ds_kwargs)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=panda_collate, drop_last=False, **common,
    )
    return train_loader, val_loader, test_loader
