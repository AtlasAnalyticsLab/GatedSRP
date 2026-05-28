"""
Dataset extension for SRP: neighbor index + h_morph computed per slide.

This module layers on top of slide_level/src/data.py rather than
replacing it — the underlying SlideFeatureDataset is reused as-is to
guarantee identical features/coords/labels/fold behavior vs. stage 2.

Additions in this layer:

  1. `build_neighbor_index(coords, stride)` — compute per-patch 3x3
     spatial neighbor indices from the slide's `/coords` using the
     known patch stride (512 level-0 pixels at 40x per CAMELYON17 h5;
     see the slide-level baseline implementation). Missing cells are represented by
     index -1 + mask False.

  2. `compute_h_morph(features, neighbor_index, neighbor_mask)` — per-patch
     local homogeneity h^morph_i = (1/|N(i)|) Σ_{j in N(i)} cos(u_i, u_j),
     where u_* are the raw UNI v1 features. Computed from the frozen
     feature extractor (UNI is never updated in this implementation), so h^morph
     is strictly slide-intrinsic and can be cached per-slide.

  3. `SRPSlideFeatureDataset` — wraps a stage-2 SlideFeatureDataset and
     adds `neighbor_index`, `neighbor_mask`, and `h_morph` to each
     returned dict.

  4. `srp_slide_collate` — extends stage-2 `slide_collate` with the three
     new fields. batch_size=1 only (same as stage 2).

Caching: for fixed-beta we keep computation on-the-fly. Each slide's
neighbor index takes ~10-50 ms and h_morph ~50-200 ms at p100
(N = 37 666, d = 1024) — negligible vs. the forward pass. If this ever
becomes a bottleneck we can cache to disk under /tmp.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Stage-2 data pipeline we reuse verbatim.
from slide_level.src.data import (
    FoldAssignment,
    N_MAX_SAFETY,
    SlideRecord,
    _deterministic_subsample,
    build_fold_assignments,
    enumerate_slides,
    filter_records,
)


# CAMELYON17 patch stride at level-0 (40x) used by the released protocol. The
# /coords field stores level-0 pixel coordinates; the integer patch
# grid is simply coords // PATCH_STRIDE_L0.
PATCH_STRIDE_L0 = 512
_NEIGHBOR_SHELLS = ("cumulative", "ring")
_NEIGHBOR_SOURCES = ("real", "shuffled")
_NEIGHBOR_WEIGHTING = ("uniform", "gaussian", "inverse_distance")


def neighbor_radius_from_window(neighbor_window: int) -> int:
    """Convert an odd window size such as 3/5/7 to radius 1/2/3."""
    if neighbor_window < 3 or neighbor_window % 2 != 1:
        raise ValueError(
            f"neighbor_window must be an odd integer >= 3, got {neighbor_window}"
        )
    return (int(neighbor_window) - 1) // 2


def neighbor_offsets(radius: int = 1, shell: str = "cumulative") -> list[tuple[int, int]]:
    """Return deterministic grid offsets for the requested neighborhood.

    `cumulative` means every offset inside the square Chebyshev ball
    of the radius, excluding self. `ring` keeps only the outer
    Chebyshev shell. The radius=1 cumulative order is the historical
    3x3 order, preserving legacy tests and run behavior.
    """
    if radius < 1:
        raise ValueError(f"neighbor radius must be >= 1, got {radius}")
    if shell not in _NEIGHBOR_SHELLS:
        raise ValueError(f"neighbor shell must be one of {_NEIGHBOR_SHELLS}, got {shell!r}")
    offsets: list[tuple[int, int]] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            dist = max(abs(dx), abs(dy))
            if shell == "ring" and dist != radius:
                continue
            offsets.append((dx, dy))
    return offsets


def neighbor_distance_weights(
    offsets: Sequence[tuple[int, int]],
    weighting: str = "uniform",
    sigma: float = 1.0,
) -> np.ndarray:
    """Per-offset weights for local-neighborhood common-mode means."""
    if weighting not in _NEIGHBOR_WEIGHTING:
        raise ValueError(
            f"neighbor weighting must be one of {_NEIGHBOR_WEIGHTING}, got {weighting!r}"
        )
    if sigma <= 0.0:
        raise ValueError(f"neighbor_weight_sigma must be positive, got {sigma}")
    if weighting == "uniform":
        return np.ones((len(offsets),), dtype=np.float32)
    d2 = np.asarray([dx * dx + dy * dy for dx, dy in offsets], dtype=np.float32)
    if weighting == "gaussian":
        return np.exp(-d2 / (2.0 * float(sigma) * float(sigma))).astype(np.float32)
    # inverse_distance: diagonal offsets naturally get a smaller weight
    # than orthogonal offsets; eps prevents the impossible zero-distance
    # self-offset from causing a divide-by-zero if this function is reused.
    return (1.0 / np.sqrt(np.maximum(d2, 1e-12))).astype(np.float32)


def _shuffle_neighbor_slots(
    neighbor_index: np.ndarray,
    neighbor_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Replace valid neighbor identities while preserving each row count.

    The mask and valid-slot positions are left untouched, so a 5-neighbor
    edge patch still has exactly five neighbor slots and a corner still
    has three. This is a locality-control ablation: model capacity and
    count distribution remain fixed while spatial identity is broken.
    """
    n, _k = neighbor_index.shape
    out = np.full_like(neighbor_index, -1)
    if n <= 1:
        return out
    rng = np.random.default_rng(seed)
    pool = np.arange(n, dtype=np.int64)
    for i in range(n):
        valid_slots = np.flatnonzero(neighbor_mask[i])
        count = int(valid_slots.size)
        if count == 0:
            continue
        candidates = pool[pool != i]
        replace = count > candidates.size
        picked = rng.choice(candidates, size=count, replace=replace)
        out[i, valid_slots] = picked.astype(np.int64)
    return out


def build_neighbor_graph(
    coords: np.ndarray,                  # (N, 2) int64 level-0 pixel coords
    stride: int = PATCH_STRIDE_L0,
    radius: int = 1,
    shell: str = "cumulative",
    source: str = "real",
    shuffle_seed: int = 0,
    weighting: str = "uniform",
    weight_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-patch spatial neighbor indices, mask, and weights.

    Each patch `i` at integer grid position `g_i = coords_i // stride`
    looks at the deterministic offsets requested by `(radius, shell)`.
    For each offset, we test whether another patch `j` lives at
    `g_i + offset`; if yes, neighbor_index[i, k] = j and mask is True;
    if no, the slot is -1/False. `source='shuffled'` keeps the same
    mask but replaces valid neighbor identities with random same-slide
    patches, preserving valid-neighbor count per token.

    The offset order is fixed and deterministic so unit tests can check
    specific positions.

    Args:
      coords  (N, 2) int64. coords[:, 0] and coords[:, 1] are the two
              pixel coordinates; the ordering (x-first vs y-first) does
              NOT affect SRP semantics because the 3x3 window is
              rotation-symmetric.
      stride  integer pixel stride per patch.

    Returns (neighbor_index, neighbor_mask, neighbor_weight):
      neighbor_index: (N, K) int64, -1 at invalid slots.
      neighbor_mask:  (N, K) bool, True at valid slots.
      neighbor_weight:(N, K) float32, 0 at invalid slots.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (N, 2); got {coords.shape}")
    if source not in _NEIGHBOR_SOURCES:
        raise ValueError(f"neighbor source must be one of {_NEIGHBOR_SOURCES}, got {source!r}")
    N = coords.shape[0]
    grid = coords // stride                        # (N, 2) int
    # Dict lookup from (gx, gy) -> patch index. Keys are tuples of ints.
    coord_to_idx: Dict[tuple, int] = {}
    for i in range(N):
        key = (int(grid[i, 0]), int(grid[i, 1]))
        # If two patches map to the same grid cell (possible if coords
        # are ambiguous), keep the first one — later ones are treated as
        # absent from neighbor lookups. This is vanishingly rare on
        # CAMELYON17 and the alternative (collision lists) would require
        # variable-fanout neighborhoods.
        if key not in coord_to_idx:
            coord_to_idx[key] = i

    offsets = neighbor_offsets(radius=radius, shell=shell)
    K = len(offsets)
    neighbor_index = np.full((N, K), -1, dtype=np.int64)
    neighbor_mask = np.zeros((N, K), dtype=np.bool_)
    slot_weights = neighbor_distance_weights(
        offsets, weighting=weighting, sigma=weight_sigma,
    )

    # Python-level loop is OK here: O(8N) and dominated by h5 I/O.
    # Using coord_to_idx keeps the inner operation O(1).
    for i in range(N):
        gx = int(grid[i, 0])
        gy = int(grid[i, 1])
        for k, (dx, dy) in enumerate(offsets):
            j = coord_to_idx.get((gx + dx, gy + dy))
            if j is not None and j != i:
                neighbor_index[i, k] = j
                neighbor_mask[i, k]  = True

    if source == "shuffled":
        neighbor_index = _shuffle_neighbor_slots(
            neighbor_index, neighbor_mask, seed=int(shuffle_seed),
        )
    neighbor_weight = np.broadcast_to(slot_weights[None, :], (N, K)).copy()
    neighbor_weight = neighbor_weight * neighbor_mask.astype(np.float32)
    return neighbor_index, neighbor_mask, neighbor_weight


def build_neighbor_index(
    coords: np.ndarray,                  # (N, 2) int64 level-0 pixel coords
    stride: int = PATCH_STRIDE_L0,
    radius: int = 1,
    shell: str = "cumulative",
    source: str = "real",
    shuffle_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper returning only index and mask."""
    neighbor_index, neighbor_mask, _neighbor_weight = build_neighbor_graph(
        coords,
        stride=stride,
        radius=radius,
        shell=shell,
        source=source,
        shuffle_seed=shuffle_seed,
        weighting="uniform",
    )
    return neighbor_index, neighbor_mask


def compute_h_morph(
    features: np.ndarray,                # (N, D) float32, raw UNI v1
    neighbor_index: np.ndarray,          # (N, 8) int64, -1 for invalid
    neighbor_mask: np.ndarray,           # (N, 8) bool
) -> np.ndarray:
    """
    Per-patch local homogeneity from FROZEN RAW UNI FEATURES.

    h^morph_i = (1/|N(i)|) Σ_{j in N(i)} cos(u_i, u_j)

    Because UNI is used as a frozen extractor and never fine-tuned
    during fixed-beta training, h^morph is strictly slide-intrinsic —
    identical across ablations, epochs, seeds, and optimization steps.

    Args:
      features       (N, D) float32 raw UNI features as read from h5.
      neighbor_index (N, 8) long.
      neighbor_mask  (N, 8) bool.

    Returns h_morph: (N,) float32. For fully-isolated patches (|N|=0)
    the value is 0.0 by convention; downstream the gate clamps to [0,1]
    anyway.
    """
    assert features.ndim == 2
    N, D = features.shape
    # Normalize once.
    norms = np.linalg.norm(features, axis=-1, keepdims=True)    # (N, 1)
    u_norm = features / np.maximum(norms, 1e-12)                # (N, D)

    # Gather neighbor rows. Clamp -1 to 0 — the mask will zero these out.
    safe_idx = np.clip(neighbor_index, 0, N - 1)                # (N, 8)
    u_neigh = u_norm[safe_idx]                                   # (N, 8, D)
    # Per-neighbor cosines: dot product of unit vectors.
    cos_per_neighbor = (u_norm[:, None, :] * u_neigh).sum(axis=-1)  # (N, 8)
    # Mask + mean.
    mask_f = neighbor_mask.astype(np.float32)
    cnt = mask_f.sum(axis=-1)                                    # (N,)
    # Guard against |N|=0 (divide-by-zero): set h=0 there.
    safe_cnt = np.where(cnt > 0, cnt, 1.0)
    h_morph = (cos_per_neighbor * mask_f).sum(axis=-1) / safe_cnt
    # Where cnt == 0, force h = 0 so the gate is identity-like for
    # fully-isolated patches.
    h_morph = np.where(cnt > 0, h_morph, 0.0)
    return h_morph.astype(np.float32)


class SRPSlideFeatureDataset(Dataset):
    """
    Dataset that returns raw UNI features, coords, label, plus
    SRP-specific tensors (neighbor_index, neighbor_mask, h_morph).

    Layered on top of stage-2 SlideFeatureDataset's data-loading logic —
    the h5 read, dtype handling, and safety subsampling are intentionally
    identical. We re-implement the subsample path inline so the neighbor
    index can be rebuilt from the SUBSAMPLED coords (otherwise indices
    would point to rows that no longer exist).
    """
    def __init__(
        self,
        records: Sequence[SlideRecord],
        subsample_cap: Optional[int] = N_MAX_SAFETY,
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
        # Read h5 directly here rather than delegating, so we can build
        # neighbor data from the possibly-subsampled coords.
        import h5py
        with h5py.File(r.h5_path, "r") as f:
            feats = np.asarray(f["features"][:], dtype=np.float32)   # (N, 1024)
            coords = np.asarray(f["coords"][:], dtype=np.int64)      # (N, 2)

        if self.subsample_cap is not None and feats.shape[0] > self.subsample_cap:
            feats, coords = _deterministic_subsample(
                feats, coords, self.subsample_cap,
            )

        neighbor_index, neighbor_mask, neighbor_weight = build_neighbor_graph(
            coords, stride=PATCH_STRIDE_L0,
            radius=self.neighbor_radius,
            shell=self.neighbor_shell,
            source=self.neighbor_source,
            shuffle_seed=self.neighbor_shuffle_seed + idx,
            weighting=self.neighbor_weighting,
            weight_sigma=self.neighbor_weight_sigma,
        )
        h_morph = compute_h_morph(feats, neighbor_index, neighbor_mask)

        return {
            "features":       torch.from_numpy(feats),            # (N, 1024)
            "coords":         torch.from_numpy(coords),           # (N, 2)
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


def srp_slide_collate(batch: List[Dict]) -> Dict:
    """
    Collate for SRP. batch_size=1 only (matches stage 2). Emits all
    SRP-specific tensors unconditionally — non-gated ablations simply
    ignore h_morph in their forward path.
    """
    assert len(batch) == 1, (
        f"srp_slide_collate expects batch_size=1 (got {len(batch)})"
    )
    b = batch[0]
    return {
        "features":       b["features"].unsqueeze(0),        # (1, N, 1024)
        "coords":         b["coords"].unsqueeze(0),          # (1, N, 2)
        "neighbor_index": b["neighbor_index"].unsqueeze(0),  # (1, N, 8)
        "neighbor_mask":  b["neighbor_mask"].unsqueeze(0),   # (1, N, 8)
        "neighbor_weight": b["neighbor_weight"].unsqueeze(0), # (1, N, K)
        "h_morph":        b["h_morph"].unsqueeze(0),         # (1, N)
        "label":          b["label"].unsqueeze(0),           # (1,)
        "slide_id":       [b["slide_id"]],
        "patient_id":     [b["patient_id"]],
        "center":         [b["center"]],
        "n_tokens":       [b["n_tokens"]],
    }


def _effective_cap(cap: Optional[int]) -> int:
    """Mirror of stage-2's cap-vs-safety handling."""
    if cap is None:
        return N_MAX_SAFETY
    return min(int(cap), N_MAX_SAFETY)


def build_srp_loaders_for_fold(
    records: Sequence[SlideRecord],
    fold: FoldAssignment,
    num_workers: int = 2,
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
    """
    Build (train, val, test) DataLoaders with SRP-extended samples.
    Same caps / safety-ceiling semantics as stage-2's equivalent.
    """
    tr_cap = _effective_cap(train_cap)
    va_cap = _effective_cap(val_cap)
    te_cap = _effective_cap(test_cap)

    train_ds = SRPSlideFeatureDataset(
        filter_records(records, fold.train_patients), subsample_cap=tr_cap,
        neighbor_radius=neighbor_radius,
        neighbor_shell=neighbor_shell,
        neighbor_source=neighbor_source,
        neighbor_shuffle_seed=neighbor_shuffle_seed,
        neighbor_weighting=neighbor_weighting,
        neighbor_weight_sigma=neighbor_weight_sigma,
    )
    val_ds = SRPSlideFeatureDataset(
        filter_records(records, fold.val_patients), subsample_cap=va_cap,
        neighbor_radius=neighbor_radius,
        neighbor_shell=neighbor_shell,
        neighbor_source=neighbor_source,
        neighbor_shuffle_seed=neighbor_shuffle_seed,
        neighbor_weighting=neighbor_weighting,
        neighbor_weight_sigma=neighbor_weight_sigma,
    )
    test_ds = SRPSlideFeatureDataset(
        filter_records(records, fold.test_patients), subsample_cap=te_cap,
        neighbor_radius=neighbor_radius,
        neighbor_shell=neighbor_shell,
        neighbor_source=neighbor_source,
        neighbor_shuffle_seed=neighbor_shuffle_seed,
        neighbor_weighting=neighbor_weighting,
        neighbor_weight_sigma=neighbor_weight_sigma,
    )

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=num_workers, collate_fn=srp_slide_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=srp_slide_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=srp_slide_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


__all__ = [
    "PATCH_STRIDE_L0",
    "neighbor_radius_from_window",
    "neighbor_offsets",
    "neighbor_distance_weights",
    "build_neighbor_graph",
    "build_neighbor_index",
    "compute_h_morph",
    "SRPSlideFeatureDataset",
    "srp_slide_collate",
    "build_srp_loaders_for_fold",
    # Re-export stage-2 helpers for convenience.
    "enumerate_slides",
    "build_fold_assignments",
    "filter_records",
]
