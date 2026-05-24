"""TCGA-to-Atlas UNI-v2 survival dataset adapter.

This module is intentionally separate from the slide-classification adapters.
Survival experiments split by case, train on WSI bags, and evaluate by
case-level risk aggregation so multi-slide cases do not receive extra weight.
"""

from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from slide_level_srp.data_ext import build_neighbor_graph, compute_h_morph


TCGA_SURVIVAL_LABEL_CSV = os.environ.get(
    "TCGA_SURVIVAL_LABEL_CSV",
    "data/labels/tcga_survival/all_matched_survival_labels_long.csv",
)
TCGA_ATLAS_FEATURE_ROOT = os.environ.get(
    "TCGA_FEATURE_ROOT",
    "data/features/tcga-to-atlas",
)
TCGA_FEATURE_KEY = os.environ.get("TCGA_FEATURE_KEY", "features/uni_v2")
TCGA_UNIV2_DIM = 1536
TCGA_SURVIVAL_COHORTS = ("KIRC", "KIRP", "LUAD", "STAD", "UCEC")
_N_MAX_SAFETY_TCGA = 49152


@dataclass(frozen=True)
class TCGASurvivalRecord:
    cohort: str
    endpoint: str
    slide_id: str
    case_id: str
    source_case_id: str
    event: int
    time_days: float
    label_discrete_original: int
    h5_path: str
    wsi_path: str


@dataclass(frozen=True)
class TCGASurvivalFoldAssignment:
    fold_id: int
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]
    all_folds: List[List[str]]
    stratification: str
    split_mode: str = "case_level_5fold"
    global_seed: Optional[int] = None


def canonical_tcga_cohort(cohort: str) -> str:
    """Normalize ``KIRC`` / ``TCGA_KIRC`` / ``TCGA-KIRC`` spellings."""
    text = cohort.strip().upper().replace("-", "_")
    if text.startswith("TCGA_"):
        text = text[len("TCGA_"):]
    if text not in TCGA_SURVIVAL_COHORTS:
        raise ValueError(f"unsupported TCGA survival cohort {cohort!r}")
    return text


def tcga_encoder_from_feature_key(feature_key: str = TCGA_FEATURE_KEY) -> str:
    """Map an H5 feature key to the AtlasPatch encoder folder name.

    Existing TCGA survival jobs use ``features/uni_v2`` and therefore keep the
    historical ``.../40x/uni_v2/20x_256/patches`` path. Encoder-selection
    ablations use the same TCGA-to-Atlas layout with the encoder folder swapped
    to ``medsiglip`` or ``vit_b_16``. Keeping the mapping in one helper avoids
    passing a second CLI flag that could drift from ``--feature_key``.
    """

    normalized = feature_key.strip().strip("/")
    if normalized.startswith("features/"):
        encoder = normalized.split("/", 1)[1]
        if encoder:
            return encoder
    return "uni_v2"


def tcga_feature_dir(
    cohort: str,
    feature_root: str = TCGA_ATLAS_FEATURE_ROOT,
    feature_key: str = TCGA_FEATURE_KEY,
) -> Path:
    cohort_short = canonical_tcga_cohort(cohort)
    encoder = tcga_encoder_from_feature_key(feature_key)
    return (
        Path(feature_root)
        / f"A-TCGA-{cohort_short}"
        / "40x"
        / encoder
        / "20x_256"
        / "patches"
    )


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _slide_id_from_filename(filename: str) -> str:
    return Path(filename).stem


def enumerate_tcga_survival_slides(
    *,
    cohort: str,
    endpoint: str = "OS",
    label_csv: str = TCGA_SURVIVAL_LABEL_CSV,
    feature_root: str = TCGA_ATLAS_FEATURE_ROOT,
    feature_key: str = TCGA_FEATURE_KEY,
    drop_nonpositive_time: bool = True,
    require_h5: bool = True,
) -> List[TCGASurvivalRecord]:
    """Join matched survival labels with TCGA-to-Atlas H5 feature files."""
    cohort_short = canonical_tcga_cohort(cohort)
    cohort_label = f"TCGA_{cohort_short}"
    endpoint = endpoint.strip().upper()
    h5_dir = tcga_feature_dir(cohort_short, feature_root, feature_key)
    if require_h5 and not h5_dir.is_dir():
        raise FileNotFoundError(f"missing TCGA feature dir: {h5_dir}")

    records: list[TCGASurvivalRecord] = []
    seen_slides: set[str] = set()
    with open(label_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("cohort", "").strip().upper() != cohort_label:
                continue
            if row.get("endpoint", "").strip().upper() != endpoint:
                continue
            if drop_nonpositive_time and _as_bool(row.get("has_nonpositive_time", "")):
                continue
            time_days = float(row["time_days"])
            if drop_nonpositive_time and time_days <= 0.0:
                continue
            slide_id = _slide_id_from_filename(row["filename"])
            if slide_id in seen_slides:
                raise RuntimeError(f"duplicate TCGA survival slide row: {slide_id}")
            seen_slides.add(slide_id)
            h5_path = h5_dir / f"{slide_id}.h5"
            if require_h5 and not h5_path.exists():
                continue
            records.append(
                TCGASurvivalRecord(
                    cohort=cohort_short,
                    endpoint=endpoint,
                    slide_id=slide_id,
                    case_id=row["case_barcode"].strip(),
                    source_case_id=row.get("source_case_id", "").strip(),
                    event=int(float(row["event"])),
                    time_days=time_days,
                    label_discrete_original=int(float(row.get("discrete_label", 0))),
                    h5_path=str(h5_path),
                    wsi_path=row.get("wsi_path", "").strip(),
                )
            )
    if not records:
        raise RuntimeError(
            f"no TCGA survival rows for cohort={cohort_short} endpoint={endpoint} "
            f"after joining labels with {h5_dir}"
        )
    _validate_case_label_consistency(records)
    records.sort(key=lambda r: (r.case_id, r.slide_id))
    return records


def _validate_case_label_consistency(records: Sequence[TCGASurvivalRecord]) -> None:
    by_case: dict[str, tuple[int, float]] = {}
    for record in records:
        key = (int(record.event), float(record.time_days))
        previous = by_case.get(record.case_id)
        if previous is None:
            by_case[record.case_id] = key
        elif previous != key:
            raise RuntimeError(
                "inconsistent survival label inside case "
                f"{record.case_id}: {previous} vs {key}"
            )


def _case_rows(records: Sequence[TCGASurvivalRecord]) -> dict[str, TCGASurvivalRecord]:
    out: dict[str, TCGASurvivalRecord] = {}
    for record in records:
        out.setdefault(record.case_id, record)
    return out


def _time_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros_like(values, dtype=np.int64)
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def _case_strata(
    case_rows: dict[str, TCGASurvivalRecord],
    n_folds: int,
) -> tuple[dict[str, str], str]:
    """Build case strata, falling back to event-only when bins get sparse."""
    ids = sorted(case_rows)
    times = np.asarray([case_rows[cid].time_days for cid in ids], dtype=np.float64)
    edges = np.unique(np.quantile(times, [0.25, 0.50, 0.75]))
    time_bins = _time_bin(times, edges)
    event_time = {
        cid: f"event{case_rows[cid].event}_time{int(bin_id)}"
        for cid, bin_id in zip(ids, time_bins)
    }
    counts = Counter(event_time.values())
    if counts and min(counts.values()) >= n_folds:
        return event_time, "event_time_quartile"
    return {cid: f"event{case_rows[cid].event}" for cid in ids}, "event"


def _partition_cases_by_stratum(
    case_ids: Sequence[str],
    strata: dict[str, str],
    *,
    n_parts: int,
    seed: int,
) -> list[list[str]]:
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for cid in case_ids:
        by_stratum[strata[cid]].append(cid)
    rng = np.random.default_rng(seed)
    parts: list[list[str]] = [[] for _ in range(n_parts)]
    for stratum, ids in sorted(by_stratum.items()):
        shuffled = sorted(ids)
        rng.shuffle(shuffled)
        for idx, cid in enumerate(shuffled):
            # Round-robin within each stratum gives every fold a comparable
            # event/censor mix while keeping the only randomness in the seed.
            parts[idx % n_parts].append(cid)
    return [sorted(part) for part in parts]


def build_tcga_survival_fold_assignments(
    records: Sequence[TCGASurvivalRecord],
    *,
    n_folds: int = 5,
    val_frac_within_train_pool: float = 0.125,
    fold_seed: int = 1,
) -> List[TCGASurvivalFoldAssignment]:
    """Build case-level 5-fold splits with an inner validation split."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >=2, got {n_folds}")
    if not (0.0 < val_frac_within_train_pool < 0.5):
        raise ValueError(
            "val_frac_within_train_pool must be in (0, 0.5), got "
            f"{val_frac_within_train_pool}"
        )
    _validate_case_label_consistency(records)
    cases = _case_rows(records)
    if len(cases) < n_folds:
        raise RuntimeError(f"cannot split {len(cases)} cases into {n_folds} folds")
    event_counts = Counter(row.event for row in cases.values())
    if any(count < n_folds for count in event_counts.values()):
        raise RuntimeError(
            f"not enough event/censor cases for {n_folds}-fold split: {dict(event_counts)}"
        )

    strata, stratification = _case_strata(cases, n_folds)
    outer_parts = _partition_cases_by_stratum(
        sorted(cases),
        strata,
        n_parts=n_folds,
        seed=fold_seed,
    )
    out: list[TCGASurvivalFoldAssignment] = []
    for fold_id in range(n_folds):
        test_cases = sorted(outer_parts[fold_id])
        train_pool = sorted(
            cid
            for idx, part in enumerate(outer_parts)
            if idx != fold_id
            for cid in part
        )
        # Validation comes from the train pool and is stratified by the same
        # case-level strata.  The number of parts makes one part roughly 10%
        # of the full cohort (12.5% of the 80% train pool).
        n_val_parts = max(2, int(round(1.0 / val_frac_within_train_pool)))
        n_val_parts = min(n_val_parts, len(train_pool))
        val_parts = _partition_cases_by_stratum(
            train_pool,
            strata,
            n_parts=n_val_parts,
            seed=fold_seed * 1000 + fold_id * 137 + 17,
        )
        val_cases = sorted(val_parts[0])
        train_cases = sorted(cid for cid in train_pool if cid not in set(val_cases))
        if not train_cases or not val_cases or not test_cases:
            raise RuntimeError(f"empty split produced for TCGA survival fold {fold_id}")
        train_set, val_set, test_set = set(train_cases), set(val_cases), set(test_cases)
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise RuntimeError(f"case leakage detected in TCGA survival fold {fold_id}")
        out.append(
            TCGASurvivalFoldAssignment(
                fold_id=fold_id,
                train_cases=train_cases,
                val_cases=val_cases,
                test_cases=test_cases,
                all_folds=outer_parts,
                stratification=stratification,
            )
        )
    return out


def build_tcga_survival_seed_holdout_assignment(
    records: Sequence[TCGASurvivalRecord],
    *,
    global_seed: int,
    n_outer_parts: int = 5,
    val_frac_within_train_pool: float = 0.125,
) -> TCGASurvivalFoldAssignment:
    """Build one reproducible case-level train/val/test split from a global seed.

    The reported classification sweeps use five global seeds where the seed
    controls both split membership and training randomness.  Survival uses the
    same idea here while preserving the approved 5-fold test fraction: each
    seed stratifies cases into five outer parts and uses the first part as the
    held-out test set.  Validation is then drawn only from the remaining train
    pool, so no case can leak across train/val/test.
    """
    if n_outer_parts < 2:
        raise ValueError(f"n_outer_parts must be >=2, got {n_outer_parts}")
    if not (0.0 < val_frac_within_train_pool < 0.5):
        raise ValueError(
            "val_frac_within_train_pool must be in (0, 0.5), got "
            f"{val_frac_within_train_pool}"
        )

    _validate_case_label_consistency(records)
    cases = _case_rows(records)
    if len(cases) < n_outer_parts:
        raise RuntimeError(
            f"cannot split {len(cases)} cases into {n_outer_parts} outer parts"
        )

    event_counts = Counter(row.event for row in cases.values())
    if any(count < n_outer_parts for count in event_counts.values()):
        raise RuntimeError(
            "not enough event/censor cases for global-seed TCGA survival split: "
            f"{dict(event_counts)}"
        )

    strata, stratification = _case_strata(cases, n_outer_parts)
    outer_parts = _partition_cases_by_stratum(
        sorted(cases),
        strata,
        n_parts=n_outer_parts,
        seed=int(global_seed),
    )
    test_cases = sorted(outer_parts[0])
    train_pool = sorted(cid for part in outer_parts[1:] for cid in part)

    # Keep the validation fraction aligned with the original 5-fold survival
    # setting: 12.5% of the 80% train pool is about 10% of all cases.  A
    # different seed stream avoids accidentally making validation membership a
    # deterministic prefix of the held-out test partition.
    n_val_parts = max(2, int(round(1.0 / val_frac_within_train_pool)))
    n_val_parts = min(n_val_parts, len(train_pool))
    val_parts = _partition_cases_by_stratum(
        train_pool,
        strata,
        n_parts=n_val_parts,
        seed=int(global_seed) * 1000 + 17,
    )
    val_cases = sorted(val_parts[0])
    train_cases = sorted(cid for cid in train_pool if cid not in set(val_cases))

    if not train_cases or not val_cases or not test_cases:
        raise RuntimeError(
            f"empty split produced for TCGA survival global_seed={global_seed}"
        )
    train_set, val_set, test_set = set(train_cases), set(val_cases), set(test_cases)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError(
            f"case leakage detected in TCGA survival global_seed={global_seed}"
        )

    return TCGASurvivalFoldAssignment(
        fold_id=-1,
        train_cases=train_cases,
        val_cases=val_cases,
        test_cases=test_cases,
        all_folds=outer_parts,
        stratification=stratification,
        split_mode="global_seed_holdout",
        global_seed=int(global_seed),
    )


def filter_tcga_survival_records(
    records: Sequence[TCGASurvivalRecord],
    case_ids: Iterable[str],
) -> List[TCGASurvivalRecord]:
    keep = set(case_ids)
    return [record for record in records if record.case_id in keep]


def make_time_bin_edges(
    records: Sequence[TCGASurvivalRecord],
    *,
    n_bins: int = 4,
) -> np.ndarray:
    """Derive discrete survival bin edges from uncensored training cases."""
    if n_bins < 2:
        raise ValueError(f"n_bins must be >=2, got {n_bins}")
    case_rows = _case_rows(records)
    event_times = np.asarray(
        [row.time_days for row in case_rows.values() if row.event == 1],
        dtype=np.float64,
    )
    if event_times.size < n_bins:
        # Low-event cohorts can be sparse after fold holdout.  Falling back to
        # all training cases keeps bins finite while metadata records the edges.
        event_times = np.asarray(
            [row.time_days for row in case_rows.values()],
            dtype=np.float64,
        )
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(event_times, quantiles))
    if edges.size == 0:
        midpoint = float(np.median(event_times)) if event_times.size else 1.0
        edges = np.asarray([midpoint], dtype=np.float64)
    return edges.astype(np.float64)


def assign_time_bin(time_days: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, float(time_days), side="right"))


def survival_split_metadata(
    records: Sequence[TCGASurvivalRecord],
    fold: TCGASurvivalFoldAssignment,
    *,
    bin_edges: np.ndarray,
    n_bins: int,
) -> dict:
    by_split = {
        "train": filter_tcga_survival_records(records, fold.train_cases),
        "val": filter_tcga_survival_records(records, fold.val_cases),
        "test": filter_tcga_survival_records(records, fold.test_cases),
    }
    meta = {
        "cohort": records[0].cohort if records else None,
        "endpoint": records[0].endpoint if records else None,
        "fold": fold.fold_id,
        "split_mode": fold.split_mode,
        "fold_seed": None,
        "global_seed": fold.global_seed,
        "stratification": fold.stratification,
        "n_bins": int(n_bins),
        "bin_edges_days": [float(x) for x in bin_edges.tolist()],
        "split_cases": {
            "train": list(fold.train_cases),
            "val": list(fold.val_cases),
            "test": list(fold.test_cases),
        },
    }
    meta["case_counts"] = {
        name: len({r.case_id for r in rows})
        for name, rows in by_split.items()
    }
    meta["slide_counts"] = {name: len(rows) for name, rows in by_split.items()}
    meta["event_counts"] = {
        name: int(sum(1 for cid, row in _case_rows(rows).items() if row.event == 1))
        for name, rows in by_split.items()
    }
    meta["censor_counts"] = {
        name: meta["case_counts"][name] - meta["event_counts"][name]
        for name in by_split
    }
    return meta


class TCGASurvivalDataset(Dataset):
    """One TCGA WSI bag plus case-level survival target."""

    def __init__(
        self,
        records: Sequence[TCGASurvivalRecord],
        *,
        bin_edges: np.ndarray,
        subsample_cap: Optional[int] = _N_MAX_SAFETY_TCGA,
        feature_key: str = TCGA_FEATURE_KEY,
        expected_dim: int = TCGA_UNIV2_DIM,
        neighbor_radius: int = 1,
        neighbor_shell: str = "cumulative",
        neighbor_source: str = "real",
        neighbor_shuffle_seed: int = 0,
        neighbor_weighting: str = "uniform",
        neighbor_weight_sigma: float = 1.0,
    ) -> None:
        self.records = list(records)
        self.bin_edges = np.asarray(bin_edges, dtype=np.float64)
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
            stride = int(h5.attrs.get("patch_size_level0", 256))

        if feats.ndim != 2 or feats.shape[1] != self.expected_dim:
            got = feats.shape[1] if feats.ndim == 2 else "non-2D"
            raise ValueError(
                f"{record.h5_path}: expected {self.expected_dim}-d features "
                f"for {self.feature_key}, got {got}"
            )
        if coords.ndim != 2 or coords.shape[0] != feats.shape[0] or coords.shape[1] < 2:
            raise ValueError(
                f"{record.h5_path}: coords shape {coords.shape} is incompatible with "
                f"features shape {feats.shape}"
            )
        if feats.shape[0] <= 0:
            raise ValueError(f"{record.h5_path}: empty TCGA feature bag")

        if self.subsample_cap is not None and feats.shape[0] > self.subsample_cap:
            # Cap very long UCEC/KIRC slides deterministically by spatial order.
            # This keeps folds reproducible and prevents heavy-tail slides from
            # triggering OOM during paper-table sweeps.
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
            "event": torch.tensor(record.event, dtype=torch.float32),
            "time_days": torch.tensor(record.time_days, dtype=torch.float32),
            "time_bin": torch.tensor(assign_time_bin(record.time_days, self.bin_edges), dtype=torch.int64),
            "slide_id": record.slide_id,
            "patient_id": record.case_id,
            "case_id": record.case_id,
            "source_case_id": record.source_case_id,
            "cohort": record.cohort,
            "endpoint": record.endpoint,
            "n_tokens": int(feats.shape[0]),
        }


def tcga_survival_collate(batch: List[Dict]) -> Dict:
    if len(batch) != 1:
        raise ValueError(f"tcga_survival_collate expects batch_size=1, got {len(batch)}")
    item = batch[0]
    return {
        "features": item["features"].unsqueeze(0),
        "coords": item["coords"].unsqueeze(0),
        "neighbor_index": item["neighbor_index"].unsqueeze(0),
        "neighbor_mask": item["neighbor_mask"].unsqueeze(0),
        "neighbor_weight": item["neighbor_weight"].unsqueeze(0),
        "h_morph": item["h_morph"].unsqueeze(0),
        "event": item["event"].unsqueeze(0),
        "time_days": item["time_days"].unsqueeze(0),
        "time_bin": item["time_bin"].unsqueeze(0),
        "slide_id": [item["slide_id"]],
        "patient_id": [item["patient_id"]],
        "case_id": [item["case_id"]],
        "source_case_id": [item["source_case_id"]],
        "cohort": [item["cohort"]],
        "endpoint": [item["endpoint"]],
        "n_tokens": [item["n_tokens"]],
    }


def build_tcga_survival_loaders_for_fold(
    records: Sequence[TCGASurvivalRecord],
    fold: TCGASurvivalFoldAssignment,
    *,
    bin_edges: np.ndarray,
    num_workers: int = 2,
    subsample_cap: Optional[int] = _N_MAX_SAFETY_TCGA,
    train_cap: Optional[int] = None,
    val_cap: Optional[int] = None,
    test_cap: Optional[int] = None,
    feature_key: str = TCGA_FEATURE_KEY,
    expected_dim: int = TCGA_UNIV2_DIM,
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
        bin_edges=bin_edges,
        feature_key=feature_key,
        expected_dim=expected_dim,
        neighbor_radius=neighbor_radius,
        neighbor_shell=neighbor_shell,
        neighbor_source=neighbor_source,
        neighbor_shuffle_seed=neighbor_shuffle_seed,
        neighbor_weighting=neighbor_weighting,
        neighbor_weight_sigma=neighbor_weight_sigma,
    )
    train_ds = TCGASurvivalDataset(
        filter_tcga_survival_records(records, fold.train_cases),
        subsample_cap=eff_train_cap,
        **ds_kwargs,
    )
    val_ds = TCGASurvivalDataset(
        filter_tcga_survival_records(records, fold.val_cases),
        subsample_cap=eff_val_cap,
        **ds_kwargs,
    )
    test_ds = TCGASurvivalDataset(
        filter_tcga_survival_records(records, fold.test_cases),
        subsample_cap=eff_test_cap,
        **ds_kwargs,
    )
    common = dict(
        num_workers=num_workers,
        collate_fn=tcga_survival_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return (
        DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False, **common),
        DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False, **common),
    )


def tcga_survival_inventory(records: Sequence[TCGASurvivalRecord]) -> dict:
    case_rows = _case_rows(records)
    return {
        "slides": len(records),
        "cases": len(case_rows),
        "events": int(sum(1 for row in case_rows.values() if row.event == 1)),
        "censored": int(sum(1 for row in case_rows.values() if row.event == 0)),
        "multi_slide_cases": int(
            sum(1 for count in Counter(r.case_id for r in records).values() if count > 1)
        ),
    }


__all__ = [
    "TCGA_SURVIVAL_LABEL_CSV",
    "TCGA_ATLAS_FEATURE_ROOT",
    "TCGA_FEATURE_KEY",
    "TCGA_UNIV2_DIM",
    "TCGA_SURVIVAL_COHORTS",
    "TCGASurvivalRecord",
    "TCGASurvivalFoldAssignment",
    "canonical_tcga_cohort",
    "tcga_encoder_from_feature_key",
    "tcga_feature_dir",
    "enumerate_tcga_survival_slides",
    "build_tcga_survival_fold_assignments",
    "build_tcga_survival_seed_holdout_assignment",
    "filter_tcga_survival_records",
    "make_time_bin_edges",
    "assign_time_bin",
    "survival_split_metadata",
    "build_tcga_survival_loaders_for_fold",
    "tcga_survival_inventory",
]
