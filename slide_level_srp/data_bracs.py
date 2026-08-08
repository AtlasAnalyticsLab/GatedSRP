"""BRACS UNI-v2 WSI adapter for the SRP TransMIL trainer.

BRACS labels are stored in ``data/labels/bracs/BRACS.xlsx`` by default, or
the path supplied through ``BRACS_XLSX``. The task is WSI-level
classification, so this adapter reads ``WSI_Information`` and uses the
``WSI label`` column, not the ROI label columns. UNI-v2 features are flat
H5 files under ``BRACS_FEATURE_ROOT``.

The workbook is parsed with the Python standard library on purpose.  Several
experiment environments used for this project do not have ``openpyxl``
installed, and launch-time dataset discovery should not depend on an optional
Excel dependency.
"""

from __future__ import annotations

import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from xml.etree import ElementTree as ET

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from slide_level_srp.data_ext import (
    apply_subsample_mode,
    build_neighbor_graph,
    compute_h_morph,
)


BRACS_XLSX = os.environ.get("BRACS_XLSX", "data/labels/bracs/BRACS.xlsx")
BRACS_FEATURE_ROOT = os.environ.get(
    "BRACS_FEATURE_ROOT",
    "data/features/bracs/patches",
)
BRACS_FEATURE_KEY = os.environ.get("BRACS_FEATURE_KEY", "features/uni_v2")
BRACS_DIM = 1536
BRACS_CLASSES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]
BRACS_SHEET = "WSI_Information"
_N_MAX_SAFETY_BRACS = 32768

_MAIN_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
_OFFICE_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


@dataclass(frozen=True)
class BRACSSlideRecord:
    slide_id: str
    patient_id: str
    center: int
    label: int
    label_name: str
    set_name: str
    roi_count: int
    h5_path: str


@dataclass(frozen=True)
class BRACSFoldAssignment:
    train_patients: List[str]
    val_patients: List[str]
    test_patients: List[str]


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    """Read Excel shared strings, including rich-text cells."""
    try:
        raw = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    out: list[str] = []
    for item in root.findall("a:si", _MAIN_NS):
        # Rich strings store one ``t`` per run.  Joining preserves labels and
        # slide IDs without needing Excel formatting support.
        parts = [node.text or "" for node in item.findall(".//a:t", _MAIN_NS)]
        out.append("".join(parts))
    return out


def _worksheet_path(zf: zipfile.ZipFile, sheet_name: str) -> str:
    """Resolve a workbook sheet name to its worksheet XML member."""
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_targets: dict[str, str] = {}
    for rel in rels.findall("r:Relationship", _REL_NS):
        rel_id = rel.attrib.get("Id", "")
        target = rel.attrib.get("Target", "")
        if target.startswith("/"):
            target = target.lstrip("/")
        else:
            target = f"xl/{target}"
        rel_targets[rel_id] = target

    for sheet in workbook.findall(".//a:sheet", _MAIN_NS):
        if sheet.attrib.get("name") != sheet_name:
            continue
        rel_id = sheet.attrib.get(f"{_OFFICE_REL_NS}id", "")
        if rel_id not in rel_targets:
            raise RuntimeError(f"BRACS workbook sheet {sheet_name!r} has no relationship target")
        return rel_targets[rel_id]
    raise RuntimeError(f"BRACS workbook is missing sheet {sheet_name!r}")


def _cell_col_index(cell_ref: str) -> int:
    """Convert an Excel column reference like ``AA12`` to a zero-based index."""
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    if not letters:
        return 0
    idx = 0
    for char in letters:
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


def _clean_excel_number(text: str) -> str:
    """Normalize Excel numeric strings while preserving non-numeric labels."""
    stripped = text.strip()
    if re.fullmatch(r"-?\d+\.0+", stripped):
        return str(int(float(stripped)))
    return stripped


def _cell_value(cell: ET.Element, shared: Sequence[str]) -> str:
    """Return a single worksheet cell as text."""
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", _MAIN_NS)).strip()

    value = cell.find("a:v", _MAIN_NS)
    if value is None or value.text is None:
        return ""

    raw = value.text
    if cell_type == "s":
        idx = int(raw)
        return shared[idx].strip() if 0 <= idx < len(shared) else ""
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return _clean_excel_number(raw)


def _read_xlsx_sheet_dicts(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Read one XLSX sheet into dictionaries using the first non-empty row."""
    with zipfile.ZipFile(path) as zf:
        shared = _xlsx_shared_strings(zf)
        sheet_xml = zf.read(_worksheet_path(zf, sheet_name))
    root = ET.fromstring(sheet_xml)

    rows: list[list[str]] = []
    for row in root.findall(".//a:sheetData/a:row", _MAIN_NS):
        cells: dict[int, str] = {}
        for cell in row.findall("a:c", _MAIN_NS):
            col = _cell_col_index(cell.attrib.get("r", ""))
            cells[col] = _cell_value(cell, shared)
        if not cells:
            continue
        max_col = max(cells)
        values = [cells.get(i, "") for i in range(max_col + 1)]
        if any(v.strip() for v in values):
            rows.append(values)

    if not rows:
        return []

    header = [value.strip() for value in rows[0]]
    out: list[dict[str, str]] = []
    for values in rows[1:]:
        row_dict: dict[str, str] = {}
        for idx, name in enumerate(header):
            if name:
                row_dict[name.strip()] = values[idx].strip() if idx < len(values) else ""
        if any(row_dict.values()):
            out.append(row_dict)
    return out


def read_bracs_wsi_rows(xlsx_path: str = BRACS_XLSX) -> list[dict[str, str]]:
    """Read BRACS WSI rows and validate the columns used by this task."""
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"missing BRACS workbook: {path}")
    rows = _read_xlsx_sheet_dicts(path, BRACS_SHEET)
    required = {"WSI Filename", "Patient Id", "RoI", "WSI label", "Set"}
    if rows:
        missing = required - set(rows[0].keys())
        if missing:
            raise RuntimeError(f"BRACS workbook is missing required WSI columns: {sorted(missing)}")
    return rows


def _parse_roi_count(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def enumerate_bracs_slides(
    xlsx_path: str = BRACS_XLSX,
    feature_root: str = BRACS_FEATURE_ROOT,
    *,
    require_h5: bool = True,
) -> List[BRACSSlideRecord]:
    """Join BRACS WSI labels with UNI-v2 H5 feature files.

    Missing H5 rows are skipped by default, matching the existing KGH/CAM16
    adapters.  The validator reports those rows explicitly before launch.
    """
    h5_root = Path(feature_root)
    if not h5_root.is_dir():
        raise FileNotFoundError(f"missing BRACS feature dir: {h5_root}")

    out: list[BRACSSlideRecord] = []
    seen: set[str] = set()
    class_to_id = {name: idx for idx, name in enumerate(BRACS_CLASSES)}
    for row in read_bracs_wsi_rows(xlsx_path):
        slide_id = row.get("WSI Filename", "").strip()
        if not slide_id:
            continue
        if slide_id in seen:
            raise RuntimeError(f"duplicate BRACS WSI row for slide_id={slide_id}")
        seen.add(slide_id)

        label_name = row.get("WSI label", "").strip()
        if label_name not in class_to_id:
            raise RuntimeError(
                f"BRACS slide {slide_id} has unsupported WSI label {label_name!r}; "
                f"expected one of {BRACS_CLASSES}"
            )

        h5_path = h5_root / f"{slide_id}.h5"
        if require_h5 and not h5_path.exists():
            continue

        patient_id = row.get("Patient Id", "").strip() or slide_id
        out.append(
            BRACSSlideRecord(
                slide_id=slide_id,
                patient_id=patient_id,
                center=0,
                label=class_to_id[label_name],
                label_name=label_name,
                set_name=row.get("Set", "").strip(),
                roi_count=_parse_roi_count(row.get("RoI", "")),
                h5_path=str(h5_path),
            )
        )

    if not out:
        raise RuntimeError(
            f"no BRACS slides found by joining xlsx_path={xlsx_path} with feature_root={h5_root}"
        )
    out.sort(key=lambda r: r.slide_id)
    return out


def _patient_label_vectors(records: Sequence[BRACSSlideRecord]) -> dict[str, np.ndarray]:
    """Return slide-count label vectors per patient group."""
    vectors: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(BRACS_CLASSES), dtype=np.int64))
    for record in records:
        vectors[record.patient_id][record.label] += 1
    return dict(vectors)


def _partition_patient_groups(
    patient_ids: Sequence[str],
    vectors: dict[str, np.ndarray],
    *,
    n_parts: int,
    seed: int,
) -> list[list[str]]:
    """Greedily split patient groups while balancing label vectors.

    ``StratifiedKFold`` is unsafe here because BRACS has repeated patients.
    This deterministic greedy splitter keeps each patient wholly inside one
    fold, then minimizes deviation from the target label distribution.
    """
    if n_parts < 2:
        raise ValueError("n_parts must be at least 2")
    if len(patient_ids) < n_parts:
        raise RuntimeError(f"cannot split {len(patient_ids)} patients into {n_parts} parts")

    rng = np.random.default_rng(seed)
    ids = list(patient_ids)
    total = np.sum([vectors[pid] for pid in ids], axis=0).astype(np.float64)
    target = total / float(n_parts)
    target_size = float(total.sum()) / float(n_parts)
    rarity = 1.0 / np.maximum(total, 1.0)
    shuffled = list(rng.permutation(ids))

    def priority(pid: str) -> tuple[float, int, int, str]:
        vec = vectors[pid]
        present = vec > 0
        # Patients carrying rare or multiple labels are placed first because
        # late assignment has less room to repair rare-class imbalance.
        rare_score = float((vec * rarity).sum())
        return (-rare_score, -int(present.sum()), -int(vec.sum()), pid)

    shuffled.sort(key=priority)
    parts: list[list[str]] = [[] for _ in range(n_parts)]
    part_vectors = [np.zeros(len(BRACS_CLASSES), dtype=np.float64) for _ in range(n_parts)]
    part_sizes = [0.0 for _ in range(n_parts)]

    for pid in shuffled:
        vec = vectors[pid].astype(np.float64)
        fold_order = list(rng.permutation(n_parts))
        scored: list[tuple[float, float, int]] = []
        for part_idx in fold_order:
            trial_vectors = [v.copy() for v in part_vectors]
            trial_sizes = list(part_sizes)
            trial_vectors[part_idx] += vec
            trial_sizes[part_idx] += float(vec.sum())
            mat = np.stack(trial_vectors, axis=0)
            # Label balance dominates; size balance is a mild tie breaker.
            # Denominators avoid over-penalizing naturally small classes.
            label_score = float(np.sum(((mat - target) ** 2) / (target + 1.0)))
            size_score = float(np.sum(((np.asarray(trial_sizes) - target_size) ** 2) / (target_size + 1.0)))
            scored.append((label_score + 0.05 * size_score, trial_sizes[part_idx], part_idx))
        _, _, best_idx = min(scored, key=lambda item: item)
        parts[best_idx].append(pid)
        part_vectors[best_idx] += vec
        part_sizes[best_idx] += float(vec.sum())

    return [sorted(part) for part in parts]


def _patient_subset_label_counts(patient_ids: Iterable[str], vectors: dict[str, np.ndarray]) -> np.ndarray:
    ids = list(patient_ids)
    if not ids:
        return np.zeros(len(BRACS_CLASSES), dtype=np.int64)
    return np.sum([vectors[pid] for pid in ids], axis=0).astype(np.int64)


def _choose_validation_patients(
    train_pool_patients: Sequence[str],
    vectors: dict[str, np.ndarray],
    *,
    val_frac: float,
    seed: int,
) -> list[str]:
    """Choose one patient-grouped validation subset from the train pool."""
    n_val_parts = max(2, int(round(1.0 / val_frac)))
    n_val_parts = min(n_val_parts, len(train_pool_patients))
    pool_counts = _patient_subset_label_counts(train_pool_patients, vectors).astype(np.float64)
    target_counts = pool_counts * float(val_frac)
    target_size = float(pool_counts.sum()) * float(val_frac)

    best: tuple[float, int, int, list[str]] | None = None
    for trial in range(1):
        parts = _partition_patient_groups(
            train_pool_patients,
            vectors,
            n_parts=n_val_parts,
            seed=seed + trial,
        )
        for part in parts:
            if not part or len(part) == len(train_pool_patients):
                continue
            counts = _patient_subset_label_counts(part, vectors).astype(np.float64)
            missing = int(np.sum(counts == 0))
            size = int(counts.sum())
            label_error = float(np.sum(np.abs(counts - target_counts) / (target_counts + 1.0)))
            size_error = abs(float(size) - target_size) / (target_size + 1.0)
            # Any missing class is a strong penalty because validation macro F1
            # and AUC become hard to interpret for a 7-way task.
            score = 1000.0 * missing + label_error + 0.25 * size_error
            candidate = (score, missing, abs(size - int(round(target_size))), list(part))
            if best is None or candidate < best:
                best = candidate

    if best is None:
        raise RuntimeError("failed to choose a BRACS validation split")
    if best[1] > 0:
        counts = _patient_subset_label_counts(best[3], vectors)
        missing_labels = [BRACS_CLASSES[i] for i, count in enumerate(counts) if count == 0]
        raise RuntimeError(
            "could not build a BRACS validation split with every class present; "
            f"missing={missing_labels}, counts={dict(zip(BRACS_CLASSES, counts.tolist()))}"
        )
    return sorted(best[3])


def build_bracs_fold_assignments(
    records: Sequence[BRACSSlideRecord],
    n_folds: int = 5,
    val_frac: float = 0.10,
    fold_seed: int = 0,
) -> List[BRACSFoldAssignment]:
    """Build patient-grouped, label-balanced 5-fold assignments."""
    if not records:
        raise RuntimeError("cannot build BRACS folds from an empty record list")
    if not (0.0 < val_frac < 0.5):
        raise ValueError(f"val_frac should be in (0, 0.5), got {val_frac}")

    vectors = _patient_label_vectors(records)
    patient_ids = sorted(vectors)
    total_counts = _patient_subset_label_counts(patient_ids, vectors)
    for label, count in enumerate(total_counts):
        if count < n_folds:
            raise RuntimeError(
                f"BRACS class {BRACS_CLASSES[label]} has only {count} slides; "
                f"cannot build {n_folds} folds."
            )

    outer_parts = _partition_patient_groups(
        patient_ids,
        vectors,
        n_parts=n_folds,
        seed=fold_seed,
    )
    out: list[BRACSFoldAssignment] = []
    for fold_idx in range(n_folds):
        test_patients = sorted(outer_parts[fold_idx])
        train_pool_patients = sorted(
            pid for idx, part in enumerate(outer_parts) if idx != fold_idx for pid in part
        )
        val_patients = _choose_validation_patients(
            train_pool_patients,
            vectors,
            val_frac=val_frac,
            seed=fold_seed * 1000 + fold_idx * 137 + 17,
        )
        val_set = set(val_patients)
        train_patients = sorted(pid for pid in train_pool_patients if pid not in val_set)

        # Fail early if a future workbook change breaks patient isolation.
        train_set, test_set = set(train_patients), set(test_patients)
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise RuntimeError(f"patient leakage detected while building BRACS fold {fold_idx}")

        out.append(
            BRACSFoldAssignment(
                train_patients=train_patients,
                val_patients=val_patients,
                test_patients=test_patients,
            )
        )
    return out


def build_bracs_global_seed_assignment(
    records: Sequence[BRACSSlideRecord],
    *,
    global_seed: int,
    test_frac: float = 0.20,
    val_frac: float = 0.10,
) -> BRACSFoldAssignment:
    """Build one patient-isolated BRACS holdout split from a global seed.

    BRACS patients can contain WSIs from multiple diagnostic labels.  The
    generic global-seed splitter expects every split unit to belong to one
    stratum, so it cannot safely operate at patient granularity here.  Reusing
    the BRACS patient-label-vector partitioner preserves patient isolation while
    still producing seed-specific, label-balanced train/val/test assignments.
    """
    if not records:
        raise RuntimeError("cannot build a BRACS global-seed split from an empty record list")
    if not (0.0 < test_frac < 0.5):
        raise ValueError(f"test_frac should be in (0, 0.5), got {test_frac}")
    if not (0.0 < val_frac < 0.5):
        raise ValueError(f"val_frac should be in (0, 0.5), got {val_frac}")

    vectors = _patient_label_vectors(records)
    patient_ids = sorted(vectors)
    total_counts = _patient_subset_label_counts(patient_ids, vectors)
    for label, count in enumerate(total_counts):
        if count < 2:
            raise RuntimeError(
                f"BRACS class {BRACS_CLASSES[label]} has only {count} slides; "
                "cannot build a train/test holdout with every class represented."
            )

    # A 20% test split is equivalent to one of five balanced patient partitions.
    # The modulo choice keeps each global seed paired across model arms while the
    # partition seed itself changes the patient assignment from seed to seed.
    n_test_parts = max(2, int(round(1.0 / test_frac)))
    n_test_parts = min(n_test_parts, len(patient_ids))
    test_parts = _partition_patient_groups(
        patient_ids,
        vectors,
        n_parts=n_test_parts,
        seed=int(global_seed),
    )
    test_idx = int(global_seed) % n_test_parts
    test_patients = sorted(test_parts[test_idx])
    train_pool_patients = sorted(
        pid for idx, part in enumerate(test_parts) if idx != test_idx for pid in part
    )
    if not test_patients or not train_pool_patients:
        raise RuntimeError(
            "BRACS global-seed split produced an empty train or test partition; "
            f"global_seed={global_seed}, n_test_parts={n_test_parts}"
        )

    # Validation is selected from the train pool only, so test patients remain
    # fully isolated.  Offset the seed to avoid choosing the validation part from
    # the same RNG ordering used for the outer test partition.
    val_patients = _choose_validation_patients(
        train_pool_patients,
        vectors,
        val_frac=val_frac,
        seed=int(global_seed) * 1000 + 17,
    )
    val_set = set(val_patients)
    train_patients = sorted(pid for pid in train_pool_patients if pid not in val_set)

    train_set, test_set = set(train_patients), set(test_patients)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError(
            "patient leakage detected while building BRACS global-seed split; "
            f"global_seed={global_seed}"
        )
    return BRACSFoldAssignment(
        train_patients=train_patients,
        val_patients=sorted(val_patients),
        test_patients=test_patients,
    )


def _filter_records(records: Sequence[BRACSSlideRecord], patient_ids: Sequence[str]) -> List[BRACSSlideRecord]:
    keep = set(patient_ids)
    return [record for record in records if record.patient_id in keep]


class BRACSSlideDataset(Dataset):
    """Emit one BRACS WSI bag in the schema expected by ``train.py``."""

    def __init__(
        self,
        records: Sequence[BRACSSlideRecord],
        subsample_cap: Optional[int] = _N_MAX_SAFETY_BRACS,
        feature_key: str = BRACS_FEATURE_KEY,
        expected_dim: int = BRACS_DIM,
        neighbor_radius: int = 1,
        neighbor_shell: str = "cumulative",
        neighbor_source: str = "real",
        neighbor_shuffle_seed: int = 0,
        neighbor_weighting: str = "uniform",
        neighbor_weight_sigma: float = 1.0,
        subsample_mode: str = "coord_uniform",
        subsample_seed: int = 0,
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
        self.subsample_mode = subsample_mode
        self.subsample_seed = int(subsample_seed)

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
            raise ValueError(f"{record.h5_path}: empty BRACS feature bag")

        if self.subsample_cap is not None and feats.shape[0] > self.subsample_cap:
            feats, coords = apply_subsample_mode(
                feats,
                coords,
                cap=self.subsample_cap,
                mode=self.subsample_mode,
                seed=self.subsample_seed,
                slide_key=f"BRACS|{record.slide_id}|{record.patient_id}",
            )

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
            "n_tokens": int(feats.shape[0]),
            "label_name": record.label_name,
            "set_name": record.set_name,
            "roi_count": record.roi_count,
        }


def bracs_slide_collate(batch: List[Dict]) -> Dict:
    if len(batch) != 1:
        raise ValueError(f"bracs_slide_collate expects batch_size=1, got {len(batch)}")
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
        "n_tokens": [item["n_tokens"]],
        "label_name": [item["label_name"]],
        "set_name": [item["set_name"]],
        "roi_count": [item["roi_count"]],
    }


def build_bracs_loaders_for_fold(
    records: Sequence[BRACSSlideRecord],
    fold: BRACSFoldAssignment,
    num_workers: int = 2,
    subsample_cap: Optional[int] = _N_MAX_SAFETY_BRACS,
    train_cap: Optional[int] = None,
    val_cap: Optional[int] = None,
    test_cap: Optional[int] = None,
    feature_key: str = BRACS_FEATURE_KEY,
    expected_dim: int = BRACS_DIM,
    neighbor_radius: int = 1,
    neighbor_shell: str = "cumulative",
    neighbor_source: str = "real",
    neighbor_shuffle_seed: int = 0,
    neighbor_weighting: str = "uniform",
    neighbor_weight_sigma: float = 1.0,
    subsample_mode: str = "coord_uniform",
    subsample_seed: int = 0,
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
        subsample_mode=subsample_mode,
        subsample_seed=subsample_seed,
    )
    train_ds = BRACSSlideDataset(_filter_records(records, fold.train_patients), eff_train_cap, **ds_kwargs)
    val_ds = BRACSSlideDataset(_filter_records(records, fold.val_patients), eff_val_cap, **ds_kwargs)
    test_ds = BRACSSlideDataset(_filter_records(records, fold.test_patients), eff_test_cap, **ds_kwargs)
    common = dict(num_workers=num_workers, collate_fn=bracs_slide_collate, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False, **common),
        DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False, **common),
    )


def bracs_label_counts(records: Sequence[BRACSSlideRecord]) -> dict[str, int]:
    """Small helper used by validation/reporting scripts."""
    counts = Counter(record.label_name for record in records)
    return {label: int(counts.get(label, 0)) for label in BRACS_CLASSES}


__all__ = [
    "BRACS_CLASSES",
    "BRACS_DIM",
    "BRACS_FEATURE_KEY",
    "BRACS_FEATURE_ROOT",
    "BRACS_XLSX",
    "BRACSSlideRecord",
    "BRACSFoldAssignment",
    "read_bracs_wsi_rows",
    "enumerate_bracs_slides",
    "build_bracs_fold_assignments",
    "build_bracs_global_seed_assignment",
    "build_bracs_loaders_for_fold",
    "bracs_label_counts",
]
