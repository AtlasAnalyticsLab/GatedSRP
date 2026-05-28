"""
ADP (Atlas of Digital Pathology) data loader for the multi-label
hierarchical patch classification task.

Expected public data layout, which can be overridden with ADP_* environment
variables:

    data/raw/adp/ADP V1.0 Release/
        ADP_EncodedLabels_Release1_Flat.csv  <- 17668 rows × (1 name + 43 labels)
        img_res_1um_bicubic/<patch>.png      <- 17668 RGB PNGs, 272×272 each, 1 µm/px
        splits/
            train.npy   <- (14134,) int32 indices into the CSV
            valid.npy   <- ( 1767,) int32 indices
            test.npy    <- ( 1767,) int32 indices

Label structure:
  43 columns of {0, 1} multi-hot indicators arranged as a 3-level
  hierarchical histological tissue type (HTT) tree. Levels are encoded by
  the dot count in the column name:
      level 1 (9):  E, C, H, S, A, M, N, G, T
      level 2 (20): E.M, E.T, C.D, C.L, C.X, H.E, H.K, H.Y, H.X, S.M,
                    S.C, S.R, A.W, A.M, N.P, N.R, N.G, G.O, G.N, G.X
      level 3 (14): E.M.S, E.M.C, E.T.S, E.T.C, E.T.X, C.D.I, C.D.R,
                    N.G.M, N.G.A, N.G.O, N.G.E, N.G.R, N.G.T, N.G.X

Pruning (one-shot, deterministic):
  Five level-3 labels under the N.G branch have ZERO training positives
  in the official train split:
      N.G.A, N.G.O, N.G.E, N.G.R, N.G.T
  These are dropped at load time; the model emits 38 logits. Pruning is
  applied to the LABEL MATRIX only — the CSV file is untouched.

Splits behavior:
  The official splits are PATCH-level random, not slide-level group splits.
  All 100 source slides appear in train; 99 of them in valid; 97 in test.
  This matches the protocol used in the ADP paper (Hosseini et al. 2019
  CVPR). Mean-AP on these splits is comparable across runs, but the
  numerical values are inflated relative to a slide-level cross-validation
  setup because patches from the same WSI share slide-specific
  artifacts (stain, focus, tissue-fold patterns).
"""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


_DEFAULT_ADP_ROOT = Path("data/raw/adp/ADP V1.0 Release")
ADP_ROOT = Path(os.environ.get("ADP_ROOT", _DEFAULT_ADP_ROOT))
ADP_CSV = Path(os.environ.get("ADP_CSV", ADP_ROOT / "ADP_EncodedLabels_Release1_Flat.csv"))
ADP_IMG_DIR = Path(os.environ.get("ADP_IMG_DIR", ADP_ROOT / "img_res_1um_bicubic"))
ADP_SPLITS_DIR = Path(os.environ.get("ADP_SPLITS_DIR", ADP_ROOT / "splits"))

# Labels with zero positives anywhere in the dataset (verified via direct
# count on the CSV). These five level-3 children of N.G are dead labels
# in Release1 Flat. We keep them out of the model output to avoid
# undefined-AP rows.
ADP_PRUNE_LABELS = ("N.G.A", "N.G.O", "N.G.E", "N.G.R", "N.G.T")


def label_level(name: str) -> int:
    """1-based hierarchy depth: 'E' -> 1, 'E.M' -> 2, 'E.M.S' -> 3."""
    return name.count(".") + 1


def get_label_columns(csv_path: Path | str = ADP_CSV,
                      prune: tuple[str, ...] = ADP_PRUNE_LABELS) -> list[str]:
    """
    Read the CSV header and return the list of LABEL columns (excluding
    'Patch Names'), with `prune` removed.
    """
    df_head = pd.read_csv(csv_path, nrows=0)
    cols = [c for c in df_head.columns if c != "Patch Names"]
    return [c for c in cols if c not in set(prune)]


def auto_prune_zero_train_labels(csv_path: Path | str = ADP_CSV,
                                 train_npy: Path | str | None = None) -> list[str]:
    """
    Returns labels with > 0 positives on the training split. Useful when
    the static prune list above might be wrong for a future release.
    Falls back to all-zero columns if `train_npy` is None.
    """
    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c != "Patch Names"]
    if train_npy is None:
        sums = df[label_cols].sum(axis=0).astype(int)
    else:
        idx = np.load(train_npy, allow_pickle=True)
        sums = df.iloc[idx][label_cols].sum(axis=0).astype(int)
    return [c for c in label_cols if sums[c] > 0]


# ---------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------
class ADPDataset(Dataset):
    """
    Multi-label patch-level classifier dataset.
    Returns (image_tensor, label_vector_float32) per __getitem__.

    label_vector is shape (n_labels,), 0./1. multi-hot.
    """

    def __init__(
        self,
        csv_path: Path | str,
        img_dir: Path | str,
        split_npy: Path | str | None = None,
        label_cols: list[str] | None = None,
        transform: transforms.Compose | None = None,
        *,
        indices: np.ndarray | list[int] | None = None,
    ) -> None:
        super().__init__()
        if label_cols is None:
            label_cols = get_label_columns(csv_path)
        if (split_npy is None) == (indices is None):
            raise ValueError(
                "ADPDataset requires exactly one split source: either "
                "`split_npy` for the official protocol or `indices` for a "
                "global-seed holdout split."
            )
        df = pd.read_csv(csv_path)
        # Keep the original CSV-row ids as the split unit.  The global-seed
        # holdout protocol varies these row ids directly, while the
        # official protocol loads the same ids from Release1's npy files.
        if indices is None:
            idx = np.load(split_npy, allow_pickle=True).astype(int)
        else:
            idx = np.asarray(indices, dtype=np.int64)
        df = df.iloc[idx].reset_index(drop=True)
        self.row_indices = idx.astype(np.int64, copy=True)
        # Keep raw 0/1 ints for memory; cast to float on retrieval.
        self.names: np.ndarray = df["Patch Names"].to_numpy()
        self.labels: np.ndarray = df[label_cols].to_numpy(dtype=np.float32)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.label_cols = list(label_cols)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = str(self.names[idx])
        img = Image.open(self.img_dir / name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = torch.from_numpy(self.labels[idx])    # already float32
        return img, target


# ---------------------------------------------------------------------
# Augmentations & loader factory
# ---------------------------------------------------------------------
def build_transforms(image_size: int = 272, train: bool = True) -> transforms.Compose:
    """
    Augmentations match the round-2 NCT-CRC recipe (HFlip + VFlip + discrete
    90° rotations + ColorJitter), keeping comparisons across datasets clean.
    Eval is just ToTensor + Normalize.

    ImageNet normalization is used despite from-scratch training because
    the pixel statistics are similar enough and switching to dataset-
    specific stats would be one more hyperparameter to defend.
    """
    norm = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if not train:
        return transforms.Compose([transforms.ToTensor(), norm])

    # Discrete 90° rotations only (degenerate ranges) — no interpolation
    # artifacts at the patch boundary. Same trick used for NCT-CRC.
    rotations = transforms.RandomChoice([
        transforms.RandomRotation(degrees=(0, 0)),
        transforms.RandomRotation(degrees=(90, 90)),
        transforms.RandomRotation(degrees=(180, 180)),
        transforms.RandomRotation(degrees=(270, 270)),
    ])
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        rotations,
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                               saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        norm,
    ])


def build_loaders(
    batch_size: int = 256,
    num_workers: int = 8,
    image_size: int = 272,
    pin_memory: bool = True,
    csv_path: Path | str = ADP_CSV,
    img_dir: Path | str = ADP_IMG_DIR,
    splits_dir: Path | str = ADP_SPLITS_DIR,
    label_cols: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Returns (train_loader, val_loader, test_loader, label_cols_used).

    label_cols defaults to the 38 post-pruning columns. The exact list is
    returned so the caller can record it as a model output spec / report
    per-class APs against it.
    """
    if label_cols is None:
        label_cols = get_label_columns(csv_path)

    train_idx = np.load(Path(splits_dir) / "train.npy", allow_pickle=True).astype(int)
    val_idx = np.load(Path(splits_dir) / "valid.npy", allow_pickle=True).astype(int)
    test_idx = np.load(Path(splits_dir) / "test.npy", allow_pickle=True).astype(int)
    return build_loaders_from_indices(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        pin_memory=pin_memory,
        csv_path=csv_path,
        img_dir=img_dir,
        label_cols=label_cols,
    )


def build_loaders_from_indices(
    *,
    train_idx: np.ndarray | list[int],
    val_idx: np.ndarray | list[int],
    test_idx: np.ndarray | list[int],
    batch_size: int = 256,
    num_workers: int = 8,
    image_size: int = 272,
    pin_memory: bool = True,
    csv_path: Path | str = ADP_CSV,
    img_dir: Path | str = ADP_IMG_DIR,
    label_cols: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build ADP loaders from explicit CSV-row indices.

    This is the data path used by the global-seed holdout protocol.  It
    intentionally shares the same transforms and DataLoader options as
    `build_loaders()` so the only experimental difference is the split
    membership controlled by `global_seed`.
    """
    if label_cols is None:
        label_cols = get_label_columns(csv_path)

    tf_train = build_transforms(image_size=image_size, train=True)
    tf_eval = build_transforms(image_size=image_size, train=False)

    train_set = ADPDataset(
        csv_path, img_dir, label_cols=label_cols,
        transform=tf_train, indices=train_idx,
    )
    val_set = ADPDataset(
        csv_path, img_dir, label_cols=label_cols,
        transform=tf_eval, indices=val_idx,
    )
    test_set = ADPDataset(
        csv_path, img_dir, label_cols=label_cols,
        transform=tf_eval, indices=test_idx,
    )

    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=(num_workers > 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            drop_last=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             drop_last=False, **common)
    return train_loader, val_loader, test_loader, list(label_cols)


def build_global_seed_split_indices(
    *,
    csv_path: Path | str = ADP_CSV,
    label_cols: list[str] | None = None,
    global_seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
) -> dict[str, np.ndarray]:
    """
    Deterministic ADP patch-level holdout split from one global seed.

    ADP's established protocol is patch-level rather than slide-grouped,
    so the split unit is one CSV row.  We stratify by the full
    post-pruning multilabel signature when enough samples exist.  Tiny
    signatures are kept in train because forcing one rare signature into
    every split can create empty train cells and destabilize BCE training.
    """
    if label_cols is None:
        label_cols = get_label_columns(csv_path)
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(
            "ADP split fractions must sum to 1.0; got "
            f"{train_frac + val_frac + test_frac:.6f}"
        )

    df = pd.read_csv(csv_path)
    labels = df[label_cols].to_numpy(dtype=np.uint8)
    by_signature: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for row_idx, label_row in enumerate(labels):
        by_signature[tuple(int(v) for v in label_row)].append(row_idx)

    rng = np.random.default_rng(int(global_seed))
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for signature in sorted(by_signature.keys()):
        idxs = np.asarray(by_signature[signature], dtype=np.int64)
        rng.shuffle(idxs)
        n = int(idxs.size)
        if n <= 1:
            train_idx.extend(idxs.tolist())
            continue

        # Keep at least one training sample per observed signature.  For
        # moderately sized signatures, insist on a test example; for
        # larger ones, also allocate validation.  This preserves the
        # requested 70/10/20 shape without sacrificing rare-label coverage
        # in the train set.
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n >= 5:
            n_test = max(1, n_test)
        if n >= 10:
            n_val = max(1, n_val)
        while n_test + n_val >= n:
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break

        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test:n_test + n_val].tolist())
        train_idx.extend(idxs[n_test + n_val:].tolist())

    # Shuffle within split so DataLoader ordering is not grouped by
    # multilabel signature.  We still return deterministic arrays.
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return {
        "train": np.asarray(train_idx, dtype=np.int64),
        "val": np.asarray(val_idx, dtype=np.int64),
        "test": np.asarray(test_idx, dtype=np.int64),
    }
