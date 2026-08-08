"""
PANDA zero-patch H5 guardrail tests.

The production PANDA extractor can leave an H5 present but empty. That
case must be removed before fold construction; otherwise positional
variant jobs fail only when the DataLoader reaches the bad slide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_panda import PandaSlideDataset, SlideRecord, enumerate_slides


def _write_panda_h5(path: Path, n_patches: int) -> None:
    """Create the minimal PANDA H5 schema used by src.data_panda."""
    with h5py.File(path, "w") as f:
        features = f.create_group("features")
        features.create_dataset(
            "uni_v2",
            data=np.zeros((n_patches, 1536), dtype=np.float32),
        )
        f.create_dataset(
            "coords",
            data=np.zeros((n_patches, 5), dtype=np.int64),
        )


def test_enumerate_slides_filters_zero_patch_h5_before_folds(tmp_path: Path) -> None:
    """A present-but-empty H5 must not become a SlideRecord."""
    h5_dir = tmp_path / "patches"
    h5_dir.mkdir()
    _write_panda_h5(h5_dir / "valid_slide.h5", n_patches=2)
    _write_panda_h5(h5_dir / "zero_slide.h5", n_patches=0)

    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "image_id,data_provider,isup_grade,gleason_score\n"
        "valid_slide,karolinska,1,3+4\n"
        "zero_slide,radboud,0,0+0\n",
        encoding="utf-8",
    )

    records = enumerate_slides(csv_path=csv_path, h5_dir=h5_dir)

    assert [r.image_id for r in records] == ["valid_slide"]


def test_dataset_rejects_direct_zero_patch_record(tmp_path: Path) -> None:
    """Direct SlideRecord construction should fail with a clear message."""
    h5_path = tmp_path / "zero_slide.h5"
    _write_panda_h5(h5_path, n_patches=0)
    ds = PandaSlideDataset([
        SlideRecord(
            image_id="zero_slide",
            data_provider="radboud",
            isup_grade=0,
            h5_path=str(h5_path),
        )
    ])

    with pytest.raises(ValueError, match="zero extracted patches"):
        _ = ds[0]
