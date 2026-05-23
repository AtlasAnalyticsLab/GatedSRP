from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_panda import SlideRecord, build_panda_global_seed_splits


def _records(per_stratum: int = 10) -> list[SlideRecord]:
    records: list[SlideRecord] = []
    for provider in ("karolinska", "radboud"):
        for grade in range(6):
            for i in range(per_stratum):
                records.append(
                    SlideRecord(
                        image_id=f"{provider}_{grade}_{i}",
                        data_provider=provider,
                        isup_grade=grade,
                        h5_path=f"/tmp/{provider}_{grade}_{i}.h5",
                    )
                )
    return records


def test_panda_global_seed_split_is_disjoint_and_stratified() -> None:
    """PANDA global holdout should mirror the slide-level 70/10/20 idea."""
    records = _records(per_stratum=10)
    split = build_panda_global_seed_splits(records, global_seed=42)

    train = set(split.train_idx)
    val = set(split.val_idx)
    test = set(split.test_idx)

    assert train
    assert val
    assert test
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == len(records)

    # With 10 slides in every provider/ISUP bucket and default fractions, each
    # bucket contributes 7 train, 1 validation, and 2 test slides.
    assert set(split.stratum_counts) == {
        f"{provider}|isup{grade}"
        for provider in ("karolinska", "radboud")
        for grade in range(6)
    }
    for counts in split.stratum_counts.values():
        assert counts == {"train": 7, "val": 1, "test": 2, "total": 10}


def test_panda_global_seed_changes_membership() -> None:
    """Different global seeds should change the held-out membership."""
    records = _records(per_stratum=10)
    split_a = build_panda_global_seed_splits(records, global_seed=42)
    split_b = build_panda_global_seed_splits(records, global_seed=43)

    assert set(split_a.test_idx) != set(split_b.test_idx)
    assert set(split_a.val_idx) != set(split_b.val_idx)
