"""Regression tests for the public typed-comparison components."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from slide_level_srp.data_ext import (
    apply_subsample_mode,
    build_neighbor_graph,
)
from scripts.collect_comparison_results import (
    load_classification_metrics,
    load_survival_metrics,
    main as collect_comparison_main,
    primary_classification_metric,
    summarize_metrics,
    summarize_runtime,
)
from scripts.collect_task_results import collect_classification
from slide_level_srp.src.dense_srp_aggregator import DenseAttentionSRPAggregator
from slide_level_srp.src.mil_baselines import (
    DSMILAggregator,
    build_mil_baseline_aggregator,
    dsmil_dual_stream_cross_entropy,
)
from slide_level_srp.src.runtime_profile import RuntimeProfiler
from slide_level_srp.src.srp_correction import PatchSRPCorrection
from slide_level_srp.src.srp_attention import (
    NystromSRPAttention,
    gather_neighbors,
    neighborhood_mean,
    streaming_neighborhood_mean,
)


def _cyclic_neighbors(batch_size: int, n_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dense eight-slot fixture without depending on slide geometry."""
    offsets = (-1, 1, -2, 2, -3, 3, -4, 4)
    index = torch.empty(batch_size, n_tokens, len(offsets), dtype=torch.long)
    for token in range(n_tokens):
        for slot, offset in enumerate(offsets):
            index[:, token, slot] = (token + offset) % n_tokens
    return index, torch.ones_like(index, dtype=torch.bool)


def test_streaming_local_mean_matches_stacked_implementation() -> None:
    """The memory-reduced implementation must preserve the exact operation."""
    torch.manual_seed(8)
    values = torch.randn(2, 3, 13, 7)
    index, mask = _cyclic_neighbors(batch_size=2, n_tokens=13)
    weights = torch.rand(2, 13, 8)

    stacked_values = gather_neighbors(values, index, mask)
    expected = neighborhood_mean(stacked_values, mask, weights)
    actual = streaming_neighborhood_mean(
        values,
        index,
        mask,
        weights,
        slot_chunk=3,
    )

    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.allclose(expected_tensor, actual_tensor, atol=1e-6, rtol=1e-6)


def test_chunked_signed_correction_matches_stacked_forward() -> None:
    """Token chunking must not alter the signed GatedSRP model output."""
    torch.manual_seed(9)
    common = {
        "dim": 48,
        "num_heads": 3,
        "num_landmarks": 8,
        "qkv_bias": True,
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        "srp_mode": "post_agg_signed_gated",
        "beta_patch_mode": "signed_gated",
        "gate_active": True,
    }
    stacked = NystromSRPAttention(
        **common,
        srp_context_impl="stacked",
        srp_correction_chunk_size=0,
    )
    chunked = NystromSRPAttention(
        **common,
        srp_context_impl="streaming_mean",
        srp_correction_chunk_size=5,
    )
    chunked.load_state_dict(stacked.state_dict())
    # Move the zero-initialized gate off identity so the comparison exercises
    # the correction rather than only comparing the shared attention core.
    with torch.no_grad():
        stacked.gate.layer_head_bias.fill_(0.6)
        chunked.gate.layer_head_bias.fill_(0.6)

    n_tokens = 17
    inputs = torch.randn(1, n_tokens + 1, 48)
    index, mask = _cyclic_neighbors(batch_size=1, n_tokens=n_tokens)
    is_real = torch.zeros(1, n_tokens + 1, dtype=torch.bool)
    is_real[:, 1:] = True
    local_homogeneity = torch.linspace(-0.25, 0.9, n_tokens).view(1, -1)

    stacked.eval()
    chunked.eval()
    with torch.no_grad():
        expected = stacked(
            inputs,
            index,
            mask,
            is_real,
            h_local=local_homogeneity,
        )
        actual = chunked(
            inputs,
            index,
            mask,
            is_real,
            h_local=local_homogeneity,
        )
    assert torch.allclose(expected, actual, atol=2e-5, rtol=2e-5)


def test_random_retained_sampling_is_paired_and_seeded() -> None:
    """A slide/run pair must always retain the same source patch rows."""
    features = np.arange(80, dtype=np.float32).reshape(20, 4)
    coords = np.arange(40, dtype=np.int64).reshape(20, 2)

    first = apply_subsample_mode(
        features,
        coords,
        cap=8,
        mode="random_retained",
        seed=42,
        slide_key="slide-a|patient-a",
    )
    repeated = apply_subsample_mode(
        features,
        coords,
        cap=8,
        mode="random_retained",
        seed=42,
        slide_key="slide-a|patient-a",
    )
    different_seed = apply_subsample_mode(
        features,
        coords,
        cap=8,
        mode="random_retained",
        seed=43,
        slide_key="slide-a|patient-a",
    )

    assert np.array_equal(first[0], repeated[0])
    assert np.array_equal(first[1], repeated[1])
    assert not np.array_equal(first[0], different_seed[0])


def test_nearest_retained_graph_uses_closest_coordinates() -> None:
    """The retained-patch graph should order neighbors by physical distance."""
    coords = np.asarray([[0, 0], [512, 0], [1536, 0], [0, 2048]], dtype=np.int64)
    index, mask, weights = build_neighbor_graph(
        coords,
        stride=512,
        radius=1,
        source="nearest_retained",
    )

    assert index.shape == (4, 8)
    assert mask[0].sum() == 3
    assert index[0, 0] == 1
    assert index[0, 1] == 2
    assert np.all(weights[mask] == 1.0)


def test_runtime_profiler_writes_machine_readable_summary(tmp_path: Path) -> None:
    """Profiling should be optional and emit portable JSON when enabled."""
    profiler = RuntimeProfiler(enabled=True, device=torch.device("cpu"))
    profiler.start("test")
    profiler.stop(n_slides=3)
    output = tmp_path / "runtime_profile.json"
    profiler.write(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["enabled"] is True
    assert payload["phases"]["test"]["measurements"][0]["slides"] == 3
    assert payload["phases"]["test"]["slides_per_second_mean"] > 0.0


def test_architecture_neutral_correction_chunking_preserves_output() -> None:
    """Chunking the public adapter must only change activation memory usage."""
    torch.manual_seed(10)
    unchunked = PatchSRPCorrection(
        12,
        hidden_dim=7,
        delta_scale=2.0,
        correction_chunk_size=0,
        checkpoint_correction=False,
    )
    chunked = PatchSRPCorrection(
        12,
        hidden_dim=7,
        delta_scale=2.0,
        correction_chunk_size=3,
        checkpoint_correction=False,
    )
    chunked.load_state_dict(unchunked.state_dict())
    # Move the identity-initialized gate so the test exercises the correction.
    with torch.no_grad():
        unchunked.gate[-1].bias.fill_(0.4)
        chunked.gate[-1].bias.fill_(0.4)

    tokens = torch.randn(2, 9, 12)
    index, mask = _cyclic_neighbors(batch_size=2, n_tokens=9)
    expected = unchunked(tokens, index, mask)
    actual = chunked(tokens, index, mask)
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_dense_mhsa_gated_srp_forward_is_finite() -> None:
    """The retained-patch dense integration must produce task logits."""
    torch.manual_seed(11)
    model = DenseAttentionSRPAggregator(
        in_dim=12,
        embed_dim=24,
        depth=2,
        num_heads=3,
        num_classes=4,
        checkpoint_mode="off",
        use_srp=True,
    )
    features = torch.randn(1, 9, 12)
    index, mask = _cyclic_neighbors(batch_size=1, n_tokens=9)
    logits = model(
        features,
        neighbor_index=index,
        neighbor_mask=mask,
        h_local=torch.rand(1, 9),
    )
    assert logits.shape == (1, 4)
    assert torch.isfinite(logits).all()


def test_mil_baseline_factories_and_dsmil_loss() -> None:
    """Every advertised MIL baseline must run on a variable-length bag."""
    torch.manual_seed(12)
    features = torch.randn(1, 7, 16)
    for kind in ("abmil", "dsmil", "official_transmil"):
        model = build_mil_baseline_aggregator(
            kind=kind,
            in_dim=16,
            num_classes=3,
        )
        logits = model(features)
        assert logits.shape == (1, 3)
        assert torch.isfinite(logits).all()
        if isinstance(model, DSMILAggregator):
            loss = dsmil_dual_stream_cross_entropy(
                model,
                logits,
                torch.tensor([1]),
            )
            assert torch.isfinite(loss)


def test_runtime_summary_includes_complete_tcga_mean() -> None:
    """The collector should reproduce the cross-cohort resource aggregation."""
    rows = []
    for index, cohort in enumerate(sorted({"KIRC", "KIRP", "LUAD", "STAD", "UCEC"})):
        rows.append({
            "dataset": cohort,
            "method": "Gated SRP",
            "profile_epochs": "3",
            "peak_reserved_gib": float(index + 1),
            "train_wsi_per_second": float(10 + index),
            "test_wsi_per_second": "",
        })
    summary = summarize_runtime(rows, ["dataset", "method", "profile_epochs"])
    tcga_mean = next(row for row in summary if row["dataset"] == "TCGA-5 mean")
    assert tcga_mean["peak_reserved_gib"] == 3.0
    assert tcga_mean["train_wsi_per_second"] == 12.0
    assert tcga_mean["test_wsi_per_second"] == ""
    assert tcga_mean["runs"] == 5


def test_strict_public_collection_skips_restricted_rows(tmp_path: Path) -> None:
    """A public rerun must not require artifacts from the restricted cohort."""
    manifest = tmp_path / "classification.tsv"
    manifest.write_text(
        "dataset\tmethod\tmethod_label\tseed\trun_name\taccess\tcommand\n"
        "cam16\tbaseline\tNA\t42\tpublic_run\tpublic\tpython train.py\n"
        "kgh\tbaseline\tNA\t42\trestricted_run\trestricted\tpython train.py\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "public_run"
    run_dir.mkdir(parents=True)
    np.savez(
        run_dir / "test_artifacts.npz",
        test_metrics=np.asarray(json.dumps({"f1": 0.75, "acc": 0.8, "auc": 0.9})),
    )

    per_run, summary = collect_classification(
        manifest,
        tmp_path / "runs",
        strict=True,
        public_only=True,
    )

    assert [row["run_name"] for row in per_run] == ["public_run"]
    assert len(summary) == 1
    assert summary[0]["dataset"] == "cam16"


def test_comparison_metric_collection_preserves_all_numeric_metrics(
    tmp_path: Path,
) -> None:
    """Typed comparison tables preserve every finite task metric."""
    classification = tmp_path / "classification"
    classification.mkdir()
    np.savez(
        classification / "test_artifacts.npz",
        test_metrics=np.asarray(json.dumps({
            "map_macro": 0.61,
            "f1": 0.72,
            "acc": 0.83,
            "label": "ignored",
        })),
    )
    survival = tmp_path / "survival"
    survival.mkdir()
    (survival / "metrics.json").write_text(
        json.dumps({
            "test": {
                "case_c_index": 0.66,
                "slide_c_index": 0.64,
                "n_cases": 80,
            }
        }),
        encoding="utf-8",
    )

    assert load_classification_metrics(classification) == {
        "map_macro": 0.61,
        "f1": 0.72,
        "acc": 0.83,
    }
    assert load_survival_metrics(survival) == {
        "case_c_index": 0.66,
        "slide_c_index": 0.64,
        "n_cases": 80.0,
    }
    assert primary_classification_metric(
        {"f1": 0.72, "kappa_quad": 0.81},
        "panda_vit4",
    ) == ("kappa_quad", 0.81)

    summary = summarize_metrics(
        [
            {"dataset": "ADP", "method": "MHSA", "metric": "map_macro", "value": 0.6},
            {"dataset": "ADP", "method": "MHSA", "metric": "map_macro", "value": 0.8},
        ],
        ["dataset", "method", "metric"],
    )
    assert summary == [{
        "dataset": "ADP",
        "method": "MHSA",
        "metric": "map_macro",
        "mean": 0.7,
        "std": pytest.approx(2 ** 0.5 / 10),
        "runs": 2,
    }]


def test_comparison_collector_writes_selected_and_all_metric_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI output contract must cover multi-metric comparison rows."""
    manifest = tmp_path / "attention_operators.tsv"
    manifest.write_text(
        "dataset\tmethod\tseed\taccess\trun_name\tcommand\n"
        "ADP\tMHSA\t42\tpublic\tadp_s42\tpython train.py\n"
        "ADP\tMHSA\t43\tpublic\tadp_s43\tpython train.py\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "runs"
    for seed, map_macro, f1 in ((42, 0.60, 0.70), (43, 0.80, 0.74)):
        run_dir = run_root / f"adp_s{seed}"
        run_dir.mkdir(parents=True)
        np.savez(
            run_dir / "test_artifacts.npz",
            test_metrics=np.asarray(json.dumps({
                "map_macro": map_macro,
                "f1": f1,
                "acc": 0.82,
            })),
        )

    out_dir = tmp_path / "collected"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_comparison_results.py",
            str(manifest),
            "--run-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
            "--public-only",
            "--strict",
        ],
    )
    collect_comparison_main()

    expected = {
        "per_seed.tsv",
        "summary.tsv",
        "all_metrics_per_seed.tsv",
        "all_metrics_summary.tsv",
    }
    assert {path.name for path in out_dir.iterdir()} == expected
    with (out_dir / "all_metrics_summary.tsv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        summary = list(csv.DictReader(handle, delimiter="\t"))
    by_metric = {row["metric"]: row for row in summary}
    assert set(by_metric) == {"acc", "f1", "map_macro"}
    assert float(by_metric["map_macro"]["mean"]) == pytest.approx(0.70)
    assert by_metric["map_macro"]["runs"] == "2"
    with (out_dir / "summary.tsv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        primary_summary = list(csv.DictReader(handle, delimiter="\t"))
    assert len(primary_summary) == 1
    assert primary_summary[0]["primary_metric"] == "map_macro"
    assert float(primary_summary[0]["mean"]) == pytest.approx(0.70)
