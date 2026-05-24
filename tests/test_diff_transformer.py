"""Tests for the reported Diff Transformer comparator."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from slide_level_srp.src.diff_transformer import (  # noqa: E402
    NystromDiffTransformerAggregator,
    NystromDifferentialAttention,
    lambda_init_fn,
)
from slide_level_srp.train import _ABLATIONS, _build_model, _model_forward  # noqa: E402


def test_diff_attention_forward_shape_and_finite_values():
    torch.manual_seed(7)
    attn = NystromDifferentialAttention(
        dim=192,
        depth_index=0,
        baseline_num_heads=6,
        num_landmarks=8,
        pinv_iterations=2,
    )
    x = torch.randn(2, 21, 192)
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert abs(attn.lambda_init - lambda_init_fn(0)) < 1e-12
    assert attn.num_heads == 3
    assert attn.head_dim == 32


def test_diff_aggregator_forward_shape_and_even_head_contract():
    torch.manual_seed(13)
    model = NystromDiffTransformerAggregator(
        in_dim=64,
        embed_dim=192,
        depth=2,
        num_heads=6,
        num_landmarks=8,
        num_classes=4,
        drop_path_rate=0.0,
        pinv_iterations=2,
        checkpoint_mode="off",
    )
    logits = model(torch.randn(1, 23, 64))
    assert logits.shape == (1, 4)
    assert torch.isfinite(logits).all()

    try:
        NystromDiffTransformerAggregator(embed_dim=192, num_heads=5)
    except ValueError as exc:
        assert "even" in str(exc)
    else:
        raise AssertionError("odd baseline head count should be rejected")


def test_train_dispatch_builds_diff_backend_without_srp_tensors():
    class Args:
        run_name = "diff_dispatch_test"
        in_dim = 64
        embed_dim = 192
        depth = 2
        num_heads = 6
        num_landmarks = 8
        num_classes = 3
        drop_path = 0.0
        pinv_iterations = 2
        checkpoint_mode = "off"
        ln_specialization = "shared"
        layerscale_init = 0.0
        no_ppeg = False

    spec = _ABLATIONS["diff_transformer"]
    assert spec["backend"] == "diff"
    model = _build_model(Args, spec["backend"], spec, torch.device("cpu"))
    batch = {"features": torch.randn(1, 17, 64)}
    logits = _model_forward(model, batch, "diff", torch.device("cpu"), ablation_spec=spec)
    assert logits.shape == (1, 3)
    assert torch.isfinite(logits).all()
