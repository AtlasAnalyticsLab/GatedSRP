"""PANDA TransMIL architecture-family smoke tests.

These tests keep the reported PANDA TransMIL rerun path honest without
touching real data or GPUs.  The important contract is that `--arch transmil`
can instantiate and forward the same four method families used by the other
WSI classification datasets: Standard SA, XSA, Differential Transformer, and
Gated SRP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _parse_train_panda_args(monkeypatch: pytest.MonkeyPatch, *extra: str):
    """Use the real CLI parser so the smoke test covers validation defaults."""
    import slide_level_srp.train_panda as train_panda

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_runner",
            "--run_name",
            "panda_transmil_smoke",
            "--arch",
            "transmil",
            "--embed_dim",
            "24",
            "--num_heads",
            "4",
            "--num_landmarks",
            "4",
            "--drop_path",
            "0.0",
            *extra,
        ],
    )
    return train_panda.parse_args()


def _tiny_batch(n_tokens: int = 5, in_dim: int = 1536) -> dict[str, torch.Tensor | list]:
    """Build one native-length PANDA slide with a valid two-neighbor graph."""
    torch.manual_seed(11)
    neighbor_index = torch.zeros(1, n_tokens, 8, dtype=torch.long)
    neighbor_mask = torch.zeros(1, n_tokens, 8, dtype=torch.bool)
    for idx in range(n_tokens):
        if idx > 0:
            neighbor_index[0, idx, 0] = idx - 1
            neighbor_mask[0, idx, 0] = True
        if idx + 1 < n_tokens:
            neighbor_index[0, idx, 1] = idx + 1
            neighbor_mask[0, idx, 1] = True
    return {
        "features": torch.randn(1, n_tokens, in_dim),
        "mask": torch.ones(1, n_tokens, dtype=torch.bool),
        "neighbor_index": neighbor_index,
        "neighbor_mask": neighbor_mask,
        "neighbor_weight": neighbor_mask.float(),
        "h_local": torch.linspace(-0.5, 0.5, n_tokens).unsqueeze(0),
    }


@pytest.mark.parametrize(
    ("mode", "extra"),
    [
        ("baseline", []),
        ("xsa_all_hard", []),
        ("diff_transformer", []),
        (
            "srp_signed_gated",
            ["--srp_r_target", "knn8", "--delta_scale", "2.0", "--gate_hidden_dim", "8"],
        ),
    ],
)
def test_panda_transmil_paper_modes_forward(monkeypatch, mode: str, extra: list[str]) -> None:
    """Every paper method must produce finite six-class PANDA logits."""
    import slide_level_srp.train_panda as train_panda

    args = _parse_train_panda_args(monkeypatch, "--mode", mode, *extra)
    torch.manual_seed(7)
    model = train_panda.build_model(args)
    model.eval()
    with torch.no_grad():
        logits = train_panda.model_forward(
            model,
            _tiny_batch(),
            torch.device("cpu"),
            args.arch,
            srp_r_target=args.srp_r_target,
            mode=args.mode,
        )
    assert logits.shape == (1, 6)
    assert torch.isfinite(logits).all()


def test_panda_transmil_diff_uses_slide_level_comparator(monkeypatch) -> None:
    """`--arch transmil --mode diff_transformer` must not use the ViT wrapper."""
    import slide_level_srp.train_panda as train_panda
    from slide_level_srp.src.diff_transformer import NystromDiffTransformerAggregator

    args = _parse_train_panda_args(monkeypatch, "--mode", "diff_transformer")
    model = train_panda.build_model(args)
    assert isinstance(model, NystromDiffTransformerAggregator)
