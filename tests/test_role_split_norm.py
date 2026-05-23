from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.role_split_norm import RoleSplitLayerNorm


def test_shared_mode_matches_layernorm():
    torch.manual_seed(1)
    x = torch.randn(3, 5, 7)
    ref = nn.LayerNorm(7)
    mod = RoleSplitLayerNorm(7, mode="shared")
    mod.copy_shared_weights_(ref)

    assert torch.allclose(mod(x), ref(x), atol=0.0, rtol=0.0)


def test_cls_patch_mode_matches_layernorm_when_affines_are_copied():
    torch.manual_seed(2)
    x = torch.randn(2, 6, 11)
    ref = nn.LayerNorm(11)
    with torch.no_grad():
        # Non-default affine values make the equivalence test meaningful: both
        # split branches must use the copied gamma/beta, not just default ones.
        ref.weight.copy_(torch.linspace(0.5, 1.5, 11))
        ref.bias.copy_(torch.linspace(-0.2, 0.2, 11))

    mod = RoleSplitLayerNorm(11, mode="cls_patch", num_cls_tokens=1)
    mod.copy_shared_weights_(ref)

    assert torch.allclose(mod(x), ref(x), atol=0.0, rtol=0.0)


def test_cls_patch_branches_can_diverge_by_role():
    x = torch.ones(1, 4, 3)
    mod = RoleSplitLayerNorm(3, mode="cls_patch", num_cls_tokens=1)
    with torch.no_grad():
        mod.cls_norm.bias.fill_(1.0)
        mod.patch_norm.bias.fill_(2.0)

    y = mod(x)
    assert torch.allclose(y[:, :1], torch.ones_like(y[:, :1]))
    assert torch.allclose(y[:, 1:], torch.full_like(y[:, 1:], 2.0))


def test_edge_cases_for_empty_role_slices():
    x = torch.randn(2, 1, 5)
    cls_only = RoleSplitLayerNorm(5, mode="cls_patch", num_cls_tokens=1)
    assert cls_only(x).shape == x.shape

    patch_only = RoleSplitLayerNorm(5, mode="cls_patch", num_cls_tokens=0)
    assert patch_only(x).shape == x.shape


def test_role_specific_gradients_are_separate():
    x = torch.randn(2, 4, 5, requires_grad=True)
    mod = RoleSplitLayerNorm(5, mode="cls_patch", num_cls_tokens=1)

    # Weight CLS and patch rows differently so both branches receive gradients
    # and the assertion would catch accidental parameter sharing.
    y = mod(x)
    loss = y[:, :1].sum() + 0.5 * y[:, 1:].sum()
    loss.backward()

    assert mod.cls_norm.weight is not mod.patch_norm.weight
    assert mod.cls_norm.bias is not mod.patch_norm.bias
    assert mod.cls_norm.weight.grad is not None
    assert mod.patch_norm.weight.grad is not None
    assert torch.isfinite(mod.cls_norm.weight.grad).all()
    assert torch.isfinite(mod.patch_norm.weight.grad).all()
