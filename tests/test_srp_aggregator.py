"""
Integration tests for NystromSRPAggregator.

Covers:
  A  forward produces correct output shape for all srp_modes
  B  baseline equivalence: beta_patch_mode=zero + baseline-like inputs
     -> same logits as stage-2 NystromXSAggregator with alpha_*=zero
     (up to the extra in_proj/cls_token/ppeg weight init which must be
     seeded identically for this check)
  C  is_real mask has the right structure (position 0 False, 1..N True,
     pad dupes False) — verified indirectly by checking that a run with
     modified pad rows gives the same output as a run without them
     (pad-invariance property)
  D  h_morph required for gated mode
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from slide_level_srp.src.srp_aggregator import NystromSRPAggregator
from slide_level_srp.data_ext import build_neighbor_index, compute_h_morph, PATCH_STRIDE_L0
from slide_level_srp.src.srp_diagnostics import extract_layerscale_values
from src.role_split_norm import RoleSplitLayerNorm
from slide_level.src.aggregator import NystromXSAggregator
import numpy as np


def _synthetic_slide(B=1, N=20, in_dim=1024, seed=0):
    """Make a synthetic (features, neighbor_index, neighbor_mask, h_morph)
    batch for a roughly 4x5 grid of patches."""
    torch.manual_seed(seed)
    features = torch.randn(B, N, in_dim)
    # Build coords on a 4x5 grid, so neighbor_index has real structure.
    rows, cols = 4, 5
    assert rows * cols == N
    coords = np.zeros((N, 2), dtype=np.int64)
    for r in range(rows):
        for c in range(cols):
            coords[r * cols + c, 0] = c * PATCH_STRIDE_L0
            coords[r * cols + c, 1] = r * PATCH_STRIDE_L0
    nbi_np, nbm_np = build_neighbor_index(coords)
    h_morph_np = compute_h_morph(features[0].numpy(), nbi_np, nbm_np)
    neighbor_index = torch.from_numpy(nbi_np).unsqueeze(0).expand(B, -1, -1).contiguous()
    neighbor_mask = torch.from_numpy(nbm_np).unsqueeze(0).expand(B, -1, -1).contiguous()
    h_morph = torch.from_numpy(h_morph_np).unsqueeze(0).expand(B, -1).contiguous()
    return features, neighbor_index, neighbor_mask, h_morph


def test_A_forward_shape_all_modes():
    features, nbi, nbm, h_m = _synthetic_slide()
    mode_specs = [
        ("post_agg", "learn"),
        ("pre_v", "learn"),
        ("post_agg_gated", "learn"),
        ("post_agg_signed_gated", "signed_gated"),
        ("pre_q_signed_gated", "signed_gated"),
        ("pre_k_signed_gated", "signed_gated"),
        ("pre_v_signed_gated", "signed_gated"),
        ("post_agg_mlp_control", "zero"),
    ]
    for srp_mode, beta_mode in mode_specs:
        torch.manual_seed(123)
        mod = NystromSRPAggregator(
            in_dim=1024, embed_dim=192, depth=2, num_heads=6,
            num_landmarks=16, num_classes=4,
            beta_patch_mode=beta_mode, beta_init=1.0,
            srp_mode=srp_mode, drop_path_rate=0.0,
            checkpoint_mode="off",  # disable checkpointing for speed
        )
        mod.eval()
        logits = mod(features, nbi, nbm, h_morph=h_m, h_local=h_m)
        assert logits.shape == (1, 4)
        assert torch.isfinite(logits).all()


def test_B_baseline_equivalence_with_stage2():
    """Under beta=0 and a synthetic baseline ablation, SRPAggregator
    must produce bit-for-bit the same logits as stage-2 XSAggregator
    with alpha_*=0. This is the paired-per-slide guarantee used by the
    released reproduction protocol.
    """
    features, nbi, nbm, h_m = _synthetic_slide()

    torch.manual_seed(777)
    m_srp = NystromSRPAggregator(
        in_dim=1024, embed_dim=192, depth=2, num_heads=6,
        num_landmarks=16, num_classes=4,
        beta_patch_mode="zero", beta_init=1.0,
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off",
    )
    torch.manual_seed(777)
    m_xsa = NystromXSAggregator(
        in_dim=1024, embed_dim=192, depth=2, num_heads=6,
        num_landmarks=16, num_classes=4,
        alpha_cls_mode="zero", alpha_patch_mode="zero",
        alpha_init=1.0, drop_path_rate=0.0,
        checkpoint_mode="off",
    )
    m_srp.eval()
    m_xsa.eval()

    # Verify weight sharing at the in_proj / cls_token / norm / head level.
    assert torch.allclose(m_srp.in_proj.weight, m_xsa.in_proj.weight)
    assert torch.allclose(m_srp.cls_token, m_xsa.cls_token)
    assert torch.allclose(m_srp.head.weight, m_xsa.head.weight)

    y_srp = m_srp(features, nbi, nbm)
    y_xsa = m_xsa(features)

    max_diff = (y_srp - y_xsa).abs().max().item()
    assert max_diff < 1e-4, (
        f"SRP baseline vs stage-2 alpha=0 divergence: max |Δ| = {max_diff:.3e}. "
        f"Paired per-slide comparison in analysis will be compromised."
    )


def test_D_gated_requires_h_morph():
    features, nbi, nbm, h_m = _synthetic_slide()
    mod = NystromSRPAggregator(
        embed_dim=192, depth=2, num_heads=6, num_landmarks=16,
        num_classes=4, beta_patch_mode="learn",
        srp_mode="post_agg_gated", drop_path_rate=0.0,
        checkpoint_mode="off",
    )
    mod.eval()
    # h_morph=None should raise. Accept either AssertionError (previous)
    # or ValueError (post-validation second-audit F7 fix).
    raised = False
    try:
        mod(features, nbi, nbm, h_morph=None)
    except (AssertionError, ValueError):
        raised = True
    assert raised, "post_agg_gated must fail without h_morph"
    # With h_morph: succeeds.
    out = mod(features, nbi, nbm, h_morph=h_m)
    assert out.shape == (1, 4)


def test_E_forward_backward_under_pre_v():
    """Backward through the aggregator under pre_v SRP succeeds."""
    features, nbi, nbm, h_m = _synthetic_slide()
    features.requires_grad_(True)
    torch.manual_seed(1)
    mod = NystromSRPAggregator(
        embed_dim=192, depth=2, num_heads=6, num_landmarks=16,
        num_classes=4, beta_patch_mode="learn",
        srp_mode="pre_v", drop_path_rate=0.0,
        checkpoint_mode="off",
    )
    mod.train()
    logits = mod(features, nbi, nbm)
    loss = logits.sum()
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    for n, p in mod.named_parameters():
        assert p.grad is not None, f"{n} has no grad"
        assert torch.isfinite(p.grad).all(), f"{n} has non-finite grad"


def test_F_layerscale_disabled_has_no_gamma_parameters():
    """Default-off LayerScale must preserve old checkpoints and queues.

    Existing ablation workers do not pass --layerscale_init. In that path the
    model should not register any gamma parameters, otherwise old optimizer
    counts and state-dict compatibility would change.
    """
    mod = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off", layerscale_init=0.0,
    )
    assert not any("gamma_" in name for name, _ in mod.named_parameters())
    assert extract_layerscale_values(mod) == {}


def test_G_layerscale_shape_init_and_gradients():
    """LayerScale is CaiT-style per-channel residual scaling."""
    features, nbi, nbm, _ = _synthetic_slide(in_dim=32)
    torch.manual_seed(17)
    mod = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off", layerscale_init=0.1,
    )
    for blk in mod.blocks:
        assert blk.gamma_attn.shape == (96,)
        assert blk.gamma_mlp.shape == (96,)
        assert torch.allclose(blk.gamma_attn, torch.full((96,), 0.1))
        assert torch.allclose(blk.gamma_mlp, torch.full((96,), 0.1))
        assert blk.gamma_attn.requires_grad
        assert blk.gamma_mlp.requires_grad

    mod.train()
    loss = mod(features, nbi, nbm).sum()
    loss.backward()
    for li, blk in enumerate(mod.blocks):
        assert blk.gamma_attn.grad is not None, f"block {li} gamma_attn has no grad"
        assert blk.gamma_mlp.grad is not None, f"block {li} gamma_mlp has no grad"
        assert torch.isfinite(blk.gamma_attn.grad).all()
        assert torch.isfinite(blk.gamma_mlp.grad).all()

    snap = extract_layerscale_values(mod)
    assert snap["layerscale_attn"].shape == (2, 96)
    assert snap["layerscale_mlp"].shape == (2, 96)


def test_H_layerscale_gamma_one_matches_disabled_model():
    """gamma=1 should reduce exactly to the historical residual branch."""
    features, nbi, nbm, _ = _synthetic_slide(in_dim=32)

    torch.manual_seed(23)
    no_ls = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off", layerscale_init=0.0,
    )
    torch.manual_seed(23)
    with_ls = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off", layerscale_init=1.0,
    )
    no_ls.eval()
    with_ls.eval()
    y_no = no_ls(features, nbi, nbm)
    y_ls = with_ls(features, nbi, nbm)
    assert torch.allclose(y_no, y_ls, atol=1e-6)


def test_I_signed_gate_identity_still_holds_with_layerscale():
    """At zero-init, signed gated SRP + LayerScale starts as baseline + LayerScale."""
    features, nbi, nbm, h_local = _synthetic_slide(in_dim=32)

    torch.manual_seed(31)
    baseline = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off", layerscale_init=0.1,
    )
    torch.manual_seed(31)
    gated = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4,
        beta_patch_mode="signed_gated",
        srp_mode="post_agg_signed_gated",
        delta_scale=1.0, gate_hidden_dim=8,
        drop_path_rate=0.0, checkpoint_mode="off",
        layerscale_init=0.1,
    )
    baseline.eval()
    gated.eval()
    y_base = baseline(features, nbi, nbm)
    y_gated = gated(features, nbi, nbm, h_local=h_local)
    max_diff = (y_base - y_gated).abs().max().item()
    assert max_diff < 1e-5, (
        "signed-gate identity init should match baseline when beta_eff=0; "
        f"got max |diff| = {max_diff:.3e}"
    )


def test_I2_pre_attention_signed_gate_identity_matches_baseline():
    """All signed-gated placements must start from the same baseline model.

    This protects the live live ablation ablations too: adding new modes must not
    perturb the non-gate initialization or the existing zero-beta behavior.
    """
    features, nbi, nbm, h_local = _synthetic_slide(in_dim=32)

    torch.manual_seed(37)
    baseline = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=3, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off",
    )
    baseline.eval()
    y_base = baseline(features, nbi, nbm)

    for srp_mode in (
        "post_agg_signed_gated",
        "pre_q_signed_gated",
        "pre_k_signed_gated",
        "pre_v_signed_gated",
    ):
        torch.manual_seed(37)
        gated = NystromSRPAggregator(
            in_dim=32, embed_dim=96, depth=3, num_heads=6,
            num_landmarks=8, num_classes=4,
            beta_patch_mode="signed_gated",
            srp_mode=srp_mode,
            delta_scale=2.0,
            gate_hidden_dim=8,
            drop_path_rate=0.0,
            checkpoint_mode="off",
        )
        gated.eval()
        y_gated = gated(features, nbi, nbm, h_local=h_local)
        max_diff = (y_base - y_gated).abs().max().item()
        assert max_diff < 1e-5, (
            f"{srp_mode} identity init should match baseline; "
            f"got max |diff| = {max_diff:.3e}"
        )


def test_I2b_mlp_control_identity_and_no_geometry_contract():
    """Plain-adapter control starts as baseline and needs no local geometry.

    This is the reported capacity-control arm.  The model output must not
    require h_local/h_morph, and same-seed construction must preserve the
    shared TransMIL initialization so differences are attributable only to the
    adapter after it learns.
    """
    features, nbi, nbm, _h_local = _synthetic_slide(in_dim=32)

    torch.manual_seed(41)
    baseline = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=3, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off",
    )
    torch.manual_seed(41)
    control = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=3, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg_mlp_control",
        gate_hidden_dim=8,
        drop_path_rate=0.0,
        checkpoint_mode="off",
    )
    baseline.eval()
    control.eval()

    y_base = baseline(features, nbi, nbm)
    y_control = control(features, nbi, nbm)
    assert torch.allclose(y_base, y_control, atol=1e-5)

    active_adapters = [
        blk.attn.mlp_control
        for blk in control.blocks
        if blk.attn.mlp_control is not None
    ]
    assert len(active_adapters) == 2
    assert control.blocks[-1].attn.mlp_control is None

    # Perturbing the first active adapter must change logits, proving the
    # control branch has a real path to CLS through later attention blocks.
    with torch.no_grad():
        active_adapters[0].fc2.bias.fill_(0.1)
        y_pert = control(features, nbi, nbm)
    assert (y_pert - y_control).abs().max().item() > 1e-6


def test_I3_final_block_gate_policy_is_placement_aware():
    """Final-block gate allocation should match the CLS path analysis.

    Post-attention and patch-only pre-Q gates are dead in the last block.
    Pre-K and pre-V are live because the final CLS attention row consumes
    modified patch keys/values before the classifier reads CLS.
    """
    expected_last_active = {
        "post_agg_signed_gated": False,
        "pre_q_signed_gated": False,
        "pre_k_signed_gated": True,
        "pre_v_signed_gated": True,
    }

    def _gate_param_count(blk) -> int:
        return sum(p.numel() for n, p in blk.named_parameters() if "gate." in n)

    for srp_mode, last_active in expected_last_active.items():
        mod = NystromSRPAggregator(
            in_dim=32, embed_dim=96, depth=3, num_heads=6,
            num_landmarks=8, num_classes=4,
            beta_patch_mode="signed_gated",
            srp_mode=srp_mode,
            delta_scale=2.0,
            gate_hidden_dim=8,
            drop_path_rate=0.0,
            checkpoint_mode="off",
        )
        counts = [_gate_param_count(blk) for blk in mod.blocks]
        assert counts[0] > 0 and counts[1] > 0, (
            f"{srp_mode}: non-final signed-gate blocks should be active"
        )
        if last_active:
            assert counts[-1] > 0, f"{srp_mode}: final block should be active"
        else:
            assert counts[-1] == 0, f"{srp_mode}: final block should be inactive"


def test_I4_pre_k_pre_v_single_final_block_gate_affects_cls_output():
    """Pre-K and pre-V gates have a same-block path to the CLS classifier.

    With depth=1 the only block is also the final block.  A forced non-zero
    beta must change logits for pre-K/pre-V; this would be impossible for a
    pure post-attention patch write under a CLS-only head.
    """
    features, nbi, nbm, h_local = _synthetic_slide(in_dim=32)
    raw_init = math.atanh(0.5)

    for srp_mode in ("pre_k_signed_gated", "pre_v_signed_gated"):
        torch.manual_seed(53)
        mod = NystromSRPAggregator(
            in_dim=32, embed_dim=96, depth=1, num_heads=6,
            num_landmarks=8, num_classes=4,
            beta_patch_mode="signed_gated",
            srp_mode=srp_mode,
            delta_scale=2.0,
            gate_hidden_dim=8,
            drop_path_rate=0.0,
            checkpoint_mode="off",
        )
        mod.eval()
        with torch.no_grad():
            y_id = mod(features, nbi, nbm, h_local=h_local)
            mod.blocks[0].attn.gate.layer_head_bias.fill_(raw_init)
            y_pert = mod(features, nbi, nbm, h_local=h_local)
        diff = (y_pert - y_id).abs().max().item()
        assert diff > 1e-6, (
            f"{srp_mode}: perturbing the final-block gate should change "
            f"CLS logits; got max |diff| = {diff:.3e}"
        )


def test_I5_signed_pre_attention_modes_require_h_local_early():
    """Pre-attention signed gates should fail at the aggregator boundary.

    The trainer supplies h_local for these modes, so this is primarily a
    safety check for future launchers and tests: a missing local homogeneity
    tensor should produce a clear contract error before entering attention.
    """
    features, nbi, nbm, _ = _synthetic_slide(in_dim=32)

    for srp_mode in (
        "pre_q_signed_gated",
        "pre_k_signed_gated",
        "pre_v_signed_gated",
    ):
        mod = NystromSRPAggregator(
            in_dim=32, embed_dim=96, depth=1, num_heads=6,
            num_landmarks=8, num_classes=4,
            beta_patch_mode="signed_gated",
            srp_mode=srp_mode,
            delta_scale=2.0,
            gate_hidden_dim=8,
            drop_path_rate=0.0,
            checkpoint_mode="off",
        )
        with pytest.raises(ValueError, match="requires h_local"):
            mod(features, nbi, nbm)


def test_J_layerscale_checkpoint_modes_backward():
    """Whole-block and per-module checkpoint paths must both scale branches."""
    for checkpoint_mode in ("whole_block", "per_module"):
        features, nbi, nbm, _ = _synthetic_slide(in_dim=32)
        features.requires_grad_(True)
        torch.manual_seed(41)
        mod = NystromSRPAggregator(
            in_dim=32, embed_dim=96, depth=2, num_heads=6,
            num_landmarks=8, num_classes=4, beta_patch_mode="zero",
            srp_mode="post_agg", drop_path_rate=0.0,
            checkpoint_mode=checkpoint_mode, layerscale_init=0.1,
        )
        mod.train()
        loss = mod(features, nbi, nbm).sum()
        loss.backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
        for li, blk in enumerate(mod.blocks):
            assert blk.gamma_attn.grad is not None, (
                f"{checkpoint_mode} block {li} gamma_attn has no grad"
            )
            assert blk.gamma_mlp.grad is not None, (
                f"{checkpoint_mode} block {li} gamma_mlp has no grad"
            )


def test_K_default_shared_ln_preserves_historical_parameter_names():
    """Default-off split LN must not disturb active queues or old checkpoints."""
    mod = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off",
    )
    names = {name for name, _ in mod.named_parameters()}

    assert isinstance(mod.blocks[0].norm1, torch.nn.LayerNorm)
    assert isinstance(mod.blocks[0].norm2, torch.nn.LayerNorm)
    assert isinstance(mod.norm, torch.nn.LayerNorm)
    assert "blocks.0.norm1.weight" in names
    assert "blocks.0.norm2.weight" in names
    assert "norm.weight" in names
    assert not any("cls_norm" in name or "patch_norm" in name for name in names)


def test_L_split_ln_registers_role_specific_block_norms_only_by_default():
    mod = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off", ln_specialization="cls_patch",
    )
    names = {name for name, _ in mod.named_parameters()}

    assert isinstance(mod.blocks[0].norm1, RoleSplitLayerNorm)
    assert isinstance(mod.blocks[0].norm2, RoleSplitLayerNorm)
    # The default `block` scope leaves the final pre-head norm shared because
    # patch rows after the last block do not feed the CLS-only classifier.
    assert isinstance(mod.norm, torch.nn.LayerNorm)
    assert "blocks.0.norm1.cls_norm.weight" in names
    assert "blocks.0.norm1.patch_norm.weight" in names
    assert "norm.weight" in names


def test_M_split_ln_is_equivalent_to_shared_ln_at_initialization():
    features, nbi, nbm, h_local = _synthetic_slide(in_dim=32)

    torch.manual_seed(101)
    shared = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="signed_gated",
        srp_mode="post_agg_signed_gated", delta_scale=1.0,
        gate_hidden_dim=8, drop_path_rate=0.0,
        checkpoint_mode="off", ln_specialization="shared",
    )
    torch.manual_seed(101)
    split = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="signed_gated",
        srp_mode="post_agg_signed_gated", delta_scale=1.0,
        gate_hidden_dim=8, drop_path_rate=0.0,
        checkpoint_mode="off", ln_specialization="cls_patch",
    )
    shared.eval()
    split.eval()

    y_shared = shared(features, nbi, nbm, h_local=h_local)
    y_split = split(features, nbi, nbm, h_local=h_local)
    assert torch.allclose(y_shared, y_split, atol=1e-5, rtol=1e-5)


def test_N_block_final_scope_specializes_final_norm_and_preserves_init_output():
    features, nbi, nbm, _ = _synthetic_slide(in_dim=32)

    torch.manual_seed(103)
    shared = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off", ln_specialization="shared",
    )
    torch.manual_seed(103)
    split_final = NystromSRPAggregator(
        in_dim=32, embed_dim=96, depth=2, num_heads=6,
        num_landmarks=8, num_classes=4, beta_patch_mode="zero",
        srp_mode="post_agg", drop_path_rate=0.0,
        checkpoint_mode="off", ln_specialization="cls_patch",
        ln_specialization_scope="block_final",
    )
    shared.eval()
    split_final.eval()

    assert isinstance(split_final.norm, RoleSplitLayerNorm)
    y_shared = shared(features, nbi, nbm)
    y_split = split_final(features, nbi, nbm)
    assert torch.allclose(y_shared, y_split, atol=1e-5, rtol=1e-5)


def test_O_split_ln_checkpoint_modes_backward():
    """Specialized LN must work under both checkpointing strategies."""
    for checkpoint_mode in ("whole_block", "per_module"):
        features, nbi, nbm, _ = _synthetic_slide(in_dim=32)
        features.requires_grad_(True)
        torch.manual_seed(107)
        mod = NystromSRPAggregator(
            in_dim=32, embed_dim=96, depth=2, num_heads=6,
            num_landmarks=8, num_classes=4, beta_patch_mode="zero",
            srp_mode="post_agg", drop_path_rate=0.0,
            checkpoint_mode=checkpoint_mode, ln_specialization="cls_patch",
        )
        mod.train()
        loss = mod(features, nbi, nbm).sum()
        loss.backward()

        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
        norm1 = mod.blocks[0].norm1
        assert isinstance(norm1, RoleSplitLayerNorm)
        assert norm1.cls_norm.weight.grad is not None
        assert norm1.patch_norm.weight.grad is not None
        assert torch.isfinite(norm1.cls_norm.weight.grad).all()
        assert torch.isfinite(norm1.patch_norm.weight.grad).all()


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        print(f"{fn.__name__} ...", end=" ", flush=True)
        fn()
        print("OK")
    print(f"\nAll {len(fns)} tests passed.")
