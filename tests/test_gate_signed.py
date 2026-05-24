"""
Unit tests for slide_level_srp/src/gate_signed.py — TokenHeadGate.

What we verify:
  test_A   forward output shape is (B, H, N, 1) for any reasonable input.
  test_B   identity init: at construction, beta_eff == 0 exactly for any
           non-degenerate input. This is the critical invariant — without
           it, raw_init=0 wouldn't reproduce the un-projected baseline.
  test_C   range bound: for any random input, beta_eff lives in
           [-delta_scale, +delta_scale]. tanh saturation guarantees this.
  test_D   gradient flow: after one backward step on a random target,
           every learnable parameter on the OUTPUT path receives non-zero
           gradient — i.e. the gate is not silently dead.
  test_E   make_token_diag / make_head_diag helpers assemble the right
           shape and skip None columns correctly.
  test_F   reproducing fixed beta via raw_init: when the layer_head_bias
           is set to arctanh(beta / delta_scale), the gate produces a
           constant beta_eff equal to the requested fixed beta. This
           verifies the §2.3 ablation recipe for non-zero init.

These tests are pure-PyTorch and do not require GPU. Run with:

  conda activate atlas_patch
  python -m pytest tests/test_gate_signed.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Repo-root on sys.path so package imports work whether pytest is launched
# from the repo root or any subdirectory.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from slide_level_srp.src.gate_signed import TokenHeadGate, make_token_diag, make_head_diag


# Test-config defaults. Small enough to keep CPU runs fast.
B, H, N = 2, 4, 16
T_TOKEN, T_HEAD = 4, 3
HIDDEN_DIM = 16


def _random_inputs(
    delta_scale: float = 2.0,
    seed: int = 0,
) -> tuple[TokenHeadGate, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    gate = TokenHeadGate(
        num_heads=H,
        num_token_features=T_TOKEN,
        num_head_features=T_HEAD,
        hidden_dim=HIDDEN_DIM,
        delta_scale=delta_scale,
    )
    token_diag = torch.randn(B, N, T_TOKEN)
    head_diag = torch.randn(B, H, N, T_HEAD)
    return gate, token_diag, head_diag


def test_A_forward_shape() -> None:
    gate, token_diag, head_diag = _random_inputs()
    beta_eff = gate(token_diag, head_diag)
    assert beta_eff.shape == (B, H, N, 1)


def test_B_identity_init_beta_zero() -> None:
    """At construction, beta_eff must be exactly 0 for every (b, h, n).

    This verifies the proposal §2.3 invariant that raw_init = 0
    produces beta_eff = 0, so the SRP correction vanishes and the
    model starts as the un-projected baseline.
    """
    gate, token_diag, head_diag = _random_inputs()
    beta_eff = gate(token_diag, head_diag)
    # Strict zero — the output-path zero-init should produce a literal
    # zero raw_logit, and tanh(0) is exactly 0 in float32.
    assert torch.all(beta_eff == 0.0), (
        f"beta_eff must be exactly 0 at init; got max abs "
        f"{beta_eff.abs().max().item()}"
    )


@pytest.mark.parametrize("delta_scale", [1.0, 2.0, 0.5])
def test_C_range_bound(delta_scale: float) -> None:
    """beta_eff is bounded in [-delta_scale, +delta_scale] regardless of
    raw_logit magnitude — tanh saturates."""
    gate, token_diag, head_diag = _random_inputs(delta_scale=delta_scale)
    # Force the output-path parameters to large values so raw_logit is
    # well into the saturating regime of tanh. This tests the bound
    # holds even when the gate is far from init.
    with torch.no_grad():
        gate.token_mlp_out.weight.fill_(100.0)
        gate.token_mlp_out.bias.fill_(100.0)
        gate.head_weight.fill_(100.0)
        gate.head_bias.fill_(100.0)
        gate.layer_head_bias.fill_(100.0)
    beta_eff = gate(token_diag, head_diag)
    assert beta_eff.abs().max().item() <= delta_scale + 1e-6
    # And the saturation should be near the bound.
    assert beta_eff.abs().max().item() > 0.99 * delta_scale


def test_D_gradient_flow_on_output_params() -> None:
    """After one backward, every output-path parameter must have a
    non-zero gradient.

    Catches the silent-dead-gate failure mode: if raw_logit is 0 at
    init AND the loss happens to depend on beta_eff only through a
    saturating non-linearity, gradients could vanish.
    """
    gate, token_diag, head_diag = _random_inputs()
    beta_eff = gate(token_diag, head_diag)
    # Use beta_eff directly as the loss target so the gradient signal
    # is clean. We sum beta_eff so the loss has a non-zero gradient
    # everywhere — would be zero if we used .mean() of an exactly-zero
    # tensor, but .sum() of a zero tensor still yields zero loss with
    # non-zero gradient at each parameter (because d(sum)/d(param) is
    # not zero even when the values are).
    loss = beta_eff.sum()
    loss.backward()

    # Output-path parameters — required to have grad.
    output_path_params = [
        ("token_mlp_out.weight", gate.token_mlp_out.weight),
        ("token_mlp_out.bias", gate.token_mlp_out.bias),
        ("head_weight", gate.head_weight),
        ("head_bias", gate.head_bias),
        ("layer_head_bias", gate.layer_head_bias),
    ]
    for name, p in output_path_params:
        assert p.grad is not None, f"{name} has no grad"
        # At init, raw_logit = 0 → d(beta_eff)/d(raw_logit) = delta_scale * 1 = 2,
        # which is non-zero. So every output-path param receives gradient.
        assert p.grad.abs().sum() > 0.0, (
            f"{name} has zero grad — gate is silently dead"
        )

    # Hidden-layer weights at STEP 1 receive exactly zero gradient by
    # construction: dL/d(hidden.weight) = dL/d(u) · token_mlp_out.weight
    # · activation_grad, and token_mlp_out.weight is zero-init. This is
    # the expected behaviour, not a bug — the output layer moves on
    # step 1's update, after which the hidden layer wakes up. Verify
    # this by taking one SGD step on the output params and re-checking
    # at step 2.
    assert gate.token_mlp_hidden.weight.grad is not None
    assert gate.token_mlp_hidden.weight.grad.abs().sum() == 0.0, (
        "expected token_mlp_hidden.weight grad = 0 at step 1 (zero "
        "output init); got non-zero — verify zero-init recipe is "
        "actually applied"
    )
    # Step the output-path parameters off zero by hand. We use a TINY
    # step (1e-4) deliberately: a larger step would push raw_logit
    # well into the saturating regime of tanh, where d(tanh)/d(raw)
    # collapses to ~0 and the hidden layer would still get zero
    # gradient at step 2 — masking the property we want to verify.
    # 1e-4 keeps raw_logit in the linear-tanh region so the chain rule
    # delivers usable gradient back to the hidden layer.
    with torch.no_grad():
        gate.token_mlp_out.weight -= 1e-4 * gate.token_mlp_out.weight.grad
        gate.token_mlp_out.bias -= 1e-4 * gate.token_mlp_out.bias.grad
    # Step 2 forward + backward — the output layer is now non-zero, so
    # backprop reaches the hidden layer.
    gate.zero_grad()
    beta_eff_2 = gate(token_diag, head_diag)
    beta_eff_2.sum().backward()
    assert gate.token_mlp_hidden.weight.grad.abs().sum() > 0.0, (
        "token_mlp_hidden.weight has zero grad even after the output "
        "layer moved — gate is not propagating into the hidden layer"
    )


def test_E_helpers_assemble_correct_shape() -> None:
    """make_token_diag and make_head_diag should produce tensors with
    the right last dimension and skip None columns."""
    h_local = torch.randn(B, N)
    nbr_cnt = torch.randn(B, N)
    norm_r = torch.randn(B, N)
    local_var = None
    td = make_token_diag(h_local, nbr_cnt, norm_r, local_var)
    assert td.shape == (B, N, 3)

    cos_yr = torch.randn(B, H, N)
    log_norm_y = torch.randn(B, H, N)
    hd = make_head_diag(cos_yr, log_norm_y=log_norm_y)
    # cos_yr (1) + abs_cos_yr (auto-derived, 1) + log_norm_y (1) = 3
    assert hd.shape == (B, H, N, 3)

    # With abs_cos_yr explicitly None we still get auto-derived: same.
    hd2 = make_head_diag(cos_yr, abs_cos_yr=None, log_norm_y=None)
    # cos_yr + abs_cos_yr (derived) = 2
    assert hd2.shape == (B, H, N, 2)


def test_F_fixed_beta_via_raw_init() -> None:
    """Setting `layer_head_bias[h] = arctanh(beta_target / delta_scale)`
    on every head should produce a constant beta_eff equal to
    beta_target across all (b, h, n).

    This verifies the §2.3 ablation recipe for non-zero init.
    """
    delta_scale = 2.0
    beta_target = 1.0
    gate, token_diag, head_diag = _random_inputs(delta_scale=delta_scale)
    raw_init = math.atanh(beta_target / delta_scale)
    with torch.no_grad():
        gate.layer_head_bias.fill_(raw_init)
    beta_eff = gate(token_diag, head_diag)
    # Token MLP output and head linear are still zero-init, so they
    # contribute 0; the only non-zero raw_logit term is layer_head_bias.
    # Therefore beta_eff should equal beta_target everywhere.
    assert torch.allclose(
        beta_eff, torch.full_like(beta_eff, beta_target), atol=1e-6,
    ), (
        f"raw_init={raw_init:.6f} should produce constant "
        f"beta_eff={beta_target}; got range "
        f"[{beta_eff.min().item():.6f}, {beta_eff.max().item():.6f}]"
    )


def test_G_negative_beta_reachable() -> None:
    """Verify the signed nature: a negative layer_head_bias produces a
    negative beta_eff. This guards against any accidental sigmoid-like
    bounding that would prevent anti-SRP."""
    delta_scale = 2.0
    gate, token_diag, head_diag = _random_inputs(delta_scale=delta_scale)
    beta_target = -1.0
    raw_init = math.atanh(beta_target / delta_scale)
    with torch.no_grad():
        gate.layer_head_bias.fill_(raw_init)
    beta_eff = gate(token_diag, head_diag)
    assert torch.allclose(
        beta_eff, torch.full_like(beta_eff, beta_target), atol=1e-6,
    )
    assert (beta_eff < 0).all()


def test_H_gate_stats_accumulator_correlation_is_pearson() -> None:
    """corr_beta_h_local saved into test_artifacts.npz must be the
    standard Pearson correlation (centered numerator AND centered
    denominator). The first implementation used uncentered norms in
    the denominator which biased the magnitude toward zero. This test
    constructs a controlled (β_eff, h_local) pair with a known Pearson
    r and verifies the persisted value matches.
    """
    from slide_level_srp.src.gate_signed import GateStatsAccumulator
    import numpy as np

    # Build a tiny model-like object with one block whose attn exposes
    # the same `_last_gate_stats` schema GateStatsAccumulator reads.
    class _FakeAttn:
        def __init__(self, beta_eff, cos_yr, y_norms, h_local, ncnt):
            self._last_gate_stats = {
                "beta_eff": beta_eff,
                "cos_yr": cos_yr,
                "y_norms": y_norms,
                "h_local": h_local,
                "neighbour_count": ncnt,
            }

    class _FakeBlock:
        def __init__(self, attn): self.attn = attn

    class _FakeModel:
        # Deliberately NOT an nn.Module — GateStatsAccumulator only
        # accesses `model.blocks[i].attn._last_gate_stats` via attribute
        # lookup, so a plain object suffices and we avoid having to
        # wrap fake blocks in nn.ModuleList.
        def __init__(self, blocks): self.blocks = blocks

    # B=1, H=2, N=8. Construct β_eff so its per-token mean (averaged
    # across heads) equals exactly h_local — Pearson r should be 1.0.
    H, N = 2, 8
    h_local = torch.linspace(-1.0, 1.0, N).unsqueeze(0)        # (1, N)
    # β_eff per-token (averaged across H, then squeezed last dim) = h_local.
    # So replicate h_local across heads, then add a head-specific perturbation
    # that averages to zero across heads to keep the per-token mean unchanged.
    base = h_local.expand(H, N).unsqueeze(0).unsqueeze(-1)     # (1, H, N, 1)
    head_pert = torch.tensor([[1.0], [-1.0]]).view(1, H, 1, 1) * 0.1
    beta_eff = base + head_pert
    # Sanity: per-token mean across heads recovers h_local exactly.
    pt_mean = beta_eff.mean(dim=1).squeeze(-1)                 # (1, N)
    assert torch.allclose(pt_mean, h_local, atol=1e-6)

    cos_yr = torch.zeros(1, H, N)
    y_norms = torch.ones(1, H, N)
    ncnt = torch.full((1, N), 8.0)
    fake_blk = _FakeBlock(_FakeAttn(beta_eff, cos_yr, y_norms, h_local, ncnt))
    fake_model = _FakeModel([fake_blk])

    acc = GateStatsAccumulator()
    acc.update(fake_model)
    out = acc.finalize()
    corr = float(out["block0_corr_beta_h_local"][0])
    assert abs(corr - 1.0) < 1e-5, (
        f"GateStatsAccumulator should report Pearson r = 1.0 when β_eff "
        f"per-token-mean equals h_local; got {corr:.6f}. The denominator "
        f"may still be using uncentered norms (biases the magnitude)."
    )

    # Construct a second pair where the analytic Pearson is exactly 0.5
    # so we verify the absolute magnitude is right (not just the sign).
    rng = np.random.default_rng(0)
    x = rng.standard_normal(40)
    eps = rng.standard_normal(40)
    y = 0.5 * x + np.sqrt(1 - 0.5**2) * eps   # ~ correlated at r=0.5
    # Convert to torch shapes expected by the accumulator.
    h_local2 = torch.from_numpy(x).float().unsqueeze(0)         # (1, 40)
    pt_mean2 = torch.from_numpy(y).float().unsqueeze(0)         # (1, 40)
    beta_eff2 = pt_mean2.unsqueeze(1).unsqueeze(-1).expand(1, H, 40, 1).clone()
    cos_yr2 = torch.zeros(1, H, 40)
    y_norms2 = torch.ones(1, H, 40)
    ncnt2 = torch.full((1, 40), 8.0)
    fake_blk2 = _FakeBlock(_FakeAttn(beta_eff2, cos_yr2, y_norms2, h_local2, ncnt2))
    fake_model2 = _FakeModel([fake_blk2])
    acc2 = GateStatsAccumulator()
    acc2.update(fake_model2)
    out2 = acc2.finalize()
    corr2 = float(out2["block0_corr_beta_h_local"][0])
    # Compute the analytic Pearson r of (x, y) for comparison.
    expected = float(np.corrcoef(x, y)[0, 1])
    # Generous tolerance because this is just verifying we are NOT
    # off by a known-broken factor (uncentered norms would skew this
    # by ~10-30% depending on the means).
    assert abs(corr2 - expected) < 1e-4, (
        f"GateStatsAccumulator's persisted corr deviates from "
        f"np.corrcoef. Persisted={corr2:.6f}, expected={expected:.6f}. "
        f"Denominator is likely uncentered."
    )


def test_I_gate_stats_accumulator_handles_empty_example() -> None:
    """GateStatsAccumulator.update() must not crash on zero-token examples.

    The regression observed on main was a signed-gate PANDA final-eval
    pass where one example had beta_eff shape (B, H, 0, 1). The previous
    reducer called min/max and divided by zero after training had already
    completed, so the run lost its final artifacts. Empty examples are
    represented with NaNs to preserve per-example array alignment.
    """
    from slide_level_srp.src.gate_signed import GateStatsAccumulator

    class _FakeAttn:
        def __init__(self, beta_eff, h_local=None):
            self._last_gate_stats = {"beta_eff": beta_eff}
            if h_local is not None:
                self._last_gate_stats["h_local"] = h_local

    class _FakeBlock:
        def __init__(self, attn): self.attn = attn

    class _FakeModel:
        def __init__(self, blocks): self.blocks = blocks

    beta_eff_empty = torch.zeros(1, H, 0, 1)
    fake_model = _FakeModel([_FakeBlock(_FakeAttn(beta_eff_empty))])
    acc = GateStatsAccumulator()
    acc.update(fake_model)
    out = acc.finalize()
    assert "block0_beta_eff_mean" in out
    assert math.isnan(float(out["block0_beta_eff_mean"][0]))
    for stat in (
        "beta_eff_min", "beta_eff_max", "beta_eff_std", "frac_neg",
        "frac_near_zero", "frac_projection", "frac_reflection",
    ):
        assert math.isnan(float(out[f"block0_{stat}"][0])), (
            f"{stat} should be NaN for an empty example"
        )

    rng = torch.Generator().manual_seed(0)
    acc2 = GateStatsAccumulator()
    for be in (
        torch.randn(1, H, 5, 1, generator=rng),
        torch.zeros(1, H, 0, 1),
        torch.randn(1, H, 5, 1, generator=rng),
    ):
        acc2.update(_FakeModel([_FakeBlock(_FakeAttn(be))]))
    arr = acc2.finalize()["block0_beta_eff_mean"]
    assert arr.shape == (3,)
    assert not math.isnan(float(arr[0]))
    assert math.isnan(float(arr[1]))
    assert not math.isnan(float(arr[2]))

    acc3 = GateStatsAccumulator()
    acc3.update(_FakeModel([
        _FakeBlock(_FakeAttn(torch.zeros(1, H, 0, 1), h_local=torch.zeros(1, 0)))
    ]))
    out3 = acc3.finalize()
    assert "block0_corr_beta_h_local" in out3
    assert math.isnan(float(out3["block0_corr_beta_h_local"][0]))


def test_J_panda_gate_diagnostics_handles_empty_example() -> None:
    """PANDA last-batch diagnostics should mirror the empty-stat policy."""
    from slide_level_srp.train_panda import gate_diagnostics

    class _FakeAttn:
        def __init__(self):
            self._last_gate_stats = {
                "beta_eff": torch.zeros(1, H, 0, 1),
                "cos_yr": torch.zeros(1, H, 0),
                "h_local": torch.zeros(1, 0),
            }

    class _FakeBlock:
        def __init__(self): self.attn = _FakeAttn()

    class _FakeModel:
        def __init__(self): self.blocks = [_FakeBlock()]

    out = gate_diagnostics(_FakeModel(), prefix="gate_last_batch")
    assert math.isnan(float(out["gate_last_batch/block0/beta_eff_mean"]))
    assert math.isnan(float(out["gate_last_batch/block0/frac_neg"]))
    assert math.isnan(float(out["gate_last_batch/block0/corr_beta_h_local"]))


if __name__ == "__main__":
    # Allow `python tests/test_gate_signed.py` to run all tests directly.
    test_A_forward_shape()
    test_B_identity_init_beta_zero()
    for ds in [1.0, 2.0, 0.5]:
        test_C_range_bound(ds)
    test_D_gradient_flow_on_output_params()
    test_E_helpers_assemble_correct_shape()
    test_F_fixed_beta_via_raw_init()
    test_G_negative_beta_reachable()
    test_H_gate_stats_accumulator_correlation_is_pearson()
    test_I_gate_stats_accumulator_handles_empty_example()
    test_J_panda_gate_diagnostics_handles_empty_example()
    print("All tests passed.")
