"""
Unit tests for NystromSRPAttention.

Covers (by design):
  A   beta=zero is identity at the patch rows (z == y everywhere)
  B   beta=1 orthogonalizes z vs r at patch rows
  C   CLS row unchanged under every srp_mode (direct projection no-op)
  D   Pad duplicates unchanged under every srp_mode
  E   Pre-V instance weeding: v_j == r_j -> v'_j ~ 0
  F   Detach policy: no gradient flows from a patch's projected output
      back through a neighbor's v_k via r_j
  G   End-to-end backward produces finite grads, including beta_patch
  H   Baseline equivalence: beta=zero + baseline inputs -> output matches
      NystromXSAAttention with alpha=0 (XSA disabled)
  I   Gated path: post_agg_gated with gate==1 equals post_agg
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from slide_level_srp.src.srp_attention import NystromSRPAttention
from slide_level.src.nystrom_xsa import NystromXSAAttention


# --- helpers --------------------------------------------------------------

def _build_attn(
    dim=192, heads=6, m=16,
    beta_mode="one", beta_init=1.0,
    srp_mode="post_agg",
    num_cls=1, pinv=6, qkv_bias=True,
    **kwargs,
):
    torch.manual_seed(0)
    return NystromSRPAttention(
        dim=dim, num_heads=heads, num_landmarks=m,
        qkv_bias=qkv_bias, attn_drop=0.0, proj_drop=0.0,
        beta_patch_mode=beta_mode, beta_init=beta_init,
        srp_mode=srp_mode, num_cls_tokens=num_cls,
        pinv_iterations=pinv,
        **kwargs,
    )


def _trivial_neighbors(B: int, N: int, device=None):
    """
    Build a trivial neighbor list: each patch claims its 8 cyclic
    neighbors [i-1, i+1, i-2, i+2, i-3, i+3, i-4, i+4] (wrapped mod N),
    or as many as exist for small N. Always has at least |N|=1 neighbor
    for every patch (if N >= 2). The mask matches.

    This is enough for mechanism tests; spatial correctness of the
    neighbor builder is covered in test_neighbor_index.py.
    """
    device = device or torch.device("cpu")
    offsets = [-1, 1, -2, 2, -3, 3, -4, 4]
    nbi = torch.full((B, N, 8), -1, dtype=torch.long, device=device)
    nbm = torch.zeros((B, N, 8), dtype=torch.bool, device=device)
    for i in range(N):
        for k, o in enumerate(offsets):
            j = (i + o) % N if N > 1 else -1
            if j != i and j >= 0 and j < N:
                nbi[:, i, k] = j
                nbm[:, i, k] = True
    return nbi, nbm


def _is_real_full(B: int, N: int, L: int, device=None):
    """is_real: True at positions 1..N, False at 0 and N+1..L-1."""
    is_real = torch.zeros(B, L, dtype=torch.bool, device=device or "cpu")
    is_real[:, 1 : 1 + N] = True
    return is_real


# --- tests ----------------------------------------------------------------

def test_A_beta_zero_is_patch_identity():
    """beta_patch=0 -> projection term is 0 -> z == y at patch rows.

    We verify this through cos_zr_post == cos_yr_pre in the capture dict,
    since z == y implies their projections on r̂ are equal.
    """
    torch.manual_seed(1)
    B, N, C = 1, 24, 192
    L = 1 + N  # no PPEG pad in this test
    x = torch.randn(B, L, C)
    nbi, nbm = _trivial_neighbors(B, N)
    is_real = _is_real_full(B, N, L)

    mod = _build_attn(beta_mode="zero", srp_mode="post_agg")
    mod._capture_stats = True
    _ = mod(x, nbi, nbm, is_real)
    stats = mod.last_stats
    mod._capture_stats = False

    # z_patch == y_patch => cos(z, r) == cos(y, r).
    cos_pre = stats["cos_yr_patch_pre"]
    cos_post = stats["cos_zr_patch_post"]
    assert torch.allclose(cos_pre, cos_post, atol=1e-5), (
        f"beta=0 should give identity at patch rows; got max |Δ| = "
        f"{(cos_pre - cos_post).abs().max().item():.3e}"
    )


def test_B_beta_one_orthogonalizes_z_vs_r():
    """beta_patch=1 -> z is orthogonal to r at every patch row."""
    torch.manual_seed(2)
    B, N, C = 1, 32, 192
    L = 1 + N
    x = torch.randn(B, L, C)
    nbi, nbm = _trivial_neighbors(B, N)
    is_real = _is_real_full(B, N, L)

    mod = _build_attn(beta_mode="one", srp_mode="post_agg")
    mod._capture_stats = True
    _ = mod(x, nbi, nbm, is_real)
    cos_zr_post = mod.last_stats["cos_zr_patch_post"]
    mod._capture_stats = False

    # Patches with |N|=0 would pass through (r̂ ≈ 0 -> cos undefined ->
    # F.cosine_similarity returns 0). Our trivial neighbor fixture gives
    # every patch >= 1 neighbor for N >= 2, so we expect ~0 everywhere.
    assert cos_zr_post.abs().max().item() < 1e-4, (
        f"beta=1 should make cos(z, r) ~ 0; got max |cos| = "
        f"{cos_zr_post.abs().max().item():.3e}"
    )


def test_C_cls_row_never_projected():
    """CLS row (position 0) receives no DIRECT SRP projection under any
    mode. The correct invariant by design is the internal
    per-forward equality:

        z_cls == y_cls   (inside a single forward pass)

    NOT cross-ablation equality: under pre_v, CLS's output legitimately
    differs from baseline because CLS attends over modified patch values
    (by design: CLS inherits SRP's effect indirectly through modified
    patch outputs). Checking z_cls(β=1) == z_cls(β=0)
    therefore fails by design on pre_v, not because of a bug.

    We verify the correct invariant by reading the raw y_cls and z_cls
    tensors that NystromSRPAttention stashes in `last_stats` specifically
    for this test.

    Full element-wise vector comparison (not just L2 norms) addresses
    audit-round-1 "Low: CLS/pad tests compare norms rather than vectors".
    Non-state_dict-based reference-module setup addresses audit-round-2
    "Medium: load_state_dict overwrites beta_patch=0".
    """
    torch.manual_seed(3)
    B, N, C = 1, 24, 192
    L = 1 + N
    x = torch.randn(B, L, C)
    nbi, nbm = _trivial_neighbors(B, N)
    is_real = _is_real_full(B, N, L)

    for srp_mode in ("post_agg", "pre_v", "post_agg_gated"):
        mod = _build_attn(beta_mode="one", srp_mode=srp_mode)
        # Sanity: β is actually 1, not accidentally 0 through some init path.
        assert mod.beta_patch.abs().max().item() > 0.5, (
            f"mod (β=1) should have non-zero beta_patch; got {mod.beta_patch}"
        )

        h_m = torch.full((B, N), 0.5, dtype=torch.float32)
        mod._capture_stats = True
        _ = mod(
            x, nbi, nbm, is_real,
            h_morph=h_m if srp_mode == "post_agg_gated" else None,
        )
        stats = mod.last_stats
        mod._capture_stats = False

        y_cls = stats["y_cls_raw"]    # (B, H, n_cls, D)
        z_cls = stats["z_cls_raw"]
        diff = (y_cls - z_cls).abs()
        # Exact equality at CLS — not approximate. SRP never writes into
        # position 0 under ANY mode, so the value z stored at CLS must
        # equal whatever y_cls the Nyström forward produced. If ANY bit
        # differs, something is silently modifying CLS.
        assert diff.max().item() == 0.0, (
            f"[{srp_mode}] z_cls ≠ y_cls within a single forward: "
            f"max |Δ| = {diff.max().item():.3e}. CLS is being touched "
            f"by SRP — clone-and-update or projection path has a bug."
        )

    # Cross-check that the comparison is non-vacuous: β=1 post_agg MUST
    # alter the patch rows vs β=0 (otherwise the projection is silently
    # a no-op and the whole test suite would pass trivially).
    mod1 = _build_attn(beta_mode="one", srp_mode="post_agg")
    mod0 = _build_attn(beta_mode="zero", srp_mode="post_agg")
    assert torch.allclose(mod1.qkv.weight, mod0.qkv.weight), (
        "Seeded construction should yield identical qkv weights."
    )
    assert mod1.beta_patch.abs().max().item() > 0.5
    assert mod0.beta_patch.abs().max().item() < 1e-8
    mod1._capture_stats = True; mod0._capture_stats = True
    _ = mod1(x, nbi, nbm, is_real)
    _ = mod0(x, nbi, nbm, is_real)
    # Read z_norm at patch rows for both, via last_stats.
    z_norm_1 = mod1.last_stats["z_norm"][:, :, 1 : 1 + N]
    z_norm_0 = mod0.last_stats["z_norm"][:, :, 1 : 1 + N]
    patch_drift = (z_norm_1 - z_norm_0).abs().max().item()
    assert patch_drift > 1e-3, (
        f"β=1 projection is inactive: patch-row z_norm matches β=0 "
        f"(max |Δ| = {patch_drift:.3e}). test_C's CLS invariant would "
        f"be vacuous if this happened."
    )


def test_C2_signed_pre_attention_modes_start_as_identity_and_do_not_write_cls():
    """Pre-Q/K/V signed-gated SRP must preserve the identity-init contract.

    The new placements edit Q/K/V before attention, but beta_eff is exactly
    zero at construction.  Therefore each mode should match the beta=0
    baseline at the full module output while still preserving the internal
    invariant that SRP never directly writes the CLS row.
    """
    torch.manual_seed(33)
    B, N, C = 1, 24, 192
    L = 1 + N
    x = torch.randn(B, L, C)
    h_local = torch.linspace(-0.5, 0.5, N).view(B, N)
    nbi, nbm = _trivial_neighbors(B, N)
    is_real = _is_real_full(B, N, L)

    ref = _build_attn(beta_mode="zero", srp_mode="post_agg")
    ref.eval()
    with torch.no_grad():
        y_ref = ref(x, nbi, nbm, is_real)

    for srp_mode in (
        "post_agg_signed_gated",
        "pre_q_signed_gated",
        "pre_k_signed_gated",
        "pre_v_signed_gated",
    ):
        mod = _build_attn(
            beta_mode="signed_gated",
            srp_mode=srp_mode,
            delta_scale=2.0,
        )
        mod.eval()
        mod._capture_stats = True
        with torch.no_grad():
            y = mod(x, nbi, nbm, is_real, h_local=h_local)
        stats = mod.last_stats
        mod._capture_stats = False

        assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-6), (
            f"{srp_mode} should be output-identical to beta=0 at "
            "identity init."
        )
        gate_stats = mod._last_gate_stats
        assert gate_stats is not None, f"{srp_mode} did not populate gate stats"
        beta_eff = gate_stats["beta_eff"]
        assert torch.all(beta_eff == 0.0), (
            f"{srp_mode} beta_eff must be exactly zero at identity init"
        )
        cls_diff = (stats["y_cls_raw"] - stats["z_cls_raw"]).abs().max().item()
        assert cls_diff == 0.0, (
            f"{srp_mode} directly wrote CLS inside SRP; max |diff|={cls_diff:.3e}"
        )


def test_C3_signed_pre_attention_modes_handle_pad_duplicates():
    """Pre-attention modes must tolerate TransMIL square-pad duplicates.

    The SRP write surface is restricted to real patch rows.  Pad duplicates
    can still participate in attention as TransMIL tokens, but they should
    not create shape errors or NaNs when the real-patch neighborhood tensor
    is shorter than the padded sequence.
    """
    torch.manual_seed(34)
    B, N, C = 1, 20, 192
    L = 1 + 25
    x = torch.randn(B, L, C)
    h_local = torch.linspace(-0.25, 0.25, N).view(B, N)
    nbi, nbm = _trivial_neighbors(B, N)
    is_real = _is_real_full(B, N, L)

    for srp_mode in (
        "pre_q_signed_gated",
        "pre_k_signed_gated",
        "pre_v_signed_gated",
    ):
        mod = _build_attn(
            beta_mode="signed_gated",
            srp_mode=srp_mode,
            delta_scale=2.0,
        )
        mod.eval()
        with torch.no_grad():
            out = mod(x, nbi, nbm, is_real, h_local=h_local)
        assert out.shape == (B, L, C)
        assert torch.isfinite(out).all(), f"{srp_mode} produced non-finite output"


def test_D_pad_duplicates_not_projected():
    """Pad-dupe rows (positions 1+N..L-1, where is_real=False) pass
    through unchanged under post_agg SRP.

    Under post_agg the v tensor is not modified, so Nyström's output
    at pad positions is bit-identical between β=0 and β=1 (only the
    clone-and-update of y at the PATCH slice changes with β). We verify
    full element-wise equality at pad positions — not just L2 norms —
    addressing audit-round-1 "Low: CLS/pad tests compare norms rather
    than vectors".

    Seed-synchronized construction (not load_state_dict) ensures the
    β=0 reference actually has β=0 — addresses audit-round-2 Medium 3.
    """
    torch.manual_seed(4)
    B, N, C = 1, 20, 192
    # Simulate post-square-pad: H*W = 25, add = 5 pad dupes.
    L = 1 + 25
    x = torch.randn(B, L, C)
    nbi, nbm = _trivial_neighbors(B, N)
    is_real = torch.zeros(B, L, dtype=torch.bool)
    is_real[:, 1 : 1 + N] = True

    # Seed-synchronized construction. Each _build_attn call resets the
    # torch RNG to state 0 before constructing; Linear init consumes
    # RNG identically across both calls, while the beta_patch buffer
    # fill is RNG-free. So qkv/proj match but beta_patch differs as
    # intended.
    mod = _build_attn(beta_mode="one", srp_mode="post_agg")
    mod_ref = _build_attn(beta_mode="zero", srp_mode="post_agg")
    assert torch.allclose(mod.qkv.weight, mod_ref.qkv.weight)
    assert mod.beta_patch.abs().max().item() > 0.5
    assert mod_ref.beta_patch.abs().max().item() < 1e-8

    captured, captured_ref = {}, {}
    def hook(module, inp, _out):
        captured["z"] = inp[0].detach().clone()
    def hook_ref(module, inp, _out):
        captured_ref["z"] = inp[0].detach().clone()
    h1 = mod.proj.register_forward_hook(hook)
    h2 = mod_ref.proj.register_forward_hook(hook_ref)
    _ = mod(x, nbi, nbm, is_real)
    _ = mod_ref(x, nbi, nbm, is_real)
    h1.remove(); h2.remove()

    # Sanity cross-check: β=1 MUST have modified patch rows vs β=0,
    # otherwise we'd be comparing two identical forwards and the test
    # would be a no-op.
    patch_diff = (
        captured["z"][:, 1 : 1 + N, :] - captured_ref["z"][:, 1 : 1 + N, :]
    ).abs().max().item()
    assert patch_diff > 1e-3, (
        f"β=1 and β=0 gave identical patch rows (max |Δ| = {patch_diff:.3e}); "
        f"projection not active — test is vacuous."
    )

    # Element-wise check on every pad-dupe position — the actual assertion.
    pad_diff = (
        captured["z"][:, 1 + N : L, :] - captured_ref["z"][:, 1 + N : L, :]
    ).abs()
    assert pad_diff.max().item() < 1e-5, (
        f"Pad-dupe rows differ between β=1 and β=0 forwards: "
        f"max |Δ| element = {pad_diff.max().item():.3e}. "
        f"SRP must not touch pad duplicates."
    )


def test_E_prev_instance_weeding():
    """Under pre_v with β̃=1 and v_j == r_j (perfectly homogeneous
    neighborhood), v'_j must be ~ 0 by design

    We construct this by making all the input features identical on the
    patch rows — then Q/K/V projections produce identical v_j across
    patches (linear map of identical inputs), so r_j = mean(v_k) = v_j.
    """
    torch.manual_seed(5)
    B, N, C = 1, 24, 192
    L = 1 + N
    # Identical features on patch rows; CLS can be different.
    x = torch.randn(B, L, C) * 0  # zeros
    x[:, 0, :] = torch.randn(C)   # CLS varies (shouldn't matter here)
    x[:, 1 : 1 + N, :] = torch.randn(C)   # SAME random vector on all patches
    nbi, nbm = _trivial_neighbors(B, N)
    is_real = _is_real_full(B, N, L)

    mod = _build_attn(beta_mode="one", srp_mode="pre_v")
    mod._capture_stats = True
    _ = mod(x, nbi, nbm, is_real)
    stats = mod.last_stats
    mod._capture_stats = False

    # ρ_j = ||v'|| / ||v||. For perfect homogeneity, ρ should be ~0.
    rho = stats["rho_patch"]          # (B, H, N)
    assert rho is not None, "pre_v should populate rho_patch"
    assert rho.max().item() < 1e-3, (
        f"Under perfect homogeneity (v_j == r_j), ρ_j should be ~0 "
        f"everywhere; got max ρ = {rho.max().item():.3e}"
    )


def test_F_detach_blocks_gradient_through_neighbors():
    """r_j is computed from v.detach(), so gradient does NOT flow from
    the projection term `(v_j · r̂_j) · r̂_j` back through the neighbors
    v_k that r_j was built from.

    Strengthened per audit: we use torch.autograd.grad to construct
    an explicit test. Let L = sum of the pre_v-modified v at patch j
    ONLY, i.e. L = v'_j.sum(). By design:
        v'_j = v_j - β̃ · (v_j · r̂_j) · r̂_j
    where r̂_j is detached. The gradient dL/dv_k for a neighbor k ≠ j
    should be EXACTLY 0 — there is no path from v'_j back to v_k.

    We verify this by taking the gradient of v'_j through the raw v
    tensor (before clone-and-update). If the detach works, dL/dv[b, h, k, :]
    for k ≠ j is exactly zero.
    """
    import torch.nn.functional as Fnn
    torch.manual_seed(6)
    B, N, C = 1, 16, 192
    L = 1 + N
    x = torch.randn(B, L, C)
    nbi, nbm = _trivial_neighbors(B, N)
    is_real = _is_real_full(B, N, L)

    mod = _build_attn(beta_mode="one", srp_mode="pre_v")
    # We construct the pre_v projection directly using the same math
    # as NystromSRPAttention.forward's pre_v branch, so we can target
    # a specific j and assert exact zero gradient for a specific k ≠ j.
    H, D = mod.num_heads, mod.head_dim
    qkv = mod.qkv(x).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
    _q, _k, v = qkv.unbind(0)
    # v is a view of the qkv output; requires_grad is inherited from
    # qkv.weight via the Linear. Retain grad so we can read it.
    v.retain_grad()
    v_patch = v[:, :, 1 : 1 + N, :]                       # (B, H, N, D)

    # Replicate the attention's gather step (detached neighbors).
    from slide_level_srp.src.srp_attention import (
        gather_neighbors, neighborhood_mean,
    )
    neighbor_v_det = gather_neighbors(v_patch.detach(), nbi, nbm)
    _r, r_hat_det, _c = neighborhood_mean(neighbor_v_det, nbm)   # detached

    beta_bh = mod.beta_patch.view(1, H, 1, 1).to(dtype=v.dtype)
    dot_vr = (v_patch * r_hat_det).sum(dim=-1, keepdim=True)
    v_patch_new = v_patch - beta_bh * dot_vr * r_hat_det

    # Pick j = 3 (interior patch under the trivial neighbor fixture).
    # j's neighbors are at positions (3 ± 1, 3 ± 2, ..., 3 ± 4) mod N.
    j = 3
    # Identify one neighbor k and one non-neighbor k'.
    neigh_of_j = set()
    for kk in range(8):
        if nbm[0, j, kk]:
            neigh_of_j.add(int(nbi[0, j, kk]))
    non_neighbor_ks = [i for i in range(N) if i != j and i not in neigh_of_j]
    assert non_neighbor_ks, "fixture must contain at least one non-neighbor of j"
    k_nn = non_neighbor_ks[0]

    # Loss: sum v'_j across heads + dims -> scalar.
    L_scalar = v_patch_new[:, :, j, :].sum()
    # Gradient of L_scalar w.r.t. v (the full qkv v tensor).
    grads = torch.autograd.grad(
        L_scalar, v, retain_graph=False, create_graph=False,
    )
    dL_dv = grads[0]                                       # (B, H, L, D)

    # dL/dv[:, :, j, :] should be non-zero (direct path v_j → v'_j).
    grad_j = dL_dv[:, :, 1 + j, :].abs().max().item()
    assert grad_j > 1e-4, (
        f"Sanity failed: direct path v_j → v'_j should give non-zero grad, "
        f"got max |dL/dv_j| = {grad_j:.3e}"
    )

    # dL/dv[:, :, k_nn, :] must be exactly zero — detach closes all paths
    # through the neighborhood mean. (Includes the CLS and pad rows, but
    # we specifically check a non-neighbor patch row.)
    grad_knn = dL_dv[:, :, 1 + k_nn, :].abs().max().item()
    assert grad_knn < 1e-8, (
        f"Detach policy violation: dL/dv_k (k non-neighbor of j) = "
        f"{grad_knn:.3e}, should be exactly 0. Gradient is leaking "
        f"through r_j — the gather/neighborhood_mean must consume "
        f"v.detach()."
    )

    # dL/dv[:, :, k, :] for k ∈ N(j) should ALSO be zero under the detach
    # policy: r_j uses detached v_k, so v'_j has no gradient dependence
    # on v_k either. This is the stronger condition the validation check asked
    # for — we're not just checking "gradients finite", we're checking
    # "the exact path we claim to have cut is actually cut."
    for k in sorted(neigh_of_j):
        grad_k = dL_dv[:, :, 1 + k, :].abs().max().item()
        assert grad_k < 1e-8, (
            f"Detach policy violation: dL/dv_k (k={k}, neighbor of j={j}) = "
            f"{grad_k:.3e}, should be exactly 0 under detached r_j."
        )


def test_G_end_to_end_backward_is_finite():
    """Every SRP mode supports backward pass with finite grads on all
    trainable parameters (including learnable beta_patch)."""
    torch.manual_seed(7)
    B, N, C = 1, 24, 192
    L = 1 + N
    x = torch.randn(B, L, C, requires_grad=True)
    nbi, nbm = _trivial_neighbors(B, N)
    is_real = _is_real_full(B, N, L)
    h_m = torch.rand(B, N, dtype=torch.float32)

    for srp_mode in ("post_agg", "pre_v", "post_agg_gated"):
        mod = _build_attn(beta_mode="learn", srp_mode=srp_mode)
        out = mod(
            x, nbi, nbm, is_real,
            h_morph=h_m if srp_mode == "post_agg_gated" else None,
        )
        loss = (out ** 2).mean()
        # Retain grad because x is shared across modes in this loop.
        mod.zero_grad(set_to_none=True)
        x.grad = None
        loss.backward()
        for n, p in mod.named_parameters():
            assert p.grad is not None, f"[{srp_mode}] {n} has no grad"
            assert torch.isfinite(p.grad).all(), (
                f"[{srp_mode}] {n} has non-finite grad"
            )


def test_H_gated_with_gate_one_equals_postagg():
    """Under post_agg_gated with h_morph clamping to 1, the effective
    β equals β_base, so the output should match post_agg exactly.

    Weights must be shared (same init) between the two modules; the
    easiest way is to build one module and load its state into the other.
    """
    torch.manual_seed(8)
    B, N, C = 1, 24, 192
    L = 1 + N
    x = torch.randn(B, L, C)
    nbi, nbm = _trivial_neighbors(B, N)
    is_real = _is_real_full(B, N, L)

    torch.manual_seed(99)
    m_postagg = _build_attn(beta_mode="one", srp_mode="post_agg")
    torch.manual_seed(99)
    m_gated = _build_attn(beta_mode="one", srp_mode="post_agg_gated")
    # Sanity: qkv and proj weights match since we seeded identically.
    assert torch.allclose(m_postagg.qkv.weight, m_gated.qkv.weight)
    assert torch.allclose(m_postagg.proj.weight, m_gated.proj.weight)
    # beta_patch buffers are identical.
    assert torch.allclose(m_postagg.beta_patch, m_gated.beta_patch)

    h_m = torch.ones(B, N, dtype=torch.float32)   # gate -> 1.0 everywhere

    y_postagg = m_postagg(x, nbi, nbm, is_real)
    y_gated   = m_gated(x, nbi, nbm, is_real, h_morph=h_m)
    max_diff = (y_postagg - y_gated).abs().max().item()
    assert max_diff < 1e-5, (
        f"gated@gate=1 should equal post_agg exactly; max |Δ| = {max_diff:.3e}"
    )


def test_I_baseline_beta_zero_matches_xsa_alpha_zero():
    """Sanity: baseline SRP (beta=0) should produce the same output as
    the stage-2 attention with alpha=0, provided QKV/proj weights match.

    We seed both modules identically and verify their outputs on a common
    input agree to bf32 precision. This secures the "no drift between
    stage-2 and slide-level SRP baseline" guarantee (by design).
    """
    torch.manual_seed(9)
    B, N, C = 1, 40, 192
    L = 1 + N
    x = torch.randn(B, L, C)
    nbi, nbm = _trivial_neighbors(B, N)
    is_real = _is_real_full(B, N, L)

    # Use explicit seeded construction (bypassing the helper's internal
    # seed(0)) so both modules share QKV/proj weights.
    torch.manual_seed(42)
    m_srp = NystromSRPAttention(
        dim=192, num_heads=6, num_landmarks=16,
        qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
        beta_patch_mode="zero", beta_init=1.0,
        srp_mode="post_agg", num_cls_tokens=1, pinv_iterations=6,
    )
    torch.manual_seed(42)
    m_xsa = NystromXSAAttention(
        dim=192, num_heads=6, num_landmarks=16,
        qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
        alpha_cls_mode="zero", alpha_patch_mode="zero",
        alpha_init=1.0, num_cls_tokens=1, pinv_iterations=6,
    )

    # Sanity: qkv and proj weights are byte-identical.
    assert torch.allclose(m_srp.qkv.weight, m_xsa.qkv.weight)
    assert torch.allclose(m_srp.proj.weight, m_xsa.proj.weight)

    y_srp = m_srp(x, nbi, nbm, is_real)
    y_xsa = m_xsa(x)
    max_diff = (y_srp - y_xsa).abs().max().item()
    assert max_diff < 1e-5, (
        f"SRP baseline diverges from XSA alpha=0: max |Δ| = {max_diff:.3e}"
    )


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        print(f"{fn.__name__} ...", end=" ", flush=True)
        fn()
        print("OK")
    print(f"\nAll {len(fns)} tests passed.")
