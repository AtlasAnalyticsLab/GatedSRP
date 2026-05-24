"""
NystromSRPAttention: Nyström self-attention + Spatial Redundancy Projection (SRP).

SRP replaces XSA's self-direction projection with a projection along the
local *spatial common-mode* direction r_i (the mean value vector over a
patch's 3x3 neighborhood). Three projection modes are supported, all
toggled by `srp_mode`:

    "post_agg"        z_i = y_i - beta * (y_i · r̂_i) r̂_i
                      (proposal §2.4; applied after Nyström aggregation)

    "pre_v"           v'_j = v_j - β̃ * (v_j · r̂_j) r̂_j, aggregate with v'
                      (proposal §2.5; projects values before mixing)

    "post_agg_gated"  β_i = β_base · g(h^morph_i), then apply post_agg
                      with the per-token β_i. h^morph is precomputed
                      from frozen raw UNI features, passed in as `h_morph`
                      (proposal §6).

    "pre_q_signed_gated" / "pre_k_signed_gated" / "pre_v_signed_gated"
                      learned signed-gated SRP applied to Q, K, or V before
                      Nyström attention. These are the CAM16/CAM17
                      pre-attention placement ablations inspired by the
                      gated-attention paper's Q/K/V gate locations.

Three structural rules hold in every mode (proposal §12.2):
  1. CLS (position 0) receives NO direct SRP projection. Its value is
     left exactly as unmodified attention produced it. (It may still
     inherit SRP's effect indirectly via modified patch values in pre_v.)
  2. Pad-duplicate rows (positions 1+N .. H*W) receive NO SRP projection.
     They are artifacts of the square-pad step for PPEG (aggregator.py
     §4.1) and have no real coordinates.
  3. Fully-isolated real patches (|N(i)|=0) pass through as z_i = y_i by
     construction — F.normalize of a zero vector yields zero, so the
     projection term is exactly zero.

Shape conventions inside the forward:
  x          (B, 1 + H*W, C) -- CLS + real patches + pad dupes
  N_real     = number of REAL patches (< H*W when add > 0)
  is_real    (B, 1 + H*W) bool -- True at real-patch slots only
  neighbor_index / neighbor_mask  (B, N_real, 8) -- real-patch rows only
  h_morph    (B, N_real) or None -- required only for post_agg_gated
  h_local    (B, N_real) or None -- required for signed-gated SRP modes

The Nyström math (QKV, landmarks, factor softmaxes, Moore-Penrose inverse,
output y) is a verbatim port of slide_level/src/nystrom_xsa.py to keep
baseline/xsa_all bit-for-bit reproducible with stage 2. Only the
post-attention projection differs.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rcd_modules import IdentitySafeRCDRecomposer, LearnedLocalContextDirection


# Beta modes for beta_patch.
#   "zero"  -> β = 0 (no projection), buffer
#   "one"   -> β = 1 (full projection / orthogonalization), buffer
#   "learn" -> β learnable nn.Parameter, init from beta_init
#   "fixed" -> β frozen at beta_init (arbitrary value), buffer
_BETA_MODES = ("zero", "one", "learn", "fixed")
# Phase-A signed-gate mode (LEARNED_GATE_SRP_PROPOSAL.md §2). Treated
# as a sentinel: when beta_patch_mode == "signed_gated", the per-head
# scalar `beta_patch` parameter is replaced by a per-(token, head)
# learned tensor produced by a TokenHeadGate.
_BETA_SIGNED_GATED = "signed_gated"

# SRP projection modes.
_SRP_MODES = (
    "post_agg",
    "pre_v",
    "pre_q_signed_gated",
    "pre_k_signed_gated",
    "pre_v_signed_gated",
    "post_agg_gated",
    "post_agg_signed_gated",
    "post_agg_signed_gated_learned_r",
    "post_agg_mlp_control",
    "post_agg_rcd",
    "post_agg_rcd_learned_r",
)
_SRP_PRE_ATTENTION_SIGNED_GATE_MODES = (
    "pre_q_signed_gated",
    "pre_k_signed_gated",
    "pre_v_signed_gated",
)
# Both modes use the original signed-gate projection surface.  The learned-r
# variant changes only how the local direction r_hat is estimated.
_SRP_SIGNED_GATE_MODES = (
    *_SRP_PRE_ATTENTION_SIGNED_GATE_MODES,
    "post_agg_signed_gated",
    "post_agg_signed_gated_learned_r",
)
_GATE_COUNT_FEATURES = ("legacy", "rawlog", "normlog", "none")


def moore_penrose_iter_inv(A: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """
    Iterative Moore-Penrose pseudo-inverse (Xiong et al. 2021).

    Re-exported verbatim from slide_level/src/nystrom_xsa.py so the
    baseline (beta=0) path through NystromSRPAttention is mathematically
    identical to NystromXSAAttention with alpha=0.
    """
    abs_A = torch.abs(A)
    col = abs_A.sum(dim=-1)
    row = abs_A.sum(dim=-2)
    # Global-scalar init (matches lucidrains nystrom-attention and
    # therefore the official TransMIL import path).
    max_col = torch.max(col)
    max_row = torch.max(row)
    V = A.transpose(-1, -2) / (max_col * max_row)

    m_ = A.shape[-1]
    I = torch.eye(m_, device=A.device, dtype=A.dtype)
    I = I.expand_as(A)
    for _ in range(iters):
        AV = A @ V
        V = 0.25 * V @ (13 * I - (AV @ (15 * I - (AV @ (7 * I - AV)))))
    return V


def _pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, int]:
    """Pad a (..., N, D) tensor at the END to make N a multiple of m."""
    N = x.shape[-2]
    remainder = N % m
    if remainder == 0:
        return x, 0
    pad = m - remainder
    return F.pad(x, (0, 0, 0, pad)), pad


def gather_neighbors(
    v_patch: torch.Tensor,                # (B, H, N, D) -- caller may pass v_patch.detach()
    neighbor_index: torch.Tensor,         # (B, N, K) long, -1 for invalid slots
    neighbor_mask: torch.Tensor,          # (B, N, K) bool, True for valid slots
) -> torch.Tensor:
    """
    Gather per-patch neighbor value vectors, returning the full stacked
    tensor of shape (B, H, N, K, D). Invalid slots (-1 index, False mask)
    are still physically present as gathered values from clamped-to-0
    index, but the caller is responsible for masking them out before any
    reduction. The mask itself is returned to the caller elsewhere.

    Rationale for returning raw stacked neighbors rather than the mean:
    h^V diagnostic (§8.2.B) requires per-neighbor cosines, not just the
    mean direction.
    """
    B, H, N, D = v_patch.shape
    # Clamp -1 entries to 0 so advanced indexing doesn't wrap-around into
    # a valid but semantically wrong row (PyTorch interprets negative
    # indices as Python-style wrap-around — index -1 reads the LAST row).
    K = neighbor_index.shape[-1]
    safe_idx = neighbor_index.clamp(min=0)                    # (B, N, K)
    batch_idx = torch.arange(B, device=v_patch.device).view(B, 1, 1).expand(B, N, K)
    # Advanced indexing: result has shape (B, N, 8, H, D) — the un-indexed
    # head axis (1 of v_patch) is appended after the indexed axes.
    neighbor_v = v_patch[batch_idx, :, safe_idx, :]            # (B, N, K, H, D)
    neighbor_v = neighbor_v.permute(0, 3, 1, 2, 4)             # (B, H, N, K, D)
    return neighbor_v


def neighborhood_mean(
    neighbor_v: torch.Tensor,              # (B, H, N, K, D)
    neighbor_mask: torch.Tensor,           # (B, N, K) bool
    neighbor_weight: Optional[torch.Tensor] = None,  # (B, N, K) float or None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the masked per-patch neighborhood mean r_j and its unit form.

    Returns (r, r_hat, cnt):
      r      (B, H, N, D)  -- masked mean of neighbor v_k (zero if |N(j)|=0)
      r_hat  (B, H, N, D)  -- unit direction (near-zero if r≈0 via eps)
      cnt    (B, 1, N, 1)  -- valid-neighbor count per patch (float)
    """
    B, H, N, K, D = neighbor_v.shape
    mask_f = neighbor_mask.unsqueeze(1).unsqueeze(-1).to(neighbor_v.dtype)   # (B, 1, N, 8, 1)
    cnt = mask_f.sum(dim=3)                                                   # (B, 1, N, 1)
    if neighbor_weight is not None:
        if neighbor_weight.shape != (B, N, K):
            raise ValueError(
                f"neighbor_weight shape mismatch: got {tuple(neighbor_weight.shape)}, "
                f"expected ({B}, {N}, {K})"
            )
        weight_f = neighbor_weight.unsqueeze(1).unsqueeze(-1).to(neighbor_v.dtype)
        mask_f = mask_f * weight_f
    s = (neighbor_v * mask_f).sum(dim=3)                                      # (B, H, N, D)
    # For fully-isolated tokens (cnt=0), keep r = 0 (clamp avoids NaN).
    denom = mask_f.sum(dim=3).clamp(min=1.0)
    r = s / denom
    r_hat = F.normalize(r, dim=-1, eps=1e-12)
    return r, r_hat, cnt


def _gate_num_token_features(mode: str, include_y_norm_mean: bool = False) -> int:
    if mode not in _GATE_COUNT_FEATURES:
        raise ValueError(f"gate_count_features must be one of {_GATE_COUNT_FEATURES}, got {mode!r}")
    base = 1  # h_local
    if mode == "legacy":
        base += 2  # count / K and raw log count, preserving old behavior.
    elif mode in ("rawlog", "normlog"):
        base += 1
    # mode == "none" keeps h_local only.
    if include_y_norm_mean:
        base += 1
    return base


def _make_token_diag(
    *,
    h_local: torch.Tensor,
    cnt_bn: torch.Tensor,
    max_neighbors: int,
    mode: str,
    y_norm_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cols = [h_local]
    if mode == "legacy":
        cols.append(cnt_bn / float(max_neighbors))
        cols.append(torch.log1p(cnt_bn))
    elif mode == "rawlog":
        cols.append(torch.log1p(cnt_bn))
    elif mode == "normlog":
        cols.append(torch.log1p(cnt_bn) / math.log1p(float(max_neighbors)))
    elif mode == "none":
        pass
    else:
        raise ValueError(f"gate_count_features must be one of {_GATE_COUNT_FEATURES}, got {mode!r}")
    if y_norm_mean is not None:
        cols.append(y_norm_mean)
    return torch.stack(cols, dim=-1)


def _matched_mlp_control_hidden_dim(
    *,
    head_dim: int,
    num_heads: int,
    gate_hidden_dim: int,
    num_token_features: int,
    num_head_features: int,
) -> int:
    """Choose a small adapter bottleneck close to the signed gate size.

    The capacity-control arm is meant to ask whether the learned gate's
    improvement is generic extra nonlinear capacity rather than SRP geometry.
    A full D x D adapter would dwarf the gate, so this helper estimates the
    active full-gate parameter count and picks the nearest shared per-head MLP
    bottleneck.  For the reported default (6 heads, head_dim 64,
    gate_hidden_dim 64), this selects rank 2: 322 adapter parameters versus
    roughly 351 gate parameters per active block.
    """
    token_branch = gate_hidden_dim * num_token_features + gate_hidden_dim
    token_out = gate_hidden_dim + 1
    head_branch = num_heads * num_head_features
    head_bias = num_heads
    layer_head_bias = num_heads
    gate_params = token_branch + token_out + head_branch + head_bias + layer_head_bias
    # Shared per-head MLP parameter count is:
    #   fc1: head_dim * r + r
    #   fc2: r * head_dim + head_dim
    # The final +head_dim term is unavoidable, so clamp at rank 1 if the gate
    # is very small in a toy test.
    rank = round((gate_params - head_dim) / max(1, (2 * head_dim + 1)))
    return max(1, int(rank))


class NoGeometryMLPControl(nn.Module):
    """Zero-initialized plain adapter for mechanism-vs-capacity controls.

    This module intentionally consumes only the post-attention token stream
    `y_patch`; it never receives neighbor vectors, `r_hat`, `h_local`, or
    `h_morph`.  The output projection is zero-initialized so the arm starts
    exactly at the baseline function, while gradients still flow into the
    output path because the hidden activation is non-zero.
    """

    def __init__(self, head_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(head_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, head_dim)
        self.reset_output_path()

    def reset_output_path(self) -> None:
        """Keep adapter insertion an exact identity at construction."""
        with torch.no_grad():
            self.fc2.weight.zero_()
            if self.fc2.bias is not None:
                self.fc2.bias.zero_()

    def forward(self, y_patch: torch.Tensor) -> torch.Tensor:
        # Linear layers operate on the last axis, so this supports
        # (B, H, N, D) directly without reshaping head/token axes.
        return self.fc2(F.gelu(self.fc1(y_patch)))


def collect_mlp_control_module_ids(model: nn.Module) -> set[int]:
    """Return ids for every MLP-control module and child module.

    The parent aggregator's broad `_init_weights()` pass must skip these
    modules, otherwise it would overwrite the zero output projection and shift
    the non-method initialization stream relative to the baseline arm.
    """
    module_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, NoGeometryMLPControl):
            module_ids.update(id(child) for child in module.modules())
    return module_ids


def reset_mlp_control_modules(modules) -> None:
    """Re-apply identity output init after parent initialization passes."""
    for module in modules:
        if isinstance(module, NoGeometryMLPControl):
            module.reset_output_path()


class NystromSRPAttention(nn.Module):
    """
    Nyström self-attention with Spatial Redundancy Projection.

    Interface differs from NystromXSAAttention only in:
      (a) forward takes (x, neighbor_index, neighbor_mask, is_real, h_morph)
      (b) alpha_* are replaced by a single beta_patch per head (no beta_cls;
          CLS is structurally untouched)
      (c) srp_mode selects post_agg / pre_v / post_agg_gated
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        num_landmarks: int = 64,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        beta_patch_mode: str = "learn",       # see _BETA_MODES + signed_gated
        beta_init: float = 1.0,
        srp_mode: str = "post_agg",           # see _SRP_MODES
        num_cls_tokens: int = 1,
        pinv_iterations: int = 6,
        # Signed-gate parameters; consumed only under
        # srp_mode == "post_agg_signed_gated".
        delta_scale: float = 2.0,
        gate_active: bool = True,
        gate_hidden_dim: int = 16,
        gate_output_init: str = "zero",
        gate_output_init_scale: float = 1.0,
        gate_init_beta0: float = 0.0,
        gate_activation: str = "tanh",
        gate_activation_temperature: float = 1.0,
        gate_factorization: str = "full",
        gate_count_features: str = "legacy",
        # Post-Phase-A refinement methods.  These are consumed only by
        # srp_mode in {"post_agg_rcd", "post_agg_rcd_learned_r"} and are
        # constructed as separate modules so original SRP / signed-gate
        # paths remain untouched.
        rcd_adapter_kind: str = "lowrank",
        rcd_rank: int = 16,
        learned_r_hidden_dim: int = 16,
        # When False, gate diagnostic inputs (head_diag includes
        # cos_yr / |cos_yr| / log_norm_y, all derived from live y) are
        # NOT detached before the gate forward. Default True is the
        # proposal §6.3 detached regime; setting False enables the
        # "live" regime that empirically beat detached on ADP by
        # +0.98 pp paired (§6.3.1).
        detach_gate_inputs: bool = True,
    ) -> None:
        super().__init__()
        # Phase-A.9 fourth-review fix F4: constructor-arg validation must
        # raise ValueError so it survives `python -O` (asserts are
        # stripped). These guard against silent wrong-config bugs.
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        if beta_patch_mode not in _BETA_MODES + (_BETA_SIGNED_GATED,):
            raise ValueError(
                f"beta_patch_mode must be in "
                f"{_BETA_MODES + (_BETA_SIGNED_GATED,)}, got {beta_patch_mode!r}"
            )
        if srp_mode not in _SRP_MODES:
            raise ValueError(
                f"srp_mode must be in {_SRP_MODES}, got {srp_mode!r}"
            )
        if num_cls_tokens < 0:
            raise ValueError(
                f"num_cls_tokens must be >= 0, got {num_cls_tokens}"
            )
        # The signed-gated path requires both axes to agree: the SRP mode
        # must be one of the signed-gate surfaces AND the beta-mode must be
        # signed_gated.  This prevents accidentally routing signed β through
        # scalar post_agg/pre_v branches where the semantics differ.
        if srp_mode in _SRP_SIGNED_GATE_MODES and beta_patch_mode != _BETA_SIGNED_GATED:
            raise ValueError(
                f"srp_mode={srp_mode!r} requires "
                f"beta_patch_mode='signed_gated' (got {beta_patch_mode!r})"
            )
        if beta_patch_mode == _BETA_SIGNED_GATED and srp_mode not in _SRP_SIGNED_GATE_MODES:
            raise ValueError(
                "beta_patch_mode='signed_gated' requires "
                f"srp_mode in {_SRP_SIGNED_GATE_MODES} (got {srp_mode!r})"
            )
        if srp_mode in ("post_agg_rcd", "post_agg_rcd_learned_r") and beta_patch_mode != "zero":
            raise ValueError(
                f"srp_mode={srp_mode!r} uses the identity-safe RCD "
                "recomposer instead of beta_patch; set beta_patch_mode='zero' "
                f"to make the disabled beta path explicit (got {beta_patch_mode!r})."
            )
        if srp_mode == "post_agg_mlp_control" and beta_patch_mode != "zero":
            raise ValueError(
                "srp_mode='post_agg_mlp_control' is a plain adapter "
                "capacity control and must set beta_patch_mode='zero' so "
                f"the SRP projection path is explicit disabled (got {beta_patch_mode!r})."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_landmarks = num_landmarks
        self.num_cls_tokens = num_cls_tokens
        self.beta_patch_mode = beta_patch_mode
        self.srp_mode = srp_mode
        self.pinv_iterations = pinv_iterations
        self.detach_gate_inputs = bool(detach_gate_inputs)
        if gate_count_features not in _GATE_COUNT_FEATURES:
            raise ValueError(
                f"gate_count_features must be one of {_GATE_COUNT_FEATURES}, "
                f"got {gate_count_features!r}"
            )
        self.gate_count_features = gate_count_features

        # QKV, proj, dropout — identical to stage 2.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # beta_patch: per-head scalar, shape (num_heads,).
        # No beta_cls — CLS is left untouched per proposal §2.4.
        if beta_patch_mode == "learn":
            self.beta_patch = nn.Parameter(
                torch.full((num_heads,), float(beta_init))
            )
        elif beta_patch_mode == _BETA_SIGNED_GATED:
            # The per-head scalar β is replaced by per-(token, head)
            # β_eff produced by a TokenHeadGate. We still register a
            # zero-valued buffer named `beta_patch` so legacy diagnostic
            # code (e.g., diagnostic_log() callers that read
            # self.beta_patch directly for capture) does not error out;
            # the actual β_eff for forward comes from self.gate.
            self.register_buffer(
                "beta_patch", torch.zeros(num_heads), persistent=True,
            )
        else:
            # "zero" → 0.0, "one" → 1.0, "fixed" → beta_init.
            # "zero"/"one" are legacy shorthand kept for backward compat with
            # the Phase-1 ablations (baseline, srp_patch_hard). "fixed" is
            # the general form used by the Phase-1.5+ β-grid ablations.
            if beta_patch_mode == "zero":
                fixed = 0.0
            elif beta_patch_mode == "one":
                fixed = 1.0
            else:  # "fixed"
                fixed = float(beta_init)
            self.register_buffer(
                "beta_patch",
                torch.full((num_heads,), fixed),
                persistent=True,
            )

        # Signed-gate state. The parent aggregator owns the placement-aware
        # final-block policy: post-attention and patch-query-only gates stay
        # disabled in the last block, while pre-K/pre-V can remain active
        # because their edited streams are consumed by the final CLS row.
        self.delta_scale = float(delta_scale)
        self.gate_active = bool(gate_active and srp_mode in _SRP_SIGNED_GATE_MODES)
        # Standalone Method 2.4 keeps the signed-gate intervention but swaps
        # the fixed neighborhood mean r_hat for the learned local scorer.  It
        # is intentionally separate from learned_r_active, which belongs to
        # the RCD recomposer path.
        self.learned_r_gate_active = bool(
            gate_active and srp_mode == "post_agg_signed_gated_learned_r"
        )
        if self.gate_active:
            # Save+restore RNG state so the gate's nn.Linear init does
            # not shift downstream init draws — same reasoning as in
            # src/vit_panda.py.
            from slide_level_srp.src.gate_signed import TokenHeadGate
            rng_state_cpu = torch.get_rng_state()
            try:
                self.gate = TokenHeadGate(
                    num_heads=num_heads,
                    # CAM17 token-level features (3): h_local,
                    # neighbour_count/8, log(1+neighbour_count). We omit
                    # `local feature variance` from the PANDA gate
                    # because Stage-3 doesn't precompute it and the
                    # extra channel is not yet validated as informative.
                    num_token_features=_gate_num_token_features(
                        self.gate_count_features,
                        include_y_norm_mean=False,
                    ),
                    # CAM17 head-level features (3): cos(y, r̂),
                    # |cos(y, r̂)|, log_norm_y — same as PANDA.
                    num_head_features=3,
                    hidden_dim=gate_hidden_dim,
                    delta_scale=self.delta_scale,
                    output_init=gate_output_init,
                    output_init_scale=gate_output_init_scale,
                    init_beta0=gate_init_beta0,
                    activation=gate_activation,
                    activation_temperature=gate_activation_temperature,
                    factorization=gate_factorization,
                )
            finally:
                torch.set_rng_state(rng_state_cpu)
        else:
            self.gate = None

        self.mlp_control_active = bool(
            gate_active and srp_mode == "post_agg_mlp_control"
        )
        if self.mlp_control_active:
            # Match the current gate's parameter scale without perturbing the
            # baseline initialization stream.  Construction-time random draws
            # are restored for downstream modules; the parent aggregator also
            # skips this adapter during its broad `_init_weights()` pass.
            rng_state_cpu = torch.get_rng_state()
            try:
                control_hidden_dim = _matched_mlp_control_hidden_dim(
                    head_dim=self.head_dim,
                    num_heads=num_heads,
                    gate_hidden_dim=gate_hidden_dim,
                    num_token_features=_gate_num_token_features(
                        self.gate_count_features,
                        include_y_norm_mean=False,
                    ),
                    num_head_features=3,
                )
                self.mlp_control = NoGeometryMLPControl(
                    head_dim=self.head_dim,
                    hidden_dim=control_hidden_dim,
                )
            finally:
                torch.set_rng_state(rng_state_cpu)
        else:
            self.mlp_control = None

        self.rcd_active = bool(
            gate_active and srp_mode in ("post_agg_rcd", "post_agg_rcd_learned_r")
        )
        self.learned_r_active = bool(
            self.rcd_active and srp_mode == "post_agg_rcd_learned_r"
        )
        if self.rcd_active or self.learned_r_gate_active:
            # Preserve the global RNG stream exactly as the signed-gate path
            # does.  RCD / learned-r modules have their own parameters, but
            # their construction should not perturb non-method init at a
            # shared seed.
            rng_state_cpu = torch.get_rng_state()
            try:
                self.rcd_recomposer = (
                    IdentitySafeRCDRecomposer(
                        head_dim=self.head_dim,
                        rank=rcd_rank,
                        adapter_kind=rcd_adapter_kind,
                    )
                    if self.rcd_active
                    else None
                )
                needs_learned_r = self.learned_r_active or self.learned_r_gate_active
                self.context_scorer = (
                    LearnedLocalContextDirection(
                        head_dim=self.head_dim,
                        hidden_dim=learned_r_hidden_dim,
                        use_h_local=True,
                    )
                    if needs_learned_r
                    else None
                )
            finally:
                torch.set_rng_state(rng_state_cpu)
        else:
            self.rcd_recomposer = None
            self.context_scorer = None
        # Per-forward gate-output cache for diagnostics (mirrors PANDA).
        self._last_gate_stats: dict | None = None
        # Differentiable per-forward gate cache used by the optional
        # training-time L2 regularizer.  `_last_gate_stats` intentionally
        # stores detached tensors for diagnostics, so the loss path needs
        # its own non-detached handle.  It is overwritten every forward and
        # therefore does not retain history across batches.
        self._last_gate_beta_eff_for_loss: Optional[torch.Tensor] = None
        self._last_rcd_stats: dict | None = None

        # Diagnostic capture toggle + stash.
        self._capture_stats = False
        self.last_stats: dict | None = None

    # --- Nyström core -------------------------------------------------

    def _nystrom_y(
        self,
        q: torch.Tensor,         # (B, H, L, D) -- caller pre-scales q
        k: torch.Tensor,         # (B, H, L, D)
        v: torch.Tensor,         # (B, H, L, D)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Run the Nyström attention path. Returns
        (y, F_soft, A_inv, B_soft, N_p, pad) where y has shape
        (B, H, L, D) at the real-token width and the Nyström factors are
        preserved for diagnostic reconstruction (cls_self_attn).
        """
        B, H, L, D = q.shape
        m = self.num_landmarks

        q_pad, pad = _pad_to_multiple(q, m)
        k_pad, _   = _pad_to_multiple(k, m)
        v_pad, _   = _pad_to_multiple(v, m)
        N_p = q_pad.shape[-2]
        seg = N_p // m

        if pad > 0:
            mask = torch.ones(B, 1, N_p, 1, device=q.device, dtype=torch.bool)
            mask[..., L:, :] = False
        else:
            mask = None

        def seg_mean(t: torch.Tensor) -> torch.Tensor:
            t = t.reshape(B, H, m, seg, D)
            if mask is not None:
                mseg = mask.reshape(B, 1, m, seg, 1).to(t.dtype)
                s = (t * mseg).sum(dim=3)
                cnt = mseg.sum(dim=3).clamp(min=1e-6)
                return s / cnt
            return t.mean(dim=3)

        q_tilde = seg_mean(q_pad)
        k_tilde = seg_mean(k_pad)

        kF_logits = q_pad @ k_tilde.transpose(-2, -1)
        kA_logits = q_tilde @ k_tilde.transpose(-2, -1)
        kB_logits = q_tilde @ k_pad.transpose(-2, -1)

        if pad > 0:
            mcol = mask.reshape(B, 1, 1, N_p)
            kB_logits = kB_logits.masked_fill(~mcol, float("-inf"))

        F_soft = kF_logits.softmax(dim=-1)
        A_soft = kA_logits.softmax(dim=-1)
        B_soft = kB_logits.softmax(dim=-1)
        F_soft = self.attn_drop(F_soft)
        B_soft = self.attn_drop(B_soft)

        A_inv = moore_penrose_iter_inv(A_soft, iters=self.pinv_iterations)

        BV = B_soft @ v_pad
        AinvBV = A_inv @ BV
        y_pad = F_soft @ AinvBV                         # (B, H, N_p, D)
        y = y_pad[..., :L, :]                           # (B, H, L, D)

        return y, F_soft, A_inv, B_soft, N_p, pad

    # --- forward -------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                    # (B, 1+H*W, C)
        neighbor_index: torch.Tensor,       # (B, N_real, 8) long
        neighbor_mask: torch.Tensor,        # (B, N_real, 8) bool
        is_real: torch.Tensor,              # (B, 1+H*W) bool
        h_morph: Optional[torch.Tensor] = None,   # (B, N_real) float, required for post_agg_gated
        h_local: Optional[torch.Tensor] = None,   # (B, N_real) float, required for post_agg_signed_gated
        neighbor_weight: Optional[torch.Tensor] = None,  # (B, N_real, K) float
    ) -> torch.Tensor:
        """
        x           padded sequence (CLS + real patches + pad dupes)
        neighbor_*  real-patch neighborhood structure (size-N_real rows only)
        is_real     per-position flag; True for positions 1..N_real, False
                    for position 0 (CLS) and positions 1+N_real..H*W (pad dupes)
        h_morph     slide-intrinsic gate signal (frozen raw UNI features);
                    required only when srp_mode == "post_agg_gated"
        h_local     per-token cosine-homogeneity input to the learned
                    signed gate (proposal §6.2); required only when
                    srp_mode == "post_agg_signed_gated". On CAM17 this
                    can be sourced from the existing h_morph if no
                    distinct h_local is precomputed (the two are
                    different definitions but functionally similar).
        """
        # Reset the gate-stats cache at the start of every forward, so
        # train.py's diagnostic hook reads only the current step.
        self._last_gate_stats = None
        self._last_gate_beta_eff_for_loss = None
        self._last_rcd_stats = None
        B, L, C = x.shape
        H, D = self.num_heads, self.head_dim
        # Phase-A.9 review fix F9: runtime input-contract checks now
        # raise ValueError so they survive `python -O`. (The
        # constructor invariants in __init__ remain plain `assert`
        # because they catch programmer errors at instantiation, not
        # data-driven runtime conditions.)
        if L < 1 + self.num_cls_tokens:
            raise ValueError(f"sequence too short: L={L}")
        if is_real.shape != (B, L):
            raise ValueError(
                f"is_real shape mismatch: got {tuple(is_real.shape)}, "
                f"expected ({B}, {L})"
            )
        # Under batch_size=1 + slide_collate, all slides in the batch have
        # the same N_real. If multi-slide batching is added later, this
        # module needs rework to handle per-slide N.
        per_batch_N = is_real.sum(dim=-1)
        if per_batch_N.min() != per_batch_N.max():
            raise ValueError(
                "NystromSRPAttention currently requires uniform N_real across "
                f"the batch (got {per_batch_N.tolist()}). batch_size=1 ensures this."
            )
        N_real = int(per_batch_N[0].item())
        if neighbor_index.ndim != 3 or neighbor_index.shape[:2] != (B, N_real):
            raise ValueError(
                f"neighbor_index shape mismatch: got {tuple(neighbor_index.shape)}, "
                f"expected ({B}, {N_real}, K)"
            )
        if neighbor_mask.shape != neighbor_index.shape:
            raise ValueError(
                f"neighbor_mask shape mismatch: got {tuple(neighbor_mask.shape)}, "
                f"expected {tuple(neighbor_index.shape)}"
            )
        if neighbor_weight is not None and neighbor_weight.shape != neighbor_index.shape:
            raise ValueError(
                f"neighbor_weight shape mismatch: got {tuple(neighbor_weight.shape)}, "
                f"expected {tuple(neighbor_index.shape)}"
            )
        neighbor_k = int(neighbor_index.shape[-1])

        # --- Q, K, V projection + head split -----------------------------
        qkv = self.qkv(x).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                       # each (B, H, L, D)

        # Keep pre-SRP references at patch rows.  The V reference is used by
        # the existing diagnostics; Q/K references are used only by the new
        # pre-attention signed-gate placements.  Slicing creates views, so
        # autograd still tracks gradients through the live stream tensors.
        q_patch_pre = q[:, :, 1 : 1 + N_real, :]      # (B, H, N_real, D)
        k_patch_pre = k[:, :, 1 : 1 + N_real, :]
        v_patch_pre = v[:, :, 1 : 1 + N_real, :]

        # --- Pre-attention signed-gated SRP (Q/K/V placement ablations) -
        v_patch_post_pre_v: Optional[torch.Tensor] = None   # populated under pre_v variants

        def _apply_pre_attention_signed_gate(
            stream: torch.Tensor,
            stream_patch: torch.Tensor,
            *,
            stream_name: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Apply learned signed SRP to real patch rows of Q/K/V.

            This helper intentionally mirrors the post-attention signed
            gate's β_eff construction, but it computes the local redundancy
            direction in the same projection space as the stream being
            edited.  That keeps the Q, K, and V ablations interpretable and
            avoids projecting Q/K against a V-space direction.
            """
            # A disabled gate is an exact identity.  The aggregator uses
            # this for final-block post-attention/pre-Q dead paths, so we do
            # not require h_local when no gate computation will happen.
            if not self.gate_active:
                return stream, stream_patch
            if h_local is None:
                raise ValueError(
                    f"srp_mode={self.srp_mode!r} with gate_active=True "
                    "requires h_local in forward()."
                )
            if h_local.shape != (B, N_real):
                raise ValueError(
                    f"h_local shape mismatch: got {tuple(h_local.shape)}, "
                    f"expected ({B}, {N_real})"
                )

            # Directions are detached for the same reason as the original
            # post-attention SRP path: the edited token receives gradients,
            # but the local mean direction is treated as a fixed geometric
            # context for this step.
            neighbor_stream_det = gather_neighbors(
                stream_patch.detach(), neighbor_index, neighbor_mask,
            )
            _r_det, r_hat_det, cnt = neighborhood_mean(
                neighbor_stream_det, neighbor_mask, neighbor_weight,
            )
            dot_ur = (stream_patch * r_hat_det).sum(dim=-1, keepdim=True)

            # Token diagnostics are placement-independent; the count channel
            # still describes the patch graph, not Q/K/V content.
            cnt_bn = cnt.squeeze(-1).squeeze(1).to(dtype=stream.dtype)
            token_diag = _make_token_diag(
                h_local=h_local.to(stream.dtype),
                cnt_bn=cnt_bn,
                max_neighbors=neighbor_k,
                mode=self.gate_count_features,
            )

            # Head diagnostics use the selected stream.  We keep the legacy
            # cache key `cos_yr` for downstream summaries, but for pre-Q/K/V
            # it should be read as cos(stream, r_hat_stream).
            eps = 1e-12
            stream_norms = stream_patch.norm(dim=-1)
            cos_ur = dot_ur.squeeze(-1) / (stream_norms + eps)
            head_diag = torch.stack(
                [cos_ur, cos_ur.abs(), torch.log1p(stream_norms)],
                dim=-1,
            )

            # Preserve the existing detach/live gate-input switch.  Only the
            # diagnostics are detached; the SRP correction below remains
            # differentiable with respect to the edited stream.
            if self.detach_gate_inputs:
                token_diag = token_diag.detach()
                head_diag = head_diag.detach()
            beta_eff = self.gate(token_diag, head_diag)
            self._last_gate_beta_eff_for_loss = beta_eff
            with torch.no_grad():
                self._last_gate_stats = {
                    "beta_eff": beta_eff.detach(),
                    "cos_yr": cos_ur.detach(),
                    "y_norms": stream_norms.detach(),
                    "h_local": h_local.detach(),
                    "neighbour_count": cnt_bn.detach(),
                    "gate_stream": stream_name,
                }

            # β_eff=0 is identity, β_eff=1 removes the local component, and
            # β_eff=2 reflects it.  Only real patch rows are written back;
            # CLS and pad duplicates preserve the incoming stream exactly.
            stream_patch_new = stream_patch - beta_eff * dot_ur * r_hat_det
            stream_new = stream.clone()
            stream_new[:, :, 1 : 1 + N_real, :] = stream_patch_new
            return stream_new, stream_patch_new

        if self.srp_mode == "pre_q_signed_gated":
            q, q_patch_pre = _apply_pre_attention_signed_gate(
                q, q_patch_pre, stream_name="q",
            )
        elif self.srp_mode == "pre_k_signed_gated":
            k, k_patch_pre = _apply_pre_attention_signed_gate(
                k, k_patch_pre, stream_name="k",
            )
        elif self.srp_mode == "pre_v_signed_gated":
            v, v_patch_post_pre_v = _apply_pre_attention_signed_gate(
                v, v_patch_pre, stream_name="v",
            )

        # Nyström core expects the query tensor to be pre-scaled.  Scaling
        # after the pre-Q edit is mathematically equivalent to editing the
        # raw projected Q and then forming QK^T / sqrt(d_h).
        q = q * self.scale

        # --- Pre-aggregation scalar SRP (legacy pre_v) ------------------
        if self.srp_mode == "pre_v":
            # Detach the neighbor gather to block within-step gradient
            # flow through v_k via r_j. Gradient still flows through
            # v_patch_pre via the (v · r_hat) dot product.
            neighbor_v_det = gather_neighbors(
                v_patch_pre.detach(), neighbor_index, neighbor_mask,
            )
            _r, r_hat_det, _cnt = neighborhood_mean(
                neighbor_v_det, neighbor_mask, neighbor_weight,
            )

            beta_bh = self.beta_patch.to(dtype=v.dtype, device=v.device).view(1, H, 1, 1)
            dot_vr = (v_patch_pre * r_hat_det).sum(dim=-1, keepdim=True)   # (B, H, N, 1)
            v_patch_new = v_patch_pre - beta_bh * dot_vr * r_hat_det       # (B, H, N, D)

            # Clone v before writing: Nyström uses v later, and autograd
            # disallows in-place modification of a tensor that is an
            # input to an op that's already been queued for backward.
            v = v.clone()
            v[:, :, 1 : 1 + N_real, :] = v_patch_new
            v_patch_post_pre_v = v_patch_new

        # --- Nyström attention ------------------------------------------
        y, F_soft, A_inv, B_soft, N_p, pad = self._nystrom_y(q, k, v)
        y_patch = y[:, :, 1 : 1 + N_real, :]          # (B, H, N_real, D)

        # --- Post-aggregation SRP (modifies y on patch rows) ------------
        r_for_diag: Optional[torch.Tensor] = None
        r_hat_for_diag: Optional[torch.Tensor] = None
        cnt_for_diag: Optional[torch.Tensor] = None
        neighbor_v_for_diag: Optional[torch.Tensor] = None

        if self.srp_mode in (
            "post_agg",
            "post_agg_gated",
            "post_agg_signed_gated",
            "post_agg_signed_gated_learned_r",
            "post_agg_mlp_control",
            "post_agg_rcd",
            "post_agg_rcd_learned_r",
        ):
            # r̂ from the ORIGINAL v (unmodified even under gated paths;
            # gating does not touch v). Detached by design.
            neighbor_v_det = gather_neighbors(
                v_patch_pre.detach(), neighbor_index, neighbor_mask,
            )
            if self.learned_r_active or self.learned_r_gate_active:
                if h_local is None:
                    raise ValueError(
                        f"srp_mode={self.srp_mode!r} requires "
                        "h_local in forward() for Method 2.4 scoring."
                    )
                if h_local.shape != (B, N_real):
                    raise ValueError(
                        f"h_local shape mismatch: got {tuple(h_local.shape)}, "
                        f"expected ({B}, {N_real})"
                    )
                r_det, r_hat_det, cnt, learned_r_weight = self.context_scorer(
                    center_v=v_patch_pre.detach(),
                    neighbor_v=neighbor_v_det,
                    neighbor_mask=neighbor_mask,
                    neighbor_weight=neighbor_weight,
                    h_local=h_local,
                )
                with torch.no_grad():
                    self._last_rcd_stats = {
                        "learned_r_weight": learned_r_weight.detach(),
                    }
            else:
                r_det, r_hat_det, cnt = neighborhood_mean(
                    neighbor_v_det, neighbor_mask, neighbor_weight,
                )
            r_for_diag = r_det
            r_hat_for_diag = r_hat_det
            cnt_for_diag = cnt
            neighbor_v_for_diag = neighbor_v_det

            dot_yr = (y_patch * r_hat_det).sum(dim=-1, keepdim=True)       # (B, H, N, 1)

            if self.srp_mode == "post_agg":
                beta_bh = self.beta_patch.to(dtype=y.dtype, device=y.device).view(1, H, 1, 1)
                z_patch = y_patch - beta_bh * dot_yr * r_hat_det
            elif self.srp_mode == "post_agg_gated":
                # Phase-A.9 fourth-review fix F4: required-input checks
                # raise ValueError so they survive `python -O`. These are
                # data-contract violations that should always be visible.
                # post_agg_gated: β_i = β_base · clamp(h_morph_i, 0, 1).
                if h_morph is None:
                    raise ValueError(
                        "srp_mode='post_agg_gated' requires h_morph to be supplied"
                    )
                if h_morph.shape != (B, N_real):
                    raise ValueError(
                        f"h_morph shape mismatch: got {tuple(h_morph.shape)}, "
                        f"expected ({B}, {N_real})"
                    )
                # Gate is slide-intrinsic (shared across heads and layers).
                # Broadcast to (B, 1, N, 1).
                gate = h_morph.clamp(0.0, 1.0).to(dtype=y.dtype).unsqueeze(1).unsqueeze(-1)
                beta_base = self.beta_patch.to(dtype=y.dtype, device=y.device).view(1, H, 1, 1)
                beta_eff = beta_base * gate                                 # (B, H, N, 1)
                z_patch = y_patch - beta_eff * dot_yr * r_hat_det
            elif self.srp_mode in _SRP_SIGNED_GATE_MODES:
                # Signed-gated SRP: per-(token, head) β_eff from a learned
                # TokenHeadGate (proposal §2).  The learned-r variant reuses
                # this exact intervention formula after replacing the fixed
                # neighborhood r_hat above with Method 2.4's local scorer.
                # Identity at init holds because β_eff = 0 under zero-init.
                if self.gate_active:
                    # Phase-A.9 fourth-review fix F4: required-input
                    # contract for the live signed-gated forward path —
                    # raise ValueError so it survives `python -O`.
                    if h_local is None:
                        raise ValueError(
                            f"srp_mode={self.srp_mode!r} with "
                            "gate_active=True requires h_local in forward()."
                        )
                    if h_local.shape != (B, N_real):
                        raise ValueError(
                            f"h_local shape mismatch: got "
                            f"{tuple(h_local.shape)}, expected ({B}, {N_real})"
                        )
                    # Per-token diagnostics (3 channels, matching the
                    # gate's num_token_features=3 in __init__).
                    cnt_bn = cnt.squeeze(-1).squeeze(1).to(dtype=y.dtype)  # (B, N)
                    token_diag = _make_token_diag(
                        h_local=h_local.to(y.dtype),
                        cnt_bn=cnt_bn,
                        max_neighbors=neighbor_k,
                        mode=self.gate_count_features,
                    )
                    # Per-head diagnostics: cos(y, r̂), |cos|, log_norm_y.
                    eps = 1e-12
                    y_norms = y_patch.norm(dim=-1)                        # (B, H, N)
                    cos_yr = dot_yr.squeeze(-1) / (y_norms + eps)         # (B, H, N)
                    head_diag = torch.stack(
                        [cos_yr, cos_yr.abs(), torch.log1p(y_norms)],
                        dim=-1,
                    )                                                     # (B, H, N, 3)
                    # Proposal §6.3 detach convention: by default, gate
                    # diagnostic inputs are stop-grad'd. self.detach_gate_inputs
                    # toggles this — see __init__ for rationale; the +0.98
                    # pp ADP detach finding (§6.3.1) is what motivates the
                    # flag.
                    if self.detach_gate_inputs:
                        token_diag = token_diag.detach()
                        head_diag = head_diag.detach()
                    beta_eff = self.gate(token_diag, head_diag)           # (B, H, N, 1)
                    self._last_gate_beta_eff_for_loss = beta_eff
                    with torch.no_grad():
                        self._last_gate_stats = {
                            "beta_eff": beta_eff.detach(),
                            "cos_yr": cos_yr.detach(),
                            "y_norms": y_norms.detach(),
                            "h_local": h_local.detach(),
                            "neighbour_count": cnt_bn.detach(),
                        }
                else:
                    # Dead-path block: β_eff = 0, equivalent to identity.
                    beta_eff = torch.zeros_like(dot_yr)
                z_patch = y_patch - beta_eff * dot_yr * r_hat_det
            elif self.srp_mode in ("post_agg_rcd", "post_agg_rcd_learned_r"):
                # Method 2.1 / corrected RCD: decompose y into common and
                # residual branches, then add zero-initialized non-shared
                # branch deltas.  post_agg_rcd_learned_r uses the same
                # recomposer but replaces the fixed neighbour mean above
                # with the learned local context direction from Method 2.4.
                if not self.rcd_active:
                    # Last block dead-path: final post-attention patch
                    # writes have no downstream consumer under CLS readout.
                    z_patch = y_patch
                else:
                    z_patch, rcd_stats = self.rcd_recomposer(y_patch, r_hat_det)
                    with torch.no_grad():
                        merged = {} if self._last_rcd_stats is None else dict(self._last_rcd_stats)
                        merged.update(rcd_stats)
                        self._last_rcd_stats = merged
            elif self.srp_mode == "post_agg_mlp_control":
                # Mechanism-vs-capacity control: add a small plain adapter to
                # the post-attention patch stream, without using r_hat,
                # neighbor vectors, h_local, or h_morph to form the model
                # output.  The r_hat computed above is diagnostic-only so this
                # arm keeps the same per-slide SRP audit files as other SRP
                # backend runs while preserving a no-geometry intervention.
                if self.mlp_control_active:
                    z_patch = y_patch + self.mlp_control(y_patch)
                else:
                    # Final-block post-attention patch writes have no CLS
                    # path, matching the signed post-attention gate policy.
                    z_patch = y_patch
            else:
                raise RuntimeError(f"Unhandled post-aggregation srp_mode: {self.srp_mode}")

        else:
            # Pre-attention placements leave the attention output itself
            # untouched: the selected Q/K/V stream has already absorbed the
            # SRP correction before Nyström attention runs.
            z_patch = y_patch
            # Still compute V-space r̂ for diagnostics (placement-comparison
            # signatures per §8.2.E want cos(y, r) under pre-attention
            # placements too).
            neighbor_v_det = gather_neighbors(
                v_patch_pre.detach(), neighbor_index, neighbor_mask,
            )
            r_det, r_hat_det, cnt = neighborhood_mean(
                neighbor_v_det, neighbor_mask, neighbor_weight,
            )
            r_for_diag = r_det
            r_hat_for_diag = r_hat_det
            cnt_for_diag = cnt
            neighbor_v_for_diag = neighbor_v_det

        # Clone-and-update: patch rows get z_patch; CLS and pad-dupe rows
        # pass through from y unchanged (proposal §12.2 rules 1 & 2).
        z = y.clone()
        z[:, :, 1 : 1 + N_real, :] = z_patch

        # --- Diagnostic capture (only when toggled) ---------------------
        if self._capture_stats:
            self.last_stats = self._capture(
                y=y, z=z, v=v,
                v_patch_pre=v_patch_pre,
                v_patch_post_pre_v=v_patch_post_pre_v,
                r=r_for_diag,
                r_hat=r_hat_for_diag,
                cnt=cnt_for_diag,
                neighbor_v_det=neighbor_v_for_diag,
                neighbor_mask=neighbor_mask,
                F_soft=F_soft, A_inv=A_inv, B_soft=B_soft,
                h_morph=h_morph,
                N_real=N_real,
            )

        # --- Output projection ------------------------------------------
        z = z.transpose(1, 2).reshape(B, L, C)
        out = self.proj(z)
        out = self.proj_drop(out)
        return out

    # --- diagnostic capture -------------------------------------------

    def _capture(
        self,
        *,
        y: torch.Tensor,                     # (B, H, L, D) pre-SRP output
        z: torch.Tensor,                     # (B, H, L, D) post-SRP output
        v: torch.Tensor,                     # (B, H, L, D) values used in Nyström
                                             #   (under pre_v this is the modified v;
                                             #    under post_agg / gated it's the original)
        v_patch_pre: torch.Tensor,           # (B, H, N_real, D) ORIGINAL patch v
        v_patch_post_pre_v: Optional[torch.Tensor],  # (B, H, N_real, D) or None
        r: torch.Tensor,                     # (B, H, N_real, D) raw neighborhood mean
        r_hat: torch.Tensor,                 # (B, H, N_real, D) unit-normalized
        cnt: torch.Tensor,                   # (B, 1, N_real, 1) valid-neighbor count
        neighbor_v_det: torch.Tensor,        # (B, H, N_real, 8, D)
        neighbor_mask: torch.Tensor,         # (B, N_real, 8) bool
        F_soft: torch.Tensor,                # (B, H, N_p, m)
        A_inv: torch.Tensor,                 # (B, H, m, m)
        B_soft: torch.Tensor,                # (B, H, m, N_p)
        h_morph: Optional[torch.Tensor],     # (B, N_real) float or None
        N_real: int,
    ) -> dict:
        """
        Captures (a) stage-2-equivalent diagnostics for strict continuity,
        (b) SRP-specific diagnostics (proposal §8.2.A, B, D).
        All output tensors are detached for memory safety.
        """
        B, H, L, D = y.shape
        n_cls = self.num_cls_tokens

        # --- Slice by role --------------------------------------------
        # CLS rows: positions 0..n_cls-1. Patch rows: positions n_cls..n_cls+N_real-1.
        # Pad-duplicate rows (positions n_cls+N_real..L-1) are intentionally
        # EXCLUDED from per-slide diagnostic means (proposal §12.2 rule 4).
        y_cls = y[:, :, :n_cls, :]
        z_cls = z[:, :, :n_cls, :]
        v_cls = v[:, :, :n_cls, :]
        y_patch = y[:, :, n_cls : n_cls + N_real, :]
        z_patch = z[:, :, n_cls : n_cls + N_real, :]

        # --- cos(y, v) and cos(z, v) per role (stage-2 continuity) -----
        # Patch-space is the SRP-relevant slice. CLS-space is kept for
        # direct comparability with stage-2 RESULTS.md.
        cos_yv_cls_pre = F.cosine_similarity(y_cls, v_cls, dim=-1).detach()        # (B, H, n_cls)
        cos_yv_cls_post = F.cosine_similarity(z_cls, v_cls, dim=-1).detach()
        cos_yv_patch_pre = F.cosine_similarity(y_patch, v_patch_pre, dim=-1).detach()
        cos_yv_patch_post = F.cosine_similarity(z_patch, v_patch_pre, dim=-1).detach()

        # --- cos(y, r) and cos(z, r) at patch rows (proposal §8.2.A) ---
        cos_yr_patch_pre = F.cosine_similarity(y_patch, r_hat, dim=-1).detach()
        cos_zr_patch_post = F.cosine_similarity(z_patch, r_hat, dim=-1).detach()

        # --- pre_v-only: cos(v, r) and per-token magnitude retention ρ ---
        if v_patch_post_pre_v is not None:
            cos_vr_patch_pre = F.cosine_similarity(
                v_patch_pre, r_hat, dim=-1,
            ).detach()
            cos_vr_patch_post = F.cosine_similarity(
                v_patch_post_pre_v, r_hat, dim=-1,
            ).detach()
            rho_patch = (
                v_patch_post_pre_v.norm(dim=-1) / (v_patch_pre.norm(dim=-1) + 1e-8)
            ).detach()
        else:
            cos_vr_patch_pre = None
            cos_vr_patch_post = None
            rho_patch = None

        # --- h^V: V-space neighborhood coherence (proposal §8.2.B) -----
        # h^V_i = (1/|N(i)|) Σ_{j in N(i)} cos(v_i.detach(), v_j.detach()).
        # Exact form requires per-neighbor cosines, not cos(v_i, mean(v_j)).
        # We have neighbor_v_det: (B, H, N_real, 8, D) already detached.
        v_patch_det = v_patch_pre.detach()                                   # (B, H, N, D)
        v_patch_norm = F.normalize(v_patch_det, dim=-1, eps=1e-12)          # (B, H, N, D)
        neighbor_norm = F.normalize(neighbor_v_det, dim=-1, eps=1e-12)       # (B, H, N, 8, D)
        # Broadcast v_patch_norm over the 8-axis.
        cos_per_neighbor = (v_patch_norm.unsqueeze(3) * neighbor_norm).sum(dim=-1)
        # Shape (B, H, N, 8). Mask invalid slots and average.
        mask_bn8 = neighbor_mask.unsqueeze(1).to(cos_per_neighbor.dtype)     # (B, 1, N, 8)
        # cnt was (B, 1, N, 1); squeeze the trailing 1 to (B, 1, N, 1) -> (B, 1, N).
        cnt_bn = cnt.squeeze(-1)                                             # (B, 1, N)
        h_V_patch = (cos_per_neighbor * mask_bn8).sum(dim=-1) / cnt_bn.clamp(min=1.0)
        h_V_patch = h_V_patch.detach()                                       # (B, H, N)

        # --- h^morph (if provided; single tensor shared across heads) ---
        h_morph_patch = h_morph.detach() if h_morph is not None else None    # (B, N) or None

        # --- legacy norms (stage-2 §8.1 continuity) -------------------
        y_norm = y.norm(dim=-1).detach()                                     # (B, H, L)
        v_norm = v.norm(dim=-1).detach()                                     # (B, H, L)
        z_norm = z.norm(dim=-1).detach()                                     # (B, H, L)

        # --- cls_self_attn (Nyström reconstruction, stage-2 formula) ---
        f_cls = F_soft[..., 0, :]              # (B, H, m)
        b_cls = B_soft[..., :, 0]              # (B, H, m)
        cls_self_attn = torch.einsum(
            "bhm,bhmn,bhn->bh", f_cls, A_inv, b_cls,
        ).detach()

        # --- Per-patch CLS attention for WSI heatmap overlays ----------
        # The model never materializes the full N x N attention matrix because
        # that would defeat the purpose of Nyström attention on large WSIs.
        # For interpretation, however, we only need the final CLS row.  We
        # reconstruct that single row as F_cls @ A_inv @ B and keep the real
        # patch columns.  Tiny negative values can appear from the pseudo-
        # inverse approximation; exporters should clamp only for visualization,
        # while the raw tensor here preserves the model-side evidence.
        AB = A_inv @ B_soft
        P_cls = torch.einsum(
            "bhm,bhmn->bhn", F_soft[:, :, 0, :], AB,
        )                                                    # (B, H, N_p)
        cls_patch_attn = P_cls[:, :, n_cls : n_cls + N_real].detach()

        # --- Per-patch Gated SRP intervention magnitude ----------------
        # This is the direct mechanism-side signal: how much the block changed
        # each patch token in attention-output space.  For post-attention SRP
        # modes the final block can be intentionally inactive because CLS reads
        # patch tokens before a same-block patch-row edit could reach the slide
        # head; downstream visualization therefore selects the last block with
        # non-zero correction instead of assuming the final block is gated.
        srp_correction_norm_patch = (y_patch - z_patch).norm(dim=-1).detach()
        srp_correction_frac_patch = (
            srp_correction_norm_patch / (y_patch.norm(dim=-1) + 1e-8)
        ).detach()

        # --- §8.2.E placement signature: cos(y_cls, r̄_cls) per (B, H) ---
        # r̄_cls = (1/N_real) Σ_j r_j -- the mean neighborhood direction
        # averaged over ALL real patches. NOT the unit-normalized
        # r̂_bar, since direction-magnitude mixing matters for the
        # "redundant content reaching CLS" interpretation. If pre_v
        # has stripped the redundancy before aggregation, cos(y_cls, r̄)
        # should be small; if post_agg hasn't touched y_cls, cos may be
        # larger. Whether this is a true asymmetry is exactly what
        # §8.2.E tests.
        r_bar = r.mean(dim=2)                                    # (B, H, D)
        y_cls_single = y[:, :, 0, :]                             # (B, H, D)
        cos_y_cls_rbar = F.cosine_similarity(y_cls_single, r_bar, dim=-1).detach()

        # --- §8.2.D3 attention-weighted magnitude retention (pre_v only) ---
        # bar_rho_cls = Σ_j a_cls,j · ||v'_j|| / Σ_j a_cls,j · ||v_j||.
        # In Nyström, the effective attention from CLS to patch j is
        # P_cls,j = Σ_m F[0, m] · (A_inv · B)[m, j]. We reconstruct it
        # here so the per-slide diagnostic is available without ever
        # materializing the full (N_p, N_p) attention matrix.
        if v_patch_post_pre_v is not None:
            # (B, H, m, N_p). One batched matmul.
            AB = A_inv @ B_soft
            # Collapse m axis against F_row_cls = F_soft[:, :, 0, :].
            P_cls = torch.einsum(
                "bhm,bhmn->bhn", F_soft[:, :, 0, :], AB,
            )                                                    # (B, H, N_p)
            # Real-patch attention weights only.
            P_cls_patch = P_cls[:, :, n_cls : n_cls + N_real]   # (B, H, N_real)
            v_norm_orig = v_patch_pre.detach().norm(dim=-1)     # (B, H, N_real)
            v_norm_new = v_patch_post_pre_v.detach().norm(dim=-1)
            numer = (P_cls_patch * v_norm_new).sum(dim=-1)       # (B, H)
            denom = (P_cls_patch * v_norm_orig).sum(dim=-1)
            bar_rho_cls = (numer / (denom + 1e-8)).detach()
        else:
            bar_rho_cls = None

        return {
            # Stage-2-compatible role-split suite (same keys as stage-2's
            # last_stats, modulo cls vs patch role splitting which the
            # downstream diagnostics accumulator does).
            "cos_yv_cls_pre":              cos_yv_cls_pre,           # (B, H, n_cls)
            "cos_yv_cls_post":             cos_yv_cls_post,          # (B, H, n_cls)
            "cos_yv_patch_pre":            cos_yv_patch_pre,         # (B, H, N_real)
            "cos_yv_patch_post":           cos_yv_patch_post,        # (B, H, N_real)
            "y_norm":                      y_norm,                    # (B, H, L)
            "v_norm":                      v_norm,                    # (B, H, L)
            "z_norm":                      z_norm,                    # (B, H, L)
            "cls_self_attn":               cls_self_attn,             # (B, H)
            "cls_patch_attn":              cls_patch_attn,            # (B, H, N)
            "srp_correction_norm_patch":    srp_correction_norm_patch, # (B, H, N)
            "srp_correction_frac_patch":    srp_correction_frac_patch, # (B, H, N)
            "num_cls_tokens":              n_cls,
            "N_real":                      N_real,
            # SRP-specific (§8.2).
            "cos_yr_patch_pre":            cos_yr_patch_pre,         # (B, H, N)
            "cos_zr_patch_post":           cos_zr_patch_post,        # (B, H, N)
            "h_V_patch":                   h_V_patch,                 # (B, H, N)
            # pre_v-only.
            "cos_vr_patch_pre":            cos_vr_patch_pre,         # (B, H, N) or None
            "cos_vr_patch_post":           cos_vr_patch_post,        # (B, H, N) or None
            "rho_patch":                   rho_patch,                 # (B, H, N) or None
            # Slide-intrinsic gate signal (when applicable).
            "h_morph_patch":               h_morph_patch,             # (B, N) or None
            # §8.2.E placement signature (cheap, always computed).
            "cos_y_cls_rbar":              cos_y_cls_rbar,           # (B, H)
            # §8.2.D3 pre-V attention-weighted retention (pre_v only).
            "bar_rho_cls":                 bar_rho_cls,               # (B, H) or None
            # Raw CLS vectors pre- and post-SRP, used by unit tests to
            # verify the internal invariant z_cls == y_cls within a
            # single forward (proposal §12.7). Detached to avoid
            # holding graph references; downstream StatsAccumulator
            # ignores them (not registered in the accumulator's key
            # roster in srp_diagnostics.py).
            "y_cls_raw":                   y_cls.detach(),            # (B, H, n_cls, D)
            "z_cls_raw":                   z_cls.detach(),            # (B, H, n_cls, D)
        }
