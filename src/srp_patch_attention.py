"""Patch-level Spatial Redundancy Projection attention.

This module is used by the ADP raw-RGB ViT architecture ablation.  Unlike the
slide-level implementation, patch-level ViT inputs have a static grid, so
neighbor indices and masks can be registered once as buffers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from slide_level_srp.data_ext import (
    neighbor_distance_weights,
    neighbor_offsets,
)
from slide_level_srp.src.srp_attention import (
    _GATE_COUNT_FEATURES,
    _gate_num_token_features,
    _make_token_diag,
)
from slide_level_srp.src.rcd_modules import (
    IdentitySafeRCDRecomposer,
    LearnedLocalContextDirection,
)


_BETA_MODES = ("zero", "one", "learn", "fixed")
# Signed-gate mode. When beta_patch_mode == "signed_gated", the per-head
# scalar beta buffer is replaced by per-(token, head) beta_eff values produced
# by a TokenHeadGate.
_BETA_SIGNED_GATED = "signed_gated"
_BETA_RCD = "rcd"
_BETA_RCD_LEARNED_R = "rcd_learned_r"
_BETA_METHOD_MODES = (_BETA_SIGNED_GATED, _BETA_RCD, _BETA_RCD_LEARNED_R)


def build_neighbor_index_for_grid(
    grid_h: int,
    grid_w: int,
    radius: int = 1,
    shell: str = "cumulative",
    source: str = "real",
    shuffle_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a static (N, K) neighbor index + mask for a regular grid_h × grid_w
    grid in row-major order.

    Returns:
      neighbor_index: (N, K) int64, with -1 at invalid slots. N = grid_h * grid_w.
      neighbor_mask:  (N, K) bool,  True at valid slots.

    Offset order matches `slide_level_srp/data_ext.py::build_neighbor_index`:
        (-1,-1) (-1,0) (-1,+1)
        ( 0,-1)        ( 0,+1)
        (+1,-1) (+1,0) (+1,+1)

    Invariants:
      * interior cells have mask.sum(dim=-1) == 8.
      * edge cells have mask.sum(dim=-1) == 5.
      * corner cells have mask.sum(dim=-1) == 3.
      * `neighbor_index[i, k] == -1` iff `neighbor_mask[i, k] == False`.
    """
    offsets = neighbor_offsets(radius=radius, shell=shell)
    N = grid_h * grid_w
    K = len(offsets)
    neighbor_index = torch.full((N, K), -1, dtype=torch.int64)
    neighbor_mask = torch.zeros((N, K), dtype=torch.bool)

    def rc_to_idx(r: int, c: int) -> int:
        return r * grid_w + c

    for r in range(grid_h):
        for c in range(grid_w):
            i = rc_to_idx(r, c)
            for k, (dr, dc) in enumerate(offsets):
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < grid_h and 0 <= c2 < grid_w:
                    neighbor_index[i, k] = rc_to_idx(r2, c2)
                    neighbor_mask[i, k] = True
    if source == "shuffled":
        shuffled = torch.full_like(neighbor_index, -1)
        gen = torch.Generator()
        gen.manual_seed(int(shuffle_seed))
        pool = torch.arange(N, dtype=torch.int64)
        for i in range(N):
            valid_slots = torch.nonzero(neighbor_mask[i], as_tuple=False).flatten()
            count = int(valid_slots.numel())
            if count == 0:
                continue
            candidates = pool[pool != i]
            order = torch.randperm(candidates.numel(), generator=gen)
            if count <= candidates.numel():
                picked = candidates[order[:count]]
            else:
                picked = candidates[order[torch.arange(count) % candidates.numel()]]
            shuffled[i, valid_slots] = picked
        neighbor_index = shuffled
    elif source != "real":
        raise ValueError("source must be 'real' or 'shuffled'")
    return neighbor_index, neighbor_mask


class PatchSRPAttention(nn.Module):
    """
    Full-softmax self-attention + post-aggregation Spatial Redundancy Projection.

    Shares the Q/K/V/proj structure with `src.xsa_attention.XSAAttention`;
    differs in the post-attention step:

        z = y − β · (y · r̂) · r̂          (SRP projection)

    vs the XSA self-direction projection (which uses v̂ instead of r̂).

    r̂ is the unit-normalized mean of per-head value vectors over the token's
    3×3 spatial neighborhood in the 14×14 patch grid. r̂ is computed from
    detached values so only the direct projection term carries gradient
    through v_i.

    CLS is never projected: position 0's output is left exactly as softmax
    attention produced it. Patch rows 1..196 are the only positions updated.

    Configuration:
      beta_patch_mode  "zero" | "one" | "learn" | "fixed"
        "zero" → β = 0 (no projection, identity)
        "one"  → β = 1 (full projection, z ⊥ r̂)
        "learn" → β is a per-head nn.Parameter, init=beta_init
        "fixed" → β = beta_init, registered as a buffer
      beta_init  initial / fixed value (default 2.0 for reflection)

    The module assumes `num_cls_tokens=1` and a grid_h × grid_w = num_patches
    layout. These are invariant for the stage-1 ViT-S/16 setup.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        grid_h: int = 14,
        grid_w: int = 14,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        beta_patch_mode: str = "fixed",
        beta_init: float = 2.0,
        num_cls_tokens: int = 1,
        # Signed-gate options. Consumed only when beta_patch_mode == "signed_gated".
        delta_scale: float = 2.0,
        gate_active: bool = True,
        gate_hidden_dim: int = 16,
        # When False, gate diagnostic inputs (`token_diag`, `head_diag`)
        # are NOT detached before passing to the gate — i.e. gradients
        # flow back through `cos_yr` / `y_norms` into y. This is the
        # "live-input" regime; the default is the detached regime (True).
        detach_gate_inputs: bool = True,
        neighbor_radius: int = 1,
        neighbor_shell: str = "cumulative",
        neighbor_source: str = "real",
        neighbor_shuffle_seed: int = 0,
        neighbor_weighting: str = "uniform",
        neighbor_weight_sigma: float = 1.0,
        gate_output_init: str = "zero",
        gate_output_init_scale: float = 1.0,
        gate_init_beta0: float = 0.0,
        gate_activation: str = "tanh",
        gate_activation_temperature: float = 1.0,
        gate_count_features: str = "legacy",
        srp_gate_placement: str = "post_agg",
        # Optional recomposition controls. Consumed only by beta_patch_mode
        # in {"rcd", "rcd_learned_r"}.
        rcd_adapter_kind: str = "lowrank",
        rcd_rank: int = 16,
        learned_r_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        # Constructor validation uses ValueError so it still runs under
        # `python -O`, where asserts would be stripped.
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        if beta_patch_mode not in _BETA_MODES + _BETA_METHOD_MODES:
            raise ValueError(
                f"beta_patch_mode must be in "
                f"{_BETA_MODES + _BETA_METHOD_MODES}, got {beta_patch_mode!r}"
            )
        if srp_gate_placement not in ("post_agg", "pre_q", "pre_k"):
            raise ValueError(
                "srp_gate_placement must be one of "
                "('post_agg', 'pre_q', 'pre_k'), got "
                f"{srp_gate_placement!r}"
            )
        if srp_gate_placement != "post_agg" and beta_patch_mode != _BETA_SIGNED_GATED:
            raise ValueError(
                "pre-attention SRP placement is currently defined only for "
                "beta_patch_mode='signed_gated'."
            )
        if num_cls_tokens != 1:
            raise ValueError(
                "patch-level ViT uses a single CLS token "
                f"(got num_cls_tokens={num_cls_tokens})"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_cls_tokens = num_cls_tokens
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w
        self.beta_patch_mode = beta_patch_mode
        self.srp_gate_placement = srp_gate_placement
        self.detach_gate_inputs = bool(detach_gate_inputs)
        if gate_count_features not in _GATE_COUNT_FEATURES:
            raise ValueError(
                f"gate_count_features must be one of {_GATE_COUNT_FEATURES}, "
                f"got {gate_count_features!r}"
            )
        self.neighbor_radius = int(neighbor_radius)
        self.neighbor_shell = neighbor_shell
        self.neighbor_source = neighbor_source
        self.neighbor_shuffle_seed = int(neighbor_shuffle_seed)
        self.neighbor_weighting = neighbor_weighting
        self.neighbor_weight_sigma = float(neighbor_weight_sigma)
        self.gate_count_features = gate_count_features

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # β: per-head scalar, shape (num_heads,). Backward-compatible modes
        # match src.xsa_attention.XSAAttention's convention.
        if beta_patch_mode == "learn":
            self.beta_patch = nn.Parameter(
                torch.full((num_heads,), float(beta_init))
            )
        elif beta_patch_mode in _BETA_METHOD_MODES:
            # Replace scalar β with per-(token, head) β_eff from a
            # TokenHeadGate, or with RCD branch maps. Keep a zero-buffer
            # named `beta_patch` for legacy diagnostic code that reads it.
            self.register_buffer(
                "beta_patch", torch.zeros(num_heads), persistent=True,
            )
        else:
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

        # Signed-gate state. Same dead-path rule as PANDA / CAM17:
        # parent ViT passes gate_active=False on the LAST block.
        self.delta_scale = float(delta_scale)
        self.gate_active = bool(
            gate_active and beta_patch_mode == _BETA_SIGNED_GATED
        )
        if self.gate_active:
            from slide_level_srp.src.gate_signed import TokenHeadGate
            rng_state_cpu = torch.get_rng_state()
            try:
                self.gate = TokenHeadGate(
                    num_heads=num_heads,
                    # Same 3 token diagnostics as CAM17:
                    # h_local, neighbour_count/8, log(1+neighbour_count).
                    num_token_features=_gate_num_token_features(
                        self.gate_count_features,
                        include_y_norm_mean=False,
                    ),
                    # Same 3 head diagnostics: cos(y, r̂), |cos|, log_norm_y.
                    num_head_features=3,
                    hidden_dim=gate_hidden_dim,
                    delta_scale=self.delta_scale,
                    output_init=gate_output_init,
                    output_init_scale=gate_output_init_scale,
                    init_beta0=gate_init_beta0,
                    activation=gate_activation,
                    activation_temperature=gate_activation_temperature,
                )
            finally:
                torch.set_rng_state(rng_state_cpu)
        else:
            self.gate = None
        self.rcd_active = bool(
            gate_active and beta_patch_mode in (_BETA_RCD, _BETA_RCD_LEARNED_R)
        )
        self.learned_r_active = bool(
            self.rcd_active and beta_patch_mode == _BETA_RCD_LEARNED_R
        )
        if self.rcd_active:
            rng_state_cpu = torch.get_rng_state()
            try:
                self.rcd_recomposer = IdentitySafeRCDRecomposer(
                    head_dim=self.head_dim,
                    rank=rcd_rank,
                    adapter_kind=rcd_adapter_kind,
                )
                self.context_scorer = (
                    LearnedLocalContextDirection(
                        head_dim=self.head_dim,
                        hidden_dim=learned_r_hidden_dim,
                        use_h_local=True,
                    )
                    if self.learned_r_active
                    else None
                )
            finally:
                torch.set_rng_state(rng_state_cpu)
        else:
            self.rcd_recomposer = None
            self.context_scorer = None
        self._last_gate_stats: dict | None = None
        self._last_rcd_stats: dict | None = None

        # Static neighbor index over the grid_h × grid_w grid. Buffers
        # move with .to(device) and survive state_dict save/load.
        nbi, nbm = build_neighbor_index_for_grid(
            grid_h, grid_w,
            radius=self.neighbor_radius,
            shell=self.neighbor_shell,
            source=self.neighbor_source,
            shuffle_seed=self.neighbor_shuffle_seed,
        )
        weights_np = neighbor_distance_weights(
            neighbor_offsets(radius=self.neighbor_radius, shell=self.neighbor_shell),
            weighting=self.neighbor_weighting,
            sigma=self.neighbor_weight_sigma,
        )
        weights = torch.from_numpy(weights_np).to(torch.float32)
        weights = weights.unsqueeze(0).expand(self.num_patches, -1).clone()
        weights = weights * nbm.to(weights.dtype)
        self.register_buffer("neighbor_index", nbi, persistent=False)
        self.register_buffer("neighbor_mask", nbm, persistent=False)
        self.register_buffer("neighbor_weight", weights, persistent=False)

        # Diagnostics.
        self._capture_stats = False
        self.last_stats: dict | None = None

    # --- SRP-specific helpers (static, buffer-based) -----------------------

    def _gather_neighborhood_values(self, v_patch: torch.Tensor) -> torch.Tensor:
        """
        Gather per-patch 3×3 neighbor value vectors.

        v_patch:  (B, H, N_p, D)  — detached values at patch rows.
        Returns:  (B, H, N_p, 8, D) — neighbor values stacked on the 8 axis.
                  Invalid slots (mask=False) are physically populated from
                  index-clamped positions and zeroed via the mask multiplication
                  in `_neighborhood_mean`. Clamp-then-mask is required because
                  PyTorch's advanced indexing treats negative ints as Python-
                  style wrap-around (index -1 → last row), which is silently
                  wrong for our -1-marked invalid slots.
        """
        B, H, N_p, D = v_patch.shape
        # Runtime shape contract.
        if N_p != self.num_patches:
            raise ValueError(
                f"expected {self.num_patches} patch rows, got {N_p}"
            )
        # Clamp -1 entries to 0 so the gather reads a valid (but semantically
        # masked-out) row. The mask handling in _neighborhood_mean zeros these.
        K = self.neighbor_index.shape[-1]
        safe_idx = self.neighbor_index.clamp(min=0)                # (N_p, K)
        # Advanced indexing: v_patch is (B, H, N_p, D). We index along the
        # N_p axis (axis=2). Result preserves the H axis.
        # Expand safe_idx to (B, N_p, 8); batch_idx is (B, N_p, 8).
        batch_idx = torch.arange(B, device=v_patch.device).view(B, 1, 1).expand(B, N_p, K)
        safe_idx_b = safe_idx.unsqueeze(0).expand(B, -1, -1)        # (B, N_p, K)
        # Result shape: (B, N_p, K, H, D). Permute to (B, H, N_p, K, D).
        neighbor_v = v_patch[batch_idx, :, safe_idx_b, :]           # (B, N_p, K, H, D)
        neighbor_v = neighbor_v.permute(0, 3, 1, 2, 4)              # (B, H, N_p, K, D)
        return neighbor_v

    def _neighborhood_mean(self, neighbor_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Masked mean of (B, H, N_p, K, D) neighbor values.
        Returns (r, r_hat, cnt):
          r       (B, H, N_p, D) raw mean (zero where |N(j)|=0; doesn't happen
                  on a dense 14×14 grid — every patch has ≥ 3 neighbors).
          r_hat   (B, H, N_p, D) unit-normalized r (eps-guarded).
          cnt     (B, 1, N_p, 1) valid-neighbor count (float) — diagnostic.
        """
        mask_f = self.neighbor_mask.to(neighbor_v.dtype)            # (N_p, 8)
        mask_f = mask_f.unsqueeze(0).unsqueeze(0).unsqueeze(-1)     # (1, 1, N_p, 8, 1)
        weight_f = self.neighbor_weight.to(neighbor_v.dtype)
        weight_f = weight_f.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        weighted_mask = mask_f * weight_f
        s = (neighbor_v * weighted_mask).sum(dim=3)                  # (B, H, N_p, D)
        cnt = mask_f.sum(dim=3)                                      # (1, 1, N_p, 1)
        denom = weighted_mask.sum(dim=3).clamp(min=1.0)
        r = s / denom                                                # (B, H, N_p, D)
        r_hat = F.normalize(r, dim=-1, eps=1e-12)
        return r, r_hat, cnt

    def _apply_signed_gate_to_patch_stream(
        self,
        stream_patch: torch.Tensor,
        *,
        h_local: torch.Tensor | None,
        stream_label: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply the signed SRP formula to one pre/post-attention patch stream.

        `stream_patch` is either Y_patch for the existing post-aggregation
        design, Q_patch for pre-Q, or K_patch for pre-K.
        The local direction is always
        computed from the same stream family, using detached neighbor rows:

            r_hat_i = normalize(mean_{j in N(i)} stopgrad(stream_j))

        This keeps the pre-attention arms faithful to the paper-derived
        design: edit one stream before the attention product while the other
        Q/K/V streams stay untouched.
        """
        B, H, N_p, _D = stream_patch.shape
        if N_p != self.num_patches:
            raise ValueError(
                f"expected {self.num_patches} patch rows, got {N_p}"
            )
        neighbor_det = self._gather_neighborhood_values(stream_patch.detach())
        _r, r_hat, cnt = self._neighborhood_mean(neighbor_det)
        dot_sr = (stream_patch * r_hat).sum(dim=-1, keepdim=True)

        if not self.gate_active:
            beta_eff = torch.zeros_like(dot_sr)
            return stream_patch, r_hat, cnt, dot_sr, beta_eff

        # h_local is computed once from layer-0 patch tokens and shared across
        # blocks; placement changes only which attention stream supplies
        # r_hat/head diagnostics.
        if h_local is None:
            raise ValueError(
                "PatchSRPAttention with beta_patch_mode='signed_gated' "
                f"and srp_gate_placement='{self.srp_gate_placement}' "
                "requires h_local in forward()."
            )
        if h_local.shape != (B, self.num_patches):
            raise ValueError(
                f"h_local shape mismatch: got {tuple(h_local.shape)}, "
                f"expected ({B}, {self.num_patches})"
            )

        cnt_bn = cnt.squeeze(-1).squeeze(1).to(stream_patch.dtype)
        cnt_bn = cnt_bn.expand(B, -1)
        token_diag = _make_token_diag(
            h_local=h_local.to(stream_patch.dtype),
            cnt_bn=cnt_bn,
            max_neighbors=self.neighbor_index.shape[-1],
            mode=self.gate_count_features,
        )

        eps = 1e-12
        stream_norms = stream_patch.norm(dim=-1)
        cos_sr = dot_sr.squeeze(-1) / (stream_norms + eps)
        head_diag = torch.stack(
            [cos_sr, cos_sr.abs(), torch.log1p(stream_norms)],
            dim=-1,
        )
        if self.detach_gate_inputs:
            token_diag = token_diag.detach()
            head_diag = head_diag.detach()

        beta_eff = self.gate(token_diag, head_diag)
        with torch.no_grad():
            # Keep canonical key names (`cos_yr`, `y_norms`) so existing
            # diagnostics and GateStatsAccumulator remain backward-compatible.
            # Under pre-Q they should be read as cos(stream, r_hat) and
            # stream norm; `gate_stream_id` records 0=post-Y, 1=pre-Q,
            # 2=pre-K.  Keeping one diagnostics key avoids collector churn
            # while making placement explicit in saved stats.
            gate_stream_id = {"q": 1, "k": 2}.get(stream_label, 0)
            self._last_gate_stats = {
                "beta_eff": beta_eff.detach(),
                "cos_yr": cos_sr.detach(),
                "y_norms": stream_norms.detach(),
                "h_local": h_local.detach(),
                "neighbour_count": cnt_bn.detach(),
                "gate_stream_id": torch.full(
                    (B,),
                    gate_stream_id,
                    dtype=torch.int64,
                    device=stream_patch.device,
                ),
            }
        projected = stream_patch - beta_eff * dot_sr * r_hat
        return projected, r_hat, cnt, dot_sr, beta_eff

    # --- forward ----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        h_local: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, 1+num_patches, C). Returns (B, 1+num_patches, C).
        h_local (optional): (B, num_patches) per-patch homogeneity,
            required only when beta_patch_mode='signed_gated' and the
            gate is active. Computed once at the parent ViT's
            forward_features and passed unchanged through every block.
        """
        # Reset gate-stats cache on every forward.
        self._last_gate_stats = None
        self._last_rcd_stats = None
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        # Runtime shape contract.
        if N != 1 + self.num_patches:
            raise ValueError(
                f"expected {1 + self.num_patches} tokens, got {N}"
            )

        # Q/K/V (identical to XSAAttention).
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                      # each (B, H, N, D)

        if (
            self.beta_patch_mode == _BETA_SIGNED_GATED
            and self.srp_gate_placement in ("pre_q", "pre_k")
        ):
            # Pre-attention arms edit one patch stream after W_Q/W_K
            # and before QK.  CLS rows are preserved.  Pre-Q still has no
            # final-block CLS-readout path, but pre-K does because the CLS
            # query attends to edited patch keys in the same block.
            if self.srp_gate_placement == "pre_q":
                q_patch, _r_hat_q, _cnt_q, _dot_qr, _beta_q = (
                    self._apply_signed_gate_to_patch_stream(
                        q[:, :, 1:, :], h_local=h_local, stream_label="q",
                    )
                )
                q = q.clone()
                q[:, :, 1:, :] = q_patch
            else:
                k_patch, _r_hat_k, _cnt_k, _dot_kr, _beta_k = (
                    self._apply_signed_gate_to_patch_stream(
                        k[:, :, 1:, :], h_local=h_local, stream_label="k",
                    )
                )
                k = k.clone()
                k[:, :, 1:, :] = k_patch

        # Standard softmax attention (explicit softmax so we could expose
        # attn_probs if needed — we don't currently capture them here,
        # but keeping the path symmetric with XSAAttention).
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale         # (B, H, N, N)
        attn_probs = attn_logits.softmax(dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        y = attn_probs @ v                                           # (B, H, N, D)

        # --- SRP projection on patch rows only -------------------------
        # Slice patch slice; CLS (pos 0) is never written.
        v_patch_pre = v[:, :, 1:, :]                                 # (B, H, N_p, D)
        y_patch = y[:, :, 1:, :]                                     # (B, H, N_p, D)

        # r̂ from detached neighbor values. Detach blocks within-step
        # gradient flow through v_k via r_j; the direct v_j dot product
        # still carries gradient, matching the slide-level implementation
        # and the slide-level implementation.
        neighbor_v_det = self._gather_neighborhood_values(v_patch_pre.detach())
        r, r_hat, cnt = self._neighborhood_mean(neighbor_v_det)

        dot_yr = (y_patch * r_hat).sum(dim=-1, keepdim=True)         # (B, H, N_p, 1)

        if self.beta_patch_mode in (_BETA_RCD, _BETA_RCD_LEARNED_R):
            if not self.rcd_active:
                # Last block dead-path: patch writes after the final
                # attention layer do not feed the CLS classifier.
                z_patch = y_patch
            else:
                if self.learned_r_active:
                    if h_local is None:
                        raise ValueError(
                            "PatchSRPAttention beta_patch_mode='rcd_learned_r' "
                            "requires h_local in forward()."
                        )
                    if h_local.shape != (B, self.num_patches):
                        raise ValueError(
                            f"h_local shape mismatch: got {tuple(h_local.shape)}, "
                            f"expected ({B}, {self.num_patches})"
                        )
                    nbm = self.neighbor_mask.unsqueeze(0).expand(B, -1, -1)
                    nbw = self.neighbor_weight.unsqueeze(0).expand(B, -1, -1)
                    _r, r_hat, cnt, learned_r_weight = self.context_scorer(
                        center_v=v_patch_pre.detach(),
                        neighbor_v=neighbor_v_det,
                        neighbor_mask=nbm,
                        neighbor_weight=nbw,
                        h_local=h_local,
                    )
                    with torch.no_grad():
                        self._last_rcd_stats = {
                            "learned_r_weight": learned_r_weight.detach(),
                        }
                z_patch, rcd_stats = self.rcd_recomposer(y_patch, r_hat)
                with torch.no_grad():
                    merged = {} if self._last_rcd_stats is None else dict(self._last_rcd_stats)
                    merged.update(rcd_stats)
                    self._last_rcd_stats = merged
        elif self.beta_patch_mode == _BETA_SIGNED_GATED:
            # Per-(token, head) β_eff from a learned gate.
            if self.srp_gate_placement in ("pre_q", "pre_k"):
                # The gate has already been applied before QK.  Do not
                # apply a second post-attention projection; this isolates
                    # placement as the only change from the default
                    # post-attention path.
                z_patch = y_patch
            else:
                if self.gate_active:
                    # Runtime input-contract checks use ValueError so they
                    # survive `python -O`.
                    if h_local is None:
                        raise ValueError(
                            "PatchSRPAttention with beta_patch_mode='signed_gated' "
                            "and gate_active=True requires h_local in forward()."
                        )
                    if h_local.shape != (B, self.num_patches):
                        raise ValueError(
                            f"h_local shape mismatch: got {tuple(h_local.shape)}, "
                            f"expected ({B}, {self.num_patches})"
                        )
                    # cnt is (1, 1, N_p, 1); broadcast to (B, N_p) for the
                    # per-token diagnostic stack.
                    cnt_bn = cnt.squeeze(-1).squeeze(1).to(y.dtype)         # (1, N_p)
                    cnt_bn = cnt_bn.expand(B, -1)                          # (B, N_p)
                    token_diag = _make_token_diag(
                        h_local=h_local.to(y.dtype),
                        cnt_bn=cnt_bn,
                        max_neighbors=self.neighbor_index.shape[-1],
                        mode=self.gate_count_features,
                    )
                    eps = 1e-12
                    y_norms = y_patch.norm(dim=-1)                         # (B, H, N_p)
                    cos_yr = dot_yr.squeeze(-1) / (y_norms + eps)          # (B, H, N_p)
                    head_diag = torch.stack(
                        [cos_yr, cos_yr.abs(), torch.log1p(y_norms)],
                        dim=-1,
                    )                                                      # (B, H, N_p, 3)
                    # The default detach convention: gate diagnostic
                    # inputs are stop-grad'd by default. Setting
                    # detach_gate_inputs=False enables the "live" regime
                    # (gradient flows back through cos_yr / y_norms into y),
                    # which is the empirically better-performing regime on
                    # ADP in this setting (+0.98 pp paired). token_diag has no
                        # y dependence so its detach is a no-op in ADP
                    # code, but is included for parity with PANDA where
                    # token_diag DOES depend on y (log_norm_y_mean column).
                    if self.detach_gate_inputs:
                        token_diag = token_diag.detach()
                        head_diag = head_diag.detach()
                    beta_eff = self.gate(token_diag, head_diag)            # (B, H, N_p, 1)
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
                z_patch = y_patch - beta_eff * dot_yr * r_hat
        else:
            # Standard (legacy) path: scalar per-head β.
            beta = self.beta_patch.to(dtype=y.dtype, device=y.device).view(1, H, 1, 1)
            z_patch = y_patch - beta * dot_yr * r_hat                # (B, H, N_p, D)

        # Clone-and-update: start from y (CLS at pos 0 intact), overwrite
        # only the patch-row slice. This preserves the "CLS untouched"
        # invariant.
        z = y.clone()
        z[:, :, 1:, :] = z_patch

        # --- Diagnostics --------------------------------------------------
        if self._capture_stats:
            # Minimal SRP diagnostic set. Skip the heavy (B, H, N, N)
            # attn_probs capture from XSAAttention since we don't use it
            # here; the downstream diagnostics machinery inherited from
            # the original XSA code path will tolerate its absence.
            cos_yr_patch_pre = F.cosine_similarity(y_patch, r_hat, dim=-1).detach()
            cos_zr_patch_post = F.cosine_similarity(z_patch, r_hat, dim=-1).detach()
            # Stage-1 stats-accumulator in src/diagnostics.py expects these
            # stage-1 keys too:
            cos_yv_pre = F.cosine_similarity(y, v, dim=-1).detach()
            cos_zv_post = F.cosine_similarity(z, v, dim=-1).detach()
            y_norm = y.norm(dim=-1).detach()
            v_norm = v.norm(dim=-1).detach()
            z_norm = z.norm(dim=-1).detach()
            self.last_stats = {
                # Stage-1 compatible keys so src/diagnostics.py extract_batch_stats
                # can consume this output unchanged.
                "attn_probs":   attn_probs.detach(),
                "cos_yv_pre":   cos_yv_pre,
                "cos_zv_post":  cos_zv_post,
                "y_norm":       y_norm,
                "v_norm":       v_norm,
                "z_norm":       z_norm,
                "num_cls_tokens": self.num_cls_tokens,
                # SRP-specific additions (ignored by stage-1 accumulator,
                # but available on the npz if consumers want them).
                "cos_yr_patch_pre":  cos_yr_patch_pre,
                "cos_zr_patch_post": cos_zr_patch_post,
            }

        z = z.transpose(1, 2).reshape(B, N, C)
        out = self.proj(z)
        out = self.proj_drop(out)
        return out
