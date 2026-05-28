"""
NystromSRPAggregator: TransMIL-style MIL aggregator threaded with SRP.

Differences vs. stage-2 NystromXSAggregator:

  (1) Each TransLayer's forward accepts (x, neighbor_index, neighbor_mask,
      is_real, h_morph) instead of just (x). All four tensors pass through
      every block identically; h_morph is unused by non-gated ablations
      but carried along to keep the forward signature uniform.

  (2) The square-pad step builds is_real on the fly. Per the released protocol,
      is_real is True at positions 1..N (real patches) and False at
      position 0 (CLS) and positions 1+N..H*W (pad duplicates).
      neighbor_index / neighbor_mask stay sized to real patches only
      (shape (B, N, 8)); the attention forward slices the padded sequence
      down to the real-patch region and applies SRP there.

  (3) PPEG is UNCHANGED. It operates on the already-reshaped 2D patch
      grid (CLS + H*W patch rows) and has no visibility into SRP's
      neighbor tensors. This is intentional: PPEG is a TransMIL-faithful
      2D-conv position mixer, not a redundancy-aware operation.

All other stage-2 design choices (drop-path linear schedule, 4 TransLayers
with PPEG inserted after block 1, LayerNorm head on CLS, etc.) are
preserved verbatim — see the slide-level baseline implementation

Activation checkpointing (by design): identical semantics to stage 2. The
checkpointed callable now closes over the extra tensors automatically
because cp.checkpoint wraps a function not a (tensor,) tuple.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

# Import unchanged stage-2 pieces directly; this keeps stage 2 as the
# single source of truth and prevents drift. See the released protocol's
# "imports vs. copies" convention.
from slide_level.src.ppeg import PPEG
from slide_level.src.aggregator import DropPath, Mlp
from src.role_split_norm import RoleSplitLayerNorm

from .rcd_modules import collect_rcd_module_ids, reset_rcd_identity_modules
from .srp_attention import (
    NystromSRPAttention,
    collect_mlp_control_module_ids,
    reset_mlp_control_modules,
)


_CHECKPOINT_MODES = ("whole_block", "per_module", "off")
_LN_SPECIALIZATIONS = ("shared", "cls_patch")
_LN_SPECIALIZATION_SCOPES = ("block", "block_final")
_FINAL_BLOCK_CLS_PATH_SRP_MODES = {
    "pre_k_signed_gated",
    "pre_v_signed_gated",
}
_H_LOCAL_REQUIRED_SRP_MODES = {
    "post_agg_signed_gated",
    "post_agg_signed_gated_learned_r",
    "pre_q_signed_gated",
    "pre_k_signed_gated",
    "pre_v_signed_gated",
    "post_agg_rcd_learned_r",
}


def _gate_active_for_block(srp_mode: str, block_idx: int, depth: int) -> bool:
    """Return the placement-aware signed-gate activity flag.

    Post-attention and patch-query-only gates stay disabled in the final
    block because their patch-row edits cannot reach a CLS-only head in the
    same block.  Pre-K and pre-V are different: CLS attention consumes patch
    keys/values before the block emits its CLS row, so those placements have
    a valid final-block path to the classifier.
    """
    return block_idx < depth - 1 or srp_mode in _FINAL_BLOCK_CLS_PATH_SRP_MODES


def _make_layer_norm(
    dim: int,
    *,
    ln_specialization: str,
    num_cls_tokens: int,
) -> nn.Module:
    """Build the norm module without changing default checkpoint keys.

    The historical path must keep plain `nn.LayerNorm` modules named
    `norm1`, `norm2`, and `norm`.  Active queues and old checkpoints rely on
    those exact keys.  We instantiate `RoleSplitLayerNorm` only when the new
    experiment explicitly asks for CLS/patch specialization.
    """
    if ln_specialization == "shared":
        return nn.LayerNorm(dim)
    if ln_specialization == "cls_patch":
        return RoleSplitLayerNorm(
            dim,
            mode="cls_patch",
            num_cls_tokens=num_cls_tokens,
        )
    raise ValueError(f"unknown ln_specialization: {ln_specialization!r}")


class SRPBlock(nn.Module):
    """
    Pre-norm TransLayer wrapping NystromSRPAttention.

    Named `SRPBlock` (not `Block`) to keep the stage-2 diagnostic
    class-name discovery machinery unambiguous — the SRP-specific
    diagnostics module discovers SRPBlock by name separately from
    stage 2's Block (see slide_level_srp/src/srp_diagnostics.py).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_landmarks: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        beta_patch_mode: str = "learn",
        beta_init: float = 1.0,
        srp_mode: str = "post_agg",
        num_cls_tokens: int = 1,
        pinv_iterations: int = 6,
        drop_path: float = 0.0,
        checkpoint_mode: str = "whole_block",
        layerscale_init: float = 0.0,
        ln_specialization: str = "shared",
        # Signed-gate options forwarded to NystromSRPAttention. The
        # parent aggregator passes a placement-aware gate_active flag:
        # final-block post-attention/pre-Q gates are dead under CLS-only
        # readout, while final-block pre-K/pre-V gates can affect CLS.
        delta_scale: float = 2.0,
        gate_active: bool = True,
        gate_hidden_dim: int = 16,
        detach_gate_inputs: bool = True,
        gate_output_init: str = "zero",
        gate_output_init_scale: float = 1.0,
        gate_init_beta0: float = 0.0,
        gate_activation: str = "tanh",
        gate_activation_temperature: float = 1.0,
        gate_factorization: str = "full",
        gate_count_features: str = "legacy",
        # Method 2.1/2.4 RCD options.  These are forwarded only to the
        # explicit RCD SRP modes, so legacy SRP and signed-gate runs keep
        # their current parameter surface and behavior.
        rcd_adapter_kind: str = "lowrank",
        rcd_rank: int = 16,
        learned_r_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        assert checkpoint_mode in _CHECKPOINT_MODES
        if layerscale_init < 0.0:
            raise ValueError(f"layerscale_init must be non-negative, got {layerscale_init}")
        if ln_specialization not in _LN_SPECIALIZATIONS:
            raise ValueError(
                f"ln_specialization must be one of {_LN_SPECIALIZATIONS}, "
                f"got {ln_specialization!r}"
            )
        self.checkpoint_mode = checkpoint_mode
        self.ln_specialization = ln_specialization

        self.norm1 = _make_layer_norm(
            dim,
            ln_specialization=ln_specialization,
            num_cls_tokens=num_cls_tokens,
        )
        self.attn = NystromSRPAttention(
            dim=dim, num_heads=num_heads, num_landmarks=num_landmarks,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
            beta_patch_mode=beta_patch_mode,
            beta_init=beta_init,
            srp_mode=srp_mode,
            num_cls_tokens=num_cls_tokens,
            pinv_iterations=pinv_iterations,
            delta_scale=delta_scale,
            gate_active=gate_active,
            gate_hidden_dim=gate_hidden_dim,
            detach_gate_inputs=detach_gate_inputs,
            gate_output_init=gate_output_init,
            gate_output_init_scale=gate_output_init_scale,
            gate_init_beta0=gate_init_beta0,
            gate_activation=gate_activation,
            gate_activation_temperature=gate_activation_temperature,
            gate_factorization=gate_factorization,
            gate_count_features=gate_count_features,
            rcd_adapter_kind=rcd_adapter_kind,
            rcd_rank=rcd_rank,
            learned_r_hidden_dim=learned_r_hidden_dim,
        )
        self.norm2 = _make_layer_norm(
            dim,
            ln_specialization=ln_specialization,
            num_cls_tokens=num_cls_tokens,
        )
        self.mlp = Mlp(in_dim=dim, hidden_dim=int(dim * mlp_ratio), drop=proj_drop)
        self.drop_path = DropPath(drop_path)
        self.layerscale_init = float(layerscale_init)
        if self.layerscale_init > 0.0:
            # CaiT LayerScale is a learnable diagonal residual gain. Store it
            # as a vector and rely on broadcasting instead of materializing
            # diag(gamma), which preserves the intended per-channel design
            # without adding unnecessary memory or compute.
            self.gamma_attn = nn.Parameter(torch.full((dim,), self.layerscale_init))
            self.gamma_mlp = nn.Parameter(torch.full((dim,), self.layerscale_init))
        else:
            # Do not register dummy parameters when disabled. This keeps old
            # checkpoints, optimizer groups, trainability counts, and active
            # queue jobs on the exact pre-LayerScale parameter surface.
            self.gamma_attn = None
            self.gamma_mlp = None

        # CLS trajectory capture — same interface as stage-2 Block.
        self._capture_cls_pipeline = False
        self.last_cls_states: dict | None = None

    # --- submodule branches used by per-module checkpointing ---

    def _apply_layerscale(
        self,
        branch: torch.Tensor,
        gamma: Optional[nn.Parameter],
    ) -> torch.Tensor:
        if gamma is None:
            return branch
        # Branch tensors are (B, L, D); LayerScale's diagonal matrix is
        # equivalent to a channel-wise multiply shared across batch and token
        # axes. Applying it before DropPath follows CaiT's residual-branch
        # formulation: x + DropPath(gamma * F(LN(x))).
        return branch * gamma.view(1, 1, -1)

    def _attn_branch(
        self, x, neighbor_index, neighbor_mask, is_real, h_morph, h_local,
        neighbor_weight,
    ):
        return self.attn(
            self.norm1(x), neighbor_index, neighbor_mask, is_real,
            h_morph, h_local, neighbor_weight,
        )

    def _mlp_branch(self, x):
        return self.mlp(self.norm2(x))

    def _forward_inner(
        self, x, neighbor_index, neighbor_mask, is_real, h_morph, h_local,
        neighbor_weight,
    ):
        attn_out = self._attn_branch(
            x, neighbor_index, neighbor_mask, is_real, h_morph, h_local,
            neighbor_weight,
        )
        attn_out = self._apply_layerscale(attn_out, self.gamma_attn)
        x_after_attn = x + self.drop_path(attn_out)
        mlp_out = self._mlp_branch(x_after_attn)
        mlp_out = self._apply_layerscale(mlp_out, self.gamma_mlp)
        x_out = x_after_attn + self.drop_path(mlp_out)

        if self._capture_cls_pipeline:
            self.last_cls_states = {
                "cls_before_attn":  x[:, 0].detach(),
                "cls_after_attn":   x_after_attn[:, 0].detach(),
                "cls_after_block":  x_out[:, 0].detach(),
            }
        return x_out

    def forward(
        self,
        x: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
        is_real: torch.Tensor,
        h_morph: Optional[torch.Tensor],
        h_local: Optional[torch.Tensor] = None,
        neighbor_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mode = self.checkpoint_mode
        if mode == "off" or not self.training:
            return self._forward_inner(
                x, neighbor_index, neighbor_mask, is_real, h_morph, h_local,
                neighbor_weight,
            )

        if mode == "whole_block":
            return cp.checkpoint(
                self._forward_inner,
                x, neighbor_index, neighbor_mask, is_real, h_morph, h_local,
                neighbor_weight,
                use_reentrant=False,
            )

        # per_module: two checkpoint boundaries (attention + MLP).
        attn_out = cp.checkpoint(
            self._attn_branch,
            x, neighbor_index, neighbor_mask, is_real, h_morph, h_local,
            neighbor_weight,
            use_reentrant=False,
        )
        attn_out = self._apply_layerscale(attn_out, self.gamma_attn)
        x_after_attn = x + self.drop_path(attn_out)
        mlp_out = cp.checkpoint(self._mlp_branch, x_after_attn, use_reentrant=False)
        mlp_out = self._apply_layerscale(mlp_out, self.gamma_mlp)
        x_out = x_after_attn + self.drop_path(mlp_out)
        if self._capture_cls_pipeline:
            self.last_cls_states = {
                "cls_before_attn":  x[:, 0].detach(),
                "cls_after_attn":   x_after_attn[:, 0].detach(),
                "cls_after_block":  x_out[:, 0].detach(),
            }
        return x_out


class NystromSRPAggregator(nn.Module):
    """
    Slide-level aggregator with Spatial Redundancy Projection.

    Same overall architecture as stage 2:
      - Linear(1024 -> embed_dim) input projection on raw UNI features
      - square-pad by feature self-replication to reach H*W patch slots
      - prepend learned CLS -> sequence length 1 + H*W
      - block 0 (SRP) -> PPEG(H, W) -> blocks 1..depth-1 (SRP)
      - LayerNorm -> read CLS -> Linear(embed_dim -> num_classes)

    The forward now takes `neighbor_index`, `neighbor_mask`, and optional
    `h_morph` in addition to `features`. The aggregator constructs
    `is_real` internally during the square-pad step, by design

    Beta / SRP parameters (all forwarded to every block):
      beta_patch_mode: "zero" | "one" | "learn"
      beta_init:       initial value under "learn"
      srp_mode:        "post_agg" | "pre_v" | "post_agg_gated"
                       plus signed-gated post/pre-Q/pre-K/pre-V modes
    """
    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        num_landmarks: int = 64,
        num_classes: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        beta_patch_mode: str = "learn",
        beta_init: float = 1.0,
        srp_mode: str = "post_agg",
        drop_path_rate: float = 0.1,
        pinv_iterations: int = 6,
        checkpoint_mode: str = "whole_block",
        layerscale_init: float = 0.0,
        ln_specialization: str = "shared",
        ln_specialization_scope: str = "block",
        # Signed-gate options (by design). Only consumed under
        # srp_mode == "post_agg_signed_gated" + beta_patch_mode == "signed_gated".
        delta_scale: float = 2.0,
        gate_hidden_dim: int = 16,
        detach_gate_inputs: bool = True,
        gate_output_init: str = "zero",
        gate_output_init_scale: float = 1.0,
        gate_init_beta0: float = 0.0,
        gate_activation: str = "tanh",
        gate_activation_temperature: float = 1.0,
        gate_factorization: str = "full",
        gate_count_features: str = "legacy",
        # Method 2.1/2.4 RCD controls.  They are inert unless `srp_mode`
        # is one of the new RCD modes.
        rcd_adapter_kind: str = "lowrank",
        rcd_rank: int = 16,
        learned_r_hidden_dim: int = 16,
        # validation PPEG-removal ablation (the reported ablation). Removes the
        # PPEG conv-position-mixing layer between blocks 0 and 1 to test
        # the architectural-mediation hypothesis: does the δ=2 reflection
        # advantage on TransMIL require PPEG, or does it survive without?
        # When False, self.ppeg becomes nn.Identity and PPEG's reshape
        # path is skipped — see forward().
        use_ppeg: bool = True,
    ) -> None:
        super().__init__()
        assert checkpoint_mode in _CHECKPOINT_MODES
        if layerscale_init < 0.0:
            raise ValueError(f"layerscale_init must be non-negative, got {layerscale_init}")
        if ln_specialization not in _LN_SPECIALIZATIONS:
            raise ValueError(
                f"ln_specialization must be one of {_LN_SPECIALIZATIONS}, "
                f"got {ln_specialization!r}"
            )
        if ln_specialization_scope not in _LN_SPECIALIZATION_SCOPES:
            raise ValueError(
                "ln_specialization_scope must be one of "
                f"{_LN_SPECIALIZATION_SCOPES}, got {ln_specialization_scope!r}"
            )
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.num_cls_tokens = 1
        self.checkpoint_mode = checkpoint_mode
        self.srp_mode = srp_mode
        self.use_ppeg = use_ppeg
        self.layerscale_init = float(layerscale_init)
        self.ln_specialization = ln_specialization
        self.ln_specialization_scope = ln_specialization_scope

        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist() if depth > 0 else []

        # Placement-aware final-block rule.  The historical post-attention
        # signed gate remains disabled in the last block because patch-row
        # writes are not consumed by a CLS-only head.  Pre-K/pre-V gates are
        # allowed in the final block because they alter patch keys/values
        # before the final CLS attention row is formed.
        self.blocks = nn.ModuleList([
            SRPBlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_landmarks=num_landmarks,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                beta_patch_mode=beta_patch_mode,
                beta_init=beta_init,
                srp_mode=srp_mode,
                num_cls_tokens=self.num_cls_tokens,
                pinv_iterations=pinv_iterations,
                drop_path=dpr[i],
                checkpoint_mode=checkpoint_mode,
                layerscale_init=self.layerscale_init,
                ln_specialization=self.ln_specialization,
                delta_scale=delta_scale,
                gate_active=_gate_active_for_block(srp_mode, i, depth),
                gate_hidden_dim=gate_hidden_dim,
                detach_gate_inputs=detach_gate_inputs,
                gate_output_init=gate_output_init,
                gate_output_init_scale=gate_output_init_scale,
                gate_init_beta0=gate_init_beta0,
                gate_activation=gate_activation,
                gate_activation_temperature=gate_activation_temperature,
                gate_factorization=gate_factorization,
                gate_count_features=gate_count_features,
                rcd_adapter_kind=rcd_adapter_kind,
                rcd_rank=rcd_rank,
                learned_r_hidden_dim=learned_r_hidden_dim,
            )
            for i in range(depth)
        ])

        # PPEG: enabled by default (TransMIL-faithful). Replaced with
        # nn.Identity under the released protocol ablation. See forward() for the
        # corresponding reshape skip.
        self.ppeg = PPEG(dim=embed_dim) if self.use_ppeg else nn.Identity()
        final_ln_specialization = (
            self.ln_specialization
            if self.ln_specialization_scope == "block_final"
            else "shared"
        )
        self.norm = _make_layer_norm(
            embed_dim,
            ln_specialization=final_ln_specialization,
            num_cls_tokens=self.num_cls_tokens,
        )
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        from slide_level_srp.src.gate_signed import TokenHeadGate, collect_gate_module_ids
        # Skip gate-internal modules during the trunc_normal pass so the
        # gate's nn.Linear init does NOT shift downstream non-gate
        # weight draws under the same seed. See
        # See slide_level_srp.src.gate_signed.collect_gate_module_ids for
        # context.
        gate_module_ids = collect_gate_module_ids(self)
        rcd_module_ids = collect_rcd_module_ids(self)
        mlp_control_module_ids = collect_mlp_control_module_ids(self)
        method_module_ids = gate_module_ids | rcd_module_ids | mlp_control_module_ids
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if id(m) in method_module_ids:
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # Depthwise PPEG convs.
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Belt-and-suspenders gate output-path init (idempotent given
        # the skip-set above). Uses the configured ablation init rather
        # than hard-resetting to zero.
        for m in self.modules():
            if isinstance(m, TokenHeadGate):
                m.reset_output_path()
        # Same belt-and-suspenders reset for RCD: the global initializer
        # intentionally skips RCD internals, but this keeps the no-op
        # start exact even if construction order changes later.
        reset_rcd_identity_modules(self.modules())
        # Keep the plain adapter control at exact identity after the parent
        # init pass for the same reason: it is a method branch, not part of the
        # shared TransMIL baseline initialization stream.
        reset_mlp_control_modules(self.modules())

    def forward(
        self,
        features: torch.Tensor,            # (B, N, in_dim)
        neighbor_index: torch.Tensor,      # (B, N, 8) long, -1 for invalid slots
        neighbor_mask: torch.Tensor,       # (B, N, 8) bool
        h_morph: Optional[torch.Tensor] = None,   # (B, N) float (ignored unless srp_mode=gated)
        h_local: Optional[torch.Tensor] = None,   # (B, N) float (required for post_agg_signed_gated)
        neighbor_weight: Optional[torch.Tensor] = None,  # (B, N, K) float or None
    ) -> torch.Tensor:
        """
        features           raw UNI v1 features (B should be 1 at train time)
        neighbor_index     per-patch neighbor indices in [0, N), -1 = no neighbor
        neighbor_mask      True at valid neighbor slots (False for -1 or self)
        h_morph            frozen-UNI-derived slide-intrinsic homogeneity; required
                           for srp_mode='post_agg_gated', ignored otherwise
        h_local            per-patch cosine-homogeneity (by design);
                           required for srp_mode='post_agg_signed_gated'.
                           Distinct from h_morph (Sobel-based) — both are
                           token-level homogeneity signals but compute
                           differently. The h_local definition matches
                           PANDA / ADP / NCT-CRC for cross-stage parity.

        Returns: (B, num_classes) logits.
        """
        B, N, _ = features.shape
        device = features.device

        # Validation: runtime input-contract checks
        # raise ValueError so they survive `python -O`.
        if neighbor_index.ndim != 3 or neighbor_index.shape[:2] != (B, N):
            raise ValueError(
                f"neighbor_index shape {tuple(neighbor_index.shape)} "
                f"!= ({B}, {N}, K)"
            )
        if neighbor_mask.shape != neighbor_index.shape:
            raise ValueError(
                f"neighbor_mask shape {tuple(neighbor_mask.shape)} "
                f"!= {tuple(neighbor_index.shape)}"
            )
        if neighbor_weight is not None and neighbor_weight.shape != neighbor_index.shape:
            raise ValueError(
                f"neighbor_weight shape {tuple(neighbor_weight.shape)} "
                f"!= {tuple(neighbor_index.shape)}"
            )
        if self.srp_mode == "post_agg_gated":
            if h_morph is None:
                raise ValueError(
                    "srp_mode='post_agg_gated' requires h_morph to be supplied"
                )
            if h_morph.shape != (B, N):
                raise ValueError(
                    f"h_morph shape {tuple(h_morph.shape)} != ({B}, {N})"
                )
        if self.srp_mode in _H_LOCAL_REQUIRED_SRP_MODES:
            if h_local is None:
                raise ValueError(
                    f"srp_mode={self.srp_mode!r} requires h_local "
                    "to be supplied"
                )
            if h_local.shape != (B, N):
                raise ValueError(
                    f"h_local shape {tuple(h_local.shape)} != ({B}, {N})"
                )

        # Input projection.
        x = self.in_proj(features)                            # (B, N, embed_dim)

        # Square-pad for PPEG: H = W = ceil(sqrt(N)), add = H*W - N.
        # Pad rows duplicate the first `add` feature rows (TransMIL's
        # approach; zero-padding would be out-of-distribution for the
        # learned Q/K/V weights).
        H = W = int(math.ceil(math.sqrt(N)))
        HW = H * W
        add = HW - N
        if add > 0:
            x = torch.cat([x, x[:, :add, :]], dim=1)          # (B, HW, embed_dim)

        # Prepend CLS.
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                        # (B, 1+HW, embed_dim)
        L = 1 + HW

        # Build is_real: True at positions 1..N; False at position 0 (CLS)
        # and positions 1+N..L-1 (pad duplicates).
        is_real = torch.zeros(B, L, device=device, dtype=torch.bool)
        if N > 0:
            is_real[:, 1 : 1 + N] = True

        # Block 0 -> PPEG -> blocks 1..depth-1 (TransMIL-faithful).
        x = self.blocks[0](
            x, neighbor_index, neighbor_mask, is_real, h_morph, h_local,
            neighbor_weight,
        )
        # PPEG sees the full (B, 1+HW, D) sequence. CLS stays at position 0.
        # Under the released protocol PPEG-removal ablation (self.use_ppeg=False),
        # self.ppeg is nn.Identity which ignores the (H, W) shape args;
        # we route around the call entirely so we don't need to forward
        # spatial dims to Identity.
        if self.use_ppeg:
            x = self.ppeg(x, H, W)
        for blk in self.blocks[1:]:
            x = blk(
                x, neighbor_index, neighbor_mask, is_real, h_morph, h_local,
                neighbor_weight,
            )

        x = self.norm(x)
        cls_out = x[:, 0]                                     # (B, embed_dim)
        return self.head(cls_out)                             # (B, num_classes)
