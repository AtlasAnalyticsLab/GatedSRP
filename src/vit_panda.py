"""
PandaSlideViT — set-style transformer for PANDA slide-level ISUP grading.

Stage-4 design (slide_level_panda/DESIGN.md §3.6):
  Input  (B, N_max, 1536)  UNI-v2 features (variable real N, padded with mask)
  ↓ Linear(1536, 384)         input projection
  ↓ prepend [CLS]              (B, 1+N_max, 384)
  # NO positional embedding — set-style, see DESIGN §3.5
  ↓ depth=4 PandaBlocks (full softmax attention with attn_mask + MLP)
  ↓ LayerNorm
  ↓ read CLS at position 0
  ↓ Linear(384, 6)             ISUP head

Why a new attention class instead of reusing src/xsa_attention.XSAAttention?
  XSAAttention.forward(self, x) does NOT accept attn_mask — only the
  diagonal-mask flag. Variable-length padding requires per-position
  masking before softmax, which the existing module doesn't support.
  Modifying XSAAttention to accept attn_mask would break Stage-1
  reproducibility unless gated; cleaner to write a sibling class here.

The XSA / SRP math (alpha- / beta-scaled post-attention projection) IS
ported in below for the SRP arm of Round 1 (panda_srp_beta2_fixed). For
Round 1's stress test we keep the simpler `panda_baseline` path
(alpha=0, no SRP) which is structurally equivalent to standard attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 8-neighbour gather + masked-mean utilities shared with Stage-3 SRP. The
# functions accept (B, H, N, D) value tensors and (B, N, 8) index/mask
# tensors; nothing PANDA-specific. Cross-validated bit-identical to
# Stage-3's `build_neighbor_index` graph (see src/data_panda.py).
from slide_level_srp.src.srp_attention import (        # noqa: E402
    _GATE_COUNT_FEATURES,
    gather_neighbors as _gather_neighbors,
    _gate_num_token_features,
    _make_token_diag,
    neighborhood_mean as _neighborhood_mean,
)
from slide_level_srp.src.rcd_modules import (          # noqa: E402
    IdentitySafeRCDRecomposer,
    LearnedLocalContextDirection,
    collect_rcd_module_ids,
    reset_rcd_identity_modules,
)

# Signed learned-gate SRP (LEARNED_GATE_SRP_PROPOSAL.md §2). Imported
# lazily-style at top so any signed_gated mode failure surfaces at
# construction, not at first forward.
from slide_level_srp.src.gate_signed import TokenHeadGate, collect_gate_module_ids  # noqa: E402

# Number of token-level diagnostic features fed into the gate's
# MLP_token. Must match the assemble order in `_make_gate_token_diag`
# below: [h_local, neighbour_count/8, log_neighbour_count, log_norm_y_mean].
# Exposed as a constant so slide_level_srp/train_panda.py can verify gate config without
# importing PandaAttention internals.
PANDA_GATE_NUM_TOKEN_FEATURES = 4
# Number of per-head diagnostic features. Order: [cos(y, r̂), |cos|, log_norm_y].
PANDA_GATE_NUM_HEAD_FEATURES = 3
_SIGNED_GATE_MODES = (
    "srp_signed_gated",
    "srp_signed_gated_learned_r",
    "srp_signed_gated_pre_q",
    "srp_signed_gated_pre_k",
)


# --- DropPath + MLP (verbatim from src/vit.py) ---------------------------

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop2 = nn.Dropout(drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


# --- Variable-length attention with optional XSA / SRP projection ---------

class PandaAttention(nn.Module):
    """
    Full softmax attention over (1 + N_max) tokens with a variable-length
    boolean mask. Supports three Round-1 ablation configurations:

      mode="baseline"      α=0 (identity post-projection)
      mode="xsa_all_hard"  α_cls=α_patch=1 (hard XSA on every token)
      mode="srp_beta2"     SRP at scalar β against r̂ — the projection
                           target r̂ is selected by `r_target`:

                             "slide_mean" : r̂ = unit(mean over real-patch v)
                                             — slide-wide mean (Round 1).
                             "knn8"       : r̂_i = unit(mean of v over the
                                             8 spatial neighbours of i,
                                             from H5 /coords) — Round 2,
                                             mirrors Stage-3 / Phase-2
                                             grid-local SRP. Requires
                                             `neighbor_index` and
                                             `neighbor_mask` in forward().

    For the stress test we instantiate `mode="baseline"` because it
    exercises the same attention/MLP/residual structure that any other
    mode would; the post-projection step adds < 1 % to wall time and
    < 1 % to peak memory.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mode: str = "baseline",
        beta: float = 2.0,
        r_target: str = "slide_mean",      # "slide_mean" | "knn8"
        num_cls_tokens: int = 1,
        # Signed-gate (proposal §2.1, §2.2) — consulted when mode is one of
        # _SIGNED_GATE_MODES.  The learned-r variant keeps the same gate
        # intervention and changes only the local direction estimator.
        delta_scale: float = 2.0,
        gate_active: bool = True,
        gate_hidden_dim: int = 16,
        # When False, gate diagnostic inputs (token_diag includes
        # log_norm_y_mean, head_diag includes cos_yr / log_norm_y_h)
        # are NOT detached before the gate forward — gradients flow
        # back through the y-derived diagnostics into y. This is the
        # "live" regime; default True is the proposal §6.3 detached
        # regime. See §6.3.1 for the +0.98 pp ADP detach finding that
        # motivates this toggle.
        detach_gate_inputs: bool = True,
        gate_output_init: str = "zero",
        gate_output_init_scale: float = 1.0,
        gate_init_beta0: float = 0.0,
        gate_activation: str = "tanh",
        gate_activation_temperature: float = 1.0,
        gate_count_features: str = "legacy",
        # Method 2.1/2.4 controls.  RCD uses both branch recomposition and
        # optional learned-r; standalone Method 2.4 uses only context_scorer.
        rcd_adapter_kind: str = "lowrank",
        rcd_rank: int = 16,
        learned_r_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        assert mode in (
            "baseline", "xsa_all_hard", "srp_beta2", "srp_signed_gated",
            "srp_signed_gated_learned_r", "srp_signed_gated_pre_q",
            "srp_signed_gated_pre_k",
            "srp_rcd", "srp_rcd_learned_r",
        )
        # The signed-gate path always uses the knn8 r̂ family. slide_mean
        # is incompatible by construction: a signed local gate over a
        # slide-wide r̂ would degenerate to a per-token rescaling of one
        # global direction, defeating the purpose of token-level
        # specialisation. If the user wants slide_mean r̂, they should
        # use mode="srp_beta2" with r_target="slide_mean" instead.
        if mode in _SIGNED_GATE_MODES or mode == "srp_rcd_learned_r":
            assert r_target == "knn8", (
                f"{mode} requires r_target='knn8'; slide_mean is not "
                "local enough for token-level learned directions."
            )
        else:
            assert r_target in ("slide_mean", "knn8")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mode = mode
        self.r_target = r_target
        self.num_cls_tokens = num_cls_tokens
        # Stored as buffer so it follows .to(device) but doesn't enter
        # the optimizer.
        self.register_buffer("beta", torch.tensor(float(beta)), persistent=True)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Signed-gate state. The gate is instantiated only when
        #   (a) mode is the signed-gate variant, AND
        #   (b) gate_active is True.
        # gate_active=False is the dead-path opt-out (proposal §2.4):
        # for the LAST attention block, the SRP correction has zero
        # downstream consumer (CLS-only readout, post-attention CLS
        # pass-through), so gate parameters in the last block would
        # receive zero gradient and look "uniformly stuck at init"
        # post-hoc. The parent module sets gate_active=False there to
        # avoid the false-negative.
        self.delta_scale = float(delta_scale)
        self.detach_gate_inputs = bool(detach_gate_inputs)
        if gate_count_features not in _GATE_COUNT_FEATURES:
            raise ValueError(
                f"gate_count_features must be one of {_GATE_COUNT_FEATURES}, "
                f"got {gate_count_features!r}"
            )
        self.gate_count_features = gate_count_features
        self.gate_active = bool(gate_active and mode in _SIGNED_GATE_MODES)
        # Standalone Method 2.4 on PANDA: preserve signed-gate beta_eff and
        # replace only the fixed knn8 r_hat with the learned local scorer.
        self.learned_r_gate_active = bool(
            gate_active and mode == "srp_signed_gated_learned_r"
        )
        if self.gate_active:
            # Save & restore the global RNG state across gate
            # construction. nn.Linear.__init__ inside TokenHeadGate
            # consumes RNG via reset_parameters() (Kaiming uniform on
            # weights, uniform on biases). Without this guard, building
            # the gate would shift the RNG state for the *subsequent*
            # nn.Linear modules in the same block (norm2 if it were
            # Linear, then mlp.fc1, fc2), making it impossible to
            # reproduce baseline parameter init at the same seed when
            # toggling between modes. With the save/restore, the gate's
            # presence is invisible to other modules' init.
            rng_state_cpu = torch.get_rng_state()
            try:
                self.gate = TokenHeadGate(
                    num_heads=num_heads,
                    num_token_features=_gate_num_token_features(
                        self.gate_count_features,
                        include_y_norm_mean=True,
                    ),
                    num_head_features=PANDA_GATE_NUM_HEAD_FEATURES,
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

        # RCD state.  The parent passes gate_active=False on the last
        # block for every post-attention patch-only method, because a
        # CLS-only readout has no downstream consumer for a final patch
        # write-back.  Reusing that flag avoids allocating dead RCD params.
        self.rcd_active = bool(
            gate_active and mode in ("srp_rcd", "srp_rcd_learned_r")
        )
        self.learned_r_active = bool(
            self.rcd_active and mode == "srp_rcd_learned_r"
        )
        if self.rcd_active or self.learned_r_gate_active:
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
        # Per-forward gate-output cache for diagnostics. Populated by
        # forward() when self.gate is not None; slide_level_srp/train_panda.py reads
        # this dict to log gate stats. Cleared at the start of every
        # forward to avoid leaking across calls.
        self._last_gate_stats: dict | None = None
        # Training-time cache for Phase11 gate containment.  The stats cache
        # above is detached on purpose for logging; this separate tensor stays
        # attached to the gate computation so `slide_level_srp/train_panda.py` can add
        # mean(beta_eff^2) to the loss without accidentally regularizing a
        # stale value from a previous batch.
        self._last_gate_beta_eff_for_loss: torch.Tensor | None = None
        self._last_rcd_stats: dict | None = None

    def _apply_signed_gate_to_patch_stream(
        self,
        stream_patch: torch.Tensor,
        *,
        neighbor_index: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
        neighbor_weight: torch.Tensor | None,
        h_local: torch.Tensor | None,
        stream_label: str,
    ) -> torch.Tensor:
        """
        Apply the signed-gated SRP formula to one patch-token stream.

        Phase18 uses this for pre-Q/pre-K:

            Q_patch' = Q_patch - beta_eff * <Q_patch, r_hat^Q> * r_hat^Q
            K_patch' = K_patch - beta_eff * <K_patch, r_hat^K> * r_hat^K

        where r_hat is the detached knn-local mean direction in the edited
        stream's feature space.
        The method mirrors the post-attention PANDA gate diagnostics, but
        swaps the edited stream from Y to Q/K.  CLS is not present in
        `stream_patch`, so callers preserve CLS by only writing back the
        patch slice.
        """
        if neighbor_index is None or neighbor_mask is None:
            raise ValueError(
                f"{self.mode} requires neighbor_index and neighbor_mask "
                "tensors in forward()."
            )

        B, H, N_max, _D = stream_patch.shape
        neighbor_stream_det = _gather_neighbors(
            stream_patch.detach(), neighbor_index, neighbor_mask,
        )
        _r, r_hat, cnt = _neighborhood_mean(
            neighbor_stream_det, neighbor_mask, neighbor_weight,
        )
        dot_sr = (stream_patch * r_hat).sum(dim=-1, keepdim=True)

        if not self.gate_active:
            # Dead-path block: final-block post-Y/pre-Q patch edits do not
            # affect the final CLS query row, so the parent disables them.
            return stream_patch

        if h_local is None:
            raise ValueError(
                f"{self.mode} with gate_active=True requires h_local in "
                "forward()."
            )
        if h_local.shape != (B, N_max):
            raise ValueError(
                f"h_local shape mismatch: got {tuple(h_local.shape)}, "
                f"expected ({B}, {N_max})"
            )

        cnt_bn = cnt.squeeze(-1).squeeze(1).to(dtype=stream_patch.dtype)
        stream_norms = stream_patch.norm(dim=-1)
        # Keep the same four token diagnostic channels as the post-Y PANDA
        # gate.  Under pre-Q, the norm channel is log_norm_Q_mean rather than
        # log_norm_Y_mean; this isolates placement while preserving gate size.
        log_norm_stream_mean = torch.log1p(stream_norms).mean(dim=1)
        token_diag = _make_token_diag(
            h_local=h_local.to(stream_patch.dtype),
            cnt_bn=cnt_bn,
            max_neighbors=neighbor_index.shape[-1],
            mode=self.gate_count_features,
            y_norm_mean=log_norm_stream_mean,
        )

        eps = 1e-12
        cos_sr = dot_sr.squeeze(-1) / (stream_norms + eps)
        head_diag = torch.stack(
            [cos_sr, cos_sr.abs(), torch.log1p(stream_norms)], dim=-1,
        )
        if self.detach_gate_inputs:
            token_diag = token_diag.detach()
            head_diag = head_diag.detach()

        beta_eff = self.gate(token_diag, head_diag)
        self._last_gate_beta_eff_for_loss = beta_eff
        with torch.no_grad():
            # Existing collectors read cos_yr/y_norms.  For pre-attention
            # modes these mean cos(stream, r_hat^stream) and ||stream||;
            # gate_stream_id marks 1=pre-Q and 2=pre-K.
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
        return stream_patch - beta_eff * dot_sr * r_hat

    def forward(
        self,
        x: torch.Tensor,
        mask_full: torch.Tensor,
        neighbor_index: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        neighbor_weight: torch.Tensor | None = None,
        h_local: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x              (B, L, C)        L = 1 + N_max (CLS at position 0)
        mask_full      (B, L) bool      True at real-patch positions + CLS
        neighbor_index (B, N_max, 8) long, REAL-PATCH indices in [0, N_max)
                       — only required when mode='srp_beta2' AND
                       r_target='knn8', or mode uses signed-gate/RCD local r.
                       Indices are into the patch slice (positions
                       1..1+N_max), NOT including the CLS at 0.
        neighbor_mask  (B, N_max, 8) bool — same shape; True at valid slots.
        neighbor_weight(B, N_max, K) float — optional common-mode weights.
        h_local        (B, N_max) float — per-patch local homogeneity,
                       precomputed once per slide in the dataset (see
                       src/data_panda.py::_compute_h_local). Required
                       under signed-gated local-r modes; ignored otherwise.
        """
        # Reset the gate-stats cache at the start of every forward.
        # _last_gate_stats is populated only on the signed-gated path
        # below; slide_level_srp/train_panda.py's diagnostic hook reads None when no
        # gate is active so callers don't need to branch.
        self._last_gate_stats = None
        self._last_gate_beta_eff_for_loss = None
        self._last_rcd_stats = None
        B, L, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                           # (B, H, L, D)

        if self.mode in ("srp_signed_gated_pre_q", "srp_signed_gated_pre_k"):
            # Phase18 pre-attention arms edit one patch stream before QK.
            # CLS rows are preserved.  Pre-Q has no direct final-block CLS
            # path; pre-K does, because the final CLS query attends to edited
            # patch keys in the same block.
            n_cls = self.num_cls_tokens
            if self.mode == "srp_signed_gated_pre_q":
                q_patch = self._apply_signed_gate_to_patch_stream(
                    q[:, :, n_cls:, :],
                    neighbor_index=neighbor_index,
                    neighbor_mask=neighbor_mask,
                    neighbor_weight=neighbor_weight,
                    h_local=h_local,
                    stream_label="q",
                )
                q = q.clone()
                q[:, :, n_cls:, :] = q_patch
            else:
                k_patch = self._apply_signed_gate_to_patch_stream(
                    k[:, :, n_cls:, :],
                    neighbor_index=neighbor_index,
                    neighbor_mask=neighbor_mask,
                    neighbor_weight=neighbor_weight,
                    h_local=h_local,
                    stream_label="k",
                )
                k = k.clone()
                k[:, :, n_cls:, :] = k_patch

        # (B, H, L, L) attention logits, with key-side mask applied
        # to the last dim before softmax. Pad keys → -inf, contributing 0
        # after softmax. Pad queries also produce attention rows but their
        # outputs are discarded downstream (CLS reads position 0; pad-row
        # gradients flow but contribute nothing meaningful to the loss).
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        # Broadcast key mask across (B, H, L_query, L_key).
        key_mask = mask_full.unsqueeze(1).unsqueeze(2)    # (B, 1, 1, L)
        attn_logits = attn_logits.masked_fill(~key_mask, float("-inf"))
        attn_probs = attn_logits.softmax(dim=-1)
        # If a query row has all keys masked (shouldn't happen here since
        # CLS is always real, even for a direct zero-patch smoke input),
        # softmax of all-neg-inf produces NaN. Guard:
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = self.attn_drop(attn_probs)
        y = attn_probs @ v                                # (B, H, L, D)

        if self.mode == "xsa_all_hard":
            # XSA: z = y - α·(y·v̂)·v̂, α=1 here. v̂ from per-head L2 norm.
            vn = F.normalize(v, dim=-1)
            dot = (y * vn).sum(dim=-1, keepdim=True)
            z = y - dot * vn
        elif self.mode == "srp_beta2":
            n_cls = self.num_cls_tokens
            beta_dt = self.beta.to(y.dtype)

            if self.r_target == "slide_mean":
                # Slide-wide mean r̂ = unit(mean over real-patch v vectors,
                # per head). Same r̂ for every patch in the slide.
                # CLS is excluded from the mean and from the SRP write-back.
                # r̂ is detached.
                patch_mask = mask_full.clone()
                patch_mask[:, :n_cls] = False                 # exclude CLS
                mf = patch_mask.to(v.dtype).unsqueeze(1).unsqueeze(-1)   # (B, 1, L, 1)
                v_det = v.detach()
                r = (v_det * mf).sum(dim=2) / mf.sum(dim=2).clamp(min=1.0)   # (B, H, D)
                r_hat = F.normalize(r, dim=-1, eps=1e-12)                    # (B, H, D)
                r_hat_b = r_hat.unsqueeze(2)                                 # (B, H, 1, D)
                # Project all rows; CLS row restored below.
                dot_yr = (y * r_hat_b).sum(dim=-1, keepdim=True)             # (B, H, L, 1)
                z = y - beta_dt * dot_yr * r_hat_b
                # CLS pass-through.
                z = z.clone()
                z[:, :, :n_cls, :] = y[:, :, :n_cls, :]

            else:  # self.r_target == "knn8"
                # Per-patch r̂ from the 8 spatial neighbours' v vectors.
                # Mirrors Stage-3 / Phase-2 grid-local SRP. Requires the
                # forward to be called with neighbor_index / neighbor_mask
                # tensors (one row per patch, shape (B, N_max, 8)). r̂ is
                # detached via gather_neighbors taking v_patch.detach().
                # Phase-A.9 second-review fix F7: runtime input contract
                # raises ValueError so it survives `python -O`.
                if neighbor_index is None or neighbor_mask is None:
                    raise ValueError(
                        "srp_beta2 with r_target='knn8' requires "
                        "neighbor_index and neighbor_mask tensors in forward()."
                    )
                # Slice patch rows: (B, H, N_max, D). CLS occupies row [0..n_cls).
                v_patch = v[:, :, n_cls:, :]                              # (B, H, N_max, D)
                y_patch = y[:, :, n_cls:, :]                              # (B, H, N_max, D)
                # Gather neighbours from detached v. Returns (B, H, N_max, 8, D).
                neighbor_v_det = _gather_neighbors(
                    v_patch.detach(), neighbor_index, neighbor_mask,
                )
                _r, r_hat, _cnt = _neighborhood_mean(
                    neighbor_v_det, neighbor_mask, neighbor_weight,
                )                                                         # r_hat: (B, H, N_max, D)
                dot_yr = (y_patch * r_hat).sum(dim=-1, keepdim=True)      # (B, H, N_max, 1)
                z_patch = y_patch - beta_dt * dot_yr * r_hat              # (B, H, N_max, D)
                # Pad rows have neighbor_mask all-False, so cnt clamps to
                # 1 and r_hat is normalize(0)=0 → projection term is zero
                # → z_patch = y_patch at pad rows. No explicit pad-zero
                # write-back needed beyond CLS restoration.
                z = y.clone()
                z[:, :, n_cls:, :] = z_patch
                # CLS already untouched (we only wrote to [n_cls:]).
        elif self.mode in ("srp_rcd", "srp_rcd_learned_r"):
            # Method 2.1 / 2.4 RCD: decompose each patch attention output
            # into context-common and residual components, then learn
            # non-shared zero-initialized branch deltas.  This is a
            # different intervention surface from scalar β or signed β_eff:
            # it starts exactly at identity and learns how to recombine the
            # two branches instead of subtracting a projection term.
            if not self.rcd_active:
                # Last block dead-path: final post-attention patch writes do
                # not feed the CLS readout, so this block intentionally has
                # no RCD parameters and behaves as the baseline.
                z = y
            else:
                n_cls = self.num_cls_tokens
                v_patch = v[:, :, n_cls:, :]                              # (B, H, N_max, D)
                y_patch = y[:, :, n_cls:, :]                              # (B, H, N_max, D)

                if self.learned_r_active:
                    # Method 2.4 remains strictly local: it learns additive
                    # neighbour logits over the supplied knn graph.  h_local
                    # is the only extra token covariate, mirroring the signed
                    # gate input without exposing labels or site metadata.
                    if neighbor_index is None or neighbor_mask is None:
                        raise ValueError(
                            "srp_rcd_learned_r requires neighbor_index and "
                            "neighbor_mask tensors in forward()."
                        )
                    if h_local is None:
                        raise ValueError(
                            "srp_rcd_learned_r requires h_local in forward()."
                        )
                    neighbor_v_det = _gather_neighbors(
                        v_patch.detach(), neighbor_index, neighbor_mask,
                    )
                    _r, r_hat, _cnt, learned_r_weight = self.context_scorer(
                        center_v=v_patch.detach(),
                        neighbor_v=neighbor_v_det,
                        neighbor_mask=neighbor_mask,
                        neighbor_weight=neighbor_weight,
                        h_local=h_local,
                    )
                    with torch.no_grad():
                        self._last_rcd_stats = {
                            "learned_r_weight": learned_r_weight.detach(),
                        }
                elif self.r_target == "slide_mean":
                    # Slide-wide fixed direction control for Method 2.1.
                    # Real patch rows only contribute to the mean; padded
                    # rows are masked out and do not become keys later.
                    patch_mask = mask_full[:, n_cls:]
                    mf = patch_mask.to(v.dtype).unsqueeze(1).unsqueeze(-1)
                    v_det = v_patch.detach()
                    r = (v_det * mf).sum(dim=2) / mf.sum(dim=2).clamp(min=1.0)
                    r_hat = F.normalize(r, dim=-1, eps=1e-12).unsqueeze(2)
                    r_hat = r_hat.expand_as(y_patch)
                else:  # self.r_target == "knn8"
                    if neighbor_index is None or neighbor_mask is None:
                        raise ValueError(
                            "srp_rcd with r_target='knn8' requires "
                            "neighbor_index and neighbor_mask tensors in forward()."
                        )
                    neighbor_v_det = _gather_neighbors(
                        v_patch.detach(), neighbor_index, neighbor_mask,
                    )
                    _r, r_hat, _cnt = _neighborhood_mean(
                        neighbor_v_det, neighbor_mask, neighbor_weight,
                    )

                z_patch, rcd_stats = self.rcd_recomposer(y_patch, r_hat)
                with torch.no_grad():
                    merged = {} if self._last_rcd_stats is None else dict(self._last_rcd_stats)
                    merged.update(rcd_stats)
                    self._last_rcd_stats = merged
                z = y.clone()
                z[:, :, n_cls:, :] = z_patch
        elif self.mode in ("srp_signed_gated_pre_q", "srp_signed_gated_pre_k"):
            # Pre-attention placement already applied the signed projection.
            # Do not add a second post-Y correction; Phase18 isolates the
            # placement change against the current post-attention design.
            z = y
        elif self.mode in _SIGNED_GATE_MODES:
            # Signed learned-gate SRP: same r̂ machinery as srp_beta2 +
            # knn8, but the per-(token, head) gate replaces scalar β.
            # srp_signed_gated_learned_r keeps this exact formula and
            # learns r_hat through Method 2.4's local-neighbour scorer.
            # Identity at init by construction (proposal §2.3): the
            # gate's output-path zero-init makes raw_logit = 0, so
            # β_eff = δ · tanh(0) = 0 and z_patch = y_patch — exactly
            # the un-projected baseline. The model learns where to
            # apply non-zero β over the course of training.
            # Phase-A.9 second-review fix F7: ValueError instead of assert.
            if neighbor_index is None or neighbor_mask is None:
                raise ValueError(
                    f"{self.mode} requires neighbor_index and "
                    "neighbor_mask tensors in forward()."
                )
            n_cls = self.num_cls_tokens
            v_patch = v[:, :, n_cls:, :]                                  # (B, H, N_max, D)
            y_patch = y[:, :, n_cls:, :]                                  # (B, H, N_max, D)
            neighbor_v_det = _gather_neighbors(
                v_patch.detach(), neighbor_index, neighbor_mask,
            )
            if self.learned_r_gate_active:
                if h_local is None:
                    raise ValueError(
                        f"{self.mode} requires h_local in forward() for "
                        "Method 2.4 scoring."
                    )
                r, r_hat, cnt, learned_r_weight = self.context_scorer(
                    center_v=v_patch.detach(),
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
                r, r_hat, cnt = _neighborhood_mean(
                    neighbor_v_det, neighbor_mask, neighbor_weight,
                )                                                         # r:(B,H,N,D), r_hat:(B,H,N,D), cnt:(B,N)
            dot_yr = (y_patch * r_hat).sum(dim=-1, keepdim=True)          # (B, H, N_max, 1)

            if self.gate_active:
                # Build per-token diagnostics (token-level, shared across heads).
                # 1. h_local: precomputed cosine homogeneity, shape (B, N).
                #    Required input — not optional, since CAM17 Phase 0
                #    confirmed h_local is the most predictive token-level
                #    gate input.
                # Phase-A.9 second-review fix F7: ValueError instead of assert.
                if h_local is None:
                    raise ValueError(
                        f"{self.mode} with gate_active=True requires "
                        "h_local in forward()."
                    )
                # 2. neighbour_count / 8: a low-support patch has
                #    unreliable r̂; the gate should learn to attenuate
                #    projection there. _neighborhood_mean returns cnt
                #    with shape (B, 1, N, 1) per the Stage-3 convention
                #    — squeeze the head and last dims to (B, N) since
                #    neighbour count is intrinsic to the patch, not
                #    head-specific.
                cnt_bn = cnt.squeeze(-1).squeeze(1).to(dtype=y.dtype)     # (B, N)
                # 4. log_norm_y_mean: per-token magnitude of y, averaged
                #    over heads. Low-magnitude tokens are noise-dominated;
                #    the gate may want to suppress projection there.
                y_norms = y_patch.norm(dim=-1)                            # (B, H, N)
                log_norm_y_mean = torch.log1p(y_norms).mean(dim=1)        # (B, N)
                token_diag = _make_token_diag(
                    h_local=h_local.to(y.dtype),
                    cnt_bn=cnt_bn,
                    max_neighbors=neighbor_index.shape[-1],
                    mode=self.gate_count_features,
                    y_norm_mean=log_norm_y_mean,
                )

                # Per-head diagnostics. cos_yr = cos(y, r̂); since r_hat
                # is unit-norm, this is just dot_yr / ‖y‖.
                eps = 1e-12
                cos_yr = (dot_yr.squeeze(-1) /
                          (y_norms + eps))                                # (B, H, N)
                abs_cos_yr = cos_yr.abs()
                log_norm_y_h = torch.log1p(y_norms)                       # (B, H, N)
                head_diag = torch.stack(
                    [cos_yr, abs_cos_yr, log_norm_y_h], dim=-1,
                )                                                         # (B, H, N, 3)

                # Proposal §6.3 detach convention: by default, gate
                # diagnostic inputs are stop-grad'd. detach_gate_inputs
                # toggles this — see __init__ for rationale; the +0.98
                # pp ADP detach finding (§6.3.1) is what motivates the
                # toggle.
                if self.detach_gate_inputs:
                    token_diag = token_diag.detach()
                    head_diag = head_diag.detach()

                # Gate forward → (B, H, N, 1).
                beta_eff = self.gate(token_diag, head_diag)
                # Keep the live beta surface for a same-step loss term.
                # Do not detach here: the L2 penalty must push the gate
                # parameters, while the detached logging dict below remains
                # safe for W&B/artifact collection.
                self._last_gate_beta_eff_for_loss = beta_eff
                # Stash diagnostics for the trainer to log. Detached so
                # the optimizer never sees these tensors as gradients.
                with torch.no_grad():
                    self._last_gate_stats = {
                        "beta_eff": beta_eff.detach(),
                        "cos_yr": cos_yr.detach(),
                        "y_norms": y_norms.detach(),
                        "h_local": h_local.detach(),
                        "neighbour_count": cnt_bn.detach(),
                    }
            else:
                # Dead-path: gate excluded for this block (proposal §2.4).
                # β_eff = 0 → z_patch = y_patch; equivalent to baseline
                # for this block. Constructing a zero tensor keeps the
                # downstream einsum shape uniform with the active path.
                beta_eff = torch.zeros_like(dot_yr)

            z_patch = y_patch - beta_eff * dot_yr * r_hat                 # (B, H, N_max, D)
            z = y.clone()
            z[:, :, n_cls:, :] = z_patch
        else:
            # baseline: identity.
            z = y

        z = z.transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(z))


# --- Block + ViT ---------------------------------------------------------

class PandaBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 mode: str = "baseline", beta: float = 2.0,
                 r_target: str = "slide_mean",
                 drop_path: float = 0.0,
                 num_cls_tokens: int = 1,
                 # Signed-gate options forwarded to PandaAttention.
                 # gate_active=False on the last block enforces the
                 # dead-path rule (proposal §2.4).
                 delta_scale: float = 2.0,
                 gate_active: bool = True,
                 gate_hidden_dim: int = 16,
                 detach_gate_inputs: bool = True,
                 gate_output_init: str = "zero",
                 gate_output_init_scale: float = 1.0,
                 gate_init_beta0: float = 0.0,
                 gate_activation: str = "tanh",
                 gate_activation_temperature: float = 1.0,
                 gate_count_features: str = "legacy",
                 # Method 2.1/2.4 options forwarded to PandaAttention.
                 rcd_adapter_kind: str = "lowrank",
                 rcd_rank: int = 16,
                 learned_r_hidden_dim: int = 16) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PandaAttention(
            dim, num_heads,
            attn_drop=attn_drop, proj_drop=proj_drop,
            mode=mode, beta=beta, r_target=r_target,
            num_cls_tokens=num_cls_tokens,
            delta_scale=delta_scale,
            gate_active=gate_active,
            gate_hidden_dim=gate_hidden_dim,
            detach_gate_inputs=detach_gate_inputs,
            gate_output_init=gate_output_init,
            gate_output_init_scale=gate_output_init_scale,
            gate_init_beta0=gate_init_beta0,
            gate_activation=gate_activation,
            gate_activation_temperature=gate_activation_temperature,
            gate_count_features=gate_count_features,
            rcd_adapter_kind=rcd_adapter_kind,
            rcd_rank=rcd_rank,
            learned_r_hidden_dim=learned_r_hidden_dim,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_dim=dim, hidden_dim=int(dim * mlp_ratio), drop=proj_drop)
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        mask_full: torch.Tensor,
        neighbor_index: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        neighbor_weight: torch.Tensor | None = None,
        h_local: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(
            self.norm1(x), mask_full,
            neighbor_index=neighbor_index, neighbor_mask=neighbor_mask,
            neighbor_weight=neighbor_weight,
            h_local=h_local,
        ))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PandaSlideViT(nn.Module):
    """
    Set-style ViT for PANDA. Default config matches DESIGN §3.6 / §4:
      depth=4, dim=384, heads=6, mlp_ratio=4, drop_path=0.1,
      num_classes=6 (ISUP), no positional embedding.
    """
    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 6,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.1,
        mode: str = "baseline",
        beta: float = 2.0,
        r_target: str = "slide_mean",
        # Signed-gate options. Only consumed under signed-gated modes.
        delta_scale: float = 2.0,
        gate_hidden_dim: int = 16,
        detach_gate_inputs: bool = True,
        gate_output_init: str = "zero",
        gate_output_init_scale: float = 1.0,
        gate_init_beta0: float = 0.0,
        gate_activation: str = "tanh",
        gate_activation_temperature: float = 1.0,
        gate_count_features: str = "legacy",
        # Method 2.1/2.4 controls.  Inert unless mode is an RCD mode.
        rcd_adapter_kind: str = "lowrank",
        rcd_rank: int = 16,
        learned_r_hidden_dim: int = 16,
        pos_mode: str = "none",
        coord_pos_dim: int = 64,
        coord_norm: str = "slide_minmax",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.r_target = r_target
        self.depth = depth
        if pos_mode not in ("none", "coord_mlp"):
            raise ValueError(f"pos_mode for PandaSlideViT must be 'none' or 'coord_mlp', got {pos_mode!r}")
        if coord_norm != "slide_minmax":
            raise ValueError(f"coord_norm must be 'slide_minmax', got {coord_norm!r}")
        self.pos_mode = pos_mode
        self.coord_norm = coord_norm
        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.coord_pos_mlp = None
        if self.pos_mode == "coord_mlp":
            self.coord_pos_mlp = nn.Sequential(
                nn.Linear(2, coord_pos_dim),
                nn.GELU(),
                nn.Linear(coord_pos_dim, embed_dim),
            )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Linear depth schedule for drop-path.
        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist() if depth > 0 else []
        # Dead-path rule (proposal §2.4): post-aggregation and pre-Q patch
        # gates have no final-block CLS consumer.  Pre-K is live in the final
        # block because the CLS query attends to edited patch keys.
        final_block_gate_live = mode == "srp_signed_gated_pre_k"
        self.blocks = nn.ModuleList([
            PandaBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop, proj_drop=proj_drop,
                mode=mode, beta=beta, r_target=r_target,
                drop_path=dpr[i],
                num_cls_tokens=1,
                delta_scale=delta_scale,
                gate_active=(i < depth - 1 or final_block_gate_live),
                gate_hidden_dim=gate_hidden_dim,
                detach_gate_inputs=detach_gate_inputs,
                gate_output_init=gate_output_init,
                gate_output_init_scale=gate_output_init_scale,
                gate_init_beta0=gate_init_beta0,
                gate_activation=gate_activation,
                gate_activation_temperature=gate_activation_temperature,
                gate_count_features=gate_count_features,
                rcd_adapter_kind=rcd_adapter_kind,
                rcd_rank=rcd_rank,
                learned_r_hidden_dim=learned_r_hidden_dim,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        # Build the skip-set BEFORE the trunc_normal pass so that gate
        # internals are excluded. Without this, trunc_normal_ on the
        # gate's two nn.Linear modules would consume RNG and shift
        # downstream non-gate weight init away from a same-seed
        # baseline build. See slide_level_srp.src.gate_signed
        # collect_gate_module_ids
        # docstring for the rationale.
        gate_module_ids = collect_gate_module_ids(self)
        rcd_module_ids = collect_rcd_module_ids(self)
        method_module_ids = gate_module_ids | rcd_module_ids
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
        # Belt-and-suspenders: even though the trunc_normal pass above
        # now skips gate internals, also re-apply the configured gate
        # output-path initialization after the parent init pass.
        for m in self.modules():
            if isinstance(m, TokenHeadGate):
                m.reset_output_path()
        reset_rcd_identity_modules(self.modules())

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        neighbor_index: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        neighbor_weight: torch.Tensor | None = None,
        h_local: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        features        (B, N_max, in_dim)
        mask            (B, N_max) bool     — True at real patches
        neighbor_index  (B, N_max, 8) long  — required when SRP mode
                                              + r_target='knn8' OR
                                              mode uses learned local r.
        neighbor_mask   (B, N_max, 8) bool
        h_local         (B, N_max) float    — required when
                                              mode='srp_signed_gated',
                                              mode='srp_signed_gated_learned_r',
                                              or mode='srp_rcd_learned_r'.
        Returns   (B, num_classes)
        """
        # Phase-A.9 second-review fix F7: ValueError instead of assert.
        if self.r_target == "knn8" or self.mode in _SIGNED_GATE_MODES or self.mode == "srp_rcd_learned_r":
            if neighbor_index is None or neighbor_mask is None:
                raise ValueError(
                    "PandaSlideViT requires neighbor_index and neighbor_mask "
                    "under r_target='knn8' or learned-local-r modes."
                )
        if self.mode in _SIGNED_GATE_MODES or self.mode == "srp_rcd_learned_r":
            if h_local is None:
                raise ValueError(
                    f"PandaSlideViT(mode={self.mode!r}) requires h_local "
                    "in forward()."
                )
        B, N_max, _ = features.shape
        x = self.in_proj(features)                                # (B, N_max, embed_dim)
        if self.coord_pos_mlp is not None:
            if coords is None:
                raise ValueError("PandaSlideViT(pos_mode='coord_mlp') requires coords in forward().")
            c = coords.to(dtype=x.dtype, device=x.device)
            if c.ndim != 3 or c.shape[0] != B or c.shape[1] != N_max or c.shape[2] < 2:
                raise ValueError(
                    "PandaSlideViT(pos_mode='coord_mlp') expects coords "
                    f"shape (B, N_max, >=2) matching features; got "
                    f"{tuple(c.shape)} for features {tuple(features.shape)}."
                )
            c = c[:, :, :2]
            if N_max > 0:
                # Empty slides have no coordinate extrema. The loader now
                # filters zero-patch PANDA H5s before folds are built, but
                # keeping the model tolerant makes direct smoke tests and
                # legacy callers behave like the non-positional path: the
                # classifier sees only CLS instead of crashing on amin/amax.
                cmin = c.amin(dim=1, keepdim=True)
                cmax = c.amax(dim=1, keepdim=True)
                c = (c - cmin) / (cmax - cmin).clamp(min=1.0)
                c = c * 2.0 - 1.0
                x = x + self.coord_pos_mlp(c)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                            # (B, 1+N_max, embed_dim)
        # Augment mask with the CLS True at position 0.
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        mask_full = torch.cat([cls_mask, mask], dim=1)            # (B, 1+N_max)

        for blk in self.blocks:
            x = blk(x, mask_full,
                    neighbor_index=neighbor_index,
                    neighbor_mask=neighbor_mask,
                    neighbor_weight=neighbor_weight,
                    h_local=h_local)
        x = self.norm(x)
        cls_out = x[:, 0]                                          # (B, embed_dim)
        return self.head(cls_out)
