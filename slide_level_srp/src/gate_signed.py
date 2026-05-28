"""
Signed learned-gate SRP module.

Implements the TokenHeadGate parameterization:

    raw_logit_{i,h} = MLP_token(token_diag_i) + Linear_head(head_diag_{i,h}) + b_{l,h}
    beta_eff_{i,h}  = delta_scale * tanh(raw_logit_{i,h})

The SRP projection then becomes per-token-per-head:

    z_{i,h} = y_{i,h} - beta_eff_{i,h} * (y_{i,h} · r_hat_{i,h}) * r_hat_{i,h}

The gate is identity-initialised: with all output-path parameters set to
zero, raw_logit = 0 and beta_eff = 0 at step 1, so the model starts as
the un-projected baseline (by design).

Why this design:

  - **Shared per-token gate (MLP_token) + per-head adjustment
    (Linear_head + per-head bias):** most of the gate decision should
    come from local context that is the same across heads (homogeneity,
    neighbour count). Each head can shift it via a small adjustment
    based on per-head diagnostics (alignment with r_hat, magnitude).
    This mirrors how attention heads specialise while still consuming
    the same per-token context.

  - **Signed via tanh, not sigmoid:** early CAM17 diagnostics showed
    rare-class ITC slides prefer beta = -1 (anti-SRP), which a
    sigmoid-bounded [0, beta_max] gate cannot reach. A tanh gate
    bounded in [-delta_scale, +delta_scale] covers identity, anti-SRP,
    projection, and reflection in a single parameterization.

  - **Zero-init only on the OUTPUT layer:** hidden-layer weights stay
    Kaiming (default) so gradients flow into the gate network as soon
    as the output layer starts moving. Zero-init on output ensures
    initial beta_eff is exactly 0 regardless of random hidden weights.

The per-layer bias `b_{l,h}` is owned here (one TokenHeadGate per
attention layer), so the layer index never has to be threaded through.
The dead-path rule (by design) is enforced *outside* this module by
not instantiating it in the last attention block — the rule is an
architectural choice of the parent transformer, not the gate itself.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_GATE_OUTPUT_INITS = (
    "zero",
    "tiny_normal",
    "xavier_uniform",
    "kaiming_uniform",
    "orthogonal",
    "constant_beta",
)
_GATE_ACTIVATIONS = (
    "tanh",
    "scaled_sigmoid",
    "sigmoid01",
    "softsign",
    "hardtanh",
    "atan",
)
_GATE_FACTORIZATIONS = (
    "full",
    "token_only",
    "head_only",
    "no_bias",
)


def bounded_signed_activation(
    raw: torch.Tensor,
    mode: str = "tanh",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Map raw gate logits to a signed value in [-1, 1].

    The signed-gate SRP parameter is `beta_eff = delta_scale * g(raw)`.
    Keeping `g(raw)` bounded is important because the gate is allowed to
    choose anti-SRP and reflection-like regimes, but it should not
    produce unbounded projection strengths from a transient logit spike.
    """
    if temperature <= 0.0:
        raise ValueError(f"activation temperature must be positive, got {temperature}")
    x = raw / float(temperature)
    if mode == "tanh":
        return torch.tanh(x)
    if mode == "scaled_sigmoid":
        return 2.0 * torch.sigmoid(x) - 1.0
    if mode == "sigmoid01":
        return torch.sigmoid(x)
    if mode == "softsign":
        return F.softsign(x)
    if mode == "hardtanh":
        return F.hardtanh(x, min_val=-1.0, max_val=1.0)
    if mode == "atan":
        return (2.0 / math.pi) * torch.atan(x)
    raise ValueError(f"unknown gate activation {mode!r}; expected {_GATE_ACTIVATIONS}")


def _inverse_bounded_activation(value: float, mode: str) -> float:
    """Return a raw logit whose bounded activation equals `value`.

    Used only for the constant-beta initialization arm. `value` is the
    normalized beta target (`beta0 / delta_scale`), so it must live
    inside (-1, 1). Endpoints are deliberately rejected because all
    smooth activations would require infinite logits there.
    """
    if not -1.0 < value < 1.0:
        raise ValueError(
            "constant_beta init requires beta0 / delta_scale to be in "
            f"(-1, 1), got {value}"
        )
    if mode == "tanh":
        return math.atanh(value)
    if mode == "scaled_sigmoid":
        return math.log((1.0 + value) / (1.0 - value))
    if mode == "sigmoid01":
        if not 0.0 < value < 1.0:
            raise ValueError(
                "constant_beta init with sigmoid01 requires beta0 / "
                f"delta_scale to be in (0, 1), got {value}"
            )
        return math.log(value / (1.0 - value))
    if mode == "softsign":
        return value / (1.0 - abs(value))
    if mode == "hardtanh":
        return value
    if mode == "atan":
        return math.tan((math.pi / 2.0) * value)
    raise ValueError(f"unknown gate activation {mode!r}; expected {_GATE_ACTIVATIONS}")


class TokenHeadGate(nn.Module):
    """
    Per-(token, head) signed gate with identity initialisation.

    Inputs to forward:
      token_diag (B, N, T_token)
        Per-token diagnostics, shared across heads. Suggested:
          [h_local, neighbour_count/8, norm_r_mean, local_variance, ...]
        See the released protocol for the full list. The exact set is dataset-
        specific; only the dimensionality T_token must match what was
        passed at construction.
      head_diag (B, H, N, T_head)
        Per-head diagnostics. Suggested:
          [cos(y, r_hat), |cos(y, r_hat)|, log_norm_y, ...]

    Output:
      beta_eff (B, H, N, 1)
        Per-(token, head) effective beta, bounded in
        [-delta_scale, +delta_scale]. Multiply the standard SRP correction
        term (y · r_hat) * r_hat by this beta_eff to obtain the gated
        projection.

    Initialisation invariant:
      All output-path parameters (token_mlp_out W and b, head linear
      weight and bias, layer-head bias) are zero. Hidden-layer weights
      stay Kaiming. Result: at step 1, raw_logit = 0 everywhere, so
      beta_eff = delta_scale * tanh(0) = 0, and the SRP correction
      vanishes — the model is exactly the un-projected baseline.
    """

    def __init__(
        self,
        num_heads: int,
        num_token_features: int,
        num_head_features: int,
        hidden_dim: int = 16,
        delta_scale: float = 2.0,
        output_init: str = "zero",
        output_init_scale: float = 1.0,
        init_beta0: float = 0.0,
        activation: str = "tanh",
        activation_temperature: float = 1.0,
        factorization: str = "full",
    ) -> None:
        super().__init__()
        # Validation: public CLI/config validation must
        # raise a real exception (kept under `python -O`).
        if delta_scale <= 0.0:
            raise ValueError(f"delta_scale must be positive, got {delta_scale}")
        if output_init not in _GATE_OUTPUT_INITS:
            raise ValueError(
                f"output_init must be one of {_GATE_OUTPUT_INITS}, got {output_init!r}"
            )
        if output_init_scale < 0.0:
            raise ValueError(
                f"output_init_scale must be non-negative, got {output_init_scale}"
            )
        if activation not in _GATE_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_GATE_ACTIVATIONS}, got {activation!r}"
            )
        if factorization not in _GATE_FACTORIZATIONS:
            raise ValueError(
                f"factorization must be one of {_GATE_FACTORIZATIONS}, got {factorization!r}"
            )
        if activation_temperature <= 0.0:
            raise ValueError(
                "activation_temperature must be positive, got "
                f"{activation_temperature}"
            )
        self.num_heads = num_heads
        self.num_token_features = num_token_features
        self.num_head_features = num_head_features
        self.hidden_dim = hidden_dim
        self.delta_scale = float(delta_scale)
        self.output_init = output_init
        self.output_init_scale = float(output_init_scale)
        self.init_beta0 = float(init_beta0)
        self.activation = activation
        self.activation_temperature = float(activation_temperature)
        self.factorization = factorization
        self.use_token_term = factorization in ("full", "token_only", "no_bias")
        self.use_head_term = factorization in ("full", "head_only", "no_bias")
        self.use_bias_term = factorization != "no_bias"

        if self.use_token_term:
            # Shared token MLP: 2 linear layers with GELU. The HIDDEN layer
            # uses default (Kaiming) init so gradients can flow once the
            # output layer leaves zero. The OUTPUT layer is zero-init below.
            self.token_mlp_hidden = nn.Linear(
                num_token_features,
                hidden_dim,
                bias=self.use_bias_term,
            )
            self.token_mlp_out = nn.Linear(
                hidden_dim,
                1,
                bias=self.use_bias_term,
            )
        else:
            self.token_mlp_hidden = None
            self.token_mlp_out = None

        if self.use_head_term:
            # Per-head linear: each head has its own (T_h,)-vector of weights
            # and a scalar bias, applied to head_diag's per-head feature row.
            # Parameterised as a 2D weight tensor of shape (H, T_h) so we can
            # batch-apply with einsum without a Python head loop.
            self.head_weight = nn.Parameter(
                torch.zeros(num_heads, num_head_features),
            )
        else:
            self.register_buffer(
                "head_weight",
                torch.zeros(num_heads, num_head_features),
                persistent=False,
            )
        if self.use_bias_term:
            self.head_bias = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_buffer(
                "head_bias", torch.zeros(num_heads), persistent=False,
            )

        # Per-(layer, head) bias. One TokenHeadGate is instantiated per
        # attention layer, so this captures both axes implicitly: the
        # outer module owns the layer dimension by holding one of these
        # gates per layer, and `b_l_h` here is the per-head bias *for
        # this layer*. Shared across all tokens within the layer.
        if self.use_bias_term:
            self.layer_head_bias = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_buffer(
                "layer_head_bias", torch.zeros(num_heads), persistent=False,
            )

        # Output-path init. Apply AFTER nn.Linear's default Kaiming init
        # has run, so hidden-layer weights are random while the output
        # path follows the requested ablation arm.
        self.reset_output_path()

    def reset_output_path(self) -> None:
        """Apply the configured output-path initialization.

        Parent models run broad `_init_weights()` passes after module
        construction. They must call this method afterwards so gate
        output-path ablations survive those generic initialization
        routines. Defaults are exact zero identity, matching the
        original signed-gate behavior.
        """
        with torch.no_grad():
            if self.token_mlp_out is not None:
                self.token_mlp_out.weight.zero_()
                if self.token_mlp_out.bias is not None:
                    self.token_mlp_out.bias.zero_()
            self.head_weight.zero_()
            self.head_bias.zero_()
            self.layer_head_bias.zero_()

            if self.output_init == "zero":
                return
            if self.output_init == "tiny_normal":
                if self.token_mlp_out is not None:
                    nn.init.normal_(
                        self.token_mlp_out.weight,
                        mean=0.0,
                        std=self.output_init_scale,
                    )
                if self.use_head_term:
                    nn.init.normal_(
                        self.head_weight,
                        mean=0.0,
                        std=self.output_init_scale,
                    )
                return
            if self.output_init == "xavier_uniform":
                if self.token_mlp_out is not None:
                    nn.init.xavier_uniform_(self.token_mlp_out.weight)
                    self.token_mlp_out.weight.mul_(self.output_init_scale)
                if self.use_head_term:
                    nn.init.xavier_uniform_(self.head_weight)
                    self.head_weight.mul_(self.output_init_scale)
                return
            if self.output_init == "kaiming_uniform":
                if self.token_mlp_out is not None:
                    nn.init.kaiming_uniform_(self.token_mlp_out.weight, a=math.sqrt(5))
                    self.token_mlp_out.weight.mul_(self.output_init_scale)
                if self.use_head_term:
                    nn.init.kaiming_uniform_(self.head_weight, a=math.sqrt(5))
                    self.head_weight.mul_(self.output_init_scale)
                return
            if self.output_init == "orthogonal":
                if self.token_mlp_out is not None:
                    nn.init.orthogonal_(self.token_mlp_out.weight)
                    self.token_mlp_out.weight.mul_(self.output_init_scale)
                if self.use_head_term:
                    nn.init.orthogonal_(self.head_weight)
                    self.head_weight.mul_(self.output_init_scale)
                return
            if self.output_init == "constant_beta":
                if not self.use_bias_term:
                    raise ValueError(
                        "constant_beta gate initialization requires a bias "
                        "term and is incompatible with factorization='no_bias'"
                    )
                raw = _inverse_bounded_activation(
                    self.init_beta0 / self.delta_scale,
                    self.activation,
                )
                # Layer/head bias is the only non-zero output-path term,
                # so every token/head starts at the same beta value.
                self.layer_head_bias.fill_(raw)
                return
            raise RuntimeError(f"unhandled gate output init {self.output_init!r}")

    def reset_identity(self) -> None:
        """
        Zero-init all output-path parameters so the gate emits
        beta_eff = 0 everywhere at construction (by design).

        This remains available for tests and exact-identity probes.
        Parent modules should normally call `reset_output_path()` after
        their broad `_init_weights()` pass so non-zero initialization
        ablation arms are preserved.

        Hidden-layer weights are NOT reset here: they should stay at
        whatever the parent module's init scheme produces (typically
        Kaiming or trunc_normal, both of which are fine — gradients
        will flow into the hidden layer once the output starts moving).
        """
        with torch.no_grad():
            if self.token_mlp_out is not None:
                self.token_mlp_out.weight.zero_()
                if self.token_mlp_out.bias is not None:
                    self.token_mlp_out.bias.zero_()
            self.head_weight.zero_()
            self.head_bias.zero_()
            self.layer_head_bias.zero_()

    def forward(
        self,
        token_diag: torch.Tensor,
        head_diag: torch.Tensor,
    ) -> torch.Tensor:
        """
        token_diag (B, N, T_token)
        head_diag  (B, H, N, T_head)

        Returns beta_eff (B, H, N, 1).
        """
        B, N, Tt = token_diag.shape
        Bh, H, Nh, Th = head_diag.shape
        # Validation: runtime tensor-shape contracts must
        # raise (kept under `python -O`). Cheaper than letting downstream
        # einsum blow up with cryptic messages.
        if B != Bh or N != Nh:
            raise ValueError(
                f"token_diag and head_diag must agree on (B, N): "
                f"got {(B, N)} vs {(Bh, Nh)}"
            )
        if H != self.num_heads:
            raise ValueError(
                f"head_diag has H={H}, gate expects num_heads={self.num_heads}"
            )
        if Tt != self.num_token_features:
            raise ValueError(
                f"token_diag has T_token={Tt}, gate expects "
                f"num_token_features={self.num_token_features}"
            )
        if Th != self.num_head_features:
            raise ValueError(
                f"head_diag has T_head={Th}, gate expects "
                f"num_head_features={self.num_head_features}"
            )

        if self.use_token_term:
            # Token MLP: (B, N, T_t) -> (B, N, hidden) -> (B, N, 1)
            u = self.token_mlp_out(F.gelu(self.token_mlp_hidden(token_diag)))
            # Broadcast token logit u across heads: (B, 1, N, 1).
            u = u.unsqueeze(1)
        else:
            u = token_diag.new_zeros(B, 1, N, 1)

        # Per-head linear: d_{b,h,n} = sum_t head_diag[b,h,n,t] * head_weight[h,t]
        # einsum keeps everything on-device and avoids a Python head loop.
        if self.use_head_term:
            d = torch.einsum(
                "bhnt,ht->bhn", head_diag, self.head_weight,
            )
            if self.use_bias_term:
                d = d + self.head_bias.view(1, H, 1)
            d = d.unsqueeze(-1)                            # (B, H, N, 1)
        else:
            d = token_diag.new_zeros(B, H, N, 1)

        # Per-(layer, head) bias broadcast across batch and tokens.
        b = self.layer_head_bias.view(1, H, 1, 1)

        # u: (B, 1, N, 1) -> broadcasts across heads with d's H.
        raw = u + d + b                                     # (B, H, N, 1)
        gate = bounded_signed_activation(
            raw,
            mode=self.activation,
            temperature=self.activation_temperature,
        )
        beta_eff = self.delta_scale * gate
        return beta_eff


class GateStatsAccumulator:
    """
    Per-example accumulator for signed-gate diagnostics across an
    eval pass. Use case: run a full validation/test pass and persist
    the gate's behaviour distribution (β_eff per block per example,
    summary stats, h_local correlations) into the run's
    test_artifacts.npz.

    Why this matters: without per-example accumulation, the only gate
    record left after training is the LAST eval batch's stats — the
    gate diagnostics are designed for cross-class /
    cross-provider / cross-length stratification, which requires
    per-example records, not single-batch snapshots.

    Usage pattern (per trainer):

        acc = GateStatsAccumulator()
        for batch in eval_loader:
            logits = model(...)              # populates each block's
                                             # _last_gate_stats cache
            acc.update(model)                # snapshot per example

        gate_stats = acc.finalize()          # returns dict of arrays
        np.savez(..., gate_stats=gate_stats)

    The accumulator is no-op when no gate-active blocks exist (i.e.,
    the model is not in a signed-gate mode), so callers can blindly
    invoke it without branching on mode.

    Per-example storage cost: ~13 floats × num_active_blocks per
    example. For PANDA (≈ 2121 val slides × 3 active blocks) that's
    ~80 KB per run — negligible compared to the existing logits
    payload.

    The accumulator stores PER-(BLOCK, EXAMPLE) summaries rather than
    full β_eff tensors because:
      - β_eff has shape (B, H, N, 1); N varies per example on PANDA
        (native length), so storing full tensors would require either
        ragged arrays or padding.
      - Downstream analyses operate on SUMMARY statistics anyway —
        mean / std / sign-bin fractions / h_local correlation. The
        full β_eff distribution can be recovered from those summaries
        at the precision needed for cross-task taxonomy claims.

    Conventions:
      - block_index keys are integer (the position in model.blocks).
        Last (dead-path) blocks are skipped automatically because
        their _last_gate_stats stays None.
      - The "frac_*" stats partition β_eff into 4 bins:
            neg         : β_eff <= -0.5         (anti-SRP regime)
            near_zero   : -0.5 < β_eff < 0.5    (effective identity)
            projection  :  0.5 <= β_eff <= 1.5  (≈ projection regime)
            reflection  :  β_eff > 1.5          (≈ reflection regime)
        Bins reflect the proposal's β-regime semantics directly so
        post-hoc stratification doesn't need to re-discretize.
    """

    def __init__(self) -> None:
        # Per-block list of per-example dicts. block_index → list of
        # per-example summary dicts in arrival order.
        self._records: dict[int, list[dict]] = {}

    def update(self, model: nn.Module) -> None:
        """Snapshot the current `_last_gate_stats` for every gate-active
        attention layer in `model`. Idempotent across the eval batch
        index — call once per batch."""
        if not hasattr(model, "blocks"):
            return
        for i, blk in enumerate(model.blocks):
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            stats = getattr(attn, "_last_gate_stats", None)
            if stats is None:
                continue
            beta_eff = stats["beta_eff"].float().detach().cpu()  # (B, H, N, 1)
            # Reduce to per-example summaries. Each example's β_eff is
            # (H, N, 1); collapse to a flat vector for the bin / mean / std.
            B = beta_eff.shape[0]
            for b in range(B):
                be = beta_eff[b].flatten()                       # (H*N,)
                n = be.numel()
                if n == 0:
                    # Some slide pipelines can surface an empty example after
                    # filtering. Keep one NaN-filled record so downstream
                    # arrays preserve per-example alignment instead of losing
                    # the completed run during final gate-stat collection.
                    rec = {
                        "beta_eff_mean":      float("nan"),
                        "beta_eff_std":       float("nan"),
                        "beta_eff_min":       float("nan"),
                        "beta_eff_max":       float("nan"),
                        "frac_neg":           float("nan"),
                        "frac_near_zero":     float("nan"),
                        "frac_projection":    float("nan"),
                        "frac_reflection":    float("nan"),
                    }
                    if "cos_yr" in stats:
                        rec["mean_cos_yr"] = float("nan")
                    if "h_local" in stats:
                        rec["corr_beta_h_local"] = float("nan")
                    self._records.setdefault(i, []).append(rec)
                    continue
                rec = {
                    "beta_eff_mean":      float(be.mean().item()),
                    "beta_eff_std":       float(be.std().item()) if n > 1 else 0.0,
                    "beta_eff_min":       float(be.min().item()),
                    "beta_eff_max":       float(be.max().item()),
                    "frac_neg":           float((be <= -0.5).sum().item() / n),
                    "frac_near_zero":     float(((be > -0.5) & (be < 0.5)).sum().item() / n),
                    "frac_projection":    float(((be >= 0.5) & (be <= 1.5)).sum().item() / n),
                    "frac_reflection":    float((be > 1.5).sum().item() / n),
                }
                if "cos_yr" in stats:
                    cos_yr = stats["cos_yr"].float().detach().cpu()
                    rec["mean_cos_yr"] = float(cos_yr[b].mean().item())
                # Pearson corr(β_eff_per_token, h_local_per_token) within
                # this example. β_eff is (H, N, 1) → mean over heads
                # gives (N,); h_local is (N,). corr is informative only
                # when both have variance.
                #
                # IMPORTANT: the Pearson formula is
                #     r = sum((x - x_mean)(y - y_mean))
                #         / (||x - x_mean|| * ||y - y_mean||)
                # i.e. the denominator uses *centered* norms. Using the
                # raw `x.norm() * y.norm()` (uncentered) biases r
                # toward smaller magnitudes, particularly when either
                # vector has a non-zero mean. This was a real bug in
                # the first implementation; the fix mirrors the W&B
                # helper in slide_level_srp/train_panda.py::gate_diagnostics.
                if "h_local" in stats:
                    hl = stats["h_local"][b].float().detach().cpu().flatten()
                    be_pt = beta_eff[b].mean(dim=0).squeeze(-1)  # (N,)
                    if (hl.numel() > 1
                        and be_pt.std() > 1e-8
                        and hl.std() > 1e-8):
                        be_c = be_pt - be_pt.mean()
                        hl_c = hl - hl.mean()
                        rec["corr_beta_h_local"] = float(
                            (be_c * hl_c).sum().item()
                            / (be_c.norm().item() * hl_c.norm().item() + 1e-12)
                        )
                    else:
                        rec["corr_beta_h_local"] = 0.0
                self._records.setdefault(i, []).append(rec)

    def finalize(self) -> dict:
        """Return a flat dict suitable for `np.savez` / `**kwargs`. Keys
        are flattened to `block{i}_{stat}` so a downstream reader can
        index without reconstructing the nested structure.

        Returns an empty dict when no gate stats were ever recorded
        (i.e., the model was not in a signed-gate mode).
        """
        import numpy as _np
        out: dict = {}
        if not self._records:
            return out
        for block_i, recs in sorted(self._records.items()):
            if not recs:
                continue
            stat_names = list(recs[0].keys())
            for stat in stat_names:
                vals = _np.array(
                    [r.get(stat, _np.nan) for r in recs],
                    dtype=_np.float32,
                )
                out[f"block{block_i}_{stat}"] = vals
        # Also save the block list so post-hoc readers know which
        # block indices were captured (useful when depth or
        # gate-active mask changes between runs).
        out["captured_block_indices"] = _np.array(
            sorted(self._records.keys()), dtype=_np.int64,
        )
        return out


def signed_gate_step_summary(model: nn.Module) -> dict[str, float]:
    """
    Lightweight per-step gate trajectory snapshot.

    Returns a small flat dict of scalars suitable for direct
    W&B logging during training, e.g.::

        {"gate_step/block0/beta_mean": ..., "gate_step/block0/frac_neg": ...}

    Reads each gate-active attention layer's `_last_gate_stats` cache
    (populated during the most recent forward) and computes:
      - beta_mean, beta_std       (overall β_eff distribution)
      - frac_neg                  (β_eff <= -0.5)
      - frac_near_zero            (-0.5 < β_eff < 0.5)
      - frac_projection           (0.5 <= β_eff <= 1.5)
      - frac_reflection           (β_eff > 1.5)

    Bins match `GateStatsAccumulator` for cross-tool comparability.
    Cost is O(B·H·N) GPU reductions per gate-active block per call —
    callers should fire this every ~25 optimizer steps, not every step.

    Returns an empty dict when no gate is active (signed_gated mode is
    off, or none of the blocks have populated `_last_gate_stats`),
    making this safe to call unconditionally.

    Why this exists: scalar α/β trajectory logs do not describe signed-gate
    motion. This helper records compact per-step summaries so gate movement
    can be checked without changing the training loop or saving large tensors.
    """
    out: dict[str, float] = {}
    if not hasattr(model, "blocks"):
        return out
    for i, blk in enumerate(model.blocks):
        attn = getattr(blk, "attn", None)
        if attn is None:
            continue
        stats = getattr(attn, "_last_gate_stats", None)
        if stats is None:
            continue
        beta_eff = stats.get("beta_eff")
        if beta_eff is None:
            continue
        # GPU-side reductions on the cached tensor; .float() is cheap
        # because beta_eff is already small (B*H*N elements per block).
        b = beta_eff.detach().float()
        beta_mean = float(b.mean().item())
        beta_std = float(b.std().item()) if b.numel() > 1 else 0.0
        frac_neg = float((b <= -0.5).float().mean().item())
        frac_near = float(((b > -0.5) & (b < 0.5)).float().mean().item())
        frac_proj = float(((b >= 0.5) & (b <= 1.5)).float().mean().item())
        frac_refl = float((b > 1.5).float().mean().item())
        prefix = f"gate_step/block{i}"
        out[f"{prefix}/beta_mean"] = beta_mean
        out[f"{prefix}/beta_std"] = beta_std
        out[f"{prefix}/frac_neg"] = frac_neg
        out[f"{prefix}/frac_near_zero"] = frac_near
        out[f"{prefix}/frac_projection"] = frac_proj
        out[f"{prefix}/frac_reflection"] = frac_refl
    return out


def collect_gate_module_ids(model: nn.Module) -> set[int]:
    """
    Return the `id()` of every nn.Module inside any TokenHeadGate in
    `model`, including the TokenHeadGate itself. Used by parent
    modules' _init_weights to SKIP gate-internal modules during their
    generic trunc_normal pass.

    Why this matters: the parent's _init_weights typically does

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

    which walks every Linear including the gate's `token_mlp_hidden`
    and `token_mlp_out`. Each `trunc_normal_` call consumes the global
    RNG, so non-gate Linears later in the iteration order receive
    different random draws than they would in a baseline build at the same
    seed, contaminating same-seed paired comparisons.

    The fix is to leave the gate's submodules at their PyTorch-default
    init (Kaiming for nn.Linear) and let the gate's own
    `reset_output_path()` apply the requested output-path ablation.
    The parent's trunc_normal pass then touches exactly the same modules
    in baseline and signed-gated builds, so non-gate weights are
    bit-identical.
    """
    out: set[int] = set()
    for m in model.modules():
        if isinstance(m, TokenHeadGate):
            for sub in m.modules():
                out.add(id(sub))
    return out


def make_token_diag(
    h_local: torch.Tensor | None,
    neighbour_count: torch.Tensor | None,
    norm_r_mean: torch.Tensor | None,
    local_var: torch.Tensor | None,
    extras: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Helper to assemble per-token diagnostics into a (B, N, T_token) tensor.

    Each input is (B, N) or (B, N, 1); None inputs are skipped (their
    feature column is dropped). `extras` is a list of arbitrary (B, N) or
    (B, N, k) tensors appended after the named features. The caller must
    construct the gate with `num_token_features` matching the actual
    number of features assembled here.

    Caller responsibility:
      - All inputs must be on the same device and dtype.
      - All inputs must agree on (B, N).
      - `neighbour_count` should already be normalised (e.g. divided by
        8) — this helper does no rescaling.
    """
    cols: list[torch.Tensor] = []

    def _add(t: torch.Tensor | None) -> None:
        if t is None:
            return
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        cols.append(t)

    _add(h_local)
    _add(neighbour_count)
    _add(norm_r_mean)
    _add(local_var)
    if extras:
        for e in extras:
            _add(e)

    # Validation: data-contract validation must
    # raise ValueError so it survives `python -O` (asserts are stripped).
    if not cols:
        raise ValueError("at least one token-level diagnostic must be provided")
    return torch.cat(cols, dim=-1)


def make_head_diag(
    cos_yr: torch.Tensor,
    abs_cos_yr: torch.Tensor | None = None,
    log_norm_y: torch.Tensor | None = None,
    extras: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Helper to assemble per-head diagnostics into a (B, H, N, T_head)
    tensor.

    Each input must be (B, H, N) or (B, H, N, 1). `abs_cos_yr` and
    `log_norm_y` default to derived values if not provided:
      abs_cos_yr = cos_yr.abs()
      log_norm_y is left out (caller must provide if wanted)
    """
    cols: list[torch.Tensor] = []

    def _add(t: torch.Tensor | None) -> None:
        if t is None:
            return
        if t.dim() == 3:
            t = t.unsqueeze(-1)
        cols.append(t)

    _add(cos_yr)
    if abs_cos_yr is None:
        # Common case: derive from cos_yr to save the caller a step.
        abs_cos_yr = cos_yr.abs()
    _add(abs_cos_yr)
    _add(log_norm_y)
    if extras:
        for e in extras:
            _add(e)

    return torch.cat(cols, dim=-1)
