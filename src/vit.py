"""
ViT-Small/16 for patch-level pathology experiments.

Standard pre-norm ViT architecture:
  Embed -> [Block * depth] -> LayerNorm -> Linear(num_classes)

Each Block is:
  x_after_attn = x + Attention(LN(x))
  x_out        = x_after_attn + MLP(LN(x_after_attn))

The Attention module is our XSAAttention (src.xsa_attention), which
applies standard softmax attention followed by the per-role alpha-scaled
self-component projection (XSA).

Capture hooks:
  Each Block exposes a `_capture_cls_pipeline` flag. When set to True,
  the block stores the CLS-token vector at three stages:
    - cls_before_attn: the pre-block input CLS vector (before LN1/attn)
    - cls_after_attn:  after the attention sublayer (with residual), i.e.
                       (x + attn(LN(x)))[:, 0]
    - cls_after_block: after the MLP sublayer (with residual), i.e. x_out[:, 0]
  These are used by src.diagnostics to compute per-layer cosine similarity
  curves for CLS evolution.

This file intentionally reimplements the small ViT stack so the attention
backends, residual path, and CLS-token path are explicit in one place.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .xsa_attention import XSAAttention
# Optional SRP attention path. Only instantiated when a Block is built with
# use_srp=True; the import has no runtime effect if SRP is not used.
from .srp_patch_attention import PatchSRPAttention


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic depth (a.k.a. DropPath). Zeros the ENTIRE residual-branch
    output for a random subset of the batch, scaling the surviving branch
    outputs by 1 / (1 - drop_prob) to preserve expected value.

    Notes:
      - Per-sample (not per-token): a single Bernoulli draw per batch row.
      - Only active when training. In eval mode this is identity, so val/test
        metrics and diagnostic captures are unaffected.
      - drop_prob = 0 is identity in all modes.

    Intended to be applied to the output of a residual branch BEFORE the
    residual add, i.e.  `x = x + drop_path(branch(x))` -- the pass-through
    residual is always intact, which matters here because our story relies
    on the residual to carry x_i back after XSA removes its v_i-aligned
    component from the attention contribution.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Shape broadcasts across all non-batch dims: (B, 1, 1, ...) for whatever rank x has.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """
    Thin nn.Module wrapper around the functional `drop_path` above so it can
    be registered as a submodule of Block. Holds a per-layer drop probability
    set by ViT's linear depth schedule.
    """
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"


class Mlp(nn.Module):
    """Two-layer MLP with GELU activation; standard transformer feed-forward."""
    def __init__(self, in_dim: int, hidden_dim: int | None = None,
                 out_dim: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    """Pre-norm transformer block: LN -> XSAAttention -> residual -> LN -> MLP -> residual.

    Drop-path (stochastic depth) is applied to BOTH residual-branch outputs
    before they are summed back into the residual stream. The residual
    pass-through itself is never dropped.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        alpha_cls_mode: str = "zero",
        alpha_patch_mode: str = "zero",
        alpha_init: float = 1.0,
        num_cls_tokens: int = 1,
        mask_diagonal: bool = False,
        drop_path: float = 0.0,
        # --- SRP attention extension --------------------------------
        # When use_srp is True, the attention module is PatchSRPAttention
        # (reflection over 3×3 spatial neighborhood mean) instead of
        # XSAAttention (self-direction projection). The two are mutually
        # exclusive per-block; XSA-specific params (alpha_*, mask_diagonal)
        # are ignored when use_srp=True, and SRP-specific params
        # (beta_*) are ignored when use_srp=False. Default is XSA to
        # preserve the existing baseline/XSA behavior.
        use_srp: bool = False,
        beta_patch_mode: str = "fixed",
        beta_init: float = 2.0,
        grid_h: int = 14,
        grid_w: int = 14,
        # Signed-gate options forwarded to PatchSRPAttention. Only
        # consumed under use_srp=True + beta_patch_mode='signed_gated'.
        delta_scale: float = 2.0,
        gate_active: bool = True,
        gate_hidden_dim: int = 16,
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
        # Method 2.1/2.4 RCD controls.  Inert unless
        # beta_patch_mode is "rcd" or "rcd_learned_r".
        rcd_adapter_kind: str = "lowrank",
        rcd_rank: int = 16,
        learned_r_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.use_srp = use_srp
        self.beta_patch_mode = beta_patch_mode
        self.norm1 = nn.LayerNorm(dim)
        if use_srp:
            self.attn = PatchSRPAttention(
                dim=dim, num_heads=num_heads,
                grid_h=grid_h, grid_w=grid_w,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=proj_drop,
                beta_patch_mode=beta_patch_mode,
                beta_init=beta_init,
                num_cls_tokens=num_cls_tokens,
                delta_scale=delta_scale,
                gate_active=gate_active,
                gate_hidden_dim=gate_hidden_dim,
                detach_gate_inputs=detach_gate_inputs,
                neighbor_radius=neighbor_radius,
                neighbor_shell=neighbor_shell,
                neighbor_source=neighbor_source,
                neighbor_shuffle_seed=neighbor_shuffle_seed,
                neighbor_weighting=neighbor_weighting,
                neighbor_weight_sigma=neighbor_weight_sigma,
                gate_output_init=gate_output_init,
                gate_output_init_scale=gate_output_init_scale,
                gate_init_beta0=gate_init_beta0,
                gate_activation=gate_activation,
                gate_activation_temperature=gate_activation_temperature,
                gate_count_features=gate_count_features,
                srp_gate_placement=srp_gate_placement,
                rcd_adapter_kind=rcd_adapter_kind,
                rcd_rank=rcd_rank,
                learned_r_hidden_dim=learned_r_hidden_dim,
            )
        else:
            self.attn = XSAAttention(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=proj_drop,
                alpha_cls_mode=alpha_cls_mode,
                alpha_patch_mode=alpha_patch_mode,
                alpha_init=alpha_init,
                num_cls_tokens=num_cls_tokens,
                mask_diagonal=mask_diagonal,
            )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_dim=dim, hidden_dim=int(dim * mlp_ratio), drop=proj_drop)
        # One DropPath instance per residual branch. drop_path=0.0 is a
        # cheap identity at forward time.
        self.drop_path = DropPath(drop_path)

        # Diagnostic capture toggle; when enabled, stores the CLS trajectory
        # within this block at three checkpoints (see module docstring).
        self._capture_cls_pipeline = False
        self.last_cls_states: dict | None = None

    def forward(
        self,
        x: torch.Tensor,
        h_local: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention sublayer with residual + drop-path on the branch output.
        # h_local is only consumed by PatchSRPAttention under signed-gated
        # or learned-r RCD mode; the legacy XSAAttention path ignores it.
        if self.use_srp and self.beta_patch_mode in ("signed_gated", "rcd_learned_r"):
            attn_out = self.attn(self.norm1(x), h_local=h_local)
        else:
            attn_out = self.attn(self.norm1(x))
        x_after_attn = x + self.drop_path(attn_out)
        # MLP sublayer with residual + drop-path on the branch output.
        mlp_out = self.mlp(self.norm2(x_after_attn))
        x_out = x_after_attn + self.drop_path(mlp_out)

        if self._capture_cls_pipeline:
            # Store CLS vectors only (position 0). Detach to avoid graph retention.
            self.last_cls_states = {
                "cls_before_attn": x[:, 0].detach(),
                "cls_after_attn": x_after_attn[:, 0].detach(),
                "cls_after_block": x_out[:, 0].detach(),
            }
        return x_out


class PatchEmbed(nn.Module):
    """
    Convolutional patch embedding (equivalent to non-overlapping linear patch
    projection). Maps (B, 3, 224, 224) -> (B, 196, 384) for patch_size=16.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 384) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, E, H/P, W/P) -> (B, H/P * W/P, E)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    """
    ViT-Small/16 with XSAAttention blocks.

    Defaults match ViT-S as commonly configured:
      embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, patch=16.

    XSA-related arguments (alpha_cls_mode, alpha_patch_mode, alpha_init) are
    forwarded uniformly to every block.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 9,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        alpha_cls_mode: str = "zero",
        alpha_patch_mode: str = "zero",
        alpha_init: float = 1.0,
        mask_diagonal: bool = False,
        drop_path_rate: float = 0.0,
        # --- SRP attention extension (see src/srp_patch_attention.py) ---
        use_srp: bool = False,
        beta_patch_mode: str = "fixed",
        beta_init: float = 2.0,
        # --- Signed-gate extension ----------------------------------
        delta_scale: float = 2.0,
        gate_hidden_dim: int = 16,
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
        # Method 2.1/2.4 RCD controls.  Inert unless
        # beta_patch_mode is "rcd" or "rcd_learned_r".
        rcd_adapter_kind: str = "lowrank",
        rcd_rank: int = 16,
        learned_r_hidden_dim: int = 16,
        use_abs_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        self.use_srp = use_srp
        self.beta_patch_mode = beta_patch_mode
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.use_abs_pos_embed = bool(use_abs_pos_embed)
        # One learned CLS token. If this is changed, XSAAttention.num_cls_tokens
        # must match (see below where we pass it through).
        self.num_cls_tokens = 1
        # Diagonal masking (a_{i,i}=0) is orthogonal to XSA's alpha modes.
        # The mask_all control variant uses mask_diagonal=True with alpha=0,
        # so that no XSA projection is applied, only the direct-self-attention
        # block. Set uniformly across all blocks.
        self.mask_diagonal = mask_diagonal
        # Drop-path schedule (stochastic depth): linearly increasing from 0
        # at the first block to `drop_path_rate` at the last block. Standard
        # DeiT/ViT recipe -- shallower layers are kept intact more often so
        # the model still learns the low-level features reliably, while
        # deeper layers (where overfitting lives) get stronger regularization.
        self.drop_path_rate = drop_path_rate
        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist() if depth > 0 else []

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learned CLS token and absolute 1D positional embedding over (CLS + patches).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_cls_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Patch grid dims for SRP neighbor lookups. For a square img_size
        # divisible by patch_size, grid is sqrt(num_patches) each side.
        grid_side = img_size // patch_size
        # Dead-path rule: for post-aggregation and pre-Q,
        # the LAST block's patch-only gate has no downstream CLS consumer.
        # Pre-K is different: the final CLS query attends to edited patch
        # keys in the same block, so its last-block gate remains live.
        final_block_gate_live = srp_gate_placement == "pre_k"
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                alpha_cls_mode=alpha_cls_mode,
                alpha_patch_mode=alpha_patch_mode,
                alpha_init=alpha_init,
                num_cls_tokens=self.num_cls_tokens,
                mask_diagonal=mask_diagonal,
                drop_path=dpr[i],
                use_srp=use_srp,
                beta_patch_mode=beta_patch_mode,
                beta_init=beta_init,
                grid_h=grid_side,
                grid_w=grid_side,
                delta_scale=delta_scale,
                gate_active=(i < depth - 1 or final_block_gate_live),
                gate_hidden_dim=gate_hidden_dim,
                detach_gate_inputs=detach_gate_inputs,
                neighbor_radius=neighbor_radius,
                neighbor_shell=neighbor_shell,
                neighbor_source=neighbor_source,
                neighbor_shuffle_seed=neighbor_shuffle_seed,
                neighbor_weighting=neighbor_weighting,
                neighbor_weight_sigma=neighbor_weight_sigma,
                gate_output_init=gate_output_init,
                gate_output_init_scale=gate_output_init_scale,
                gate_init_beta0=gate_init_beta0,
                gate_activation=gate_activation,
                gate_activation_temperature=gate_activation_temperature,
                gate_count_features=gate_count_features,
                srp_gate_placement=srp_gate_placement,
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
        # Standard ViT initialization: truncated-normal for positional/CLS and
        # all Linear weights; zeros for biases; LayerNorm gamma=1, beta=0.
        # This is the classic DeiT/timm init.
        from slide_level_srp.src.gate_signed import TokenHeadGate, collect_gate_module_ids
        # Skip gate-internal modules during the trunc_normal pass so
        # gate Linear init does NOT shift downstream non-gate weight
        # draws under the same seed. See
        # gate_signed.collect_gate_module_ids docstring for context.
        gate_module_ids = collect_gate_module_ids(self)
        from slide_level_srp.src.rcd_modules import collect_rcd_module_ids, reset_rcd_identity_modules
        rcd_module_ids = collect_rcd_module_ids(self)
        method_module_ids = gate_module_ids | rcd_module_ids
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
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
        # Belt-and-suspenders gate identity init.
        for m in self.modules():
            if isinstance(m, TokenHeadGate):
                m.reset_output_path()
        reset_rcd_identity_modules(self.modules())

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, 224, 224) -> (B, 196, 384) -> prepend CLS -> add pos embed -> blocks -> final LN
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        if self.use_abs_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # Compute h_local once per forward from layer-0 patch tokens. It is
        # detached so gradients do not flow through this diagnostic and shared
        # across blocks. It is computed only when the gate is actually used;
        # otherwise None is ignored downstream.
        h_local: torch.Tensor | None = None
        if self.use_srp and self.beta_patch_mode in ("signed_gated", "rcd_learned_r"):
            with torch.no_grad():
                # Use the FIRST block's static neighbour graph (all
                # blocks share the same grid). Patch tokens occupy
                # positions [1..1+num_patches).
                num_patches = self.blocks[0].attn.num_patches
                patch_tokens = x[:, 1:1 + num_patches, :].detach()  # (B, N_p, E)
                nbr_idx = self.blocks[0].attn.neighbor_index         # (N_p, 8)
                nbr_msk = self.blocks[0].attn.neighbor_mask          # (N_p, 8)
                pn = torch.nn.functional.normalize(
                    patch_tokens, dim=-1, eps=1e-12,
                )                                                    # (B, N_p, E)
                # Gather neighbours: clamp -1 idx to 0 for safe gather,
                # then mask out the invalid slots.
                safe_idx = nbr_idx.clamp(min=0)                      # (N_p, 8)
                nbr = pn[:, safe_idx, :]                             # (B, N_p, 8, E)
                cos = (nbr * pn[:, :, None, :]).sum(dim=-1)          # (B, N_p, 8)
                cos = cos * nbr_msk[None, :, :].to(cos.dtype)
                vc = nbr_msk.sum(dim=-1).clamp(min=1).to(cos.dtype)  # (N_p,)
                h_local = cos.sum(dim=-1) / vc                       # (B, N_p)
        for blk in self.blocks:
            x = blk(x, h_local=h_local)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        # Classification head reads the CLS token only (position 0).
        cls_out = x[:, 0]
        return self.head(cls_out)
