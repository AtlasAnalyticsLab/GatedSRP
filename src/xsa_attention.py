"""
XSA (Exclusive Self Attention) module with per-role adaptive alpha.

Given standard multi-head self-attention output Y (per head), XSA applies
a post-attention projection that partially or fully removes the component
of Y along each token's own value vector v_i:

    Z_i = Y_i - alpha_role(i) * ( (Y_i @ v_i) / ||v_i||^2 ) * v_i

Using Vn = v / ||v||, this is equivalent to

    Z = Y - alpha * sum(Y * Vn, dim=head_dim, keepdim=True) * Vn

alpha is selected per token based on role:
  - positions [0 .. num_cls_tokens-1] use alpha_cls
  - positions [num_cls_tokens .. N-1]   use alpha_patch

Each alpha is a vector of shape (num_heads,) so the model can decide per-head,
per-role, how much self-component to remove.

Three modes are supported for each role independently:
  "zero"  -> alpha fixed at 0.0  (standard self-attention, XSA disabled)
  "one"   -> alpha fixed at 1.0  (the paper's hard XSA)
  "learn" -> alpha is an nn.Parameter initialized at alpha_init (default 1.0)

Notes on stats capture:
  When self._capture_stats is True, the forward pass stashes four detached
  tensors into self.last_stats so downstream diagnostics code can compute:
      attn_probs:  (B, H, N, N)  -- needed for CLS self-attention a_{cls,cls}.
      cos_yv_pre:  (B, H, N)     -- cos(y_i, v_i) per token, BEFORE projection.
                                    This is the paper's "attention similarity
                                    bias" (cosine between attention output and
                                    token's own value). Central interpretability
                                    quantity: under alpha=0 this is what a
                                    standard SA produces; under alpha=1 the
                                    post-projection version is 0 by construction
                                    but the pre-projection value shows what the
                                    attention mechanism is still trying to do.
      cos_zv_post: (B, H, N)     -- cos(z_i, v_i) per token, AFTER projection.
                                    With alpha=1 this should be ~0 everywhere;
                                    with alpha in (0,1) it equals roughly
                                    (1-alpha_h) * cos_yv_pre per head; with
                                    learnable alpha its evolution tells us
                                    what the model chose.
      num_cls_tokens: (int)      -- recorded so diagnostics can split per token
                                    across CLS vs patch roles without needing
                                    a separate channel.
  Capturing is off by default; it should only be toggled during evaluation
  via src.diagnostics.set_capture_mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_ALPHA_MODES = ("zero", "one", "learn")


class XSAAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        alpha_cls_mode: str = "zero",
        alpha_patch_mode: str = "zero",
        alpha_init: float = 1.0,
        num_cls_tokens: int = 1,
        mask_diagonal: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert alpha_cls_mode in _ALPHA_MODES, f"alpha_cls_mode must be in {_ALPHA_MODES}"
        assert alpha_patch_mode in _ALPHA_MODES, f"alpha_patch_mode must be in {_ALPHA_MODES}"
        assert num_cls_tokens >= 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # Standard scaled dot-product attention scale; kept separate from XSA.
        self.scale = self.head_dim ** -0.5
        self.num_cls_tokens = num_cls_tokens
        self.alpha_cls_mode = alpha_cls_mode
        self.alpha_patch_mode = alpha_patch_mode
        # Diagonal masking (a_{i,i} = 0) is a "cheaper XSA" control:
        # it prevents direct self-attention but does NOT remove the
        # indirect self-component that XSA's orthogonal projection also
        # removes. Combined with alpha=0 on both roles, this gives the
        # pure diagonal-masking baseline for the mechanism-isolation run.
        # Can be combined orthogonally with any alpha configuration.
        self.mask_diagonal = mask_diagonal

        # Q, K, V are produced by a single linear; this is the standard fused form.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Register alpha either as Parameter (mode=="learn") or buffer (fixed).
        # Buffer is used for fixed modes so the value moves with .to(device)
        # but does NOT receive gradients. This keeps a uniform attribute name
        # (self.alpha_cls / self.alpha_patch) regardless of mode.
        self._make_alpha("alpha_cls", alpha_cls_mode, alpha_init, num_heads)
        self._make_alpha("alpha_patch", alpha_patch_mode, alpha_init, num_heads)

        # Stats capture toggle. Off by default; flipped on during eval.
        # When on, forward() stashes attention probabilities into self.last_stats.
        self._capture_stats = False
        self.last_stats: dict | None = None

    def _make_alpha(self, name: str, mode: str, init_val: float, n_heads: int) -> None:
        """Create self.<name> as Parameter or buffer depending on mode."""
        if mode == "learn":
            setattr(self, name, nn.Parameter(torch.full((n_heads,), float(init_val))))
        else:
            fixed = 0.0 if mode == "zero" else 1.0
            # register_buffer ensures correct device/dtype movement without grads.
            self.register_buffer(name, torch.full((n_heads,), fixed), persistent=True)

    def _build_alpha_per_token(self, N: int, device, dtype) -> torch.Tensor:
        """
        Concatenate alpha_cls and alpha_patch along the token axis.
        Returns tensor of shape (1, num_heads, N, 1) for broadcasting with
        per-head per-token scaling of the projection.
        """
        # alpha_cls, alpha_patch each have shape (num_heads,)
        n_cls = self.num_cls_tokens
        n_patch = N - n_cls

        # Shape (num_heads, num_cls_tokens) then (num_heads, num_patches)
        a_cls = self.alpha_cls.to(device=device, dtype=dtype).unsqueeze(-1).expand(-1, n_cls)
        a_patch = self.alpha_patch.to(device=device, dtype=dtype).unsqueeze(-1).expand(-1, n_patch)

        # Concat along the token axis: (num_heads, N)
        alpha = torch.cat([a_cls, a_patch], dim=1)
        # Broadcast shape for (B, num_heads, N, head_dim): (1, num_heads, N, 1)
        return alpha.unsqueeze(0).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)  where N = num_cls_tokens + num_patches and C = dim.
        Returns: (B, N, C)
        """
        B, N, C = x.shape

        # (B, N, 3C) -> (B, N, 3, H, D_head) -> (3, B, H, N, D_head)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, N, D_head)

        # Standard attention: use explicit softmax (not SDPA) so we can expose
        # attention probabilities for CLS diagnostics. Cost is modest at ViT-S
        # scale; we prefer transparency over a marginal speedup here.
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale           # (B, H, N, N)
        if self.mask_diagonal:
            # Set the diagonal to -inf BEFORE softmax, so a_{i,i} = 0
            # exactly after normalization. This is the "no direct
            # self-attention" baseline; XSA's projection (when active)
            # additionally removes indirect self-leakage via correlated v_j.
            # Single eye-mask allocated per forward; cost is negligible.
            eye = torch.eye(N, device=attn_logits.device, dtype=torch.bool)
            attn_logits = attn_logits.masked_fill(eye, float("-inf"))
        attn_probs = attn_logits.softmax(dim=-1)                       # (B, H, N, N)
        attn_probs_drop = self.attn_drop(attn_probs)
        y = attn_probs_drop @ v                                        # (B, H, N, D_head)

        # --- XSA projection step -------------------------------------------
        # Vn is v normalized to unit length along the head dimension.
        # F.normalize defaults to eps=1e-12 which avoids division by zero
        # when some v_i is (near-)zero; in that case Vn rows become ~0 and
        # the projection term becomes 0 (i.e. z_i = y_i), which is safe.
        vn = F.normalize(v, dim=-1)                                    # (B, H, N, D_head)
        # Dot product of y with Vn along head_dim: (B, H, N, 1)
        dot = (y * vn).sum(dim=-1, keepdim=True)

        # Per-token alpha: uses alpha_cls on CLS positions and alpha_patch
        # on patch positions. Shape (1, H, N, 1), broadcasts cleanly with
        # dot and vn.
        alpha = self._build_alpha_per_token(N, device=y.device, dtype=y.dtype)
        z = y - alpha * dot * vn                                       # (B, H, N, D_head)
        # -------------------------------------------------------------------

        # Capture diagnostics BEFORE reshaping heads away, since cos(y, v)
        # and the per-head norms are meaningful per head (dim=-1 = head_dim).
        if self._capture_stats:
            # cos(y, v): paper's central interpretability quantity (the
            # "attention similarity bias"). cos(z, v) verifies what XSA
            # actually removed. Both are per (batch, head, token) with
            # dim=-1 = head_dim. F.cosine_similarity uses eps=1e-8 by
            # default which is sufficient for the bf16/fp32 magnitudes
            # seen in practice here.
            cos_yv_pre = F.cosine_similarity(y, v, dim=-1).detach()    # (B, H, N)
            cos_zv_post = F.cosine_similarity(z, v, dim=-1).detach()   # (B, H, N)
            # Per-head L2 norms along head_dim. Together these give
            # magnitude diagnostics orthogonal to cos:
            # ||z||/||y|| = fraction of attention-output magnitude preserved
            # by the (possibly partial) XSA projection. Logged separately by
            # role in src/diagnostics.extract_batch_stats.
            y_norm = y.norm(dim=-1).detach()                            # (B, H, N)
            v_norm = v.norm(dim=-1).detach()                            # (B, H, N)
            z_norm = z.norm(dim=-1).detach()                            # (B, H, N)
            self.last_stats = {
                "attn_probs": attn_probs.detach(),
                "cos_yv_pre": cos_yv_pre,
                "cos_zv_post": cos_zv_post,
                "y_norm": y_norm,
                "v_norm": v_norm,
                "z_norm": z_norm,
                "num_cls_tokens": self.num_cls_tokens,
            }

        # Concatenate heads back to (B, N, C) and apply output projection.
        z = z.transpose(1, 2).reshape(B, N, C)
        out = self.proj(z)
        out = self.proj_drop(out)

        return out
