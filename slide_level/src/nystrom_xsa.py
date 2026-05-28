"""
Nyström self-attention + XSA (Exclusive Self Attention) projection.

This module combines two components:

  1. Nyström attention approximation, closely following Xiong et al. 2021
     ("Nyströmformer") and Shao et al. 2021 (TransMIL). The Nyström
     output y has the SAME shape (B, H, N, head_dim) as standard softmax
     attention -- only how y is computed differs. This matters because
     XSA (step 2) is a pure function of y and v and does not care how y
     was produced.

  2. XSA post-attention projection, ported verbatim from stage 1
     (src/xsa_attention.py). Per-role adaptive alpha (one alpha vector
     per head for CLS, another for patch), three modes per role:
       "zero"  -> alpha = 0 (standard SA, XSA disabled)
       "one"   -> alpha = 1 (hard XSA)
       "learn" -> alpha is a learnable nn.Parameter (init=alpha_init)

Nyström details (Xiong et al. 2021, faithful):

    Given Q, K, V each (B, H, N, D_head) with scale s = 1/sqrt(D_head):
      Q_s = Q * s
      Q_tilde = segment-mean(Q_s, m)   # (B, H, m, D_head), landmarks
      K_tilde = segment-mean(K,   m)   # (B, H, m, D_head), landmarks

      F  = softmax(Q_s @ K_tilde^T)    # (B, H, N, m)
      A  = softmax(Q_tilde @ K_tilde^T)  # (B, H, m, m)
      B  = softmax(Q_tilde @ K^T)      # (B, H, m, N)
      A_inv = moore_penrose_inv(A)     # iterative Newton-Schulz style
      y  = F @ A_inv @ B @ V           # (B, H, N, D_head)

    Landmarks are segment-mean pooled (not selected tokens), which is
    the paper's canonical choice and what TransMIL follows. If N is not
    divisible by m, we pad at the END (so CLS at index 0 stays put) and
    mask the pad columns of B before its softmax so they contribute 0.

XSA details (identical to stage 1):

    Vn = v / ||v||  (eps=1e-12 via F.normalize)
    z_i = y_i - alpha_role(i) * (y_i · Vn_i) * Vn_i

Diagnostics (see stage-1 src/diagnostics.py for semantics):
    cls_self_attn:   (B, H) -- a_{cls,cls} = F[b, h, 0, :] @ A_inv[b, h, :, :]
                               @ B[b, h, :, 0], precomputed (the Nyström
                               equivalent of a_{cls,cls} from full attention;
                               see the released protocol).
    cos_yv_pre:      (B, H, N) -- cos(y_i, v_i) before projection
    cos_zv_post:     (B, H, N) -- cos(z_i, v_i) after projection
    y_norm, v_norm, z_norm: (B, H, N) -- per-head L2 norms
    num_cls_tokens:  int       -- role boundary for CLS vs patch split

The stats dict intentionally does NOT include attn_probs (B, H, N, N);
Nyström never materializes it and for N ~ 37k that tensor would be
gigabytes. cls_self_attn is precomputed here instead.

Optional depthwise conv_skip (Xiong et al., a "per-position linear"
added to y, originally kernel-33 1D conv on V): intentionally OMITTED
here. Note: the lucidrains `nystrom-attention` package used by official
TransMIL DOES enable it (`residual=True` is the default in that
package's API and TransMIL does not override it). We drop it for our
MIL setting because a 1D depthwise conv over a token sequence with no
natural ordering is not geometrically meaningful; keeping Nyström pure
also makes the XSA analysis cleaner. This is a deliberate deviation
from the official TransMIL import path -- see the released protocol for the
full list of architectural deviations.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_ALPHA_MODES = ("zero", "one", "learn")


def moore_penrose_iter_inv(A: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """
    Iterative Moore-Penrose pseudo-inverse via the Razon-Schulz-style
    cubic iteration used by Xiong et al. 2021 ("Nystromformer").

    Formula per iteration (paper, Eq. 8):
        V_{n+1} = 0.25 * V_n * (13 I - A V_n (15 I - A V_n (7 I - A V_n)))

    Initialization matches the **lucidrains `nystrom-attention`** package
    used by the official TransMIL repo (so our Nystrom factor computation
    is numerically bit-compatible with TransMIL's import path):

        z = A.transpose(-1,-2) / (torch.max(col_sums) * torch.max(row_sums))

    where `torch.max` with no dim returns a SINGLE global scalar over the
    entire (batch, head, matrix) tensor -- not per-slice. This yields a
    smaller init than a per-slice max for slices whose own max is below
    the batch-wide max, but the convergence criterion ||I - A V_0||_op < 1
    is still satisfied because the init scaling is conservative (smaller
    ||V_0|| implies larger convergence margin).

    Earlier versions of this file used per-(batch, head) row/col maxima.
    The audit flagged that as a real numerical discrepancy vs. the
    official TransMIL dependency path; switching to the global-scalar
    form eliminates it. 6 iterations is the paper's + lucidrains' default.

    Input:  A of shape (..., m, m). Must be a softmax output (row-sums = 1).
    Output: A^+ (approximate Moore-Penrose inverse), same shape.
    """
    # Per lucidrains nystrom-attention: naming here follows their variable
    # names exactly for cross-referenceability, even though `col` ends up
    # holding row sums and `row` holds column sums (their quirk, not ours).
    abs_A = torch.abs(A)
    col = abs_A.sum(dim=-1)              # (..., m) -- row sums in math terms
    row = abs_A.sum(dim=-2)              # (..., m) -- col sums in math terms
    # Single global scalars over all leading + inner dims.
    max_col = torch.max(col)
    max_row = torch.max(row)
    V = A.transpose(-1, -2) / (max_col * max_row)

    m_ = A.shape[-1]
    # Identity broadcastable over any leading dims (B, H, ...).
    I = torch.eye(m_, device=A.device, dtype=A.dtype)
    I = I.expand_as(A)
    for _ in range(iters):
        AV = A @ V
        # 0.25 * V * (13 I - AV (15 I - AV (7 I - AV)))
        V = 0.25 * V @ (13 * I - (AV @ (15 * I - (AV @ (7 * I - AV)))))
    return V


def _pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, int]:
    """
    Pad a (..., N, D) tensor along the N axis at the END with zeros so
    N becomes a multiple of m. Returns (padded_tensor, pad_count).

    Padding at the end (not the start) keeps CLS at position 0, which
    both XSA's per-role alpha splitting and the cls_self_attn diagnostic
    rely on.
    """
    N = x.shape[-2]
    remainder = N % m
    if remainder == 0:
        return x, 0
    pad = m - remainder
    # F.pad pads the LAST dim by default; we want to pad the -2 dim (N).
    # pad spec for F.pad on a (..., N, D): (0, 0) on D, then (0, pad) on N.
    return F.pad(x, (0, 0, 0, pad)), pad


class NystromXSAAttention(nn.Module):
    """
    Nyström self-attention with an XSA post-projection and per-role
    adaptive alpha. Drop-in replacement for full-attention + XSA at
    slide scale.

    Args:
      dim:              embedding dim (= num_heads * head_dim)
      num_heads:        (default 6 for ViT-S-like config)
      num_landmarks m:  Nyström landmark count. Default 64, which is
                        lucidrains `nystrom-attention`'s library default.
                        Official TransMIL uses `dim // 2` (= 256 at
                        their dim=512); we use a smaller m as a
                        deliberate deviation for compute economy -- see
                        the released protocol
      qkv_bias, attn_drop, proj_drop: same as standard attention
      alpha_cls_mode, alpha_patch_mode: "zero" | "one" | "learn"
      alpha_init:       init value when mode == "learn"
      num_cls_tokens:   role boundary; tokens [0..num_cls_tokens-1] are
                        CLS, rest are patch (default 1)
      pinv_iterations:  Moore-Penrose iteration count (default 6)

    Forward: x (B, N, C) -> (B, N, C)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        num_landmarks: int = 64,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        alpha_cls_mode: str = "zero",
        alpha_patch_mode: str = "zero",
        alpha_init: float = 1.0,
        num_cls_tokens: int = 1,
        pinv_iterations: int = 6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert alpha_cls_mode in _ALPHA_MODES, f"alpha_cls_mode must be in {_ALPHA_MODES}"
        assert alpha_patch_mode in _ALPHA_MODES, f"alpha_patch_mode must be in {_ALPHA_MODES}"
        assert num_cls_tokens >= 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_landmarks = num_landmarks
        self.num_cls_tokens = num_cls_tokens
        self.alpha_cls_mode = alpha_cls_mode
        self.alpha_patch_mode = alpha_patch_mode
        self.pinv_iterations = pinv_iterations

        # Fused QKV projection, same as stage 1.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Per-role alpha: Parameter if "learn", buffer otherwise.
        self._make_alpha("alpha_cls", alpha_cls_mode, alpha_init, num_heads)
        self._make_alpha("alpha_patch", alpha_patch_mode, alpha_init, num_heads)

        # Diagnostic capture toggle.
        self._capture_stats = False
        self.last_stats: dict | None = None

    def _make_alpha(self, name: str, mode: str, init_val: float, n_heads: int) -> None:
        """Create self.<name> as Parameter or buffer depending on mode."""
        if mode == "learn":
            setattr(self, name, nn.Parameter(torch.full((n_heads,), float(init_val))))
        else:
            fixed = 0.0 if mode == "zero" else 1.0
            self.register_buffer(name, torch.full((n_heads,), fixed), persistent=True)

    def _build_alpha_per_token(self, N: int, device, dtype) -> torch.Tensor:
        """
        Concatenate alpha_cls and alpha_patch along the token axis.
        Returns shape (1, num_heads, N, 1) for broadcasting.
        """
        n_cls = self.num_cls_tokens
        n_patch = N - n_cls
        a_cls = self.alpha_cls.to(device=device, dtype=dtype).unsqueeze(-1).expand(-1, n_cls)
        a_patch = self.alpha_patch.to(device=device, dtype=dtype).unsqueeze(-1).expand(-1, n_patch)
        alpha = torch.cat([a_cls, a_patch], dim=1)          # (num_heads, N)
        return alpha.unsqueeze(0).unsqueeze(-1)              # (1, num_heads, N, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C) with N = num_cls_tokens + num_patches.
        """
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        m = self.num_landmarks

        # --- Q, K, V projection + head split (standard fused form) -------
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                              # each (B, H, N, D)
        # Scale Q ONCE. All three Nyström kernels then inherit the scale
        # through Q_tilde = segment-mean(Q_s, m), which is mathematically
        # equivalent to scaling each kernel output by `self.scale`.
        q = q * self.scale

        # --- Pad N to multiple of m ---------------------------------------
        # Pad at the END (not the start as lucidrains' nystrom-attention
        # does) so CLS stays at position 0. The pad amount is <= m-1.
        q_pad, pad = _pad_to_multiple(q, m)                  # (B, H, N', D)
        k_pad, _   = _pad_to_multiple(k, m)                  # (B, H, N', D)
        v_pad, _   = _pad_to_multiple(v, m)                  # (B, H, N', D)
        N_p = q_pad.shape[-2]
        seg = N_p // m                                        # segment size per landmark

        # Mask: (B, 1, N_p, 1), True where real token, False where pad.
        # Broadcast-compatible with (B, H, N_p, D) tensors.
        if pad > 0:
            mask = torch.ones(B, 1, N_p, 1, device=x.device, dtype=torch.bool)
            mask[..., N:, :] = False
        else:
            mask = None

        # --- Landmarks via segment-mean pooling ---------------------------
        # Reshape (B, H, N_p, D) -> (B, H, m, seg, D), then mean over seg.
        def seg_mean(t: torch.Tensor) -> torch.Tensor:
            # t shape: (B, H, N_p, D)
            t = t.reshape(B, H, m, seg, D)
            if mask is not None:
                # Weighted mean with mask. mask shape broadcast:
                # (B, 1, N_p, 1) -> (B, 1, m, seg, 1)
                mseg = mask.reshape(B, 1, m, seg, 1).to(t.dtype)
                s = (t * mseg).sum(dim=3)                   # (B, H, m, D)
                cnt = mseg.sum(dim=3).clamp(min=1e-6)        # (B, 1, m, 1)
                return s / cnt
            return t.mean(dim=3)

        q_tilde = seg_mean(q_pad)                            # (B, H, m, D)
        k_tilde = seg_mean(k_pad)                            # (B, H, m, D)

        # --- Three softmax factors ----------------------------------------
        # F: (B, H, N_p, m)
        kF_logits = q_pad @ k_tilde.transpose(-2, -1)
        # A: (B, H, m, m)
        kA_logits = q_tilde @ k_tilde.transpose(-2, -1)
        # B: (B, H, m, N_p)
        kB_logits = q_tilde @ k_pad.transpose(-2, -1)

        # Masks applied BEFORE softmax:
        #   F softmax is over the m landmark dim (last dim). Landmarks
        #   are all real (they are segment means of real tokens only
        #   under masked seg_mean above, so no -inf needed on F).
        #   B softmax is over the N_p token dim (last dim). Pad columns
        #   must be masked out so they contribute 0 after softmax --
        #   otherwise pad tokens leak weight into V.
        if pad > 0:
            mcol = mask.reshape(B, 1, 1, N_p)                # broadcast over (H, m)
            kB_logits = kB_logits.masked_fill(~mcol, float("-inf"))

        # F and B softmax dims are dim=-1 (standard).
        F_soft = kF_logits.softmax(dim=-1)                   # (B, H, N_p, m)
        A_soft = kA_logits.softmax(dim=-1)                   # (B, H, m, m)
        B_soft = kB_logits.softmax(dim=-1)                   # (B, H, m, N_p)

        # attn_drop applies to the combined attention probabilities. In
        # Nyström the natural place is on F and B (the two "user-side"
        # softmaxes); A is an internal landmark-landmark softmax. For
        # the implementation we keep drop=0 (default) so this is a no-op unless the
        # caller flips it on.
        F_soft = self.attn_drop(F_soft)
        B_soft = self.attn_drop(B_soft)

        # --- Moore-Penrose inverse of A -----------------------------------
        A_inv = moore_penrose_iter_inv(A_soft, iters=self.pinv_iterations)

        # --- Nyström attention output y -----------------------------------
        # Parenthesize for lower peak memory:
        #   (B, H, m, N_p) @ (B, H, N_p, D) = (B, H, m, D)
        #   (B, H, m, m)   @ (B, H, m,   D) = (B, H, m, D)
        #   (B, H, N_p, m) @ (B, H, m,   D) = (B, H, N_p, D)
        BV = B_soft @ v_pad                                  # (B, H, m, D)
        AinvBV = A_inv @ BV                                   # (B, H, m, D)
        y_pad = F_soft @ AinvBV                               # (B, H, N_p, D)

        # Drop pad rows to recover (B, H, N, D).
        y = y_pad[..., :N, :]

        # --- XSA projection (stage 1 semantics verbatim) ------------------
        # Vn = v / ||v||, clipped by F.normalize's eps (1e-12).
        vn = F.normalize(v, dim=-1)                          # (B, H, N, D)
        dot = (y * vn).sum(dim=-1, keepdim=True)             # (B, H, N, 1)
        alpha = self._build_alpha_per_token(N, device=y.device, dtype=y.dtype)
        z = y - alpha * dot * vn                             # (B, H, N, D)

        # --- Diagnostic capture -------------------------------------------
        if self._capture_stats:
            # cos(y, v) pre/post XSA, per (batch, head, token).
            cos_yv_pre = F.cosine_similarity(y, v, dim=-1).detach()   # (B, H, N)
            cos_zv_post = F.cosine_similarity(z, v, dim=-1).detach()  # (B, H, N)
            y_norm = y.norm(dim=-1).detach()                           # (B, H, N)
            v_norm = v.norm(dim=-1).detach()                           # (B, H, N)
            z_norm = z.norm(dim=-1).detach()                           # (B, H, N)
            # CLS self-attention a_{cls,cls}: precompute directly from
            # Nyström factors. The full attention matrix would be
            # P = F @ A_inv @ B of shape (B, H, N_p, N_p); we want only
            # P[:, :, 0, 0] since CLS is at position 0 after end-padding.
            # Compute as: F[:, :, 0, :] @ A_inv[:, :, :, :] @ B[:, :, :, 0].
            # Shapes: (B, H, m) @ (B, H, m, m) @ (B, H, m) -> (B, H).
            f_cls = F_soft[..., 0, :]                       # (B, H, m)
            b_cls = B_soft[..., :, 0]                       # (B, H, m)
            cls_self_attn = torch.einsum(
                "bhm,bhmn,bhn->bh", f_cls, A_inv, b_cls,
            ).detach()                                        # (B, H)

            # Real WSI heatmaps need the final CLS-to-patch row, not only the
            # scalar CLS self-attention diagnostic.  Reconstructing just this
            # row keeps the capture memory bounded: the full N x N Nyström
            # attention matrix is never materialized for WSI-scale bags.
            AB = A_inv @ B_soft                               # (B, H, m, N_p)
            P_cls = torch.einsum(
                "bhm,bhmn->bhn", F_soft[:, :, 0, :], AB,
            )                                                  # (B, H, N_p)
            n_cls = int(self.num_cls_tokens)
            cls_patch_attn = P_cls[:, :, n_cls:N].detach()     # (B, H, N_patch)

            self.last_stats = {
                # Per (batch, head): a_{cls,cls}. Note: Nyström's P is an
                # APPROXIMATION of the full softmax attention matrix, so
                # this value is the model's effective self-attention on
                # CLS under the Nyström approximation, not the "true"
                # softmax self-attention. See the released protocol
                "cls_self_attn": cls_self_attn,
                "cls_patch_attn": cls_patch_attn,
                "cos_yv_pre": cos_yv_pre,
                "cos_zv_post": cos_zv_post,
                "y_norm": y_norm,
                "v_norm": v_norm,
                "z_norm": z_norm,
                "num_cls_tokens": self.num_cls_tokens,
            }

        # --- Output projection --------------------------------------------
        z = z.transpose(1, 2).reshape(B, N, C)
        out = self.proj(z)
        out = self.proj_drop(out)
        return out
