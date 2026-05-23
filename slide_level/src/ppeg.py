"""
PPEG (Pyramid Position Encoding Generator) from TransMIL (Shao et al. 2021).

PPEG is TransMIL's alternative to learned absolute positional embeddings.
Instead of a table indexed by position, it mixes local neighborhoods of
the reshaped 2D token grid with three depthwise convolutions (kernel 7,
5, 3), then adds the convolved output back to the original features.
CLS is carried through unchanged.

Faithful to the official TransMIL implementation:
  - Three depthwise conv2d layers (groups=dim) with kernels 7, 5, 3.
  - Skip connection: out = conv7(x) + x + conv5(x) + conv3(x).
  - CLS token at position 0 is separated, never convolved, then re-joined.
  - The caller supplies (H, W) such that H * W == number of patch tokens.
    The grid is padded to square by duplicating the first few patch
    features BEFORE PPEG is called; see NystromXSAggregator.forward.

Rationale on depthwise: the conv mixes spatial context at fixed channel,
adding a positional signal that is translation-equivariant over the
square grid. For MIL with no natural ordering this is only "kind of"
positional, but we follow TransMIL exactly because (a) our scientific
question is about XSA, not PPEG, and (b) deviating from the reference
architecture would confound the comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PPEG(nn.Module):
    """
    TransMIL's PPEG block. In signature and semantics this matches
    Shao et al. 2021 verbatim.

    Args:
      dim: embedding dimension (channel count).
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        # kernel=7, stride=1, padding=3 (=7//2); groups=dim gives depthwise.
        self.conv7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1,
                               padding=3, groups=dim)
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1,
                               padding=2, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                               padding=1, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: (B, 1+H*W, D) with CLS at position 0.
        H, W: reshape dimensions for the patch tokens (H*W == x.shape[1]-1).
        Returns: (B, 1+H*W, D) with CLS at position 0 unchanged.
        """
        B, L, C = x.shape
        assert L == 1 + H * W, (
            f"PPEG expects L == 1 + H*W; got L={L}, H={H}, W={W} (1+HW={1+H*W})"
        )
        cls = x[:, :1, :]                 # (B, 1, D)  -- unchanged
        feat = x[:, 1:, :]                # (B, HW, D)
        # (B, HW, D) -> (B, D, H, W)
        cnn = feat.transpose(1, 2).contiguous().view(B, C, H, W)
        # Per TransMIL: three depthwise convs summed with a skip.
        cnn = self.conv7(cnn) + cnn + self.conv5(cnn) + self.conv3(cnn)
        # Flatten back to token sequence: (B, D, H, W) -> (B, HW, D).
        feat = cnn.flatten(2).transpose(1, 2).contiguous()
        return torch.cat([cls, feat], dim=1)
