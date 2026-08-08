"""Token-role specialized LayerNorm utilities.

The specialization tested here uses a deliberately narrow intervention:
normal LayerNorm statistics are still computed per token, but the learned
affine parameters are split by token role.  In the current ViT-style slide
models that role split is `[CLS]` versus patch tokens.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn


LayerNormShape = Union[int, tuple[int, ...], list[int]]


class RoleSplitLayerNorm(nn.Module):
    """LayerNorm with separate affine branches for CLS and patch tokens.

    The module expects sequence tensors shaped like `(B, L, C)` and splits
    along the token axis `L`.  This is intentionally not a two-stream model:
    the outputs are concatenated back into one token sequence before attention,
    so CLS and patch tokens still communicate normally.
    """

    def __init__(
        self,
        normalized_shape: LayerNormShape,
        *,
        mode: str = "cls_patch",
        num_cls_tokens: int = 1,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"shared", "cls_patch"}:
            raise ValueError(f"unknown RoleSplitLayerNorm mode: {mode!r}")
        if num_cls_tokens < 0:
            raise ValueError(
                f"num_cls_tokens must be non-negative, got {num_cls_tokens}"
            )

        self.mode = mode
        self.num_cls_tokens = int(num_cls_tokens)
        self.normalized_shape = normalized_shape
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)

        if mode == "shared":
            # Shared mode is provided for equivalence tests and future reuse.
            # Production default models still instantiate plain nn.LayerNorm
            # directly so old checkpoint keys stay unchanged.
            self.norm = nn.LayerNorm(
                normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
        else:
            # Two independent affine parameter sets are the entire
            # specialization.  Means/variances remain token-local exactly as
            # in standard nn.LayerNorm.
            self.cls_norm = nn.LayerNorm(
                normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
            self.patch_norm = nn.LayerNorm(
                normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(
                "RoleSplitLayerNorm expects a token sequence with shape "
                f"(B, L, ...), got {tuple(x.shape)}"
            )
        if self.mode == "shared":
            return self.norm(x)

        n_tokens = x.shape[1]
        n_cls = min(self.num_cls_tokens, n_tokens)

        if n_cls == 0:
            # This edge case is useful for tests and for future tokenizers
            # without a global token.  It should behave like patch-only LN.
            return self.patch_norm(x)
        if n_cls == n_tokens:
            # Avoid concatenating an empty patch slice.  This also keeps
            # gradients and dtype behavior identical when only CLS is present.
            return self.cls_norm(x)

        cls = self.cls_norm(x[:, :n_cls, ...])
        patch = self.patch_norm(x[:, n_cls:, ...])
        return torch.cat((cls, patch), dim=1)

    @torch.no_grad()
    def copy_shared_weights_(self, shared_norm: nn.LayerNorm) -> "RoleSplitLayerNorm":
        """Copy one shared LayerNorm's affine state into every active branch."""

        if not isinstance(shared_norm, nn.LayerNorm):
            raise TypeError(
                "copy_shared_weights_ expects an nn.LayerNorm source, got "
                f"{type(shared_norm)!r}"
            )

        if self.mode == "shared":
            self.norm.load_state_dict(shared_norm.state_dict())
        else:
            # Both branches start from the same affine state so a split-LN
            # model can be made exactly equivalent to a shared-LN model before
            # training lets the branches diverge.
            state = shared_norm.state_dict()
            self.cls_norm.load_state_dict(state)
            self.patch_norm.load_state_dict(state)
        return self

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, mode={self.mode!r}, "
            f"num_cls_tokens={self.num_cls_tokens}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
