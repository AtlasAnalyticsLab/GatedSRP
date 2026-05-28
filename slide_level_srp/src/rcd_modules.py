"""Identity-safe Residual Context Decomposition modules.

These modules implement the post-signed-gate refinement plan without changing
the existing SRP or signed-gate paths.  They are intentionally small and
zero-output-initialized so a newly enabled RCD arm starts from the same
attention output as the original model and must learn any intervention.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


_RCD_ADAPTER_KINDS = ("lowrank", "diag")


class IdentitySafeRCDRecomposer(nn.Module):
    """Factorized common/residual recomposer for Method 2.1.

    Given attention output ``y`` and a context direction ``r_hat``, the
    decomposition is exact:

        common = proj_r(y)
        residual = y - common

    The module returns ``y + delta_residual + delta_common``.  The two deltas
    use non-shared branch maps, which avoids the degenerate "generic adapter on
    y" interpretation called out in ``NEXT_METHOD_REFINEMENT_PLAN.md``.  The
    delta output is exactly zero after ``reset_identity()``, so the initial
    function is identity even though branch-specific parameters exist.
    """

    def __init__(
        self,
        head_dim: int,
        rank: int = 16,
        adapter_kind: str = "lowrank",
    ) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        if adapter_kind not in _RCD_ADAPTER_KINDS:
            raise ValueError(
                f"adapter_kind must be one of {_RCD_ADAPTER_KINDS}, "
                f"got {adapter_kind!r}"
            )
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.adapter_kind = adapter_kind

        if adapter_kind == "lowrank":
            bottleneck = min(int(rank), int(head_dim))
            self.res_down = nn.Linear(head_dim, bottleneck)
            self.res_up = nn.Linear(bottleneck, head_dim)
            self.com_down = nn.Linear(head_dim, bottleneck)
            self.com_up = nn.Linear(bottleneck, head_dim)
            self.register_parameter("res_diag_delta", None)
            self.register_parameter("com_diag_delta", None)
        else:
            # Diagonal branch maps are useful when the capacity control needs
            # to stay very close to the signed-gate parameter count.  Zero
            # deltas mean residual + common is returned exactly at init.
            self.res_down = None
            self.res_up = None
            self.com_down = None
            self.com_up = None
            self.res_diag_delta = nn.Parameter(torch.zeros(head_dim))
            self.com_diag_delta = nn.Parameter(torch.zeros(head_dim))

        self.reset_identity()

    def reset_identity(self) -> None:
        """Restore the exact identity function without touching down maps.

        Parent models run their own global initialization pass.  Calling this
        after that pass re-zeroes only the branch output surfaces, preserving an
        exact no-op start while leaving hidden projections ready to receive
        gradients once the zero output maps move.
        """
        if self.adapter_kind == "lowrank":
            nn.init.zeros_(self.res_up.weight)
            nn.init.zeros_(self.res_up.bias)
            nn.init.zeros_(self.com_up.weight)
            nn.init.zeros_(self.com_up.bias)
        else:
            nn.init.zeros_(self.res_diag_delta)
            nn.init.zeros_(self.com_diag_delta)

    def _branch_delta(self, x: torch.Tensor, branch: str) -> torch.Tensor:
        if self.adapter_kind == "lowrank":
            if branch == "residual":
                return self.res_up(F.gelu(self.res_down(x)))
            return self.com_up(F.gelu(self.com_down(x)))
        if branch == "residual":
            return x * self.res_diag_delta.to(dtype=x.dtype, device=x.device)
        return x * self.com_diag_delta.to(dtype=x.dtype, device=x.device)

    def forward(self, y: torch.Tensor, r_hat: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return recomposed output and lightweight detached diagnostics.

        Shapes:
          y, r_hat: ``(B, H, N, D)``.
        """
        if y.shape != r_hat.shape:
            raise ValueError(
                f"RCD y/r_hat shape mismatch: got {tuple(y.shape)} and "
                f"{tuple(r_hat.shape)}"
            )
        common = (y * r_hat).sum(dim=-1, keepdim=True) * r_hat
        residual = y - common
        delta_res = self._branch_delta(residual, "residual")
        delta_com = self._branch_delta(common, "common")
        out = y + delta_res + delta_com
        with torch.no_grad():
            stats = {
                "delta_residual_norm": delta_res.norm(dim=-1).detach(),
                "delta_common_norm": delta_com.norm(dim=-1).detach(),
                "common_norm": common.norm(dim=-1).detach(),
                "residual_norm": residual.norm(dim=-1).detach(),
            }
        return out, stats


class LearnedLocalContextDirection(nn.Module):
    """Local-neighbour context-direction scorer for Method 2.4.

    The scorer learns an additive logit over the existing local-neighbour
    weights.  The additive logit head is zero-initialized, so the initial
    softmax recovers the original normalized neighbour weighting.  This keeps
    Method 2.4 isolated: any change in context direction must be learned, and
    the arm can be compared directly with fixed-neighbour RCD.
    """

    def __init__(
        self,
        head_dim: int,
        hidden_dim: int = 16,
        use_h_local: bool = True,
    ) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        self.head_dim = int(head_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_h_local = bool(use_h_local)

        self.center_proj = nn.Linear(head_dim, hidden_dim, bias=False)
        self.neighbor_proj = nn.Linear(head_dim, hidden_dim, bias=False)
        self.h_local_proj = nn.Linear(1, hidden_dim, bias=False) if use_h_local else None
        self.score = nn.Linear(hidden_dim, 1)
        self.reset_identity()

    def reset_identity(self) -> None:
        # Zero additive scores make the learned softmax equal to the original
        # base weighting.  The projection layers are deliberately left at their
        # ordinary initialization so the scorer can start learning as soon as
        # the output score head moves.
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(
        self,
        *,
        center_v: torch.Tensor,
        neighbor_v: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_weight: torch.Tensor | None = None,
        h_local: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(r, r_hat, cnt, weights)``.

        Shapes:
          center_v: ``(B, H, N, D)``
          neighbor_v: ``(B, H, N, K, D)``
          neighbor_mask: ``(B, N, K)``
          neighbor_weight: optional ``(B, N, K)``
          h_local: optional ``(B, N)`` required when ``use_h_local=True``
        """
        if neighbor_v.ndim != 5:
            raise ValueError(f"neighbor_v must be 5-D, got {tuple(neighbor_v.shape)}")
        B, H, N, K, D = neighbor_v.shape
        if center_v.shape != (B, H, N, D):
            raise ValueError(
                f"center_v shape mismatch: got {tuple(center_v.shape)}, "
                f"expected ({B}, {H}, {N}, {D})"
            )
        if neighbor_mask.shape != (B, N, K):
            raise ValueError(
                f"neighbor_mask shape mismatch: got {tuple(neighbor_mask.shape)}, "
                f"expected ({B}, {N}, {K})"
            )
        if neighbor_weight is not None and neighbor_weight.shape != (B, N, K):
            raise ValueError(
                f"neighbor_weight shape mismatch: got {tuple(neighbor_weight.shape)}, "
                f"expected ({B}, {N}, {K})"
            )
        if self.use_h_local:
            if h_local is None:
                raise ValueError("LearnedLocalContextDirection requires h_local")
            if h_local.shape != (B, N):
                raise ValueError(
                    f"h_local shape mismatch: got {tuple(h_local.shape)}, "
                    f"expected ({B}, {N})"
                )

        center_h = self.center_proj(center_v).unsqueeze(3)       # (B, H, N, 1, R)
        neighbor_h = self.neighbor_proj(neighbor_v)              # (B, H, N, K, R)
        hidden = center_h + neighbor_h
        if self.use_h_local:
            h = h_local.to(dtype=center_v.dtype, device=center_v.device)
            h = self.h_local_proj(h.unsqueeze(-1)).unsqueeze(1).unsqueeze(3)
            hidden = hidden + h
        learned_logits = self.score(torch.tanh(hidden)).squeeze(-1)  # (B, H, N, K)

        valid = neighbor_mask.to(dtype=torch.bool, device=center_v.device).unsqueeze(1)
        if neighbor_weight is None:
            base_logits = torch.zeros((B, 1, N, K), dtype=center_v.dtype, device=center_v.device)
        else:
            base_w = neighbor_weight.to(dtype=center_v.dtype, device=center_v.device).clamp_min(1e-12)
            base_logits = base_w.log().unsqueeze(1)

        logits = base_logits + learned_logits
        logits = logits.masked_fill(~valid, -1.0e9)
        weights = torch.softmax(logits, dim=3) * valid.to(dtype=center_v.dtype)
        # All-invalid rows would otherwise retain a meaningless uniform
        # softmax before masking.  The post-mask weights are zero, and this
        # explicit renormalization keeps partially-valid rows normalized.
        weights = weights / weights.sum(dim=3, keepdim=True).clamp_min(1e-12)
        r = (neighbor_v * weights.unsqueeze(-1)).sum(dim=3)
        r_hat = F.normalize(r, dim=-1, eps=1e-12)
        cnt = neighbor_mask.to(dtype=center_v.dtype, device=center_v.device).sum(dim=2)
        cnt = cnt.view(B, 1, N, 1)
        return r, r_hat, cnt, weights.detach()


def collect_rcd_module_ids(model: nn.Module) -> set[int]:
    """Return ids of RCD modules and descendants for init-skip passes."""
    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, (IdentitySafeRCDRecomposer, LearnedLocalContextDirection)):
            ids.update(id(m) for m in module.modules())
    return ids


def reset_rcd_identity_modules(modules: Iterable[nn.Module]) -> None:
    """Reset every RCD refinement module in ``modules`` to no-op behavior."""
    for module in modules:
        if isinstance(module, (IdentitySafeRCDRecomposer, LearnedLocalContextDirection)):
            module.reset_identity()
