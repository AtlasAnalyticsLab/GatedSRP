"""Architecture-neutral post-attention Gated SRP correction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PatchSRPCorrection(nn.Module):
    """Apply a signed local-redundancy correction to patch tokens.

    This compact module is used when adapting Gated SRP to third-party slide
    encoders whose attention implementation cannot reuse
    :class:`NystromSRPAttention` directly. It consumes the same coordinate
    neighbor graph and preserves the identity-start invariant.
    """

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int = 16,
        delta_scale: float = 1.0,
        correction_chunk_size: int = 8192,
        checkpoint_correction: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive")
        if delta_scale <= 0.0:
            raise ValueError("delta_scale must be positive")
        if correction_chunk_size < 0:
            raise ValueError("correction_chunk_size must be non-negative")
        self.delta_scale = float(delta_scale)
        self.correction_chunk_size = int(correction_chunk_size)
        self.checkpoint_correction = bool(checkpoint_correction)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2 + 2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        # A zero output layer makes the insertion exactly equivalent to the
        # unmodified architecture at initialization.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _apply_gate(
        self,
        token_chunk: torch.Tensor,
        local_direction: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        projection = (
            (token_chunk * local_direction).sum(dim=-1, keepdim=True)
            * local_direction
        )
        has_neighbor = (neighbor_mask.sum(dim=-1, keepdim=True) > 0).to(
            token_chunk.dtype
        )
        token_norm = token_chunk.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        cosine = (
            (token_chunk * local_direction).sum(dim=-1, keepdim=True)
            / token_norm
        )
        relative_projection = projection.norm(dim=-1, keepdim=True) / token_norm
        # The diagnostics guide only the scalar gate. Detaching them prevents
        # the gate branch from changing the local direction estimator itself.
        gate_input = torch.cat(
            [
                token_chunk.detach(),
                local_direction.detach(),
                cosine.detach(),
                relative_projection.detach(),
            ],
            dim=-1,
        )
        beta = self.delta_scale * torch.tanh(self.gate(gate_input))
        return token_chunk - beta * has_neighbor * projection

    def _forward_range(
        self,
        tokens: torch.Tensor,
        start: int,
        end: int,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_weight: torch.Tensor,
    ) -> torch.Tensor:
        token_chunk = tokens[:, start:end]
        index_chunk = neighbor_index[:, start:end]
        mask_chunk = neighbor_mask[:, start:end]
        weight_chunk = neighbor_weight[:, start:end]
        bsz, n_chunk, dim = token_chunk.shape
        numerator = tokens.new_zeros((bsz, n_chunk, dim))
        denominator = tokens.new_zeros((bsz, n_chunk, 1))
        # Streaming over neighbor slots avoids a B x N x K x D temporary and
        # keeps the adapter usable on native-length slide bags.
        for slot in range(index_chunk.shape[-1]):
            safe_index = index_chunk[..., slot].clamp(min=0)
            gathered = tokens.gather(
                1, safe_index.unsqueeze(-1).expand(-1, -1, dim)
            )
            slot_weight = weight_chunk[..., slot].unsqueeze(-1)
            numerator = numerator + gathered * slot_weight
            denominator = denominator + slot_weight
        local_mean = numerator / denominator.clamp_min(1.0)
        local_direction = F.normalize(local_mean, dim=-1, eps=1.0e-6)
        return self._apply_gate(token_chunk, local_direction, mask_chunk)

    def forward(
        self,
        tokens: torch.Tensor,
        neighbor_index: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
        *,
        neighbor_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if neighbor_index is None or neighbor_mask is None:
            return tokens
        if tokens.ndim != 3 or neighbor_index.ndim != 3:
            raise ValueError(
                "tokens and neighbor_index must have shapes (B,N,D) and (B,N,K)"
            )
        bsz, n_tokens, _ = tokens.shape
        if neighbor_index.shape[:2] != (bsz, n_tokens):
            raise ValueError("neighbor_index must align with patch tokens")
        if neighbor_mask.shape != neighbor_index.shape:
            raise ValueError("neighbor_mask must match neighbor_index")

        mask = neighbor_mask.to(device=tokens.device, dtype=torch.bool)
        if neighbor_weight is None:
            weight = mask.to(tokens.dtype)
        else:
            if neighbor_weight.shape != neighbor_index.shape:
                raise ValueError("neighbor_weight must match neighbor_index")
            weight = neighbor_weight.to(tokens.device, tokens.dtype) * mask.to(
                tokens.dtype
            )
        index = neighbor_index.to(tokens.device)

        chunk_size = self.correction_chunk_size
        if chunk_size == 0 or n_tokens <= chunk_size:
            return self._forward_range(tokens, 0, n_tokens, index, mask, weight)

        chunks = []
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            if self.training and self.checkpoint_correction and tokens.requires_grad:
                # Captured indices are immutable for this forward. Checkpointing
                # trades recomputation for lower activation memory.
                def chunk_forward(
                    source: torch.Tensor,
                    chunk_start: int = start,
                    chunk_end: int = end,
                ) -> torch.Tensor:
                    return self._forward_range(
                        source,
                        chunk_start,
                        chunk_end,
                        index,
                        mask,
                        weight,
                    )

                corrected = checkpoint(
                    chunk_forward,
                    tokens,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                corrected = self._forward_range(
                    tokens, start, end, index, mask, weight
                )
            chunks.append(corrected)
        return torch.cat(chunks, dim=1)


__all__ = ["PatchSRPCorrection"]
