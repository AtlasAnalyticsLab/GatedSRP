"""Minimal forward pass for a slide-level Gated SRP aggregator.

This example uses synthetic patch features and a regular coordinate grid so
the script can run without downloading WSI data. Real use should replace the
random features and coordinates with H5 arrays from the embedding pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Running `python examples/minimal_slide_forward.py` sets sys.path to the
# examples directory. Add the repository root so the local packages resolve
# without requiring an editable install for this smoke example.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slide_level_srp.data_ext import build_neighbor_graph
from slide_level_srp.src.srp_aggregator import NystromSRPAggregator


def compute_h_local(
    features: torch.Tensor,
    neighbor_index: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the local homogeneity signal used by the signed gate.

    The gate needs a token-level cue that says whether a patch is locally
    similar to its neighbors. This mirrors the training path while keeping the
    example self-contained.
    """

    eps = 1e-12
    features_norm = features / (features.norm(dim=-1, keepdim=True) + eps)
    safe_idx = neighbor_index.clamp(min=0)
    batch_size, n_tokens, neighbor_slots = safe_idx.shape
    batch_idx = torch.arange(features.shape[0]).view(batch_size, 1, 1)
    batch_idx = batch_idx.expand(batch_size, n_tokens, neighbor_slots)
    neighbor_features = features_norm[batch_idx, safe_idx, :]
    cosine = (neighbor_features * features_norm[:, :, None, :]).sum(dim=-1)
    cosine = cosine * neighbor_mask.to(cosine.dtype)
    counts = neighbor_mask.sum(dim=-1).to(cosine.dtype).clamp(min=1.0)
    return cosine.sum(dim=-1) / counts


def main() -> None:
    torch.manual_seed(7)

    grid_size = 8
    n_tokens = grid_size * grid_size
    in_dim = 1536
    stride = 256

    # Synthetic coordinates mimic a non-overlapping patch grid. The neighbor
    # builder works the same way for level-0 WSI pixel coordinates.
    coords = np.array(
        [(x * stride, y * stride) for y in range(grid_size) for x in range(grid_size)],
        dtype=np.int64,
    )
    neighbor_index_np, neighbor_mask_np, _ = build_neighbor_graph(coords, stride=stride)

    features = torch.randn(1, n_tokens, in_dim)
    neighbor_index = torch.from_numpy(neighbor_index_np).unsqueeze(0)
    neighbor_mask = torch.from_numpy(neighbor_mask_np).unsqueeze(0)
    h_local = compute_h_local(features, neighbor_index, neighbor_mask)

    model = NystromSRPAggregator(
        in_dim=in_dim,
        embed_dim=384,
        depth=4,
        num_heads=6,
        num_classes=2,
        beta_patch_mode="signed_gated",
        srp_mode="post_agg_signed_gated",
        delta_scale=1.0,
        gate_hidden_dim=16,
        gate_output_init="zero",
    )

    logits = model(
        features,
        neighbor_index,
        neighbor_mask,
        h_local=h_local,
    )
    print(logits.shape)


if __name__ == "__main__":
    main()
