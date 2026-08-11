# Integrating GatedSRP

GatedSRP changes the patch-token attention update, not the task head. A model
needs patch coordinates, a local neighbor graph, and access to the per-patch
attention stream. Existing positional encoding, residual connections, and CLS
readout can remain unchanged.

## Integration Contract

| Input | Shape | Meaning |
|---|---:|---|
| Patch features | `(B, N, D_in)` | Input feature bag, usually frozen patch-encoder embeddings. |
| Coordinates | `(B, N, 2)` | Level-0 pixel coordinates or integer grid positions. |
| Neighbor indices | `(B, N, K)` | Source patch indices; `-1` marks a missing coordinate cell. |
| Neighbor mask | `(B, N, K)` | Valid-neighbor mask aligned with the indices. |
| Neighbor weights | `(B, N, K)` | Optional uniform, Gaussian, or inverse-distance weights. |
| Local homogeneity | `(B, N)` | Mean feature cosine similarity to valid neighbors for the token gate. |

The public loaders construct these fields from each H5 `/coords` array. To use
the same graph builder in another pipeline:

```python
import torch

from slide_level_srp.data_ext import build_neighbor_graph, compute_h_morph

# coords_np: (N, 2), features_np: (N, D)
index_np, mask_np, weight_np = build_neighbor_graph(
    coords_np,
    stride=512,
    radius=1,
    source="real",
    weighting="uniform",
)
h_local_np = compute_h_morph(features_np, index_np, mask_np)

neighbor_index = torch.from_numpy(index_np).unsqueeze(0)
neighbor_mask = torch.from_numpy(mask_np).unsqueeze(0)
neighbor_weight = torch.from_numpy(weight_np).unsqueeze(0)
h_local = torch.from_numpy(h_local_np).unsqueeze(0)
```

Set `stride` to the level-0 distance between adjacent extracted patches. The
AtlasPatch loaders use dataset-specific defaults where the H5 file does not
store this metadata.

## Option 1: Use the Nyström Aggregator

This is the shortest route for a TransMIL-style one-slide-per-batch model.

```python
from slide_level_srp.src.srp_aggregator import NystromSRPAggregator

model = NystromSRPAggregator(
    in_dim=1536,
    embed_dim=384,
    depth=4,
    num_heads=6,
    num_classes=2,
    beta_patch_mode="signed_gated",
    srp_mode="post_agg_signed_gated",
    delta_scale=1.0,
    gate_hidden_dim=16,
    gate_output_init="zero",
    gate_activation="tanh",
    srp_context_impl="streaming_mean",
    srp_correction_chunk_size=32768,
)

logits = model(
    features,
    neighbor_index,
    neighbor_mask,
    h_local=h_local,
    neighbor_weight=neighbor_weight,
)
```

`features`, graph tensors, and `h_local` must be on the same device as the
model. The implementation supports variable slide lengths with batch size 1.

## Option 2: Add a Post-Attention Hook

`PatchSRPCorrection` is an architecture-neutral module used by the SPAN and
LongNet adapters. Insert it immediately after a patch-token attention update
and before that update enters the residual path.

```python
import torch
from torch import nn

from slide_level_srp.src.srp_correction import PatchSRPCorrection

class AttentionBlockWithGatedSRP(nn.Module):
    def __init__(self, dim: int, attention: nn.Module):
        super().__init__()
        self.attn = attention
        self.srp = PatchSRPCorrection(
            dim,
            hidden_dim=32,
            delta_scale=2.0,
            correction_chunk_size=8192,
        )

    def forward(self, x, neighbor_index, neighbor_mask, neighbor_weight=None):
        residual = x
        update = self.attn(x)

        # Keep CLS outside the local patch graph.
        patch_update = self.srp(
            update[:, 1:],
            neighbor_index,
            neighbor_mask,
            neighbor_weight=neighbor_weight,
        )
        update = torch.cat([update[:, :1], patch_update], dim=1)
        return residual + update
```

This compact adapter estimates the local direction in the exposed
post-attention patch stream. Because it receives an already merged token
update, its learned coefficient is token-specific rather than head-specific.
When your attention implementation exposes per-head outputs and value vectors,
use the value vectors for the local direction as in Option 3; that matches the
Nyström implementation directly.

## Option 3: Implement the Operator Directly

For per-head attention output `y` and value vectors `v`:

```python
# y, v: (B, H, N, D)
# neighbor_index, neighbor_mask: (B, N, K)
neighbor_v = gather_neighbors(v.detach(), neighbor_index, neighbor_mask)
_, r_hat, neighbor_count = neighborhood_mean(
    neighbor_v,
    neighbor_mask,
    neighbor_weight,
)

projection = (y * r_hat).sum(dim=-1, keepdim=True) * r_hat
beta = signed_gate(token_diagnostics, head_diagnostics)
z = y - beta * projection
```

Reuse `TokenHeadGate` from `slide_level_srp/src/gate_signed.py` when possible.
Its output path is zero-initialized and its shape validation catches graph/token
misalignment before a long training run.

## Fixed-Grid ViT

For a fixed patch grid, `src.srp_patch_attention.PatchSRPAttention` builds the
local grid internally:

```python
from src.srp_patch_attention import PatchSRPAttention

attn = PatchSRPAttention(
    dim=384,
    num_heads=6,
    grid_h=17,
    grid_w=17,
    beta_patch_mode="signed_gated",
    delta_scale=2.0,
    gate_hidden_dim=16,
    gate_output_init="zero",
)
```

The ADP trainer demonstrates this path on raw RGB patches.

## Invariants to Preserve

- Initialize the gate output path at zero so the inserted module starts as an
  exact identity.
- Correct real patch rows only; do not project CLS or padding rows.
- Build paired comparison arms from the same train/validation/test split and,
  when capping patches, the same retained subset.
- Rebuild neighbor indices after subsampling. The dense experiment uses
  deterministic random retention followed by nearest retained-coordinate
  neighbors.
- In the Nyström path, construct the local direction from detached per-head
  value vectors, preserving gradients through `y` and the gate. The portable
  `PatchSRPCorrection` adapter instead estimates the direction from the exposed
  post-attention token stream and keeps gradients through that correction
  geometry; this is the adapter used for the released SPAN and LongNet rows.
- For a post-attention patch-only update, place the correction before a later
  CLS-mixing block. A patch write after the final CLS readout has no effect.
- Keep the coefficient bounded. Use `fixed` for `delta*tanh(g)` or
  `direct_beta_softclip` for `2*tanh(g/2)`.

## Known Integrations

| Architecture | Public implementation | Setup |
|---|---|---|
| Nyström/TransMIL | `NystromSRPAggregator` | Core dependencies only. |
| Dense MHSA | `DenseAttentionSRPAggregator` | Core dependencies; cap at 1,024 for the released comparison. |
| SPAN | `OfficialSPANAggregator` | SPAN checkout, OmegaConf, compatible DGL. |
| Prov-GigaPath LongNet | `OfficialGigaPathLongNetAggregator` | Prov-GigaPath checkout and flash-attn or xFormers. |

Use [ARCHITECTURES.md](ARCHITECTURES.md) for optional dependency setup and
[slide_backbones.tsv](../configs/slide_backbones.tsv) for
complete paired commands.
