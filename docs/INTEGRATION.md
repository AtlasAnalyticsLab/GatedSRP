# Integrating Gated SRP

This page is for using Gated SRP outside the bundled reproduction manifests.
The shortest path is to reuse the provided attention classes. The most portable
path is to implement the projection update around your own attention block.

## Inputs You Need

Gated SRP needs the same token sequence as your attention layer plus local
neighborhood information.

| Input | Shape | Meaning |
|---|---:|---|
| `features` | `(B, N, D_in)` | Frozen or trainable patch features before the slide aggregator. |
| `coords` | `(N, 2)` or `(B, N, 2)` | Level-0 patch coordinates or grid coordinates. |
| `neighbor_index` | `(B, N, K)` | Neighbor token indices for each real patch, `-1` for invalid slots. |
| `neighbor_mask` | `(B, N, K)` | Boolean mask for valid neighbor slots. |
| `h_local` | `(B, N)` | Mean cosine similarity between each patch feature and its neighbors. Required by signed-gated SRP. |

The bundled WSI datasets build `neighbor_index`, `neighbor_mask`, and
`h_morph` in their loaders. The slide trainer computes `h_local` on device
from the raw feature tensor and neighbor graph.

For a runnable synthetic example, see
[examples/minimal_slide_forward.py](../examples/minimal_slide_forward.py).

## Option A: Slide-Level Nystrom Aggregator

Use this when your model resembles a TransMIL-style WSI aggregator with one
slide bag per forward pass.

```python
import torch

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
)

features = torch.randn(1, 2048, 1536)
neighbor_index = torch.full((1, 2048, 8), -1, dtype=torch.long)
neighbor_mask = torch.zeros((1, 2048, 8), dtype=torch.bool)
h_local = torch.zeros((1, 2048))

logits = model(
    features,
    neighbor_index,
    neighbor_mask,
    h_local=h_local,
)
```

For real data, do not use the dummy neighbor graph above. Build the graph from
patch coordinates using `slide_level_srp.data_ext.build_neighbor_graph`, or use
one of the bundled dataset loaders.

## Option B: Fixed-Grid Full Attention

Use this when you have a ViT-style fixed patch grid, such as a 14 x 14 or
17 x 17 image-token grid.

```python
from src.srp_patch_attention import PatchSRPAttention

attn = PatchSRPAttention(
    dim=384,
    num_heads=6,
    grid_h=14,
    grid_w=14,
    beta_patch_mode="signed_gated",
    delta_scale=2.0,
    gate_hidden_dim=16,
    gate_output_init="zero",
)
```

`PatchSRPAttention` constructs the grid neighborhood internally. Pass `h_local`
when using signed-gated or learned-local-reference modes.

## Option C: Implement the Update in Your Own Attention

If you already have an attention implementation, add Gated SRP after computing
per-head attention output `y` and values `v`.

```python
# y: (B, H, N, D) attention output for patch tokens
# v: (B, H, N, D) value vectors for patch tokens
# neighbor_index: (B, N, K), neighbor_mask: (B, N, K)

neighbor_v = gather_neighbors(v.detach(), neighbor_index, neighbor_mask)
r = masked_mean(neighbor_v, neighbor_mask)          # (B, H, N, D)
r_hat = normalize(r)

dot = (y * r_hat).sum(dim=-1, keepdim=True)
beta = signed_gate(token_diag, head_diag)           # (B, H, N, 1)
z = y - beta * dot * r_hat
```

Keep these invariants:

- Start from `beta = 0` so the model initially matches the base attention layer.
- Do not directly project CLS tokens. Let CLS read changed patch tokens in the
  next attention operation.
- Use real-patch masks so padding and square-pad duplicate rows are excluded.
- For post-attention patch-only updates, avoid relying on the final block if the
  prediction head immediately reads CLS; there is no later attention step for
  patch edits to reach CLS.

## Reproduction Configurations

The reported Gated SRP configurations are encoded in the manifests:

- WSI classification: [configs/paper_classification.tsv](../configs/paper_classification.tsv)
- TCGA survival: [configs/paper_tcga_survival.tsv](../configs/paper_tcga_survival.tsv)
- Architecture ablation: [configs/paper_architecture_ablation.tsv](../configs/paper_architecture_ablation.tsv)
- Design ablation: [configs/paper_design_ablation.tsv](../configs/paper_design_ablation.tsv)
- Patch-encoder ablation: [configs/paper_patch_encoder_ablation.tsv](../configs/paper_patch_encoder_ablation.tsv)

Use these as known-good starting points before changing gate range, hidden
width, placement, neighborhood window, or patch encoder.
