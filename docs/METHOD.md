# Method Overview

Gated Spatial Redundancy Projection (Gated SRP) is an attention correction for
computational pathology transformers. It is motivated by a simple WSI-specific
fact: nearby tissue patches often carry highly similar morphology, stain,
texture, and cellular content. A standard attention layer can repeatedly mix
that locally common component into patch tokens, making subtle diagnostic or
prognostic deviations harder to preserve.

Gated SRP keeps the ordinary attention layer and adds a small geometric update
around it.

## Core Update

For each attention head and patch token:

1. Compute the normal attention output `y_i`.
2. Build a local reference direction `r_i` by averaging the value vectors of
   spatial neighbors from the coordinate grid.
3. Normalize it to `r_hat_i`.
4. Predict a signed coefficient `beta_i` with a small token/head gate.
5. Correct only the neighborhood-aligned component:

```text
z_i = y_i - beta_i * <y_i, r_hat_i> * r_hat_i
```

`beta_i = 0` preserves the original attention output. Positive values suppress
the locally common component. Larger positive values can reflect that component.
Negative values can preserve or amplify local context when the task needs it.

## Design Choices

| Choice | Why it matters |
|---|---|
| Local spatial reference | Redundancy is estimated from neighboring patches rather than from a global slide mean. |
| Signed adaptive gate | The model can decide whether to subtract, preserve, or reflect local common content per token and head. |
| Identity-safe initialization | The signed gate starts at `beta_i = 0`, so the model begins as the base attention layer. |
| CLS direct pass-through | The CLS row is not directly projected; it receives SRP effects only through changed patch tokens. |
| Small parameter overhead | The reported WSI runs add only `+0.004%` to `+0.034%` parameters depending on dataset configuration. |

## Where It Lives

| Use case | Implementation |
|---|---|
| Slide-level TransMIL-style Nystrom attention | `slide_level_srp/src/srp_attention.py::NystromSRPAttention` |
| Full-softmax ViT on fixed patch grids | `src/srp_patch_attention.py::PatchSRPAttention` |
| Slide-level model wrapper | `slide_level_srp/src/srp_aggregator.py::NystromSRPAggregator` |
| Shared signed gate | `slide_level_srp/src/gate_signed.py::TokenHeadGate` |
| Neighbor graph and homogeneity inputs | `slide_level_srp/data_ext.py` |

## Reported Evidence

The bundled reference tables show the main trends:

- Gated SRP obtains the best mean case-level C-index on all five TCGA survival
  cohorts in the reported comparison.
- It improves the NA baseline on 12 of 16 reported classification metrics.
- It gives the best mean classification AUC on three of five classification
  datasets.
- The architecture-choice ablation confirms the correction also works in dense
  full-attention settings on ADP patches and PANDA dense ViT.

See [RESULTS.md](RESULTS.md) for the table files and
[REPRODUCING.md](REPRODUCING.md) for exact commands.
