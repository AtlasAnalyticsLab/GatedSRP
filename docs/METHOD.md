# Method

Gated Spatial Redundancy Projection (GatedSRP) is a post-attention correction
for spatially organized pathology tokens. It separates the component of each
patch update that aligns with its local neighborhood and lets a small signed
gate decide how that component should be treated.

<p align="center">
  <img src="../assets/gatedsrp_overview.png" width="960" alt="GatedSRP method overview">
</p>

## Motivation

Adjacent whole-slide image patches often share morphology, stain, texture, and
cell composition. Their feature vectors are therefore more locally similar
than patches in a typical natural image. Repeatedly mixing the common component
can obscure the smaller deviations that carry diagnosis or prognosis.

<p align="center">
  <img src="../assets/local_redundancy.png" width="960" alt="Local feature redundancy in natural images and pathology slides">
</p>

GatedSRP uses this spatial structure without replacing self-attention. The base
attention output, positional mechanism, residual path, and task head remain in
place.

## Local Direction

For patch token `i` and attention head `h`, let `N(i)` be the available patches
in an odd square coordinate window. The default `3x3` window has at most eight
neighbors and excludes the center patch.

```text
r_i,h     = weighted_mean({v_j,h : j in N(i)})
r_hat_i,h = r_i,h / ||r_i,h||
c_i,h     = <y_i,h, r_hat_i,h> r_hat_i,h
```

`v` is the value stream and `y` is the attention update. Missing coordinate
cells are masked. An isolated patch gets a zero local direction and therefore
passes through unchanged.

## Signed Gate

The gate combines token-level local diagnostics with head-level alignment and
magnitude diagnostics:

```text
g_i,h    = MLP_token(d_i) + Linear_head(e_i,h) + b_layer,head
beta_i,h = delta * tanh(g_i,h)
z_i,h    = y_i,h - beta_i,h c_i,h
```

The output path of the gate is initialized to zero. Consequently `g=0`,
`beta=0`, and `z=y` at initialization for every token and head.

| Coefficient | Geometric effect |
|---:|---|
| `< 0` | Adds the neighborhood-aligned component. |
| `0` | Identity; leaves the attention update unchanged. |
| `1` | Removes the neighborhood-aligned component. |
| `2` | Reflects the neighborhood-aligned component. |

The selected dataset configurations use a fixed bound `delta`. The direct
bounded alternative removes this selected range:

```text
beta = 2 * tanh(g / 2)
```

Both parameterizations are exposed through `--gate_delta_mode`; their results
are in [coefficient_parameterizations.tsv](../results/coefficient_parameterizations.tsv).

## Where the Update Is Applied

The default slide-level implementation applies GatedSRP after the attention
aggregation and before the residual update. Only real patch rows are corrected:

- CLS tokens are never directly projected.
- Padded or duplicated rows are not treated as real patches.
- The final post-attention patch-only correction is disabled when no later
  attention block exists for corrected patches to influence CLS.
- Local directions are detached from the correction path while gradients still
  flow through the attention update and signed gate.

The code also exposes controlled pre-Q, pre-K, pre-V, fixed-beta, and learned
local-direction variants used by the component comparisons.

## Scaling to Native-Length Slides

A direct neighbor gather allocates a tensor shaped `(B, H, N, K, D)`. The
released slide implementation computes the same weighted mean by streaming
neighbor slots and can apply the correction in token chunks. This preserves the
operation while reducing peak memory to tensors proportional to `(B, H, N, D)`.

Use:

```text
--srp_context_impl streaming_mean
--srp_correction_chunk_size 32768
```

Set the chunk size to `0` to disable token chunking. Unit tests compare stacked,
streamed, and chunked outputs numerically.

## Learned Regimes

<p align="center">
  <img src="../assets/signed_gate_examples.png" width="900" alt="Examples of signed gate behavior on PANDA and TCGA-KIRC">
</p>

The coefficient is adaptive rather than a fixed pathology label. Across 410
final checkpoint exports, most token means are between identity and projection;
none of the export means exceed `1.5`. Individual examples include weakly
negative PANDA behavior and KIRC behavior above projection strength. The
complete dataset-level fractions are stored in
[coefficient_behavior.tsv](../results/coefficient_behavior.tsv).

## Implementation Map

| Component | File |
|---|---|
| Signed token/head gate | `slide_level_srp/src/gate_signed.py` |
| Nyström attention correction | `slide_level_srp/src/srp_attention.py` |
| TransMIL-style aggregator | `slide_level_srp/src/srp_aggregator.py` |
| Architecture-neutral correction | `slide_level_srp/src/srp_correction.py` |
| Dense MHSA variant | `slide_level_srp/src/dense_srp_aggregator.py` |
| SPAN and LongNet adapters | `slide_level_srp/src/official_architectures.py` |
| Coordinate graph and local homogeneity | `slide_level_srp/data_ext.py` |
| Fixed-grid ViT correction | `src/srp_patch_attention.py` |

See [INTEGRATION.md](INTEGRATION.md) for integration contracts and
[RESULTS.md](RESULTS.md) for quantitative evidence.
