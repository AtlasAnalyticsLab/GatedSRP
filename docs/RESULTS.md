# Results

The `results/` directory contains compact aggregate tables and per-seed records.
Unless stated otherwise, means and sample standard deviations use seeds
`42-46`.

## Classification Tasks

[classification_summary.tsv](../results/classification_summary.tsv)
reports F1, accuracy, and AUC for CAMELYON16, CAMELYON17, KGH, PANDA, and BRACS;
PANDA additionally uses quadratic kappa as its selected task metric.

Across the five selected classification metrics, the mean GatedSRP-minus-NA
change is `+0.0108`: four dataset changes are positive and one is negative.
Across all 16 classification metrics, 12 changes are positive. This is a
descriptive aggregate rather than a claim of uniform improvement; its 95%
confidence interval crosses zero.

Exact paired comparisons are in
[dataset_statistics.tsv](../results/dataset_statistics.tsv), and seed-level
metrics are in
[classification_per_seed.tsv](../results/classification_per_seed.tsv).

## Survival Tasks

[survival_summary.tsv](../results/survival_summary.tsv) reports
case-level C-index for TCGA-KIRC, KIRP, LUAD, STAD, and UCEC.

| Cohort | NA | GatedSRP | Mean change |
|---|---:|---:|---:|
| KIRC | 0.7110 | 0.7257 | +0.0147 |
| KIRP | 0.7247 | 0.7648 | +0.0401 |
| LUAD | 0.5513 | 0.5832 | +0.0319 |
| STAD | 0.5910 | 0.6171 | +0.0261 |
| UCEC | 0.6756 | 0.6973 | +0.0217 |

All five cohort-level mean changes are positive. Their mean paired change is
`+0.0269`, 95% CI `[0.0148, 0.0389]`, `t(4)=6.191`, two-sided `p=0.0035`.
The source values are in
[cross_dataset_statistics.tsv](../results/cross_dataset_statistics.tsv); per-seed
survival metrics are in
[survival_per_seed.tsv](../results/survival_per_seed.tsv).

The selected gate settings are stored in
[survival_selected_settings.tsv](../results/survival_selected_settings.tsv)
and encoded directly in
[survival_tasks.tsv](../configs/survival_tasks.tsv).

## MIL Models

[mil_models.tsv](../results/mil_models.tsv) compares ABMIL, DSMIL,
official TransMIL, and GatedSRP on four selected TCGA cohorts.

| Method | KIRP | LUAD | STAD | UCEC |
|---|---:|---:|---:|---:|
| ABMIL | 0.7224 | 0.5553 | 0.5905 | 0.6529 |
| DSMIL | 0.7253 | 0.5780 | 0.5833 | 0.6621 |
| Official TransMIL | 0.7318 | 0.5422 | 0.5856 | 0.6460 |
| GatedSRP | **0.7648** | **0.5832** | **0.6171** | **0.6973** |

All seed-level values are in
[mil_models_per_seed.tsv](../results/mil_models_per_seed.tsv). The
GatedSRP rows reuse the corresponding survival-task runs rather than retraining
the same configuration under a second name.

## Slide Backbones

[slide_backbones.tsv](../results/slide_backbones.tsv)
contains paired baseline/GatedSRP results for official slide-level SPAN,
Prov-GigaPath LongNet, and dense MHSA on KIRP, LUAD, STAD, PANDA, and BRACS.

The experiment demonstrates that the correction can be inserted into multiple
attention families without a pretrained slide checkpoint or family-specific
GatedSRP tuning. It does not show uniform gains: some pairs improve, some are
effectively unchanged, and some decrease. This distinction is important when
adapting the mechanism to a new architecture.

Dense MHSA uses a deterministic 1,024-patch random retained subset per slide
and seed, shared by the paired arms, followed by nearest retained-coordinate
neighbors. Per-seed values are in
[slide_backbones_per_seed.tsv](../results/slide_backbones_per_seed.tsv).

## Attention Operators And Patch Encoders

These tables vary the token mixer and frozen patch representation while keeping
the associated task protocol fixed.

| Evaluation | Result table | Command manifest |
|---|---|---|
| Dense operators on ADP and PANDA | [attention_operators.tsv](../results/attention_operators.tsv) | [attention_operators.tsv](../configs/attention_operators.tsv) |
| UNI-v2, MedSigLIP-448, and ViT-B/16 | [patch_encoders.tsv](../results/patch_encoders.tsv) | [patch_encoders.tsv](../configs/patch_encoders.tsv) |

## Neighborhood Size

[neighborhood_sizes.tsv](../results/neighborhood_sizes.tsv) compares `3x3`, `5x5`, and
`7x7` cumulative neighborhoods on KIRP, LUAD, STAD, KGH, PANDA, and BRACS. The
`3x3` setting has the best mean selected metric on all six evaluated tasks.
Seed-level values are in
[neighborhood_sizes_per_seed.tsv](../results/neighborhood_sizes_per_seed.tsv).

The larger windows are exact but increase neighbor work and memory. The public
implementation can stream neighbor slots and chunk token corrections for
memory-constrained new runs. The quality manifests retain the reduction order
used by the bundled checkpoints; see [METHOD.md](METHOD.md#scaling-to-native-length-slides)
before changing those flags in a reference reproduction.

## Coefficient Parameterization

[coefficient_parameterizations.tsv](../results/coefficient_parameterizations.tsv) compares the selected
fixed-range gate

```text
beta = delta * tanh(g)
```

with the direct bounded alternative

```text
beta = 2 * tanh(g / 2).
```

The direct parameterization removes the dataset-selected `delta`. It remains
competitive and is slightly higher on PANDA, while the selected fixed range is
higher on the other five tasks. Seed-level values are in
[coefficient_parameterizations_per_seed.tsv](../results/coefficient_parameterizations_per_seed.tsv).

## Runtime and Memory

[runtime_efficiency.tsv](../results/runtime_efficiency.tsv) contains a three-epoch,
seed-42 profile. The table reports peak CUDA reserved memory and synchronized
WSI throughput.

| Scope | Method | Peak GiB | Train WSI/s | Test WSI/s |
|---|---|---:|---:|---:|
| PANDA | NA | 0.47 | 11.806 | 46.588 |
| PANDA | XSA | 0.52 | 12.081 | 53.365 |
| PANDA | Diff | 0.57 | 7.680 | 33.493 |
| PANDA | GatedSRP | 0.49 | 9.790 | 32.962 |
| Five-cohort TCGA mean | NA | 1.80 | 1.910 | - |
| Five-cohort TCGA mean | XSA | 1.92 | 1.903 | - |
| Five-cohort TCGA mean | Diff | 2.08 | 1.896 | - |
| Five-cohort TCGA mean | GatedSRP | 4.69 | 1.925 | - |

The full TCGA GatedSRP path retains per-token local directions, correction
coefficients, and corrected tokens; it does not materialize dense attention
weights. Runtime values are machine-dependent and should be compared only
under the same host, storage, and software conditions.

## Learned Coefficient Regimes

<p align="center">
  <img src="../assets/signed_gate_examples.png" width="900" alt="Examples of learned signed coefficient regimes">
</p>

[coefficient_behavior.tsv](../results/coefficient_behavior.tsv) summarizes 410
final checkpoint exports. Most token means fall between identity and projection,
and no export mean is in the `beta>1.5` reflection bin. The examples show that
individual slides can still express weak negative or above-projection behavior.

## Projection And Gate Components

These comparisons isolate one GatedSRP design variable at a time.

| Evaluation | Result table | Command manifest |
|---|---|---|
| Fixed projection | [projection_variants.tsv](../results/projection_variants.tsv) | [component_variants.tsv](../configs/component_variants.tsv) |
| Gate range | [coefficient_ranges.tsv](../results/coefficient_ranges.tsv) | [component_variants.tsv](../configs/component_variants.tsv) |
| Gate-input gradients | [gradient_paths.tsv](../results/gradient_paths.tsv) | [component_variants.tsv](../configs/component_variants.tsv) |
| Gate factorization | [gate_factorizations.tsv](../results/gate_factorizations.tsv) | [component_variants.tsv](../configs/component_variants.tsv) |
| Gate initialization | [gate_initializations.tsv](../results/gate_initializations.tsv) | [component_variants.tsv](../configs/component_variants.tsv) |

See [REPRODUCING.md](REPRODUCING.md) for launch and collection commands.
