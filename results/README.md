# Reference Tables

This directory contains the aggregate and seed-level values used to validate a
fresh run. Metrics are mean +/- sample standard deviation across five seeds
unless a table states otherwise.

## Prediction Tasks

| File | Purpose |
|---|---|
| `classification_summary.tsv` | Classification comparison across CAMELYON16/17, KGH, PANDA, and BRACS. |
| `classification_per_seed.tsv` | Seed-level classification metrics for all 100 task rows. |
| `survival_summary.tsv` | Case-level C-index comparison across five TCGA cohorts. |
| `survival_per_seed.tsv` | Seed-level TCGA survival metrics for all 100 task rows. |
| `survival_selected_settings.tsv` | Cohort-specific hyperparameters for the GatedSRP survival runs. |

## Attention And MIL Architectures

| File | Purpose |
|---|---|
| `attention_operators.tsv` | Dense attention operators evaluated on ADP and PANDA. |
| `slide_backbones.tsv` | Aggregate SPAN, Prov-GigaPath LongNet, and dense-MHSA compatibility results. |
| `slide_backbones_per_seed.tsv` | Seed-level slide-backbone values. |
| `mil_models.tsv` | ABMIL, DSMIL, TransMIL, and GatedSRP survival comparison. |
| `mil_models_per_seed.tsv` | Seed-level values for all four MIL comparison methods. |

## Representations And Spatial Context

| File | Purpose |
|---|---|
| `patch_encoders.tsv` | UNI-v2, MedSigLIP-448, and ViT-B/16 patch encoders. |
| `neighborhood_sizes.tsv` | 3x3, 5x5, and 7x7 local-neighborhood comparison. |
| `neighborhood_sizes_per_seed.tsv` | Seed-level local-neighborhood values. |

## GatedSRP Design

| File | Purpose |
|---|---|
| `projection_variants.tsv` | Fixed projection-strength variants. |
| `coefficient_ranges.tsv` | Signed versus nonnegative coefficient ranges. |
| `gradient_paths.tsv` | Detached versus live gate-input gradients. |
| `gate_factorizations.tsv` | Token, head, and bias gate factorization. |
| `gate_initializations.tsv` | Gate-output initialization. |
| `coefficient_parameterizations.tsv` | Fixed-range and directly parameterized signed-coefficient comparison. |
| `coefficient_parameterizations_per_seed.tsv` | Seed-level coefficient-parameterization values. |
| `coefficient_behavior.tsv` | Distribution of learned coefficient regimes across checkpoint exports. |

## Efficiency And Statistics

| File | Purpose |
|---|---|
| `runtime_efficiency.tsv` | Peak reserved memory and train/test throughput. |
| `cross_dataset_statistics.tsv` | Across-dataset paired summaries. |
| `dataset_statistics.tsv` | Per-dataset paired values used by the statistical analysis. |

These TSVs contain aggregate or experimental metrics only. Restricted KGH
labels, slide identifiers, images, and embeddings are not distributed.
