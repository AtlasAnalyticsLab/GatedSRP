# Reference Result Tables

Reference TSVs are stored in [results](../results).

## Headline Summary

| Claim checked by the bundled tables | Where to inspect |
|---|---|
| Best mean case-level C-index on all five TCGA survival cohorts | [tcga_survival_main_table.tsv](../results/tcga_survival_main_table.tsv) |
| Improved NA on 12 of 16 reported classification metrics | [classification_main_table.tsv](../results/classification_main_table.tsv) |
| Best mean AUC on three of five classification datasets | [classification_main_table.tsv](../results/classification_main_table.tsv) |
| Works in dense full-attention settings, not only Nystrom attention | [ablation_architecture_choice.tsv](../results/ablation_architecture_choice.tsv) |

## Classification

[classification_main_table.tsv](../results/classification_main_table.tsv) contains mean +/- sample standard deviation over five seeds for:

- CAMELYON16: F1, accuracy, AUC
- CAMELYON17: F1, accuracy, AUC
- KGH: F1, accuracy, AUC
- PANDA: quadratic kappa, F1, accuracy, AUC
- BRACS: F1, accuracy, AUC

PANDA uses the TransMIL architecture path in `python -m slide_level_srp.train_panda --arch transmil`, matching the reference table.

## TCGA Survival

[tcga_survival_main_table.tsv](../results/tcga_survival_main_table.tsv) reports case-level C-index for TCGA-KIRC, KIRP, LUAD, STAD, and UCEC.

The reported Gated SRP configs are:

```text
KIRC: delta_scale=1.5, gate_hidden_dim=128
KIRP: delta_scale=2.0, gate_hidden_dim=32
LUAD: delta_scale=1.0, gate_hidden_dim=128
STAD: delta_scale=0.5, gate_hidden_dim=128
UCEC: delta_scale=1.5, gate_hidden_dim=32
```

These are encoded directly in [configs/paper_tcga_survival.tsv](../configs/paper_tcga_survival.tsv).

## Ablations

The latest reported ablation tables are stored as compact TSVs:

- [ablation_fixed_projection.tsv](../results/ablation_fixed_projection.tsv)
- [ablation_gate_range.tsv](../results/ablation_gate_range.tsv)
- [ablation_gate_gradients.tsv](../results/ablation_gate_gradients.tsv)
- [ablation_gate_factorization.tsv](../results/ablation_gate_factorization.tsv)
- [ablation_gate_initialization.tsv](../results/ablation_gate_initialization.tsv)
- [ablation_patch_encoder.tsv](../results/ablation_patch_encoder.tsv)
- [ablation_architecture_choice.tsv](../results/ablation_architecture_choice.tsv)

Their corresponding runnable manifests are:

- [paper_design_ablation.tsv](../configs/paper_design_ablation.tsv)
- [paper_patch_encoder_ablation.tsv](../configs/paper_patch_encoder_ablation.tsv)
- [paper_architecture_ablation.tsv](../configs/paper_architecture_ablation.tsv)

ADP in the architecture-choice table uses the official Release1 patch split.
PANDA in that table uses the dense `vit4` slide model; the main PANDA table uses
the TransMIL path.
