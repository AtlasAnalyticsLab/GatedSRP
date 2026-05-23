# Paper Result Tables

Reference TSVs are stored in [results/paper](../results/paper).

## Classification

[classification_main_table.tsv](../results/paper/classification_main_table.tsv) contains the table values for:

- CAMELYON16: F1, accuracy, AUC
- CAMELYON17: F1, accuracy, AUC
- KGH: F1, accuracy, AUC
- PANDA: quadratic kappa, F1, accuracy, AUC
- BRACS: F1, accuracy, AUC

PANDA uses the TransMIL architecture path in `python -m panda.train --arch transmil`, matching the paper table.

## TCGA Survival

[tcga_survival_main_table.tsv](../results/paper/tcga_survival_main_table.tsv) reports case-level C-index for TCGA-KIRC, KIRP, LUAD, STAD, and UCEC.

The reported Gated SRP configs are:

```text
KIRC: delta_scale=1.5, gate_hidden_dim=128
KIRP: delta_scale=2.0, gate_hidden_dim=32
LUAD: delta_scale=1.0, gate_hidden_dim=128
STAD: delta_scale=0.5, gate_hidden_dim=128
UCEC: delta_scale=1.5, gate_hidden_dim=32
```

These are encoded directly in [configs/paper_tcga_survival.tsv](../configs/paper_tcga_survival.tsv).
