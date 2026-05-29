# Released Label Metadata

This directory contains the label metadata consumed by the released manifests.
Raw WSIs are not included.

| Dataset | File |
|---|---|
| CAMELYON16 | `camelyon16/slides.csv` |
| CAMELYON17 | `camelyon17/stages.csv` |
| PANDA | `panda/train.csv` |
| BRACS | `bracs/BRACS.xlsx` |
| TCGA survival | `tcga_survival/all_matched_survival_labels_long.csv` |
| ADP | `adp/ADP_EncodedLabels_Release1_Flat.csv` |

KGH labels are not included because KGH is a private cohort. For local KGH
runs, place the private raw slides under the class-folder layout documented in
`docs/DATASETS.md`, or pass a private `KGH_LABEL_CSV` outside this repository.
