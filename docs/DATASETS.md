# Datasets

This repository includes the label metadata used by the released runs under
`data/labels/`. Raw WSIs and extracted H5 embeddings are not redistributed;
download each dataset from its official source, place raw files under
`data/raw/`, and write AtlasPatch H5 embeddings under `data/features/`.

## Expected Layout

```text
data/
  raw/
    camelyon16/training/{tumor,normal}/
    camelyon17/images/
    kgh/{train,test}/{CP_HP,CP_SSL,CP_TA,CP_TVA}/
    panda/train_images/
    bracs/...
    adp/ADP V1.0 Release/
  labels/
    camelyon16/slides.csv
    camelyon17/stages.csv
    kgh/slides.csv
    panda/train.csv
    bracs/BRACS.xlsx
    tcga_survival/all_matched_survival_labels_long.csv
    adp/ADP_EncodedLabels_Release1_Flat.csv
  features/
    camelyon16/{tumor,normal}/uni_v2/20x_256/patches/*.h5
    camelyon17/patches/*.h5
    kgh/patches/*.h5
    panda/patches/*.h5
    bracs/patches/*.h5
    tcga-to-atlas/A-TCGA-{KIRC,KIRP,LUAD,STAD,UCEC}/40x/uni_v2/20x_256/patches/*.h5
    kgh/{medsiglip_448,vit_b_16}/patches/*.h5
    panda/{medsiglip_448,vit_b_16}/patches/*.h5
    bracs/{medsiglip_448,vit_b_16}/patches/*.h5
```

You can use different roots by editing or sourcing [configs/paths.example.env](../configs/paths.example.env).

## Dataset Sources

| Dataset | Source | Labels Used |
|---|---|---|
| CAMELYON16 | CAMELYON download page via Grand Challenge: https://camelyon17.grand-challenge.org/Download/ | `data/labels/camelyon16/slides.csv`; binary tumor/normal from the official training folders. The official test set is not used. |
| CAMELYON17 | CAMELYON17 Grand Challenge: https://camelyon17.grand-challenge.org/ | `stages.csv`, four classes: negative, ITC, micro, macro. Unlabeled test slides are not used. |
| PANDA | PANDA Grand Challenge page points to Kaggle: https://panda.grand-challenge.org/data/ | `data/labels/panda/train.csv`, ISUP grade 0-5. |
| BRACS | BRACS site: https://www.bracs.icar.cnr.it/ | `BRACS.xlsx`, sheet `WSI_Information`, column `WSI label`. |
| TCGA survival | NCI GDC Data Portal: https://portal.gdc.cancer.gov/ | OS labels from the matched label CSV described below. |
| KGH | Local KGH cohort access is required. | `data/labels/kgh/slides.csv`; four classes: `CP_HP`, `CP_SSL`, `CP_TA`, `CP_TVA`. Normal slides are excluded. |
| ADP | ADP Release1 data. | `data/labels/adp/ADP_EncodedLabels_Release1_Flat.csv` with the official train/valid/test split files from the raw release. |

## Which Data Each Manifest Needs

| Manifest | Required datasets |
|---|---|
| `paper_classification.tsv` | CAMELYON16, CAMELYON17, KGH, PANDA, BRACS |
| `paper_tcga_survival.tsv` | TCGA-KIRC, TCGA-KIRP, TCGA-LUAD, TCGA-STAD, TCGA-UCEC |
| `paper_architecture_ablation.tsv` | ADP Release1 and PANDA |
| `paper_design_ablation.tsv` | KIRP, LUAD, STAD, KGH, PANDA, BRACS, plus selected CAMELYON settings encoded in the manifest |
| `paper_patch_encoder_ablation.tsv` | KGH, PANDA, BRACS, TCGA-KIRP, TCGA-LUAD, TCGA-STAD with the requested encoder features |

## Inclusion Rules

The code mirrors the reported filtering:

- CAMELYON16: labeled training slides only, tumor vs normal.
- CAMELYON17: valid labeled training slides only; missing H5 slides are skipped before splitting.
- KGH: four disease subtype classes only from `data/labels/kgh/slides.csv`; normal slides are excluded.
- PANDA: one zero-patch slide is excluded by H5 validation before split construction.
- BRACS: WSI-level seven-class labels; slides without H5 features are skipped.
- TCGA: OS endpoint, non-positive survival times removed, case-level splitting and case-level C-index.
- ADP: used only for the architecture-choice ablation; it is a raw-RGB
  patch-level multilabel task, not a slide-level WSI benchmark.

## ADP Layout

Place the extracted ADP Release1 directory at:

```text
data/raw/adp/ADP V1.0 Release/
  img_res_1um_bicubic/*.png
  splits/{train,valid,test}.npy
data/labels/adp/ADP_EncodedLabels_Release1_Flat.csv
```

Five level-3 labels with zero training positives are pruned by the loader, so
the ADP ViT emits 38 logits. Override the default paths with `ADP_ROOT`,
`ADP_CSV`, `ADP_IMG_DIR`, and `ADP_SPLITS_DIR` if your files are elsewhere.

## TCGA Label CSV

`slide_level_srp/data_tcga_survival.py` expects a long CSV with at least:

```text
cohort,endpoint,filename,case_barcode,source_case_id,event,time_days,discrete_label,has_nonpositive_time,wsi_path
```

Rows should use cohorts `TCGA_KIRC`, `TCGA_KIRP`, `TCGA_LUAD`, `TCGA_STAD`, and `TCGA_UCEC`; `endpoint` should be `OS`.
