# Datasets

This repository includes the label metadata used by the released runs under
`data/labels/`. Raw WSIs and extracted H5 embeddings are not redistributed. The
examples below use `data/raw/` and `data/features/` as a compact local
convention, but the scripts do not require that layout: raw datasets can stay
on shared server storage, and the training manifests can point to absolute
feature paths through environment variables.

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

## Existing Server Storage

If the datasets already exist on your server, do not move or duplicate them just
to match the example tree. Use one of these patterns:

```bash
cp configs/paths.example.env .env.local
```

Edit `.env.local` so raw and feature roots point to your storage, for example:

```bash
export CAM17_RAW_ROOT=/shared/pathology/CAMELYON17/images
export CAM17_UNIV2_ROOT=/scratch/features/camelyon17/patches
export PANDA_RAW_ROOT=/shared/pathology/PANDA/train_images
export PANDA_FEATURE_ROOT=/scratch/features/panda/patches
export TCGA_RAW_ROOT=/scratch/gatedsrp/tcga-to-atlas
export TCGA_FEATURE_ROOT=/scratch/features/tcga-to-atlas
```

Then source it before extraction, validation, or training:

```bash
source .env.local
```

For TCGA slides that were already downloaded from GDC, stage symlinks from the
existing storage root into the cohort layout expected by the AtlasPatch launcher:

```bash
python scripts/stage_tcga_existing_slides.py \
  --source /shared/gdc/tcga-slides \
  --label-csv data/labels/tcga_survival/all_matched_survival_labels_long.csv \
  --out-root "$TCGA_RAW_ROOT" \
  --mode symlink
```

If the existing slides are split across multiple roots, repeat `--source`. The
same path can also be used through the TCGA helper:

```bash
TCGA_EXISTING_SLIDE_DIRS="/shared/gdc/root1:/shared/gdc/root2" \
TCGA_RAW_ROOT="$TCGA_RAW_ROOT" \
bash scripts/download_tcga_slides.sh
```

When `TCGA_EXISTING_SLIDE_DIRS` is set, the helper skips `gdc-client` download
and only stages links for the exact OS slides in the released label CSV. If the
source tree itself contains symlinked directories, set
`TCGA_STAGE_FOLLOW_SYMLINKS=1`; if duplicate filenames are expected, inspect
them or set `TCGA_STAGE_DUPLICATE_POLICY=first`.

## Dataset Sources

| Dataset | Source | Labels Used |
|---|---|---|
| CAMELYON16 | CAMELYON download page via Grand Challenge: https://camelyon17.grand-challenge.org/Download/ | `data/labels/camelyon16/slides.csv`; binary tumor/normal from the official training folders. The official test set is not used. |
| CAMELYON17 | CAMELYON17 Grand Challenge: https://camelyon17.grand-challenge.org/ | `stages.csv`, four classes: negative, ITC, micro, macro. Unlabeled test slides are not used. |
| PANDA | PANDA Grand Challenge page points to Kaggle: https://panda.grand-challenge.org/data/ | `data/labels/panda/train.csv`, ISUP grade 0-5. |
| BRACS | BRACS site: https://www.bracs.icar.cnr.it/ | `BRACS.xlsx`, sheet `WSI_Information`, column `WSI label`. |
| TCGA survival | NCI GDC Data Portal: https://portal.gdc.cancer.gov/ | OS labels from the matched label CSV described below. Use `scripts/download_tcga_slides.sh` or the manual GDC manifest workflow below to download the exact SVS files. |
| KGH | Private cohort; local access is required. | Labels are not redistributed. The loader derives the four private class labels from `${KGH_RAW_ROOT}/{train,test}/{CP_HP,CP_SSL,CP_TA,CP_TVA}/`; normal slides are excluded. |
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
- KGH: four disease subtype class folders only; normal slides are excluded and no KGH labels are shipped.
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

## TCGA Slide Download

The TCGA label CSV includes the exact SVS filenames used by the survival
experiments. You can generate a GDC manifest from those filenames with the
public GDC Files API
(https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/) and
download the matching open-access slide images with the GDC Data Transfer Tool.

Install `gdc-client` from the GDC Data Transfer Tool page:
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

Then run the end-to-end helper:

```bash
bash scripts/download_tcga_slides.sh
```

The full OS manifest contains 2,025 SVS files and is about 1,975.66 GiB, so run
the command on storage suitable for roughly 2 TB of WSI data. `gdc-client` can
be rerun with the same manifest to resume interrupted downloads.

The helper performs three steps:

```bash
python scripts/prepare_tcga_gdc_manifest.py \
  --label-csv data/labels/tcga_survival/all_matched_survival_labels_long.csv \
  --endpoint OS \
  --manifest-out data/raw/tcga-to-atlas/gdc_manifest_tcga_os.tsv \
  --metadata-out data/raw/tcga-to-atlas/gdc_slide_metadata_tcga_os.tsv

gdc-client download \
  -m data/raw/tcga-to-atlas/gdc_manifest_tcga_os.tsv \
  -d data/raw/tcga-to-atlas/gdc-client-downloads

python scripts/link_tcga_gdc_downloads.py \
  --metadata data/raw/tcga-to-atlas/gdc_slide_metadata_tcga_os.tsv \
  --download-dir data/raw/tcga-to-atlas/gdc-client-downloads \
  --out-root data/raw/tcga-to-atlas \
  --mode symlink \
  --overwrite
```

After linking, the raw slide layout is:

```text
data/raw/tcga-to-atlas/
  A-TCGA-KIRC/*.svs
  A-TCGA-KIRP/*.svs
  A-TCGA-LUAD/*.svs
  A-TCGA-STAD/*.svs
  A-TCGA-UCEC/*.svs
```

If the slides are already downloaded elsewhere, skip the download and stage
symlinks instead:

```bash
python scripts/stage_tcga_existing_slides.py \
  --source /shared/gdc/tcga-slides \
  --out-root data/raw/tcga-to-atlas \
  --mode symlink
```

If the GDC API changes or a network policy blocks API access, use the GDC Data
Portal manually: filter to each TCGA project, select `Slide Image` / `SVS`
files whose names appear in `data/labels/tcga_survival/all_matched_survival_labels_long.csv`
for `endpoint=OS` and `has_nonpositive_time=False`, download a manifest from
the cart, then continue from the `gdc-client download` and linking steps above.
