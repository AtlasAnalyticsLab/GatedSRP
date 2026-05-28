# Embeddings

The main WSI runs use frozen 20x, 256-pixel AtlasPatch UNI-v2 patch
embeddings. The patch-encoder ablation also uses MedSigLIP-448 and ViT-B/16
features in the same one-H5-file-per-WSI format. ADP is the exception: it
loads raw RGB patch PNGs directly.

## H5 Contract

Each H5 file must contain:

```text
/coords            shape (N, >=2), integer level-0 patch coordinates
/features/uni_v2   shape (N, 1536), float patch embeddings
```

`patch_size_level0` is read from H5 attributes when present. The code falls back to dataset-specific strides when the attribute is missing.

Validate a feature directory before training:

```bash
python scripts/validate_h5_embeddings.py \
  --root "$CAM17_UNIV2_ROOT" \
  --feature-key features/uni_v2 \
  --expected-dim 1536
```

## Extraction Template

Install AtlasPatch in the same environment, then run:

```bash
INPUT_DIR=data/raw/camelyon17/images \
OUTPUT_DIR=data/features/camelyon17 \
bash scripts/extract_with_atlaspatch.sh
```

AtlasPatch should write `data/features/camelyon17/patches/*.h5`. Point the corresponding environment variable at the `patches` directory:

```bash
export CAM17_UNIV2_ROOT=data/features/camelyon17/patches
```

For large cohorts, run extraction once and treat the H5 files as immutable
inputs to the training manifests. The row order of `/coords` and every
`/features/<encoder>` matrix must match because the neighbor graph indexes rows
directly.

## Dataset-Specific Targets

| Dataset | Output root passed to trainer |
|---|---|
| CAMELYON16 | `CAM16_UNIV2_ROOT=data/features/camelyon16` with class subfolders `tumor/uni_v2/20x_256/patches` and `normal/uni_v2/20x_256/patches`. |
| CAMELYON17 | `CAM17_UNIV2_ROOT=data/features/camelyon17/patches`. |
| KGH | `KGH_FEATURE_ROOT=data/features/kgh/patches`. |
| PANDA | `PANDA_FEATURE_ROOT=data/features/panda/patches`. |
| BRACS | `BRACS_FEATURE_ROOT=data/features/bracs/patches`. |
| TCGA | `TCGA_FEATURE_ROOT=data/features/tcga-to-atlas`; the trainer adds `A-TCGA-{cohort}/40x/uni_v2/20x_256/patches`. |

If your extraction tool writes a different key, either normalize to `/features/uni_v2` or set the matching `*_FEATURE_KEY` variable and keep `--in_dim 1536`.

## Patch-Encoder Ablation Keys

The patch-encoder manifest expects the following feature dimensions and H5
keys:

| Encoder | H5 key | `--in_dim` |
|---|---|---:|
| UNI-v2 | `features/uni_v2` | 1536 |
| MedSigLIP-448 | `features/medsiglip` | 1152 |
| ViT-B/16 | `features/vit_b_16` | 768 |

For KGH, PANDA, and BRACS, point the corresponding root variables in
`configs/paths.example.env` at the encoder-specific `patches/` directory. For
TCGA, keep the same `TCGA_FEATURE_ROOT` tree and set the encoder-specific
feature key variables if your H5 key names differ.
