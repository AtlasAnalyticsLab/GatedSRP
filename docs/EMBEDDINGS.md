# Embeddings

The WSI task and comparison runs use frozen 20x, 256-pixel AtlasPatch UNI-v2
patch embeddings. The patch-encoder comparison also uses MedSigLIP-448 and
ViT-B/16 features in the same one-H5-file-per-WSI format. ADP is the
exception: it loads raw RGB patch PNGs directly.

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

Install the native OpenSlide library, then AtlasPatch in the environment used
for feature extraction. Conda users already receive OpenSlide through
`environment.yml`; on Ubuntu or Debian use:

```bash
sudo apt-get install openslide-tools
```

Install the Python package and SAM2:

```bash
python -m pip install atlas-patch
python -m pip install git+https://github.com/facebookresearch/sam2.git
export HF_TOKEN=your_huggingface_token
```

For `uv`:

```bash
uv pip install atlas-patch
uv pip install git+https://github.com/facebookresearch/sam2.git
```

UNI-v2 is gated on Hugging Face. Request access to `MahmoodLab/UNI2-h` and set
`HF_TOKEN` before extraction. AtlasPatch skips an existing per-slide H5 by
default; pass `--force` to the launcher only when you intend to rebuild it.

Then run the dataset-aware launcher. It calls `atlaspatch process` with
`--patch-size 256 --target-mag 20 --feature-extractors uni_v2` and writes the
H5 files into the directory layout consumed by the trainers. `--input` and
`--output` can be absolute paths on shared storage; the examples use repo-local
paths only as placeholders.

```bash
# CAMELYON17: writes data/features/camelyon17/patches/*.h5
python scripts/extract_atlaspatch_embeddings.py \
  --dataset camelyon17 \
  --input "${CAM17_RAW_ROOT:-data/raw/camelyon17/images}" \
  --output "$(dirname "${CAM17_UNIV2_ROOT:-data/features/camelyon17/patches}")"

# PANDA: writes data/features/panda/patches/*.h5
python scripts/extract_atlaspatch_embeddings.py \
  --dataset panda \
  --input "${PANDA_RAW_ROOT:-data/raw/panda/train_images}" \
  --output "$(dirname "${PANDA_FEATURE_ROOT:-data/features/panda/patches}")"

# KGH: private cohort; stages the four class folders and excludes Normal.
python scripts/extract_atlaspatch_embeddings.py \
  --dataset kgh \
  --input "${KGH_RAW_ROOT:-data/raw/kgh}" \
  --output "$(dirname "${KGH_FEATURE_ROOT:-data/features/kgh/patches}")"

# BRACS: point --input at the directory containing BRACS WSIs.
python scripts/extract_atlaspatch_embeddings.py \
  --dataset bracs \
  --input "${BRACS_RAW_ROOT:-data/raw/bracs}" \
  --output "$(dirname "${BRACS_FEATURE_ROOT:-data/features/bracs/patches}")"
```

CAMELYON16 keeps the class-specific feature folders expected by the loader:

```bash
python scripts/extract_atlaspatch_embeddings.py \
  --dataset camelyon16 \
  --input "${CAM16_RAW_ROOT:-data/raw/camelyon16/training}" \
  --output "${CAM16_UNIV2_ROOT:-data/features/camelyon16}"
```

If CAMELYON16 normal and tumor slides are stored in separate server paths, pass
them explicitly:

```bash
python scripts/extract_atlaspatch_embeddings.py \
  --dataset camelyon16 \
  --input /unused/when/class/dirs/are/set \
  --cam16-normal-dir /shared/CAMELYON16/normal-slides \
  --cam16-tumor-dir /shared/CAMELYON16/tumor-slides \
  --output "${CAM16_UNIV2_ROOT:-data/features/camelyon16}"
```

TCGA survival uses one output tree per cohort:

```bash
# First download, or stage already downloaded, raw TCGA SVS files.
bash scripts/download_tcga_slides.sh

# Then extract one AtlasPatch H5 tree per cohort.
for cohort in KIRC KIRP LUAD STAD UCEC; do
  python scripts/extract_atlaspatch_embeddings.py \
    --dataset tcga \
    --cohort "$cohort" \
    --input "${TCGA_RAW_ROOT:-data/raw/tcga-to-atlas}/A-TCGA-${cohort}" \
    --output "${TCGA_FEATURE_ROOT:-data/features/tcga-to-atlas}"
done
```

The generic shell wrapper remains available for one-off extraction:

```bash
INPUT_DIR="${CAM17_RAW_ROOT:-data/raw/camelyon17/images}" \
OUTPUT_DIR="$(dirname "${CAM17_UNIV2_ROOT:-data/features/camelyon17/patches}")" \
bash scripts/extract_with_atlaspatch.sh
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

## Patch-Encoder Features

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
