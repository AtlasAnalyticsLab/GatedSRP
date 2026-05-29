#!/usr/bin/env bash
# Generate the TCGA GDC manifest, download the SVS files with gdc-client, and
# arrange them in the raw directory layout used by the AtlasPatch launcher.
# If TCGA_EXISTING_SLIDE_DIRS is set, skip downloading and stage those existing
# files instead. Separate multiple source roots with ":".
set -euo pipefail

LABEL_CSV="${LABEL_CSV:-${TCGA_SURVIVAL_LABEL_CSV:-data/labels/tcga_survival/all_matched_survival_labels_long.csv}}"
RAW_ROOT="${RAW_ROOT:-${TCGA_RAW_ROOT:-data/raw/tcga-to-atlas}}"
MANIFEST="${MANIFEST:-$RAW_ROOT/gdc_manifest_tcga_os.tsv}"
METADATA="${METADATA:-$RAW_ROOT/gdc_slide_metadata_tcga_os.tsv}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${TCGA_GDC_DOWNLOAD_DIR:-$RAW_ROOT/gdc-client-downloads}}"
STAGE_MODE="${STAGE_MODE:-symlink}"
TCGA_EXISTING_SLIDE_DIRS="${TCGA_EXISTING_SLIDE_DIRS:-${TCGA_EXISTING_SLIDE_DIR:-}}"
TCGA_STAGE_DUPLICATE_POLICY="${TCGA_STAGE_DUPLICATE_POLICY:-error}"
TCGA_STAGE_ALLOW_MISSING="${TCGA_STAGE_ALLOW_MISSING:-0}"
TCGA_STAGE_FOLLOW_SYMLINKS="${TCGA_STAGE_FOLLOW_SYMLINKS:-0}"

if [[ -n "$TCGA_EXISTING_SLIDE_DIRS" ]]; then
  stage_cmd=(python scripts/stage_tcga_existing_slides.py
    --label-csv "$LABEL_CSV"
    --endpoint OS
    --out-root "$RAW_ROOT"
    --mode "$STAGE_MODE"
    --duplicate-policy "$TCGA_STAGE_DUPLICATE_POLICY"
    --overwrite)
  if [[ "$TCGA_STAGE_ALLOW_MISSING" == "1" ]]; then
    stage_cmd+=(--allow-missing)
  fi
  if [[ "$TCGA_STAGE_FOLLOW_SYMLINKS" == "1" ]]; then
    stage_cmd+=(--follow-symlinks)
  fi
  IFS=":" read -r -a source_roots <<< "$TCGA_EXISTING_SLIDE_DIRS"
  for source_root in "${source_roots[@]}"; do
    if [[ -n "$source_root" ]]; then
      stage_cmd+=(--source "$source_root")
    fi
  done
  "${stage_cmd[@]}"
  exit 0
fi

if ! command -v gdc-client >/dev/null 2>&1; then
  cat >&2 <<'EOF'
gdc-client was not found on PATH.
Install the GDC Data Transfer Tool from:
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
EOF
  exit 127
fi

python scripts/prepare_tcga_gdc_manifest.py \
  --label-csv "$LABEL_CSV" \
  --endpoint OS \
  --manifest-out "$MANIFEST" \
  --metadata-out "$METADATA"

mkdir -p "$DOWNLOAD_DIR"
gdc-client download -m "$MANIFEST" -d "$DOWNLOAD_DIR"

python scripts/link_tcga_gdc_downloads.py \
  --metadata "$METADATA" \
  --download-dir "$DOWNLOAD_DIR" \
  --out-root "$RAW_ROOT" \
  --mode symlink \
  --overwrite
