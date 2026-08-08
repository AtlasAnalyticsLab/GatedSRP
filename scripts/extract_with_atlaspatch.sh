#!/usr/bin/env bash
# Minimal AtlasPatch UNI-v2 extraction template.
#
# This intentionally stays dataset-agnostic: point INPUT_DIR at a folder of raw
# WSIs and OUTPUT_DIR at the dataset-specific feature directory documented in
# docs/EMBEDDINGS.md. AtlasPatch writes one H5 per slide under
# "$OUTPUT_DIR/patches" with /coords and /features/uni_v2.
#
# For the supported dataset layouts, prefer:
#   python scripts/extract_atlaspatch_embeddings.py --dataset <name> ...
set -euo pipefail

: "${INPUT_DIR:?set INPUT_DIR to a directory of WSI files}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the AtlasPatch output directory}"

FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-24}"
FEATURE_NUM_WORKERS="${FEATURE_NUM_WORKERS:-2}"
PATCH_WORKERS="${PATCH_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
ATLASPATCH_FORCE="${ATLASPATCH_FORCE:-0}"

force_args=()
if [[ "$ATLASPATCH_FORCE" == "1" ]]; then
  force_args+=(--force)
fi

atlaspatch process "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --patch-size 256 \
  --target-mag 20 \
  --feature-extractors uni_v2 \
  --feature-batch-size "$FEATURE_BATCH_SIZE" \
  --feature-num-workers "$FEATURE_NUM_WORKERS" \
  --feature-precision "${FEATURE_PRECISION:-float16}" \
  --patch-workers "$PATCH_WORKERS" \
  --device "$DEVICE" \
  --feature-device "$DEVICE" \
  "${force_args[@]}"
