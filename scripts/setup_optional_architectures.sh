#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${GATEDSRP_EXTERNAL_DIR:-${ROOT_DIR}/external}"
SPAN_DIR="${GATEDSRP_SPAN_ROOT:-${EXTERNAL_DIR}/SPAN}"
GIGAPATH_DIR="${GATEDSRP_GIGAPATH_ROOT:-${EXTERNAL_DIR}/prov-gigapath}"

SPAN_COMMIT="08e4ba08900f151d6b618d5e13595a1ab2f12164"
GIGAPATH_COMMIT="3505f87e197d167522be491bb3f18fb5a08ca584"

mkdir -p "${EXTERNAL_DIR}"

if [[ ! -d "${SPAN_DIR}/.git" ]]; then
  git clone https://github.com/wwyi1828/SPAN.git "${SPAN_DIR}"
fi
git -C "${SPAN_DIR}" fetch origin "${SPAN_COMMIT}"
git -C "${SPAN_DIR}" checkout --detach "${SPAN_COMMIT}"

if [[ ! -d "${GIGAPATH_DIR}/.git" ]]; then
  git clone https://github.com/prov-gigapath/prov-gigapath.git "${GIGAPATH_DIR}"
fi
git -C "${GIGAPATH_DIR}" fetch origin "${GIGAPATH_COMMIT}"
git -C "${GIGAPATH_DIR}" checkout --detach "${GIGAPATH_COMMIT}"

printf 'SPAN: %s\n' "${SPAN_DIR}"
printf 'Prov-GigaPath: %s\n' "${GIGAPATH_DIR}"
printf '%s\n' 'Install the optional Python/CUDA dependencies described in docs/ARCHITECTURES.md.'
