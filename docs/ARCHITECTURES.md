# Architecture Integrations

GatedSRP can be placed after the patch-token attention update of another slide
encoder without replacing that encoder's token mixing, positional encoding, or
readout. The public adapters cover official slide-level SPAN and
Prov-GigaPath LongNet; dense MHSA is implemented directly in this repository.

## Prepare Optional Source Trees

```bash
bash scripts/setup_optional_architectures.sh
```

By default, this creates:

```text
external/
  SPAN/
  prov-gigapath/
```

Existing checkouts can remain on shared storage:

```bash
export GATEDSRP_SPAN_ROOT=/shared/code/SPAN
export GATEDSRP_GIGAPATH_ROOT=/shared/code/prov-gigapath
```

The setup script checks out SPAN revision
`08e4ba08900f151d6b618d5e13595a1ab2f12164` and Prov-GigaPath revision
`3505f87e197d167522be491bb3f18fb5a08ca584`, matching the evaluated source.

## SPAN

`OfficialSPANAggregator` uses SPAN's slide-level encoder and task aggregators.
It converts H5 coordinates to SPAN's integer grid, preserves SPAN's global
tokens and sparse blocks, and inserts a zero-initialized correction into the
patch-token transformer update for the GatedSRP arm.

The completed runs used OmegaConf 2.3.0, torchdata 0.7.1, and the Linux
`dgl==2.1.0+cu121` wheel alongside PyTorch 2.6.0. Install the matching CUDA
wheel for your platform; the evaluated environment used:

```bash
python -m pip install omegaconf==2.3.0 torchdata==0.7.1
python -m pip install 'dgl==2.1.0+cu121' \
  -f https://data.dgl.ai/wheels/cu121/repo.html
```

The adapter installs a process-local GraphBolt import stub because this SPAN
path uses DGL graph construction, message passing, and `edge_softmax`, but not
GraphBolt. It does not modify the installed DGL package.

```bash
python scripts/run_manifest.py configs/slide_backbones.tsv \
  --dataset=KIRP --architecture=SPAN \
  --variant=baseline --seed=42 --dry-run
```

## Prov-GigaPath LongNet

`OfficialGigaPathLongNetAggregator` instantiates the official `LongNetViT`
encoder with its coordinate positional table and final CLS readout. The paired
GatedSRP arm corrects patch-token attention updates before the residual path.
The released commands use the official 12-layer, 768-dimensional
configuration.

WSI-scale execution requires `flash-attn` or xFormers. The completed runs used
`xformers==0.0.29.post3`, installed without dependency resolution so it did not
replace the pinned PyTorch build:

```bash
python -m pip install xformers==0.0.29.post3 --no-deps
```

A standard PyTorch attention fallback is available only for small shape checks:

```bash
export GATEDSRP_ALLOW_LONGNET_TORCH_FALLBACK=1
```

That fallback materializes dense attention scores and should not be used for
native-length WSI experiments.

After installing optional dependencies, inspect both backends before launching
a full group:

```bash
python - <<'PY'
from slide_level_srp.src.official_architectures import (
    official_architecture_dependency_status,
)

print(official_architecture_dependency_status())
PY
```

The expected values begin with `ready` for both `span` and `longnet`.

```bash
python scripts/run_manifest.py configs/slide_backbones.tsv \
  --dataset=STAD --architecture='Prov-GigaPath LongNet' \
  --variant=gated_srp --seed=42 --dry-run
```

## Dense MHSA

`DenseAttentionSRPAggregator` replaces Nyström attention with standard dense
multi-head self-attention. The comparison keeps at most 1,024 patches per
slide. A deterministic random subset is selected from each slide for each seed,
and both paired arms use that same subset. The neighborhood graph is then
rebuilt from nearest retained coordinates.

```bash
python scripts/run_manifest.py configs/slide_backbones.tsv \
  --dataset=BRACS --architecture='Dense MHSA' \
  --variant=gated_srp --seed=42 --dry-run
```

## Training Policy

No pretrained slide-level checkpoint is loaded for these comparisons. Each
family is trained with the same frozen patch embeddings, task split, optimizer
schedule, and task head as its paired arm. GatedSRP uses the same selected gate
configuration as the corresponding dataset task run; there is no
architecture-specific gate search.

Results are deliberately interpreted as compatibility evidence. They are mixed
across architecture families and datasets, so the table does not imply that
adding GatedSRP must improve every external model.

See [slide_backbones.tsv](../results/slide_backbones.tsv) for
aggregate values and
[slide_backbones_per_seed.tsv](../results/slide_backbones_per_seed.tsv)
for all 150 seed-level results.
