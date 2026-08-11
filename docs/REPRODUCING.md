# Reproducing the Evaluations

Every released run is an explicit row in `configs/*.tsv`. A row fixes the
dataset, method, seed, split protocol, model arguments, and output name. The
recommended workflow is to prepare one dataset, dry-run one row, execute it,
inspect its artifacts, and only then launch the full five-seed group.

## 1. Create the Environment

Choose one setup path. The package pins match the completed runs. The examples
below use PyTorch 2.6.0 with CUDA 12.4.

### Conda

```bash
conda env create -f environment.yml
conda activate gatedsrp
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

### venv + pip

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

### uv

```bash
python -m pip install uv
uv venv --python 3.10
source .venv/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
uv pip install --no-deps -e .
```

For CPU tests, install the matching CPU wheel instead. Full WSI training is
intended for CUDA GPUs.

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

```bash
python -m pytest tests -q
```

## 2. Prepare Data and Paths

Follow [DATASETS.md](DATASETS.md) to acquire raw slides and
[EMBEDDINGS.md](EMBEDDINGS.md) to generate AtlasPatch H5 files. Public labels
used by the manifests are under `data/labels/`.

The data do not need to live inside the repository:

```bash
cp configs/paths.example.env .env.local
# Edit roots to point at local or shared storage.
source .env.local
```

At minimum, each WSI needs an H5 file containing `/coords` and the configured
feature key. Validate each inventory before training:

```bash
python scripts/validate_h5_embeddings.py \
  --root "$PANDA_FEATURE_ROOT" \
  --feature-key features/uni_v2 \
  --expected-dim 1536
```

KGH rows are marked `access=restricted`. They remain useful to authorized
users, but cannot be reproduced from this repository alone because KGH labels,
slides, and embeddings are not distributed.

## 3. Select a Manifest

The manifests are indexed by the experimental variable they evaluate:

| Type | Evaluation | Rows | Manifest | Default output root |
|---|---|---:|---|---|
| Prediction task | WSI classification | 100 | `configs/classification_tasks.tsv` | `runs/classification` |
| Prediction task | TCGA overall survival | 100 | `configs/survival_tasks.tsv` | `runs/survival_tasks` |
| Attention | Dense operators on ADP and PANDA | 20 | `configs/attention_operators.tsv` | `runs/attention_operators` |
| Slide backbone | SPAN, LongNet, and dense MHSA | 150 | `configs/slide_backbones.tsv` | `runs/slide_backbones` |
| MIL model | ABMIL, DSMIL, and TransMIL | 60 | `configs/mil_models.tsv` | `runs/mil_models` |
| Patch representation | UNI-v2, MedSigLIP-448, and ViT-B/16 | 180 | `configs/patch_encoders.tsv` | `runs/patch_encoders` |
| GatedSRP design | Projection, range, gradients, factorization, and initialization | 480 | `configs/component_variants.tsv` | `runs/component_variants` |
| Spatial context | `3x3`, `5x5`, and `7x7` neighborhoods | 90 | `configs/neighborhood_sizes.tsv` | `runs/neighborhood_sizes` |
| Coefficient design | Selected range versus direct bounded beta | 60 | `configs/coefficient_parameterizations.tsv` | `runs/coefficient_parameterizations` |
| Efficiency | Three-epoch PANDA and five-cohort TCGA profiles | 24 | `configs/runtime_efficiency.tsv` | `runs/runtime_efficiency` |

Full quality evaluations use seeds `42, 43, 44, 45, 46`. The efficiency profile uses
seed 42 and three epochs because it measures execution rather than final model
quality.

## 4. Dry-Run and Launch

`run_manifest.py` accepts exact `column=value` filters. It prints the selected
command before execution.

```bash
# One public classification row
python scripts/run_manifest.py configs/classification_tasks.tsv \
  --where dataset=cam16 --where method=baseline --where seed=42 --dry-run

# One TCGA survival row
python scripts/run_manifest.py configs/survival_tasks.tsv \
  --where cohort=KIRP --where method=gated_post_attention --where seed=42 --dry-run

# One neighborhood-size row
python scripts/run_manifest.py configs/neighborhood_sizes.tsv \
  --where dataset=LUAD --where window=3x3 --where seed=42 --dry-run
```

Remove `--dry-run` to execute. Run all publicly accessible rows with the access
filter:

```bash
python scripts/run_manifest.py configs/survival_tasks.tsv --where access=public
python scripts/run_manifest.py configs/neighborhood_sizes.tsv --where access=public
```

Authorized KGH users can omit the access filter. Every manifest uses the same
`public`/`restricted` convention.

Use `--start-at <run_name>` to resume a manifest from a named row. Existing run
directories are not silently deleted; choose a new output root or remove an
incomplete run explicitly before relaunching it.

## 5. Optional Architectures

SPAN and Prov-GigaPath LongNet are not vendored. Prepare their pinned source
checkouts:

```bash
bash scripts/setup_optional_architectures.sh
```

SPAN requires OmegaConf and a DGL build compatible with the installed
PyTorch/CUDA version. LongNet requires flash-attn or xFormers at WSI scale. See
[ARCHITECTURES.md](ARCHITECTURES.md) before running
`configs/slide_backbones.tsv`.

Dense MHSA rows use at most 1,024 patches. For each slide and seed, the baseline
and GatedSRP arms receive the same deterministic random retained subset; the
local graph is rebuilt using nearest retained coordinates.

## 6. Collect Results

The task collector handles classification and survival manifests:

```bash
python scripts/collect_task_results.py --public-only --strict
```

This writes task summaries to `results/rerun/`.

For a first one-row run, repeat the launch filters during collection. Filters
are applied before `--strict` checks, so this validates exactly the selected
artifact instead of requiring the full matrix:

```bash
python scripts/collect_task_results.py \
  --where cohort=KIRP \
  --where method=gated_post_attention \
  --where seed=42 \
  --public-only --strict
```

The generic collector accepts each typed comparison manifest and its output
root:

```bash
python scripts/collect_comparison_results.py configs/attention_operators.tsv \
  --run-root "${GATEDSRP_ATTENTION_OUT:-runs/attention_operators}" \
  --public-only --strict

python scripts/collect_comparison_results.py configs/component_variants.tsv \
  --run-root "${GATEDSRP_COMPONENT_OUT:-runs/component_variants}" \
  --public-only --strict

python scripts/collect_comparison_results.py configs/patch_encoders.tsv \
  --run-root "${GATEDSRP_PATCH_ENCODER_OUT:-runs/patch_encoders}" \
  --public-only --strict

python scripts/collect_comparison_results.py configs/slide_backbones.tsv \
  --run-root "${GATEDSRP_BACKBONE_OUT:-runs/slide_backbones}" \
  --public-only --strict

python scripts/collect_comparison_results.py configs/mil_models.tsv \
  --run-root "${GATEDSRP_MIL_OUT:-runs/mil_models}" \
  --public-only --strict

python scripts/collect_comparison_results.py configs/coefficient_parameterizations.tsv \
  --run-root "${GATEDSRP_COEFFICIENT_OUT:-runs/coefficient_parameterizations}" \
  --public-only --strict

python scripts/collect_comparison_results.py configs/neighborhood_sizes.tsv \
  --run-root "${GATEDSRP_NEIGHBORHOOD_OUT:-runs/neighborhood_sizes}" \
  --public-only --strict

python scripts/collect_comparison_results.py configs/runtime_efficiency.tsv \
  --run-root "${GATEDSRP_RUNTIME_OUT:-runs/runtime_efficiency}" \
  --public-only --strict
```

Each invocation writes selected-metric `per_seed.tsv` and `summary.tsv`, plus
`all_metrics_per_seed.tsv` and `all_metrics_summary.tsv`, under
`results/rerun/<manifest-name>/`. The all-metric tables reconstruct
multi-metric comparisons such as ADP mAP/F1/accuracy/AUC.
Profiler manifests also write `runtime.tsv`, including the derived
five-cohort TCGA mean. Compare those summaries with the bundled reference
tables listed in [RESULTS.md](RESULTS.md).

## 7. Expected Run Artifacts

Classification run directories contain:

```text
<run_name>/
  best.pt
  test_artifacts.npz
  ...
```

TCGA survival run directories contain:

```text
<run_name>/
  best.pt
  metrics.json
  test_slide_predictions.csv
  test_case_predictions.csv
  ...
```

Rows launched with `--profile_runtime` additionally write
`runtime_profile.json` with synchronized phase throughput and CUDA peak memory.

## Reproduction Rules

- Use the five global-seed holdouts encoded in each manifest.
- Keep paired method arms on the same seed and split.
- Keep the context implementation flags encoded in each command. Nyström
  quality rows pin the reduction order used for their reference checkpoints;
  the efficiency manifest pins the streamed, chunked implementation it
  measures.
- Do not cap native-length WSI task runs. The 1,024-patch cap belongs only to
  the explicit dense-retained-patch comparison.
- Keep `features` and `coords` row-aligned when combining encoder outputs.
- Report case-level C-index for TCGA survival and sample standard deviation
  across the five seeds.
- Treat runtime values as machine-dependent. Compare methods on the same host,
  software environment, data storage, and profile schedule.

No pretrained slide-level model is required. Manifests regenerate checkpoints
locally; see [CHECKPOINTS.md](CHECKPOINTS.md).
