# Reproducing the Reported Runs

## 1. Environment

Choose one setup path. PyTorch is installed explicitly so you can select the
CUDA or CPU wheel that matches your machine.

### Conda

```bash
conda env create -f environment.yml
conda activate gatedsrp
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

### venv + pip

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

### uv

```bash
uv sync --python 3.10
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only smoke tests, install the CPU PyTorch wheel instead.

## 2. Data and Embeddings

1. Download datasets as described in [DATASETS.md](DATASETS.md).
2. Extract UNI-v2 H5 embeddings as described in [EMBEDDINGS.md](EMBEDDINGS.md).
3. Source path variables:

```bash
source configs/paths.example.env
```

4. Validate H5 inventories:

```bash
python scripts/validate_h5_embeddings.py --root "$CAM17_UNIV2_ROOT" --feature-key "$CAM17_UNIV2_FEATURE_KEY" --expected-dim 1536
python scripts/validate_h5_embeddings.py --root "$PANDA_FEATURE_ROOT" --feature-key features/uni_v2 --expected-dim 1536
```

Repeat for the other feature roots.

## 3. Run Classification

Run one row first:

```bash
python scripts/run_manifest.py configs/paper_classification.tsv \
  --where dataset=cam16 --where method=baseline --where seed=42
```

Run all classification rows:

```bash
python scripts/run_manifest.py configs/paper_classification.tsv
```

This writes runs under `GATEDSRP_CLASSIFICATION_OUT`, defaulting to `runs/classification`.

## 4. Run TCGA Survival

Run one row first:

```bash
python scripts/run_manifest.py configs/paper_tcga_survival.tsv \
  --where cohort=KIRC --where method=baseline --where seed=42
```

Run all survival rows:

```bash
python scripts/run_manifest.py configs/paper_tcga_survival.tsv
```

This writes runs under `GATEDSRP_TCGA_SURVIVAL_OUT`, defaulting to `runs/tcga_survival_global_seed_main`.

## 5. Collect Tables

```bash
python scripts/collect_paper_results.py --strict
```

Collected rerun summaries are written to `results/rerun/`. Reference tables are in `results/`.

## Notes on Compute

The reported runs use batch size 1 with gradient accumulation. Large WSI bags can require a GPU with substantial memory; reduce pressure by lowering workers, using smaller shard batches during embedding extraction, or setting split caps only for debugging. Do not use caps for final reproduction unless you report that change.
