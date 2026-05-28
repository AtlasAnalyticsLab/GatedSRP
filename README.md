# GatedSRP

Reproducibility code for **Gated Spatial Redundancy Projection for Pathology Transformer Attentions**.

This repository contains only the reproducibility training paths:

- Classification: CAMELYON16, CAMELYON17, KGH, PANDA, and BRACS.
- Survival: TCGA-KIRC, TCGA-KIRP, TCGA-LUAD, TCGA-STAD, and TCGA-UCEC.
- Architecture ablation: ADP raw-RGB patches and PANDA dense ViT.
- Design ablations: fixed projection, gate range, gate gradients, gate
  factorization, gate initialization, and patch encoder.
- Methods: NA, XSA, Diff, and Gated SRP.

The repository is organized around a short quick start, detailed docs in
`docs/`, runnable manifests in `configs/`, and reference tables in
`results/`.

## Quick Start

Choose one environment path.

Conda:

```bash
conda env create -f environment.yml
conda activate gatedsrp
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

venv + pip:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

uv:

```bash
uv sync --python 3.10
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only smoke tests, install the CPU PyTorch wheel instead.

Prepare datasets and frozen H5 embeddings as described in:

- [docs/DATASETS.md](docs/DATASETS.md)
- [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md)

Then set paths:

```bash
source configs/paths.example.env
```

Run a smoke command:

```bash
python scripts/run_manifest.py configs/paper_classification.tsv \
  --where dataset=cam16 --where method=baseline --where seed=42 --dry-run
```

Run the lightweight unit tests:

```bash
python -m pytest tests -q
```

Run the full manifests:

```bash
python scripts/run_manifest.py configs/paper_classification.tsv
python scripts/run_manifest.py configs/paper_tcga_survival.tsv
python scripts/run_manifest.py configs/paper_architecture_ablation.tsv
python scripts/run_manifest.py configs/paper_design_ablation.tsv
python scripts/run_manifest.py configs/paper_patch_encoder_ablation.tsv
```

Collect main-table rerun metrics:

```bash
python scripts/collect_paper_results.py --strict
```

## Repository Layout

| Path | Purpose |
|---|---|
| `slide_level_srp/` | Gated SRP, Diff Transformer comparator, slide/TCGA trainers, and dataset adapters. |
| `slide_level/` | Baseline TransMIL/XSA components reused by the SRP trainer. |
| `patch_level_adp/` | ADP raw-RGB ViT training entry point for the architecture ablation. |
| `slide_level_srp/train_panda.py` and `src/` | PANDA training entry point plus PANDA data/model helpers. |
| `configs/` | Exact reported-run manifests and example path environment variables. |
| `scripts/` | Manifest runner, H5 validator, AtlasPatch extraction template, and result collector. |
| `results/` | Reference result tables for reproducing the manuscript numbers. |
| `docs/` | Dataset, embedding, and reproduction details. |

## Reference Results

Reference tables are bundled as TSV files:

- [classification_main_table.tsv](results/classification_main_table.tsv)
- [tcga_survival_main_table.tsv](results/tcga_survival_main_table.tsv)
- [ablation_architecture_choice.tsv](results/ablation_architecture_choice.tsv)
- [ablation_fixed_projection.tsv](results/ablation_fixed_projection.tsv)
- [ablation_gate_range.tsv](results/ablation_gate_range.tsv)
- [ablation_gate_gradients.tsv](results/ablation_gate_gradients.tsv)
- [ablation_gate_factorization.tsv](results/ablation_gate_factorization.tsv)
- [ablation_gate_initialization.tsv](results/ablation_gate_initialization.tsv)
- [ablation_patch_encoder.tsv](results/ablation_patch_encoder.tsv)

The manifests use five global seeds (`42` to `46`) for every dataset-method pair.
ADP uses its official Release1 train/validation/test split for every method;
the seed controls model initialization and training randomness.

## Citation

Citation metadata will be added after publication details are available.

## License

This repository is released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. See [LICENSE](LICENSE).
