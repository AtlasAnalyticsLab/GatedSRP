# GatedSRP

Reproducibility code for **Gated Spatial Redundancy Projection for Pathology Transformer Attentions**.

This repository contains only the paper-facing training paths:

- Classification: CAMELYON16, CAMELYON17, KGH, PANDA, and BRACS.
- Survival: TCGA-KIRC, TCGA-KIRP, TCGA-LUAD, TCGA-STAD, and TCGA-UCEC.
- Methods: Standard self-attention, XSA, Differential Transformer, and Gated SRP.

The organization follows the public-release style used by
[AtlasAnalyticsLab/MOOZY](https://github.com/AtlasAnalyticsLab/MOOZY): a short quick start here, detailed docs in `docs/`, runnable manifests in `configs/`, and paper tables in `results/paper/`.

## Quick Start

```bash
conda env create -f environment.yml
conda activate gatedsrp
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Prepare datasets and UNI-v2 H5 embeddings as described in:

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

Run the paper manifests:

```bash
python scripts/run_manifest.py configs/paper_classification.tsv
python scripts/run_manifest.py configs/paper_tcga_survival.tsv
```

Collect rerun metrics:

```bash
python scripts/collect_paper_results.py --strict
```

## Repository Layout

| Path | Purpose |
|---|---|
| `slide_level_srp/` | Gated SRP, Diff Transformer comparator, slide/TCGA trainers, and dataset adapters. |
| `slide_level/` | Baseline TransMIL/XSA components reused by the SRP trainer. |
| `panda/` and `src/` | PANDA training entry point plus PANDA data/model helpers used for the paper PANDA row. |
| `configs/` | Exact paper run manifests and example path environment variables. |
| `scripts/` | Manifest runner, H5 validator, AtlasPatch extraction template, and result collector. |
| `results/paper/` | Paper-facing reference tables copied from the audited internal run package. |
| `docs/` | Dataset, embedding, and reproduction details. |

## Reference Results

Paper tables are bundled as TSV files:

- [classification_main_table.tsv](results/paper/classification_main_table.tsv)
- [tcga_survival_main_table.tsv](results/paper/tcga_survival_main_table.tsv)

The manifests use five global seeds (`42` to `46`) for every dataset-method pair.

## Citation

Citation metadata will be added after the paper review process is complete.
