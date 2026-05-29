# Checkpoints

This release does not bundle trained model checkpoints.

The reported tables are intended to be reproduced by rerunning the checked-in
manifests in `configs/` after preparing the datasets and H5 embeddings. Each
manifest row fixes the run name, method, dataset, seed, split mode, and trainer
arguments. The resulting local training outputs include `best.pt` files under
the configured `runs/` directory.

## Why Checkpoints Are Not Included

- Checkpoints are generated outputs, not required inputs for reproduction.
- The full set of trained weights is large and redundant with the runnable
  manifests.
- Git source clones should stay lightweight; checkpoints, H5 embeddings, raw
  WSIs, logs, and run folders are intentionally ignored by `.gitignore`.

## Regenerate Checkpoints

For a single classification run:

```bash
python scripts/run_manifest.py configs/paper_classification.tsv \
  --where dataset=cam16 --where method=baseline --where seed=42
```

For a single TCGA survival run:

```bash
python scripts/run_manifest.py configs/paper_tcga_survival.tsv \
  --where cohort=KIRC --where method=baseline --where seed=42
```

The trainer writes the best checkpoint inside the selected output root, for
example `runs/classification/<run_name>/best.pt` or
`runs/tcga_survival_global_seed_main/<run_name>/best.pt`.

## Optional External Weights

If fixed pretrained weights are released later, they should be hosted outside
the source repository and accompanied by a manifest containing:

- manifest row or `run_name`
- dataset/cohort and seed
- model family and feature key
- checkpoint filename
- SHA256 checksum

The source repository should then link to that external model repository rather
than storing binary weights directly in Git.
