# Checkpoints

Trained checkpoints are not stored in this Git repository. They are generated
outputs of the released five-seed manifests, not prerequisites for training or
evaluation.

## What Each Run Produces

Every trainer writes its best task checkpoint under the configured output root:

```text
runs/<study>/<run_name>/best.pt
```

Classification runs also save `test_artifacts.npz`; TCGA survival runs save
`metrics.json`, slide predictions, and case-level predictions. These metric
artifacts are sufficient for the result collectors.

## Regenerate a Checkpoint

```bash
python scripts/run_manifest.py configs/classification_tasks.tsv \
  --where dataset=cam16 --where method=baseline --where seed=42

python scripts/run_manifest.py configs/survival_tasks.tsv \
  --where cohort=KIRP --where method=gated_post_attention --where seed=42
```

The exact model, split, optimizer, feature key, and seed are visible in the
selected TSV row.

## Why Weights Are Separate

- The full matrix contains hundreds of seed-specific models.
- Raw checkpoints are much larger than the source and result tables.
- No pretrained slide encoder or GatedSRP-specific pretraining is needed to run
  the released protocols.
- Reproduction from the manifests verifies the complete training path rather
  than only checkpoint inference.

For convenience inference, a future model hosting release can provide selected
task checkpoints with their manifest row, dataset, feature key, seed, and model
family. Until such a release is linked here, regenerate the desired `best.pt`
from its manifest row.

H5 embeddings are also too large for the source repository. They can be
generated with AtlasPatch using [EMBEDDINGS.md](EMBEDDINGS.md) or distributed as
a separate data artifact.
