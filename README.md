# scALTER

This repository provides a single-command scALTER pipeline for transposable element count extraction, view construction, and model training.

## Scripts

- `scripts/scALTER.py`
  Main pipeline entry point. Runs the three steps in order and exposes the commonly tuned parameters.
- `scripts/extract_counts.py`
  Extracts unique and multi TE count tables from a 10x BAM.
- `scripts/build_views.py`
  Builds aligned unique, multi, and merge sparse matrix views.
- `scripts/train_model.py`
  Trains the scALTER model and writes latent representations, reconstructed means, and checkpoints.

## Quick Start

```bash
/qiyang/Anaconda/conda/envs/teexp/bin/python /qiyang/GitHub/scALTER/scripts/scALTER.py
```

The PBMC8k defaults are:

- BAM: `/qiyang/TEexp/Data/IRescue/human_pbmc8k/star_output/pbmc8k/pbmc8k_Aligned.sortedByCoord.out.bam`
- whitelist: `/qiyang/TEexp/Data/IRescue/human_pbmc8k/star_output/pbmc8k/pbmc8k_Solo.out/Gene/filtered/barcodes.tsv`
- TE GTF: `/qiyang/TEexp/Data/dataset_human/hg38/hg38_TE_subfamily.exclusive.gtf`
- results: `/qiyang/GitHub/scALTER/results/pbmc8k/my_subfamily`

## Example With Parameters

```bash
/qiyang/Anaconda/conda/envs/teexp/bin/python /qiyang/GitHub/scALTER/scripts/scALTER.py \
  --workers 48 \
  --reducer-workers 2 \
  --epochs 300 \
  --n-latent 32 \
  --batch-size 128 \
  --learning-rate 1e-3
```

## Pipeline Outputs

Step 1 writes BAM-derived count tables:

- `counts/pbmc8k_unique.tsv`
- `counts/pbmc8k_multi.tsv`

Step 2 writes the three preserved model input views:

- `views/aligned_npz/unique.npz`
- `views/aligned_npz/multi.npz`
- `views/aligned_npz/merge.npz`
- `views/h5ad/pbmc8k_subfamily_u_m_aligned.h5ad`

Step 3 writes model outputs:

- `model/scalter_weights.pt`
- `model/scalter_checkpoint.pt`
- `model/training_history.json`
- `model/run_config.json`
- `model/mean_u.tsv`
- `model/mean_m.tsv`
- `model/mean_merge.tsv`
- `model/latent_mu.tsv`
- `model/latent_std.tsv`

Each step script can still be run independently, but `scripts/scALTER.py` is the recommended entry point.
