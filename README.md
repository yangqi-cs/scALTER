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

Activate the conda environment that contains scALTER dependencies, then run:

```bash
python /qiyang/GitHub/scALTER/scripts/scALTER.py \
  --bam /path/to/alignments.bam \
  --whitelist /path/to/barcodes.tsv \
  --te-annotation-gtf /path/to/te_annotation.gtf
```

By default, outputs are written under:

```text
/qiyang/GitHub/scALTER/results/
```

## Example With Parameters

```bash
python /qiyang/GitHub/scALTER/scripts/scALTER.py \
  --bam /path/to/alignments.bam \
  --whitelist /path/to/barcodes.tsv \
  --te-annotation-gtf /path/to/te_annotation.gtf \
  --sample-prefix sample1 \
  --threads 48 \
  --reducer-threads 2 \
  --count-likelihood nb \
  --epochs 300 \
  --n-latent 32 \
  --batch-size 128 \
  --learning-rate 1e-3
```

## Required Inputs

- `--bam`: input alignment file in BAM format.
- `--whitelist`: cell barcode whitelist, usually the filtered 10x `barcodes.tsv`.
- `--te-annotation-gtf`: TE annotation file in GTF format.

Before running the count extraction step, `scripts/scALTER.py` opens the BAM and checks that it is readable and that the requested cell and UMI tags are present in mapped reads.

## Pipeline Outputs

Step 1 writes BAM-derived count tables:

- `counts/scalter_unique.tsv`
- `counts/scalter_multi.tsv`

Step 2 writes the three preserved model input views:

- `views/aligned_npz/unique.npz`
- `views/aligned_npz/multi.npz`
- `views/aligned_npz/merge.npz`
- `views/h5ad/scalter_subfamily_u_m_aligned.h5ad`

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
