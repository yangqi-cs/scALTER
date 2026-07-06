# scALTER

## Introduction
scALTER is a framework for single-cell TE expression reconstruction and latent representation learning.

<div align="center">
<img src="figures/workflow.png" width="90%">
</div>

## Installation
Follow the steps below to set up scALTER:

#### 1. Clone the Repository

Retrieve the latest version of scALTER from the GitHub repository:

```bash
git clone https://github.com/yangqi-cs/scALTER.git
cd scALTER
```

#### 2. Set Up the Conda Environment

Create and activate the conda environment:

```bash
conda env create -f env.yml
conda activate scALTER
```

## Usage
- #### scripts/scALTER.py

Run the full scALTER pipeline, including count extraction, view construction, and model training. Required inputs:

```text
--bam                  Input alignment file in BAM format
--whitelist            Cell barcode whitelist, usually filtered barcodes.tsv
--te-annotation-gtf    TE annotation file in GTF format
```

Common options:

```text
--result-root          Root output directory for counts/, views/, and model/
--cell-tag             BAM tag for cell barcodes
--umi-tag              BAM tag for UMIs
--sample-prefix        Prefix used for count table names
--threads              Number of threads for count extraction
--reducer-threads      Number of threads for count table reduction
--align-mode           How to align unique and multi matrices: union or intersection
--count-likelihood     Count likelihood used by the model: nb or zinb
--n-hidden             Hidden dimension
--n-latent             Latent representation dimension
--n-layers             Number of neural network layers
--dropout-rate         Dropout rate
--kl-weight            KL loss weight
--learning-rate        Learning rate
--batch-size           Training batch size
--epochs               Maximum number of training epochs
```

- #### scripts/extract_counts.py

Extract unique and multi TE count tables from a BAM file.

- #### scripts/build_views.py

Build aligned unique, multi, and merge sparse matrix views.

- #### scripts/train_model.py

Train the scALTER model and export reconstructed means, latent representations, and checkpoints.

## Examples
Run the full pipeline:

```bash
python scripts/scALTER.py \
  --bam /path/to/alignments.bam \
  --whitelist /path/to/barcodes.tsv \
  --te-annotation-gtf /path/to/te_annotation.gtf
```

Run with commonly tuned parameters:

```bash
python scripts/scALTER.py \
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

By default, scALTER writes results under:

```text
/qiyang/GitHub/scALTER/results/
```

The main output files include:

```text
counts/scalter_unique.tsv
counts/scalter_multi.tsv
views/aligned_npz/unique.npz
views/aligned_npz/multi.npz
views/aligned_npz/merge.npz
model/scalter_weights.pt
model/scalter_checkpoint.pt
model/latent_mu.tsv
model/latent_std.tsv
```

## Contact
:e-mail: **Yang Qi** (yang.qi@mail.nwpu.edu.cn)

School of Computer Science, Northwestern Polytechnical University, Xi’an, Shaanxi 710072, China
