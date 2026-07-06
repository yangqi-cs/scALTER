# scALTER

This repository contains a runnable three-step scALTER pipeline organized from the PBMC8k reference scripts.

## Layout

- `scripts/1_extract_u_m_counts_by_cb_bedtools_parallel.py`
  Extracts TE unique and multi counts from a 10x BAM using `samtools` and `bedtools`.
- `scripts/2_build_u_m_views.py`
  Converts long `*_unique.tsv` and `*_multi.tsv` files into aligned sparse matrices.
- `scripts/3_train_scvi_poe_unique_multi_merge_best_model.py`
  Trains the three-view scVI-PoE model using unique, multi, and merge views.
- `scripts/run_scalter_pbmc8k.sh`
  Runs the three steps sequentially with PBMC8k defaults.

## Default Run

```bash
bash /qiyang/GitHub/scALTER/scripts/run_scalter_pbmc8k.sh
```

The default input BAM and whitelist are inherited from the PBMC8k reference workflow:

- BAM: `/qiyang/TEexp/Data/IRescue/human_pbmc8k/star_output/pbmc8k/pbmc8k_Aligned.sortedByCoord.out.bam`
- whitelist: `/qiyang/TEexp/Data/IRescue/human_pbmc8k/star_output/pbmc8k/pbmc8k_Solo.out/Gene/filtered/barcodes.tsv`
- TE GTF: `/qiyang/TEexp/Data/dataset_human/hg38/hg38_TE_subfamily.exclusive.gtf`

## Preserved Outputs

Step 1 keeps BAM-derived long-format count inputs:

- `results/pbmc8k/my_subfamily/1_subfamily_u_m/pbmc8k_unique.tsv`
- `results/pbmc8k/my_subfamily/1_subfamily_u_m/pbmc8k_multi.tsv`

Step 2 keeps the three model input views:

- `results/pbmc8k/my_subfamily/2_subfamily_u_m_aligned/aligned_npz/unique.npz`
- `results/pbmc8k/my_subfamily/2_subfamily_u_m_aligned/aligned_npz/multi.npz`
- `results/pbmc8k/my_subfamily/2_subfamily_u_m_aligned/aligned_npz/merge.npz`
- `results/pbmc8k/my_subfamily/2_subfamily_u_m_aligned/h5ad/pbmc8k_subfamily_u_m_aligned.h5ad`

Step 3 keeps model outputs:

- `results/pbmc8k/my_subfamily/3_cross_view/1_7_scvi_poe_unique_multi_merge_best_model/scvi_poe_weights.pt`
- `results/pbmc8k/my_subfamily/3_cross_view/1_7_scvi_poe_unique_multi_merge_best_model/scvi_poe_model_checkpoint.pt`
- `results/pbmc8k/my_subfamily/3_cross_view/1_7_scvi_poe_unique_multi_merge_best_model/training_history.json`
- `results/pbmc8k/my_subfamily/3_cross_view/1_7_scvi_poe_unique_multi_merge_best_model/run_config.json`
- `mean_u.tsv`, `mean_m.tsv`, `mean_merge.tsv`, `latent_mu.tsv`, and `latent_std.tsv`

## Common Overrides

```bash
WORKERS=48 REDUCER_WORKERS=2 bash /qiyang/GitHub/scALTER/scripts/run_scalter_pbmc8k.sh
```

Each Python script also exposes `--help` for dataset-specific paths and training parameters.
