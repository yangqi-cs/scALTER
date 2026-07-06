#!/usr/bin/env python3
"""Command-line entry point for the scALTER pipeline."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_PYTHON = "/qiyang/Anaconda/conda/envs/teexp/bin/python"
DEFAULT_RESULT_ROOT = "/qiyang/GitHub/scALTER/results/pbmc8k"


def str_path(path):
    return str(path)


def add_optional(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def add_bool(cmd, flag, enabled):
    if enabled:
        cmd.append(flag)


def run_step(step_name, cmd):
    print("\n" + "=" * 80)
    print(step_name)
    print("=" * 80)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the scALTER count extraction, view construction, and model training pipeline."
    )

    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Python executable used for all steps.")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--base-dir", default=None, help="Default: <result-root>/my_<te-level>.")
    parser.add_argument("--te-level", choices=["subfamily", "locus"], default="subfamily")
    parser.add_argument("--sample-prefix", default="pbmc8k")

    parser.add_argument(
        "--bam",
        default="/qiyang/TEexp/Data/IRescue/human_pbmc8k/star_output/pbmc8k/pbmc8k_Aligned.sortedByCoord.out.bam",
    )
    parser.add_argument(
        "--whitelist",
        default="/qiyang/TEexp/Data/IRescue/human_pbmc8k/star_output/pbmc8k/pbmc8k_Solo.out/Gene/filtered/barcodes.tsv",
    )
    parser.add_argument(
        "--gtf-file",
        default="/qiyang/TEexp/Data/dataset_human/hg38/hg38_TE_subfamily.exclusive.gtf",
    )

    parser.add_argument("--counts-dir", default=None, help="Default: <base-dir>/counts.")
    parser.add_argument("--views-dir", default=None, help="Default: <base-dir>/views.")
    parser.add_argument("--model-dir", default=None, help="Default: <base-dir>/model.")
    parser.add_argument("--tmp-dir", default=None)

    parser.add_argument("--samtools", default=None)
    parser.add_argument("--bedtools", default=None)
    parser.add_argument("--awk", default=None)
    parser.add_argument("--cell-tag", default="CB")
    parser.add_argument("--umi-tag", default="UB")
    parser.add_argument("--min-mapq", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--reducer-workers", type=int, default=1)
    parser.add_argument("--overwrite-counts", action="store_true")
    parser.add_argument("--reuse-hits", action="store_true")
    parser.add_argument("--keep-count-tmp", action="store_true")
    parser.add_argument("--canonical-gtf-to-bed", action="store_true")

    parser.add_argument("--align-mode", choices=["union", "intersection"], default="union")

    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-latent", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--gene-likelihood", choices=["nb", "zinb"], default="nb")
    parser.add_argument("--kl-weight", type=float, default=0.00001)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--predict-batch-size", type=int, default=128)
    parser.add_argument("--output-chunk-features", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early-stop", type=int, default=15)
    parser.add_argument("--reduce-lr", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-batch-norm", action="store_true")
    parser.add_argument("--use-layer-norm", action="store_true")

    parser.add_argument("--skip-counts", action="store_true")
    parser.add_argument("--skip-views", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = Path(args.base_dir or os.path.join(args.result_root, f"my_{args.te_level}"))
    counts_dir = Path(args.counts_dir or base_dir / "counts")
    views_dir = Path(args.views_dir or base_dir / "views")
    model_dir = Path(args.model_dir or base_dir / "model")

    if not args.skip_counts:
        cmd = [
            args.python,
            str_path(script_dir / "extract_counts.py"),
            "--te-level", args.te_level,
            "--sample-prefix", args.sample_prefix,
            "--bam", args.bam,
            "--whitelist", args.whitelist,
            "--gtf-file", args.gtf_file,
            "--output-dir", str_path(counts_dir),
            "--cell-tag", args.cell_tag,
            "--umi-tag", args.umi_tag,
            "--min-mapq", str(args.min_mapq),
            "--workers", str(args.workers),
            "--reducer-workers", str(args.reducer_workers),
        ]
        add_optional(cmd, "--tmp-dir", args.tmp_dir)
        add_optional(cmd, "--samtools", args.samtools)
        add_optional(cmd, "--bedtools", args.bedtools)
        add_optional(cmd, "--awk", args.awk)
        add_bool(cmd, "--no-skip-existing", args.overwrite_counts)
        add_bool(cmd, "--reuse-hits", args.reuse_hits)
        add_bool(cmd, "--keep-tmp", args.keep_count_tmp)
        add_bool(cmd, "--canonical-gtf-to-bed", args.canonical_gtf_to_bed)
        run_step("Step 1/3: extracting unique and multi counts", cmd)

    if not args.skip_views:
        cmd = [
            args.python,
            str_path(script_dir / "build_views.py"),
            "--te-level", args.te_level,
            "--sample-prefix", args.sample_prefix,
            "--base-dir", str_path(base_dir),
            "--input-dir", str_path(counts_dir),
            "--output-dir", str_path(views_dir),
            "--align-mode", args.align_mode,
        ]
        run_step("Step 2/3: building aligned input views", cmd)

    if not args.skip_model:
        cmd = [
            args.python,
            str_path(script_dir / "train_model.py"),
            "--te-level", args.te_level,
            "--base-dir", str_path(base_dir),
            "--data-dir", str_path(views_dir / "aligned_npz"),
            "--output-dir", str_path(model_dir),
            "--n-hidden", str(args.n_hidden),
            "--n-latent", str(args.n_latent),
            "--n-layers", str(args.n_layers),
            "--dropout-rate", str(args.dropout_rate),
            "--gene-likelihood", args.gene_likelihood,
            "--kl-weight", str(args.kl_weight),
            "--learning-rate", str(args.learning_rate),
            "--batch-size", str(args.batch_size),
            "--predict-batch-size", str(args.predict_batch_size),
            "--output-chunk-features", str(args.output_chunk_features),
            "--epochs", str(args.epochs),
            "--early-stop", str(args.early_stop),
            "--reduce-lr", str(args.reduce_lr),
            "--random-seed", str(args.random_seed),
        ]
        add_bool(cmd, "--no-batch-norm", args.no_batch_norm)
        add_bool(cmd, "--use-layer-norm", args.use_layer_norm)
        run_step("Step 3/3: training the scALTER model", cmd)


if __name__ == "__main__":
    main()
