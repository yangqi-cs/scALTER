#!/usr/bin/env python3
"""Command-line entry point for the scALTER pipeline."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_PYTHON = sys.executable
DEFAULT_RESULT_ROOT = "/qiyang/GitHub/scALTER/results"


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
    sys.stdout.flush()
    subprocess.run(cmd, check=True)


def validate_input_files(args):
    bam = Path(args.bam)
    whitelist = Path(args.whitelist)
    te_gtf = Path(args.te_annotation_gtf)

    if bam.suffix.lower() != ".bam":
        raise ValueError(f"--bam must point to a .bam file: {bam}")
    if not bam.exists():
        raise FileNotFoundError(f"BAM file does not exist: {bam}")
    if not whitelist.exists():
        raise FileNotFoundError(f"Whitelist file does not exist: {whitelist}")
    if not (
        str(te_gtf).endswith(".gtf")
        or str(te_gtf).endswith(".gtf.gz")
    ):
        raise ValueError(f"--te-annotation-gtf must point to a .gtf or .gtf.gz file: {te_gtf}")
    if not te_gtf.exists():
        raise FileNotFoundError(f"TE annotation GTF does not exist: {te_gtf}")

    try:
        import pysam
    except ImportError as exc:
        raise ImportError(
            "pysam is required to validate BAM files before running scALTER."
        ) from exc

    found_cell_tag = False
    found_umi_tag = False
    inspected = 0
    max_reads = 100000
    with pysam.AlignmentFile(str(bam), "rb") as bam_file:
        for read in bam_file.fetch(until_eof=True):
            if read.is_unmapped:
                continue
            inspected += 1
            found_cell_tag = found_cell_tag or read.has_tag(args.cell_tag)
            found_umi_tag = found_umi_tag or read.has_tag(args.umi_tag)
            if found_cell_tag and found_umi_tag:
                break
            if inspected >= max_reads:
                break

    if inspected == 0:
        raise ValueError(f"No mapped reads were found while checking BAM: {bam}")
    missing = []
    if not found_cell_tag:
        missing.append(args.cell_tag)
    if not found_umi_tag:
        missing.append(args.umi_tag)
    if missing:
        raise ValueError(
            "Could not find required BAM tag(s) in the first "
            f"{inspected} mapped reads: {', '.join(missing)}"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the scALTER count extraction, view construction, and model training pipeline."
    )

    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Python executable used for all steps.")
    parser.add_argument(
        "--result-root",
        default=DEFAULT_RESULT_ROOT,
        help="Root output directory. scALTER writes raw_exp/, model/, recon_exp/, and latent/ under this path.",
    )

    parser.add_argument(
        "--bam",
        required=True,
        help="Input alignment file in BAM format.",
    )
    parser.add_argument(
        "--whitelist",
        required=True,
        help="Cell barcode whitelist, usually the filtered 10x barcodes.tsv file.",
    )
    parser.add_argument(
        "--te-annotation-gtf",
        required=True,
        help="TE annotation file in GTF format.",
    )

    parser.add_argument("--tmp-dir", default=None)

    parser.add_argument("--samtools", default=None)
    parser.add_argument("--bedtools", default=None)
    parser.add_argument("--awk", default=None)
    parser.add_argument("--cell-tag", default="CB")
    parser.add_argument("--umi-tag", default="UB")
    parser.add_argument("--min-mapq", type=int, default=0)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--reducer-threads", type=int, default=1)
    parser.add_argument("--overwrite-counts", action="store_true")
    parser.add_argument("--reuse-hits", action="store_true")
    parser.add_argument("--keep-count-tmp", action="store_true")
    parser.add_argument("--canonical-gtf-to-bed", action="store_true")

    parser.add_argument("--align-mode", choices=["union", "intersection"], default="union")

    parser.add_argument("--count-likelihood", choices=["nb", "zinb"], default="nb")
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-latent", type=int, default=32)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
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
    result_root = Path(args.result_root)
    raw_exp_dir = result_root / "raw_exp"
    model_dir = result_root / "model"
    recon_dir = result_root / "recon_exp"
    latent_dir = result_root / "latent"
    tmp_root = result_root / "tmp"
    count_tmp_dir = tmp_root / "counts"
    bedtools_tmp_dir = Path(args.tmp_dir) if args.tmp_dir else tmp_root / "bedtools"

    if args.skip_counts and not args.skip_views:
        raise ValueError(
            "--skip-counts cannot be used when views need to be built because "
            "count TSV files are treated as temporary intermediates."
        )

    if not args.skip_counts:
        validate_input_files(args)

    if not args.skip_counts:
        tmp_root.mkdir(parents=True, exist_ok=True)
        if count_tmp_dir.exists():
            shutil.rmtree(count_tmp_dir)
        count_tmp_dir.mkdir(parents=True, exist_ok=True)
        if args.tmp_dir is None and bedtools_tmp_dir.exists() and not args.reuse_hits:
            shutil.rmtree(bedtools_tmp_dir)

        cmd = [
            args.python,
            str_path(script_dir / "extract_counts.py"),
            "--bam", args.bam,
            "--whitelist", args.whitelist,
            "--te-annotation-gtf", args.te_annotation_gtf,
            "--output-dir", str_path(count_tmp_dir),
            "--tmp-dir", str_path(bedtools_tmp_dir),
            "--cell-tag", args.cell_tag,
            "--umi-tag", args.umi_tag,
            "--min-mapq", str(args.min_mapq),
            "--threads", str(args.threads),
            "--reducer-threads", str(args.reducer_threads),
        ]
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
            "--input-dir", str_path(count_tmp_dir),
            "--output-dir", str_path(raw_exp_dir),
            "--align-mode", args.align_mode,
        ]
        run_step("Step 2/3: building aligned input views", cmd)

    if not args.keep_count_tmp and count_tmp_dir.exists():
        shutil.rmtree(count_tmp_dir)

    if not args.skip_model:
        cmd = [
            args.python,
            str_path(script_dir / "train_model.py"),
            "--data-dir", str_path(raw_exp_dir),
            "--output-dir", str_path(model_dir),
            "--recon-dir", str_path(recon_dir),
            "--latent-dir", str_path(latent_dir),
            "--count-likelihood", args.count_likelihood,
            "--n-hidden", str(args.n_hidden),
            "--n-layers", str(args.n_layers),
            "--n-latent", str(args.n_latent),
            "--dropout-rate", str(args.dropout_rate),
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
