#!/usr/bin/env python3
"""
Accelerated 10x U/M TE counting for scALTER.

This keeps the counting policy used by the original scALTER count extractor:
  - feature = gene_id from repeat_region rows in the TE GTF
  - valid cell barcode filtering by per-sample *_barcodes.txt
  - UMI deduplication by (cell, TE, UMI)
  - U matrix: NH == 1 and the alignment overlaps exactly one TE feature
  - M matrix: all other TE-overlapping alignments, weighted by 1 / n_overlapped_TE

The expensive read/TE overlap step is delegated to standard command-line genomics tools and scheduled as sample-by-chromosome tasks. Per-sample reducers then apply the original U/M and UMI logic to the hit streams.
"""

# Example production run:
# python -u /qiyang/GitHub/scALTER/scripts/extract_counts.py --threads 32 --reducer-threads 1

import argparse
import gzip
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import timedelta
from multiprocessing import Pool, cpu_count

import pysam


# ============================================================
# ==================  USER CONFIGURATION  ===================
# ============================================================

# Analysis mode
TE_LEVEL = "subfamily"  # "subfamily" or "locus"

# Input files
SAMPLE_NAME = "scalter"
BAM = None
WHITELIST = None
TE_ANNOTATION_GTF = None

# Output directories
OUTPUT_DIR = None
# None = /qiyang/GitHub/scALTER/results/_tmp_counts
TMP_DIR = None
# None = <OUTPUT_DIR>/_tmp_bedtools

# Tool paths
DEFAULT_ENV_BIN = None
SAMTOOLS = None  # None = auto-detect from the active environment/PATH
BEDTOOLS = None  # None = auto-detect from the active environment/PATH
AWK = None       # None = auto-detect gawk/awk

# BAM tags and filters
CELL_TAG = "CB"
UMI_TAG = "UB"
MIN_MAPQ = 0

# Concurrency
THREADS = None
# None = max(1, min(cpu_count() - 4, 32))
REDUCER_THREADS = None
# None = max(1, min(8, THREADS))

# Run behavior
SKIP_EXISTING = True
REUSE_HITS = False
KEEP_TMP = False

# Keep False to mirror the original Python interval behavior as closely as
# possible. Set True only if you intentionally want canonical GTF->BED coords.
CANONICAL_GTF_TO_BED = False


def default_output_dir():
    return "/qiyang/GitHub/scALTER/results/_tmp_counts"


def resolve_tool(name, explicit=None):
    if explicit:
        return explicit
    if DEFAULT_ENV_BIN:
        candidate = os.path.join(DEFAULT_ENV_BIN, name)
        if os.path.exists(candidate):
            return candidate
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"Cannot find {name}. Pass --{name} or add it to PATH."
    )


def run_cmd(cmd):
    strict_cmd = f"set -euo pipefail; {cmd}"
    subprocess.run(strict_cmd, shell=True, executable="/bin/bash", check=True)


def get_bam_chromosomes(bam_file):
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        return set(bam.references)


def get_bam_chrom_order(bam_file):
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        return list(bam.references)


def parse_gene_id(attrs):
    for attr in attrs.split(";"):
        attr = attr.strip()
        if attr.startswith("gene_id "):
            parts = attr.split(" ", 1)
            if len(parts) == 2:
                return parts[1].replace('"', "").strip()
    return None


def write_te_beds(gtf_file, valid_chroms, bed_dir, preserve_original_coords=True):
    """
    Write one sorted BED4 file per chromosome.

    preserve_original_coords=True intentionally mirrors the older Python
    interval logic, which used Interval(GTF_start, GTF_end + 1) against pysam's
    0-based read coordinates. That is not the canonical GTF->BED conversion,
    but it preserves existing U/M outputs as closely as possible.
    """
    os.makedirs(bed_dir, exist_ok=True)
    per_chrom = defaultdict(list)

    with open(gtf_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, feature_type = fields[0], fields[2]
            if chrom not in valid_chroms or feature_type != "repeat_region":
                continue
            gene_id = parse_gene_id(fields[8])
            if not gene_id:
                continue

            gtf_start = int(fields[3])
            gtf_end = int(fields[4])
            if preserve_original_coords:
                bed_start = gtf_start
                bed_end = gtf_end + 1
            else:
                bed_start = gtf_start - 1
                bed_end = gtf_end

            if bed_start < bed_end:
                per_chrom[chrom].append((bed_start, bed_end, gene_id))

    bed_files = {}
    total_intervals = 0
    for chrom, rows in per_chrom.items():
        rows.sort(key=lambda x: (x[0], x[1], x[2]))
        path = os.path.join(bed_dir, f"{chrom}.te.bed")
        with open(path, "w", buffering=1024 * 1024) as out:
            for start, end, gene_id in rows:
                out.write(f"{chrom}\t{start}\t{end}\t{gene_id}\n")
        bed_files[chrom] = path
        total_intervals += len(rows)

    return bed_files, total_intervals


def load_valid_barcodes(barcode_file):
    with open(barcode_file) as f:
        return {line.strip() for line in f if line.strip()}


def discover_samples(input_dir, barcode_dir):
    barcode_files = sorted(
        f for f in os.listdir(barcode_dir) if f.endswith("_barcodes.txt")
    )
    samples = []
    skipped = []

    for barcode_name in barcode_files:
        prefix = barcode_name.replace("_barcodes.txt", "")
        bam_matches = sorted(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if prefix in f and (f.endswith(".bam") or ".bam." in f)
        )
        if not bam_matches:
            skipped.append(prefix)
            continue
        samples.append(
            {
                "prefix": prefix,
                "bam_file": bam_matches[0],
                "barcode_file": os.path.join(barcode_dir, barcode_name),
            }
        )

    return samples, skipped


def shell_quote(path):
    return "'" + path.replace("'", "'\"'\"'") + "'"


def resolve_awk(explicit=None):
    if explicit:
        return explicit
    for name in ("gawk", "awk"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError("Cannot find awk/gawk. Pass --awk.")


def intersect_one_chrom(args):
    (
        sample,
        chrom,
        ref_bed,
        hit_file,
        samtools,
        bedtools,
        awk,
        cell_tag,
        umi_tag,
        min_mapq,
        reuse_hits,
    ) = args

    if reuse_hits and os.path.exists(hit_file) and os.path.getsize(hit_file) > 0:
        return {
            "prefix": sample["prefix"],
            "chrom": chrom,
            "hit_file": hit_file,
            "skipped": True,
        }

    os.makedirs(os.path.dirname(hit_file), exist_ok=True)
    tmp_file = hit_file + ".tmp"

    bam_q = shell_quote(sample["bam_file"])
    barcode_q = shell_quote(sample["barcode_file"])
    ref_q = shell_quote(ref_bed)
    out_q = shell_quote(tmp_file)
    chrom_q = shell_quote(chrom)
    samtools_q = shell_quote(samtools)
    bedtools_q = shell_quote(bedtools)
    awk_q = shell_quote(awk)

    # The awk stage keeps SAM headers, filters barcodes/MAPQ, extracts tags,
    # and appends CB/UMI/NH/MAPQ to QNAME before bamtobed. The last awk stage
    # writes one row per alignment-TE hit:
    # aln_id, chrom, start, end, CB, UMI, NH, TE
    cmd = f"""
{samtools_q} view -h {bam_q} {chrom_q} |
{awk_q} -v OFS='\\t' \\
     -v cb_tag='{cell_tag}' \\
     -v umi_tag='{umi_tag}' \\
     -v bc_file={barcode_q} \\
     -v min_mapq='{min_mapq}' '
BEGIN {{
    while ((getline line < bc_file) > 0) {{
        sub(/\\r$/, "", line);
        if (line != "") valid[line] = 1;
    }}
    close(bc_file);
}}
$1 ~ /^@/ {{ print; next }}
{{
    mapq = $5 + 0;
    if (mapq < min_mapq) next;

    cb = ""; umi = ""; nh = "";
    for (i = 12; i <= NF; i++) {{
        split($i, tag, ":");
        if (tag[1] == cb_tag) cb = tag[3];
        else if (tag[1] == umi_tag) umi = tag[3];
        else if (tag[1] == "NH") nh = tag[3];
    }}
    if (cb == "" || !(cb in valid)) next;
    if (umi == "") umi = ".";
    if (nh == "") {{
        if (mapq == 0) nh = 2;
        else if (mapq == 255 || mapq >= 20) nh = 1;
        else nh = 2;
    }}
    $1 = cb "|" umi "|" nh "|" mapq "|" NR;
    print;
}}' |
{samtools_q} view -u - |
{bedtools_q} bamtobed -i stdin |
{bedtools_q} intersect -a stdin -b {ref_q} -wa -wb -sorted |
{awk_q} -v OFS='\\t' '
{{
    split($4, q, /\\|/);
    cb = q[1];
    umi = q[2];
    nh = q[3];
    print $4, $1, $2, $3, cb, umi, nh, $10;
}}' |
gzip -c > {out_q}
mv {out_q} {shell_quote(hit_file)}
"""
    try:
        run_cmd(cmd)
        return {
            "prefix": sample["prefix"],
            "chrom": chrom,
            "hit_file": hit_file,
            "skipped": False,
        }
    except Exception:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise


def apply_alignment_counts(
    cell,
    umi,
    nh,
    te_types,
    unique_counts,
    multi_counts,
    seen_umis,
    stats,
    enable_umi_dedup=True,
):
    if not te_types:
        return

    try:
        nh_int = int(nh)
    except ValueError:
        nh_int = 2

    if nh_int == 1:
        stats["unique_mapped"] += 1
    else:
        stats["multi_mapped"] += 1

    te_count = len(te_types)
    has_umi = umi != "."
    if not has_umi:
        stats["no_umi_hit_alignments"] += 1

    if te_count == 1 and nh_int == 1:
        te = next(iter(te_types))
        if enable_umi_dedup and has_umi:
            if umi in seen_umis[cell][te]:
                stats["umi_duplicates"] += 1
                return
            seen_umis[cell][te].add(umi)
        unique_counts[cell][te] += 1
        stats["unique_te_reads"] += 1
        return

    weight = 1.0 / te_count
    for te in te_types:
        if enable_umi_dedup and has_umi:
            if umi in seen_umis[cell][te]:
                stats["umi_duplicates"] += 1
                continue
            seen_umis[cell][te].add(umi)
        multi_counts[cell][te] += weight
    stats["multi_te_reads"] += 1


def reduce_hit_file(
    hit_file,
    unique_counts,
    multi_counts,
    seen_umis,
    stats,
    enable_umi_dedup=True,
):
    current_key = None
    current_cell = None
    current_umi = None
    current_nh = None
    current_tes = set()

    def flush_current():
        if current_key is not None:
            apply_alignment_counts(
                current_cell,
                current_umi,
                current_nh,
                current_tes,
                unique_counts,
                multi_counts,
                seen_umis,
                stats,
                enable_umi_dedup=enable_umi_dedup,
            )

    with gzip.open(hit_file, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            aln_id, chrom, start, end, cell, umi, nh, te = parts[:8]
            key = (aln_id, chrom, start, end)
            if current_key is None:
                current_key = key
                current_cell = cell
                current_umi = umi
                current_nh = nh
                current_tes = {te}
            elif key == current_key:
                current_tes.add(te)
            else:
                flush_current()
                stats["hit_alignments"] += 1
                current_key = key
                current_cell = cell
                current_umi = umi
                current_nh = nh
                current_tes = {te}

    if current_key is not None:
        flush_current()
        stats["hit_alignments"] += 1


def write_long_tsv(output_file, counts, float_values=False):
    with open(output_file, "w", buffering=1024 * 1024) as out:
        lines = []
        for cell in sorted(counts.keys()):
            for te in sorted(counts[cell].keys()):
                value = counts[cell][te]
                if float_values:
                    lines.append(f"{cell}\t{te}\t{value:.4f}\n")
                else:
                    lines.append(f"{cell}\t{te}\t{value}\n")
                if len(lines) >= 10000:
                    out.writelines(lines)
                    lines = []
        if lines:
            out.writelines(lines)


def reduce_sample(args):
    sample, hit_files, output_dir, enable_umi_dedup, skip_existing = args
    sample_name = sample["name"]
    unique_output = os.path.join(output_dir, "unique.tsv")
    multi_output = os.path.join(output_dir, "multi.tsv")

    if (
        skip_existing
        and os.path.exists(unique_output)
        and os.path.exists(multi_output)
    ):
        return {
            "success": True,
            "prefix": sample_name,
            "sample": sample_name,
            "skipped": True,
            "unique_file": unique_output,
            "multi_file": multi_output,
        }

    unique_counts = defaultdict(lambda: defaultdict(int))
    multi_counts = defaultdict(lambda: defaultdict(float))
    seen_umis = defaultdict(lambda: defaultdict(set))
    stats = {
        "hit_alignments": 0,
        "unique_mapped": 0,
        "multi_mapped": 0,
        "no_umi_hit_alignments": 0,
        "umi_duplicates": 0,
        "unique_te_reads": 0,
        "multi_te_reads": 0,
    }

    start_time = time.time()
    for hit_file in hit_files:
        if os.path.exists(hit_file) and os.path.getsize(hit_file) > 0:
            reduce_hit_file(
                hit_file,
                unique_counts,
                multi_counts,
                seen_umis,
                stats,
                enable_umi_dedup=enable_umi_dedup,
            )

    os.makedirs(output_dir, exist_ok=True)
    write_long_tsv(unique_output, unique_counts, float_values=False)
    write_long_tsv(multi_output, multi_counts, float_values=True)

    stats["unique_cells"] = len(unique_counts)
    stats["multi_cells"] = len(multi_counts)
    stats["unique_tes"] = len(
        {te for cell_tes in unique_counts.values() for te in cell_tes}
    )
    stats["multi_tes"] = len(
        {te for cell_tes in multi_counts.values() for te in cell_tes}
    )

    elapsed = timedelta(seconds=int(time.time() - start_time))
    return {
        "success": True,
        "prefix": sample_name,
        "sample": sample_name,
        "skipped": False,
        "unique_file": unique_output,
        "multi_file": multi_output,
        "stats": stats,
        "elapsed_time": elapsed,
    }


def signal_handler(sig, frame):
    print("\nInterrupted. Existing completed hit files can be reused with --reuse-hits.")
    sys.exit(130)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fast 10x U/M TE counter."
    )
    parser.add_argument("--bam", required=True)
    parser.add_argument("--whitelist", required=True)
    parser.add_argument("--te-annotation-gtf", required=True)
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Default: OUTPUT_DIR, or /qiyang/GitHub/scALTER/results/_tmp_counts when OUTPUT_DIR is None",
    )
    parser.add_argument(
        "--tmp-dir",
        default=TMP_DIR,
        help="Default: TMP_DIR, or <output-dir>/_tmp_bedtools when TMP_DIR is None",
    )
    parser.add_argument("--samtools", default=SAMTOOLS)
    parser.add_argument("--bedtools", default=BEDTOOLS)
    parser.add_argument("--awk", default=AWK)
    parser.add_argument("--cell-tag", default=CELL_TAG)
    parser.add_argument("--umi-tag", default=UMI_TAG)
    parser.add_argument("--min-mapq", type=int, default=MIN_MAPQ)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--reducer-threads", type=int, default=REDUCER_THREADS)
    parser.add_argument("--skip-existing", action="store_true", default=SKIP_EXISTING)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--reuse-hits", action="store_true", default=REUSE_HITS)
    parser.add_argument("--no-reuse-hits", dest="reuse_hits", action="store_false")
    parser.add_argument("--keep-tmp", action="store_true", default=KEEP_TMP)
    parser.add_argument("--remove-tmp", dest="keep_tmp", action="store_false")
    parser.add_argument(
        "--canonical-gtf-to-bed",
        action="store_true",
        default=CANONICAL_GTF_TO_BED,
        help="Use canonical GTF->BED conversion. Off by default to preserve the original script's coordinate behavior.",
    )
    parser.add_argument(
        "--preserve-original-coords",
        dest="canonical_gtf_to_bed",
        action="store_false",
        help="Use the original script's coordinate behavior.",
    )
    return parser


def main():
    signal.signal(signal.SIGINT, signal_handler)
    args = build_arg_parser().parse_args()

    samtools = resolve_tool("samtools", args.samtools)
    bedtools = resolve_tool("bedtools", args.bedtools)
    awk = resolve_awk(args.awk)

    gtf_file = args.te_annotation_gtf
    output_dir = args.output_dir or default_output_dir()

    tmp_dir = args.tmp_dir or os.path.join(output_dir, "_tmp_bedtools")
    bed_dir = os.path.join(tmp_dir, "te_bed")
    hit_root = os.path.join(tmp_dir, "hits")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    total_cores = cpu_count()
    threads = args.threads or max(1, min(total_cores - 4, 32))
    reducer_threads = args.reducer_threads or max(1, min(8, threads))

    print("=" * 80)
    print("Fast 10x U/M TE counting")
    print("=" * 80)
    print(f"Sample:          {SAMPLE_NAME}")
    print(f"BAM:             {args.bam}")
    print(f"Whitelist:       {args.whitelist}")
    print(f"GTF:             {gtf_file}")
    print(f"Output dir:      {output_dir}")
    print(f"Temporary dir:   {tmp_dir}")
    print(f"samtools:        {samtools}")
    print(f"bedtools:        {bedtools}")
    print(f"awk:             {awk}")
    print(f"Threads:         {threads}")
    print(f"Reducer threads: {reducer_threads}")
    print(f"TE level:        {TE_LEVEL}")
    print(f"Coordinate mode: {'canonical' if args.canonical_gtf_to_bed else 'preserve-original'}")
    print()

    for path in (args.bam, args.whitelist, gtf_file):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    samples = [
        {
            "name": SAMPLE_NAME,
            "prefix": SAMPLE_NAME,
            "bam_file": args.bam,
            "barcode_file": args.whitelist,
        }
    ]

    completed_samples = []
    active_samples = []
    for sample in samples:
        unique_output = os.path.join(output_dir, "unique.tsv")
        multi_output = os.path.join(output_dir, "multi.tsv")
        if (
            args.skip_existing
            and os.path.exists(unique_output)
            and os.path.exists(multi_output)
        ):
            completed_samples.append(sample)
        else:
            active_samples.append(sample)
    samples = active_samples

    print(f"Prepared {len(samples)} sample")
    if completed_samples:
        print(
            f"Skipped {len(completed_samples)} samples with existing U/M outputs: "
            + ", ".join(s["name"] for s in completed_samples)
        )
    print()

    if not samples:
        print("All requested samples already have outputs. Nothing to do.")
        return

    valid_chroms = get_bam_chromosomes(samples[0]["bam_file"])
    chrom_order = get_bam_chrom_order(samples[0]["bam_file"])

    print("Preparing per-chromosome TE BED files ...")
    start = time.time()
    bed_files, total_intervals = write_te_beds(
        gtf_file,
        valid_chroms,
        bed_dir,
        preserve_original_coords=not args.canonical_gtf_to_bed,
    )
    elapsed = timedelta(seconds=int(time.time() - start))
    print(f"  Wrote {len(bed_files)} chrom BEDs, {total_intervals:,} intervals in {elapsed}")
    print()

    chroms = [chrom for chrom in chrom_order if chrom in bed_files]
    if not chroms:
        raise RuntimeError("No shared chromosomes between BAM and TE GTF.")

    task_args = []
    sample_hit_files = defaultdict(list)
    for sample in samples:
        for chrom in chroms:
            hit_file = os.path.join(hit_root, sample["prefix"], f"{chrom}.hits.tsv.gz")
            sample_hit_files[sample["prefix"]].append(hit_file)
            task_args.append(
                (
                    sample,
                    chrom,
                    bed_files[chrom],
                    hit_file,
                    samtools,
                    bedtools,
                    awk,
                    args.cell_tag,
                    args.umi_tag,
                    args.min_mapq,
                    args.reuse_hits,
                )
            )

    print(f"Phase 1: running {len(task_args)} sample-chromosome intersect tasks ...")
    total_start = time.time()
    phase_start = time.time()
    with Pool(processes=threads) as pool:
        intersect_results = pool.map(intersect_one_chrom, task_args)
    phase_elapsed = timedelta(seconds=int(time.time() - phase_start))
    skipped_hits = sum(1 for r in intersect_results if r["skipped"])
    print(f"  Intersections done in {phase_elapsed}; reused {skipped_hits} existing hit files")
    print()

    print("Phase 2: reducing hit streams into U/M long TSVs ...")
    phase_start = time.time()
    reduce_args = [
        (
            sample,
            sample_hit_files[sample["prefix"]],
            output_dir,
            True,
            args.skip_existing,
        )
        for sample in samples
    ]
    with Pool(processes=min(reducer_threads, len(reduce_args))) as pool:
        reduce_results = pool.map(reduce_sample, reduce_args)
    phase_elapsed = timedelta(seconds=int(time.time() - phase_start))
    print(f"  Reduction done in {phase_elapsed}")
    print()

    successful = sum(1 for r in reduce_results if r.get("success"))
    total_elapsed = timedelta(seconds=int(time.time() - total_start))
    print("=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)
    print(f"Samples completed: {successful}/{len(reduce_results)}")
    print(f"Total time:        {total_elapsed}")
    for result in reduce_results:
        prefix = result["prefix"]
        if result.get("skipped"):
            print(f"  {prefix}: skipped existing outputs")
            continue
        stats = result.get("stats", {})
        print(
            f"  {prefix}: hit_alignments={stats.get('hit_alignments', 0):,}, "
            f"U={stats.get('unique_te_reads', 0):,}, "
            f"M={stats.get('multi_te_reads', 0):,}, "
            f"UMI_dup={stats.get('umi_duplicates', 0):,}"
        )
    print(f"Output directory: {output_dir}")

    if args.keep_tmp:
        print(f"Temporary hit files kept at: {tmp_dir}")
        print("Use --reuse-hits to reuse them for a reduction-only rerun.")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Removed temporary directory: {tmp_dir}")


if __name__ == "__main__":
    main()
