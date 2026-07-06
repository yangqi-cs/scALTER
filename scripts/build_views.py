#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build aligned unique/multi/merge sparse matrices from scALTER long TSV output.

Input:
  unique.tsv
  multi.tsv

Output:
  /qiyang/GitHub/scALTER/results/views/
    raw_exp/
      barcodes.tsv
      features.tsv
      obs_metadata.csv
      var_metadata.csv
      unique.npz
      multi.npz
      merge.npz
      manifest.json
    h5ad/
      scalter_subfamily_u_m_aligned.h5ad

The h5ad stores:
  adata.obs              aligned cell barcodes and batch metadata
  adata.var              aligned feature names
  adata.layers["unique"] unique sparse matrix
  adata.layers["multi"]  multi sparse matrix
  adata.X                unique + multi sparse matrix, suitable as default model input
"""

# Example production run:
# python -u /qiyang/GitHub/scALTER/scripts/build_views.py

import json
import argparse
import os
import platform
import sys
from datetime import datetime

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


# =========================
# User configuration
# =========================
TE_LEVEL = "subfamily"
SAMPLE_NAME = "scalter"

# Existing output path from scripts/extract_counts.py.
BASE_DIR = "/qiyang/GitHub/scALTER/results"
INPUT_DIR = os.path.join(BASE_DIR, "_tmp_counts")

OUTPUT_DIR = os.path.join(BASE_DIR, "views")
RAW_EXP_DIR = os.path.join(OUTPUT_DIR, "raw_exp")
H5AD_DIR = os.path.join(OUTPUT_DIR, "h5ad")

UNIQUE_TSV = os.path.join(INPUT_DIR, "unique.tsv")
MULTI_TSV = os.path.join(INPUT_DIR, "multi.tsv")

# "union" keeps all cells/features from either matrix and fills missing values
# with zero. This avoids dropping TE features that appear only in multi.
ALIGN_MODE = "union"  # "union" or "intersection"

DTYPE = np.float32


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build aligned scALTER U/M/merge matrix views from long TSV files."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing unique.tsv and multi.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for raw_exp/ and h5ad/ outputs.",
    )
    parser.add_argument("--unique-tsv", default=None)
    parser.add_argument("--multi-tsv", default=None)
    parser.add_argument(
        "--align-mode",
        choices=["union", "intersection"],
        default=ALIGN_MODE,
    )
    return parser


def configure_from_args(args):
    global TE_LEVEL, INPUT_DIR, OUTPUT_DIR
    global RAW_EXP_DIR, H5AD_DIR, UNIQUE_TSV, MULTI_TSV, ALIGN_MODE

    INPUT_DIR = args.input_dir or "/qiyang/GitHub/scALTER/results/_tmp_counts"
    OUTPUT_DIR = args.output_dir or "/qiyang/GitHub/scALTER/results/views"
    RAW_EXP_DIR = os.path.join(OUTPUT_DIR, "raw_exp")
    H5AD_DIR = os.path.join(OUTPUT_DIR, "h5ad")
    UNIQUE_TSV = args.unique_tsv or os.path.join(INPUT_DIR, "unique.tsv")
    MULTI_TSV = args.multi_tsv or os.path.join(INPUT_DIR, "multi.tsv")
    ALIGN_MODE = args.align_mode


def ensure_inputs():
    missing = [p for p in [UNIQUE_TSV, MULTI_TSV] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing input TSV files:\n" + "\n".join(missing))

    os.makedirs(RAW_EXP_DIR, exist_ok=True)
    os.makedirs(H5AD_DIR, exist_ok=True)


def read_long_tsv(path):
    print(f"Reading: {path}")
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["cell", "feature", "count"],
        dtype={"cell": "string", "feature": "string", "count": DTYPE},
    )
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    df["cell"] = df["cell"].astype(str)
    df["feature"] = df["feature"].astype(str)
    df["count"] = df["count"].astype(DTYPE)

    # Make repeated rows deterministic and compact before sparse construction.
    df = df.groupby(["cell", "feature"], as_index=False, sort=False)["count"].sum()
    print(
        f"  rows={df.shape[0]:,}, cells={df['cell'].nunique():,}, "
        f"features={df['feature'].nunique():,}, total={df['count'].sum():.3f}"
    )
    return df


def align_axes(unique_df, multi_df, mode="union"):
    if mode not in {"union", "intersection"}:
        raise ValueError("ALIGN_MODE must be 'union' or 'intersection'")

    u_cells = set(unique_df["cell"])
    m_cells = set(multi_df["cell"])
    u_features = set(unique_df["feature"])
    m_features = set(multi_df["feature"])

    if mode == "union":
        barcodes = sorted(u_cells | m_cells)
        features = sorted(u_features | m_features)
    else:
        barcodes = sorted(u_cells & m_cells)
        features = sorted(u_features & m_features)

    if not barcodes:
        raise ValueError("No aligned barcodes.")
    if not features:
        raise ValueError("No aligned features.")

    print(f"Aligned mode: {mode}")
    print(f"  unique cells={len(u_cells):,}, multi cells={len(m_cells):,}, aligned={len(barcodes):,}")
    print(f"  unique features={len(u_features):,}, multi features={len(m_features):,}, aligned={len(features):,}")

    return pd.Index(barcodes, name="barcode"), pd.Index(features, name="feature")


def build_sparse_matrix(df, barcodes, features):
    cell_to_idx = pd.Series(np.arange(len(barcodes), dtype=np.int64), index=barcodes)
    feature_to_idx = pd.Series(np.arange(len(features), dtype=np.int64), index=features)

    row = df["cell"].map(cell_to_idx)
    col = df["feature"].map(feature_to_idx)
    keep = row.notna() & col.notna()

    dropped = int((~keep).sum())
    if dropped:
        print(f"  Dropped {dropped:,} rows outside aligned axes")

    row = row[keep].astype(np.int64).to_numpy()
    col = col[keep].astype(np.int64).to_numpy()
    data = df.loc[keep, "count"].astype(DTYPE).to_numpy()

    mat = sparse.coo_matrix(
        (data, (row, col)),
        shape=(len(barcodes), len(features)),
        dtype=DTYPE,
    ).tocsr()
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


def write_list(index, path):
    with open(path, "w") as f:
        for item in index.astype(str):
            f.write(f"{item}\n")


def build_metadata(barcodes, features, unique_mat, multi_mat):
    obs = pd.DataFrame(index=barcodes.astype(str))
    obs.index.name = "barcode"
    obs["barcode"] = obs.index.astype(str)
    obs["barcode_core"] = obs["barcode"].str.replace(r"-\d+$", "", regex=True)
    obs["batch"] = SAMPLE_NAME
    obs["sample"] = SAMPLE_NAME
    obs["unique_counts"] = np.asarray(unique_mat.sum(axis=1)).ravel().astype(DTYPE)
    obs["multi_counts"] = np.asarray(multi_mat.sum(axis=1)).ravel().astype(DTYPE)
    obs["total_counts"] = obs["unique_counts"] + obs["multi_counts"]
    obs["unique_n_features"] = np.diff(unique_mat.indptr).astype(np.int32)
    obs["multi_n_features"] = np.diff(multi_mat.indptr).astype(np.int32)
    obs["total_n_features"] = np.diff((unique_mat + multi_mat).tocsr().indptr).astype(np.int32)

    var = pd.DataFrame(index=features.astype(str))
    var.index.name = "feature"
    var["feature"] = var.index.astype(str)
    var["unique_n_cells"] = np.diff(unique_mat.tocsc().indptr).astype(np.int32)
    var["multi_n_cells"] = np.diff(multi_mat.tocsc().indptr).astype(np.int32)
    var["total_n_cells"] = np.diff((unique_mat + multi_mat).tocsc().indptr).astype(np.int32)
    var["unique_counts"] = np.asarray(unique_mat.sum(axis=0)).ravel().astype(DTYPE)
    var["multi_counts"] = np.asarray(multi_mat.sum(axis=0)).ravel().astype(DTYPE)
    var["total_counts"] = var["unique_counts"] + var["multi_counts"]

    return obs, var


def save_raw_exp(barcodes, features, obs, var, unique_mat, multi_mat, merge_mat, manifest):
    print(f"Saving raw expression NPZ output: {RAW_EXP_DIR}")
    write_list(barcodes, os.path.join(RAW_EXP_DIR, "barcodes.tsv"))
    write_list(features, os.path.join(RAW_EXP_DIR, "features.tsv"))
    obs.to_csv(os.path.join(RAW_EXP_DIR, "obs_metadata.csv"))
    var.to_csv(os.path.join(RAW_EXP_DIR, "var_metadata.csv"))
    sparse.save_npz(os.path.join(RAW_EXP_DIR, "unique.npz"), unique_mat)
    sparse.save_npz(os.path.join(RAW_EXP_DIR, "multi.npz"), multi_mat)
    sparse.save_npz(os.path.join(RAW_EXP_DIR, "merge.npz"), merge_mat)
    with open(os.path.join(RAW_EXP_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def save_h5ad(obs, var, unique_mat, multi_mat, merge_mat, manifest):
    print(f"Saving h5ad output: {H5AD_DIR}")
    adata = ad.AnnData(X=merge_mat, obs=obs.copy(), var=var.copy())
    adata.layers["unique"] = unique_mat.copy()
    adata.layers["multi"] = multi_mat.copy()
    adata.uns["scalter_um_manifest"] = manifest
    adata.uns["matrix_semantics"] = {
        "X": "unique + multi",
        "layers_unique": "unique TE counts",
        "layers_multi": "fractional multi/ambiguous TE counts",
    }

    out_file = os.path.join(H5AD_DIR, f"{SAMPLE_NAME}_{TE_LEVEL}_u_m_aligned.h5ad")
    adata.write_h5ad(out_file, compression="gzip")
    return out_file


def build_manifest(barcodes, features, unique_mat, multi_mat, merge_mat):
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/build_views.py",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "te_level": TE_LEVEL,
        "sample": SAMPLE_NAME,
        "align_mode": ALIGN_MODE,
        "input_unique_tsv": UNIQUE_TSV,
        "input_multi_tsv": MULTI_TSV,
        "output_dir": OUTPUT_DIR,
        "shape": {
            "n_cells": int(len(barcodes)),
            "n_features": int(len(features)),
        },
        "nnz": {
            "unique": int(unique_mat.nnz),
            "multi": int(multi_mat.nnz),
            "merge": int(merge_mat.nnz),
        },
        "sum": {
            "unique": float(unique_mat.sum()),
            "multi": float(multi_mat.sum()),
            "merge": float(merge_mat.sum()),
        },
        "dtype": str(DTYPE),
        "files": {
            "barcodes": "raw_exp/barcodes.tsv",
            "features": "raw_exp/features.tsv",
            "obs_metadata": "raw_exp/obs_metadata.csv",
            "var_metadata": "raw_exp/var_metadata.csv",
            "unique_npz": "raw_exp/unique.npz",
            "multi_npz": "raw_exp/multi.npz",
            "merge_npz": "raw_exp/merge.npz",
            "h5ad": f"h5ad/{SAMPLE_NAME}_{TE_LEVEL}_u_m_aligned.h5ad",
        },
    }


def main():
    args = build_arg_parser().parse_args()
    configure_from_args(args)

    print("=" * 90)
    print("scALTER U/M TSV -> RAW EXPRESSION + H5AD")
    print("=" * 90)
    print(f"TE_LEVEL:        {TE_LEVEL}")
    print(f"SAMPLE:          {SAMPLE_NAME}")
    print(f"INPUT_DIR:       {INPUT_DIR}")
    print(f"OUTPUT_DIR:      {OUTPUT_DIR}")
    print(f"ALIGN_MODE:      {ALIGN_MODE}")
    print("=" * 90)

    ensure_inputs()
    unique_df = read_long_tsv(UNIQUE_TSV)
    multi_df = read_long_tsv(MULTI_TSV)

    barcodes, features = align_axes(unique_df, multi_df, mode=ALIGN_MODE)

    print("Building unique sparse matrix...")
    unique_mat = build_sparse_matrix(unique_df, barcodes, features)
    print(f"  unique shape={unique_mat.shape}, nnz={unique_mat.nnz:,}, sum={float(unique_mat.sum()):.3f}")

    print("Building multi sparse matrix...")
    multi_mat = build_sparse_matrix(multi_df, barcodes, features)
    print(f"  multi shape={multi_mat.shape}, nnz={multi_mat.nnz:,}, sum={float(multi_mat.sum()):.3f}")

    merge_mat = (unique_mat + multi_mat).tocsr().astype(DTYPE)
    merge_mat.sum_duplicates()
    merge_mat.eliminate_zeros()
    print(f"  merge shape={merge_mat.shape}, nnz={merge_mat.nnz:,}, sum={float(merge_mat.sum()):.3f}")

    obs, var = build_metadata(barcodes, features, unique_mat, multi_mat)
    manifest = build_manifest(barcodes, features, unique_mat, multi_mat, merge_mat)

    save_raw_exp(barcodes, features, obs, var, unique_mat, multi_mat, merge_mat, manifest)
    h5ad_file = save_h5ad(obs, var, unique_mat, multi_mat, merge_mat, manifest)

    print("=" * 90)
    print("Done.")
    print(f"Raw expression dir: {RAW_EXP_DIR}")
    print(f"H5AD file:       {h5ad_file}")
    print("=" * 90)


if __name__ == "__main__":
    main()
