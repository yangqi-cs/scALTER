#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/qiyang/Anaconda/conda/envs/teexp/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/qiyang/GitHub/scALTER}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/pbmc8k}"
TE_LEVEL="${TE_LEVEL:-subfamily}"
SAMPLE_PREFIX="${SAMPLE_PREFIX:-pbmc8k}"
WORKERS="${WORKERS:-32}"
REDUCER_WORKERS="${REDUCER_WORKERS:-1}"

BASE_DIR="${RESULT_ROOT}/my_${TE_LEVEL}"
LOG_DIR="${RESULT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "[1/3] Extracting unique/multi TE counts from BAM"
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/1_extract_u_m_counts_by_cb_bedtools_parallel.py" \
  --te-level "${TE_LEVEL}" \
  --sample-prefix "${SAMPLE_PREFIX}" \
  --output-dir "${BASE_DIR}/1_${TE_LEVEL}_u_m" \
  --workers "${WORKERS}" \
  --reducer-workers "${REDUCER_WORKERS}" \
  2>&1 | tee "${LOG_DIR}/1_extract_u_m_counts_by_cb_bedtools_parallel.log"

echo "[2/3] Building aligned unique/multi/merge views"
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/2_build_u_m_views.py" \
  --te-level "${TE_LEVEL}" \
  --sample-prefix "${SAMPLE_PREFIX}" \
  --base-dir "${BASE_DIR}" \
  --input-dir "${BASE_DIR}/1_${TE_LEVEL}_u_m" \
  --output-dir "${BASE_DIR}/2_${TE_LEVEL}_u_m_aligned" \
  2>&1 | tee "${LOG_DIR}/2_build_u_m_views.log"

echo "[3/3] Training scVI-PoE model"
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/3_train_scvi_poe_unique_multi_merge_best_model.py" \
  --te-level "${TE_LEVEL}" \
  --base-dir "${BASE_DIR}" \
  --data-dir "2_${TE_LEVEL}_u_m_aligned/aligned_npz" \
  --output-dir "3_cross_view/1_7_scvi_poe_unique_multi_merge_best_model" \
  2>&1 | tee "${LOG_DIR}/3_train_scvi_poe_unique_multi_merge_best_model.log"

echo "Done. Results are under ${BASE_DIR}"
