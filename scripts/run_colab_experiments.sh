#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 SUBJECT_MANIFEST LANDMARK_MANIFEST DATA_ROOT CACHE_DIR OUTPUT_DIR"
  exit 2
fi

SUBJECT_MANIFEST=$1
LANDMARK_MANIFEST=$2
DATA_ROOT=$3
CACHE_DIR=$4
OUTPUT_DIR=$5

COMMON=(
  --task core_15
  --subject-manifest "$SUBJECT_MANIFEST"
  --region-manifest "$LANDMARK_MANIFEST"
  --data-root "$DATA_ROOT"
  --cache-dir "$CACHE_DIR"
  --output-dir "$OUTPUT_DIR"
  --view-policy service
  --selection-metric grade_qwk
  --epochs 15
  --batch-size 4
  --num-workers 4
)

ALL_VIEW_COMMON=(
  --task core_15
  --subject-manifest "$SUBJECT_MANIFEST"
  --region-manifest "$LANDMARK_MANIFEST"
  --data-root "$DATA_ROOT"
  --cache-dir "$CACHE_DIR"
  --output-dir "$OUTPUT_DIR"
  --view-policy all
  --selection-metric grade_qwk
  --epochs 15
  --batch-size 4
  --num-workers 4
)

skin-train "${ALL_VIEW_COMMON[@]}" \
  --run-name landmark_all_regression_mean \
  --grade-objective regression \
  --fusion mean

skin-train "${COMMON[@]}" \
  --run-name landmark_service_regression_mean \
  --grade-objective regression \
  --fusion mean

skin-train "${COMMON[@]}" \
  --run-name landmark_service_softmax_mean \
  --grade-objective softmax \
  --fusion mean

skin-train "${COMMON[@]}" \
  --run-name landmark_service_ordinal_mean \
  --grade-objective ordinal \
  --fusion mean

skin-train "${COMMON[@]}" \
  --run-name landmark_service_softmax_concat \
  --grade-objective softmax \
  --fusion concat

skin-train "${COMMON[@]}" \
  --run-name landmark_service_softmax_attention \
  --grade-objective softmax \
  --fusion attention
