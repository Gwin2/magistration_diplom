#!/usr/bin/env bash
set -euo pipefail

TS_MANAGEMENT_URL="${TS_MANAGEMENT_URL:-http://localhost:8081}"
MODEL_NAME="${1:-uav_detector}"
MAR_FILE="${2:-uav_detector.mar}"
INITIAL_WORKERS="${INITIAL_WORKERS:-1}"
SYNCHRONOUS="${SYNCHRONOUS:-true}"

curl -X POST "${TS_MANAGEMENT_URL}/models" \
  -F "url=${MAR_FILE}" \
  -F "model_name=${MODEL_NAME}" \
  -F "initial_workers=${INITIAL_WORKERS}" \
  -F "synchronous=${SYNCHRONOUS}"

echo
echo "Model '${MODEL_NAME}' registration request sent."
