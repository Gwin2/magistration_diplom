#!/usr/bin/env bash
set -euo pipefail

TS_INFERENCE_URL="${TS_INFERENCE_URL:-http://localhost:8080}"
MODEL_NAME="${1:-uav_detector}"
IMAGE_PATH="${2:?Provide image path as second arg}"

curl -X POST "${TS_INFERENCE_URL}/predictions/${MODEL_NAME}" \
  -T "${IMAGE_PATH}"

echo
