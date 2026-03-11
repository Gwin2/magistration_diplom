#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/yolos_tiny.yaml}"
RUNS_DIR="${2:-runs}"
REPORTS_DIR="${3:-reports}"
METADATA_CSV="${4:-}"

uav-vit train --config "$CONFIG"
uav-vit evaluate --config "$CONFIG" --split test
uav-vit summarize --runs-dir "$RUNS_DIR" --output-dir "$REPORTS_DIR"

if [ -n "$METADATA_CSV" ]; then
  uav-vit analyze-conditions --config "$CONFIG" --metadata-csv "$METADATA_CSV" --column weather --split test
  uav-vit analyze-conditions --config "$CONFIG" --metadata-csv "$METADATA_CSV" --column quality --split test
  uav-vit analyze-conditions --config "$CONFIG" --metadata-csv "$METADATA_CSV" --column maneuver --split test
fi

echo "Pipeline completed."
