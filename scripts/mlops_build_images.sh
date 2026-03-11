#!/usr/bin/env bash
set -euo pipefail

TRAINER_IMAGE="${TRAINER_IMAGE:-uav-vit-trainer:latest}"
MLFLOW_IMAGE="${MLFLOW_IMAGE:-uav-vit-mlflow:latest}"
TORCHSERVE_IMAGE="${TORCHSERVE_IMAGE:-uav-vit-torchserve:latest}"

docker build -t "$TRAINER_IMAGE" -f Dockerfile .
docker build -t "$MLFLOW_IMAGE" -f docker/mlflow/Dockerfile .
docker build -t "$TORCHSERVE_IMAGE" -f docker/torchserve/Dockerfile .

echo "Images built:"
echo "  $TRAINER_IMAGE"
echo "  $MLFLOW_IMAGE"
echo "  $TORCHSERVE_IMAGE"
