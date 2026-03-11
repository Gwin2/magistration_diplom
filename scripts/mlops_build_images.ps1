param(
  [string]$TrainerImage = "uav-vit-trainer:latest",
  [string]$MlflowImage = "uav-vit-mlflow:latest",
  [string]$TorchserveImage = "uav-vit-torchserve:latest"
)

$ErrorActionPreference = "Stop"

docker build -t $TrainerImage -f Dockerfile .
docker build -t $MlflowImage -f docker/mlflow/Dockerfile .
docker build -t $TorchserveImage -f docker/torchserve/Dockerfile .

Write-Host "Images built:"
Write-Host "  $TrainerImage"
Write-Host "  $MlflowImage"
Write-Host "  $TorchserveImage"
