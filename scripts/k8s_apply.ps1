$ErrorActionPreference = "Stop"

kubectl apply -k k8s/base
kubectl -n uav-mlops wait --for=condition=available deployment/postgres --timeout=180s
kubectl -n uav-mlops wait --for=condition=available deployment/minio --timeout=180s
kubectl -n uav-mlops wait --for=condition=complete job/minio-init --timeout=180s
kubectl -n uav-mlops wait --for=condition=available deployment/mlflow --timeout=180s
kubectl -n uav-mlops wait --for=condition=available deployment/torchserve --timeout=180s
kubectl -n uav-mlops wait --for=condition=available deployment/pushgateway --timeout=180s
kubectl -n uav-mlops wait --for=condition=available deployment/prometheus --timeout=180s
kubectl -n uav-mlops wait --for=condition=available deployment/grafana --timeout=180s

Write-Host "Kubernetes stack is ready in namespace uav-mlops."
