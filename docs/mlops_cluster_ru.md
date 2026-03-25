# MLOps кластер

Документ описывает локальный и cluster-friendly контур для обучения, оценки и инференса.

## Состав стека

- tracking и artifacts: `MLflow + Postgres + MinIO`
- control plane: `control-api`
- observability: `Prometheus + Grafana + Alertmanager + Pushgateway + exporters`
- experiment review: `TensorBoard` через `control-api`
- serving: `TorchServe`
- orchestration: `Docker Compose` локально, `Kubernetes` для кластера

## Предварительные требования

- Docker и Docker Compose
- Python `3.10+`
- для Kubernetes: `kubectl` и рабочий storage class для PVC

## Docker Compose

### Базовый режим

Поднимает storage, tracking, control plane, UI и observability без тяжёлых train/inference сервисов.

```bash
cp .env.example .env
docker compose up -d --build
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

Контейнеры в compose не используют фиксированные `container_name`, поэтому стек можно безопасно запускать как из терминала, так и из IDE. Docker будет создавать project-scoped имена вида `magistration_diplom-mlflow-1`.

### Полный режим

```bash
docker compose --profile training --profile inference up -d --build
```

### Профили

- `training`: `trainer`, `run-metrics-exporter`
- `inference`: `torchserve`

## Проверка сервисов

- UI: `http://localhost:${UI_HOST_PORT}` (`18090`)
- Control API: `http://localhost:${CONTROL_API_HOST_PORT}/health` (`18070`)
- MLflow: `http://localhost:${MLFLOW_HOST_PORT}` (`15000`)
- MinIO API: `http://localhost:${MINIO_API_HOST_PORT}` (`19000`)
- MinIO Console: `http://localhost:${MINIO_CONSOLE_HOST_PORT}` (`19001`)
- Prometheus: `http://localhost:${PROMETHEUS_HOST_PORT}` (`19090`)
- Grafana: `http://localhost:${GRAFANA_HOST_PORT}` (`13000`)
- Pushgateway: `http://localhost:${PUSHGATEWAY_HOST_PORT}` (`19091`)
- Alertmanager: `http://localhost:${ALERTMANAGER_HOST_PORT}` (`19093`)
- Process Exporter: `http://localhost:${PROCESS_EXPORTER_HOST_PORT}` (`19256`)
- TorchServe ping: `http://localhost:${TORCHSERVE_INFERENCE_HOST_PORT}/ping` (`18080`)

После изменения портов выполните `docker compose down` и затем `docker compose up -d --build`.

Для обычной остановки и последующего старта используйте:

```bash
docker compose stop
docker compose start
```

## Train/Eval с логированием

Через CLI:

```bash
docker compose run --rm --profile training trainer uav-vit train --config configs/experiments/yolos_tiny.yaml
docker compose run --rm --profile training trainer uav-vit evaluate --config configs/experiments/yolos_tiny.yaml --split test
```

Через UI:

- открыть `Studio`
- выбрать или создать YAML
- запустить `Start Training` или `Run Evaluation`
- наблюдать jobs, MLflow и TensorBoard из вкладки `Experiments`

## TorchServe

Export модели:

```bash
python scripts/export_torchserve.py \
  --config configs/experiments/yolos_tiny.yaml \
  --checkpoint runs/yolos_tiny_uav/best.pt \
  --model-name uav_detector \
  --export-path model-store \
  --force
```

Регистрация:

```bash
bash scripts/torchserve_register_model.sh uav_detector uav_detector.mar
```

Smoke test:

```bash
bash scripts/torchserve_infer.sh uav_detector /path/to/frame.jpg
```

## Kubernetes

### Сборка образов

```bash
bash scripts/mlops_build_images.sh
```

После публикации образов обновите `image:` в manifest'ах внутри `k8s/base`.

### Deployment

```bash
bash scripts/k8s_apply.sh
kubectl -n uav-mlops get pods
```

### Jobs в кластере

```bash
kubectl -n uav-mlops delete job uav-train --ignore-not-found
kubectl -n uav-mlops apply -f k8s/base/training-job.yaml
kubectl -n uav-mlops logs -f job/uav-train

kubectl -n uav-mlops delete job uav-evaluate --ignore-not-found
kubectl -n uav-mlops apply -f k8s/base/evaluation-job.yaml
kubectl -n uav-mlops logs -f job/uav-evaluate
```

## Хранение артефактов

- MLflow metadata: PostgreSQL
- MLflow artifacts: MinIO bucket `mlflow`
- TensorBoard logs и checkpoints: `runs/*`
- TorchServe model-store: `model-store/`

## Очистка

```bash
bash scripts/k8s_cleanup.sh
docker compose down -v
```
