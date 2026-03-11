# MLOps кластер: MLflow + TorchServe + Prometheus + Grafana + Kubernetes + Docker Compose

Документ описывает production-like контур для обучения и инференса:

- Tracking и артефакты: `MLflow + Postgres + MinIO`
- Serving: `TorchServe`
- Observability: `Prometheus + Grafana + Pushgateway + exporters`
- Alerting and governance: `Alertmanager + resource limits + process-exporter`
- Оркестрация: `Docker Compose` (локально), `Kubernetes` (кластер)

## 1. Предварительные требования

- Docker + Docker Compose
- Python `3.10+`
- для Kubernetes: `kubectl` и кластер с динамическим provisioner PVC

## 2. Локальный MLOps стек через Docker Compose

## 2.1 Подъем сервисов

Linux/macOS:

```bash
cp .env.example .env
docker compose up -d --build
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

## 2.2 Проверка сервисов

- MLflow UI: `http://localhost:${MLFLOW_HOST_PORT}` (`15000`)
- MinIO API: `http://localhost:${MINIO_API_HOST_PORT}` (`19000`)
- MinIO Console: `http://localhost:${MINIO_CONSOLE_HOST_PORT}` (`19001`)
- TorchServe health: `http://localhost:${TORCHSERVE_INFERENCE_HOST_PORT}/ping` (`18080`)
- TorchServe management: `http://localhost:${TORCHSERVE_MANAGEMENT_HOST_PORT}/models` (`18081`)
- Prometheus: `http://localhost:${PROMETHEUS_HOST_PORT}` (`19090`)
- Grafana: `http://localhost:${GRAFANA_HOST_PORT}` (`13000`)
- Pushgateway: `http://localhost:${PUSHGATEWAY_HOST_PORT}` (`19091`)
- Alertmanager: `http://localhost:${ALERTMANAGER_HOST_PORT}` (`19093`)
- Process Exporter: `http://localhost:${PROCESS_EXPORTER_HOST_PORT}` (`19256`)
- Split UI Control Center: `http://localhost:${UI_HOST_PORT}` (`18090`)

Порты задаются через `.env` (см. `.env.example`), поэтому их можно менять без правки `docker-compose.yml`.
После изменения портов выполните: `docker compose down && docker compose up -d --build`.

Логин Grafana берется из `.env`:

- `GRAFANA_ADMIN_USER`
- `GRAFANA_ADMIN_PASSWORD`

Автоматически загружаются дашборды:

- `UAV System Overview`
- `UAV Training and Inference`
- `UAV Resource Control`

По умолчанию `trainer`, `run-metrics-exporter` и `torchserve` отключены через profiles.
Для полного контура:

```bash
docker compose --profile training --profile inference up -d --build
```

## 2.3 Обучение и оценка с логированием в MLflow

```bash
docker compose run --rm trainer uav-vit train --config configs/experiments/yolos_tiny.yaml
docker compose run --rm trainer uav-vit evaluate --config configs/experiments/yolos_tiny.yaml --split test
```

Примечание: в YAML-конфигах включен блок `mlflow`, поэтому параметры, метрики и артефакты отправляются в MLflow автоматически.
Также train/eval отправляют live-метрики в Pushgateway для Grafana.

## 2.4 Публикация модели в TorchServe

```bash
python scripts/export_torchserve.py \
  --config configs/experiments/yolos_tiny.yaml \
  --checkpoint runs/yolos_tiny_uav/best.pt \
  --model-name uav_detector \
  --export-path model-store \
  --force
```

```bash
bash scripts/torchserve_register_model.sh uav_detector uav_detector.mar
```

Smoke-test инференса:

```bash
bash scripts/torchserve_infer.sh uav_detector /path/to/frame.jpg
```

## 3. Kubernetes кластер

## 3.1 Сборка образов

```bash
bash scripts/mlops_build_images.sh
```

После сборки опубликуйте образы в реестр и замените `image:` в:

- `k8s/base/mlflow-deployment.yaml`
- `k8s/base/torchserve-deployment.yaml`
- `k8s/base/training-job.yaml`
- `k8s/base/evaluation-job.yaml`

## 3.2 Деплой инфраструктуры

```bash
bash scripts/k8s_apply.sh
kubectl -n uav-mlops get pods
```

## 3.3 Запуск обучения и оценки в кластере

```bash
kubectl -n uav-mlops delete job uav-train --ignore-not-found
kubectl -n uav-mlops apply -f k8s/base/training-job.yaml
kubectl -n uav-mlops logs -f job/uav-train

kubectl -n uav-mlops delete job uav-evaluate --ignore-not-found
kubectl -n uav-mlops apply -f k8s/base/evaluation-job.yaml
kubectl -n uav-mlops logs -f job/uav-evaluate
```

## 3.4 Доступ к сервисам кластера

```bash
kubectl -n uav-mlops port-forward svc/mlflow 5000:5000
kubectl -n uav-mlops port-forward svc/torchserve 8080:8080 8081:8081
kubectl -n uav-mlops port-forward svc/prometheus 9090:9090
kubectl -n uav-mlops port-forward svc/grafana 3000:3000
```

## 4. Хранение артефактов

- MLflow metadata: PostgreSQL
- MLflow artifacts: MinIO bucket `mlflow`
- TorchServe model-store: PVC `model-store-pvc`
- Чекпоинты и метрики экспериментов: PVC `runs-pvc`

## 5. Очистка

```bash
bash scripts/k8s_cleanup.sh
docker compose down -v
```

Дополнительно: описание дашбордов и источников метрик в [grafana_web_view_ru.md](grafana_web_view_ru.md).
