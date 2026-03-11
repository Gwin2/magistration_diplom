# UAV ViT Thesis Framework

Фреймворк для магистерской работы по детекции БПЛА в сложных условиях: плохая погода, низкое качество кадра, резкие маневры и малые объекты.

## Возможности

- Единый research pipeline: `video -> COCO -> train/eval -> отчеты`
- Модели ViT/DETR через `transformers` + реестр для кастомных архитектур
- Метрики: `mAP`, `mAP50`, `mAP75`, `mAR`, latency/FPS
- Анализ по условиям: `weather`, `quality`, `maneuver`
- MLOps-слой: `MLflow + TorchServe + MinIO + Postgres` (`docker compose` и `k8s`)

## Структура проекта

- `src/uav_vit` - код пайплайна обучения, оценки, аналитики
- `configs/experiments` - YAML-конфиги экспериментов
- `scripts` - скрипты bootstrap, pipeline, MLOps и k8s
- `docs` - методология, шаблоны результатов, деплой
- `k8s/base` - базовые манифесты кластера

## Быстрый старт (локально)

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install
```

## Подготовка данных (video -> COCO)

Ожидаемые поля CSV:

- `video_name`, `frame_idx`, `x_min`, `y_min`, `x_max`, `y_max`, `class_name`
- опционально: `weather`, `quality`, `maneuver`, `split`

```bash
uav-vit convert-video \
  --video-dir data/raw/videos \
  --annotations-csv data/raw/annotations.csv \
  --output-dir data/processed/uav_coco
```

## Обучение, оценка, отчеты

```bash
uav-vit train --config configs/experiments/yolos_tiny.yaml
uav-vit evaluate --config configs/experiments/yolos_tiny.yaml --split test
uav-vit summarize --runs-dir runs --output-dir reports
```

Артефакты:

- `runs/<experiment>/best.pt`, `runs/<experiment>/metrics.csv`
- `reports/summary.csv`
- `reports/summary.tex`

## Анализ по условиям

```bash
uav-vit analyze-conditions \
  --config configs/experiments/yolos_tiny.yaml \
  --metadata-csv data/processed/uav_coco/frame_metadata.csv \
  --column weather \
  --split test
```

Для других групп используйте `--column quality` или `--column maneuver`.

## Полный прогон

Windows:

```powershell
.\scripts\run_full_pipeline.ps1 -Config configs/experiments/yolos_tiny.yaml -MetadataCsv data/processed/uav_coco/frame_metadata.csv
```

Linux/macOS:

```bash
bash scripts/run_full_pipeline.sh configs/experiments/yolos_tiny.yaml runs reports data/processed/uav_coco/frame_metadata.csv
```

## Кастомные архитектуры

1. Добавьте модуль, например `src/custom_models/my_detector.py`.
2. Зарегистрируйте билдер: `@register_model("my_detector")`.
3. Укажите в YAML:
   - `model.name: my_detector`
   - `model.custom_modules: ["custom_models.my_detector"]`

Подробнее: [docs/architecture_extension.md](docs/architecture_extension.md)

## MLOps стек (MLflow + TorchServe + Grafana)

Базовый режим (UI + мониторинг, без запуска обучения/инференса):

```bash
cp .env.example .env
docker compose up -d --build
```

Полный режим (добавить сервисы обучения и инференса):

```bash
docker compose --profile training --profile inference up -d --build
```

Если нужен только один контур:

```bash
docker compose --profile training up -d --build
docker compose --profile inference up -d --build
```

Сервисы по умолчанию:

- Split UI Control Center: `http://localhost:${UI_HOST_PORT}` (`18090`)
- MLflow UI: `http://localhost:${MLFLOW_HOST_PORT}` (`15000`)
- MinIO: `http://localhost:${MINIO_API_HOST_PORT}` (`19000`)
  console: `http://localhost:${MINIO_CONSOLE_HOST_PORT}` (`19001`)
- Prometheus: `http://localhost:${PROMETHEUS_HOST_PORT}` (`19090`)
- Grafana: `http://localhost:${GRAFANA_HOST_PORT}` (`13000`)
- Pushgateway: `http://localhost:${PUSHGATEWAY_HOST_PORT}` (`19091`)
- Blackbox Exporter: `http://localhost:${BLACKBOX_HOST_PORT}` (`19116`)
- Alertmanager: `http://localhost:${ALERTMANAGER_HOST_PORT}` (`19093`)
- Process Exporter: `http://localhost:${PROCESS_EXPORTER_HOST_PORT}` (`19256`)

Все host-порты вынесены в переменные `.env`, поэтому при конфликте достаточно поменять значение без редактирования `docker-compose.yml`.
После изменения портов выполните `docker compose down` и затем `docker compose up -d --build`.

Сервисы `trainer`, `run-metrics-exporter` и `torchserve` вынесены в `profiles` и не стартуют в базовом режиме.
Это позволяет поднимать UI и мониторинг отдельно от тяжелых ML-компонентов.

Для UI доступны разделы:

- Fine-Tuning Studio (ручная настройка и генерация конфигов)
- Auto Tuning (автоплан дообучения)
- CI/CD Pipeline Control
- Resource Governance (генерация `docker-compose.override.yml` по ресурсным профилям)
- Container and Process Consumption (live-потребление CPU/RAM/threads/network по каждому контейнеру и процессу)
- Process Control (команды и переходы в сервисы)
- Live Metrics + встроенные Grafana dashboards

Через UI-шлюз (`http://localhost:${UI_HOST_PORT}`) доступны все HTTP-сервисы стека:
`/api/grafana`, `/api/prometheus`, `/api/mlflow`, `/api/alertmanager`, `/api/minio`,
`/api/minio-console`, `/api/pushgateway`, `/api/blackbox-exporter`, `/api/cadvisor`,
`/api/node-exporter`, `/api/process-exporter`, `/api/postgres-exporter`,
`/api/torchserve`, `/api/torchserve-mgmt`, `/api/torchserve-metrics`.

В Grafana автоматически провиженятся дашборды:

- `UAV System Overview`
- `UAV Training and Inference`
- `UAV Resource Control`

Экспорт и публикация модели:

```bash
python scripts/export_torchserve.py \
  --config configs/experiments/yolos_tiny.yaml \
  --checkpoint runs/yolos_tiny_uav/best.pt \
  --model-name uav_detector \
  --export-path model-store \
  --force

bash scripts/torchserve_register_model.sh uav_detector uav_detector.mar
bash scripts/torchserve_infer.sh uav_detector /path/to/frame.jpg
```

Kubernetes:

```bash
bash scripts/k8s_apply.sh
kubectl -n uav-mlops get pods
```

Полная инструкция: [docs/mlops_cluster_ru.md](docs/mlops_cluster_ru.md)

## Документация

- [Карта документации](docs/README_ru.md)
- [Развертывание и git-автоматизация](docs/deployment_ru.md)
- [MLOps кластер](docs/mlops_cluster_ru.md)
- [Grafana Web View](docs/grafana_web_view_ru.md)
- [Split UI Control Center](docs/ui_control_center_ru.md)
- [Полный контроль ресурсов](docs/resource_control_ru.md)
- [Расширение архитектур](docs/architecture_extension.md)
- [Методология исследования](docs/thesis_methodology_ru.md)
- [Полный аналитический каркас](docs/full_analysis_framework_ru.md)
- [Шаблон результатов](docs/results_template.md)
- [Шаблон выводов](docs/conclusions_ru_template.md)
