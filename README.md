# UAV ViT Thesis Platform

Платформа для магистерской работы по обнаружению БПЛА в сложных условиях: низкое качество кадра, плохая погода, малые размеры цели, резкие маневры и нестабильный фон.

## Что внутри

- research pipeline: `video -> COCO -> train/eval -> reports`
- базовые ViT/DETR-модели через `transformers` и реестр для собственных архитектур
- нативный `Mission Control UI` для датасетов, конфигов, экспериментов, TensorBoard, MLflow и TorchServe
- MLOps-стек: `MLflow + Postgres + MinIO + Prometheus + Grafana + Alertmanager + TorchServe`
- локальный запуск через `docker compose` и deployment в `k8s`

## Структура проекта

- `src/uav_vit` — обучение, оценка, аналитика, control plane и serving-интеграции
- `configs/experiments` — YAML-конфиги экспериментов
- `scripts` — bootstrap, export, deployment и cluster scripts
- `monitoring` — Prometheus, Grafana, Alertmanager, exporters
- `ui` — фронтенд Mission Control
- `docs` — методология, развёртывание, MLOps и шаблоны результатов
- `k8s/base` — базовые Kubernetes manifest'ы

## Быстрый старт без Docker

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

## Подготовка данных

Ожидаемые поля CSV:

- `video_name`, `frame_idx`, `x_min`, `y_min`, `x_max`, `y_max`, `class_name`
- опционально: `weather`, `quality`, `maneuver`, `split`

```bash
uav-vit convert-video \
  --video-dir data/raw/videos \
  --annotations-csv data/raw/annotations.csv \
  --output-dir data/processed/uav_coco
```

## Обучение, оценка, отчёты

```bash
uav-vit train --config configs/experiments/yolos_tiny.yaml
uav-vit evaluate --config configs/experiments/yolos_tiny.yaml --split test
uav-vit summarize --runs-dir runs --output-dir reports
```

Артефакты:

- `runs/<experiment>/best.pt`, `runs/<experiment>/metrics.csv`, `runs/<experiment>/tensorboard/`
- `reports/summary.csv`
- `reports/summary.tex`

## Mission Control UI

UI доступен по `http://localhost:${UI_HOST_PORT}` (по умолчанию `18090`) и работает как единая операторская панель.

Основные разделы:

- `Overview` — health stack, KPI, рекомендации по лучшим запускам и встроенные Grafana dashboards
- `Datasets` — загрузка архивов, регистрация готовых каталогов и скачивание dataset bundle
- `Studio` — редактирование YAML-конфигов, библиотека архитектур, шаблоны для своих моделей и запуск train/eval
- `Experiments` — список запусков, фильтры, сравнение, теги, рейтинг, заметки, jobs и TensorBoard
- `Serving` — регистрация моделей в TorchServe, inference probe и единый gateway ко всем web-сервисам
- `Resources` — генерация resource override, потребление по контейнерам и процессам

Поддерживаются темы интерфейса `Flight`, `Horizon`, `Paper`, `Signal`, переключение плотности UI и отключение анимаций.

## Compose-стек

Сначала подготовьте `.env`:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Базовый режим: UI, control plane, tracking, monitoring и storage.

```bash
docker compose up -d --build
```

Полный режим с обучением и инференсом:

```bash
docker compose --profile training --profile inference up -d --build
```

Отдельные контуры:

```bash
docker compose --profile training up -d --build
docker compose --profile inference up -d --build
```

### Сервисы по умолчанию

- UI: `http://localhost:${UI_HOST_PORT}` (`18090`)
- Control API: `http://localhost:${CONTROL_API_HOST_PORT}` (`18070`)
- MLflow: `http://localhost:${MLFLOW_HOST_PORT}` (`15000`)
- MinIO API: `http://localhost:${MINIO_API_HOST_PORT}` (`19000`)
- MinIO Console: `http://localhost:${MINIO_CONSOLE_HOST_PORT}` (`19001`)
- Prometheus: `http://localhost:${PROMETHEUS_HOST_PORT}` (`19090`)
- Grafana: `http://localhost:${GRAFANA_HOST_PORT}` (`13000`)
- Pushgateway: `http://localhost:${PUSHGATEWAY_HOST_PORT}` (`19091`)
- Alertmanager: `http://localhost:${ALERTMANAGER_HOST_PORT}` (`19093`)
- Process Exporter: `http://localhost:${PROCESS_EXPORTER_HOST_PORT}` (`19256`)
- TorchServe inference: `http://localhost:${TORCHSERVE_INFERENCE_HOST_PORT}` (`18080`)
- TorchServe management: `http://localhost:${TORCHSERVE_MANAGEMENT_HOST_PORT}` (`18081`)

Через UI-proxy доступны:

- `/api/control`
- `/api/mlflow`
- `/api/grafana`
- `/api/prometheus`
- `/api/tensorboard`
- `/api/alertmanager`
- `/api/minio`, `/api/minio-console`
- `/api/torchserve`, `/api/torchserve-mgmt`, `/api/torchserve-metrics`
- `/api/cadvisor`, `/api/node-exporter`, `/api/process-exporter`, `/api/postgres-exporter`

## Кастомные архитектуры

1. Создайте модуль, например `src/custom_models/my_detector.py`.
2. Зарегистрируйте билдер: `@register_model("my_detector")`.
3. Укажите в YAML:
   - `model.name: my_detector`
   - `model.custom_modules: ["custom_models.my_detector"]`
4. При желании редактируйте и сохраняйте шаблон прямо из `Studio` во фронтенде.

Подробнее: [docs/architecture_extension.md](docs/architecture_extension.md)

## Контроль качества и CI

Локальные проверки:

```bash
ruff check src tests
ruff format --check src tests
pytest
node --check ui/app.js
docker compose config
```

## Документация

- [Карта документации](docs/README_ru.md)
- [Развёртывание и git-автоматизация](docs/deployment_ru.md)
- [MLOps кластер](docs/mlops_cluster_ru.md)
- [Grafana Web View](docs/grafana_web_view_ru.md)
- [Mission Control UI](docs/ui_control_center_ru.md)
- [Полный контроль ресурсов](docs/resource_control_ru.md)
- [Расширение архитектур](docs/architecture_extension.md)
- [Методология исследования](docs/thesis_methodology_ru.md)
- [Полный аналитический каркас](docs/full_analysis_framework_ru.md)
- [Шаблон результатов](docs/results_template.md)
- [Шаблон выводов](docs/conclusions_ru_template.md)
