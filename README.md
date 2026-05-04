# UAV ViT Thesis Platform

Платформа для магистерской работы по обнаружению БПЛА в сложных условиях: низкое качество кадра, плохая погода, малые размеры цели, резкие маневры и нестабильный фон.

## Что внутри

- **Research pipeline**: `video -> COCO -> train/eval -> reports`
- **ViT/DETR модели**: базовые архитектуры через `transformers` + реестр для собственных моделей
- **Mission Control UI**: единая панель управления с 8-шаговым мастером обучения
- **MLOps стек**: `MLflow + Postgres + MinIO + Prometheus + Grafana + Alertmanager + TorchServe`
- **Deployment**: локальный запуск через `docker compose` и production-ready `k8s` манифесты

## Структура проекта

```
├── src/uav_vit/           # Обучение, оценка, аналитика, control plane
├── configs/experiments/   # YAML-конфиги экспериментов
├── scripts/              # Bootstrap, export, deployment скрипты
├── monitoring/           # Prometheus, Grafana, Alertmanager, exporters
├── ui/                   # Mission Control фронтенд (8 шагов)
├── docs/                 # Документация и шаблоны
├── k8s/base/            # Kubernetes манифесты
└── tests/               # Тесты и интеграционные проверки
```

## Быстрый старт

### 1. Установка зависимостей

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install
```

### 2. Подготовка данных

Ожидаемые поля CSV:
- `video_name`, `frame_idx`, `x_min`, `y_min`, `x_max`, `y_max`, `class_name`
- опционально: `weather`, `quality`, `maneuver`, `split`

```bash
uav-vit convert-video \
  --video-dir data/raw/videos \
  --annotations-csv data/raw/annotations.csv \
  --output-dir data/processed/uav_coco
```

### 3. Обучение и оценка

```bash
# Обучение модели
uav-vit train --config configs/experiments/yolos_tiny.yaml

# Оценка на тестовой выборке
uav-vit evaluate --config configs/experiments/yolos_tiny.yaml --split test

# Генерация отчётов
uav-vit summarize --runs-dir runs --output-dir reports
```

**Артефакты:**
- `runs/<experiment>/best.pt` — чекпоинт модели
- `runs/<experiment>/metrics.csv` — метрики
- `runs/<experiment>/tensorboard/` — логи для TensorBoard
- `reports/summary.csv` и `reports/summary.tex` — сводные отчёты

## Mission Control UI

UI доступен по `http://localhost:${UI_HOST_PORT}` (по умолчанию `18090`) и работает как единая операторская панель с **8-шаговым мастером обучения**.

### 8 шагов обучения нейронной сети

1. **Data** — загрузка датасетов, подготовка COCO-формата
2. **Architecture** — конструктор модели: выбор базовой архитектуры или сборка кастомной
3. **Configuration** — параметры обучения: optimizer, scheduler, аугментации
4. **Training** — запуск эксперимента, мониторинг в реальном времени
5. **Evaluation** — метрики на валидации/тесте, визуализация предсказаний
6. **Deploy** — экспорт модели, регистрация в TorchServe
7. **Metrics** — live-дашборды Prometheus + Grafana
8. **Resources** — потребление GPU/CPU/RAM по контейнерам и процессам

### Навигация

- **Последовательный режим**: автоматический переход между шагами после выполнения действий
- **Ручное управление**: кнопки Back/Next для навигации
- **Прямой доступ**: клики по шагам в боковой панели
- **Сохранение прогресса**: состояние сохраняется в localStorage браузера

### Темы и настройки

Поддерживаются темы интерфейса: `Flight`, `Horizon`, `Paper`, `Signal`. Доступно переключение плотности UI и отключение анимаций.

**Подробнее**: [docs/ui_control_center_ru.md](docs/ui_control_center_ru.md)

## Compose-стек

### 1. Подготовка окружения

Сначала подготовьте `.env` файл:

**Linux/macOS:**
```bash
cp .env.example .env
```

**Windows PowerShell:**
```powershell
Copy-Item .env.example .env
```

### 2. Запуск сервисов

**Базовый режим** (UI, control plane, tracking, monitoring, storage):
```bash
docker compose up -d --build
```

**Полный режим** (с обучением и инференсом):
```bash
docker compose --profile training --profile inference up -d --build
```

**Отдельные контуры:**
```bash
# Только обучение
docker compose --profile training up -d --build

# Только инференс
docker compose --profile inference up -d --build
```

**Остановка и перезапуск:**
```bash
docker compose stop
docker compose start
```

### 3. Особенности архитектуры

Контейнеры создаются **без фиксированных `container_name`**, что обеспечивает:
- Безопасный запуск из CLI и IDE без конфликтов имён
- Корректную работу `docker compose stop` → `docker compose start`
- Project-scoped имена вида `magistration_diplom-mlflow-1`, `magistration_diplom-ui-1`

### 4. Сервисы по умолчанию

| Сервис | URL | Порт |
|--------|-----|------|
| UI | `http://localhost:${UI_HOST_PORT}` | 18090 |
| Control API | `http://localhost:${CONTROL_API_HOST_PORT}` | 18070 |
| MLflow | `http://localhost:${MLFLOW_HOST_PORT}` | 15000 |
| MinIO API | `http://localhost:${MINIO_API_HOST_PORT}` | 19000 |
| MinIO Console | `http://localhost:${MINIO_CONSOLE_HOST_PORT}` | 19001 |
| Prometheus | `http://localhost:${PROMETHEUS_HOST_PORT}` | 19090 |
| Grafana | `http://localhost:${GRAFANA_HOST_PORT}` | 13000 |
| Pushgateway | `http://localhost:${PUSHGATEWAY_HOST_PORT}` | 19091 |
| Alertmanager | `http://localhost:${ALERTMANAGER_HOST_PORT}` | 19093 |
| Process Exporter | `http://localhost:${PROCESS_EXPORTER_HOST_PORT}` | 19256 |
| TorchServe Inference | `http://localhost:${TORCHSERVE_INFERENCE_HOST_PORT}` | 18080 |
| TorchServe Management | `http://localhost:${TORCHSERVE_MANAGEMENT_HOST_PORT}` | 18081 |

### 5. UI Proxy endpoints

Через UI-proxy доступны все сервисы:
- `/api/control` — Control Plane API
- `/api/mlflow` — MLflow tracking
- `/api/grafana` — Grafana dashboards
- `/api/prometheus` — Prometheus metrics
- `/api/tensorboard` — TensorBoard
- `/api/alertmanager` — Alertmanager
- `/api/minio`, `/api/minio-console` — MinIO
- `/api/torchserve`, `/api/torchserve-mgmt`, `/api/torchserve-metrics` — TorchServe
- `/api/cadvisor`, `/api/node-exporter`, `/api/process-exporter`, `/api/postgres-exporter` — Exporters

### 6. Troubleshooting

**Ошибка**: `Conflict. The container name ... is already in use`

**Решение:**
1. Обновите состояние Compose-приложения в IDE (`Reload` / `Refresh`)
2. Убедитесь, что стек не запущен из другого окна/каталога
3. Выполните очистку:
   ```bash
   docker compose down --remove-orphans
   docker compose up -d --build
   ```

**Быстрый рестарт** (если менялись только сервисы):
```bash
docker compose stop
docker compose start
```

**Диагностика:**
```bash
docker compose ps -a
docker ps -a
```

## Кастомные архитектуры

### Быстрый старт

1. **Создайте модуль** модели, например `src/custom_models/my_detector.py`
2. **Зарегистрируйте билдер** с декоратором: `@register_model("my_detector")`
3. **Укажите в YAML-конфиге**:
   ```yaml
   model:
     name: my_detector
     custom_modules: ["custom_models.my_detector"]
   ```
4. **Редактируйте через UI**: сохраняйте шаблон прямо из `Studio` во фронтенде
5. **Используйте конструктор**: получите рекомендации по `goal` и dataset tags, preview автоматически синхронизируется в code/YAML editors

**Подробнее**: [docs/architecture_extension.md](docs/architecture_extension.md)

## Контроль качества и CI

### Локальные проверки

```bash
# Python linting & formatting
ruff check src tests
ruff format --check src tests

# Тесты
pytest

# JavaScript валидация
node --check ui/app.js

# Docker Compose валидация
docker compose config
```

### Pre-commit хуки

Автоматически применяются при коммите:
- Форматирование кода (ruff-format)
- Линтинг (ruff)
- Проверка типов (mypy)

## Документация

### Основная документация

| Документ | Описание |
|----------|----------|
| [Карта документации](docs/README_ru.md) | Навигация по всей документации |
| [Развёртывание и git-автоматизация](docs/deployment_ru.md) | Deployment скрипты, CI/CD |
| [MLOps кластер](docs/mlops_cluster_ru.md) | Архитектура MLOps платформы |
| [Grafana Web View](docs/grafana_web_view_ru.md) | Дашборды и мониторинг |
| [Mission Control UI](docs/ui_control_center_ru.md) | Руководство по UI |
| [Полный контроль ресурсов](docs/resource_control_ru.md) | Управление GPU/CPU/RAM |
| [Расширение архитектур](docs/architecture_extension.md) | Создание кастомных моделей |

### Для диссертации

| Документ | Назначение |
|----------|------------|
| [Методология исследования](docs/thesis_methodology_ru.md) | Научная методология |
| [Полный аналитический каркас](docs/full_analysis_framework_ru.md) | Framework анализа результатов |
| [Шаблон результатов](docs/results_template.md) | Оформление экспериментов |
| [Шаблон выводов](docs/conclusions_ru_template.md) | Формулировка выводов |

---

**Лицензия**: MIT  
**Контакты**: [ваш email]  
**Статус**: Активная разработка
