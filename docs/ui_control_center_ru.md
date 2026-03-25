# Mission Control UI

Документ описывает нативный web-интерфейс `uav-ui`, который объединяет датасеты, конфиги, архитектуры, эксперименты, TensorBoard, MLflow, Grafana и TorchServe в одном операторском окне.

## Назначение

UI нужен для того, чтобы:

- загружать и регистрировать датасеты без ручного редактирования файловой структуры
- редактировать YAML-конфиги и шаблоны моделей прямо в браузере
- запускать train/eval через `control-api` и отслеживать lifecycle jobs
- просматривать TensorBoard, MLflow, рекомендации системы и результаты сравнения запусков
- добавлять теги, рейтинг и заметки к экспериментам
- регистрировать модели в TorchServe и выполнять инференс через интерфейс
- контролировать нагрузку по контейнерам и процессам и генерировать `docker-compose.override.yml`

## Где доступен

- URL: `http://localhost:${UI_HOST_PORT}` (по умолчанию `18090`)
- сервис: `uav-ui` (Nginx + static frontend + reverse proxy)

## Разделы интерфейса

### 1. Overview

- status-панель по `control-api`, `MLflow`, `Prometheus`, `Grafana`, `TensorBoard`, `Pushgateway`, `TorchServe`, `Alertmanager`
- KPI-карточки по `mAP50`, `mAP75`, `latency`, `FPS`, `service health`, `host CPU`, `host memory`, `firing alerts`
- рекомендации по лучшим экспериментам
- встроенные Grafana dashboards: `System`, `Training`, `Resources`

### 2. Datasets

- upload архивов (`zip`, `tar`, `tgz`, `gz`)
- регистрация существующих каталогов без копирования
- поиск по имени, пути и тегам
- скачивание dataset bundle через UI

### 3. Studio

- библиотека конфигов
- YAML editor
- библиотека архитектур
- конструктор архитектур: выбор base detector, layer catalog, builder для classifier/bbox heads
- автоподстановка параметров слоёв и рекомендации по dataset profile/goal
- editor для source code и шаблонов конфигурации
- запуск train/eval и сохранение текущего YAML в configs

### 4. Experiments

- список запусков из локального workspace и MLflow
- фильтрация по имени, модели, статусу и `min mAP50`
- compare table
- редактирование metadata: `tags`, `rating`, `note`
- jobs panel и console logs
- встроенный TensorBoard

### 5. Serving

- регистрация и удаление моделей в TorchServe
- inference probe для изображений
- быстрые ссылки ко всем сервисам через UI-proxy

### 6. Resources

- выбор resource profile (`balanced`, `training`, `inference`, `eco`)
- генерация `docker-compose.override.yml`
- мониторинг container/process consumption

## Reverse proxy через UI

Через единый UI host доступны:

- `/api/control`
- `/api/mlflow`
- `/api/grafana`
- `/api/prometheus`
- `/api/tensorboard`
- `/api/alertmanager`
- `/api/pushgateway`
- `/api/minio`, `/api/minio-console`
- `/api/torchserve`, `/api/torchserve-mgmt`, `/api/torchserve-metrics`
- `/api/cadvisor`, `/api/node-exporter`, `/api/process-exporter`, `/api/postgres-exporter`

## Визуальная система

- темы: `Flight`, `Horizon`, `Paper`, `Signal`
- плотность: `Compact`, `Comfort`
- режим анимации: `Motion on/off`
- настройки сохраняются в `localStorage`

## Техническая схема

- `ui/index.html`, `ui/styles.css`, `ui/app.js` — frontend
- `docker/ui/nginx.conf` — reverse proxy до web-сервисов стека
- `src/uav_vit/control/app.py` — FastAPI control layer
- `src/uav_vit/control/workspace.py` — управление workspace, jobs, configs, datasets
- `src/uav_vit/control/mlops.py` — bridges к MLflow, TensorBoard и TorchServe

## Запуск

Базовый режим:

```bash
docker compose up -d --build
```

Полный режим с train/inference:

```bash
docker compose --profile training --profile inference up -d --build
```
