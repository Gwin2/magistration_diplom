# Grafana Web View

Документ описывает встроенные Grafana dashboards, их источники данных и маршрут доступа через UI.

## Источники данных

- `Prometheus` (`uid=prometheus`) — основной datasource
- `Pushgateway` — live-метрики train/eval
- `run-metrics-exporter` — агрегаты из `runs/*/metrics.csv`
- `blackbox-exporter` — доступность сервисов
- `postgres-exporter` — состояние MLflow metadata DB
- `cadvisor` и `node-exporter` — контейнерные и host metrics
- `process-exporter` — CPU/RAM/threads по процессам
- `Alertmanager` — активные алерты и их маршрутизация

## Доступ

Прямой доступ:

- Grafana: `http://localhost:${GRAFANA_HOST_PORT}` (`13000`)
- Prometheus: `http://localhost:${PROMETHEUS_HOST_PORT}` (`19090`)

Через Mission Control UI:

- Grafana root: `http://localhost:${UI_HOST_PORT}/api/grafana/`
- health: `http://localhost:${UI_HOST_PORT}/api/grafana/api/health`
- dashboards открываются встроенно в `Overview`

Grafana настроена на работу из subpath через:

- `GF_SERVER_ROOT_URL`
- `GF_SERVER_SERVE_FROM_SUB_PATH=true`

## Dashboards

### 1. UAV System Overview

Показывает:

- `probe_success` по сервисам
- host CPU/RAM
- базовую загрузку контейнеров
- состояние PostgreSQL и observability layers

### 2. UAV Training and Inference

Показывает:

- `train loss`
- `mAP50`, `mAP75`
- latency/FPS
- прогресс train/eval runs
- ключевые inference-метрики

### 3. UAV Resource Control

Показывает:

- host CPU/RAM/disk usage
- firing alerts
- container CPU/RAM/network
- top processes by CPU/RAM/threads

## Встраивание в UI

Во вкладке `Overview` доступен переключатель встроенных dashboards:

- `System`
- `Training`
- `Resources`

UI автоматически добавляет параметр темы (`light` или `dark`) к iframe в зависимости от выбранной темы интерфейса.

## Быстрый запуск

```bash
cp .env.example .env
docker compose up -d --build
```

Для train/inference сервисов:

```bash
docker compose --profile training --profile inference up -d --build
```
