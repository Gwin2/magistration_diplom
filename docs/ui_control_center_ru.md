# Split UI Control Center

Документ описывает web-интерфейс `uav-ui`, который объединяет операционные сценарии MLOps без обязательного старта обучения и инференса.

## Назначение

UI построен как разделенная панель (split view) для:

- дообучения и настройки гиперпараметров
- контроля CI/CD и этапов доставки
- мониторинга состояния сервисов
- просмотра KPI и встроенных Grafana дашбордов
- ручной и автоматической настройки параметров

## Где доступен

- URL: `http://localhost:${UI_HOST_PORT}` (default `18090`)
- Сервис: `uav-ui` (Nginx + статический frontend)

## Что внутри

1. Fine-Tuning Studio:
   - выбор архитектуры
   - настройка `learning_rate`, `epochs`, `batch_size`, `score_threshold`
   - генерация конфигурации ручного дообучения

2. Auto Tuning:
   - стратегии (`bayesian`, `grid`, `population`)
   - objective (`mAP50`, `latency`, `fps`)
   - генерация плана trial-ов

3. CI/CD Pipeline Control:
   - визуальный прогон стадий pipeline
   - индикатор текущего этапа

4. Process Control:
   - генерация команд train/eval/deploy
   - быстрые переходы во все HTTP-сервисы стека через `/api/...`

5. Live Metrics:
   - KPI карточки по `mAP50`, `mAP75`, `latency`, `fps`, `health`
   - KPI по ресурсам `host_cpu`, `host_mem`, `host_disk`, `firing_alerts`
   - данные читаются через прокси `/api/prometheus/...`
   - fallback-режим визуализации при отсутствии метрик

6. Container and Process Consumption:
   - таблица контейнеров: CPU%, memory MB, network RX KB/s
   - таблица процессов: CPU%, memory MB, threads
   - источник данных: Prometheus (`cadvisor`, `node-exporter`, `process-exporter`)

7. Embedded Dashboards:
   - `UAV System Overview`
   - `UAV Training and Inference`
   - `UAV Resource Control`

## Полный доступ к сервисам через UI

UI содержит reverse proxy и дает единый вход в сервисы:

- `/api/grafana`
- `/api/prometheus`
- `/api/mlflow`
- `/api/alertmanager`
- `/api/minio`
- `/api/minio-console`
- `/api/pushgateway`
- `/api/blackbox-exporter`
- `/api/cadvisor`
- `/api/node-exporter`
- `/api/process-exporter`
- `/api/postgres-exporter`
- `/api/torchserve`
- `/api/torchserve-mgmt`
- `/api/torchserve-metrics`

## Техническая схема

- `ui/index.html`, `ui/styles.css`, `ui/app.js` — frontend
- `docker/ui/nginx.conf` — reverse proxy до всех web-сервисов (`prometheus`, `mlflow`, `grafana`, `alertmanager`, `minio`, `exporters`, `torchserve`)
- `docker-compose.yml`:
  - `ui` работает в базовом режиме
  - `trainer`, `run-metrics-exporter`, `torchserve` вынесены в `profiles`

## Запуск

Базовый режим:

```bash
docker compose up -d --build
```

Полный режим с обучением/инференсом:

```bash
docker compose --profile training --profile inference up -d --build
```
