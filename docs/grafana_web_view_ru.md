# Grafana Web View

Документ описывает, какие метрики и процессы визуализируются в Grafana и откуда берутся данные.

## Источники данных

- Prometheus datasource (`uid=prometheus`)
- Pushgateway: live-метрики train/eval из Python-пайплайна
- Run Metrics Exporter: агрегированные метрики из `runs/*/metrics.csv`
- Blackbox exporter: доступность сервисов (`mlflow`, `minio`, `torchserve`, `grafana`, `ui`)
- Postgres exporter: состояние БД метаданных MLflow
- cAdvisor + node-exporter (docker compose): контейнерные и хостовые ресурсы
- Process Exporter: потребление CPU/RAM/threads по процессам
- Alertmanager: централизованная маршрутизация алертов Prometheus

## Split UI Control Center

`http://localhost:${UI_HOST_PORT}` (`18090`) — разделенный web-интерфейс для операционного управления:

- ручное дообучение (гиперпараметры + генерация конфига)
- автонастройка (план trial-ов)
- контроль CI/CD этапов
- контроль процесса и команды запуска
- live KPI + встраивание Grafana-дешбордов
- отдельные таблицы по контейнерам и процессам (CPU/RAM/network/threads)
- доступ ко всем HTTP-сервисам стека через UI proxy (`/api/...`)

## Дашборды

## 1) UAV System Overview

Показывает состояние инфраструктуры:

- доступность endpoints (`probe_success`)
- загрузка CPU/RAM хоста
- CPU/RAM контейнеров
- состояние PostgreSQL
- количество отслеживаемых run-артефактов

## 2) UAV Training and Inference

Показывает ML-метрики:

- train loss (live + из CSV)
- `mAP50` на валидации и оценке
- latency/FPS
- прогресс по эпохам
- лучшая checkpoint-метрика

## 3) UAV Resource Control

Дешборд полного контроля ресурсов:

- host CPU/RAM/disk usage
- активные алерты и доступность сервисов
- утилизация CPU/RAM контейнеров относительно лимитов
- network throughput контейнеров
- топ процессов по CPU и RAM

## URL и доступ

- Grafana: `http://localhost:${GRAFANA_HOST_PORT}` (`13000`)
- Prometheus: `http://localhost:${PROMETHEUS_HOST_PORT}` (`19090`)
- Split UI: `http://localhost:${UI_HOST_PORT}` (`18090`)
- Alertmanager: `http://localhost:${ALERTMANAGER_HOST_PORT}` (`19093`)
- Process Exporter: `http://localhost:${PROCESS_EXPORTER_HOST_PORT}` (`19256`)

Логин/пароль Grafana:

- `GRAFANA_ADMIN_USER`
- `GRAFANA_ADMIN_PASSWORD`

## Быстрый запуск

```bash
cp .env.example .env
docker compose up -d --build
```

Для запуска обучения/инференса дополнительно включайте профили:

```bash
docker compose --profile training --profile inference up -d --build
```

После запуска откройте Grafana и папку дашбордов `UAV-MLOps`.
