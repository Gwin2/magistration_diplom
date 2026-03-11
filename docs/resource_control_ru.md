# Полный контроль ресурсов

Документ описывает контур управления ресурсами в `docker-compose` и web-интерфейсе.

## Что включено

1. Resource limits на сервисах:
   - `cpus`
   - `mem_limit`
   - `mem_reservation`
   - `pids_limit`
   - ротация логов (`max-size`, `max-file`)

2. Monitoring:
   - `cAdvisor` — контейнерные метрики
   - `node-exporter` — host-метрики
   - `process-exporter` — метрики процессов
   - `postgres-exporter`, `blackbox-exporter`

3. Alerting:
   - Prometheus rules (`monitoring/prometheus/alerts.yml`)
   - Alertmanager (`http://localhost:${ALERTMANAGER_HOST_PORT}`, default `19093`)

4. UI Control:
   - `Resource Governance` (генерация resource override)
   - `Container and Process Consumption` (детализация по контейнерам/процессам)
   - доступ ко всем web-сервисам через UI proxy (`/api/...`)

## Основные алерты

- `HostHighCPU`
- `HostHighMemory`
- `HostLowDiskSpace`
- `ContainerCPUThrottled`
- `ContainerMemoryNearLimit`
- `ServiceEndpointDown`
- `PrometheusTargetDown`

## Быстрый запуск

```bash
docker compose up -d --build
```

Полный контур с train/inference:

```bash
docker compose --profile training --profile inference up -d --build
```

Все внешние порты задаются в `.env` (см. `.env.example`), поэтому конфликты с уже занятыми портами решаются сменой значений переменных.
