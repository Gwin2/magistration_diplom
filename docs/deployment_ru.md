# Развертывание и git-автоматизация

Документ описывает базовый запуск проекта для разработки и экспериментов.
Для MLOps-кластера используйте отдельный гайд: [mlops_cluster_ru.md](mlops_cluster_ru.md).

## Требования

- Python `3.10+`
- Git
- Docker + Docker Compose (опционально)
- Kubernetes + kubectl (опционально, для кластера)

## Локальная настройка (Windows PowerShell)

```powershell
.\scripts\bootstrap.ps1
.\scripts\git_setup.ps1
```

## Локальная настройка (Linux/macOS)

```bash
bash scripts/bootstrap.sh
bash scripts/git_setup.sh
```

## Что настраивается автоматически

- создание `.venv`
- установка проекта и dev-зависимостей
- установка `pre-commit` hooks
- базовая конфигурация git

## Проверка после установки

```bash
uav-vit --help
pytest
```

## Базовый рабочий цикл

```bash
uav-vit train --config configs/experiments/yolos_tiny.yaml
uav-vit evaluate --config configs/experiments/yolos_tiny.yaml --split test
uav-vit summarize --runs-dir runs --output-dir reports
```

## Docker (локальный контейнер trainer)

```bash
docker compose build trainer
docker compose run --rm trainer uav-vit --help
```

## Grafana/Prometheus web view

```bash
cp .env.example .env
docker compose up -d --build
```

После запуска:

- Grafana: `http://localhost:${GRAFANA_HOST_PORT}` (default `13000`)
- Prometheus: `http://localhost:${PROMETHEUS_HOST_PORT}` (default `19090`)

При конфликте портов измените значения в `.env` и перезапустите стек:
`docker compose down && docker compose up -d --build`.

Подробно: [grafana_web_view_ru.md](grafana_web_view_ru.md)

## CI

В репозитории настроен GitHub Actions workflow:

- lint: `ruff check`, `ruff format --check`
- tests: `pytest`
