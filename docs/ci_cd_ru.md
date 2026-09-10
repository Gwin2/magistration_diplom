# CI/CD и диагностика

CI запускается для pull request, push в `main`/`master` и вручную. CD запускается для тега `vX.Y.Z` или вручную.

## Что защищено

| Проверка | Что может сломаться | Как чинить |
| --- | --- | --- |
| `ruff check` | Python-ошибка, плохой импорт, небезопасный шаблон | `python -m ruff check src tests --fix`, затем повторить CI |
| `pytest` + coverage/JUnit | Регрессия поведения API, данных или мониторинга | Открыть `python-test-results`, исправить первый failing test |
| `node --check` | Синтаксическая ошибка UI | `node --check ui/app.js`, исправить строку из лога |
| YAML validator | Повреждённый workflow, конфиг мониторинга или manifest | Исправить файл и повторить `python -B scripts/ci/validate_yaml.py` |
| `docker compose config` | Неверная переменная, сервис или volume | Проверить `.env.example` и `docker-compose.yml` |
| Kustomize + kubeconform | Kubernetes-ресурс не применится | Сначала исправить ошибку render/schema, потом запускать deploy |
| Trivy | Уязвимость в зависимостях проекта | Обновить зависимость или base image; unfixed-уязвимость не блокирует gate |
| CodeQL | Небезопасный поток данных в Python/JavaScript | Открыть Security → Code scanning и исправить source/sink |
| advisory `ruff format`, `mypy`, `pip-audit` | Накопленный style/type/dependency debt | Скачать `advisory-diagnostics`, чинить по одному отчёту |

Обязательные gates: Ruff lint, тесты, UI/config validation, Trivy. Advisory-отчёты не блокируют merge, пока не закрыт существующий baseline-долг в `control`, `data`, `ui.builder`, `dataset` и `monitoring`.

## CD

Тег `v1.2.3` публикует в GitHub Container Registry три образа:

```text
ghcr.io/<owner>/uav-vit-trainer:v1.2.3
ghcr.io/<owner>/uav-vit-mlflow:v1.2.3
ghcr.io/<owner>/uav-vit-torchserve:v1.2.3
```

Каждый образ получает SBOM, provenance и SHA-тег. Ручной запуск принимает `image_tag`.

Production Kubernetes deploy пока не автоматизируется: `k8s/base` содержит `your-registry/*` и development secrets. После появления production overlay и GitHub Environment `production` deploy добавляется отдельным approval-gate.

## Локальный preflight

```bash
python -m ruff check src tests
pytest --cov=src --cov-report=term-missing
node --check ui/app.js
python -B scripts/ci/validate_yaml.py
docker compose --env-file .env.example config --quiet
kubectl kustomize k8s/base > /tmp/k8s.yaml
```

Первый failed gate — причина. Последующие ошибки часто каскадируют и отдельно не требуют диагностики.
