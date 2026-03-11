PYTHON ?= python
CONFIG ?= configs/experiments/yolos_tiny.yaml

.PHONY: setup lint test train eval summarize compose-up compose-down k8s-up k8s-down

setup:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest

train:
	uav-vit train --config $(CONFIG)

eval:
	uav-vit evaluate --config $(CONFIG)

summarize:
	uav-vit summarize --runs-dir runs --output-dir reports

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

k8s-up:
	bash scripts/k8s_apply.sh

k8s-down:
	bash scripts/k8s_cleanup.sh
