from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL_KEYS = {"experiment", "paths", "model", "train", "eval", "data"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must be a mapping.")

    missing = REQUIRED_TOP_LEVEL_KEYS - set(config.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Config at {config_path} is missing sections: {missing_str}")

    normalized = deepcopy(config)
    normalized["model"] = _normalize_model_config(normalized["model"])
    return normalized


def _normalize_model_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    if "id2label" in model_cfg:
        model_cfg["id2label"] = {int(key): value for key, value in model_cfg["id2label"].items()}
    if "label2id" in model_cfg:
        model_cfg["label2id"] = {str(key): int(value) for key, value in model_cfg["label2id"].items()}
    if "label2id" not in model_cfg and "id2label" in model_cfg:
        model_cfg["label2id"] = {value: key for key, value in model_cfg["id2label"].items()}
    if "id2label" not in model_cfg and "label2id" in model_cfg:
        model_cfg["id2label"] = {value: key for key, value in model_cfg["label2id"].items()}
    model_cfg.setdefault("custom_modules", [])
    return model_cfg
