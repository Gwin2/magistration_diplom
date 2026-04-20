from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from uav_vit.logging_config import get_logger
from uav_vit.utils import optional_import

logger = get_logger(__name__)


def _flatten_dict(data: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in data.items():
        joined_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, dict):
            output.update(_flatten_dict(value, parent_key=joined_key))
            continue
        if isinstance(value, (str, int, float, bool)):
            output[joined_key] = value
        elif value is None:
            output[joined_key] = "null"
        else:
            output[joined_key] = str(value)
    return output


def _is_enabled(config: dict[str, Any]) -> bool:
    cfg = config.get("mlflow", {})
    if bool(cfg.get("enabled", False)):
        return True
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _import_mlflow() -> Any | None:
    """Import mlflow optionally.

    Deprecated: Use uav_vit.utils.optional_import instead.
    """
    return optional_import("mlflow")


@contextmanager
def mlflow_run(config: dict[str, Any], phase: str) -> Iterator[Any | None]:
    if not _is_enabled(config):
        yield None
        return

    mlflow = _import_mlflow()
    if mlflow is None:
        logger.warning(
            "[mlflow] MLflow is enabled but package is not installed. Logging is skipped."
        )
        yield None
        return

    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = (
        mlflow_cfg.get("experiment_name")
        or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or config["experiment"]["name"]
    )
    mlflow.set_experiment(experiment_name)

    run_name = mlflow_cfg.get("run_name") or f"{config['experiment']['name']}-{phase}"
    mlflow.start_run(run_name=run_name)
    try:
        mlflow.set_tags(
            {
                "phase": phase,
                "model.name": str(config["model"].get("name", "unknown")),
                "dataset.train_annotations": str(config["paths"].get("train_annotations", "")),
                "dataset.val_annotations": str(config["paths"].get("val_annotations", "")),
                "dataset.test_annotations": str(config["paths"].get("test_annotations", "")),
            }
        )
        params = _flatten_dict(config)
        try:
            mlflow.log_params(params)
        except Exception as exc:
            logger.warning(f"[mlflow] Failed to log params: {exc}")
        yield mlflow
    except Exception:
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.end_run(status="FINISHED")


def log_metrics(mlflow: Any | None, metrics: dict[str, float], step: int | None = None) -> None:
    if mlflow is None:
        return
    prepared = {
        str(key): float(value) for key, value in metrics.items() if _is_float_compatible(value)
    }
    if step is None:
        mlflow.log_metrics(prepared)
    else:
        mlflow.log_metrics(prepared, step=step)


def log_artifact_if_exists(
    mlflow: Any | None, path: str | Path, artifact_path: str | None = None
) -> None:
    if mlflow is None:
        return
    local_path = Path(path)
    if not local_path.exists():
        return
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path)


def _is_float_compatible(value: Any) -> bool:
    return isinstance(value, (int, float))
