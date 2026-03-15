from __future__ import annotations

from pathlib import Path
from typing import Any


def tensorboard_writer(config: dict[str, Any], phase: str) -> Any | None:
    if not bool(config.get("tensorboard", {}).get("enabled", True)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    log_dir = Path(config["paths"]["output_dir"]) / "tensorboard" / phase
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def log_tensorboard_metrics(writer: Any | None, metrics: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(key, float(value), global_step=step)


def close_tensorboard_writer(writer: Any | None) -> None:
    if writer is None:
        return
    writer.flush()
    writer.close()
