from __future__ import annotations

import csv
import importlib
import json
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uav_vit.data import CocoDetectionDataset, collate_fn
from uav_vit.engine.evaluator import benchmark_latency, evaluate_model
from uav_vit.integrations import (
    close_tensorboard_writer,
    log_artifact_if_exists,
    log_metrics,
    log_tensorboard_metrics,
    mlflow_run,
    tensorboard_writer,
)
from uav_vit.logging_config import get_logger
from uav_vit.models import build_model
from uav_vit.monitoring import PrometheusPusher, build_push_config
from uav_vit.utils.seed import set_seed

logger = get_logger(__name__)


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _to_device_labels(labels: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    moved: list[dict[str, Any]] = []
    for item in labels:
        moved_item: dict[str, Any] = {}
        for key, value in item.items():
            moved_item[key] = value.to(device) if torch.is_tensor(value) else value
        moved.append(moved_item)
    return moved


def _maybe_import_custom_modules(config: dict[str, Any]) -> None:
    for module_name in config["model"].get("custom_modules", []):
        importlib.import_module(module_name)


def _create_dataloader(
    dataset: CocoDetectionDataset,
    image_processor: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(collate_fn, image_processor=image_processor),
    )


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device
) -> dict[str, Any]:
    # SECURITY FIX: Use weights_only=True to prevent arbitrary code execution
    # This restricts loading to only tensor data, not arbitrary Python objects
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = payload.get("model_state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.info(f"[checkpoint] Missing keys: {len(missing)}")
    if unexpected:
        logger.info(f"[checkpoint] Unexpected keys: {len(unexpected)}")
    return payload


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["experiment"]["seed"]))
    _maybe_import_custom_modules(config)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = build_model(config)
    model = model_bundle.model
    image_processor = model_bundle.image_processor

    train_dataset = CocoDetectionDataset(
        images_dir=config["paths"]["train_images"],
        annotations_path=config["paths"]["train_annotations"],
        image_processor=image_processor,
    )
    val_dataset = CocoDetectionDataset(
        images_dir=config["paths"]["val_images"],
        annotations_path=config["paths"]["val_annotations"],
        image_processor=image_processor,
    )

    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"]["num_workers"])

    train_loader = _create_dataloader(
        train_dataset,
        image_processor=image_processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = _create_dataloader(
        val_dataset,
        image_processor=image_processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    device = _select_device(str(config["train"]["device"]))
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    use_amp = bool(config["train"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = int(config["train"]["epochs"])
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 0.0))
    checkpoint_metric = str(config["train"].get("checkpoint_metric", "map"))
    checkpoint_mode = str(config["train"].get("checkpoint_mode", "max"))
    maximize = checkpoint_mode == "max"
    best_metric = float("-inf") if maximize else float("inf")
    experiment_name = str(config["experiment"]["name"])
    model_name = str(config["model"]["name"])
    monitoring_pusher = PrometheusPusher(
        push_config=build_push_config(config, phase="train"),
        experiment=experiment_name,
        model=model_name,
    )

    metrics_file = output_dir / "metrics.csv"
    headers = [
        "epoch",
        "train_loss",
        "map",
        "map_50",
        "map_75",
        "mar_100",
        "latency_ms",
        "fps",
    ]
    with metrics_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

    summary: dict[str, Any] = {}
    tb_writer = tensorboard_writer(config, phase="train")
    try:
        with mlflow_run(config, phase="train") as mlflow:
            for epoch in range(1, epochs + 1):
                model.train()
                running_loss = 0.0
                num_steps = 0
                progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
                for step, batch in enumerate(progress, start=1):
                    pixel_values = batch["pixel_values"].to(device)
                    pixel_mask = batch.get("pixel_mask")
                    if pixel_mask is not None:
                        pixel_mask = pixel_mask.to(device)
                    labels = _to_device_labels(batch["labels"], device)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(
                            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
                        )
                        loss = outputs.loss

                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += float(loss.item())
                    num_steps += 1
                    progress.set_postfix({"loss": f"{loss.item():.4f}"})

                    if step % int(config["train"].get("log_interval", 20)) == 0:
                        logger.info(f"[train] epoch={epoch} step={step} loss={loss.item():.4f}")

                train_loss = running_loss / max(num_steps, 1)
                val_metrics = evaluate_model(
                    model=model,
                    image_processor=image_processor,
                    dataloader=val_loader,
                    device=device,
                    score_threshold=float(config["eval"]["score_threshold"]),
                )
                latency_metrics = benchmark_latency(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    warmup_iters=int(config["eval"]["latency_warmup_iters"]),
                    latency_iters=int(config["eval"]["latency_iters"]),
                )

                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "map": val_metrics.get("map", 0.0),
                    "map_50": val_metrics.get("map_50", 0.0),
                    "map_75": val_metrics.get("map_75", 0.0),
                    "mar_100": val_metrics.get("mar_100", 0.0),
                    "latency_ms": latency_metrics["latency_ms"],
                    "fps": latency_metrics["fps"],
                }

                with metrics_file.open("a", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    writer.writerow(row)
                monitoring_pusher.push_train_epoch(
                    epoch=epoch,
                    metrics={k: float(v) for k, v in row.items() if k != "epoch"},
                )
                metric_payload = {
                    "train_loss": float(train_loss),
                    "val_map": float(row["map"]),
                    "val_map_50": float(row["map_50"]),
                    "val_map_75": float(row["map_75"]),
                    "val_mar_100": float(row["mar_100"]),
                    "latency_ms": float(row["latency_ms"]),
                    "fps": float(row["fps"]),
                }
                log_metrics(mlflow, metric_payload, step=epoch)
                log_tensorboard_metrics(tb_writer, metric_payload, step=epoch)

                last_checkpoint = output_dir / "last.pt"
                payload = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": row,
                    "config": config,
                }
                torch.save(payload, last_checkpoint)

                current_metric = float(row.get(checkpoint_metric, 0.0))
                is_better = (
                    current_metric > best_metric if maximize else current_metric < best_metric
                )
                if is_better:
                    best_metric = current_metric
                    torch.save(payload, output_dir / "best.pt")

                logger.info(
                    f"[val] epoch={epoch} map={row['map']:.4f} map50={row['map_50']:.4f} "
                    f"latency_ms={row['latency_ms']:.2f} fps={row['fps']:.2f}"
                )
            summary = {
                "best_metric_name": checkpoint_metric,
                "best_metric_value": best_metric,
                "output_dir": str(output_dir),
            }
            with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)

            log_metrics(
                mlflow,
                {
                    "best_metric_value": float(best_metric),
                },
            )
            log_tensorboard_metrics(
                tb_writer, {"best_metric_value": float(best_metric)}, step=epochs
            )
            monitoring_pusher.push_train_summary(
                best_metric_name=checkpoint_metric,
                best_metric_value=float(best_metric),
            )
            log_artifact_if_exists(mlflow, metrics_file, artifact_path="training")
            log_artifact_if_exists(mlflow, output_dir / "summary.json", artifact_path="training")
            if bool(config.get("mlflow", {}).get("log_checkpoints", True)):
                log_artifact_if_exists(mlflow, output_dir / "best.pt", artifact_path="checkpoints")
                log_artifact_if_exists(mlflow, output_dir / "last.pt", artifact_path="checkpoints")
    finally:
        close_tensorboard_writer(tb_writer)
    return summary
