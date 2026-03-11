from __future__ import annotations

import importlib
import json
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from uav_vit.data import CocoDetectionDataset, collate_fn
from uav_vit.engine.evaluator import benchmark_latency, evaluate_model
from uav_vit.engine.trainer import _select_device, load_checkpoint
from uav_vit.integrations import log_artifact_if_exists, log_metrics, mlflow_run
from uav_vit.monitoring import PrometheusPusher, build_push_config
from uav_vit.models import build_model


def _maybe_import_custom_modules(config: dict[str, Any]) -> None:
    for module_name in config["model"].get("custom_modules", []):
        importlib.import_module(module_name)


def evaluate_from_config(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    split: str = "test",
) -> dict[str, float]:
    _maybe_import_custom_modules(config)
    bundle = build_model(config)
    model = bundle.model
    image_processor = bundle.image_processor

    split_images = config["paths"][f"{split}_images"]
    split_annotations = config["paths"][f"{split}_annotations"]
    dataset = CocoDetectionDataset(
        images_dir=split_images,
        annotations_path=split_annotations,
        image_processor=image_processor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(collate_fn, image_processor=image_processor),
    )

    device = _select_device(str(config["train"]["device"]))
    model.to(device)

    resolved_checkpoint = checkpoint_path
    if resolved_checkpoint is None:
        best_path = Path(config["paths"]["output_dir"]) / "best.pt"
        resolved_checkpoint = best_path if best_path.exists() else None
    if resolved_checkpoint is not None:
        load_checkpoint(model, resolved_checkpoint, device)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    monitoring_pusher = PrometheusPusher(
        push_config=build_push_config(config, phase=f"evaluate-{split}"),
        experiment=str(config["experiment"]["name"]),
        model=str(config["model"]["name"]),
    )
    with mlflow_run(config, phase=f"evaluate-{split}") as mlflow:
        metrics = evaluate_model(
            model=model,
            image_processor=image_processor,
            dataloader=dataloader,
            device=device,
            score_threshold=float(config["eval"]["score_threshold"]),
        )
        latency = benchmark_latency(
            model=model,
            dataloader=dataloader,
            device=device,
            warmup_iters=int(config["eval"]["latency_warmup_iters"]),
            latency_iters=int(config["eval"]["latency_iters"]),
        )
        output = {**metrics, **latency}

        out_file = output_dir / f"{split}_metrics.json"
        with out_file.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)
        monitoring_pusher.push_evaluation(split=split, metrics={k: float(v) for k, v in output.items()})
        log_metrics(mlflow, {f"{split}_{k}": float(v) for k, v in output.items()})
        log_artifact_if_exists(mlflow, out_file, artifact_path="evaluation")
    return output
