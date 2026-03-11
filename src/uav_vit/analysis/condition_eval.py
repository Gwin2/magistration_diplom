from __future__ import annotations

import importlib
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from uav_vit.data import CocoDetectionDataset, collate_fn
from uav_vit.engine.evaluator import evaluate_model
from uav_vit.engine.trainer import _select_device, load_checkpoint
from uav_vit.models import build_model


def _maybe_import_custom_modules(config: dict[str, Any]) -> None:
    for module_name in config["model"].get("custom_modules", []):
        importlib.import_module(module_name)


def evaluate_by_condition(
    config: dict[str, Any],
    metadata_csv: str | Path,
    condition_column: str,
    split: str = "test",
    checkpoint_path: str | Path | None = None,
) -> Path:
    _maybe_import_custom_modules(config)
    bundle = build_model(config)
    model = bundle.model
    image_processor = bundle.image_processor

    dataset = CocoDetectionDataset(
        images_dir=config["paths"][f"{split}_images"],
        annotations_path=config["paths"][f"{split}_annotations"],
        image_processor=image_processor,
    )

    metadata = pd.read_csv(metadata_csv)
    if condition_column not in metadata.columns:
        raise ValueError(f"Column '{condition_column}' was not found in {metadata_csv}")
    metadata = metadata[metadata["split"] == split].copy()
    if metadata.empty:
        raise ValueError(f"No metadata rows found for split='{split}' in {metadata_csv}")

    file_name_to_indices: dict[str, list[int]] = {}
    for idx, image_id in enumerate(dataset.image_ids):
        image_info = dataset.coco.loadImgs(image_id)[0]
        file_name = image_info["file_name"]
        file_name_to_indices.setdefault(file_name, []).append(idx)

    device = _select_device(str(config["train"]["device"]))
    model.to(device)
    if checkpoint_path is None:
        best_path = Path(config["paths"]["output_dir"]) / "best.pt"
        checkpoint_path = best_path if best_path.exists() else None
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, device)

    rows: list[dict[str, Any]] = []
    for condition_value, group_df in metadata.groupby(condition_column):
        subset_indices: list[int] = []
        for file_name in group_df["file_name"].astype(str).unique():
            subset_indices.extend(file_name_to_indices.get(file_name, []))
        unique_indices = sorted(set(subset_indices))
        if not unique_indices:
            continue

        subset = Subset(dataset, unique_indices)
        loader = DataLoader(
            subset,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["train"]["num_workers"]),
            pin_memory=torch.cuda.is_available(),
            collate_fn=partial(collate_fn, image_processor=image_processor),
        )
        metrics = evaluate_model(
            model=model,
            image_processor=image_processor,
            dataloader=loader,
            device=device,
            score_threshold=float(config["eval"]["score_threshold"]),
        )

        rows.append(
            {
                "condition_column": condition_column,
                "condition_value": str(condition_value),
                "num_images": len(unique_indices),
                "map": metrics.get("map", 0.0),
                "map_50": metrics.get("map_50", 0.0),
                "map_75": metrics.get("map_75", 0.0),
                "mar_100": metrics.get("mar_100", 0.0),
            }
        )

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}_{condition_column}_metrics.csv"
    pd.DataFrame(rows).sort_values(by="map_50", ascending=False).to_csv(out_path, index=False)
    return out_path
