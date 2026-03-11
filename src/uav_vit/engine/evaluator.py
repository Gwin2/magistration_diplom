from __future__ import annotations

import time
from typing import Any

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm


def _to_device_labels(labels: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    moved: list[dict[str, Any]] = []
    for item in labels:
        moved_item: dict[str, Any] = {}
        for key, value in item.items():
            moved_item[key] = value.to(device) if torch.is_tensor(value) else value
        moved.append(moved_item)
    return moved


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    image_processor: Any,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    score_threshold: float = 0.1,
) -> dict[str, float]:
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = batch["orig_sizes"].to(device)
        processed = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )

        predictions = [
            {
                "boxes": pred["boxes"].detach().cpu(),
                "scores": pred["scores"].detach().cpu(),
                "labels": pred["labels"].detach().cpu(),
            }
            for pred in processed
        ]
        targets = [
            {"boxes": target["boxes"].detach().cpu(), "labels": target["labels"].detach().cpu()}
            for target in batch["targets"]
        ]
        metric.update(predictions, targets)

    result = metric.compute()
    output: dict[str, float] = {}
    for key, value in result.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                output[key] = float(value.item())
            else:
                output[key] = float(value.float().mean().item())
        else:
            output[key] = float(value)
    return output


@torch.no_grad()
def benchmark_latency(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    warmup_iters: int = 10,
    latency_iters: int = 50,
) -> dict[str, float]:
    model.eval()
    iterator = iter(dataloader)
    try:
        batch = next(iterator)
    except StopIteration:
        return {"latency_ms": 0.0, "fps": 0.0}

    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)
    batch_size = pixel_values.shape[0]

    for _ in range(max(warmup_iters, 0)):
        _ = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings: list[float] = []
    for _ in range(max(latency_iters, 1)):
        start = time.perf_counter()
        _ = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)

    mean_seconds = sum(timings) / len(timings)
    latency_ms = mean_seconds * 1000.0
    fps = batch_size / mean_seconds if mean_seconds > 0 else 0.0
    return {"latency_ms": latency_ms, "fps": fps}
