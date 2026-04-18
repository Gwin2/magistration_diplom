from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDetectionDataset(Dataset):
    """COCO dataset wrapper adapted for HuggingFace object detection models."""

    def __init__(self, images_dir: str | Path, annotations_path: str | Path, image_processor: Any) -> None:
        self.images_dir = Path(images_dir)
        self.coco = COCO(str(annotations_path))
        self.image_ids = sorted(self.coco.getImgIds())
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.images_dir / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        annotation_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        annotations = self.coco.loadAnns(annotation_ids)

        coco_annotations: list[dict[str, Any]] = []
        metric_boxes: list[list[float]] = []
        metric_labels: list[int] = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            if w <= 1e-6 or h <= 1e-6:
                continue
            coco_annotations.append(
                {
                    "id": ann["id"],
                    "category_id": int(ann["category_id"]),
                    "bbox": [x, y, w, h],
                    "area": float(ann.get("area", w * h)),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )
            metric_boxes.append([x, y, x + w, y + h])
            metric_labels.append(int(ann["category_id"]))

        formatted_target = {"image_id": image_id, "annotations": coco_annotations}
        encoded_inputs = self.image_processor(images=image, annotations=formatted_target, return_tensors="pt")
        labels = encoded_inputs["labels"][0]
        pixel_values = encoded_inputs["pixel_values"].squeeze(0)

        if metric_boxes:
            target_boxes = torch.tensor(metric_boxes, dtype=torch.float32)
            target_labels = torch.tensor(metric_labels, dtype=torch.int64)
        else:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_labels = torch.zeros((0,), dtype=torch.int64)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "targets": {"boxes": target_boxes, "labels": target_labels},
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "image_id": image_id,
        }


def collate_fn(batch: list[dict[str, Any]], image_processor: Any) -> dict[str, Any]:
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    targets = [item["targets"] for item in batch]
    orig_sizes = torch.stack([item["orig_size"] for item in batch])
    image_ids = [item["image_id"] for item in batch]

    if hasattr(image_processor, "pad"):
        encoded = image_processor.pad(pixel_values, return_tensors="pt")
        batch_out: dict[str, Any] = {"pixel_values": encoded["pixel_values"], "labels": labels}
        if "pixel_mask" in encoded:
            batch_out["pixel_mask"] = encoded["pixel_mask"]
    else:
        batch_out = {"pixel_values": torch.stack(pixel_values), "labels": labels}

    batch_out["targets"] = targets
    batch_out["orig_sizes"] = orig_sizes
    batch_out["image_ids"] = image_ids
    return batch_out
