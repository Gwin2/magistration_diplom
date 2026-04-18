"""COCO detection dataset for UAV ViT framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO


class CocoDetectionDataset(torch.utils.data.Dataset):
    """COCO-format detection dataset with image processor integration.

    Args:
        images_dir: Path to directory containing image files.
        annotations_path: Path to COCO-format annotations JSON file.
        image_processor: Image processor from transformers or custom implementation.
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_path: str | Path,
        image_processor: Any,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.image_processor = image_processor

        self.coco = COCO(str(self.annotations_path))
        self.image_ids = sorted(self.coco.getImgIds())

        if len(self.image_ids) == 0:
            raise ValueError(f"No images found in {self.annotations_path}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        file_name = image_info["file_name"]

        image_path = self.images_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.image_processor is not None:
            processed = self.image_processor(images=image, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)
            pixel_mask = processed.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.squeeze(0)
                target["pixel_mask"] = pixel_mask
        else:
            pixel_values = torch.as_tensor(image).permute(2, 0, 1).float() / 255.0

        return {
            "pixel_values": pixel_values,
            "labels": target,
        }


def collate_fn(batch: list[dict[str, Any]], image_processor: Any | None = None) -> dict[str, Any]:
    """Collate function for DataLoader with COCO detection data.

    Args:
        batch: List of samples from CocoDetectionDataset.
        image_processor: Optional image processor for batching logic.

    Returns:
        Batched dictionary with pixel_values and labels.
    """
    pixel_values_list = [item["pixel_values"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    if image_processor is not None and hasattr(image_processor, "pad"):
        batched = image_processor.pad(
            [
                {"pixel_values": pv, **lbl}
                for pv, lbl in zip(pixel_values_list, labels_list, strict=False)
            ],
            return_tensors="pt",
        )
        pixel_values = batched["pixel_values"]
        pixel_mask = batched.get("pixel_mask")
        labels = []
        for i in range(len(labels_list)):
            label_dict = {k: v[i] if k != "image_id" else v for k, v in labels_list[i].items()}
            labels.append(label_dict)
    else:
        max_h = max(pv.shape[1] for pv in pixel_values_list)
        max_w = max(pv.shape[2] for pv in pixel_values_list)
        channels = pixel_values_list[0].shape[0]

        pixel_values = torch.zeros(len(batch), channels, max_h, max_w)
        for i, pv in enumerate(pixel_values_list):
            pixel_values[i, :, : pv.shape[1], : pv.shape[2]] = pv

        labels = labels_list

    result: dict[str, Any] = {"pixel_values": pixel_values, "labels": labels}
    if "pixel_mask" in locals() and pixel_mask is not None:
        result["pixel_mask"] = pixel_mask

    return result
