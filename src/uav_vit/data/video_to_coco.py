"""Video to COCO conversion utilities for UAV ViT framework."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class VideoToCocoConfig:
    """Configuration for video-to-COCO conversion.

    Args:
        video_dir: Path to directory containing video files.
        annotations_csv: Path to CSV file with frame-level annotations.
        output_dir: Path to output directory for COCO format data.
        train_ratio: Ratio of data for training split.
        val_ratio: Ratio of data for validation split.
        test_ratio: Ratio of data for test split.
        seed: Random seed for reproducibility.
        normalized_boxes: Whether bounding boxes are normalized (0-1).
        image_format: Image file format extension (e.g., 'jpg', 'png').
    """

    video_dir: Path
    annotations_csv: Path
    output_dir: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    seed: int = 42
    normalized_boxes: bool = False
    image_format: str = "jpg"


def _assign_splits(
    frame_table: Any,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Any:
    """Assign train/val/test splits to frames grouped by video.

    Args:
        frame_table: pandas DataFrame with video_name and frame columns.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        seed: Random seed.

    Returns:
        DataFrame with added 'split' column.
    """
    import pandas as pd

    random.seed(seed)
    video_names = frame_table["video_name"].unique()
    random.shuffle(video_names)

    n_videos = len(video_names)
    n_train = max(1, int(round(n_videos * train_ratio)))
    n_val = max(1, int(round(n_videos * val_ratio)))
    n_test = n_videos - n_train - n_val

    if n_test < 1:
        n_test = 1
        n_train = max(1, n_videos - n_val - n_test)

    train_videos = set(video_names[:n_train])
    val_videos = set(video_names[n_train : n_train + n_val])
    test_videos = set(video_names[n_train + n_val :])

    def get_split(video_name: str) -> str:
        if video_name in train_videos:
            return "train"
        if video_name in val_videos:
            return "val"
        return "test"

    splits = frame_table["video_name"].apply(get_split)
    result = frame_table.copy()
    result["split"] = splits
    return result


def convert_video_annotations_to_coco(config: VideoToCocoConfig) -> dict[str, Any]:
    """Convert video annotations to COCO format with train/val/test splits.

    Args:
        config: VideoToCocoConfig with paths and parameters.

    Returns:
        Dictionary with statistics about the conversion.
    """
    import pandas as pd

    random.seed(config.seed)
    np.random.seed(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (config.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (config.output_dir / "annotations").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.annotations_csv)

    if "split" not in df.columns:
        df = _assign_splits(df, config.train_ratio, config.val_ratio, config.test_ratio, config.seed)

    category_map: dict[str, int] = {}
    category_id = 1

    images_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    annotations_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    image_id_counter = 1
    ann_id_counter = 1

    stats = {
        "total_frames": 0,
        "splits": {},
        "categories": [],
    }

    video_paths = {v.name: v for v in config.video_dir.iterdir() if v.suffix.lower() in [".mp4", ".avi", ".mov"]}

    grouped = df.groupby(["video_name", "split"])
    for (video_name, split), group in grouped:
        if video_name not in video_paths:
            continue

        video_path = video_paths[video_name]
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            continue

        frame_indices = sorted(group["frame_idx"].unique())

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if not ret:
                continue

            frame_rows = group[group["frame_idx"] == frame_idx]

            image_filename = f"{video_name.stem}_{frame_idx:06d}.{config.image_format}"
            image_path = config.output_dir / "images" / split / image_filename

            if config.image_format.lower() == "jpg":
                cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(image_path), frame)

            height, width = frame.shape[:2]

            image_info = {
                "id": image_id_counter,
                "file_name": image_filename,
                "width": width,
                "height": height,
            }
            images_by_split[split].append(image_info)

            for _, row in frame_rows.iterrows():
                category_name = str(row.get("category", "uav"))
                if category_name not in category_map:
                    category_map[category_name] = category_id
                    category_id += 1

                cat_id = category_map[category_name]

                if "bbox_x" in row and "bbox_y" in row and "bbox_w" in row and "bbox_h" in row:
                    x = float(row["bbox_x"])
                    y = float(row["bbox_y"])
                    w = float(row["bbox_w"])
                    h = float(row["bbox_h"])
                elif "bbox" in row:
                    bbox_str = row["bbox"]
                    if isinstance(bbox_str, str):
                        bbox_parts = bbox_str.strip("[]()").split(",")
                        if len(bbox_parts) >= 4:
                            x, y, w, h = map(float, bbox_parts[:4])
                        else:
                            continue
                    else:
                        continue
                else:
                    continue

                if config.normalized_boxes:
                    x = x * width
                    y = y * height
                    w = w * width
                    h = h * height

                if w <= 0 or h <= 0:
                    continue

                area = w * h

                annotation = {
                    "id": ann_id_counter,
                    "image_id": image_id_counter,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0,
                }
                annotations_by_split[split].append(annotation)
                ann_id_counter += 1

            image_id_counter += 1
            stats["total_frames"] += 1

        cap.release()

    categories = [{"id": cid, "name": name} for name, cid in category_map.items()]
    stats["categories"] = categories

    for split in ["train", "val", "test"]:
        coco_data = {
            "info": {
                "description": f"UAV detection dataset - {split} split",
                "version": "1.0",
                "year": 2025,
            },
            "licenses": [],
            "images": images_by_split[split],
            "annotations": annotations_by_split[split],
            "categories": categories,
        }

        ann_file = config.output_dir / "annotations" / f"instances_{split}.json"
        with ann_file.open("w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        split_stats = {
            "num_images": len(images_by_split[split]),
            "num_annotations": len(annotations_by_split[split]),
        }
        stats["splits"][split] = split_stats

    return stats
