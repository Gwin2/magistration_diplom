"""Data module for UAV ViT detection framework.

This module provides dataset classes and utilities for working with COCO-format
detection data, including video-to-COCO conversion.
"""

from uav_vit.data.video_to_coco import (
    VideoToCocoConfig,
    _assign_splits,
    convert_video_annotations_to_coco,
)

__all__ = [
    "VideoToCocoConfig",
    "_assign_splits",
    "convert_video_annotations_to_coco",
]

try:
    from uav_vit.data.dataset import CocoDetectionDataset, collate_fn

    __all__.extend(["CocoDetectionDataset", "collate_fn"])
except ImportError:
    pass
