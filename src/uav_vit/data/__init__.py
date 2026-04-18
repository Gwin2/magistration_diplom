"""Data module for UAV ViT detection framework.

This module provides dataset classes and utilities for working with COCO-format
detection data, including video-to-COCO conversion.
"""

from uav_vit.data.dataset import CocoDetectionDataset, collate_fn
from uav_vit.data.video_to_coco import VideoToCocoConfig, convert_video_annotations_to_coco

__all__ = [
    "CocoDetectionDataset",
    "collate_fn",
    "VideoToCocoConfig",
    "convert_video_annotations_to_coco",
]
