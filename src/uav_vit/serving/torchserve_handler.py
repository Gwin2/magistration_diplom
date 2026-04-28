from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

from uav_vit.config import load_yaml
from uav_vit.models import build_model


class UAVObjectDetectionHandler(BaseHandler):
    """TorchServe handler for UAV object detection models."""

    def __init__(self) -> None:
        super().__init__()
        self.initialized = False
        self.device = torch.device("cpu")
        self.model: Any = None
        self.image_processor: Any = None
        self.score_threshold = 0.2

    def initialize(self, context: Any) -> None:
        properties = context.system_properties
        model_dir = Path(properties.get("model_dir", "."))
        gpu_id = properties.get("gpu_id")
        if torch.cuda.is_available() and gpu_id is not None:
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        manifest = context.manifest
        serialized_file = manifest["model"].get("serializedFile", "best.pt")
        checkpoint_path = model_dir / serialized_file

        config_name = os.environ.get("TS_CONFIG_FILENAME", "inference_config.yaml")
        config_path = model_dir / config_name
        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing model config file {config_name} in model archive. "
                "Use scripts/export_torchserve.py to pack checkpoint with config."
            )

        config = load_yaml(config_path)
        bundle = build_model(config)
        self.model = bundle.model
        self.image_processor = bundle.image_processor

        # SECURITY FIX: Use weights_only=True to prevent arbitrary code execution
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = payload.get("model_state_dict", payload)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.score_threshold = float(os.environ.get("TS_SCORE_THRESHOLD", "0.2"))
        self.initialized = True

    def preprocess(
        self, data: list[dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]]]:
        images: list[Image.Image] = []
        sizes: list[tuple[int, int]] = []
        for row in data:
            payload = row.get("data") or row.get("body")
            if payload is None:
                raise ValueError("Request item does not contain 'data' or 'body'.")
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            width, height = image.size
            images.append(image)
            sizes.append((height, width))

        encoded = self.image_processor(images=images, return_tensors="pt")
        batch = {"pixel_values": encoded["pixel_values"].to(self.device)}
        if "pixel_mask" in encoded:
            batch["pixel_mask"] = encoded["pixel_mask"].to(self.device)
        return batch, sizes

    def inference(
        self, data: tuple[dict[str, torch.Tensor], list[tuple[int, int]]], *args: Any, **kwargs: Any
    ) -> Any:
        inputs, sizes = data
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor(sizes, dtype=torch.int64, device=self.device)
        predictions = self.image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.score_threshold,
            target_sizes=target_sizes,
        )
        return predictions

    def postprocess(self, data: Any) -> list[dict[str, Any]]:
        response: list[dict[str, Any]] = []
        for item in data:
            boxes = item["boxes"].detach().cpu().tolist()
            scores = item["scores"].detach().cpu().tolist()
            labels = item["labels"].detach().cpu().tolist()
            response.append(
                {
                    "boxes": [[float(v) for v in box] for box in boxes],
                    "scores": [float(v) for v in scores],
                    "labels": [int(v) for v in labels],
                }
            )
        return response
