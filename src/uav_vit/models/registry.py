from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

ModelBuilder = Callable[[dict[str, Any]], "ModelBundle"]
MODEL_REGISTRY: dict[str, ModelBuilder] = {}


@dataclass
class ModelBundle:
    model: Any
    image_processor: Any
    name: str


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def build_model(config: dict[str, Any]) -> ModelBundle:
    model_name = config["model"]["name"]
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available}")
    return MODEL_REGISTRY[model_name](config)


def _build_hf_model(config: dict[str, Any], default_checkpoint: str) -> ModelBundle:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    model_cfg = config["model"]
    checkpoint = model_cfg.get("checkpoint") or default_checkpoint
    num_labels = int(model_cfg["num_labels"])
    id2label = {int(key): value for key, value in model_cfg["id2label"].items()}
    label2id = {str(key): int(value) for key, value in model_cfg["label2id"].items()}

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        ignore_mismatched_sizes=True,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return ModelBundle(model=model, image_processor=image_processor, name=model_cfg["name"])


@register_model("yolos_tiny")
def build_yolos_tiny(config: dict[str, Any]) -> ModelBundle:
    return _build_hf_model(config, default_checkpoint="hustvl/yolos-tiny")


@register_model("detr_resnet50")
def build_detr_resnet50(config: dict[str, Any]) -> ModelBundle:
    return _build_hf_model(config, default_checkpoint="facebook/detr-resnet-50")


@register_model("hf_auto")
def build_hf_auto(config: dict[str, Any]) -> ModelBundle:
    model_cfg = config["model"]
    checkpoint = model_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("model.checkpoint is required for model.name=hf_auto")
    return _build_hf_model(config, default_checkpoint=checkpoint)
