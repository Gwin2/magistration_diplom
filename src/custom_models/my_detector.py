from __future__ import annotations

from transformers import AutoImageProcessor, AutoModelForObjectDetection

from uav_vit.models import ModelBundle, register_model


@register_model("my_detector")
def build_my_detector(config: dict) -> ModelBundle:
    """Template for custom model registration."""
    checkpoint = config["model"].get("checkpoint") or "facebook/detr-resnet-50"
    # SECURITY FIX: Use specific revision to prevent supply chain attacks
    revision = config["model"].get("revision")  # Optional: specify exact model version
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        revision=revision,
        ignore_mismatched_sizes=True,
        num_labels=int(config["model"]["num_labels"]),
        id2label={int(k): v for k, v in config["model"]["id2label"].items()},
        label2id={str(k): int(v) for k, v in config["model"]["label2id"].items()},
    )
    processor = AutoImageProcessor.from_pretrained(checkpoint, revision=revision)
    return ModelBundle(model=model, image_processor=processor, name="my_detector")
