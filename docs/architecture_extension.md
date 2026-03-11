# Extending with Custom Architectures

## 1. Create custom module

Example file: `src/custom_models/my_detector.py`

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from uav_vit.models import ModelBundle, register_model


@register_model("my_detector")
def build_my_detector(config: dict) -> ModelBundle:
    checkpoint = config["model"].get("checkpoint") or "facebook/detr-resnet-50"
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=config["model"]["num_labels"],
        id2label=config["model"]["id2label"],
        label2id=config["model"]["label2id"],
        ignore_mismatched_sizes=True,
    )
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    return ModelBundle(model=model, image_processor=processor, name="my_detector")
```

## 2. Reference custom module in YAML

```yaml
model:
  name: my_detector
  custom_modules:
    - custom_models.my_detector
```

## 3. Train as usual

```bash
uav-vit train --config configs/experiments/custom_model_template.yaml
```

## Notes

- You can register any model that accepts `pixel_values`, optional `pixel_mask`, and `labels`.
- Keep output format compatible with HuggingFace object detection models for evaluator reuse.
