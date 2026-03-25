# ruff: noqa: E501
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from uav_vit.control.state import slugify

BASE_MODEL_LIBRARY: dict[str, dict[str, Any]] = {
    "detr_resnet50": {
        "id": "detr_resnet50",
        "label": "DETR ResNet-50",
        "checkpoint": "facebook/detr-resnet-50",
        "family": "detr",
        "summary": "Balanced baseline for general UAV detection.",
    },
    "yolos_tiny": {
        "id": "yolos_tiny",
        "label": "YOLOS Tiny",
        "checkpoint": "hustvl/yolos-tiny",
        "family": "yolos",
        "summary": "Compact baseline when latency matters more than capacity.",
    },
    "hf_auto": {
        "id": "hf_auto",
        "label": "HF Auto",
        "checkpoint": "facebook/detr-resnet-50",
        "family": "hf",
        "summary": "Bring your own checkpoint and keep the constructor-driven heads.",
    },
}

GOAL_LIBRARY = [
    {
        "id": "balanced",
        "label": "Balanced",
        "summary": "Keep head depth moderate and training stable.",
    },
    {
        "id": "accuracy",
        "label": "Accuracy",
        "summary": "Prefer deeper heads and more capacity for difficult scenes.",
    },
    {
        "id": "latency",
        "label": "Latency",
        "summary": "Prefer fewer layers and lighter checkpoints.",
    },
    {
        "id": "stability",
        "label": "Stability",
        "summary": "Add normalization and regularization for noisy data.",
    },
]

LAYER_LIBRARY = [
    {
        "type": "linear",
        "label": "Linear",
        "description": "Dense projection on token features.",
        "params": [
            {"name": "out_features", "type": "int", "default": 256, "min": 16, "step": 16},
            {"name": "bias", "type": "bool", "default": True},
        ],
    },
    {
        "type": "gelu",
        "label": "GELU",
        "description": "Smooth activation for transformer-style heads.",
        "params": [],
    },
    {
        "type": "relu",
        "label": "ReLU",
        "description": "Low-cost activation for faster heads.",
        "params": [{"name": "inplace", "type": "bool", "default": True}],
    },
    {
        "type": "dropout",
        "label": "Dropout",
        "description": "Regularization for difficult weather or small datasets.",
        "params": [
            {"name": "p", "type": "float", "default": 0.1, "min": 0.0, "max": 0.8, "step": 0.05}
        ],
    },
    {
        "type": "layer_norm",
        "label": "LayerNorm",
        "description": "Stabilizes token features before prediction.",
        "params": [
            {
                "name": "eps",
                "type": "float",
                "default": 1e-5,
                "min": 1e-7,
                "max": 1e-3,
                "step": 1e-5,
            }
        ],
    },
    {
        "type": "transformer_encoder",
        "label": "Transformer Encoder",
        "description": "Adds token mixing before the prediction head.",
        "params": [
            {"name": "nhead", "type": "int", "default": 8, "min": 1, "step": 1},
            {"name": "dim_feedforward", "type": "int", "default": 1024, "min": 128, "step": 64},
            {
                "name": "dropout",
                "type": "float",
                "default": 0.1,
                "min": 0.0,
                "max": 0.8,
                "step": 0.05,
            },
            {"name": "norm_first", "type": "bool", "default": True},
        ],
    },
    {
        "type": "residual_mlp",
        "label": "Residual MLP",
        "description": "Residual feed-forward refinement for dense prediction.",
        "params": [
            {
                "name": "expansion",
                "type": "float",
                "default": 2.0,
                "min": 1.0,
                "max": 8.0,
                "step": 0.5,
            },
            {
                "name": "dropout",
                "type": "float",
                "default": 0.1,
                "min": 0.0,
                "max": 0.8,
                "step": 0.05,
            },
            {"name": "activation", "type": "enum", "default": "gelu", "options": ["gelu", "relu"]},
        ],
    },
    {
        "type": "identity",
        "label": "Identity",
        "description": "No-op layer, useful for temporary staging.",
        "params": [],
    },
]

LAYER_DEFAULTS = {
    item["type"]: {param["name"]: param["default"] for param in item["params"]}
    for item in LAYER_LIBRARY
}
LAYER_INDEX = {item["type"]: item for item in LAYER_LIBRARY}


def list_constructor_catalog() -> dict[str, Any]:
    return {
        "base_models": list(BASE_MODEL_LIBRARY.values()),
        "goals": GOAL_LIBRARY,
        "layers": LAYER_LIBRARY,
        "templates": {
            "default_blueprint": default_blueprint(),
        },
    }


def default_blueprint(
    name: str = "custom_detector_builder",
    base_model: str = "detr_resnet50",
    goal: str = "balanced",
) -> dict[str, Any]:
    preset = BASE_MODEL_LIBRARY.get(base_model, BASE_MODEL_LIBRARY["detr_resnet50"])
    return {
        "name": name,
        "base_model": preset["id"],
        "checkpoint": preset["checkpoint"],
        "goal": goal,
        "dataset_id": "",
        "labels": ["uav"],
        "train_backbone": goal in {"accuracy", "balanced"},
        "head_specs": {
            "classifier": [
                layer_with_defaults("linear", out_features=256),
                layer_with_defaults("gelu"),
                layer_with_defaults("dropout", p=0.1),
            ],
            "bbox": [
                layer_with_defaults("linear", out_features=256),
                layer_with_defaults("relu"),
                layer_with_defaults("dropout", p=0.05),
            ],
        },
    }


def layer_with_defaults(layer_type: str, **overrides: Any) -> dict[str, Any]:
    params = dict(LAYER_DEFAULTS.get(layer_type, {}))
    params.update(overrides)
    return {"type": layer_type, "params": params}


def normalize_blueprint(raw_blueprint: dict[str, Any] | None) -> dict[str, Any]:
    payload = deepcopy(raw_blueprint or {})
    normalized = default_blueprint(
        name=str(payload.get("name") or "custom_detector_builder"),
        base_model=str(payload.get("base_model") or "detr_resnet50"),
        goal=str(payload.get("goal") or "balanced"),
    )
    normalized["checkpoint"] = str(
        payload.get("checkpoint")
        or normalized["checkpoint"]
        or BASE_MODEL_LIBRARY[normalized["base_model"]]["checkpoint"]
    )
    normalized["dataset_id"] = str(payload.get("dataset_id") or "")
    labels = payload.get("labels") or normalized["labels"]
    if not isinstance(labels, list):
        labels = [str(labels)]
    normalized["labels"] = [str(label).strip() for label in labels if str(label).strip()] or ["uav"]
    normalized["train_backbone"] = bool(payload.get("train_backbone", normalized["train_backbone"]))

    raw_head_specs = (
        payload.get("head_specs") if isinstance(payload.get("head_specs"), dict) else {}
    )
    normalized_head_specs: dict[str, list[dict[str, Any]]] = {}
    for head_name in ("classifier", "bbox"):
        raw_layers = raw_head_specs.get(head_name, normalized["head_specs"][head_name])
        if not isinstance(raw_layers, list):
            raw_layers = normalized["head_specs"][head_name]
        normalized_layers: list[dict[str, Any]] = []
        for raw_layer in raw_layers:
            if not isinstance(raw_layer, dict):
                continue
            layer_type = str(raw_layer.get("type") or "").strip()
            if layer_type not in LAYER_INDEX:
                continue
            params = dict(LAYER_DEFAULTS[layer_type])
            raw_params = raw_layer.get("params", {})
            if isinstance(raw_params, dict):
                for param in LAYER_INDEX[layer_type]["params"]:
                    name = param["name"]
                    if name in raw_params:
                        params[name] = _coerce_param(raw_params[name], param)
            normalized_layers.append({"type": layer_type, "params": params})
        normalized_head_specs[head_name] = normalized_layers or deepcopy(
            normalized["head_specs"][head_name]
        )

    normalized["head_specs"] = normalized_head_specs
    return normalized


def recommend_blueprint(
    raw_blueprint: dict[str, Any] | None,
    dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blueprint = normalize_blueprint(raw_blueprint)
    goal = blueprint["goal"]
    dataset_tags = _dataset_tags(dataset)
    dataset_size = int((dataset or {}).get("file_count") or 0)
    notes: list[str] = []

    if goal == "latency":
        blueprint["base_model"] = "yolos_tiny"
        blueprint["checkpoint"] = BASE_MODEL_LIBRARY["yolos_tiny"]["checkpoint"]
        blueprint["train_backbone"] = False
        blueprint["head_specs"]["classifier"] = [
            layer_with_defaults("linear", out_features=192),
            layer_with_defaults("relu"),
            layer_with_defaults("dropout", p=0.05),
        ]
        blueprint["head_specs"]["bbox"] = [
            layer_with_defaults("linear", out_features=128),
            layer_with_defaults("relu"),
        ]
        notes.append("Latency goal switches the constructor to a lighter YOLOS head profile.")
    elif goal == "accuracy":
        blueprint["base_model"] = "detr_resnet50"
        blueprint["checkpoint"] = BASE_MODEL_LIBRARY["detr_resnet50"]["checkpoint"]
        blueprint["train_backbone"] = True
        blueprint["head_specs"]["classifier"] = [
            layer_with_defaults("layer_norm"),
            layer_with_defaults("transformer_encoder", nhead=8, dim_feedforward=1024, dropout=0.1),
            layer_with_defaults("residual_mlp", expansion=2.5, dropout=0.1, activation="gelu"),
            layer_with_defaults("dropout", p=0.1),
        ]
        blueprint["head_specs"]["bbox"] = [
            layer_with_defaults("layer_norm"),
            layer_with_defaults("residual_mlp", expansion=2.0, dropout=0.05, activation="relu"),
            layer_with_defaults("linear", out_features=256),
            layer_with_defaults("gelu"),
        ]
        notes.append("Accuracy goal deepens both heads and keeps the backbone trainable.")
    elif goal == "stability":
        blueprint["head_specs"]["classifier"] = [
            layer_with_defaults("layer_norm"),
            layer_with_defaults("linear", out_features=256),
            layer_with_defaults("gelu"),
            layer_with_defaults("dropout", p=0.15),
        ]
        blueprint["head_specs"]["bbox"] = [
            layer_with_defaults("layer_norm"),
            layer_with_defaults("linear", out_features=192),
            layer_with_defaults("relu"),
            layer_with_defaults("dropout", p=0.1),
        ]
        blueprint["train_backbone"] = dataset_size > 3500
        notes.append(
            "Stability goal adds normalization and regularization for noisy training loops."
        )

    if dataset_size and dataset_size < 1500:
        blueprint["train_backbone"] = False
        notes.append("Small dataset: freeze the base detector and keep head depth compact.")
    elif dataset_size > 8000 and goal != "latency":
        blueprint["train_backbone"] = True
        notes.append("Larger dataset: allow backbone tuning to exploit more capacity.")

    if dataset_tags & {"fog", "rain", "snow", "night", "lowlight", "blur"}:
        blueprint["head_specs"]["classifier"].insert(0, layer_with_defaults("layer_norm"))
        _increase_dropout(blueprint["head_specs"]["classifier"], minimum=0.15)
        _increase_dropout(blueprint["head_specs"]["bbox"], minimum=0.1)
        notes.append("Weather and low-visibility tags suggest extra normalization and dropout.")

    if dataset_tags & {"small-object", "small", "distant", "zoom"} and goal != "latency":
        if blueprint["base_model"] == "yolos_tiny":
            blueprint["base_model"] = "detr_resnet50"
            blueprint["checkpoint"] = BASE_MODEL_LIBRARY["detr_resnet50"]["checkpoint"]
        blueprint["head_specs"]["classifier"].append(
            layer_with_defaults("transformer_encoder", nhead=8, dim_feedforward=768, dropout=0.1)
        )
        notes.append(
            "Small-object conditions benefit from stronger token mixing before classification."
        )

    if dataset_tags & {"maneuver", "fast", "occlusion"}:
        blueprint["head_specs"]["bbox"].append(
            layer_with_defaults("residual_mlp", expansion=2.0, dropout=0.05, activation="relu")
        )
        notes.append("Fast motion and occlusion tags add residual refinement to the bbox head.")

    preview = build_constructor_preview(blueprint, dataset)
    preview["notes"] = notes + preview["notes"]
    return preview


def build_constructor_preview(
    raw_blueprint: dict[str, Any] | None,
    dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blueprint = normalize_blueprint(raw_blueprint)
    config = build_config_from_blueprint(blueprint, dataset=dataset)
    model_slug = slugify(blueprint["name"])
    summary = {
        "model_slug": model_slug,
        "base_model": blueprint["base_model"],
        "checkpoint": blueprint["checkpoint"],
        "num_labels": len(blueprint["labels"]),
        "classifier_layers": len(blueprint["head_specs"]["classifier"]),
        "bbox_layers": len(blueprint["head_specs"]["bbox"]),
        "train_backbone": blueprint["train_backbone"],
    }
    notes = [
        f"Constructor builds detection heads on top of {BASE_MODEL_LIBRARY[blueprint['base_model']]['label']}.",
        f"Classifier head depth: {summary['classifier_layers']} layers, bbox head depth: {summary['bbox_layers']} layers.",
    ]
    return {
        "blueprint": blueprint,
        "summary": summary,
        "notes": notes,
        "config": config,
        "config_yaml": yaml.safe_dump(config, allow_unicode=False, sort_keys=False),
        "source_code": render_constructor_source(blueprint),
    }


def build_config_from_blueprint(
    blueprint: dict[str, Any],
    dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_slug = slugify(blueprint["name"])
    labels = list(blueprint["labels"])
    dataset_paths = _build_dataset_paths(dataset)
    goal = blueprint["goal"]
    learning_rate = 2.0e-5
    epochs = 30
    if goal == "latency":
        learning_rate = 3.0e-5
        epochs = 20
    elif goal == "accuracy":
        learning_rate = 1.5e-5
        epochs = 40
    elif goal == "stability":
        learning_rate = 1.0e-5
        epochs = 35

    return {
        "experiment": {"name": model_slug, "seed": 42},
        "paths": {
            **dataset_paths,
            "output_dir": f"runs/{model_slug}",
        },
        "mlflow": {
            "enabled": True,
            "tracking_uri": "http://mlflow:5000",
            "experiment_name": "uav-vit-thesis",
            "run_name": model_slug,
            "log_checkpoints": True,
        },
        "tensorboard": {"enabled": True},
        "model": {
            "name": model_slug,
            "base_model": blueprint["base_model"],
            "checkpoint": blueprint["checkpoint"],
            "num_labels": len(labels),
            "id2label": {str(index): label for index, label in enumerate(labels)},
            "label2id": {label: index for index, label in enumerate(labels)},
            "train_backbone": blueprint["train_backbone"],
            "custom_modules": [f"custom_models.{model_slug}"],
        },
        "train": {
            "device": "auto",
            "epochs": epochs,
            "batch_size": 4,
            "learning_rate": learning_rate,
            "weight_decay": 1.0e-4,
            "num_workers": 4,
            "grad_clip_norm": 1.0,
            "log_interval": 20,
            "mixed_precision": True,
            "eval_every_epoch": True,
            "checkpoint_metric": "map",
            "checkpoint_mode": "max",
        },
        "eval": {
            "score_threshold": 0.1,
            "latency_warmup_iters": 10,
            "latency_iters": 50,
        },
        "data": {
            "processor_size": 960 if goal == "accuracy" else 800,
            "normalize_boxes": False,
        },
        "constructor": {
            "goal": goal,
            "blueprint": blueprint,
        },
    }


def render_constructor_source(blueprint: dict[str, Any]) -> str:
    slug = slugify(blueprint["name"])
    blueprint_json = json.dumps(blueprint, indent=4, sort_keys=True)
    checkpoint = blueprint["checkpoint"]
    return (
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n"
        "import torch\n"
        "from torch import nn\n"
        "from transformers import AutoImageProcessor, AutoModelForObjectDetection\n\n"
        "from uav_vit.models import ModelBundle, register_model\n\n\n"
        f"HEAD_BLUEPRINT = {blueprint_json}\n\n\n"
        "class ResidualMLPBlock(nn.Module):\n"
        "    def __init__(self, features: int, expansion: float = 2.0, dropout: float = 0.1, activation: str = 'gelu') -> None:\n"
        "        super().__init__()\n"
        "        hidden = max(int(features * float(expansion)), features)\n"
        "        self.norm = nn.LayerNorm(features)\n"
        "        self.fc1 = nn.Linear(features, hidden)\n"
        "        self.fc2 = nn.Linear(hidden, features)\n"
        "        self.dropout = nn.Dropout(float(dropout))\n"
        "        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU(inplace=True)\n\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        residual = x\n"
        "        x = self.norm(x)\n"
        "        x = self.fc1(x)\n"
        "        x = self.activation(x)\n"
        "        x = self.dropout(x)\n"
        "        x = self.fc2(x)\n"
        "        x = self.dropout(x)\n"
        "        return residual + x\n\n\n"
        "def _safe_nhead(requested: int, features: int) -> int:\n"
        "    requested = max(int(requested), 1)\n"
        "    for candidate in [requested, 8, 6, 4, 3, 2, 1]:\n"
        "        if candidate <= features and features % candidate == 0:\n"
        "            return candidate\n"
        "    return 1\n\n\n"
        "def _build_layers(input_dim: int, layers: list[dict[str, Any]], output_dim: int) -> nn.Sequential:\n"
        "    modules: list[nn.Module] = []\n"
        "    current_dim = input_dim\n"
        "    for layer in layers:\n"
        "        layer_type = str(layer.get('type'))\n"
        "        params = dict(layer.get('params', {}))\n"
        "        if layer_type == 'linear':\n"
        "            out_features = int(params.get('out_features', current_dim))\n"
        "            modules.append(nn.Linear(current_dim, out_features, bias=bool(params.get('bias', True))))\n"
        "            current_dim = out_features\n"
        "        elif layer_type == 'gelu':\n"
        "            modules.append(nn.GELU())\n"
        "        elif layer_type == 'relu':\n"
        "            modules.append(nn.ReLU(inplace=bool(params.get('inplace', True))))\n"
        "        elif layer_type == 'dropout':\n"
        "            modules.append(nn.Dropout(float(params.get('p', 0.1))))\n"
        "        elif layer_type == 'layer_norm':\n"
        "            modules.append(nn.LayerNorm(current_dim, eps=float(params.get('eps', 1e-5))))\n"
        "        elif layer_type == 'transformer_encoder':\n"
        "            nhead = _safe_nhead(int(params.get('nhead', 8)), current_dim)\n"
        "            modules.append(\n"
        "                nn.TransformerEncoderLayer(\n"
        "                    d_model=current_dim,\n"
        "                    nhead=nhead,\n"
        "                    dim_feedforward=int(params.get('dim_feedforward', max(current_dim * 4, 256))),\n"
        "                    dropout=float(params.get('dropout', 0.1)),\n"
        "                    batch_first=True,\n"
        "                    norm_first=bool(params.get('norm_first', True)),\n"
        "                )\n"
        "            )\n"
        "        elif layer_type == 'residual_mlp':\n"
        "            modules.append(\n"
        "                ResidualMLPBlock(\n"
        "                    current_dim,\n"
        "                    expansion=float(params.get('expansion', 2.0)),\n"
        "                    dropout=float(params.get('dropout', 0.1)),\n"
        "                    activation=str(params.get('activation', 'gelu')),\n"
        "                )\n"
        "            )\n"
        "        elif layer_type == 'identity':\n"
        "            modules.append(nn.Identity())\n"
        "        else:\n"
        "            raise ValueError(f'Unsupported constructor layer: {layer_type}')\n"
        "    modules.append(nn.Linear(current_dim, output_dim))\n"
        "    return nn.Sequential(*modules)\n\n\n"
        "def _resolve_hidden_size(model: nn.Module) -> int:\n"
        "    for owner in [getattr(model, 'config', None), getattr(getattr(model, 'model', None), 'config', None)]:\n"
        "        if owner is None:\n"
        "            continue\n"
        "        for field in ('hidden_size', 'd_model'):\n"
        "            value = getattr(owner, field, None)\n"
        "            if value:\n"
        "                return int(value)\n"
        "    return 256\n\n\n"
        f"@register_model('{slug}')\n"
        f"def build_{slug}(config: dict[str, Any]) -> ModelBundle:\n"
        f"    checkpoint = config['model'].get('checkpoint') or '{checkpoint}'\n"
        "    model = AutoModelForObjectDetection.from_pretrained(\n"
        "        checkpoint,\n"
        "        ignore_mismatched_sizes=True,\n"
        "        num_labels=int(config['model']['num_labels']),\n"
        "        id2label={int(key): value for key, value in config['model']['id2label'].items()},\n"
        "        label2id={str(key): int(value) for key, value in config['model']['label2id'].items()},\n"
        "    )\n"
        "    hidden_size = _resolve_hidden_size(model)\n"
        "    class_output_dim = int(config['model']['num_labels']) + 1\n"
        "    if not hasattr(model, 'class_labels_classifier') or not hasattr(model, 'bbox_predictor'):\n"
        "        raise ValueError('Selected checkpoint does not expose editable DETR/YOLOS detection heads.')\n"
        "    model.class_labels_classifier = _build_layers(\n"
        "        hidden_size,\n"
        "        HEAD_BLUEPRINT['head_specs']['classifier'],\n"
        "        class_output_dim,\n"
        "    )\n"
        "    model.bbox_predictor = _build_layers(\n"
        "        hidden_size,\n"
        "        HEAD_BLUEPRINT['head_specs']['bbox'],\n"
        "        4,\n"
        "    )\n"
        "    processor = AutoImageProcessor.from_pretrained(checkpoint)\n"
        f"    return ModelBundle(model=model, image_processor=processor, name='{slug}')\n"
    )


def _coerce_param(value: Any, param: dict[str, Any]) -> Any:
    kind = param["type"]
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return str(value)


def _dataset_tags(dataset: dict[str, Any] | None) -> set[str]:
    if not dataset:
        return set()
    tags = {str(tag).strip().lower() for tag in dataset.get("tags", []) if str(tag).strip()}
    description = str(dataset.get("description") or "").lower()
    for token in [
        "fog",
        "rain",
        "snow",
        "night",
        "lowlight",
        "blur",
        "small",
        "distant",
        "zoom",
        "maneuver",
        "fast",
        "occlusion",
    ]:
        if token in description:
            tags.add(token)
    return tags


def _increase_dropout(layers: list[dict[str, Any]], minimum: float) -> None:
    for layer in layers:
        if layer["type"] == "dropout":
            layer["params"]["p"] = max(float(layer["params"].get("p", 0.1)), minimum)
            return
    layers.append(layer_with_defaults("dropout", p=minimum))


def _build_dataset_paths(dataset: dict[str, Any] | None) -> dict[str, str]:
    default_root = "data/processed/uav_coco"
    if not dataset or not dataset.get("path"):
        return {
            "train_images": f"{default_root}/images/train",
            "val_images": f"{default_root}/images/val",
            "test_images": f"{default_root}/images/test",
            "train_annotations": f"{default_root}/annotations/instances_train.json",
            "val_annotations": f"{default_root}/annotations/instances_val.json",
            "test_annotations": f"{default_root}/annotations/instances_test.json",
        }

    dataset_root = Path(str(dataset["path"]).replace("\\", "/"))
    return {
        "train_images": str(dataset_root / "images" / "train").replace("\\", "/"),
        "val_images": str(dataset_root / "images" / "val").replace("\\", "/"),
        "test_images": str(dataset_root / "images" / "test").replace("\\", "/"),
        "train_annotations": str(dataset_root / "annotations" / "instances_train.json").replace(
            "\\", "/"
        ),
        "val_annotations": str(dataset_root / "annotations" / "instances_val.json").replace(
            "\\", "/"
        ),
        "test_annotations": str(dataset_root / "annotations" / "instances_test.json").replace(
            "\\", "/"
        ),
    }
