"""Pydantic schemas for UAV ViT configuration validation.

This module provides strict type checking and validation for configuration files
using Pydantic models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExperimentConfig(BaseModel):
    """Experiment configuration."""

    name: str = Field(..., min_length=1, description="Experiment name")
    description: str | None = Field(default=None, description="Optional experiment description")
    tags: list[str] = Field(default_factory=list, description="Experiment tags for filtering")


class PathsConfig(BaseModel):
    """Path configuration for datasets and artifacts."""

    train_annotations: str | Path | None = Field(default=None, description="Train annotations path")
    val_annotations: str | Path | None = Field(
        default=None, description="Validation annotations path"
    )
    test_annotations: str | Path | None = Field(default=None, description="Test annotations path")
    images_dir: str | Path | None = Field(default=None, description="Images directory")
    checkpoint_dir: str | Path = Field(default="checkpoints", description="Checkpoint directory")
    artifact_dir: str | Path = Field(default="artifacts", description="Artifacts directory")

    @field_validator("train_annotations", "val_annotations", "test_annotations", mode="before")
    @classmethod
    def convert_none_strings(cls, value: Any) -> Any:
        """Convert 'null' strings to None."""
        if isinstance(value, str) and value.lower() in ("none", "null", ""):
            return None
        return value


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str = Field(..., min_length=1, description="Model name/identifier")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    num_labels: int | None = Field(default=None, ge=1, description="Number of detection classes")
    id2label: dict[int, str] | None = Field(default=None, description="ID to label mapping")
    label2id: dict[str, int] | None = Field(default=None, description="Label to ID mapping")
    custom_modules: list[str] = Field(default_factory=list, description="Custom module paths")

    @field_validator("id2label", mode="before")
    @classmethod
    def normalize_id2label(cls, value: Any) -> dict[int, str] | None:
        """Normalize id2label keys to integers."""
        if value is None:
            return None
        if isinstance(value, dict):
            return {int(k): v for k, v in value.items()}
        return value

    @field_validator("label2id", mode="before")
    @classmethod
    def normalize_label2id(cls, value: Any) -> dict[str, int] | None:
        """Normalize label2id values to integers."""
        if value is None:
            return None
        if isinstance(value, dict):
            return {str(k): int(v) for k, v in value.items()}
        return value


class TrainConfig(BaseModel):
    """Training configuration."""

    device: str = Field(default="auto", description="Device (auto, cuda, cpu)")
    epochs: int = Field(default=100, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=8, ge=1, description="Training batch size")
    learning_rate: float = Field(default=1e-4, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay")
    warmup_epochs: int = Field(default=5, ge=0, description="Warmup epochs")
    gradient_clip: float | None = Field(default=None, gt=0, description="Gradient clipping value")
    eval_every: int = Field(default=1, ge=1, description="Evaluate every N epochs")
    save_every: int = Field(default=1, ge=1, description="Save checkpoint every N epochs")
    mixed_precision: bool = Field(default=True, description="Use mixed precision training")
    num_workers: int = Field(default=4, ge=0, description="DataLoader workers")


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    batch_size: int = Field(default=8, ge=1, description="Evaluation batch size")
    iou_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.75],
        description="IoU thresholds for mAP calculation",
    )
    max_detections: int = Field(default=100, ge=1, description="Maximum detections per image")
    device: str = Field(default="auto", description="Device for evaluation")


class DataConfig(BaseModel):
    """Data augmentation and preprocessing configuration."""

    image_size: int | tuple[int, int] = Field(default=640, description="Input image size")
    augment: bool = Field(default=True, description="Enable data augmentation")
    normalize: bool = Field(default=True, description="Normalize images")
    mean: list[float] | None = Field(default=None, description="Normalization mean")
    std: list[float] | None = Field(default=None, description="Normalization std")


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""

    enabled: bool = Field(default=False, description="Enable MLflow tracking")
    tracking_uri: str | None = Field(default=None, description="MLflow tracking URI")
    experiment_name: str | None = Field(default=None, description="MLflow experiment name")
    run_name: str | None = Field(default=None, description="MLflow run name")


class PushGatewayConfig(BaseModel):
    """Prometheus PushGateway configuration."""

    enabled: bool = Field(default=False, description="Enable PushGateway export")
    url: str | None = Field(default=None, description="PushGateway URL")
    job: str = Field(default="uav-vit", description="Job name")
    instance: str | None = Field(default=None, description="Instance name")
    timeout_seconds: int = Field(default=5, ge=1, description="Request timeout")


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    pushgateway: PushGatewayConfig = Field(default_factory=PushGatewayConfig)


class Config(BaseModel):
    """Root configuration model for UAV ViT framework."""

    model_config = ConfigDict(extra="allow")

    experiment: ExperimentConfig
    paths: PathsConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig
    data: DataConfig
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @field_validator("model", mode="before")
    @classmethod
    def ensure_model_defaults(cls, value: Any) -> dict[str, Any]:
        """Ensure model config has required defaults."""
        if isinstance(value, dict):
            value.setdefault("custom_modules", [])
            value.setdefault("pretrained", True)
        return value


def validate_config(config_dict: dict[str, Any]) -> Config:
    """Validate and parse configuration dictionary.

    Args:
        config_dict: Raw configuration dictionary from YAML.

    Returns:
        Validated Config model instance.

    Raises:
        ValidationError: If configuration validation fails.
    """
    return Config.model_validate(config_dict)
