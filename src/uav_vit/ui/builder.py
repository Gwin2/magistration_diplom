"""
Core logic for the drag-and-drop neural network architecture constructor.
Includes layer definitions, compatibility rules, and validation logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Supported neural network layer types."""

    CONV2D = "Conv2D"
    MAXPOOL2D = "MaxPool2D"
    AVGPOOL2D = "AvgPool2D"
    BATCHNORM = "BatchNorm2D"
    DROPOUT = "Dropout"
    LINEAR = "Linear"
    FLATTEN = "Flatten"
    VIT_BLOCK = "ViTBlock"
    EMBEDDING = "Embedding"
    ATTENTION = "Attention"
    RESIDUAL = "ResidualBlock"
    LAYER_NORM = "LayerNorm"
    RELU = "ReLU"
    SOFTMAX = "Softmax"


class ActivationType(Enum):
    """Supported activation functions."""

    RELU = "ReLU"
    GELU = "GELU"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"
    SOFTMAX = "Softmax"
    LEAKY_RELU = "LeakyReLU"


@dataclass
class LayerConstraints:
    """Defines constraints and recommendations for a layer type."""

    allowed_predecessors: list[LayerType] = field(default_factory=list)
    allowed_successors: list[LayerType] = field(default_factory=list)
    recommended_position: str | None = None
    max_count: int | None = None
    requires_input_shape: bool = True
    output_changes_shape: bool = True
    description: str = ""
    efficiency_tips: list[str] = field(default_factory=list)


@dataclass
class LayerDefinition:
    """Complete definition of a neural network layer."""

    type: LayerType
    name: str
    params: dict[str, Any]
    activation: ActivationType | None = None
    constraints: LayerConstraints = field(default_factory=LayerConstraints)


LAYER_RULES: dict[LayerType, LayerConstraints] = {
    LayerType.CONV2D: LayerConstraints(
        allowed_predecessors=[
            LayerType.CONV2D,
            LayerType.BATCHNORM,
            LayerType.RELU,
            LayerType.EMBEDDING,
        ],
        allowed_successors=[
            LayerType.CONV2D,
            LayerType.BATCHNORM,
            LayerType.MAXPOOL2D,
            LayerType.AVGPOOL2D,
            LayerType.DROPOUT,
            LayerType.RELU,
            LayerType.LINEAR,
        ],
        recommended_position="early",
        description="2D Convolutional layer for feature extraction.",
        efficiency_tips=[
            "Use smaller kernel sizes (3x3) stacked instead of large kernels.",
            "Consider depthwise separable convolutions for mobile deployment.",
            "Follow with BatchNorm for better convergence.",
        ],
    ),
    LayerType.MAXPOOL2D: LayerConstraints(
        allowed_predecessors=[LayerType.CONV2D, LayerType.RELU, LayerType.BATCHNORM],
        allowed_successors=[LayerType.CONV2D, LayerType.MAXPOOL2D, LayerType.FLATTEN],
        recommended_position="early",
        description="Max pooling for spatial dimensionality reduction.",
        efficiency_tips=[
            "Use after activation functions for better feature selection.",
            "Consider using strided convolutions as an alternative.",
        ],
    ),
    LayerType.BATCHNORM: LayerConstraints(
        allowed_predecessors=[
            LayerType.CONV2D,
            LayerType.LINEAR,
            LayerType.MAXPOOL2D,
        ],
        allowed_successors=[LayerType.RELU, LayerType.CONV2D, LayerType.LINEAR],
        description="Batch normalization for training stability.",
        efficiency_tips=[
            "Place before activation function for standard ResNet architecture.",
            "Disable during inference or use eval mode.",
        ],
    ),
    LayerType.DROPOUT: LayerConstraints(
        allowed_predecessors=[LayerType.LINEAR, LayerType.CONV2D, LayerType.RELU],
        allowed_successors=[LayerType.LINEAR, LayerType.CONV2D],
        recommended_position="late",
        description="Regularization technique to prevent overfitting.",
        efficiency_tips=[
            "Use higher rates (0.5) for fully connected layers.",
            "Lower rates (0.1-0.3) for convolutional layers.",
            "Disable during inference.",
        ],
    ),
    LayerType.FLATTEN: LayerConstraints(
        allowed_predecessors=[
            LayerType.CONV2D,
            LayerType.MAXPOOL2D,
            LayerType.AVGPOOL2D,
            LayerType.BATCHNORM,
            LayerType.RELU,
        ],
        allowed_successors=[LayerType.LINEAR, LayerType.DROPOUT],
        requires_input_shape=True,
        output_changes_shape=True,
        description="Flattens multi-dimensional input to 1D.",
        efficiency_tips=["Ensure input dimensions are known before flattening."],
    ),
    LayerType.LINEAR: LayerConstraints(
        allowed_predecessors=[LayerType.FLATTEN, LayerType.LINEAR, LayerType.DROPOUT],
        allowed_successors=[
            LayerType.RELU,
            LayerType.DROPOUT,
            LayerType.LINEAR,
            LayerType.SOFTMAX,
        ],
        recommended_position="late",
        description="Fully connected layer.",
        efficiency_tips=[
            "Reduce size progressively in classifier head.",
            "Use global average pooling instead of Flatten+Linear when possible.",
        ],
    ),
    LayerType.VIT_BLOCK: LayerConstraints(
        allowed_predecessors=[LayerType.VIT_BLOCK, LayerType.EMBEDDING],
        allowed_successors=[LayerType.VIT_BLOCK, LayerType.LAYER_NORM, LayerType.LINEAR],
        recommended_position="middle",
        description="Vision Transformer block with self-attention.",
        efficiency_tips=[
            "Use multiple blocks (6-12) for deep feature extraction.",
            "Ensure embedding dimension matches across blocks.",
        ],
    ),
    LayerType.EMBEDDING: LayerConstraints(
        allowed_predecessors=[],
        allowed_successors=[LayerType.VIT_BLOCK, LayerType.LINEAR],
        requires_input_shape=True,
        description="Patch embedding for Vision Transformers.",
        efficiency_tips=[
            "Match patch size to image resolution appropriately.",
            "Position embeddings are crucial for ViT performance.",
        ],
    ),
    LayerType.RESIDUAL: LayerConstraints(
        allowed_predecessors=[LayerType.CONV2D, LayerType.BATCHNORM, LayerType.RELU],
        allowed_successors=[LayerType.CONV2D, LayerType.BATCHNORM, LayerType.RELU],
        description="Residual block for deep networks.",
        efficiency_tips=[
            "Use skip connections to enable deeper networks.",
            "Ensure matching dimensions or use projection shortcuts.",
        ],
    ),
    LayerType.RELU: LayerConstraints(
        allowed_predecessors=[
            LayerType.CONV2D,
            LayerType.LINEAR,
            LayerType.BATCHNORM,
        ],
        allowed_successors=[
            LayerType.CONV2D,
            LayerType.LINEAR,
            LayerType.MAXPOOL2D,
            LayerType.DROPOUT,
        ],
        description="Rectified Linear Unit activation.",
        efficiency_tips=["Most commonly used activation for hidden layers."],
    ),
    LayerType.SOFTMAX: LayerConstraints(
        allowed_predecessors=[LayerType.LINEAR],
        allowed_successors=[],
        recommended_position="late",
        description="Softmax activation for classification.",
        efficiency_tips=["Typically used as the final layer for classification."],
    ),
    LayerType.AVGPOOL2D: LayerConstraints(
        allowed_predecessors=[LayerType.CONV2D, LayerType.RELU, LayerType.BATCHNORM],
        allowed_successors=[LayerType.CONV2D, LayerType.FLATTEN, LayerType.LINEAR],
        recommended_position="early",
        description="Average pooling for spatial dimensionality reduction.",
        efficiency_tips=[
            "Provides smoother downsampling compared to max pooling.",
            "Often used in architectures like ResNet and Inception.",
        ],
    ),
    LayerType.ATTENTION: LayerConstraints(
        allowed_predecessors=[LayerType.EMBEDDING, LayerType.LAYER_NORM, LayerType.LINEAR],
        allowed_successors=[LayerType.LAYER_NORM, LayerType.LINEAR, LayerType.DROPOUT],
        recommended_position="middle",
        description="Self-attention mechanism for capturing long-range dependencies.",
        efficiency_tips=[
            "Use multi-head attention for better representation learning.",
            "Combine with LayerNorm for stable training.",
        ],
    ),
    LayerType.LAYER_NORM: LayerConstraints(
        allowed_predecessors=[LayerType.ATTENTION, LayerType.LINEAR, LayerType.VIT_BLOCK],
        allowed_successors=[LayerType.ATTENTION, LayerType.LINEAR, LayerType.VIT_BLOCK],
        description="Layer normalization for training stability.",
        efficiency_tips=[
            "Place before or after attention/linear layers depending on architecture style.",
            "Essential for Transformer-based models.",
        ],
    ),
    LayerType.VIT_BLOCK: LayerConstraints(
        allowed_predecessors=[LayerType.LAYER_NORM, LayerType.EMBEDDING],
        allowed_successors=[LayerType.LAYER_NORM, LayerType.LINEAR],
        recommended_position="middle",
        description="Vision Transformer block with self-attention and MLP.",
        efficiency_tips=[
            "Stack multiple ViT blocks for deeper models.",
            "Use skip connections for gradient flow.",
        ],
    ),
    LayerType.EMBEDDING: LayerConstraints(
        allowed_predecessors=[],
        allowed_successors=[LayerType.VIT_BLOCK, LayerType.LAYER_NORM, LayerType.ATTENTION],
        recommended_position="early",
        description="Embedding layer for converting tokens to vectors.",
        efficiency_tips=[
            "Ensure embedding dimension matches model hidden size.",
            "Consider positional embeddings for sequence data.",
        ],
    ),
    LayerType.RESIDUAL: LayerConstraints(
        allowed_predecessors=[LayerType.CONV2D, LayerType.BATCHNORM, LayerType.RELU],
        allowed_successors=[LayerType.CONV2D, LayerType.BATCHNORM, LayerType.RELU],
        recommended_position="middle",
        description="Residual block for deep network training.",
        efficiency_tips=[
            "Use skip connections to enable very deep networks.",
            "Combine with BatchNorm for best results.",
        ],
    ),
}

DEFAULT_ACTIVATIONS: dict[LayerType, ActivationType] = {
    LayerType.CONV2D: ActivationType.RELU,
    LayerType.LINEAR: ActivationType.RELU,
    LayerType.VIT_BLOCK: ActivationType.GELU,
}


@dataclass
class LayerNode:
    """Represents a node in the neural network architecture graph."""

    id: str
    layer_type: LayerType
    params: dict[str, Any]
    position: int
    activation: ActivationType | None = None
    connections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "id": self.id,
            "layer_type": self.layer_type.value,
            "params": self.params,
            "position": self.position,
            "activation": self.activation.value if self.activation else None,
            "connections": self.connections,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerNode:
        """Deserialize node from dictionary."""
        return cls(
            id=data["id"],
            layer_type=LayerType(data["layer_type"]),
            params=data["params"],
            position=data["position"],
            activation=(
                ActivationType(data["activation"]) if data.get("activation") else None
            ),
            connections=data.get("connections", []),
        )


class ArchitectureValidator:
    """Validates neural network architectures for compatibility and efficiency."""

    def __init__(self):
        self.rules = LAYER_RULES
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.recommendations: list[str] = []

    def validate_layer_sequence(
        self, layers: list[LayerNode]
    ) -> tuple[bool, list[str], list[str], list[str]]:
        """Validate a sequence of layers for compatibility."""
        self.errors = []
        self.warnings = []
        self.recommendations = []

        if not layers:
            self.errors.append("Architecture cannot be empty.")
            return False, self.errors, self.warnings, self.recommendations

        first_layer = layers[0]
        if first_layer.layer_type == LayerType.LINEAR:
            self.warnings.append(
                "Starting with Linear layer is unusual. Consider Embedding or Conv2D."
            )

        for i in range(len(layers) - 1):
            current = layers[i]
            next_layer = layers[i + 1]
            self._check_compatibility(current, next_layer, i)

        self._check_global_constraints(layers)

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings, self.recommendations

    def _check_compatibility(
        self, current: LayerNode, next_layer: LayerNode, index: int
    ) -> None:
        """Check compatibility between two consecutive layers."""
        current_rules = self.rules.get(current.layer_type)
        next_rules = self.rules.get(next_layer.layer_type)

        if not current_rules or not next_rules:
            msg = (
                f"No rules defined for transition: "
                f"{current.layer_type.value} -> {next_layer.layer_type.value}"
            )
            self.warnings.append(msg)
            return

        if (
            current_rules.allowed_successors
            and next_layer.layer_type not in current_rules.allowed_successors
        ):
            msg = (
                f"Layer {index}: {next_layer.layer_type.value} is not compatible "
                f"after {current.layer_type.value}."
            )
            self.errors.append(msg)
            rec_msg = (
                f"Consider adding an intermediate layer between "
                f"{current.layer_type.value} and {next_layer.layer_type.value}."
            )
            self.recommendations.append(rec_msg)

        if (
            next_rules.allowed_predecessors
            and current.layer_type not in next_rules.allowed_predecessors
        ):
            msg = (
                f"Layer {index + 1}: {current.layer_type.value} is not a typical "
                f"predecessor for {next_layer.layer_type.value}."
            )
            self.warnings.append(msg)

        if current_rules.efficiency_tips and index == 0:
            self.recommendations.extend(current_rules.efficiency_tips[:2])

    def _check_global_constraints(self, layers: list[LayerNode]) -> None:
        """Check global architecture constraints."""
        layer_counts: dict[LayerType, int] = {}

        for layer in layers:
            layer_counts[layer.layer_type] = layer_counts.get(layer.layer_type, 0) + 1

        if LayerType.FLATTEN in layer_counts and LayerType.LINEAR not in layer_counts:
            self.warnings.append(
                "Flatten layer detected without subsequent Linear layer."
            )

        last_layer = layers[-1]
        if last_layer.layer_type not in [
            LayerType.LINEAR,
            LayerType.SOFTMAX,
            LayerType.VIT_BLOCK,
        ]:
            self.warnings.append(
                "Architecture should typically end with Linear or classification layer."
            )

        for layer_type, count in layer_counts.items():
            rules = self.rules.get(layer_type)
            if rules and rules.max_count and count > rules.max_count:
                self.warnings.append(
                    f"Layer type {layer_type.value} appears {count} times, "
                    f"recommended maximum is {rules.max_count}."
                )

        for i, layer in enumerate(layers):
            rules = self.rules.get(layer.layer_type)
            if rules and rules.recommended_position:
                relative_pos = i / max(len(layers) - 1, 1)
                if rules.recommended_position == "early" and relative_pos > 0.5:
                    self.warnings.append(
                        f"{layer.layer_type.value} is typically used in early layers."
                    )
                elif rules.recommended_position == "late" and relative_pos < 0.5:
                    self.warnings.append(
                        f"{layer.layer_type.value} is typically used in later layers."
                    )


class NetworkBuilder:
    """Builder class for constructing neural network architectures."""

    def __init__(self):
        self.layers: list[LayerNode] = []
        self.validator = ArchitectureValidator()
        self._counter = 0

    def add_layer(
        self,
        layer_type: LayerType,
        params: dict[str, Any] | None = None,
        activation: ActivationType | None = None,
        position: int | None = None,
    ) -> LayerNode:
        """Add a layer to the architecture."""
        self._counter += 1
        layer_id = f"layer_{self._counter}"

        if params is None:
            params = self._get_default_params(layer_type)

        if activation is None:
            activation = DEFAULT_ACTIVATIONS.get(layer_type)

        if position is None:
            position = len(self.layers)

        node = LayerNode(
            id=layer_id,
            layer_type=layer_type,
            params=params,
            position=position,
            activation=activation,
        )

        self.layers.insert(position, node)
        self._update_positions()
        return node

    def remove_layer(self, layer_id: str) -> bool:
        """Remove a layer from the architecture."""
        for i, layer in enumerate(self.layers):
            if layer.id == layer_id:
                self.layers.pop(i)
                self._update_positions()
                return True
        return False

    def move_layer(self, layer_id: str, new_position: int) -> bool:
        """Move a layer to a new position."""
        layer_to_move = None
        for i, layer in enumerate(self.layers):
            if layer.id == layer_id:
                layer_to_move = self.layers.pop(i)
                break

        if layer_to_move is None:
            return False

        new_position = max(0, min(new_position, len(self.layers)))
        self.layers.insert(new_position, layer_to_move)
        self._update_positions()
        return True

    def validate(self) -> tuple[bool, list[str], list[str], list[str]]:
        """Validate the current architecture."""
        return self.validator.validate_layer_sequence(self.layers)

    def get_architecture_summary(self) -> dict[str, Any]:
        """Get a summary of the current architecture."""
        is_valid, errors, warnings, recommendations = self.validate()

        return {
            "valid": is_valid,
            "layer_count": len(self.layers),
            "layers": [layer.to_dict() for layer in self.layers],
            "errors": errors,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    def _update_positions(self) -> None:
        """Update position indices for all layers."""
        for i, layer in enumerate(self.layers):
            layer.position = i

    def _get_default_params(self, layer_type: LayerType) -> dict[str, Any]:
        """Get default parameters for a layer type."""
        defaults = {
            LayerType.CONV2D: {"in_channels": 3, "out_channels": 64, "kernel_size": 3},
            LayerType.MAXPOOL2D: {"kernel_size": 2, "stride": 2},
            LayerType.BATCHNORM: {"num_features": 64},
            LayerType.DROPOUT: {"p": 0.5},
            LayerType.LINEAR: {"in_features": 512, "out_features": 10},
            LayerType.FLATTEN: {},
            LayerType.VIT_BLOCK: {"dim": 768, "num_heads": 12},
            LayerType.EMBEDDING: {"img_size": 224, "patch_size": 16, "dim": 768},
        }
        return defaults.get(layer_type, {})

    def clear(self) -> None:
        """Clear all layers from the builder."""
        self.layers = []
        self._counter = 0

    def export_to_config(self) -> dict[str, Any]:
        """Export architecture to configuration dictionary."""
        summary = self.get_architecture_summary()
        return {
            "architecture": {
                "layers": summary["layers"],
                "valid": summary["valid"],
            },
            "validation": {
                "errors": summary["errors"],
                "warnings": summary["warnings"],
                "recommendations": summary["recommendations"],
            },
        }


# ============================================================================
# UI Helper Constants and Functions
# ============================================================================

LAYER_INFO: dict[str, dict[str, Any]] = {
    layer_type.value: {
        "name": layer_type.value,
        "description": LAYER_RULES[layer_type].description,
        "params": {
            "in_channels": "Number of input channels",
            "out_channels": "Number of output channels",
            "kernel_size": "Size of the convolving kernel",
            "stride": "Stride of the convolution",
            "padding": "Zero-padding added to both sides of input",
            "dropout_rate": "Probability of an element to be zeroed",
            "num_features": "Number of features for BatchNorm",
            "hidden_size": "Size of hidden dimension",
            "num_heads": "Number of attention heads",
        },
    }
    for layer_type in LayerType
}

POSITION_ADVICE: dict[str, dict[str, str]] = {
    layer_type.value: {
        "early": "Recommended at the beginning of the network for feature extraction.",
        "middle": "Works best in the middle layers for processing.",
        "late": "Typically used near the output for classification/regression.",
        "anywhere": "Can be placed anywhere in the architecture.",
    }
    for layer_type in LayerType
    if LAYER_RULES[layer_type].recommended_position
}

EFFICIENCY_TIPS: dict[str, list[str]] = {
    layer_type.value: LAYER_RULES[layer_type].efficiency_tips
    for layer_type in LayerType
    if LAYER_RULES[layer_type].efficiency_tips
}


def validate_layer_sequence(layers: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Validate a sequence of layers given as dictionaries.
    
    Args:
        layers: List of layer configurations with 'type' and 'params' keys.
        
    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    validator = ArchitectureValidator()
    issues = []
    
    for i, layer_config in enumerate(layers):
        layer_type_str = layer_config.get("type", "")
        try:
            layer_type = LayerType(layer_type_str)
        except ValueError:
            issues.append(f"Layer {i}: Unknown layer type '{layer_type_str}'")
            continue
        
        params = layer_config.get("params", {})
        node = LayerNode(
            id=f"layer_{i}",
            layer_type=layer_type,
            params=params,
            position=i,
        )
        
        if i > 0:
            prev_layer_type = LayerType(layers[i - 1]["type"])
            constraints = LAYER_RULES[layer_type]
            if constraints.allowed_predecessors and prev_layer_type not in constraints.allowed_predecessors:
                issues.append(
                    f"Layer {i} ({layer_type.value}): {prev_layer_type.value} is not recommended before {layer_type.value}"
                )
    
    return len(issues) == 0, issues


def get_compatibility_issues(layers: list[dict[str, Any]]) -> list[str]:
    """Get compatibility issues for a layer sequence.
    
    Args:
        layers: List of layer configurations.
        
    Returns:
        List of compatibility issue descriptions.
    """
    _, issues = validate_layer_sequence(layers)
    return issues


def build_model_from_layers(layers: list[dict[str, Any]]) -> Any:
    """Build a PyTorch model from layer configurations.
    
    Args:
        layers: List of layer configurations.
        
    Returns:
        A PyTorch nn.Sequential model.
    """
    import torch.nn as nn
    
    pytorch_layers = []
    
    for layer_config in layers:
        layer_type_str = layer_config.get("type", "")
        params = layer_config.get("params", {})
        
        try:
            layer_type = LayerType(layer_type_str)
        except ValueError:
            continue
        
        if layer_type == LayerType.CONV2D:
            pytorch_layers.append(
                nn.Conv2d(
                    in_channels=params.get("in_channels", 3),
                    out_channels=params.get("out_channels", 64),
                    kernel_size=params.get("kernel_size", 3),
                    stride=params.get("stride", 1),
                    padding=params.get("padding", 1),
                )
            )
        elif layer_type == LayerType.MAXPOOL2D:
            pytorch_layers.append(
                nn.MaxPool2d(
                    kernel_size=params.get("kernel_size", 2),
                    stride=params.get("stride", 2),
                )
            )
        elif layer_type == LayerType.BATCHNORM:
            pytorch_layers.append(
                nn.BatchNorm2d(num_features=params.get("num_features", 64))
            )
        elif layer_type == LayerType.RELU:
            pytorch_layers.append(nn.ReLU())
        elif layer_type == LayerType.DROPOUT:
            pytorch_layers.append(nn.Dropout(p=params.get("dropout_rate", 0.5)))
        elif layer_type == LayerType.FLATTEN:
            pytorch_layers.append(nn.Flatten())
        elif layer_type == LayerType.LINEAR:
            pytorch_layers.append(
                nn.Linear(
                    in_features=params.get("in_features", 512),
                    out_features=params.get("out_features", 10),
                )
            )
        elif layer_type == LayerType.SOFTMAX:
            pytorch_layers.append(nn.Softmax(dim=1))
        elif layer_type == LayerType.AVGPOOL2D:
            pytorch_layers.append(
                nn.AvgPool2d(
                    kernel_size=params.get("kernel_size", 2),
                    stride=params.get("stride", 2),
                )
            )
        elif layer_type == LayerType.LAYER_NORM:
            pytorch_layers.append(
                nn.LayerNorm(normalized_shape=params.get("normalized_shape", 512))
            )
        elif layer_type == LayerType.ATTENTION:
            # Simplified attention placeholder
            pytorch_layers.append(nn.Identity())
    
    return nn.Sequential(*pytorch_layers)


def get_architecture_examples() -> dict[str, list[dict[str, Any]]]:
    """Return example architectures.
    
    Returns:
        Dictionary of example name to layer configuration list.
    """
    return {
        "Simple CNN": [
            {"type": "Conv2D", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3}},
            {"type": "ReLU", "params": {}},
            {"type": "MaxPool2D", "params": {"kernel_size": 2}},
            {"type": "Conv2D", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3}},
            {"type": "ReLU", "params": {}},
            {"type": "Flatten", "params": {}},
            {"type": "Linear", "params": {"in_features": 64, "out_features": 10}},
            {"type": "Softmax", "params": {}},
        ],
        "ViT-like": [
            {"type": "Embedding", "params": {"input_dim": 768, "embed_dim": 768}},
            {"type": "LayerNorm", "params": {"normalized_shape": 768}},
            {"type": "Attention", "params": {"embed_dim": 768, "num_heads": 12}},
            {"type": "LayerNorm", "params": {"normalized_shape": 768}},
            {"type": "Linear", "params": {"in_features": 768, "out_features": 10}},
        ],
        "ResNet Block": [
            {"type": "Conv2D", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3}},
            {"type": "BatchNorm2D", "params": {"num_features": 64}},
            {"type": "ReLU", "params": {}},
            {"type": "Conv2D", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3}},
            {"type": "BatchNorm2D", "params": {"num_features": 64}},
            {"type": "ReLU", "params": {}},
        ],
    }
