"""
Tests for the neural network architecture builder UI module.
"""

from uav_vit.ui.builder import (
    LAYER_RULES,
    ActivationType,
    ArchitectureValidator,
    LayerNode,
    LayerType,
    NetworkBuilder,
)


class TestLayerType:
    """Test LayerType enum."""

    def test_layer_types_exist(self):
        """Test that common layer types are defined."""
        assert LayerType.CONV2D.value == "Conv2D"
        assert LayerType.LINEAR.value == "Linear"
        assert LayerType.MAXPOOL2D.value == "MaxPool2D"
        assert LayerType.BATCHNORM.value == "BatchNorm2D"
        assert LayerType.DROPOUT.value == "Dropout"
        assert LayerType.FLATTEN.value == "Flatten"
        assert LayerType.VIT_BLOCK.value == "ViTBlock"
        assert LayerType.EMBEDDING.value == "Embedding"


class TestActivationType:
    """Test ActivationType enum."""

    def test_activation_types_exist(self):
        """Test that common activation types are defined."""
        assert ActivationType.RELU.value == "ReLU"
        assert ActivationType.GELU.value == "GELU"
        assert ActivationType.SOFTMAX.value == "Softmax"


class TestLayerNode:
    """Test LayerNode dataclass."""

    def test_create_layer_node(self):
        """Test creating a layer node."""
        node = LayerNode(
            id="test_1",
            layer_type=LayerType.CONV2D,
            params={"in_channels": 3, "out_channels": 64, "kernel_size": 3},
            position=0,
            activation=ActivationType.RELU,
        )

        assert node.id == "test_1"
        assert node.layer_type == LayerType.CONV2D
        assert node.position == 0
        assert node.activation == ActivationType.RELU
        assert node.params["in_channels"] == 3

    def test_layer_node_to_dict(self):
        """Test serializing layer node to dictionary."""
        node = LayerNode(
            id="test_2",
            layer_type=LayerType.LINEAR,
            params={"in_features": 512, "out_features": 10},
            position=1,
            activation=ActivationType.RELU,
        )

        data = node.to_dict()

        assert data["id"] == "test_2"
        assert data["layer_type"] == "Linear"
        assert data["position"] == 1
        assert data["activation"] == "ReLU"
        assert data["params"]["in_features"] == 512

    def test_layer_node_from_dict(self):
        """Test deserializing layer node from dictionary."""
        data = {
            "id": "test_3",
            "layer_type": "MaxPool2D",
            "params": {"kernel_size": 2, "stride": 2},
            "position": 2,
            "activation": None,
            "connections": [],
        }

        node = LayerNode.from_dict(data)

        assert node.id == "test_3"
        assert node.layer_type == LayerType.MAXPOOL2D
        assert node.position == 2
        assert node.activation is None
        assert node.params["kernel_size"] == 2


class TestArchitectureValidator:
    """Test ArchitectureValidator class."""

    def test_validate_empty_architecture(self):
        """Test validation of empty architecture."""
        validator = ArchitectureValidator()
        is_valid, errors, warnings, recommendations = validator.validate_layer_sequence([])

        assert is_valid is False
        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_validate_valid_cnn(self):
        """Test validation of a valid CNN architecture."""
        validator = ArchitectureValidator()
        layers = [
            LayerNode(
                id="l1",
                layer_type=LayerType.CONV2D,
                params={"in_channels": 3, "out_channels": 32},
                position=0,
            ),
            LayerNode(
                id="l2",
                layer_type=LayerType.MAXPOOL2D,
                params={"kernel_size": 2},
                position=1,
            ),
            LayerNode(
                id="l3",
                layer_type=LayerType.FLATTEN,
                params={},
                position=2,
            ),
            LayerNode(
                id="l4",
                layer_type=LayerType.LINEAR,
                params={"in_features": 512, "out_features": 10},
                position=3,
            ),
        ]

        is_valid, errors, warnings, recommendations = validator.validate_layer_sequence(layers)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_sequence(self):
        """Test validation detects invalid layer sequences."""
        validator = ArchitectureValidator()
        layers = [
            LayerNode(
                id="l1",
                layer_type=LayerType.FLATTEN,
                params={},
                position=0,
            ),
            LayerNode(
                id="l2",
                layer_type=LayerType.MAXPOOL2D,
                params={"kernel_size": 2},
                position=1,
            ),
        ]

        is_valid, errors, warnings, _ = validator.validate_layer_sequence(layers)

        # Flatten followed by MaxPool should be invalid
        assert is_valid is False or len(warnings) > 0


class TestNetworkBuilder:
    """Test NetworkBuilder class."""

    def test_create_builder(self):
        """Test creating a network builder."""
        builder = NetworkBuilder()
        assert len(builder.layers) == 0
        assert builder._counter == 0

    def test_add_layer(self):
        """Test adding layers to builder."""
        builder = NetworkBuilder()

        layer1 = builder.add_layer(
            layer_type=LayerType.CONV2D,
            params={"in_channels": 3, "out_channels": 64},
        )

        assert len(builder.layers) == 1
        assert layer1.id == "layer_1"
        assert layer1.layer_type == LayerType.CONV2D
        assert layer1.position == 0

    def test_add_multiple_layers(self):
        """Test adding multiple layers."""
        builder = NetworkBuilder()

        builder.add_layer(LayerType.CONV2D, {"in_channels": 3, "out_channels": 32})
        builder.add_layer(LayerType.BATCHNORM, {"num_features": 32})
        builder.add_layer(LayerType.MAXPOOL2D, {"kernel_size": 2})

        assert len(builder.layers) == 3
        assert builder.layers[0].layer_type == LayerType.CONV2D
        assert builder.layers[1].layer_type == LayerType.BATCHNORM
        assert builder.layers[2].layer_type == LayerType.MAXPOOL2D

    def test_remove_layer(self):
        """Test removing a layer."""
        builder = NetworkBuilder()

        layer1 = builder.add_layer(LayerType.CONV2D, {})
        builder.add_layer(LayerType.BATCHNORM, {})
        builder.add_layer(LayerType.MAXPOOL2D, {})

        assert len(builder.layers) == 3

        success = builder.remove_layer(layer1.id)
        assert success is True
        assert len(builder.layers) == 2
        assert builder.layers[0].layer_type == LayerType.BATCHNORM

    def test_move_layer(self):
        """Test moving a layer to new position."""
        builder = NetworkBuilder()

        builder.add_layer(LayerType.CONV2D, {})
        builder.add_layer(LayerType.BATCHNORM, {})
        layer3 = builder.add_layer(LayerType.MAXPOOL2D, {})

        # Move last layer to first position
        success = builder.move_layer(layer3.id, 0)

        assert success is True
        assert builder.layers[0].layer_type == LayerType.MAXPOOL2D
        assert builder.layers[1].layer_type == LayerType.CONV2D

    def test_validate_architecture(self):
        """Test validating built architecture."""
        builder = NetworkBuilder()

        builder.add_layer(LayerType.CONV2D, {"in_channels": 3, "out_channels": 32})
        builder.add_layer(LayerType.MAXPOOL2D, {"kernel_size": 2})
        builder.add_layer(LayerType.FLATTEN, {})
        builder.add_layer(LayerType.LINEAR, {"in_features": 512, "out_features": 10})

        is_valid, errors, warnings, recommendations = builder.validate()

        assert is_valid is True
        assert len(errors) == 0

    def test_get_architecture_summary(self):
        """Test getting architecture summary."""
        builder = NetworkBuilder()

        builder.add_layer(LayerType.CONV2D, {"in_channels": 3, "out_channels": 64})
        builder.add_layer(LayerType.LINEAR, {"in_features": 512, "out_features": 10})

        summary = builder.get_architecture_summary()

        assert summary["valid"] is True
        assert summary["layer_count"] == 2
        assert len(summary["layers"]) == 2
        assert "errors" in summary
        assert "warnings" in summary
        assert "recommendations" in summary

    def test_clear_builder(self):
        """Test clearing the builder."""
        builder = NetworkBuilder()

        builder.add_layer(LayerType.CONV2D, {})
        builder.add_layer(LayerType.LINEAR, {})

        assert len(builder.layers) == 2

        builder.clear()

        assert len(builder.layers) == 0
        assert builder._counter == 0

    def test_export_to_config(self):
        """Test exporting architecture to config."""
        builder = NetworkBuilder()

        builder.add_layer(LayerType.CONV2D, {"in_channels": 3, "out_channels": 64})
        builder.add_layer(LayerType.LINEAR, {"in_features": 512, "out_features": 10})

        config = builder.export_to_config()

        assert "architecture" in config
        assert "validation" in config
        assert config["architecture"]["valid"] is True
        assert len(config["architecture"]["layers"]) == 2


class TestLayerRules:
    """Test layer compatibility rules."""

    def test_rules_defined_for_all_types(self):
        """Test that rules are defined for all layer types."""
        for layer_type in LayerType:
            # Not all layer types need rules, but common ones should have them
            if layer_type in [
                LayerType.CONV2D,
                LayerType.LINEAR,
                LayerType.MAXPOOL2D,
                LayerType.BATCHNORM,
                LayerType.DROPOUT,
                LayerType.FLATTEN,
            ]:
                assert layer_type in LAYER_RULES

    def test_conv2d_rules(self):
        """Test Conv2D layer rules."""
        rules = LAYER_RULES[LayerType.CONV2D]

        assert LayerType.BATCHNORM in rules.allowed_successors
        assert LayerType.MAXPOOL2D in rules.allowed_successors
        assert rules.recommended_position == "early"
        assert len(rules.efficiency_tips) > 0

    def test_linear_rules(self):
        """Test Linear layer rules."""
        rules = LAYER_RULES[LayerType.LINEAR]

        assert LayerType.FLATTEN in rules.allowed_predecessors
        assert rules.recommended_position == "late"


class TestIntegration:
    """Integration tests for the builder module."""

    def test_build_simple_cnn(self):
        """Test building a simple CNN architecture."""
        builder = NetworkBuilder()

        # Build a simple CNN
        builder.add_layer(
            LayerType.CONV2D,
            {"in_channels": 3, "out_channels": 32, "kernel_size": 3},
        )
        builder.add_layer(LayerType.BATCHNORM, {"num_features": 32})
        builder.add_layer(LayerType.RELU, {})
        builder.add_layer(LayerType.MAXPOOL2D, {"kernel_size": 2, "stride": 2})

        builder.add_layer(
            LayerType.CONV2D,
            {"in_channels": 32, "out_channels": 64, "kernel_size": 3},
        )
        builder.add_layer(LayerType.BATCHNORM, {"num_features": 64})
        builder.add_layer(LayerType.RELU, {})
        builder.add_layer(LayerType.MAXPOOL2D, {"kernel_size": 2, "stride": 2})

        builder.add_layer(LayerType.FLATTEN, {})
        builder.add_layer(LayerType.DROPOUT, {"p": 0.5})
        builder.add_layer(LayerType.LINEAR, {"in_features": 64 * 7 * 7, "out_features": 128})
        builder.add_layer(LayerType.RELU, {})
        builder.add_layer(LayerType.DROPOUT, {"p": 0.5})
        builder.add_layer(LayerType.LINEAR, {"in_features": 128, "out_features": 10})

        summary = builder.get_architecture_summary()

        assert summary["layer_count"] == 14
        # Should be valid or have only warnings
        assert summary["valid"] or len(summary["errors"]) == 0

    def test_build_vit_like_architecture(self):
        """Test building a ViT-like architecture."""
        builder = NetworkBuilder()

        builder.add_layer(
            LayerType.EMBEDDING,
            {"img_size": 224, "patch_size": 16, "dim": 768},
        )
        builder.add_layer(LayerType.VIT_BLOCK, {"dim": 768, "num_heads": 12})
        builder.add_layer(LayerType.VIT_BLOCK, {"dim": 768, "num_heads": 12})
        builder.add_layer(LayerType.VIT_BLOCK, {"dim": 768, "num_heads": 12})
        builder.add_layer(LayerType.LINEAR, {"in_features": 768, "out_features": 10})

        summary = builder.get_architecture_summary()

        assert summary["layer_count"] == 5
        assert summary["valid"] or len(summary["errors"]) == 0
