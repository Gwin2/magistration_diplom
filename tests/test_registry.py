from uav_vit.models.registry import MODEL_REGISTRY, register_model


def test_builtin_models_registered() -> None:
    assert "yolos_tiny" in MODEL_REGISTRY
    assert "detr_resnet50" in MODEL_REGISTRY
    assert "hf_auto" in MODEL_REGISTRY


def test_register_model_decorator() -> None:
    name = "tmp_test_model"
    if name in MODEL_REGISTRY:
        del MODEL_REGISTRY[name]

    @register_model(name)
    def _builder(config):  # type: ignore[no-untyped-def]
        return None

    assert name in MODEL_REGISTRY
    assert MODEL_REGISTRY[name] is _builder
