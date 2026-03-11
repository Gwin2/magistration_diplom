from uav_vit.integrations.mlflow_logger import _flatten_dict, mlflow_run


def test_flatten_dict_nested_keys() -> None:
    flattened = _flatten_dict({"a": {"b": 1}, "c": True, "d": None})
    assert flattened["a.b"] == 1
    assert flattened["c"] is True
    assert flattened["d"] == "null"


def test_mlflow_run_disabled_returns_none(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    config = {
        "experiment": {"name": "exp"},
        "model": {"name": "model"},
        "paths": {},
    }
    with mlflow_run(config, phase="train") as mlflow:
        assert mlflow is None
