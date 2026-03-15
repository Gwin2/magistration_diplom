from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from uav_vit.control.app import create_app

BASE_CONFIG = """\
experiment:
  name: demo_run
  seed: 42
paths:
  train_images: data/processed/demo/images/train
  val_images: data/processed/demo/images/val
  test_images: data/processed/demo/images/test
  train_annotations: data/processed/demo/annotations/instances_train.json
  val_annotations: data/processed/demo/annotations/instances_val.json
  test_annotations: data/processed/demo/annotations/instances_test.json
  output_dir: runs/demo_run
model:
  name: yolos_tiny
  checkpoint: hustvl/yolos-tiny
  num_labels: 1
  id2label:
    "0": uav
  label2id:
    uav: 0
train:
  device: cpu
  epochs: 1
  batch_size: 1
  learning_rate: 0.0001
  weight_decay: 0.0
  num_workers: 0
  grad_clip_norm: 1.0
  log_interval: 10
  mixed_precision: false
  eval_every_epoch: true
  checkpoint_metric: map
  checkpoint_mode: max
eval:
  score_threshold: 0.1
  latency_warmup_iters: 1
  latency_iters: 1
data:
  processor_size: 800
  normalize_boxes: false
mlflow:
  enabled: true
  tracking_uri: http://mlflow:5000
  experiment_name: uav-vit-thesis
  run_name: demo_run
"""


class DummyProcess:
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.pid = 4242
        self._running = True

    def poll(self) -> int | None:
        return None if self._running else 0

    def terminate(self) -> None:
        self._running = False

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        self._running = False
        return 0

    def kill(self) -> None:
        self._running = False


def seed_workspace(root: Path) -> None:
    (root / "configs" / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "src" / "custom_models").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed" / "demo").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "demo_run").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "experiments" / "demo.yaml").write_text(BASE_CONFIG, encoding="utf-8")
    (root / "src" / "custom_models" / "__init__.py").write_text(
        '"""User-defined models can be registered here."""\n',
        encoding="utf-8",
    )
    (root / "runs" / "demo_run" / "metrics.csv").write_text(
        "epoch,train_loss,map,map_50,map_75,mar_100,latency_ms,fps\n"
        "1,1.0,0.35,0.52,0.28,0.44,24.5,18.2\n",
        encoding="utf-8",
    )


def make_client(tmp_path: Path, monkeypatch) -> TestClient:  # type: ignore[no-untyped-def]
    seed_workspace(tmp_path)
    monkeypatch.setenv("UAV_CONTROL_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("UAV_CONTROL_STATE_DIR", str(tmp_path / "workspace_state"))
    monkeypatch.setattr(
        "uav_vit.control.mlops.TensorBoardManager.ensure_started",
        lambda self: {"running": False, "url": "/api/tensorboard/"},
    )
    monkeypatch.setattr(
        "uav_vit.control.mlops.TensorBoardManager.status",
        lambda self: {"running": False, "url": "/api/tensorboard/"},
    )
    monkeypatch.setattr("uav_vit.control.mlops.TorchServeBridge.ping", lambda self: False)
    monkeypatch.setattr(
        "uav_vit.control.mlops.TorchServeBridge.list_models",
        lambda self: {"available": False, "models": []},
    )
    monkeypatch.setattr("uav_vit.control.mlops.MlflowBridge.list_runs", lambda self, limit=200: [])
    monkeypatch.setattr(
        "uav_vit.control.mlops.MlflowBridge.apply_ui_metadata",
        lambda self, run_name, tags=None, rating=None, note=None: True,
    )
    return TestClient(create_app())


def test_dataset_upload_and_register(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with make_client(tmp_path, monkeypatch) as client:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zip_file:
            zip_file.writestr("frames/frame_001.txt", "demo")

        response = client.post(
            "/datasets/upload",
            data={
                "dataset_name": "Uploaded Demo",
                "description": "demo upload",
                "tags": "fog,night",
            },
            files={"file": ("demo.zip", archive.getvalue(), "application/zip")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["uploaded"] is True
        uploaded = next(item for item in payload["items"] if item["name"] == "Uploaded Demo")

        response = client.get(f"/datasets/{uploaded['id']}/download")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

        response = client.post(
            "/datasets/register",
            json={
                "name": "Registered Demo",
                "path": "data/processed/demo",
                "description": "existing path",
                "tags": ["processed"],
            },
        )
        assert response.status_code == 200
        assert any(item["name"] == "Registered Demo" for item in response.json()["items"])


def test_config_architecture_and_experiment_metadata(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    with make_client(tmp_path, monkeypatch) as client:
        response = client.put(
            "/configs/demo_copy", json={"config_yaml": BASE_CONFIG.replace("demo_run", "demo_copy")}
        )
        assert response.status_code == 200
        assert response.json()["name"] == "demo_copy"

        response = client.get("/configs/demo_copy")
        assert response.status_code == 200
        assert "experiment:" in response.json()["config_yaml"]

        response = client.post(
            "/architectures",
            json={
                "name": "my_custom_detector",
                "description": "custom detector",
                "tags": ["custom", "vit"],
                "config_yaml": BASE_CONFIG.replace("yolos_tiny", "my_custom_detector"),
                "source_code": (
                    "from uav_vit.models import ModelBundle, register_model\n\n"
                    "@register_model('my_custom_detector')\n"
                    "def build_my_custom_detector(config):\n"
                    "    return ModelBundle(\n"
                    "        model=None,\n"
                    "        image_processor=None,\n"
                    "        name='my_custom_detector',\n"
                    "    )\n"
                ),
            },
        )
        assert response.status_code == 200
        assert response.json()["id"] == "my_custom_detector"
        assert (tmp_path / "src" / "custom_models" / "my_custom_detector.py").exists()

        response = client.get("/experiments")
        assert response.status_code == 200
        assert any(item["run_name"] == "demo_run" for item in response.json()["items"])

        response = client.post(
            "/experiments/demo_run/metadata",
            json={"tags": ["best", "night"], "rating": 5, "note": "ready for serving"},
        )
        assert response.status_code == 200

        response = client.get("/experiments")
        experiment = next(
            item for item in response.json()["items"] if item["run_name"] == "demo_run"
        )
        assert experiment["rating"] == 5
        assert "best" in experiment["tags"]
        assert experiment["note"] == "ready for serving"


def test_train_job_lifecycle(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("uav_vit.control.workspace.subprocess.Popen", DummyProcess)

    with make_client(tmp_path, monkeypatch) as client:
        response = client.post("/jobs/train", json={"config_name": "demo"})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        response = client.get("/jobs")
        assert response.status_code == 200
        assert any(
            item["id"] == job_id and item["status"] == "running"
            for item in response.json()["items"]
        )

        response = client.get(f"/jobs/{job_id}/logs")
        assert response.status_code == 200

        response = client.post(f"/jobs/{job_id}/stop")
        assert response.status_code == 200
        assert response.json()["job"]["status"] == "stopped"
