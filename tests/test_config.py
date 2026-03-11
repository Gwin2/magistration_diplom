from pathlib import Path

from uav_vit.config import load_yaml


def test_load_yaml_normalizes_label_maps(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  name: test
  seed: 1
paths:
  train_images: a
  val_images: b
  test_images: c
  train_annotations: d
  val_annotations: e
  test_annotations: f
  output_dir: runs/test
model:
  name: hf_auto
  checkpoint: foo/bar
  num_labels: 1
  id2label:
    "0": uav
train:
  device: cpu
  epochs: 1
  batch_size: 1
  learning_rate: 0.001
  weight_decay: 0
  num_workers: 0
eval:
  score_threshold: 0.1
  latency_warmup_iters: 1
  latency_iters: 1
data:
  processor_size: 800
""",
        encoding="utf-8",
    )
    cfg = load_yaml(config_path)
    assert cfg["model"]["id2label"] == {0: "uav"}
    assert cfg["model"]["label2id"] == {"uav": 0}
