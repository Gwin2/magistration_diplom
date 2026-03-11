from pathlib import Path

import pandas as pd

from uav_vit.monitoring.prometheus_push import build_push_config
from uav_vit.monitoring.run_metrics_exporter import parse_runs


def test_parse_runs_extracts_latest_and_best(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "exp1"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "map": 0.1,
                "map_50": 0.2,
                "map_75": 0.05,
                "mar_100": 0.3,
                "latency_ms": 12,
                "fps": 8,
            },
            {
                "epoch": 2,
                "train_loss": 0.8,
                "map": 0.15,
                "map_50": 0.25,
                "map_75": 0.07,
                "mar_100": 0.35,
                "latency_ms": 11,
                "fps": 9,
            },
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)

    rows = parse_runs(tmp_path / "runs")
    assert len(rows) == 1
    row = rows[0]
    assert row.run_name == "exp1"
    assert row.epoch == 2
    assert row.map50 == 0.25
    assert row.best_map50 == 0.25


def test_build_push_config_uses_env_when_enabled_absent(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    cfg = {
        "experiment": {"name": "exp"},
        "model": {"name": "model"},
        "paths": {},
        "monitoring": {},
    }
    push_cfg = build_push_config(cfg, phase="train")
    assert push_cfg is not None
    assert push_cfg.gateway_url == "http://pushgateway:9091"
