from pathlib import Path

import pandas as pd

from uav_vit.analysis.summarize import summarize_runs


def test_summarize_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "exp1"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "map": 0.2,
                "map_50": 0.3,
                "map_75": 0.1,
                "mar_100": 0.4,
                "latency_ms": 10.0,
                "fps": 8.0,
            },
            {
                "epoch": 2,
                "train_loss": 0.8,
                "map": 0.25,
                "map_50": 0.35,
                "map_75": 0.12,
                "mar_100": 0.45,
                "latency_ms": 12.0,
                "fps": 7.2,
            },
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)

    output = summarize_runs(runs_dir=tmp_path / "runs", output_dir=tmp_path / "reports")
    assert output["num_runs"] == 1
    assert (tmp_path / "reports" / "summary.csv").exists()
    assert (tmp_path / "reports" / "summary.tex").exists()
