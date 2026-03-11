from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def summarize_runs(runs_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    runs_path = Path(runs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for metrics_file in runs_path.glob("*/metrics.csv"):
        run_dir = metrics_file.parent
        df = pd.read_csv(metrics_file)
        if df.empty:
            continue

        best_idx = df["map"].idxmax()
        best_row = df.loc[best_idx]
        latency_ms = float(best_row.get("latency_ms", 0.0))
        map50 = float(best_row.get("map_50", 0.0))
        efficiency = map50 / latency_ms if latency_ms > 0 else 0.0

        rows.append(
            {
                "run_name": run_dir.name,
                "best_epoch": int(best_row["epoch"]),
                "map": float(best_row.get("map", 0.0)),
                "map_50": map50,
                "map_75": float(best_row.get("map_75", 0.0)),
                "mar_100": float(best_row.get("mar_100", 0.0)),
                "latency_ms": latency_ms,
                "fps": float(best_row.get("fps", 0.0)),
                "efficiency_map50_per_ms": efficiency,
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "run_name",
                "best_epoch",
                "map",
                "map_50",
                "map_75",
                "mar_100",
                "latency_ms",
                "fps",
                "efficiency_map50_per_ms",
            ]
        )
    else:
        summary_df = summary_df.sort_values(by="map_50", ascending=False).reset_index(drop=True)

    summary_csv = output_path / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_latex = output_path / "summary.tex"
    with summary_latex.open("w", encoding="utf-8") as file:
        file.write(summary_df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))

    return {
        "num_runs": int(len(summary_df)),
        "summary_csv": str(summary_csv),
        "summary_tex": str(summary_latex),
    }
