from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class RunRow:
    run_name: str
    epoch: float
    train_loss: float
    map_value: float
    map50: float
    map75: float
    mar100: float
    fps: float
    latency_ms: float
    best_map50: float


def parse_runs(runs_dir: Path) -> list[RunRow]:
    rows: list[RunRow] = []
    for metrics_file in sorted(runs_dir.glob("*/metrics.csv")):
        run_name = metrics_file.parent.name
        try:
            frame = pd.read_csv(metrics_file)
        except Exception:
            continue
        if frame.empty:
            continue
        last = frame.iloc[-1]
        best_map50 = float(frame["map_50"].max()) if "map_50" in frame.columns else 0.0
        rows.append(
            RunRow(
                run_name=run_name,
                epoch=float(last.get("epoch", 0.0)),
                train_loss=float(last.get("train_loss", 0.0)),
                map_value=float(last.get("map", 0.0)),
                map50=float(last.get("map_50", 0.0)),
                map75=float(last.get("map_75", 0.0)),
                mar100=float(last.get("mar_100", 0.0)),
                fps=float(last.get("fps", 0.0)),
                latency_ms=float(last.get("latency_ms", 0.0)),
                best_map50=best_map50,
            )
        )
    return rows


def _create_gauges(gauge_cls: Any) -> dict[str, Any]:
    return {
        "train_loss": gauge_cls(
            "uav_run_train_loss_last", "Latest training loss from run metrics.csv", ["run_name"]
        ),
        "map": gauge_cls(
            "uav_run_map_last", "Latest validation mAP from run metrics.csv", ["run_name"]
        ),
        "map50": gauge_cls(
            "uav_run_map_50_last", "Latest validation mAP@50 from run metrics.csv", ["run_name"]
        ),
        "map75": gauge_cls(
            "uav_run_map_75_last", "Latest validation mAP@75 from run metrics.csv", ["run_name"]
        ),
        "mar100": gauge_cls(
            "uav_run_mar_100_last", "Latest validation mAR@100 from run metrics.csv", ["run_name"]
        ),
        "fps": gauge_cls("uav_run_fps_last", "Latest FPS from run metrics.csv", ["run_name"]),
        "latency": gauge_cls(
            "uav_run_latency_ms_last", "Latest latency in ms from run metrics.csv", ["run_name"]
        ),
        "epoch": gauge_cls("uav_run_epoch_last", "Latest epoch from run metrics.csv", ["run_name"]),
        "best_map50": gauge_cls(
            "uav_run_map_50_best", "Best validation mAP@50 from run metrics.csv", ["run_name"]
        ),
        "last_scrape_ts": gauge_cls(
            "uav_run_exporter_last_scrape_timestamp", "Last scrape unix timestamp"
        ),
        "run_count": gauge_cls(
            "uav_run_exporter_runs_total", "Number of runs discovered in runs directory"
        ),
    }


def export_runs(runs_dir: Path, gauges: dict[str, Any]) -> None:
    rows = parse_runs(runs_dir)
    gauges["run_count"].set(float(len(rows)))
    for row in rows:
        label = {"run_name": row.run_name}
        gauges["epoch"].labels(**label).set(row.epoch)
        gauges["train_loss"].labels(**label).set(row.train_loss)
        gauges["map"].labels(**label).set(row.map_value)
        gauges["map50"].labels(**label).set(row.map50)
        gauges["map75"].labels(**label).set(row.map75)
        gauges["mar100"].labels(**label).set(row.mar100)
        gauges["fps"].labels(**label).set(row.fps)
        gauges["latency"].labels(**label).set(row.latency_ms)
        gauges["best_map50"].labels(**label).set(row.best_map50)
    gauges["last_scrape_ts"].set(time.time())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prometheus exporter for UAV run metrics.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--port", type=int, default=9108)
    parser.add_argument("--interval", type=int, default=15)
    return parser


def main() -> None:
    try:
        from prometheus_client import Gauge, start_http_server
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("prometheus_client is required for run metrics exporter.") from exc

    args = build_parser().parse_args()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    gauges = _create_gauges(Gauge)
    start_http_server(args.port)
    print(f"[exporter] Serving /metrics on 0.0.0.0:{args.port}, runs_dir={args.runs_dir}")

    while True:
        export_runs(args.runs_dir, gauges)
        time.sleep(max(args.interval, 1))


if __name__ == "__main__":
    main()
