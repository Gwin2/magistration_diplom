from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from uav_vit.config import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UAV ViT thesis framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a detection model")
    train_parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment YAML config"
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on val/test split")
    eval_parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment YAML config"
    )
    eval_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Optional checkpoint path"
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Dataset split for evaluation",
    )

    convert_parser = subparsers.add_parser(
        "convert-video", help="Convert annotated videos to COCO format"
    )
    convert_parser.add_argument("--video-dir", type=str, required=True)
    convert_parser.add_argument("--annotations-csv", type=str, required=True)
    convert_parser.add_argument("--output-dir", type=str, required=True)
    convert_parser.add_argument("--train-ratio", type=float, default=0.7)
    convert_parser.add_argument("--val-ratio", type=float, default=0.2)
    convert_parser.add_argument("--test-ratio", type=float, default=0.1)
    convert_parser.add_argument("--seed", type=int, default=42)
    convert_parser.add_argument("--normalized-boxes", action="store_true")
    convert_parser.add_argument("--image-format", type=str, default="jpg")

    summary_parser = subparsers.add_parser(
        "summarize",
        help="Aggregate run metrics and generate CSV/LaTeX tables",
    )
    summary_parser.add_argument("--runs-dir", type=str, default="runs")
    summary_parser.add_argument("--output-dir", type=str, default="reports")

    condition_parser = subparsers.add_parser(
        "analyze-conditions",
        help="Evaluate metrics separately for weather/quality/maneuver groups",
    )
    condition_parser.add_argument("--config", type=str, required=True)
    condition_parser.add_argument("--metadata-csv", type=str, required=True)
    condition_parser.add_argument("--column", type=str, required=True)
    condition_parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    condition_parser.add_argument("--checkpoint", type=str, default=None)

    return parser


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        from uav_vit.engine import train_from_config

        config = load_yaml(args.config)
        result = train_from_config(config)
        _print_json(result)
        return

    if args.command == "evaluate":
        from uav_vit.engine import evaluate_from_config

        config = load_yaml(args.config)
        result = evaluate_from_config(config, checkpoint_path=args.checkpoint, split=args.split)
        _print_json(result)
        return

    if args.command == "convert-video":
        from uav_vit.data.video_to_coco import VideoToCocoConfig, convert_video_annotations_to_coco

        convert_cfg = VideoToCocoConfig(
            video_dir=Path(args.video_dir),
            annotations_csv=Path(args.annotations_csv),
            output_dir=Path(args.output_dir),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            normalized_boxes=args.normalized_boxes,
            image_format=args.image_format,
        )
        stats = convert_video_annotations_to_coco(convert_cfg)
        _print_json(stats)
        return

    if args.command == "summarize":
        from uav_vit.analysis import summarize_runs

        summary = summarize_runs(runs_dir=args.runs_dir, output_dir=args.output_dir)
        _print_json(summary)
        return

    if args.command == "analyze-conditions":
        from uav_vit.analysis import evaluate_by_condition

        config = load_yaml(args.config)
        out_path = evaluate_by_condition(
            config=config,
            metadata_csv=args.metadata_csv,
            condition_column=args.column,
            split=args.split,
            checkpoint_path=args.checkpoint,
        )
        _print_json({"output_csv": str(out_path)})
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
