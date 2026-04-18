from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package model checkpoint as TorchServe MAR archive."
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--model-name", required=True, help="TorchServe model name.")
    parser.add_argument("--version", default="1.0", help="Model version.")
    parser.add_argument("--export-path", default="model-store", help="Output directory for .mar.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing archive.")
    return parser


def run_command(command: list[str]) -> None:
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            "torch-model-archiver failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )
    if process.stdout.strip():
        print(process.stdout.strip())


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    export_path = Path(args.export_path).resolve()
    export_path.mkdir(parents=True, exist_ok=True)

    handler_path = Path("src/uav_vit/serving/torchserve_handler.py").resolve()
    if not handler_path.exists():
        raise FileNotFoundError(f"Handler file not found: {handler_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        packaged_config = tmp_path / "inference_config.yaml"
        shutil.copy2(config_path, packaged_config)

        command = [
            "torch-model-archiver",
            "--model-name",
            args.model_name,
            "--version",
            str(args.version),
            "--serialized-file",
            str(checkpoint_path),
            "--handler",
            str(handler_path),
            "--extra-files",
            str(packaged_config),
            "--export-path",
            str(export_path),
        ]
        if args.force:
            command.append("--force")

        run_command(command)

    mar_path = export_path / f"{args.model_name}.mar"
    print(f"Model archive ready: {mar_path}")


if __name__ == "__main__":
    main()
