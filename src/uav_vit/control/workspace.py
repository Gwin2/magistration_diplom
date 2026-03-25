from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from uav_vit.config import load_yaml
from uav_vit.control.state import ControlStateStore, JobRecord, slugify, utc_now
from uav_vit.models.registry import MODEL_REGISTRY


def _normalize_tags(raw_tags: list[str] | None) -> list[str]:
    return sorted({tag.strip() for tag in (raw_tags or []) if tag and tag.strip()})


def _raw_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} must contain a mapping.")
    return payload


def _build_default_custom_source(model_name: str, checkpoint: str) -> str:
    return (
        "from __future__ import annotations\n\n"
        "from transformers import AutoImageProcessor, AutoModelForObjectDetection\n\n"
        "from uav_vit.models import ModelBundle, register_model\n\n\n"
        f'@register_model("{model_name}")\n'
        f"def build_{model_name}(config: dict) -> ModelBundle:\n"
        '    checkpoint = config["model"].get("checkpoint") or '
        f'"{checkpoint}"\n'
        "    model = AutoModelForObjectDetection.from_pretrained(\n"
        "        checkpoint,\n"
        "        ignore_mismatched_sizes=True,\n"
        '        num_labels=int(config["model"]["num_labels"]),\n'
        '        id2label={int(k): v for k, v in config["model"]["id2label"].items()},\n'
        '        label2id={str(k): int(v) for k, v in config["model"]["label2id"].items()},\n'
        "    )\n"
        "    processor = AutoImageProcessor.from_pretrained(checkpoint)\n"
        f'    return ModelBundle(model=model, image_processor=processor, name="{model_name}")\n'
    )


class WorkspaceService:
    def __init__(self, store: ControlStateStore) -> None:
        self.store = store
        self.processes: dict[str, subprocess.Popen[str]] = {}

    def list_configs(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for config_path in sorted(self.store.configs_dir.glob("*.yaml")):
            try:
                config = load_yaml(config_path)
            except Exception as exc:
                rows.append(
                    {
                        "name": config_path.stem,
                        "path": str(config_path.relative_to(self.store.workspace_root)),
                        "valid": False,
                        "error": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "name": config_path.stem,
                    "path": str(config_path.relative_to(self.store.workspace_root)),
                    "valid": True,
                    "experiment_name": str(config["experiment"]["name"]),
                    "model_name": str(config["model"]["name"]),
                    "output_dir": str(config["paths"]["output_dir"]),
                    "epochs": int(config["train"]["epochs"]),
                    "batch_size": int(config["train"]["batch_size"]),
                }
            )
        return rows

    def load_config(self, config_name: str) -> dict[str, Any]:
        config_path = self.store.configs_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config '{config_name}' not found.")
        return _raw_yaml(config_path)

    def save_config(self, config_name: str, config: dict[str, Any]) -> dict[str, Any]:
        normalized_name = slugify(config_name)
        config_path = self.store.configs_dir / f"{normalized_name}.yaml"
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=False, sort_keys=False),
            encoding="utf-8",
        )
        return {
            "name": normalized_name,
            "path": str(config_path.relative_to(self.store.workspace_root)),
        }

    def list_datasets(self) -> list[dict[str, Any]]:
        metadata = self.store.load_metadata().get("datasets", {})
        dataset_paths: dict[str, Path] = {}
        for dataset_dir in self.store.discover_dataset_directories():
            dataset_id = self.store.dataset_id_for_path(dataset_dir)
            dataset_paths[dataset_id] = dataset_dir

        for dataset_id, meta in metadata.items():
            raw_path = meta.get("path")
            if not raw_path:
                continue
            candidate = (self.store.workspace_root / raw_path).resolve()
            if candidate.exists() and candidate.is_dir():
                dataset_paths.setdefault(dataset_id, candidate)

        rows: list[dict[str, Any]] = []
        for dataset_id, dataset_dir in dataset_paths.items():
            stats = self.store.file_stats(dataset_dir)
            meta = metadata.get(dataset_id, {})
            rows.append(
                {
                    "id": dataset_id,
                    "name": meta.get("name", dataset_dir.name),
                    "path": str(dataset_dir.relative_to(self.store.workspace_root)),
                    "description": meta.get("description", ""),
                    "tags": meta.get("tags", []),
                    "file_count": stats["file_count"],
                    "size_bytes": stats["size_bytes"],
                    "updated_at": meta.get("updated_at"),
                }
            )
        rows.sort(key=lambda item: item["name"])
        return rows

    def save_custom_architecture(
        self,
        name: str,
        config: dict[str, Any],
        source_code: str | None,
        description: str = "",
        tags: list[str] | None = None,
        blueprint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        slug = slugify(name)
        self.store.custom_models_dir.mkdir(parents=True, exist_ok=True)
        init_path = self.store.custom_models_dir / "__init__.py"
        if not init_path.exists():
            init_path.write_text(
                '"""User-defined models can be registered here."""\n',
                encoding="utf-8",
            )
        source_path = self.store.custom_models_dir / f"{slug}.py"
        checkpoint = str(config.get("model", {}).get("checkpoint") or "facebook/detr-resnet-50")
        source = source_code or _build_default_custom_source(slug, checkpoint)
        source_path.write_text(source, encoding="utf-8")

        config_copy = dict(config)
        config_copy.setdefault("model", {})
        config_copy["model"]["name"] = slug
        config_copy["model"]["custom_modules"] = [f"custom_models.{slug}"]
        config_path = self.store.configs_dir / f"{slug}.yaml"
        config_path.write_text(
            yaml.safe_dump(config_copy, allow_unicode=False, sort_keys=False),
            encoding="utf-8",
        )

        self.store.update_metadata_entry(
            "architectures",
            slug,
            {
                "name": name,
                "description": description,
                "tags": _normalize_tags(tags),
                "blueprint": blueprint,
                "config_path": str(config_path.relative_to(self.store.workspace_root)),
                "source_path": str(source_path.relative_to(self.store.workspace_root)),
                "updated_at": utc_now(),
            },
        )
        return self.get_architecture(slug)

    def list_architectures(self) -> list[dict[str, Any]]:
        metadata = self.store.load_metadata().get("architectures", {})
        rows: list[dict[str, Any]] = []

        for model_name in sorted(MODEL_REGISTRY):
            meta = metadata.get(model_name, {})
            rows.append(
                {
                    "id": model_name,
                    "name": meta.get("name", model_name),
                    "kind": "builtin"
                    if model_name in {"yolos_tiny", "detr_resnet50", "hf_auto"}
                    else "custom",
                    "description": meta.get("description", ""),
                    "tags": meta.get("tags", []),
                    "has_blueprint": bool(meta.get("blueprint")),
                    "config_path": meta.get("config_path"),
                    "source_path": meta.get("source_path"),
                }
            )

        for source_path in sorted(self.store.custom_models_dir.glob("*.py")):
            if source_path.name == "__init__.py":
                continue
            slug = source_path.stem
            if any(row["id"] == slug for row in rows):
                continue
            meta = metadata.get(slug, {})
            rows.append(
                {
                    "id": slug,
                    "name": meta.get("name", slug),
                    "kind": "custom",
                    "description": meta.get("description", ""),
                    "tags": meta.get("tags", []),
                    "has_blueprint": bool(meta.get("blueprint")),
                    "config_path": meta.get(
                        "config_path",
                        str(
                            (self.store.configs_dir / f"{slug}.yaml").relative_to(
                                self.store.workspace_root
                            )
                        ),
                    ),
                    "source_path": meta.get(
                        "source_path", str(source_path.relative_to(self.store.workspace_root))
                    ),
                }
            )
        rows.sort(key=lambda item: (item["kind"], item["name"]))
        return rows

    def get_architecture(self, slug: str) -> dict[str, Any]:
        meta = self.store.load_metadata().get("architectures", {}).get(slug, {})
        source_path = self.store.custom_models_dir / f"{slug}.py"
        config_path = self.store.configs_dir / f"{slug}.yaml"
        source_code = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
        config = _raw_yaml(config_path) if config_path.exists() else {}
        return {
            "id": slug,
            "name": meta.get("name", slug),
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
            "blueprint": meta.get("blueprint"),
            "source_code": source_code,
            "config": config,
            "config_path": str(config_path.relative_to(self.store.workspace_root))
            if config_path.exists()
            else None,
            "source_path": str(source_path.relative_to(self.store.workspace_root))
            if source_path.exists()
            else None,
        }

    def launch_job(
        self,
        kind: str,
        config: dict[str, Any],
        split: str | None = None,
        save_as_config_name: str | None = None,
    ) -> JobRecord:
        if kind not in {"train", "evaluate"}:
            raise ValueError(f"Unsupported job kind: {kind}")

        job_id = self.store.next_job_id(kind)
        run_name = str(config.get("mlflow", {}).get("run_name") or job_id)
        experiment_name = str(config.get("experiment", {}).get("name") or run_name)
        output_dir = str(config.get("paths", {}).get("output_dir") or f"runs/{run_name}")

        config_copy = json.loads(json.dumps(config))
        config_copy.setdefault("mlflow", {})
        config_copy["mlflow"]["enabled"] = True
        config_copy["mlflow"]["run_name"] = run_name
        config_copy.setdefault("experiment", {})
        config_copy["experiment"]["name"] = experiment_name
        config_copy.setdefault("paths", {})
        config_copy["paths"]["output_dir"] = output_dir

        if save_as_config_name:
            saved = self.save_config(save_as_config_name, config_copy)
            config_path = self.store.workspace_root / saved["path"]
        else:
            config_path = self.store.generated_configs_dir / f"{job_id}.yaml"
            config_path.write_text(
                yaml.safe_dump(config_copy, allow_unicode=False, sort_keys=False),
                encoding="utf-8",
            )

        command = [sys.executable, "-m", "uav_vit.cli", "train", "--config", str(config_path)]
        if kind == "evaluate":
            command = [
                sys.executable,
                "-m",
                "uav_vit.cli",
                "evaluate",
                "--config",
                str(config_path),
            ]
            if split:
                command.extend(["--split", split])

        log_path = self.store.logs_dir / f"{job_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=self.store.workspace_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.processes[job_id] = process

        record = JobRecord(
            id=job_id,
            kind=kind,
            status="running",
            command=command,
            config_path=str(config_path.relative_to(self.store.workspace_root)),
            generated_config_path=str(config_path.relative_to(self.store.workspace_root)),
            experiment_name=experiment_name,
            run_name=run_name,
            output_dir=output_dir,
            split=split,
            pid=process.pid,
            log_path=str(log_path.relative_to(self.store.workspace_root)),
            started_at=utc_now(),
        )
        jobs = self.store.read_jobs()
        jobs.insert(0, record)
        self.store.write_jobs(jobs)
        return record

    def refresh_jobs(self) -> list[JobRecord]:
        jobs = self.store.read_jobs()
        changed = False
        for job in jobs:
            process = self.processes.get(job.id)
            if job.status != "running":
                continue
            if process is None:
                if job.pid and self._pid_is_alive(job.pid):
                    continue
                job.status = "finished"
                job.finished_at = job.finished_at or utc_now()
                changed = True
                continue
            exit_code = process.poll()
            if exit_code is None:
                continue
            job.exit_code = int(exit_code)
            job.status = "completed" if exit_code == 0 else "failed"
            job.finished_at = utc_now()
            changed = True
            self.processes.pop(job.id, None)
        if changed:
            self.store.write_jobs(jobs)
        return jobs

    def stop_job(self, job_id: str) -> JobRecord:
        jobs = self.store.read_jobs()
        for job in jobs:
            if job.id != job_id:
                continue
            process = self.processes.get(job_id)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            job.status = "stopped"
            job.finished_at = utc_now()
            job.exit_code = -1
            self.processes.pop(job_id, None)
            self.store.write_jobs(jobs)
            return job
        raise FileNotFoundError(f"Job '{job_id}' not found.")

    def read_job_logs(self, job_id: str, tail_lines: int = 120) -> str:
        jobs = self.store.read_jobs()
        for job in jobs:
            if job.id != job_id or not job.log_path:
                continue
            log_path = self.store.workspace_root / job.log_path
            if not log_path.exists():
                return ""
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-tail_lines:])
        raise FileNotFoundError(f"Job '{job_id}' not found.")

    def update_experiment_metadata(
        self,
        experiment_key: str,
        tags: list[str] | None = None,
        rating: int | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"updated_at": utc_now()}
        if tags is not None:
            payload["tags"] = _normalize_tags(tags)
        if rating is not None:
            payload["rating"] = max(0, min(int(rating), 5))
        if note is not None:
            payload["note"] = note
        return self.store.update_metadata_entry("experiments", experiment_key, payload)

    def local_experiment_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for metrics_file in sorted(self.store.runs_dir.glob("*/metrics.csv")):
            run_name = metrics_file.parent.name
            try:
                frame = pd.read_csv(metrics_file)
            except Exception:
                continue
            if frame.empty:
                continue
            best_column = "map_50" if "map_50" in frame.columns else "map"
            best_idx = frame[best_column].idxmax()
            best_row = frame.loc[best_idx]
            last_row = frame.iloc[-1]
            rows.append(
                {
                    "key": run_name,
                    "run_name": run_name,
                    "experiment_name": run_name,
                    "model_name": self._guess_model_name(run_name),
                    "source": "local",
                    "status": "completed",
                    "best_epoch": int(best_row.get("epoch", 0)),
                    "last_epoch": int(last_row.get("epoch", 0)),
                    "map": float(best_row.get("map", 0.0)),
                    "map_50": float(best_row.get("map_50", 0.0)),
                    "map_75": float(best_row.get("map_75", 0.0)),
                    "mar_100": float(best_row.get("mar_100", 0.0)),
                    "latency_ms": float(best_row.get("latency_ms", 0.0)),
                    "fps": float(best_row.get("fps", 0.0)),
                    "train_loss": float(last_row.get("train_loss", 0.0)),
                    "output_dir": str(metrics_file.parent.relative_to(self.store.workspace_root)),
                    "metrics_path": str(metrics_file.relative_to(self.store.workspace_root)),
                }
            )
        return rows

    def _guess_model_name(self, run_name: str) -> str:
        for config_item in self.list_configs():
            if config_item.get("experiment_name") == run_name:
                return str(config_item.get("model_name", "unknown"))
        return "unknown"

    def build_recommendations(self, experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not experiments:
            return []
        max_fps = max(float(item.get("fps", 0.0)) for item in experiments) or 1.0
        min_latency = min(float(item.get("latency_ms", 0.0)) for item in experiments) or 1.0
        recommendations: list[dict[str, Any]] = []
        for item in experiments:
            rating = float(item.get("rating", 0) or 0)
            map50 = float(item.get("map_50", 0.0))
            fps = float(item.get("fps", 0.0))
            latency = float(item.get("latency_ms", 0.0))
            score = (
                (map50 * 0.62)
                + ((fps / max_fps) * 0.2)
                + ((min_latency / max(latency, 1e-6)) * 0.13)
            )
            score += rating * 0.01
            recommendations.append(
                {
                    "key": item["key"],
                    "run_name": item["run_name"],
                    "score": round(score, 4),
                    "summary": (f"mAP50={map50:.3f}, FPS={fps:.2f}, latency={latency:.2f} ms"),
                }
            )
        recommendations.sort(key=lambda entry: entry["score"], reverse=True)
        return recommendations[:5]

    def _pid_is_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
