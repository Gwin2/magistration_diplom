from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    collapsed = re.sub(r"_+", "_", normalized).strip("_")
    return collapsed or "item"


@dataclass
class JobRecord:
    id: str
    kind: str
    status: str
    command: list[str]
    config_path: str
    experiment_name: str
    run_name: str
    output_dir: str
    generated_config_path: str | None = None
    split: str | None = None
    pid: int | None = None
    log_path: str | None = None
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ControlStateStore:
    def __init__(self, workspace_root: Path | None = None) -> None:
        root = workspace_root or Path(os.environ.get("UAV_CONTROL_WORKSPACE", Path.cwd()))
        self.workspace_root = root.resolve()
        self.data_root = self.workspace_root / "data"
        self.configs_dir = self.workspace_root / "configs" / "experiments"
        self.custom_models_dir = self.workspace_root / "src" / "custom_models"
        self.runs_dir = self.workspace_root / "runs"
        self.reports_dir = self.workspace_root / "reports"
        self.model_store_dir = self.workspace_root / "model-store"
        self.state_dir = Path(
            os.environ.get("UAV_CONTROL_STATE_DIR", str(self.workspace_root / "workspace_state"))
        ).resolve()
        self.logs_dir = self.state_dir / "logs"
        self.generated_configs_dir = self.state_dir / "generated_configs"
        self.uploads_dir = self.data_root / "uploads"
        self.downloads_dir = self.state_dir / "downloads"
        self.metadata_file = self.state_dir / "ui_metadata.json"
        self.jobs_file = self.state_dir / "jobs.json"

        for directory in [
            self.state_dir,
            self.logs_dir,
            self.generated_configs_dir,
            self.downloads_dir,
            self.uploads_dir,
            self.runs_dir,
            self.reports_dir,
            self.model_store_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_metadata(self) -> dict[str, Any]:
        if not self.metadata_file.exists():
            return {"datasets": {}, "experiments": {}, "architectures": {}}
        try:
            return json.loads(self.metadata_file.read_text(encoding="utf-8"))
        except Exception:
            return {"datasets": {}, "experiments": {}, "architectures": {}}

    def save_metadata(self, payload: dict[str, Any]) -> None:
        self.metadata_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update_metadata_entry(
        self, section: str, key: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        payload = self.load_metadata()
        section_entries = payload.setdefault(section, {})
        current = section_entries.get(key, {})
        current.update(values)
        section_entries[key] = current
        self.save_metadata(payload)
        return current

    def read_jobs(self) -> list[JobRecord]:
        if not self.jobs_file.exists():
            return []
        try:
            rows = json.loads(self.jobs_file.read_text(encoding="utf-8"))
        except Exception:
            return []
        jobs: list[JobRecord] = []
        for row in rows:
            try:
                jobs.append(JobRecord(**row))
            except TypeError:
                continue
        return jobs

    def write_jobs(self, jobs: list[JobRecord]) -> None:
        self.jobs_file.write_text(
            json.dumps([job.to_dict() for job in jobs], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def next_job_id(self, prefix: str) -> str:
        return f"{slugify(prefix)}_{int(time.time() * 1000)}"

    def discover_dataset_directories(self) -> list[Path]:
        candidates: list[Path] = []
        roots = [
            self.data_root / "raw",
            self.data_root / "processed",
            self.uploads_dir,
        ]
        for root in roots:
            if not root.exists():
                continue
            for child in sorted(root.iterdir()):
                if child.is_dir():
                    candidates.append(child)
        return candidates

    def dataset_id_for_path(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.workspace_root)
        return slugify(str(relative).replace("\\", "/"))

    def get_dataset_path(self, dataset_id: str) -> Path | None:
        for path in self.discover_dataset_directories():
            if self.dataset_id_for_path(path) == dataset_id:
                return path
        metadata = self.load_metadata().get("datasets", {})
        stored = metadata.get(dataset_id, {})
        raw_path = stored.get("path")
        if raw_path:
            candidate = (self.workspace_root / raw_path).resolve()
            if candidate.exists():
                return candidate
        return None

    def register_dataset(
        self,
        name: str,
        path: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = (self.workspace_root / resolved).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        dataset_id = self.dataset_id_for_path(resolved)
        return self.update_metadata_entry(
            "datasets",
            dataset_id,
            {
                "name": name,
                "path": str(resolved.relative_to(self.workspace_root)),
                "description": description,
                "tags": sorted({tag.strip() for tag in (tags or []) if tag.strip()}),
                "updated_at": utc_now(),
            },
        )

    def create_dataset_archive(self, dataset_id: str) -> Path:
        dataset_path = self.get_dataset_path(dataset_id)
        if dataset_path is None:
            raise FileNotFoundError(f"Dataset '{dataset_id}' was not found.")
        archive_base = self.downloads_dir / f"{dataset_id}-{int(time.time())}"
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=dataset_path)
        return Path(archive_path)

    def file_stats(self, path: Path) -> dict[str, Any]:
        total_size = 0
        file_count = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                file_count += 1
                try:
                    total_size += entry.stat().st_size
                except OSError:
                    continue
        return {"file_count": file_count, "size_bytes": total_size}
