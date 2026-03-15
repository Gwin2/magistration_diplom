from __future__ import annotations

import shutil
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Any

import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Path, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from uav_vit.control.mlops import MlflowBridge, TensorBoardManager, TorchServeBridge
from uav_vit.control.state import ControlStateStore, slugify
from uav_vit.control.workspace import WorkspaceService


class DatasetRegisterPayload(BaseModel):
    name: str
    path: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class ConfigSavePayload(BaseModel):
    config_yaml: str


class ArchitectureSavePayload(BaseModel):
    name: str
    config_yaml: str
    source_code: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class JobLaunchPayload(BaseModel):
    config_name: str | None = None
    config_yaml: str | None = None
    save_as_config_name: str | None = None
    split: str | None = None


class ExperimentMetadataPayload(BaseModel):
    tags: list[str] | None = None
    rating: int | None = None
    note: str | None = None


class TorchServeRegisterPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    registered_name: str = Field(
        validation_alias="model_name", serialization_alias="model_name"
    )
    archive_file: str
    initial_workers: int = 1
    synchronous: bool = True


def create_app() -> FastAPI:
    store = ControlStateStore()
    workspace = WorkspaceService(store)
    mlflow = MlflowBridge()
    tensorboard = TensorBoardManager(store)
    torchserve = TorchServeBridge()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # noqa: ARG001
        with suppress(Exception):
            tensorboard.ensure_started()
        yield
        with suppress(Exception):
            tensorboard.stop()

    app = FastAPI(title="UAV ViT Control API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.store = store
    app.state.workspace = workspace
    app.state.mlflow = mlflow
    app.state.tensorboard = tensorboard
    app.state.torchserve = torchserve

    @app.get("/health")
    def health() -> dict[str, Any]:
        datasets = workspace.list_datasets()
        architectures = workspace.list_architectures()
        jobs = [job.to_dict() for job in workspace.refresh_jobs()]
        running_jobs = sum(1 for job in jobs if job["status"] == "running")
        return {
            "status": "ok",
            "workspace_root": str(store.workspace_root),
            "datasets": len(datasets),
            "architectures": len(architectures),
            "jobs_total": len(jobs),
            "jobs_running": running_jobs,
            "tensorboard": tensorboard.status(),
            "torchserve_available": torchserve.ping(),
            "mlflow_enabled": bool(mlflow.tracking_uri),
        }

    @app.get("/catalog")
    def catalog() -> dict[str, Any]:
        jobs = [job.to_dict() for job in workspace.refresh_jobs()]
        experiments = _list_experiments(workspace, mlflow, limit=200)
        return {
            "datasets": workspace.list_datasets(),
            "configs": workspace.list_configs(),
            "architectures": workspace.list_architectures(),
            "jobs": jobs,
            "experiments": experiments["items"],
            "recommendations": experiments["recommendations"],
            "tensorboard": tensorboard.status(),
        }

    @app.get("/datasets")
    def list_datasets() -> dict[str, Any]:
        return {"items": workspace.list_datasets()}

    @app.post("/datasets/register")
    def register_dataset(payload: DatasetRegisterPayload) -> dict[str, Any]:
        workspace.store.register_dataset(
            name=payload.name,
            path=payload.path,
            description=payload.description,
            tags=payload.tags,
        )
        return {"items": workspace.list_datasets()}

    @app.post("/datasets/upload")
    async def upload_dataset(
        dataset_name: Annotated[str, Form(...)],
        file: Annotated[UploadFile, File(...)],
        description: Annotated[str, Form()] = "",
        tags: Annotated[str, Form()] = "",
    ) -> dict[str, Any]:
        dataset_slug = slugify(dataset_name)
        upload_dir = store.uploads_dir / dataset_slug
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        archive_name = file.filename or f"{dataset_slug}.zip"
        archive_path = store.state_dir / archive_name
        content = await file.read()
        archive_path.write_bytes(content)

        extracted = False
        try:
            shutil.unpack_archive(str(archive_path), str(upload_dir))
            extracted = True
        except (shutil.ReadError, ValueError):
            (upload_dir / archive_name).write_bytes(content)
        finally:
            with suppress(OSError):
                archive_path.unlink()

        workspace.store.register_dataset(
            name=dataset_name,
            path=str(upload_dir.relative_to(store.workspace_root)),
            description=description,
            tags=_parse_tags(tags),
        )
        return {
            "uploaded": True,
            "archive_extracted": extracted,
            "items": workspace.list_datasets(),
        }

    @app.get("/datasets/{dataset_id}/download")
    def download_dataset(dataset_id: str) -> FileResponse:
        try:
            archive_path = store.create_dataset_archive(dataset_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            path=archive_path, filename=archive_path.name, media_type="application/zip"
        )

    @app.get("/configs")
    def list_configs() -> dict[str, Any]:
        return {"items": workspace.list_configs()}

    @app.get("/configs/{config_name}")
    def get_config(config_name: str) -> dict[str, Any]:
        try:
            config = workspace.load_config(config_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "name": config_name,
            "config": config,
            "config_yaml": yaml.safe_dump(config, allow_unicode=False, sort_keys=False),
        }

    @app.put("/configs/{config_name}")
    def save_config(config_name: str, payload: ConfigSavePayload) -> dict[str, Any]:
        config = _parse_yaml_mapping(payload.config_yaml)
        saved = workspace.save_config(config_name, config)
        return {
            **saved,
            "config_yaml": yaml.safe_dump(config, allow_unicode=False, sort_keys=False),
        }

    @app.get("/architectures")
    def list_architectures() -> dict[str, Any]:
        return {"items": workspace.list_architectures()}

    @app.get("/architectures/{architecture_id}")
    def get_architecture(architecture_id: str) -> dict[str, Any]:
        data = workspace.get_architecture(architecture_id)
        data["config_yaml"] = yaml.safe_dump(
            data.get("config", {}), allow_unicode=False, sort_keys=False
        )
        return data

    @app.post("/architectures")
    def save_architecture(payload: ArchitectureSavePayload) -> dict[str, Any]:
        config = _parse_yaml_mapping(payload.config_yaml)
        result = workspace.save_custom_architecture(
            name=payload.name,
            config=config,
            source_code=payload.source_code.strip() or None,
            description=payload.description,
            tags=payload.tags,
        )
        result["config_yaml"] = yaml.safe_dump(
            result.get("config", {}), allow_unicode=False, sort_keys=False
        )
        return result

    @app.get("/jobs")
    def list_jobs() -> dict[str, Any]:
        jobs = [job.to_dict() for job in workspace.refresh_jobs()]
        return {"items": jobs, "tensorboard": tensorboard.status()}

    @app.post("/jobs/train")
    def launch_train_job(payload: JobLaunchPayload) -> dict[str, Any]:
        config = _resolve_launch_config(workspace, payload)
        job = workspace.launch_job(
            kind="train",
            config=config,
            save_as_config_name=payload.save_as_config_name,
        )
        return {"job": job.to_dict()}

    @app.post("/jobs/evaluate")
    def launch_eval_job(payload: JobLaunchPayload) -> dict[str, Any]:
        config = _resolve_launch_config(workspace, payload)
        job = workspace.launch_job(
            kind="evaluate",
            config=config,
            split=payload.split or "test",
            save_as_config_name=payload.save_as_config_name,
        )
        return {"job": job.to_dict()}

    @app.post("/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict[str, Any]:
        try:
            job = workspace.stop_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"job": job.to_dict()}

    @app.get("/jobs/{job_id}/logs", response_class=PlainTextResponse)
    def job_logs(job_id: str, tail_lines: int = Query(default=160, ge=10, le=5000)) -> str:
        try:
            return workspace.read_job_logs(job_id, tail_lines=tail_lines)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/experiments")
    def list_experiments(
        search: str = "",
        tag: str | None = None,
        model_filter: str | None = Query(default=None, alias="model_name"),
        status: str | None = None,
        min_map50: float | None = None,
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> dict[str, Any]:
        payload = _list_experiments(workspace, mlflow, limit=limit)
        items = _filter_experiments(
            payload["items"],
            search=search,
            tag=tag,
            model_name=model_filter,
            status=status,
            min_map50=min_map50,
        )
        payload["items"] = items
        payload["recommendations"] = workspace.build_recommendations(items)
        return payload

    @app.get("/experiments/compare")
    def compare_experiments(
        keys: Annotated[list[str] | None, Query()] = None,
    ) -> dict[str, Any]:
        selected_keys = _expand_keys(keys or [])
        payload = _list_experiments(workspace, mlflow, limit=500)
        items = [item for item in payload["items"] if item["key"] in selected_keys]
        return {
            "items": items,
            "columns": [
                "run_name",
                "experiment_name",
                "model_name",
                "map_50",
                "map_75",
                "fps",
                "latency_ms",
                "rating",
                "status",
            ],
        }

    @app.post("/experiments/{experiment_key}/metadata")
    def update_experiment_metadata(
        experiment_key: str,
        payload: ExperimentMetadataPayload,
    ) -> dict[str, Any]:
        data = workspace.update_experiment_metadata(
            experiment_key,
            tags=payload.tags,
            rating=payload.rating,
            note=payload.note,
        )
        mlflow.apply_ui_metadata(
            experiment_key,
            tags=payload.tags,
            rating=payload.rating,
            note=payload.note,
        )
        return data

    @app.get("/tensorboard/status")
    def tensorboard_status() -> dict[str, Any]:
        return tensorboard.status()

    @app.post("/tensorboard/start")
    def tensorboard_start() -> dict[str, Any]:
        return tensorboard.ensure_started()

    @app.get("/torchserve/models")
    def list_torchserve_models() -> dict[str, Any]:
        try:
            payload = torchserve.list_models()
        except RuntimeError as exc:
            return {"available": False, "models": [], "error": str(exc)}
        return {"available": True, **payload}

    @app.post("/torchserve/register")
    def register_torchserve_model(payload: TorchServeRegisterPayload) -> dict[str, Any]:
        try:
            result = torchserve.register_model(
                model_name=payload.registered_name,
                archive_file=payload.archive_file,
                initial_workers=payload.initial_workers,
                synchronous=payload.synchronous,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return result

    @app.delete("/torchserve/models/{model_name}")
    def unregister_torchserve_model(
        registered_model_name: Annotated[str, Path(alias="model_name")]
    ) -> dict[str, Any]:
        try:
            result = torchserve.unregister_model(registered_model_name)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return result

    @app.post("/torchserve/predict")
    async def torchserve_predict(
        inference_model_name: Annotated[str, Form(alias="model_name")],
        file: Annotated[UploadFile, File(...)],
    ) -> Any:
        payload = await file.read()
        content_type = file.content_type or "application/octet-stream"
        try:
            return torchserve.predict(
                model_name=inference_model_name,
                payload=payload,
                content_type=content_type,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


def _parse_tags(raw_tags: str) -> list[str]:
    return sorted({item.strip() for item in raw_tags.split(",") if item and item.strip()})


def _parse_yaml_mapping(text: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400, detail="Config must contain a YAML mapping at the top level."
        )
    return payload


def _resolve_launch_config(
    workspace: WorkspaceService, payload: JobLaunchPayload
) -> dict[str, Any]:
    if payload.config_yaml:
        return _parse_yaml_mapping(payload.config_yaml)
    if payload.config_name:
        try:
            return workspace.load_config(payload.config_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(
        status_code=400, detail="Either config_name or config_yaml must be provided."
    )


def _list_experiments(
    workspace: WorkspaceService,
    mlflow: MlflowBridge,
    limit: int,
) -> dict[str, Any]:
    local_rows = workspace.local_experiment_rows()
    mlflow_rows = mlflow.list_runs(limit=limit)
    metadata = workspace.store.load_metadata().get("experiments", {})

    merged: dict[str, dict[str, Any]] = {}
    for row in local_rows:
        merged[row["key"]] = dict(row)
    for row in mlflow_rows:
        existing = merged.get(row["key"])
        if existing is None:
            merged[row["key"]] = dict(row)
            continue
        combined = dict(row)
        combined.update(existing)
        combined["source"] = "hybrid"
        if row.get("mlflow_run_id"):
            combined["mlflow_run_id"] = row["mlflow_run_id"]
        if row.get("status") and existing.get("status") not in {"running", "completed"}:
            combined["status"] = row["status"]
        merged[row["key"]] = combined

    items = list(merged.values())
    for item in items:
        local_meta = metadata.get(item["key"], {})
        if local_meta.get("tags"):
            item["tags"] = local_meta["tags"]
        else:
            item["tags"] = item.get("tags", [])
        if "rating" in local_meta:
            item["rating"] = local_meta["rating"]
        else:
            item["rating"] = item.get("rating")
        if "note" in local_meta:
            item["note"] = local_meta["note"]
        else:
            item["note"] = item.get("note", "")

    items.sort(
        key=lambda row: (
            row.get("status") != "running",
            -(float(row.get("map_50", 0.0) or 0.0)),
            -(float(row.get("fps", 0.0) or 0.0)),
            row.get("run_name", ""),
        )
    )
    items = items[:limit]
    return {"items": items, "recommendations": workspace.build_recommendations(items)}


def _filter_experiments(
    items: list[dict[str, Any]],
    search: str = "",
    tag: str | None = None,
    model_name: str | None = None,
    status: str | None = None,
    min_map50: float | None = None,
) -> list[dict[str, Any]]:
    search_term = search.strip().lower()
    filtered: list[dict[str, Any]] = []
    for item in items:
        haystack = " ".join(
            [
                str(item.get("run_name", "")),
                str(item.get("experiment_name", "")),
                str(item.get("model_name", "")),
                " ".join(item.get("tags", [])),
            ]
        ).lower()
        if search_term and search_term not in haystack:
            continue
        if tag and tag not in item.get("tags", []):
            continue
        if model_name and str(item.get("model_name")) != model_name:
            continue
        if status and str(item.get("status")) != status:
            continue
        if min_map50 is not None and float(item.get("map_50", 0.0) or 0.0) < min_map50:
            continue
        filtered.append(item)
    return filtered


def _expand_keys(raw_keys: list[str]) -> list[str]:
    values: list[str] = []
    for item in raw_keys:
        for part in item.split(","):
            candidate = part.strip()
            if candidate:
                values.append(candidate)
    return values


def main() -> None:
    uvicorn.run(
        "uav_vit.control.app:create_app",
        host="0.0.0.0",
        port=8010,
        factory=True,
    )
