from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from uav_vit.control.state import ControlStateStore


class MlflowBridge:
    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")

    def list_runs(self, limit: int = 200) -> list[dict[str, Any]]:
        client = self._client()
        if client is None:
            return []

        try:
            experiments = client.search_experiments()
        except Exception:
            return []
        if not experiments:
            return []

        experiment_names = {str(item.experiment_id): item.name for item in experiments}
        experiment_ids = list(experiment_names)
        try:
            runs = client.search_runs(
                experiment_ids=experiment_ids,
                max_results=max(limit, 200),
                order_by=["attributes.start_time DESC"],
            )
        except Exception:
            return []

        rows: list[dict[str, Any]] = []
        for run in runs:
            metrics = run.data.metrics or {}
            params = run.data.params or {}
            tags = run.data.tags or {}
            run_name = (
                getattr(run.info, "run_name", None) or tags.get("mlflow.runName") or run.info.run_id
            )
            tag_list = [
                item.strip()
                for item in str(tags.get("ui.tags", "")).split(",")
                if item and item.strip()
            ]
            rows.append(
                {
                    "key": run_name,
                    "run_name": run_name,
                    "experiment_name": experiment_names.get(str(run.info.experiment_id), ""),
                    "model_name": tags.get("model.name")
                    or params.get("model.name")
                    or params.get("model_name")
                    or "unknown",
                    "source": "mlflow",
                    "status": str(run.info.status).lower(),
                    "mlflow_run_id": run.info.run_id,
                    "started_at": getattr(run.info, "start_time", None),
                    "ended_at": getattr(run.info, "end_time", None),
                    "map": float(metrics.get("val_map", metrics.get("test_map", 0.0)) or 0.0),
                    "map_50": float(
                        metrics.get(
                            "val_map_50", metrics.get("test_map_50", metrics.get("map_50", 0.0))
                        )
                        or 0.0
                    ),
                    "map_75": float(
                        metrics.get(
                            "val_map_75", metrics.get("test_map_75", metrics.get("map_75", 0.0))
                        )
                        or 0.0
                    ),
                    "mar_100": float(
                        metrics.get(
                            "val_mar_100",
                            metrics.get("test_mar_100", metrics.get("mar_100", 0.0)),
                        )
                        or 0.0
                    ),
                    "latency_ms": float(
                        metrics.get(
                            "latency_ms",
                            metrics.get("test_latency_ms", metrics.get("val_latency_ms", 0.0)),
                        )
                        or 0.0
                    ),
                    "fps": float(
                        metrics.get("fps", metrics.get("test_fps", metrics.get("val_fps", 0.0)))
                        or 0.0
                    ),
                    "train_loss": float(metrics.get("train_loss", 0.0) or 0.0),
                    "best_metric_value": float(metrics.get("best_metric_value", 0.0) or 0.0),
                    "tags": sorted(set(tag_list)),
                    "rating": self._coerce_rating(tags.get("ui.rating")),
                    "note": str(tags.get("ui.note", "")),
                }
            )
        return rows[:limit]

    def apply_ui_metadata(
        self,
        run_name: str,
        tags: list[str] | None = None,
        rating: int | None = None,
        note: str | None = None,
    ) -> bool:
        client = self._client()
        if client is None:
            return False
        run = self._find_run(client, run_name)
        if run is None:
            return False

        run_id = run.info.run_id
        try:
            if tags is not None:
                client.set_tag(run_id, "ui.tags", ",".join(sorted(set(tags))))
            if rating is not None:
                client.set_tag(run_id, "ui.rating", str(max(0, min(int(rating), 5))))
            if note is not None:
                client.set_tag(run_id, "ui.note", note)
        except Exception:
            return False
        return True

    def _find_run(self, client: Any, run_name: str) -> Any | None:
        try:
            experiments = client.search_experiments()
        except Exception:
            return None
        experiment_ids = [str(item.experiment_id) for item in experiments]
        if not experiment_ids:
            return None
        try:
            runs = client.search_runs(
                experiment_ids=experiment_ids,
                max_results=1000,
                order_by=["attributes.start_time DESC"],
            )
        except Exception:
            return None
        for run in runs:
            current_name = getattr(run.info, "run_name", None) or run.data.tags.get(
                "mlflow.runName"
            )
            if current_name == run_name:
                return run
        return None

    def _client(self) -> Any | None:
        if not self.tracking_uri:
            return None
        try:
            from mlflow.tracking import MlflowClient
        except ImportError:
            return None
        return MlflowClient(tracking_uri=self.tracking_uri)

    def _coerce_rating(self, value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return max(0, min(int(value), 5))
        except (TypeError, ValueError):
            return None


class TorchServeBridge:
    def __init__(
        self,
        management_url: str | None = None,
        inference_url: str | None = None,
    ) -> None:
        self.management_url = management_url or os.environ.get(
            "TORCHSERVE_MANAGEMENT_URL",
            "http://torchserve:8081",
        )
        self.inference_url = inference_url or os.environ.get(
            "TORCHSERVE_INFERENCE_URL",
            "http://torchserve:8080",
        )

    def ping(self) -> bool:
        try:
            self._request(self.inference_url, "/ping")
        except RuntimeError:
            return False
        return True

    def list_models(self) -> dict[str, Any]:
        return self._request(self.management_url, "/models")

    def register_model(
        self,
        model_name: str,
        archive_file: str,
        initial_workers: int = 1,
        synchronous: bool = True,
    ) -> dict[str, Any]:
        return self._request(
            self.management_url,
            "/models",
            method="POST",
            params={
                "model_name": model_name,
                "url": archive_file,
                "initial_workers": max(1, int(initial_workers)),
                "synchronous": str(bool(synchronous)).lower(),
            },
        )

    def unregister_model(self, model_name: str) -> dict[str, Any]:
        return self._request(self.management_url, f"/models/{model_name}", method="DELETE")

    def predict(self, model_name: str, payload: bytes, content_type: str) -> Any:
        return self._request(
            self.inference_url,
            f"/predictions/{model_name}",
            method="POST",
            data=payload,
            headers={"Content-Type": content_type},
        )

    def _request(
        self,
        base_url: str,
        path: str,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
    ) -> Any:
        url = f"{base_url.rstrip('/')}{path}"
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        request = Request(url=url, method=method, data=data)
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        try:
            with urlopen(request, timeout=8) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type", "")
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(message or f"TorchServe HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"TorchServe is unavailable: {exc.reason}") from exc

        text = body.decode("utf-8", errors="replace")
        if "application/json" in content_type:
            return json.loads(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}


class TensorBoardManager:
    def __init__(
        self,
        store: ControlStateStore,
        host: str = "0.0.0.0",
        port: int = 6006,
        path_prefix: str = "/api/tensorboard",
    ) -> None:
        self.store = store
        self.host = host
        self.port = port
        self.path_prefix = path_prefix
        self.process: subprocess.Popen[str] | None = None

    def ensure_started(self) -> dict[str, Any]:
        if self.process is not None and self.process.poll() is None:
            return self.status()

        command = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(self.store.runs_dir),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--path_prefix",
            self.path_prefix,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=self.store.workspace_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return self.status()

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def status(self) -> dict[str, Any]:
        running = self.process is not None and self.process.poll() is None
        return {
            "running": running,
            "pid": self.process.pid if running and self.process is not None else None,
            "logdir": str(self.store.runs_dir),
            "port": self.port,
            "path_prefix": self.path_prefix,
            "url": f"{self.path_prefix}/",
        }
