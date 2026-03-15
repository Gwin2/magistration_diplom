from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any


def _import_prometheus_client() -> Any | None:
    try:
        import prometheus_client  # type: ignore[import-not-found]
    except ImportError:
        return None
    return prometheus_client


@dataclass
class PrometheusPushConfig:
    enabled: bool
    gateway_url: str
    job_name: str
    instance: str
    timeout: int = 5


def _monitoring_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("monitoring", {})


def build_push_config(config: dict[str, Any], phase: str) -> PrometheusPushConfig | None:
    mon_cfg = _monitoring_config(config)
    push_cfg = mon_cfg.get("pushgateway", {})

    enabled = bool(push_cfg.get("enabled", False))
    env_url = os.environ.get("PUSHGATEWAY_URL", "")
    gateway_url = str(push_cfg.get("url", env_url)).strip()
    if not enabled and not env_url:
        return None
    if not gateway_url:
        return None

    job_name = str(push_cfg.get("job", f"uav-vit-{phase}"))
    instance = str(push_cfg.get("instance", socket.gethostname()))
    timeout = int(push_cfg.get("timeout_seconds", 5))
    return PrometheusPushConfig(
        enabled=True,
        gateway_url=gateway_url,
        job_name=job_name,
        instance=instance,
        timeout=timeout,
    )


class PrometheusPusher:
    def __init__(
        self, push_config: PrometheusPushConfig | None, experiment: str, model: str
    ) -> None:
        self.push_config = push_config
        self.experiment = experiment
        self.model = model
        self.prometheus_client = _import_prometheus_client()
        if self.push_config is not None and self.prometheus_client is None:
            print(
                "[monitoring] prometheus_client is not installed. Pushgateway export is disabled."
            )
            self.push_config = None

    @property
    def enabled(self) -> bool:
        return self.push_config is not None and self.prometheus_client is not None

    def push_train_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        if not self.enabled:
            return
        registry = self.prometheus_client.CollectorRegistry()
        labels = {"experiment": self.experiment, "model": self.model}

        self._gauge("uav_train_epoch", "Current training epoch", labels, registry).set(float(epoch))
        self._gauge("uav_train_loss", "Training loss", labels, registry).set(
            float(metrics.get("train_loss", 0.0))
        )
        self._gauge("uav_val_map", "Validation mAP", labels, registry).set(
            float(metrics.get("map", 0.0))
        )
        self._gauge("uav_val_map_50", "Validation mAP@50", labels, registry).set(
            float(metrics.get("map_50", 0.0))
        )
        self._gauge("uav_val_map_75", "Validation mAP@75", labels, registry).set(
            float(metrics.get("map_75", 0.0))
        )
        self._gauge("uav_val_mar_100", "Validation mAR@100", labels, registry).set(
            float(metrics.get("mar_100", 0.0))
        )
        self._gauge("uav_val_latency_ms", "Validation latency ms", labels, registry).set(
            float(metrics.get("latency_ms", 0.0))
        )
        self._gauge("uav_val_fps", "Validation FPS", labels, registry).set(
            float(metrics.get("fps", 0.0))
        )
        self._push(registry)

    def push_train_summary(self, best_metric_name: str, best_metric_value: float) -> None:
        if not self.enabled:
            return
        registry = self.prometheus_client.CollectorRegistry()
        labels = {"experiment": self.experiment, "model": self.model, "metric": best_metric_name}
        self._gauge("uav_best_metric_value", "Best checkpoint metric value", labels, registry).set(
            float(best_metric_value)
        )
        self._push(registry)

    def push_evaluation(self, split: str, metrics: dict[str, float]) -> None:
        if not self.enabled:
            return
        registry = self.prometheus_client.CollectorRegistry()
        labels = {"experiment": self.experiment, "model": self.model, "split": split}
        self._gauge("uav_eval_map", "Evaluation mAP", labels, registry).set(
            float(metrics.get("map", 0.0))
        )
        self._gauge("uav_eval_map_50", "Evaluation mAP@50", labels, registry).set(
            float(metrics.get("map_50", 0.0))
        )
        self._gauge("uav_eval_map_75", "Evaluation mAP@75", labels, registry).set(
            float(metrics.get("map_75", 0.0))
        )
        self._gauge("uav_eval_mar_100", "Evaluation mAR@100", labels, registry).set(
            float(metrics.get("mar_100", 0.0))
        )
        self._gauge("uav_eval_latency_ms", "Evaluation latency ms", labels, registry).set(
            float(metrics.get("latency_ms", 0.0))
        )
        self._gauge("uav_eval_fps", "Evaluation FPS", labels, registry).set(
            float(metrics.get("fps", 0.0))
        )
        self._push(registry)

    def _gauge(
        self,
        metric_name: str,
        help_text: str,
        labels: dict[str, str],
        registry: Any,
    ) -> Any:
        label_names = list(labels.keys())
        gauge = self.prometheus_client.Gauge(
            metric_name,
            help_text,
            labelnames=label_names,
            registry=registry,
        )
        return gauge.labels(**labels)

    def _push(self, registry: Any) -> None:
        if self.push_config is None:
            return
        grouping_key = {"instance": self.push_config.instance}
        try:
            self.prometheus_client.push_to_gateway(
                self.push_config.gateway_url,
                job=self.push_config.job_name,
                registry=registry,
                grouping_key=grouping_key,
                timeout=self.push_config.timeout,
            )
        except Exception as exc:
            print(f"[monitoring] Pushgateway export failed: {exc}")
