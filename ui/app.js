const statusChecks = {
  control: "/api/control/health",
  mlflow: "/api/mlflow/",
  prometheus: "/api/prometheus/-/healthy",
  grafana: "/api/grafana/api/health",
  tensorboard: "/api/control/tensorboard/status",
  pushgateway: "/api/pushgateway/metrics",
  torchserve: "/api/control/torchserve/models",
  alertmanager: "/api/alertmanager/-/healthy",
};

const metricQueries = {
  map50: "max(uav_eval_map50)",
  map75: "max(uav_eval_map75)",
  latency: "avg(uav_inference_latency_ms)",
  fps: "avg(uav_inference_fps)",
  health: "avg(probe_success)",
  host_cpu: "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
  host_mem: "(1 - (avg(node_memory_MemAvailable_bytes) / avg(node_memory_MemTotal_bytes))) * 100",
  firing_alerts: "sum(ALERTS{alertstate=\"firing\"})",
};

const metricFallback = {
  map50: 0.41,
  map75: 0.23,
  latency: 38.4,
  fps: 19.2,
  health: 0.94,
  host_cpu: 34.0,
  host_mem: 52.0,
  firing_alerts: 0,
};

const kpiFormatters = {
  map50: (value) => value.toFixed(3),
  map75: (value) => value.toFixed(3),
  latency: (value) => value.toFixed(1),
  fps: (value) => value.toFixed(1),
  health: (value) => `${(value * 100).toFixed(1)}%`,
  host_cpu: (value) => `${value.toFixed(1)}%`,
  host_mem: (value) => `${value.toFixed(1)}%`,
  firing_alerts: (value) => `${Math.round(value)}`,
};

const fallbackContainers = [
  { name: "uav-control-api", cpu: 3.2, mem: 440, rx: 82 },
  { name: "uav-prometheus", cpu: 7.6, mem: 680, rx: 210 },
  { name: "uav-grafana", cpu: 2.1, mem: 310, rx: 74 },
  { name: "uav-mlflow", cpu: 4.4, mem: 420, rx: 95 },
  { name: "uav-minio", cpu: 3.2, mem: 385, rx: 112 },
];

const fallbackProcesses = [
  { name: "uvicorn", cpu: 4.8, mem: 220, threads: 17 },
  { name: "prometheus", cpu: 8.4, mem: 540, threads: 18 },
  { name: "grafana", cpu: 2.6, mem: 290, threads: 29 },
  { name: "postgres", cpu: 3.3, mem: 245, threads: 14 },
  { name: "minio", cpu: 3.8, mem: 362, threads: 21 },
];

const uiStorageKeys = {
  theme: "uav_ui_theme",
  motion: "uav_ui_motion",
  density: "uav_ui_density",
  layout: "uav_ui_panel_layouts",
};

const allowedThemes = new Set(["flight", "horizon", "paper", "signal"]);
const allowedDensity = new Set(["compact", "comfortable"]);
const metricHistory = {};
Object.keys(metricQueries).forEach((key) => {
  metricHistory[key] = Array.from({ length: 18 }, () => metricFallback[key]);
});

const uiState = {
  datasets: [],
  configs: [],
  architectures: [],
  experiments: [],
  experimentUniverse: [],
  recommendations: [],
  jobs: [],
  tensorboard: null,
  torchserve: { available: false, models: [] },
  selectedConfigName: "",
  selectedArchitectureId: "",
  selectedExperimentKey: "",
  selectedJobId: "",
  compareKeys: new Set(),
  serviceStatus: Object.fromEntries(Object.keys(statusChecks).map((service) => [service, null])),
  constructorCatalog: null,
  constructorBlueprint: null,
  constructorPreview: null,
};

const constructorDragState = {
  mode: null,
  layerType: "",
  sourceHead: "",
  sourceIndex: -1,
};

const panelDragState = {
  panelId: "",
  sourcePaneId: "",
  element: null,
  targetPaneId: "",
};

const defaultConfigText = `experiment:\n  name: new_experiment\n  seed: 42\n\npaths:\n  train_images: data/processed/uav_coco/images/train\n  val_images: data/processed/uav_coco/images/val\n  test_images: data/processed/uav_coco/images/test\n  train_annotations: data/processed/uav_coco/annotations/instances_train.json\n  val_annotations: data/processed/uav_coco/annotations/instances_val.json\n  test_annotations: data/processed/uav_coco/annotations/instances_test.json\n  output_dir: runs/new_experiment\n\nmlflow:\n  enabled: true\n  tracking_uri: http://mlflow:5000\n  experiment_name: uav-vit-thesis\n  run_name: new_experiment\n  log_checkpoints: true\n\ntensorboard:\n  enabled: true\n\nmodel:\n  name: yolos_tiny\n  checkpoint: hustvl/yolos-tiny\n  num_labels: 1\n  id2label:\n    \"0\": uav\n  label2id:\n    uav: 0\n  train_backbone: true\n  custom_modules: []\n\ntrain:\n  device: auto\n  epochs: 30\n  batch_size: 4\n  learning_rate: 2.0e-5\n  weight_decay: 1.0e-4\n  num_workers: 4\n  grad_clip_norm: 1.0\n  log_interval: 20\n  mixed_precision: true\n  eval_every_epoch: true\n  checkpoint_metric: map\n  checkpoint_mode: max\n\neval:\n  score_threshold: 0.1\n  latency_warmup_iters: 10\n  latency_iters: 50\n\ndata:\n  processor_size: 800\n  normalize_boxes: false\n`;

const defaultArchitectureConfigText = `experiment:\n  name: custom_detector\n  seed: 42\n\npaths:\n  train_images: data/processed/uav_coco/images/train\n  val_images: data/processed/uav_coco/images/val\n  test_images: data/processed/uav_coco/images/test\n  train_annotations: data/processed/uav_coco/annotations/instances_train.json\n  val_annotations: data/processed/uav_coco/annotations/instances_val.json\n  test_annotations: data/processed/uav_coco/annotations/instances_test.json\n  output_dir: runs/custom_detector\n\nmlflow:\n  enabled: true\n  tracking_uri: http://mlflow:5000\n  experiment_name: uav-vit-thesis\n  run_name: custom_detector\n  log_checkpoints: true\n\ntensorboard:\n  enabled: true\n\nmodel:\n  name: my_detector\n  checkpoint: facebook/detr-resnet-50\n  num_labels: 1\n  id2label:\n    \"0\": uav\n  label2id:\n    uav: 0\n  train_backbone: true\n  custom_modules:\n    - custom_models.my_detector\n\ntrain:\n  device: auto\n  epochs: 30\n  batch_size: 4\n  learning_rate: 2.0e-5\n  weight_decay: 1.0e-4\n  num_workers: 4\n  grad_clip_norm: 1.0\n  log_interval: 20\n  mixed_precision: true\n  eval_every_epoch: true\n  checkpoint_metric: map\n  checkpoint_mode: max\n\neval:\n  score_threshold: 0.1\n  latency_warmup_iters: 10\n  latency_iters: 50\n\ndata:\n  processor_size: 800\n  normalize_boxes: false\n`;

const defaultArchitectureSource = `from __future__ import annotations\n\nfrom transformers import AutoImageProcessor, AutoModelForObjectDetection\n\nfrom uav_vit.models import ModelBundle, register_model\n\n\n@register_model(\"my_detector\")\ndef build_my_detector(config: dict) -> ModelBundle:\n    checkpoint = config[\"model\"].get(\"checkpoint\") or \"facebook/detr-resnet-50\"\n    model = AutoModelForObjectDetection.from_pretrained(\n        checkpoint,\n        ignore_mismatched_sizes=True,\n        num_labels=int(config[\"model\"][\"num_labels\"]),\n        id2label={int(key): value for key, value in config[\"model\"][\"id2label\"].items()},\n        label2id={str(key): int(value) for key, value in config[\"model\"][\"label2id\"].items()},\n    )\n    processor = AutoImageProcessor.from_pretrained(checkpoint)\n    return ModelBundle(model=model, image_processor=processor, name=\"my_detector\")\n`;

const fallbackConstructorCatalog = {
  base_models: [
    { id: "detr_resnet50", label: "DETR ResNet-50", checkpoint: "facebook/detr-resnet-50", summary: "Balanced baseline for general UAV detection." },
    { id: "yolos_tiny", label: "YOLOS Tiny", checkpoint: "hustvl/yolos-tiny", summary: "Compact baseline when latency matters more than capacity." },
    { id: "hf_auto", label: "HF Auto", checkpoint: "facebook/detr-resnet-50", summary: "Bring your own checkpoint and keep constructor-driven heads." },
  ],
  goals: [
    { id: "balanced", label: "Balanced" },
    { id: "accuracy", label: "Accuracy" },
    { id: "latency", label: "Latency" },
    { id: "stability", label: "Stability" },
  ],
  layers: [
    { type: "linear", label: "Linear", description: "Dense projection on token features.", params: [{ name: "out_features", type: "int", default: 256 }, { name: "bias", type: "bool", default: true }] },
    { type: "gelu", label: "GELU", description: "Smooth activation for transformer-style heads.", params: [] },
    { type: "relu", label: "ReLU", description: "Low-cost activation for faster heads.", params: [{ name: "inplace", type: "bool", default: true }] },
    { type: "dropout", label: "Dropout", description: "Regularization for difficult weather or small datasets.", params: [{ name: "p", type: "float", default: 0.1 }] },
    { type: "layer_norm", label: "LayerNorm", description: "Stabilizes token features before prediction.", params: [{ name: "eps", type: "float", default: 0.00001 }] },
    { type: "transformer_encoder", label: "Transformer Encoder", description: "Adds token mixing before the prediction head.", params: [{ name: "nhead", type: "int", default: 8 }, { name: "dim_feedforward", type: "int", default: 1024 }, { name: "dropout", type: "float", default: 0.1 }, { name: "norm_first", type: "bool", default: true }] },
    { type: "residual_mlp", label: "Residual MLP", description: "Residual feed-forward refinement for dense prediction.", params: [{ name: "expansion", type: "float", default: 2.0 }, { name: "dropout", type: "float", default: 0.1 }, { name: "activation", type: "enum", default: "gelu", options: ["gelu", "relu"] }] },
    { type: "identity", label: "Identity", description: "No-op layer, useful for temporary staging.", params: [] },
  ],
  templates: {
    default_blueprint: {
      name: "custom_detector_builder",
      base_model: "detr_resnet50",
      checkpoint: "facebook/detr-resnet-50",
      goal: "balanced",
      dataset_id: "",
      labels: ["uav"],
      train_backbone: true,
      head_specs: {
        classifier: [
          { type: "linear", params: { out_features: 256, bias: true } },
          { type: "gelu", params: {} },
          { type: "dropout", params: { p: 0.1 } },
        ],
        bbox: [
          { type: "linear", params: { out_features: 256, bias: true } },
          { type: "relu", params: { inplace: true } },
          { type: "dropout", params: { p: 0.05 } },
        ],
      },
    },
  },
};

function withTimeout(url, options = {}, timeoutMs = 7000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
}

function safeReadStorage(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeWriteStorage(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // ignore storage errors
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function slugify(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "") || "item";
}

function formatBytes(size) {
  const value = Number(size) || 0;
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

function formatMetric(value, digits = 3) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : "--";
}

function formatNullable(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "--";
  return formatMetric(value, digits);
}

function formatDate(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function parseMaybeJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function debounce(fn, wait = 300) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function getConstructorCatalog() {
  return uiState.constructorCatalog || fallbackConstructorCatalog;
}

function getConstructorLayerMap() {
  return new Map((getConstructorCatalog().layers || []).map((layer) => [layer.type, layer]));
}

function makeDefaultConstructorBlueprint() {
  return deepClone(getConstructorCatalog().templates?.default_blueprint || fallbackConstructorCatalog.templates.default_blueprint);
}

function getConstructorHeadLabel(headName) {
  return headName === "bbox" ? "BBox head" : "Classifier head";
}

function getConstructorLayerTone(layerType) {
  if (["gelu", "relu"].includes(layerType)) return "activation";
  if (["dropout", "layer_norm"].includes(layerType)) return "stability";
  if (["transformer_encoder", "residual_mlp"].includes(layerType)) return "mixer";
  if (["identity"].includes(layerType)) return "utility";
  return "projection";
}

function createConstructorLayer(layerType) {
  const schema = getConstructorLayerMap().get(layerType);
  if (!schema) return null;
  return {
    type: layerType,
    params: Object.fromEntries((schema.params || []).map((param) => [param.name, param.default])),
  };
}

uiState.constructorCatalog = fallbackConstructorCatalog;
uiState.constructorBlueprint = makeDefaultConstructorBlueprint();

async function parseResponseError(response) {
  const text = await response.text();
  const payload = parseMaybeJson(text);
  if (payload && payload.detail) return String(payload.detail);
  return text || `${response.status} ${response.statusText}`;
}

async function apiRequest(path, options = {}, expect = "json") {
  const response = await withTimeout(path, options, 12000);
  if (!response.ok) {
    throw new Error(await parseResponseError(response));
  }
  if (expect === "text") return response.text();
  if (expect === "raw") return response;
  return response.json();
}

function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text);
  }
  return Promise.reject(new Error("Clipboard unavailable"));
}

function setTheme(theme, persist = true) {
  const nextTheme = allowedThemes.has(theme) ? theme : "flight";
  document.body.dataset.theme = nextTheme;
  document.querySelectorAll(".theme-btn[data-theme]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.theme === nextTheme);
  });
  if (persist) safeWriteStorage(uiStorageKeys.theme, nextTheme);
  window.dispatchEvent(new CustomEvent("uav-theme-change", { detail: { theme: nextTheme } }));
}

function setMotion(mode, persist = true) {
  const nextMode = mode === "off" ? "off" : "on";
  document.body.dataset.motion = nextMode;
  const button = document.getElementById("motion-toggle");
  if (button) {
    const enabled = nextMode === "on";
    button.textContent = `Motion: ${enabled ? "on" : "off"}`;
    button.setAttribute("aria-pressed", String(!enabled));
  }
  if (persist) safeWriteStorage(uiStorageKeys.motion, nextMode);
}

function setDensity(density, persist = true) {
  const nextDensity = allowedDensity.has(density) ? density : "compact";
  document.body.dataset.density = nextDensity;
  document.querySelectorAll(".density-btn[data-density]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.density === nextDensity);
  });
  if (persist) safeWriteStorage(uiStorageKeys.density, nextDensity);
}

function setupAppearanceControls() {
  setTheme(safeReadStorage(uiStorageKeys.theme) || document.body.dataset.theme || "flight", false);
  setMotion(safeReadStorage(uiStorageKeys.motion) || document.body.dataset.motion || "on", false);
  setDensity(safeReadStorage(uiStorageKeys.density) || document.body.dataset.density || "compact", false);
  document.querySelectorAll(".theme-btn[data-theme]").forEach((btn) => btn.addEventListener("click", () => setTheme(btn.dataset.theme)));
  document.querySelectorAll(".density-btn[data-density]").forEach((btn) => btn.addEventListener("click", () => setDensity(btn.dataset.density)));
  document.getElementById("motion-toggle")?.addEventListener("click", () => setMotion(document.body.dataset.motion === "off" ? "on" : "off"));
}

function setupQuickActionButtons() {
  document.querySelectorAll("[data-jump-tab]").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.jumpTab));
  });
}

function setServiceStatus(serviceName, isUp) {
  uiState.serviceStatus[serviceName] = Boolean(isUp);
  const element = document.querySelector(`.status-pill[data-service="${serviceName}"]`);
  if (!element) return;
  element.classList.remove("status-up", "status-down");
  element.classList.add(isUp ? "status-up" : "status-down");
}

function renderOperationalHeader() {
  const serviceValues = Object.values(uiState.serviceStatus);
  const onlineCount = serviceValues.filter((value) => value === true).length;
  const offlineCount = serviceValues.filter((value) => value === false).length;
  const runningJobs = uiState.jobs.filter((job) => job.status === "running").length;
  const trackedRuns = uiState.experimentUniverse.length || uiState.experiments.length;
  const bestRun = [...(uiState.experimentUniverse.length ? uiState.experimentUniverse : uiState.experiments)]
    .filter((item) => Number.isFinite(Number(item.map_50)))
    .sort((left, right) => Number(right.map_50) - Number(left.map_50))[0];
  const servedModels = uiState.torchserve.available ? (uiState.torchserve.models || []).length : 0;

  document.getElementById("status-online-count").textContent = String(onlineCount);
  document.getElementById("status-offline-count").textContent = String(offlineCount);
  document.getElementById("status-running-jobs").textContent = String(runningJobs);
  document.getElementById("hero-tracked-runs").textContent = String(trackedRuns);
  document.getElementById("hero-running-jobs").textContent = String(runningJobs);
  document.getElementById("hero-best-map50").textContent = bestRun ? formatMetric(bestRun.map_50, 3) : "--";
  document.getElementById("hero-served-models").textContent = String(servedModels);

  const healthLabel = document.getElementById("hero-health-label");
  const healthNote = document.getElementById("hero-health-note");
  if (!healthLabel || !healthNote) return;

  if (serviceValues.every((value) => value === null)) {
    healthLabel.textContent = "Awaiting telemetry";
    healthNote.textContent = "Service checks and catalog sync are still loading.";
    return;
  }
  if (offlineCount === 0 && onlineCount > 0) {
    healthLabel.textContent = "Stack nominal";
    healthNote.textContent = `${onlineCount}/${serviceValues.length} services reachable. TensorBoard ${uiState.tensorboard?.running ? "ready" : "standby"}.`;
    return;
  }
  if (onlineCount === 0) {
    healthLabel.textContent = "Stack offline";
    healthNote.textContent = "No service health checks succeeded yet. Inspect Control API and compose logs.";
    return;
  }
  healthLabel.textContent = "Partial readiness";
  healthNote.textContent = `${onlineCount}/${serviceValues.length} services reachable, ${offlineCount} need attention.`;
}

async function refreshServiceStatus() {
  await Promise.all(Object.entries(statusChecks).map(async ([name, url]) => {
    try {
      const response = await withTimeout(url, {}, 5000);
      if (!response.ok) {
        setServiceStatus(name, false);
        return;
      }
      if (name === "tensorboard") {
        const payload = await response.json();
        setServiceStatus(name, Boolean(payload.running));
        return;
      }
      if (name === "torchserve") {
        const payload = await response.json();
        setServiceStatus(name, Boolean(payload.available));
        return;
      }
      setServiceStatus(name, true);
    } catch {
      setServiceStatus(name, false);
    }
  }));
  renderOperationalHeader();
}

function parsePromVector(payload) {
  if (!payload || payload.status !== "success") return [];
  const result = payload.data && payload.data.result;
  if (!Array.isArray(result)) return [];
  return result.map((item) => {
    const raw = item.value && item.value[1];
    const value = Number(raw);
    if (!Number.isFinite(value)) return null;
    return { metric: item.metric || {}, value };
  }).filter(Boolean);
}

function parsePromValue(payload) {
  const vector = parsePromVector(payload);
  return vector.length ? vector[0].value : null;
}

async function queryValue(query) {
  const response = await withTimeout(`/api/prometheus/api/v1/query?query=${encodeURIComponent(query)}`, {}, 5500);
  if (!response.ok) throw new Error("prometheus_query_failed");
  const payload = await response.json();
  const value = parsePromValue(payload);
  if (value === null) throw new Error("empty_result");
  return value;
}

async function queryVector(query) {
  const response = await withTimeout(`/api/prometheus/api/v1/query?query=${encodeURIComponent(query)}`, {}, 5500);
  if (!response.ok) throw new Error("prometheus_query_failed");
  return parsePromVector(await response.json());
}

function applyMetricNoise(metricKey, last) {
  const noise = (Math.random() - 0.5) * 0.06;
  if (metricKey === "latency") return Math.max(10, last + noise * 40);
  if (metricKey === "fps") return Math.max(2, last + noise * 20);
  if (metricKey === "health") return Math.min(1, Math.max(0, last + noise * 0.4));
  if (metricKey === "firing_alerts") return Math.max(0, Math.round(last + noise * 4));
  return Math.max(0, last + noise * 30);
}

async function queryMetric(metricKey) {
  try {
    return await queryValue(metricQueries[metricKey]);
  } catch {
    const last = metricHistory[metricKey][metricHistory[metricKey].length - 1] || metricFallback[metricKey];
    return applyMetricNoise(metricKey, last);
  }
}

function drawSparkline(metricKey) {
  const card = document.querySelector(`.kpi-card[data-metric="${metricKey}"]`);
  if (!card) return;
  const values = metricHistory[metricKey];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(0.0001, max - min);
  const path = values.map((value, index) => {
    const x = (index / (values.length - 1)) * 120;
    const y = 34 - ((value - min) / span) * 28;
    return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  card.querySelector("path")?.setAttribute("d", path);
}

function setKpi(metricKey, value) {
  const card = document.querySelector(`.kpi-card[data-metric="${metricKey}"]`);
  if (!card) return;
  card.querySelector(".kpi-value").textContent = kpiFormatters[metricKey](value);
  metricHistory[metricKey].push(value);
  while (metricHistory[metricKey].length > 24) metricHistory[metricKey].shift();
  drawSparkline(metricKey);
}

async function refreshMetrics() {
  await Promise.all(Object.keys(metricQueries).map(async (metricKey) => setKpi(metricKey, await queryMetric(metricKey))));
}

function normalizeContainerName(rawName) {
  return rawName ? rawName.replace(/^\//, "") : "unknown";
}

function pickProcessName(metric) {
  return metric.groupname || metric.name || metric.comm || metric.job || "unknown";
}

function toNumberFixed(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "--";
}

function renderTableRows(tbodyId, rows, columns) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  tbody.innerHTML = rows.map((row) => `<tr>${columns.map((column) => `<td>${column(row)}</td>`).join("")}</tr>`).join("");
}
async function refreshContainerUsage() {
  try {
    const [cpuRows, memRows, rxRows] = await Promise.all([
      queryVector('topk(20, sum by (name) (rate(container_cpu_usage_seconds_total{name!=""}[5m])) * 100)'),
      queryVector('topk(20, sum by (name) (container_memory_working_set_bytes{name!=""}) / 1024 / 1024)'),
      queryVector('topk(20, sum by (name) (rate(container_network_receive_bytes_total{name!=""}[5m])) / 1024)'),
    ]);
    const byName = new Map();
    cpuRows.forEach((row) => byName.set(normalizeContainerName(row.metric.name), { name: normalizeContainerName(row.metric.name), cpu: row.value, mem: null, rx: null }));
    memRows.forEach((row) => {
      const name = normalizeContainerName(row.metric.name);
      const entry = byName.get(name) || { name, cpu: null, mem: null, rx: null };
      entry.mem = row.value;
      byName.set(name, entry);
    });
    rxRows.forEach((row) => {
      const name = normalizeContainerName(row.metric.name);
      const entry = byName.get(name) || { name, cpu: null, mem: null, rx: null };
      entry.rx = row.value;
      byName.set(name, entry);
    });
    renderTableRows("container-usage-body", Array.from(byName.values()).sort((a, b) => (b.cpu || 0) - (a.cpu || 0)).slice(0, 12), [
      (row) => escapeHtml(row.name),
      (row) => toNumberFixed(row.cpu),
      (row) => toNumberFixed(row.mem),
      (row) => toNumberFixed(row.rx),
    ]);
    return true;
  } catch {
    renderTableRows("container-usage-body", fallbackContainers, [
      (row) => escapeHtml(row.name),
      (row) => toNumberFixed(row.cpu),
      (row) => toNumberFixed(row.mem),
      (row) => toNumberFixed(row.rx),
    ]);
    return false;
  }
}

async function refreshProcessUsage() {
  try {
    const [cpuRows, memRows, threadRows] = await Promise.all([
      queryVector("topk(20, sum by (groupname) (rate(namedprocess_namegroup_cpu_seconds_total[5m])) * 100)"),
      queryVector('topk(20, sum by (groupname) (namedprocess_namegroup_memory_bytes{memtype="resident"}) / 1024 / 1024)'),
      queryVector("topk(20, sum by (groupname) (namedprocess_namegroup_num_threads))"),
    ]);
    const byName = new Map();
    cpuRows.forEach((row) => byName.set(pickProcessName(row.metric), { name: pickProcessName(row.metric), cpu: row.value, mem: null, threads: null }));
    memRows.forEach((row) => {
      const name = pickProcessName(row.metric);
      const entry = byName.get(name) || { name, cpu: null, mem: null, threads: null };
      entry.mem = row.value;
      byName.set(name, entry);
    });
    threadRows.forEach((row) => {
      const name = pickProcessName(row.metric);
      const entry = byName.get(name) || { name, cpu: null, mem: null, threads: null };
      entry.threads = row.value;
      byName.set(name, entry);
    });
    renderTableRows("process-usage-body", Array.from(byName.values()).sort((a, b) => (b.cpu || 0) - (a.cpu || 0)).slice(0, 12), [
      (row) => escapeHtml(row.name),
      (row) => toNumberFixed(row.cpu),
      (row) => toNumberFixed(row.mem),
      (row) => (Number.isFinite(row.threads) ? Math.round(row.threads).toString() : "--"),
    ]);
    return true;
  } catch {
    renderTableRows("process-usage-body", fallbackProcesses, [
      (row) => escapeHtml(row.name),
      (row) => toNumberFixed(row.cpu),
      (row) => toNumberFixed(row.mem),
      (row) => String(row.threads),
    ]);
    return false;
  }
}

async function refreshUsageBreakdown() {
  const [containersOk, processesOk] = await Promise.all([refreshContainerUsage(), refreshProcessUsage()]);
  document.getElementById("usage-updated-at").textContent =
    containersOk && processesOk
      ? `Updated from Prometheus at ${new Date().toLocaleTimeString()}`
      : `Partial data mode at ${new Date().toLocaleTimeString()} (fallback values used for unavailable metrics).`;
}

const resourceProfiles = {
  balanced: { postgres: { cpus: 1.0, memMb: 1024 }, minio: { cpus: 1.0, memMb: 1024 }, mlflow: { cpus: 0.8, memMb: 768 }, prometheus: { cpus: 1.5, memMb: 2048 }, grafana: { cpus: 0.8, memMb: 768 }, "control-api": { cpus: 1.0, memMb: 1024 }, ui: { cpus: 0.5, memMb: 256 } },
  training: { postgres: { cpus: 1.2, memMb: 1536 }, minio: { cpus: 1.2, memMb: 1536 }, mlflow: { cpus: 1.0, memMb: 1024 }, trainer: { cpus: 6.0, memMb: 12288 }, prometheus: { cpus: 2.0, memMb: 3072 }, grafana: { cpus: 1.0, memMb: 1024 }, "control-api": { cpus: 1.2, memMb: 1536 }, ui: { cpus: 0.6, memMb: 384 } },
  inference: { postgres: { cpus: 1.0, memMb: 1024 }, minio: { cpus: 1.0, memMb: 1024 }, mlflow: { cpus: 0.8, memMb: 768 }, torchserve: { cpus: 4.0, memMb: 8192 }, prometheus: { cpus: 1.6, memMb: 2048 }, grafana: { cpus: 1.0, memMb: 1024 }, "control-api": { cpus: 1.0, memMb: 1024 }, ui: { cpus: 0.6, memMb: 384 } },
  eco: { postgres: { cpus: 0.7, memMb: 768 }, minio: { cpus: 0.7, memMb: 768 }, mlflow: { cpus: 0.6, memMb: 512 }, prometheus: { cpus: 1.0, memMb: 1024 }, grafana: { cpus: 0.6, memMb: 512 }, "control-api": { cpus: 0.6, memMb: 512 }, ui: { cpus: 0.4, memMb: 192 } },
};

function formatMemValue(memMb) {
  return memMb >= 1024 ? `${(memMb / 1024).toFixed(1)}g` : `${Math.round(memMb)}m`;
}

function renderResourcePlan() {
  const profile = document.getElementById("resource-profile").value;
  const cpuMultiplier = Number(document.getElementById("resource-cpu-multiplier").value);
  const memMultiplier = Number(document.getElementById("resource-mem-multiplier").value);
  const reservationPercent = Number(document.getElementById("resource-reservation").value) / 100;
  const base = resourceProfiles[profile] || resourceProfiles.balanced;
  const lines = ["# docker-compose.override.yml", `# generated: ${new Date().toISOString()}`, "services:"];
  Object.entries(base).forEach(([serviceName, config]) => {
    const cpus = Math.max(0.1, config.cpus * cpuMultiplier);
    const memMb = Math.max(128, config.memMb * memMultiplier);
    const reservationMb = Math.max(96, memMb * reservationPercent);
    lines.push(`  ${serviceName}:`, `    cpus: \"${cpus.toFixed(2)}\"`, `    mem_limit: ${formatMemValue(memMb)}`, `    mem_reservation: ${formatMemValue(reservationMb)}`);
  });
  document.getElementById("resource-override").textContent = lines.join("\n");
}

function setHint(targetId, text, state = "info") {
  const element = document.getElementById(targetId);
  if (!element) return;
  element.textContent = text;
  element.dataset.state = state;
}

function activateTab(tabId, updateHash = true) {
  const targetPane = document.getElementById(tabId);
  if (!targetPane) return;
  document.querySelectorAll(".tab-btn[data-tab-target]").forEach((button) => button.classList.toggle("is-active", button.dataset.tabTarget === tabId));
  document.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.toggle("is-active", pane.id === tabId));
  window.dispatchEvent(new CustomEvent("uav-tab-activate", { detail: { tabId } }));
  if (updateHash) history.replaceState(null, "", `#${tabId}`);
}

function setupTabs() {
  const workspace = document.querySelector(".workspace");
  document.querySelectorAll(".tab-btn[data-tab-target]").forEach((button) => button.addEventListener("click", () => {
    activateTab(button.dataset.tabTarget);
    workspace?.classList.remove("menu-open");
  }));
  document.getElementById("menu-toggle")?.addEventListener("click", () => workspace?.classList.toggle("menu-open"));
  const hashTarget = window.location.hash.replace("#", "");
  activateTab(hashTarget || document.querySelector(".tab-btn.is-active")?.dataset.tabTarget || "tab-overview", false);
}

function readPanelLayoutState() {
  return parseMaybeJson(safeReadStorage(uiStorageKeys.layout)) || {};
}

function writePanelLayoutState(layouts) {
  safeWriteStorage(uiStorageKeys.layout, JSON.stringify(layouts));
}

function getPanePanels(pane) {
  return Array.from(pane?.querySelectorAll(":scope > .panel") || []);
}

function ensurePanelLayoutId(panel, index = 0) {
  if (!panel) return "";
  if (panel.dataset.panelId) return panel.dataset.panelId;
  const paneId = panel.closest(".tab-pane-grid")?.id || "workspace";
  const title = panel.querySelector(".panel-title-row h2")?.textContent || panel.querySelector("h2")?.textContent || `panel_${index}`;
  const panelId = `${paneId}:${slugify(title)}`;
  panel.dataset.panelId = panelId;
  return panelId;
}

function persistPaneLayout(pane) {
  if (!pane?.id) return;
  const layouts = readPanelLayoutState();
  layouts[pane.id] = getPanePanels(pane).map((panel, index) => ensurePanelLayoutId(panel, index));
  writePanelLayoutState(layouts);
}

function restorePaneLayout(pane) {
  if (!pane?.id) return;
  const layouts = readPanelLayoutState();
  const savedOrder = Array.isArray(layouts[pane.id]) ? layouts[pane.id] : [];
  if (!savedOrder.length) return;
  const panels = getPanePanels(pane);
  const panelMap = new Map(panels.map((panel, index) => [ensurePanelLayoutId(panel, index), panel]));
  const ordered = savedOrder.map((panelId) => panelMap.get(panelId)).filter(Boolean);
  panels.forEach((panel) => {
    if (!ordered.includes(panel)) ordered.push(panel);
  });
  ordered.forEach((panel) => pane.appendChild(panel));
}

function clearPanelDropIndicators() {
  document.querySelectorAll(".panel.is-drop-target").forEach((panel) => {
    panel.classList.remove("is-drop-target");
    delete panel.dataset.dropPosition;
  });
}

function clearPanelDragState() {
  panelDragState.panelId = "";
  panelDragState.sourcePaneId = "";
  panelDragState.targetPaneId = "";
  panelDragState.element?.classList.remove("is-dragging");
  panelDragState.element = null;
  document.body.classList.remove("is-panel-dragging");
  document.querySelectorAll(".tab-pane-grid.is-drop-zone").forEach((pane) => pane.classList.remove("is-drop-zone"));
  clearPanelDropIndicators();
}

function resolveDropPosition(panel, clientX, clientY) {
  const rect = panel.getBoundingClientRect();
  const withinVerticalBand = clientY >= rect.top && clientY <= rect.bottom;
  const withinHorizontalBand = clientX >= rect.left && clientX <= rect.right;
  const horizontalDistance = Math.abs(clientX - (rect.left + rect.width / 2));
  const verticalDistance = Math.abs(clientY - (rect.top + rect.height / 2));
  if (withinVerticalBand && horizontalDistance > verticalDistance) {
    return clientX < rect.left + rect.width / 2 ? "before" : "after";
  }
  if (withinHorizontalBand && verticalDistance >= horizontalDistance) {
    return clientY < rect.top + rect.height / 2 ? "before" : "after";
  }
  return clientY < rect.top + rect.height / 2 ? "before" : "after";
}

function findNearestPanel(pane, clientX, clientY) {
  const candidates = getPanePanels(pane).filter((panel) => panel !== panelDragState.element);
  if (!candidates.length) return null;
  return candidates.reduce((best, panel) => {
    const rect = panel.getBoundingClientRect();
    const dx = clientX - (rect.left + rect.width / 2);
    const dy = clientY - (rect.top + rect.height / 2);
    const distance = Math.hypot(dx, dy);
    if (!best || distance < best.distance) return { panel, distance };
    return best;
  }, null)?.panel || null;
}

function updatePaneDropIndicator(pane, clientX, clientY) {
  const nearestPanel = findNearestPanel(pane, clientX, clientY);
  clearPanelDropIndicators();
  if (!nearestPanel) return null;
  nearestPanel.dataset.dropPosition = resolveDropPosition(nearestPanel, clientX, clientY);
  nearestPanel.classList.add("is-drop-target");
  return nearestPanel;
}

function commitPanelDrop(targetPane, targetPanel = null) {
  const sourcePanel = panelDragState.element;
  if (!sourcePanel || !targetPane) return;
  if (targetPanel && targetPanel !== sourcePanel) {
    const position = targetPanel.dataset.dropPosition || "before";
    targetPane.insertBefore(sourcePanel, position === "after" ? targetPanel.nextSibling : targetPanel);
  } else {
    targetPane.appendChild(sourcePanel);
  }
  persistPaneLayout(targetPane);
  if (panelDragState.sourcePaneId && panelDragState.sourcePaneId !== targetPane.id) {
    const sourcePane = document.getElementById(panelDragState.sourcePaneId);
    if (sourcePane) persistPaneLayout(sourcePane);
  }
  clearPanelDragState();
}

function decoratePanelTitleRow(panel) {
  const row = panel.querySelector(".panel-title-row");
  if (!row || row.dataset.decorated === "true") return;
  row.dataset.decorated = "true";
  const children = Array.from(row.children);
  const heading = children[0] || null;
  const side = document.createElement("div");
  side.className = "panel-header-meta";
  children.slice(1).forEach((child) => side.appendChild(child));
  const handle = document.createElement("button");
  handle.type = "button";
  handle.className = "panel-drag-handle";
  handle.draggable = true;
  handle.title = "Drag panel";
  handle.setAttribute("aria-label", "Drag panel");
  handle.innerHTML = '<span></span><span></span><span></span>';
  handle.addEventListener("dragstart", (event) => {
    const sourcePanel = event.currentTarget.closest(".panel");
    const sourcePane = sourcePanel?.closest(".tab-pane-grid");
    if (!sourcePanel || !sourcePane) return;
    panelDragState.panelId = ensurePanelLayoutId(sourcePanel);
    panelDragState.sourcePaneId = sourcePane.id;
    panelDragState.targetPaneId = sourcePane.id;
    panelDragState.element = sourcePanel;
    sourcePanel.classList.add("is-dragging");
    document.body.classList.add("is-panel-dragging");
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", panelDragState.panelId);
  });
  handle.addEventListener("dragend", () => clearPanelDragState());
  side.appendChild(handle);
  if (heading) {
    row.replaceChildren(heading, side);
  } else {
    row.replaceChildren(side);
  }
}

function bindPanelDragEvents(panel) {
  if (!panel || panel.dataset.dragBound === "true") return;
  panel.dataset.dragBound = "true";
  panel.classList.add("layout-panel");
  panel.addEventListener("dragover", (event) => {
    if (!panelDragState.element || panel === panelDragState.element) return;
    event.preventDefault();
    clearPanelDropIndicators();
    panel.dataset.dropPosition = resolveDropPosition(panel, event.clientX, event.clientY);
    panel.classList.add("is-drop-target");
  });
  panel.addEventListener("dragleave", (event) => {
    if (!panel.contains(event.relatedTarget)) {
      panel.classList.remove("is-drop-target");
      delete panel.dataset.dropPosition;
    }
  });
  panel.addEventListener("drop", (event) => {
    if (!panelDragState.element || panel === panelDragState.element) return;
    event.preventDefault();
    const targetPane = panel.closest(".tab-pane-grid");
    if (!targetPane) return;
    commitPanelDrop(targetPane, panel);
  });
}

function setupPanelGridLayout() {
  const panes = Array.from(document.querySelectorAll(".tab-pane-grid"));
  panes.forEach((pane) => {
    getPanePanels(pane).forEach((panel, index) => {
      panel.dataset.defaultOrder = panel.dataset.defaultOrder || String(index);
      ensurePanelLayoutId(panel, index);
      decoratePanelTitleRow(panel);
      bindPanelDragEvents(panel);
    });
    restorePaneLayout(pane);
    pane.addEventListener("dragover", (event) => {
      if (!panelDragState.element) return;
      event.preventDefault();
      if (!event.target.closest(".panel")) updatePaneDropIndicator(pane, event.clientX, event.clientY);
      pane.classList.add("is-drop-zone");
      panelDragState.targetPaneId = pane.id;
    });
    pane.addEventListener("dragleave", (event) => {
      if (!pane.contains(event.relatedTarget)) pane.classList.remove("is-drop-zone");
    });
    pane.addEventListener("drop", (event) => {
      if (!panelDragState.element) return;
      event.preventDefault();
      pane.classList.remove("is-drop-zone");
      const targetPanel = event.target.closest(".panel") || pane.querySelector(".panel.is-drop-target");
      commitPanelDrop(pane, targetPanel);
    });
  });
  document.getElementById("layout-reset")?.addEventListener("click", () => {
    try {
      window.localStorage.removeItem(uiStorageKeys.layout);
    } catch {
      // ignore storage errors
    }
    panes.forEach((pane) => {
      getPanePanels(pane)
        .sort((left, right) => Number(left.dataset.defaultOrder || 0) - Number(right.dataset.defaultOrder || 0))
        .forEach((panel) => pane.appendChild(panel));
    });
    clearPanelDragState();
  });
}

function resolveDashboardSrc(rawSrc) {
  try {
    const url = new URL(rawSrc, window.location.origin);
    url.searchParams.set("theme", document.body.dataset.theme === "paper" ? "light" : "dark");
    return `${url.pathname}${url.search}`;
  } catch {
    return rawSrc;
  }
}

function setupDashboardSwitcher() {
  const frame = document.getElementById("dashboard-frame");
  const shell = document.getElementById("dashboard-shell");
  const loader = document.getElementById("dashboard-loader");
  const status = document.getElementById("dashboard-status");
  const openLink = document.getElementById("dashboard-open");
  const tabs = Array.from(document.querySelectorAll(".dash-tab[data-src]"));
  let loadingTimeout = null;
  let activeTab = tabs[0] || null;
  const setLoading = (isLoading) => {
    shell.classList.toggle("loading", isLoading);
    loader?.setAttribute("aria-hidden", String(!isLoading));
  };
  const setStatus = (text, state = "info") => {
    status.textContent = text;
    status.dataset.state = state;
  };
  const activateDashboard = (tab, forceReload = false) => {
    activeTab = tab;
    tabs.forEach((item) => item.classList.toggle("is-active", item === tab));
    const src = resolveDashboardSrc(tab.dataset.src || "");
    if (!src) return;
    openLink.href = src;
    if (forceReload || frame.getAttribute("src") !== src) {
      setLoading(true);
      setStatus("Loading dashboard...", "info");
      clearTimeout(loadingTimeout);
      loadingTimeout = setTimeout(() => {
        setLoading(false);
        setStatus("Dashboard loads too long. Use Open full for diagnostics.", "warn");
      }, 12000);
      frame.setAttribute("src", src);
    }
  };
  frame.addEventListener("load", () => {
    clearTimeout(loadingTimeout);
    setLoading(false);
    setStatus("Dashboard ready", "ok");
  });
  frame.addEventListener("error", () => {
    clearTimeout(loadingTimeout);
    setLoading(false);
    setStatus("Dashboard failed to load.", "error");
  });
  tabs.forEach((tab) => tab.addEventListener("click", () => activateDashboard(tab)));
  document.getElementById("dashboard-reload")?.addEventListener("click", () => activateDashboard(activeTab || tabs[0], true));
  window.addEventListener("uav-theme-change", () => activeTab && activateDashboard(activeTab, true));
  window.addEventListener("uav-tab-activate", (event) => event?.detail?.tabId === "tab-overview" && !frame.getAttribute("src") && activateDashboard(activeTab || tabs[0], true));
  if (tabs[0]) activateDashboard(tabs[0]);
}
function setupTensorBoardFrame() {
  const frame = document.getElementById("tensorboard-frame");
  const shell = document.getElementById("tensorboard-shell");
  const loader = document.getElementById("tensorboard-loader");
  const status = document.getElementById("tensorboard-status");
  let loadingTimeout = null;
  const setLoading = (isLoading) => {
    shell.classList.toggle("loading", isLoading);
    loader?.setAttribute("aria-hidden", String(!isLoading));
  };
  const setStatus = (text, state = "info") => {
    status.textContent = text;
    status.dataset.state = state;
  };
  const loadTensorBoard = async (forceReload = false) => {
    try {
      const tensorboard = await apiRequest("/api/control/tensorboard/start", { method: "POST" });
      uiState.tensorboard = tensorboard;
      renderOperationalHeader();
      if (!tensorboard.running) {
        setStatus("TensorBoard is not running.", "warn");
        return;
      }
      if (forceReload || frame.getAttribute("src") !== "/api/tensorboard/") {
        setLoading(true);
        setStatus("Loading TensorBoard...", "info");
        clearTimeout(loadingTimeout);
        loadingTimeout = setTimeout(() => {
          setLoading(false);
          setStatus("TensorBoard is taking too long to respond.", "warn");
        }, 12000);
        frame.setAttribute("src", "/api/tensorboard/");
      }
    } catch (error) {
      setLoading(false);
      setStatus(String(error.message || error), "error");
    }
  };
  frame.addEventListener("load", () => {
    clearTimeout(loadingTimeout);
    setLoading(false);
    setStatus("TensorBoard ready", "ok");
    renderOperationalHeader();
  });
  frame.addEventListener("error", () => {
    clearTimeout(loadingTimeout);
    setLoading(false);
    setStatus("TensorBoard failed to load.", "error");
    renderOperationalHeader();
  });
  document.getElementById("tensorboard-reload")?.addEventListener("click", () => loadTensorBoard(true));
  window.addEventListener("uav-tab-activate", (event) => event?.detail?.tabId === "tab-experiments" && !frame.getAttribute("src") && loadTensorBoard(true));
  loadTensorBoard(false);
}

function renderSummary() {
  document.getElementById("summary-datasets").textContent = String(uiState.datasets.length);
  document.getElementById("summary-configs").textContent = String(uiState.configs.length);
  document.getElementById("summary-architectures").textContent = String(uiState.architectures.length);
  document.getElementById("summary-experiments").textContent = String(uiState.experimentUniverse.length || uiState.experiments.length);
  document.getElementById("summary-jobs").textContent = String(uiState.jobs.length);
  document.getElementById("summary-running").textContent = String(uiState.jobs.filter((job) => job.status === "running").length);
  renderOperationalHeader();
}

function renderRecommendations() {
  const container = document.getElementById("recommendations-list");
  if (!uiState.recommendations.length) {
    container.innerHTML = '<div class="empty-state">No recommendations yet. Run training or evaluation to populate this block.</div>';
    return;
  }
  container.innerHTML = uiState.recommendations.map((item, index) => `
    <article class="recommendation-card">
      <p class="recommendation-rank">Top ${index + 1}</p>
      <h4>${escapeHtml(item.run_name)}</h4>
      <p>${escapeHtml(item.summary)}</p>
      <strong>score ${formatMetric(item.score, 4)}</strong>
    </article>
  `).join("");
}

function renderDatasets() {
  const search = (document.getElementById("dataset-search")?.value || "").trim().toLowerCase();
  const rows = uiState.datasets.filter((dataset) => !search || [dataset.name, dataset.path, ...(dataset.tags || [])].join(" ").toLowerCase().includes(search));
  document.getElementById("dataset-list").innerHTML = rows.length ? rows.map((dataset) => `
    <article class="dataset-card">
      <div class="dataset-card-head"><div><h3>${escapeHtml(dataset.name)}</h3><p>${escapeHtml(dataset.path)}</p></div><button class="btn btn-inline dataset-download" type="button" data-dataset-id="${escapeHtml(dataset.id)}">Download</button></div>
      <div class="chip-row">${(dataset.tags || []).map((tag) => `<span class="chip">${escapeHtml(tag)}</span>`).join("") || '<span class="chip">untagged</span>'}</div>
      <div class="dataset-stats"><span>${dataset.file_count || 0} files</span><span>${formatBytes(dataset.size_bytes)}</span><span>${dataset.updated_at ? formatDate(dataset.updated_at) : "no metadata yet"}</span></div>
      <p class="dataset-description">${escapeHtml(dataset.description || "No description")}</p>
    </article>
  `).join("") : '<div class="empty-state">No datasets found for the current filter.</div>';
}

function renderConfigs() {
  const search = (document.getElementById("config-search")?.value || "").trim().toLowerCase();
  const rows = uiState.configs.filter((item) => !search || [item.name, item.experiment_name, item.model_name].join(" ").toLowerCase().includes(search));
  document.getElementById("config-list").innerHTML = rows.length ? rows.map((item) => `
    <button class="list-item ${item.name === uiState.selectedConfigName ? "is-selected" : ""}" type="button" data-config-load="${escapeHtml(item.name)}">
      <span>${escapeHtml(item.name)}</span>
      <small>${escapeHtml(item.model_name || "unknown model")} • ${item.epochs || "--"} ep</small>
    </button>
  `).join("") : '<div class="empty-state">No configs match the current filter.</div>';
}

function renderArchitectures() {
  const search = (document.getElementById("architecture-search")?.value || "").trim().toLowerCase();
  const rows = uiState.architectures.filter((item) => !search || [item.id, item.name, item.description, ...(item.tags || [])].join(" ").toLowerCase().includes(search));
  document.getElementById("architecture-list").innerHTML = rows.length ? rows.map((item) => `
    <button class="list-item ${item.id === uiState.selectedArchitectureId ? "is-selected" : ""}" type="button" data-architecture-load="${escapeHtml(item.id)}">
      <span>${escapeHtml(item.name)}</span>
      <small>${escapeHtml(item.kind)}${item.has_blueprint ? " • builder" : ""}${item.tags?.length ? ` • ${escapeHtml(item.tags.join(", "))}` : ""}</small>
    </button>
  `).join("") : '<div class="empty-state">No architectures match the current filter.</div>';
}

function populateConstructorDatasetOptions() {
  const select = document.getElementById("constructor-dataset");
  if (!select) return;
  const currentValue = uiState.constructorBlueprint?.dataset_id || "";
  const options = ['<option value="">Auto / none</option>'].concat(
    uiState.datasets.map((dataset) => `<option value="${escapeHtml(dataset.id)}">${escapeHtml(dataset.name)} • ${dataset.file_count || 0} files</option>`),
  );
  select.innerHTML = options.join("");
  select.value = uiState.datasets.some((dataset) => dataset.id === currentValue) ? currentValue : "";
}

function renderConstructorSummary() {
  const container = document.getElementById("constructor-summary");
  if (!container) return;
  const preview = uiState.constructorPreview;
  if (!preview) {
    container.innerHTML = '<div class="empty-state">Generate a preview or request a recommendation to see constructor notes.</div>';
    return;
  }
  const cards = [];
  if (preview.summary) {
    cards.push(`
      <article class="recommendation-card">
        <p class="recommendation-rank">Preview</p>
        <h4>${escapeHtml(preview.summary.model_slug || uiState.constructorBlueprint?.name || "custom_detector")}</h4>
        <p>Base: ${escapeHtml(preview.summary.base_model || "--")} • classifier: ${preview.summary.classifier_layers || 0} • bbox: ${preview.summary.bbox_layers || 0}</p>
        <strong>${preview.summary.train_backbone ? "Backbone trainable" : "Backbone frozen"}</strong>
      </article>
    `);
  }
  (preview.notes || []).forEach((note, index) => {
    cards.push(`
      <article class="recommendation-card">
        <p class="recommendation-rank">Note ${index + 1}</p>
        <p>${escapeHtml(note)}</p>
      </article>
    `);
  });
  container.innerHTML = cards.join("") || '<div class="empty-state">No constructor guidance yet.</div>';
}

function renderConstructorLayerCatalog() {
  const container = document.getElementById("constructor-layer-catalog");
  if (!container) return;
  const layers = getConstructorCatalog().layers || [];
  container.innerHTML = layers.map((layer) => `
    <article class="layer-palette-card layer-tone-${escapeHtml(getConstructorLayerTone(layer.type))}" draggable="true" data-layer-drag-type="${escapeHtml(layer.type)}">
      <div class="layer-palette-meta">
        <span class="layer-family">${escapeHtml(getConstructorLayerTone(layer.type))}</span>
        <span class="layer-drop-note">Drag</span>
      </div>
      <div>
        <h4>${escapeHtml(layer.label)}</h4>
        <p>${escapeHtml(layer.description || "")}</p>
      </div>
      <div class="chip-row layer-param-chips">
        ${(layer.params || []).length ? layer.params.map((param) => `<span class="chip">${escapeHtml(param.name)}</span>`).join("") : '<span class="chip">auto</span>'}
      </div>
      <div class="inline-actions compact-actions">
        <button class="btn btn-inline" type="button" data-layer-add="${escapeHtml(layer.type)}" data-layer-target="classifier">Add to classifier</button>
        <button class="btn btn-inline" type="button" data-layer-add="${escapeHtml(layer.type)}" data-layer-target="bbox">Add to bbox</button>
      </div>
    </article>
  `).join("");
}

function renderConstructorStack(headName) {
  const container = document.getElementById(`constructor-${headName}-stack`);
  if (!container) return;
  const blueprint = uiState.constructorBlueprint || makeDefaultConstructorBlueprint();
  const layers = blueprint.head_specs?.[headName] || [];
  const layerMap = getConstructorLayerMap();
  const cards = layers.map((layer, index) => {
    const schema = layerMap.get(layer.type) || { label: layer.type, params: [] };
    const paramsHtml = (schema.params || []).map((param) => {
      const value = layer.params?.[param.name];
      if (param.type === "bool") {
        return `<label class="layer-param"><span>${escapeHtml(param.name)}</span><input type="checkbox" data-layer-param="${escapeHtml(param.name)}" data-head-name="${escapeHtml(headName)}" data-layer-index="${index}" ${value ? "checked" : ""} /></label>`;
      }
      if (param.type === "enum") {
        return `<label class="layer-param"><span>${escapeHtml(param.name)}</span><select data-layer-param="${escapeHtml(param.name)}" data-head-name="${escapeHtml(headName)}" data-layer-index="${index}">${(param.options || []).map((option) => `<option value="${escapeHtml(option)}" ${String(value) === String(option) ? "selected" : ""}>${escapeHtml(option)}</option>`).join("")}</select></label>`;
      }
      const inputType = param.type === "int" || param.type === "float" ? "number" : "text";
      const step = param.type === "int" ? "1" : "any";
      return `<label class="layer-param"><span>${escapeHtml(param.name)}</span><input type="${inputType}" step="${step}" value="${escapeHtml(value ?? param.default ?? "")}" data-layer-param="${escapeHtml(param.name)}" data-head-name="${escapeHtml(headName)}" data-layer-index="${index}" /></label>`;
    }).join("");
    return `
      <button class="stack-drop-slot" type="button" data-drop-head="${escapeHtml(headName)}" data-drop-index="${index}">
        <span>Drop into ${escapeHtml(getConstructorHeadLabel(headName))}</span>
      </button>
      <article class="layer-card layer-tone-${escapeHtml(getConstructorLayerTone(layer.type))}" draggable="true" data-stack-layer="true" data-head-name="${escapeHtml(headName)}" data-layer-index="${index}">
        <div class="layer-card-head">
          <div>
            <div class="layer-card-meta"><span class="layer-order">#${index + 1}</span><span class="layer-family">${escapeHtml(getConstructorLayerTone(layer.type))}</span></div>
            <h4>${escapeHtml(schema.label || layer.type)}</h4>
            <p>${escapeHtml(layer.type)}</p>
          </div>
          <div class="inline-actions compact-actions">
            <span class="drag-handle" aria-hidden="true">::</span>
            <button class="btn btn-inline" type="button" data-layer-move="up" data-head-name="${escapeHtml(headName)}" data-layer-index="${index}">Up</button>
            <button class="btn btn-inline" type="button" data-layer-move="down" data-head-name="${escapeHtml(headName)}" data-layer-index="${index}">Down</button>
            <button class="btn btn-inline" type="button" data-layer-remove="true" data-head-name="${escapeHtml(headName)}" data-layer-index="${index}">Remove</button>
          </div>
        </div>
        <div class="layer-param-grid">${paramsHtml || '<div class="empty-state">No parameters for this layer.</div>'}</div>
      </article>
    `;
  }).join("");

  container.innerHTML = `
    <div class="builder-lane-copy">
      <strong>${escapeHtml(getConstructorHeadLabel(headName))}</strong>
      <span>${layers.length ? `${layers.length} block${layers.length === 1 ? "" : "s"} in this lane` : "Drop your first block here"}</span>
    </div>
    ${!layers.length ? `<div class="empty-state builder-empty">This lane is empty. Drag a tile here or tap “Add to ${escapeHtml(headName)}”.</div>` : ""}
    ${cards}
    <button class="stack-drop-slot stack-drop-slot-end" type="button" data-drop-head="${escapeHtml(headName)}" data-drop-index="${layers.length}">
      <span>Add to the end of ${escapeHtml(getConstructorHeadLabel(headName))}</span>
    </button>
  `;
}

function renderConstructorBuilder() {
  const blueprint = uiState.constructorBlueprint || makeDefaultConstructorBlueprint();
  document.getElementById("constructor-name").value = blueprint.name || "";
  document.getElementById("constructor-checkpoint").value = blueprint.checkpoint || "";
  document.getElementById("constructor-labels").value = (blueprint.labels || []).join(", ");

  const baseModelSelect = document.getElementById("constructor-base-model");
  const goalSelect = document.getElementById("constructor-goal");
  if (baseModelSelect) {
    baseModelSelect.innerHTML = (getConstructorCatalog().base_models || []).map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}</option>`).join("");
    baseModelSelect.value = blueprint.base_model || "";
  }
  if (goalSelect) {
    goalSelect.innerHTML = (getConstructorCatalog().goals || []).map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}</option>`).join("");
    goalSelect.value = blueprint.goal || "balanced";
  }

  populateConstructorDatasetOptions();
  renderConstructorLayerCatalog();
  renderConstructorStack("classifier");
  renderConstructorStack("bbox");
  renderConstructorSummary();
}

function renderExperiments() {
  const tbody = document.getElementById("experiments-body");
  tbody.innerHTML = uiState.experiments.length ? uiState.experiments.map((item) => `
    <tr class="experiment-row ${item.key === uiState.selectedExperimentKey ? "row-selected" : ""}" data-experiment-key="${escapeHtml(item.key)}">
      <td><input class="compare-checkbox" type="checkbox" data-compare-key="${escapeHtml(item.key)}" ${uiState.compareKeys.has(item.key) ? "checked" : ""} /></td>
      <td>${escapeHtml(item.run_name)}</td>
      <td>${escapeHtml(item.model_name || "unknown")}</td>
      <td><span class="job-badge status-${escapeHtml(item.status || "unknown")}">${escapeHtml(item.status || "unknown")}</span></td>
      <td>${formatNullable(item.map_50, 3)}</td>
      <td>${formatNullable(item.fps, 2)}</td>
      <td>${formatNullable(item.latency_ms, 2)}</td>
      <td>${item.rating ?? "--"}</td>
      <td>${(item.tags || []).map((tag) => `<span class="chip inline-chip">${escapeHtml(tag)}</span>`).join(" ")}</td>
    </tr>
  `).join("") : '<tr><td colspan="9" class="empty-cell">No experiments for current filters.</td></tr>';
}

function renderCompareTable() {
  const selected = uiState.experiments.filter((item) => uiState.compareKeys.has(item.key));
  document.getElementById("compare-shell").innerHTML = selected.length ? `
    <table>
      <thead><tr><th>Run</th><th>Model</th><th>mAP50</th><th>mAP75</th><th>FPS</th><th>Latency</th><th>Rating</th></tr></thead>
      <tbody>${selected.map((item) => `
        <tr><td>${escapeHtml(item.run_name)}</td><td>${escapeHtml(item.model_name || "unknown")}</td><td>${formatNullable(item.map_50, 3)}</td><td>${formatNullable(item.map_75, 3)}</td><td>${formatNullable(item.fps, 2)}</td><td>${formatNullable(item.latency_ms, 2)}</td><td>${item.rating ?? "--"}</td></tr>
      `).join("")}</tbody>
    </table>
  ` : '<div class="empty-state">Select runs in the table to compare them here.</div>';
}

function populateExperimentMetadata() {
  const selected = uiState.experiments.find((item) => item.key === uiState.selectedExperimentKey);
  document.getElementById("experiment-selected-badge").textContent = selected ? selected.run_name : "No selection";
  document.getElementById("experiment-tags").value = selected?.tags?.join(", ") || "";
  document.getElementById("experiment-rating").value = selected?.rating ?? "";
  document.getElementById("experiment-note").value = selected?.note || "";
}

function renderJobs() {
  document.getElementById("job-list").innerHTML = uiState.jobs.length ? uiState.jobs.map((job) => `
    <article class="job-card ${job.id === uiState.selectedJobId ? "is-selected" : ""}">
      <div class="job-card-head"><div><h3>${escapeHtml(job.run_name || job.id)}</h3><p>${escapeHtml(job.kind)} • ${escapeHtml(job.experiment_name || "--")}</p></div><span class="job-badge status-${escapeHtml(job.status)}">${escapeHtml(job.status)}</span></div>
      <div class="job-meta"><span>${escapeHtml(job.output_dir || "--")}</span><span>${job.pid || "--"}</span></div>
      <div class="inline-actions compact-actions"><button class="btn btn-inline" type="button" data-job-logs="${escapeHtml(job.id)}">Logs</button>${job.status === "running" ? `<button class="btn btn-inline" type="button" data-job-stop="${escapeHtml(job.id)}">Stop</button>` : ""}</div>
    </article>
  `).join("") : '<div class="empty-state">No jobs have been launched yet.</div>';
  renderOperationalHeader();
}

function renderTorchServeModels() {
  const rows = uiState.torchserve.models || [];
  document.getElementById("torchserve-models").innerHTML = !uiState.torchserve.available
    ? '<div class="empty-state">TorchServe is not available in the current stack profile.</div>'
    : rows.length
      ? rows.map((item) => `
        <article class="torchserve-card"><div><h3>${escapeHtml(item.modelName || item.model_name || "unknown")}</h3><p>${escapeHtml(item.modelUrl || item.model_url || "model-store")}</p></div><button class="btn btn-inline" type="button" data-torchserve-delete="${escapeHtml(item.modelName || item.model_name || "")}">Unregister</button></article>
      `).join("")
      : '<div class="empty-state">No TorchServe models are currently registered.</div>';
  renderOperationalHeader();
}
function updateModelFilterOptions() {
  const select = document.getElementById("experiment-model-filter");
  const currentValue = select.value;
  const modelNames = Array.from(new Set((uiState.experimentUniverse.length ? uiState.experimentUniverse : uiState.experiments).map((item) => item.model_name || "unknown"))).sort();
  select.innerHTML = ['<option value="">all</option>'].concat(modelNames.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)).join("");
  select.value = modelNames.includes(currentValue) || currentValue === "" ? currentValue : "";
}

function resetConfigEditor() {
  uiState.selectedConfigName = "";
  document.getElementById("config-name").value = "new_experiment";
  document.getElementById("config-editor").value = defaultConfigText;
  document.getElementById("config-active-badge").textContent = "New config";
  renderConfigs();
}

function resetArchitectureEditor() {
  uiState.selectedArchitectureId = "";
  document.getElementById("architecture-name").value = "my_detector";
  document.getElementById("architecture-tags").value = "custom, research";
  document.getElementById("architecture-description").value = "Custom object detector for UAV scenes.";
  document.getElementById("architecture-config").value = defaultArchitectureConfigText;
  document.getElementById("architecture-source").value = defaultArchitectureSource;
  document.getElementById("architecture-active-badge").textContent = "New model";
  resetConstructorBuilder();
  renderArchitectures();
}

function mergeUniverse(existing, next) {
  const byKey = new Map();
  [...existing, ...next].forEach((item) => byKey.set(item.key, item));
  return Array.from(byKey.values());
}

async function loadCatalog() {
  const payload = await apiRequest("/api/control/catalog");
  uiState.datasets = payload.datasets || [];
  uiState.configs = payload.configs || [];
  uiState.architectures = payload.architectures || [];
  uiState.jobs = payload.jobs || [];
  uiState.experiments = payload.experiments || [];
  uiState.experimentUniverse = payload.experiments || [];
  uiState.recommendations = payload.recommendations || [];
  uiState.tensorboard = payload.tensorboard || null;
  renderSummary();
  renderRecommendations();
  renderDatasets();
  renderConfigs();
  renderArchitectures();
  renderConstructorBuilder();
  renderExperiments();
  renderCompareTable();
  populateExperimentMetadata();
  renderJobs();
  updateModelFilterOptions();
}

async function loadDatasets() {
  uiState.datasets = (await apiRequest("/api/control/datasets")).items || [];
  renderDatasets();
  populateConstructorDatasetOptions();
  renderSummary();
}

async function loadConfigs() {
  uiState.configs = (await apiRequest("/api/control/configs")).items || [];
  renderConfigs();
  renderSummary();
}

async function loadConfigDetail(name) {
  const payload = await apiRequest(`/api/control/configs/${encodeURIComponent(name)}`);
  uiState.selectedConfigName = payload.name;
  document.getElementById("config-name").value = payload.name;
  document.getElementById("config-editor").value = payload.config_yaml || defaultConfigText;
  document.getElementById("config-active-badge").textContent = payload.name;
  renderConfigs();
}

async function loadArchitectures() {
  uiState.architectures = (await apiRequest("/api/control/architectures")).items || [];
  renderArchitectures();
  renderSummary();
}

async function loadArchitectureDetail(id) {
  const payload = await apiRequest(`/api/control/architectures/${encodeURIComponent(id)}`);
  uiState.selectedArchitectureId = payload.id;
  document.getElementById("architecture-name").value = payload.name || payload.id;
  document.getElementById("architecture-tags").value = (payload.tags || []).join(", ");
  document.getElementById("architecture-description").value = payload.description || "";
  document.getElementById("architecture-config").value = payload.config_yaml || defaultArchitectureConfigText;
  document.getElementById("architecture-source").value = payload.source_code || defaultArchitectureSource;
  document.getElementById("architecture-active-badge").textContent = payload.name || payload.id;
  if (payload.blueprint) {
    uiState.constructorBlueprint = deepClone(payload.blueprint);
  } else {
    uiState.constructorBlueprint = makeDefaultConstructorBlueprint();
    uiState.constructorBlueprint.name = payload.name || payload.id || "custom_detector_builder";
  }
  renderConstructorBuilder();
  renderArchitectures();
}

function collectConstructorBlueprintFromForm() {
  const blueprint = deepClone(uiState.constructorBlueprint || makeDefaultConstructorBlueprint());
  blueprint.name = document.getElementById("constructor-name").value.trim() || blueprint.name || "custom_detector_builder";
  blueprint.base_model = document.getElementById("constructor-base-model").value || blueprint.base_model;
  blueprint.goal = document.getElementById("constructor-goal").value || blueprint.goal;
  blueprint.checkpoint = document.getElementById("constructor-checkpoint").value.trim() || blueprint.checkpoint;
  blueprint.dataset_id = document.getElementById("constructor-dataset").value || "";
  blueprint.labels = document.getElementById("constructor-labels").value.split(",").map((item) => item.trim()).filter(Boolean);
  return blueprint;
}

function syncConstructorPreview(preview, message) {
  uiState.constructorPreview = preview;
  uiState.constructorBlueprint = deepClone(preview.blueprint || uiState.constructorBlueprint || makeDefaultConstructorBlueprint());
  document.getElementById("architecture-name").value = uiState.constructorBlueprint.name || "custom_detector_builder";
  document.getElementById("architecture-config").value = preview.config_yaml || defaultArchitectureConfigText;
  document.getElementById("architecture-source").value = preview.source_code || defaultArchitectureSource;
  document.getElementById("architecture-active-badge").textContent = preview.summary?.model_slug || uiState.constructorBlueprint.name || "Builder preview";
  renderConstructorBuilder();
  setHint("constructor-status", message, "ok");
}

async function loadConstructorCatalog() {
  try {
    uiState.constructorCatalog = await apiRequest("/api/control/architectures/constructor/catalog");
    if (!uiState.constructorBlueprint) {
      uiState.constructorBlueprint = makeDefaultConstructorBlueprint();
    }
  } catch {
    uiState.constructorCatalog = fallbackConstructorCatalog;
    if (!uiState.constructorBlueprint) {
      uiState.constructorBlueprint = makeDefaultConstructorBlueprint();
    }
  }
  renderConstructorBuilder();
}

async function generateConstructorPreview(message = "Constructor preview generated.") {
  const blueprint = collectConstructorBlueprintFromForm();
  const preview = await apiRequest("/api/control/architectures/constructor/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      blueprint,
      dataset_id: blueprint.dataset_id || null,
    }),
  });
  syncConstructorPreview(preview, message);
}

async function requestConstructorRecommendation() {
  const blueprint = collectConstructorBlueprintFromForm();
  const preview = await apiRequest("/api/control/architectures/constructor/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      blueprint,
      dataset_id: blueprint.dataset_id || null,
    }),
  });
  syncConstructorPreview(preview, "Recommendation applied to the constructor.");
}

function addConstructorLayer(headName, layerType) {
  const layer = createConstructorLayer(layerType);
  if (!layer) return;
  const blueprint = collectConstructorBlueprintFromForm();
  blueprint.head_specs[headName].push(layer);
  uiState.constructorBlueprint = blueprint;
  renderConstructorBuilder();
}

function insertConstructorLayer(headName, index, layerType) {
  const layer = createConstructorLayer(layerType);
  if (!layer) return;
  const blueprint = collectConstructorBlueprintFromForm();
  const layers = blueprint.head_specs?.[headName];
  if (!Array.isArray(layers)) return;
  const safeIndex = Math.max(0, Math.min(Number(index) || 0, layers.length));
  layers.splice(safeIndex, 0, layer);
  uiState.constructorBlueprint = blueprint;
  renderConstructorBuilder();
}

function moveConstructorLayer(headName, index, direction) {
  const blueprint = collectConstructorBlueprintFromForm();
  const layers = blueprint.head_specs?.[headName];
  if (!Array.isArray(layers) || !layers[index]) return;
  const nextIndex = direction === "up" ? index - 1 : index + 1;
  if (nextIndex < 0 || nextIndex >= layers.length) return;
  [layers[index], layers[nextIndex]] = [layers[nextIndex], layers[index]];
  uiState.constructorBlueprint = blueprint;
  renderConstructorBuilder();
}

function moveConstructorLayerTo(headName, index, targetHeadName, targetIndex) {
  const blueprint = collectConstructorBlueprintFromForm();
  const sourceLayers = blueprint.head_specs?.[headName];
  const targetLayers = blueprint.head_specs?.[targetHeadName];
  if (!Array.isArray(sourceLayers) || !Array.isArray(targetLayers) || !sourceLayers[index]) return;
  const [layer] = sourceLayers.splice(index, 1);
  let safeIndex = Math.max(0, Math.min(Number(targetIndex) || 0, targetLayers.length));
  if (headName === targetHeadName && safeIndex > index) {
    safeIndex -= 1;
  }
  targetLayers.splice(safeIndex, 0, layer);
  uiState.constructorBlueprint = blueprint;
  renderConstructorBuilder();
}

function removeConstructorLayer(headName, index) {
  const blueprint = collectConstructorBlueprintFromForm();
  const layers = blueprint.head_specs?.[headName];
  if (!Array.isArray(layers)) return;
  layers.splice(index, 1);
  uiState.constructorBlueprint = blueprint;
  renderConstructorBuilder();
}

function updateConstructorLayerParam(headName, index, paramName, rawValue, inputType) {
  const blueprint = collectConstructorBlueprintFromForm();
  const layer = blueprint.head_specs?.[headName]?.[index];
  if (!layer) return;
  if (inputType === "checkbox") {
    layer.params[paramName] = Boolean(rawValue);
  } else if (inputType === "number") {
    layer.params[paramName] = rawValue === "" ? "" : Number(rawValue);
  } else {
    layer.params[paramName] = rawValue;
  }
  uiState.constructorBlueprint = blueprint;
}

function clearConstructorDropHints() {
  document.querySelectorAll(".stack-drop-slot.is-drop-target, .builder-stack.is-drop-target").forEach((node) => {
    node.classList.remove("is-drop-target");
  });
}

function handleConstructorDrop(headName, index) {
  if (constructorDragState.mode === "new" && constructorDragState.layerType) {
    insertConstructorLayer(headName, index, constructorDragState.layerType);
    setHint("constructor-status", `${getConstructorHeadLabel(headName)} updated with ${constructorDragState.layerType}.`, "ok");
    clearConstructorDragState();
    scheduleConstructorPreview();
    return;
  }
  if (constructorDragState.mode === "move" && constructorDragState.sourceHead) {
    moveConstructorLayerTo(constructorDragState.sourceHead, constructorDragState.sourceIndex, headName, index);
    setHint("constructor-status", `Moved block into ${getConstructorHeadLabel(headName)}.`, "ok");
    clearConstructorDragState();
    scheduleConstructorPreview();
  }
}

function clearConstructorDragState() {
  constructorDragState.mode = null;
  constructorDragState.layerType = "";
  constructorDragState.sourceHead = "";
  constructorDragState.sourceIndex = -1;
  document.querySelectorAll(".layer-palette-card.is-dragging, .layer-card.is-dragging").forEach((node) => node.classList.remove("is-dragging"));
  clearConstructorDropHints();
}

const scheduleConstructorPreview = debounce(() => generateConstructorPreview("Constructor preview refreshed.").catch((error) => setHint("constructor-status", String(error.message || error), "error")), 500);

function resetConstructorBuilder() {
  uiState.constructorBlueprint = makeDefaultConstructorBlueprint();
  uiState.constructorPreview = null;
  renderConstructorBuilder();
  setHint("constructor-status", "Constructor reset to the default blueprint.", "warn");
}

function buildExperimentQuery() {
  const params = new URLSearchParams();
  const search = document.getElementById("experiment-search").value.trim();
  const model = document.getElementById("experiment-model-filter").value;
  const status = document.getElementById("experiment-status-filter").value;
  const minMap50 = document.getElementById("experiment-min-map50").value;
  if (search) params.set("search", search);
  if (model) params.set("model_name", model);
  if (status) params.set("status", status);
  if (minMap50) params.set("min_map50", minMap50);
  return params.toString();
}

async function loadExperiments() {
  const query = buildExperimentQuery();
  const payload = await apiRequest(`/api/control/experiments${query ? `?${query}` : ""}`);
  uiState.experiments = payload.items || [];
  uiState.recommendations = payload.recommendations || [];
  uiState.experimentUniverse = mergeUniverse(uiState.experimentUniverse, uiState.experiments);
  renderSummary();
  renderRecommendations();
  renderExperiments();
  renderCompareTable();
  populateExperimentMetadata();
  updateModelFilterOptions();
}

async function loadJobs() {
  const payload = await apiRequest("/api/control/jobs");
  uiState.jobs = payload.items || [];
  uiState.tensorboard = payload.tensorboard || uiState.tensorboard;
  renderJobs();
  renderSummary();
}

async function loadJobLogs(jobId) {
  uiState.selectedJobId = jobId;
  document.getElementById("job-log-label").textContent = jobId;
  document.getElementById("job-log-output").textContent = await apiRequest(`/api/control/jobs/${encodeURIComponent(jobId)}/logs`, {}, "text");
  renderJobs();
}

async function loadTorchServe() {
  const payload = await apiRequest("/api/control/torchserve/models");
  uiState.torchserve = { available: Boolean(payload.available), models: payload.models || [] };
  renderTorchServeModels();
  setHint("torchserve-status", payload.available ? `Loaded ${uiState.torchserve.models.length} registered model(s).` : payload.error || "TorchServe unavailable.", payload.available ? "ok" : "warn");
}

async function saveCurrentConfig() {
  const name = slugify(document.getElementById("config-name").value || uiState.selectedConfigName || "new_experiment");
  const payload = await apiRequest(`/api/control/configs/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_yaml: document.getElementById("config-editor").value }),
  });
  uiState.selectedConfigName = payload.name;
  document.getElementById("config-active-badge").textContent = payload.name;
  setHint("config-status", `Config ${payload.name} saved.`, "ok");
  await loadConfigs();
}

async function saveArchitecture() {
  const constructorBlueprint = collectConstructorBlueprintFromForm();
  const payload = await apiRequest("/api/control/architectures", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: document.getElementById("architecture-name").value.trim(),
      description: document.getElementById("architecture-description").value.trim(),
      tags: document.getElementById("architecture-tags").value.split(",").map((item) => item.trim()).filter(Boolean),
      config_yaml: document.getElementById("architecture-config").value,
      source_code: document.getElementById("architecture-source").value,
      blueprint: constructorBlueprint,
    }),
  });
  uiState.selectedArchitectureId = payload.id;
  if (payload.blueprint) {
    uiState.constructorBlueprint = deepClone(payload.blueprint);
  }
  document.getElementById("architecture-active-badge").textContent = payload.name || payload.id;
  setHint("architecture-status", `Architecture ${payload.name || payload.id} saved.`, "ok");
  await loadArchitectures();
}

async function uploadDataset() {
  const form = document.getElementById("dataset-upload-form");
  const formData = new FormData(form);
  const file = formData.get("file");
  if (!(file instanceof File) || !file.name) {
    setHint("dataset-status", "Select an archive to upload.", "error");
    return;
  }
  const response = await withTimeout("/api/control/datasets/upload", { method: "POST", body: formData }, 45000);
  if (!response.ok) {
    setHint("dataset-status", await parseResponseError(response), "error");
    return;
  }
  const payload = await response.json();
  uiState.datasets = payload.items || [];
  renderDatasets();
  populateConstructorDatasetOptions();
  renderSummary();
  setHint("dataset-status", payload.archive_extracted ? "Dataset archive uploaded and extracted." : "Dataset uploaded.", "ok");
  form.reset();
}

async function registerDataset() {
  const formData = new FormData(document.getElementById("dataset-register-form"));
  const payload = {
    name: String(formData.get("name") || "").trim(),
    path: String(formData.get("path") || "").trim(),
    description: String(formData.get("description") || "").trim(),
    tags: String(formData.get("tags") || "").split(",").map((item) => item.trim()).filter(Boolean),
  };
  if (!payload.name || !payload.path) {
    setHint("dataset-status", "Both dataset name and path are required.", "error");
    return;
  }
  const response = await apiRequest("/api/control/datasets/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  uiState.datasets = response.items || [];
  renderDatasets();
  populateConstructorDatasetOptions();
  renderSummary();
  setHint("dataset-status", `Dataset ${payload.name} registered.`, "ok");
  document.getElementById("dataset-register-form").reset();
}

async function launchJob(kind) {
  const configYaml = document.getElementById("config-editor").value.trim();
  if (!configYaml) {
    setHint("studio-status", "Config editor is empty.", "error");
    return;
  }
  const response = await apiRequest(`/api/control/jobs/${kind === "train" ? "train" : "evaluate"}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      config_yaml: configYaml,
      save_as_config_name: document.getElementById("launch-save-as").value.trim() || null,
      split: document.getElementById("launch-split").value,
    }),
  });
  setHint("studio-status", `${kind === "train" ? "Training" : "Evaluation"} job ${response.job.id} started.`, "ok");
  await loadJobs();
  setTimeout(() => loadExperiments().catch(() => {}), 2000);
  activateTab("tab-experiments");
}
async function stopJob(jobId) {
  await apiRequest(`/api/control/jobs/${encodeURIComponent(jobId)}/stop`, { method: "POST" });
  await loadJobs();
}

async function saveExperimentMetadata() {
  if (!uiState.selectedExperimentKey) {
    setHint("experiment-meta-status", "Select an experiment first.", "error");
    return;
  }
  await apiRequest(`/api/control/experiments/${encodeURIComponent(uiState.selectedExperimentKey)}/metadata`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      tags: document.getElementById("experiment-tags").value.split(",").map((item) => item.trim()).filter(Boolean),
      rating: document.getElementById("experiment-rating").value || null,
      note: document.getElementById("experiment-note").value,
    }),
  });
  setHint("experiment-meta-status", `Metadata saved for ${uiState.selectedExperimentKey}.`, "ok");
  await loadExperiments();
}

async function registerTorchServeModel() {
  const modelName = document.getElementById("torchserve-model-name").value.trim();
  const archiveFile = document.getElementById("torchserve-archive-file").value.trim();
  if (!modelName || !archiveFile) {
    setHint("torchserve-status", "Model name and archive file are required.", "error");
    return;
  }
  await apiRequest("/api/control/torchserve/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: modelName,
      archive_file: archiveFile,
      initial_workers: Number(document.getElementById("torchserve-workers").value),
    }),
  });
  setHint("torchserve-status", `Model ${modelName} registered.`, "ok");
  await loadTorchServe();
}

async function unregisterTorchServeModel(modelName) {
  await apiRequest(`/api/control/torchserve/models/${encodeURIComponent(modelName)}`, { method: "DELETE" });
  setHint("torchserve-status", `Model ${modelName} unregistered.`, "warn");
  await loadTorchServe();
}

async function runInference() {
  const file = document.getElementById("inference-file").files?.[0];
  const modelName = document.getElementById("inference-model-name").value.trim();
  if (!modelName || !file) {
    document.getElementById("inference-output").textContent = "Model name and image are required.";
    return;
  }
  const formData = new FormData();
  formData.set("model_name", modelName);
  formData.set("file", file);
  const response = await withTimeout("/api/control/torchserve/predict", { method: "POST", body: formData }, 30000);
  document.getElementById("inference-output").textContent = response.ok ? JSON.stringify(await response.json(), null, 2) : await parseResponseError(response);
}

function setupDatasetsHandlers() {
  document.getElementById("dataset-upload-btn").addEventListener("click", () => uploadDataset().catch((error) => setHint("dataset-status", String(error.message || error), "error")));
  document.getElementById("dataset-register-btn").addEventListener("click", () => registerDataset().catch((error) => setHint("dataset-status", String(error.message || error), "error")));
  document.getElementById("datasets-refresh").addEventListener("click", () => loadDatasets().catch((error) => setHint("dataset-status", String(error.message || error), "error")));
  document.getElementById("dataset-search").addEventListener("input", renderDatasets);
  document.getElementById("dataset-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-dataset-id]");
    if (button) window.open(`/api/control/datasets/${encodeURIComponent(button.dataset.datasetId)}/download`, "_blank", "noopener");
  });
}

function setupStudioHandlers() {
  document.getElementById("config-new").addEventListener("click", resetConfigEditor);
  document.getElementById("configs-refresh").addEventListener("click", () => loadConfigs().catch((error) => setHint("config-status", String(error.message || error), "error")));
  document.getElementById("config-search").addEventListener("input", renderConfigs);
  document.getElementById("config-save").addEventListener("click", () => saveCurrentConfig().catch((error) => setHint("config-status", String(error.message || error), "error")));
  document.getElementById("config-copy").addEventListener("click", async () => {
    try {
      await copyText(document.getElementById("config-editor").value);
      setHint("config-status", "YAML copied to clipboard.", "ok");
    } catch {
      setHint("config-status", "Clipboard access is unavailable.", "error");
    }
  });
  document.getElementById("config-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-config-load]");
    if (button) loadConfigDetail(button.dataset.configLoad).catch((error) => setHint("config-status", String(error.message || error), "error"));
  });
  document.getElementById("architecture-new").addEventListener("click", resetArchitectureEditor);
  document.getElementById("architectures-refresh").addEventListener("click", () => loadArchitectures().catch((error) => setHint("architecture-status", String(error.message || error), "error")));
  document.getElementById("architecture-search").addEventListener("input", renderArchitectures);
  document.getElementById("architecture-save").addEventListener("click", () => saveArchitecture().catch((error) => setHint("architecture-status", String(error.message || error), "error")));
  document.getElementById("architecture-use-config").addEventListener("click", () => {
    document.getElementById("config-editor").value = document.getElementById("architecture-config").value || defaultConfigText;
    document.getElementById("config-name").value = slugify(document.getElementById("architecture-name").value || "new_experiment");
    document.getElementById("config-active-badge").textContent = "Template loaded";
    setHint("config-status", "Architecture template loaded into config editor.", "ok");
  });
  document.getElementById("architecture-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-architecture-load]");
    if (button) loadArchitectureDetail(button.dataset.architectureLoad).catch((error) => setHint("architecture-status", String(error.message || error), "error"));
  });
  ["constructor-name", "constructor-dataset", "constructor-base-model", "constructor-goal", "constructor-checkpoint", "constructor-labels"].forEach((id) => {
    document.getElementById(id).addEventListener(id.includes("name") || id.includes("checkpoint") || id.includes("labels") ? "input" : "change", (event) => {
      if (id === "constructor-base-model") {
        const selected = (getConstructorCatalog().base_models || []).find((item) => item.id === event.target.value);
        if (selected) {
          document.getElementById("constructor-checkpoint").value = selected.checkpoint || "";
        }
      }
      uiState.constructorBlueprint = collectConstructorBlueprintFromForm();
      if (["constructor-dataset", "constructor-base-model", "constructor-goal"].includes(id)) {
        renderConstructorBuilder();
      }
      scheduleConstructorPreview();
    });
  });
  document.getElementById("constructor-recommend").addEventListener("click", () => requestConstructorRecommendation().catch((error) => setHint("constructor-status", String(error.message || error), "error")));
  document.getElementById("constructor-generate").addEventListener("click", () => generateConstructorPreview().catch((error) => setHint("constructor-status", String(error.message || error), "error")));
  document.getElementById("constructor-reset").addEventListener("click", resetConstructorBuilder);
  document.getElementById("constructor-layer-catalog").addEventListener("click", (event) => {
    const button = event.target.closest("[data-layer-add]");
    if (!button) return;
    addConstructorLayer(button.dataset.layerTarget, button.dataset.layerAdd);
    scheduleConstructorPreview();
  });
  document.getElementById("constructor-layer-catalog").addEventListener("dragstart", (event) => {
    const card = event.target.closest("[data-layer-drag-type]");
    if (!card) return;
    constructorDragState.mode = "new";
    constructorDragState.layerType = card.dataset.layerDragType || "";
    constructorDragState.sourceHead = "";
    constructorDragState.sourceIndex = -1;
    card.classList.add("is-dragging");
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = "copy";
      event.dataTransfer.setData("text/plain", constructorDragState.layerType);
    }
    setHint("constructor-status", `Dragging ${constructorDragState.layerType}. Drop it into a lane.`, "ok");
  });
  document.getElementById("constructor-layer-catalog").addEventListener("dragend", clearConstructorDragState);
  ["constructor-classifier-stack", "constructor-bbox-stack"].forEach((containerId) => {
    const container = document.getElementById(containerId);
    container.addEventListener("click", (event) => {
      const removeButton = event.target.closest("[data-layer-remove]");
      if (removeButton) {
        removeConstructorLayer(removeButton.dataset.headName, Number(removeButton.dataset.layerIndex));
        scheduleConstructorPreview();
        return;
      }
      const moveButton = event.target.closest("[data-layer-move]");
      if (moveButton) {
        moveConstructorLayer(moveButton.dataset.headName, Number(moveButton.dataset.layerIndex), moveButton.dataset.layerMove);
        scheduleConstructorPreview();
        return;
      }
      const dropSlot = event.target.closest("[data-drop-head][data-drop-index]");
      if (dropSlot && constructorDragState.mode) {
        handleConstructorDrop(dropSlot.dataset.dropHead, Number(dropSlot.dataset.dropIndex));
      }
    });
    container.addEventListener("input", (event) => {
      const input = event.target.closest("[data-layer-param]");
      if (!input) return;
      updateConstructorLayerParam(input.dataset.headName, Number(input.dataset.layerIndex), input.dataset.layerParam, input.type === "checkbox" ? input.checked : input.value, input.type);
      scheduleConstructorPreview();
    });
    container.addEventListener("change", (event) => {
      const input = event.target.closest("[data-layer-param]");
      if (!input) return;
      updateConstructorLayerParam(input.dataset.headName, Number(input.dataset.layerIndex), input.dataset.layerParam, input.type === "checkbox" ? input.checked : input.value, input.type);
      scheduleConstructorPreview();
    });
    container.addEventListener("dragstart", (event) => {
      const card = event.target.closest("[data-stack-layer]");
      if (!card) return;
      constructorDragState.mode = "move";
      constructorDragState.layerType = "";
      constructorDragState.sourceHead = card.dataset.headName || "";
      constructorDragState.sourceIndex = Number(card.dataset.layerIndex);
      card.classList.add("is-dragging");
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", `${constructorDragState.sourceHead}:${constructorDragState.sourceIndex}`);
      }
      setHint("constructor-status", `Moving a block inside the builder. Drop it where it should live.`, "ok");
    });
    container.addEventListener("dragend", clearConstructorDragState);
    container.addEventListener("dragover", (event) => {
      if (!constructorDragState.mode) return;
      event.preventDefault();
      const target = event.target.closest("[data-drop-head][data-drop-index]") || event.currentTarget;
      clearConstructorDropHints();
      target.classList.add("is-drop-target");
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = constructorDragState.mode === "new" ? "copy" : "move";
      }
    });
    container.addEventListener("dragleave", (event) => {
      if (!event.currentTarget.contains(event.relatedTarget)) {
        clearConstructorDropHints();
      }
    });
    container.addEventListener("drop", (event) => {
      if (!constructorDragState.mode) return;
      event.preventDefault();
      const dropTarget = event.target.closest("[data-drop-head][data-drop-index]");
      if (dropTarget) {
        handleConstructorDrop(dropTarget.dataset.dropHead, Number(dropTarget.dataset.dropIndex));
        return;
      }
      const headName = event.currentTarget.dataset.headName;
      if (headName) {
        const layerCount = (uiState.constructorBlueprint?.head_specs?.[headName] || []).length;
        handleConstructorDrop(headName, layerCount);
      }
    });
  });
  document.getElementById("launch-train").addEventListener("click", () => launchJob("train").catch((error) => setHint("studio-status", String(error.message || error), "error")));
  document.getElementById("launch-evaluate").addEventListener("click", () => launchJob("evaluate").catch((error) => setHint("studio-status", String(error.message || error), "error")));
}

function setupExperimentsHandlers() {
  const debouncedRefresh = debounce(() => loadExperiments().catch((error) => setHint("experiment-meta-status", String(error.message || error), "error")), 320);
  ["experiment-search", "experiment-model-filter", "experiment-status-filter", "experiment-min-map50"].forEach((id) => document.getElementById(id).addEventListener(id === "experiment-search" ? "input" : "change", debouncedRefresh));
  document.getElementById("experiments-refresh").addEventListener("click", () => loadExperiments().catch((error) => setHint("experiment-meta-status", String(error.message || error), "error")));
  document.getElementById("experiment-save-meta").addEventListener("click", () => saveExperimentMetadata().catch((error) => setHint("experiment-meta-status", String(error.message || error), "error")));
  document.getElementById("experiments-body").addEventListener("click", (event) => {
    const checkbox = event.target.closest("[data-compare-key]");
    if (checkbox) {
      checkbox.checked ? uiState.compareKeys.add(checkbox.dataset.compareKey) : uiState.compareKeys.delete(checkbox.dataset.compareKey);
      renderCompareTable();
      return;
    }
    const row = event.target.closest("[data-experiment-key]");
    if (!row) return;
    uiState.selectedExperimentKey = row.dataset.experimentKey;
    renderExperiments();
    populateExperimentMetadata();
  });
  document.getElementById("job-list").addEventListener("click", (event) => {
    const logsButton = event.target.closest("[data-job-logs]");
    if (logsButton) {
      loadJobLogs(logsButton.dataset.jobLogs).catch((error) => document.getElementById("job-log-output").textContent = String(error.message || error));
      return;
    }
    const stopButton = event.target.closest("[data-job-stop]");
    if (stopButton) stopJob(stopButton.dataset.jobStop).catch((error) => document.getElementById("job-log-output").textContent = String(error.message || error));
  });
  document.getElementById("job-log-refresh").addEventListener("click", () => uiState.selectedJobId && loadJobLogs(uiState.selectedJobId).catch((error) => document.getElementById("job-log-output").textContent = String(error.message || error)));
}

function setupServingHandlers() {
  document.getElementById("torchserve-register").addEventListener("click", () => registerTorchServeModel().catch((error) => setHint("torchserve-status", String(error.message || error), "error")));
  document.getElementById("torchserve-refresh").addEventListener("click", () => loadTorchServe().catch((error) => setHint("torchserve-status", String(error.message || error), "error")));
  document.getElementById("torchserve-models").addEventListener("click", (event) => {
    const button = event.target.closest("[data-torchserve-delete]");
    if (button) unregisterTorchServeModel(button.dataset.torchserveDelete).catch((error) => setHint("torchserve-status", String(error.message || error), "error"));
  });
  document.getElementById("run-inference").addEventListener("click", () => runInference().catch((error) => document.getElementById("inference-output").textContent = String(error.message || error)));
}

function setupResourceHandlers() {
  document.getElementById("generate-resource-plan").addEventListener("click", renderResourcePlan);
  document.getElementById("copy-resource-plan").addEventListener("click", async () => {
    try {
      await copyText(document.getElementById("resource-override").textContent);
      setHint("usage-updated-at", "Resource override copied to clipboard.", "ok");
    } catch {
      setHint("usage-updated-at", "Clipboard access is unavailable.", "error");
    }
  });
  ["resource-profile", "resource-cpu-multiplier", "resource-mem-multiplier", "resource-reservation"].forEach((id) => {
    document.getElementById(id).addEventListener("change", renderResourcePlan);
    document.getElementById(id).addEventListener("input", renderResourcePlan);
  });
}

async function bootstrapData() {
  await loadConstructorCatalog();
  try {
    await loadCatalog();
  } catch (error) {
    setHint("config-status", `Catalog load failed: ${String(error.message || error)}`, "error");
  }
  try {
    await loadTorchServe();
  } catch (error) {
    setHint("torchserve-status", String(error.message || error), "warn");
  }
}

function bootstrap() {
  setupTabs();
  setupAppearanceControls();
  setupPanelGridLayout();
  setupQuickActionButtons();
  setupDashboardSwitcher();
  setupTensorBoardFrame();
  setupDatasetsHandlers();
  setupStudioHandlers();
  setupExperimentsHandlers();
  setupServingHandlers();
  setupResourceHandlers();
  resetConfigEditor();
  resetArchitectureEditor();
  renderResourcePlan();
  renderOperationalHeader();
  bootstrapData();
  refreshServiceStatus();
  refreshMetrics();
  refreshUsageBreakdown();
  setInterval(refreshServiceStatus, 15000);
  setInterval(refreshMetrics, 9000);
  setInterval(refreshUsageBreakdown, 12000);
  setInterval(() => loadJobs().catch(() => {}), 10000);
  setInterval(() => loadExperiments().catch(() => {}), 15000);
  setInterval(() => uiState.selectedJobId && loadJobLogs(uiState.selectedJobId).catch(() => {}), 8000);
}

bootstrap();
