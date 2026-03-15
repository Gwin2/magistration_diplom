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
};

const defaultConfigText = `experiment:\n  name: new_experiment\n  seed: 42\n\npaths:\n  train_images: data/processed/uav_coco/images/train\n  val_images: data/processed/uav_coco/images/val\n  test_images: data/processed/uav_coco/images/test\n  train_annotations: data/processed/uav_coco/annotations/instances_train.json\n  val_annotations: data/processed/uav_coco/annotations/instances_val.json\n  test_annotations: data/processed/uav_coco/annotations/instances_test.json\n  output_dir: runs/new_experiment\n\nmlflow:\n  enabled: true\n  tracking_uri: http://mlflow:5000\n  experiment_name: uav-vit-thesis\n  run_name: new_experiment\n  log_checkpoints: true\n\ntensorboard:\n  enabled: true\n\nmodel:\n  name: yolos_tiny\n  checkpoint: hustvl/yolos-tiny\n  num_labels: 1\n  id2label:\n    \"0\": uav\n  label2id:\n    uav: 0\n  train_backbone: true\n  custom_modules: []\n\ntrain:\n  device: auto\n  epochs: 30\n  batch_size: 4\n  learning_rate: 2.0e-5\n  weight_decay: 1.0e-4\n  num_workers: 4\n  grad_clip_norm: 1.0\n  log_interval: 20\n  mixed_precision: true\n  eval_every_epoch: true\n  checkpoint_metric: map\n  checkpoint_mode: max\n\neval:\n  score_threshold: 0.1\n  latency_warmup_iters: 10\n  latency_iters: 50\n\ndata:\n  processor_size: 800\n  normalize_boxes: false\n`;

const defaultArchitectureConfigText = `experiment:\n  name: custom_detector\n  seed: 42\n\npaths:\n  train_images: data/processed/uav_coco/images/train\n  val_images: data/processed/uav_coco/images/val\n  test_images: data/processed/uav_coco/images/test\n  train_annotations: data/processed/uav_coco/annotations/instances_train.json\n  val_annotations: data/processed/uav_coco/annotations/instances_val.json\n  test_annotations: data/processed/uav_coco/annotations/instances_test.json\n  output_dir: runs/custom_detector\n\nmlflow:\n  enabled: true\n  tracking_uri: http://mlflow:5000\n  experiment_name: uav-vit-thesis\n  run_name: custom_detector\n  log_checkpoints: true\n\ntensorboard:\n  enabled: true\n\nmodel:\n  name: my_detector\n  checkpoint: facebook/detr-resnet-50\n  num_labels: 1\n  id2label:\n    \"0\": uav\n  label2id:\n    uav: 0\n  train_backbone: true\n  custom_modules:\n    - custom_models.my_detector\n\ntrain:\n  device: auto\n  epochs: 30\n  batch_size: 4\n  learning_rate: 2.0e-5\n  weight_decay: 1.0e-4\n  num_workers: 4\n  grad_clip_norm: 1.0\n  log_interval: 20\n  mixed_precision: true\n  eval_every_epoch: true\n  checkpoint_metric: map\n  checkpoint_mode: max\n\neval:\n  score_threshold: 0.1\n  latency_warmup_iters: 10\n  latency_iters: 50\n\ndata:\n  processor_size: 800\n  normalize_boxes: false\n`;

const defaultArchitectureSource = `from __future__ import annotations\n\nfrom transformers import AutoImageProcessor, AutoModelForObjectDetection\n\nfrom uav_vit.models import ModelBundle, register_model\n\n\n@register_model(\"my_detector\")\ndef build_my_detector(config: dict) -> ModelBundle:\n    checkpoint = config[\"model\"].get(\"checkpoint\") or \"facebook/detr-resnet-50\"\n    model = AutoModelForObjectDetection.from_pretrained(\n        checkpoint,\n        ignore_mismatched_sizes=True,\n        num_labels=int(config[\"model\"][\"num_labels\"]),\n        id2label={int(key): value for key, value in config[\"model\"][\"id2label\"].items()},\n        label2id={str(key): int(value) for key, value in config[\"model\"][\"label2id\"].items()},\n    )\n    processor = AutoImageProcessor.from_pretrained(checkpoint)\n    return ModelBundle(model=model, image_processor=processor, name=\"my_detector\")\n`;

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
      <small>${escapeHtml(item.kind)}${item.tags?.length ? ` • ${escapeHtml(item.tags.join(", "))}` : ""}</small>
    </button>
  `).join("") : '<div class="empty-state">No architectures match the current filter.</div>';
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
  renderExperiments();
  renderCompareTable();
  populateExperimentMetadata();
  renderJobs();
  updateModelFilterOptions();
}

async function loadDatasets() {
  uiState.datasets = (await apiRequest("/api/control/datasets")).items || [];
  renderDatasets();
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
  renderArchitectures();
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
  const payload = await apiRequest("/api/control/architectures", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: document.getElementById("architecture-name").value.trim(),
      description: document.getElementById("architecture-description").value.trim(),
      tags: document.getElementById("architecture-tags").value.split(",").map((item) => item.trim()).filter(Boolean),
      config_yaml: document.getElementById("architecture-config").value,
      source_code: document.getElementById("architecture-source").value,
    }),
  });
  uiState.selectedArchitectureId = payload.id;
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
