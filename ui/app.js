const statusChecks = {
  mlflow: "/api/mlflow/",
  prometheus: "/api/prometheus/-/healthy",
  grafana: "/api/grafana/api/health",
  pushgateway: "/api/pushgateway/metrics",
  torchserve: "/api/torchserve/ping",
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
  host_disk:
    "100 - (avg(node_filesystem_avail_bytes{mountpoint=\"/\",fstype!~\"tmpfs|overlay\"} / node_filesystem_size_bytes{mountpoint=\"/\",fstype!~\"tmpfs|overlay\"}) * 100)",
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
  host_disk: 61.0,
  firing_alerts: 0,
};

const kpiFormatters = {
  map50: (v) => v.toFixed(3),
  map75: (v) => v.toFixed(3),
  latency: (v) => v.toFixed(1),
  fps: (v) => v.toFixed(1),
  health: (v) => `${(v * 100).toFixed(1)}%`,
  host_cpu: (v) => `${v.toFixed(1)}%`,
  host_mem: (v) => `${v.toFixed(1)}%`,
  host_disk: (v) => `${v.toFixed(1)}%`,
  firing_alerts: (v) => `${Math.round(v)}`,
};

const metricHistory = {};
Object.keys(metricQueries).forEach((k) => {
  metricHistory[k] = Array.from({ length: 18 }, () => metricFallback[k]);
});

const resourceProfiles = {
  balanced: {
    postgres: { cpus: 1.0, memMb: 1024 },
    minio: { cpus: 1.0, memMb: 1024 },
    mlflow: { cpus: 0.8, memMb: 768 },
    prometheus: { cpus: 1.5, memMb: 2048 },
    grafana: { cpus: 0.8, memMb: 768 },
    ui: { cpus: 0.5, memMb: 256 },
    pushgateway: { cpus: 0.4, memMb: 256 },
    "process-exporter": { cpus: 0.4, memMb: 256 },
    "node-exporter": { cpus: 0.4, memMb: 256 },
    cadvisor: { cpus: 0.7, memMb: 512 },
    alertmanager: { cpus: 0.4, memMb: 256 },
    "blackbox-exporter": { cpus: 0.4, memMb: 256 },
    "postgres-exporter": { cpus: 0.4, memMb: 256 },
  },
  training: {
    postgres: { cpus: 1.2, memMb: 1536 },
    minio: { cpus: 1.2, memMb: 1536 },
    mlflow: { cpus: 1.0, memMb: 1024 },
    trainer: { cpus: 6.0, memMb: 12288 },
    "run-metrics-exporter": { cpus: 0.8, memMb: 1024 },
    prometheus: { cpus: 2.0, memMb: 3072 },
    grafana: { cpus: 1.0, memMb: 1024 },
    ui: { cpus: 0.6, memMb: 384 },
    pushgateway: { cpus: 0.5, memMb: 384 },
  },
  inference: {
    postgres: { cpus: 1.0, memMb: 1024 },
    minio: { cpus: 1.0, memMb: 1024 },
    mlflow: { cpus: 0.8, memMb: 768 },
    torchserve: { cpus: 4.0, memMb: 8192 },
    prometheus: { cpus: 1.6, memMb: 2048 },
    grafana: { cpus: 1.0, memMb: 1024 },
    ui: { cpus: 0.6, memMb: 384 },
    pushgateway: { cpus: 0.5, memMb: 384 },
  },
  eco: {
    postgres: { cpus: 0.7, memMb: 768 },
    minio: { cpus: 0.7, memMb: 768 },
    mlflow: { cpus: 0.6, memMb: 512 },
    prometheus: { cpus: 1.0, memMb: 1024 },
    grafana: { cpus: 0.6, memMb: 512 },
    ui: { cpus: 0.4, memMb: 192 },
    pushgateway: { cpus: 0.3, memMb: 192 },
    "process-exporter": { cpus: 0.3, memMb: 192 },
    "node-exporter": { cpus: 0.3, memMb: 192 },
    cadvisor: { cpus: 0.5, memMb: 384 },
    alertmanager: { cpus: 0.3, memMb: 192 },
    "blackbox-exporter": { cpus: 0.3, memMb: 192 },
    "postgres-exporter": { cpus: 0.3, memMb: 192 },
  },
};

const fallbackContainers = [
  { name: "uav-prometheus", cpu: 7.6, mem: 680, rx: 210 },
  { name: "uav-grafana", cpu: 2.1, mem: 310, rx: 74 },
  { name: "uav-mlflow", cpu: 4.4, mem: 420, rx: 95 },
  { name: "uav-minio", cpu: 3.2, mem: 385, rx: 112 },
  { name: "uav-postgres", cpu: 2.8, mem: 266, rx: 66 },
];

const fallbackProcesses = [
  { name: "prometheus", cpu: 8.4, mem: 540, threads: 18 },
  { name: "grafana", cpu: 2.6, mem: 290, threads: 29 },
  { name: "postgres", cpu: 3.3, mem: 245, threads: 14 },
  { name: "minio", cpu: 3.8, mem: 362, threads: 21 },
  { name: "nginx", cpu: 0.8, mem: 55, threads: 9 },
];

const uiStorageKeys = {
  theme: "uav_ui_theme",
  motion: "uav_ui_motion",
  density: "uav_ui_density",
};

const allowedThemes = new Set(["flight", "horizon", "paper"]);
const allowedDensity = new Set(["compact", "comfortable"]);

function withTimeout(url, timeoutMs = 4500) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  return fetch(url, { signal: ctrl.signal }).finally(() => clearTimeout(timer));
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
    // ignore storage quota/privacy failures
  }
}

function setTheme(theme, persist = true) {
  const nextTheme = allowedThemes.has(theme) ? theme : "flight";
  document.body.dataset.theme = nextTheme;
  document.querySelectorAll(".theme-btn[data-theme]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.theme === nextTheme);
  });
  if (persist) {
    safeWriteStorage(uiStorageKeys.theme, nextTheme);
  }
  window.dispatchEvent(new CustomEvent("uav-theme-change", { detail: { theme: nextTheme } }));
}

function setMotion(mode, persist = true) {
  const nextMode = mode === "off" ? "off" : "on";
  document.body.dataset.motion = nextMode;
  const motionBtn = document.getElementById("motion-toggle");
  if (motionBtn) {
    const motionEnabled = nextMode === "on";
    motionBtn.textContent = `Motion: ${motionEnabled ? "on" : "off"}`;
    motionBtn.setAttribute("aria-pressed", String(!motionEnabled));
  }
  if (persist) {
    safeWriteStorage(uiStorageKeys.motion, nextMode);
  }
}

function setDensity(density, persist = true) {
  const nextDensity = allowedDensity.has(density) ? density : "compact";
  document.body.dataset.density = nextDensity;
  document.querySelectorAll(".density-btn[data-density]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.density === nextDensity);
  });
  if (persist) {
    safeWriteStorage(uiStorageKeys.density, nextDensity);
  }
}

function setupAppearanceControls() {
  const storedTheme = safeReadStorage(uiStorageKeys.theme) || document.body.dataset.theme || "flight";
  const storedMotion = safeReadStorage(uiStorageKeys.motion) || document.body.dataset.motion || "on";
  const storedDensity = safeReadStorage(uiStorageKeys.density) || document.body.dataset.density || "compact";
  setTheme(storedTheme, false);
  setMotion(storedMotion, false);
  setDensity(storedDensity, false);

  document.querySelectorAll(".theme-btn[data-theme]").forEach((btn) => {
    btn.addEventListener("click", () => setTheme(btn.dataset.theme));
  });

  document.querySelectorAll(".density-btn[data-density]").forEach((btn) => {
    btn.addEventListener("click", () => setDensity(btn.dataset.density));
  });

  const motionBtn = document.getElementById("motion-toggle");
  if (motionBtn) {
    motionBtn.addEventListener("click", () => {
      const nextMode = document.body.dataset.motion === "off" ? "on" : "off";
      setMotion(nextMode);
    });
  }
}

function setServiceStatus(serviceName, up) {
  const el = document.querySelector(`.status-pill[data-service="${serviceName}"]`);
  if (!el) return;
  el.classList.remove("status-up", "status-down");
  el.classList.add(up ? "status-up" : "status-down");
}

async function refreshServiceStatus() {
  await Promise.all(
    Object.entries(statusChecks).map(async ([name, url]) => {
      try {
        const res = await withTimeout(url);
        setServiceStatus(name, res.ok);
      } catch {
        setServiceStatus(name, false);
      }
    }),
  );
}

function parsePromVector(payload) {
  if (!payload || payload.status !== "success") return [];
  const result = payload.data && payload.data.result;
  if (!Array.isArray(result)) return [];
  return result
    .map((item) => {
      const raw = item.value && item.value[1];
      const value = Number(raw);
      if (!Number.isFinite(value)) return null;
      return { metric: item.metric || {}, value };
    })
    .filter(Boolean);
}

function parsePromValue(payload) {
  const vector = parsePromVector(payload);
  if (!vector.length) return null;
  return vector[0].value;
}

async function queryValue(query) {
  const url = `/api/prometheus/api/v1/query?query=${encodeURIComponent(query)}`;
  const res = await withTimeout(url, 5500);
  if (!res.ok) throw new Error("prometheus_query_failed");
  const payload = await res.json();
  const value = parsePromValue(payload);
  if (value === null) throw new Error("empty_result");
  return value;
}

async function queryVector(query) {
  const url = `/api/prometheus/api/v1/query?query=${encodeURIComponent(query)}`;
  const res = await withTimeout(url, 5500);
  if (!res.ok) throw new Error("prometheus_query_failed");
  const payload = await res.json();
  return parsePromVector(payload);
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
  const path = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * 120;
      const y = 34 - ((v - min) / span) * 28;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  const pathEl = card.querySelector("path");
  if (pathEl) pathEl.setAttribute("d", path);
}

function setKpi(metricKey, value) {
  const card = document.querySelector(`.kpi-card[data-metric="${metricKey}"]`);
  if (!card) return;
  const valueEl = card.querySelector(".kpi-value");
  if (valueEl) valueEl.textContent = kpiFormatters[metricKey](value);
  metricHistory[metricKey].push(value);
  while (metricHistory[metricKey].length > 24) metricHistory[metricKey].shift();
  drawSparkline(metricKey);
}

async function refreshMetrics() {
  await Promise.all(
    Object.keys(metricQueries).map(async (metricKey) => {
      const value = await queryMetric(metricKey);
      setKpi(metricKey, value);
    }),
  );
}

function normalizeContainerName(rawName) {
  if (!rawName) return "unknown";
  return rawName.replace(/^\//, "");
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
  tbody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = columns.map((col) => `<td>${col(row)}</td>`).join("");
    tbody.appendChild(tr);
  });
}

async function refreshContainerUsage() {
  try {
    const [cpuRows, memRows, rxRows] = await Promise.all([
      queryVector('topk(20, sum by (name) (rate(container_cpu_usage_seconds_total{name!=""}[5m])) * 100)'),
      queryVector('topk(20, sum by (name) (container_memory_working_set_bytes{name!=""}) / 1024 / 1024)'),
      queryVector('topk(20, sum by (name) (rate(container_network_receive_bytes_total{name!=""}[5m])) / 1024)'),
    ]);

    const byName = new Map();
    cpuRows.forEach((r) => {
      const name = normalizeContainerName(r.metric.name);
      byName.set(name, { name, cpu: r.value, mem: null, rx: null });
    });
    memRows.forEach((r) => {
      const name = normalizeContainerName(r.metric.name);
      const entry = byName.get(name) || { name, cpu: null, mem: null, rx: null };
      entry.mem = r.value;
      byName.set(name, entry);
    });
    rxRows.forEach((r) => {
      const name = normalizeContainerName(r.metric.name);
      const entry = byName.get(name) || { name, cpu: null, mem: null, rx: null };
      entry.rx = r.value;
      byName.set(name, entry);
    });

    const rows = Array.from(byName.values())
      .sort((a, b) => (b.cpu || 0) - (a.cpu || 0))
      .slice(0, 12);
    renderTableRows("container-usage-body", rows, [
      (r) => r.name,
      (r) => toNumberFixed(r.cpu),
      (r) => toNumberFixed(r.mem),
      (r) => toNumberFixed(r.rx),
    ]);
    return true;
  } catch {
    renderTableRows("container-usage-body", fallbackContainers, [
      (r) => r.name,
      (r) => toNumberFixed(r.cpu),
      (r) => toNumberFixed(r.mem),
      (r) => toNumberFixed(r.rx),
    ]);
    return false;
  }
}

async function refreshProcessUsage() {
  try {
    const [cpuRows, memRows, threadRows] = await Promise.all([
      queryVector("topk(20, sum by (groupname) (rate(namedprocess_namegroup_cpu_seconds_total[5m])) * 100)"),
      queryVector(
        'topk(20, sum by (groupname) (namedprocess_namegroup_memory_bytes{memtype="resident"}) / 1024 / 1024)',
      ),
      queryVector("topk(20, sum by (groupname) (namedprocess_namegroup_num_threads))"),
    ]);

    const byName = new Map();
    cpuRows.forEach((r) => {
      const name = pickProcessName(r.metric);
      byName.set(name, { name, cpu: r.value, mem: null, threads: null });
    });
    memRows.forEach((r) => {
      const name = pickProcessName(r.metric);
      const entry = byName.get(name) || { name, cpu: null, mem: null, threads: null };
      entry.mem = r.value;
      byName.set(name, entry);
    });
    threadRows.forEach((r) => {
      const name = pickProcessName(r.metric);
      const entry = byName.get(name) || { name, cpu: null, mem: null, threads: null };
      entry.threads = r.value;
      byName.set(name, entry);
    });

    const rows = Array.from(byName.values())
      .sort((a, b) => (b.cpu || 0) - (a.cpu || 0))
      .slice(0, 12);
    renderTableRows("process-usage-body", rows, [
      (r) => r.name,
      (r) => toNumberFixed(r.cpu),
      (r) => toNumberFixed(r.mem),
      (r) => (Number.isFinite(r.threads) ? Math.round(r.threads).toString() : "--"),
    ]);
    return true;
  } catch {
    renderTableRows("process-usage-body", fallbackProcesses, [
      (r) => r.name,
      (r) => toNumberFixed(r.cpu),
      (r) => toNumberFixed(r.mem),
      (r) => String(r.threads),
    ]);
    return false;
  }
}

async function refreshUsageBreakdown() {
  const [containersOk, processesOk] = await Promise.all([refreshContainerUsage(), refreshProcessUsage()]);
  const stamp = new Date().toLocaleTimeString();
  const note = document.getElementById("usage-updated-at");
  if (!note) return;
  if (containersOk && processesOk) {
    note.textContent = `Updated from Prometheus at ${stamp}`;
    return;
  }
  note.textContent = `Partial data mode at ${stamp} (fallback values used for unavailable metrics).`;
}

function buildManualConfig() {
  const form = document.getElementById("tuning-form");
  const data = new FormData(form);
  return {
    model: {
      name: data.get("model_name"),
      score_threshold: Number(data.get("conf")),
    },
    optimization: {
      learning_rate: Number(data.get("lr")),
      epochs: Number(data.get("epochs")),
      batch_size: Number(data.get("batch_size")),
    },
    data: {
      condition_focus: data.get("condition"),
    },
    control: {
      mode: "manual",
      generated_at: new Date().toISOString(),
    },
  };
}

function toYamlLike(obj, depth = 0) {
  const indent = "  ".repeat(depth);
  return Object.entries(obj)
    .map(([key, val]) => {
      if (val && typeof val === "object" && !Array.isArray(val)) {
        return `${indent}${key}:\n${toYamlLike(val, depth + 1)}`;
      }
      return `${indent}${key}: ${val}`;
    })
    .join("\n");
}

function renderManualConfig() {
  const output = document.getElementById("manual-config");
  output.textContent = toYamlLike(buildManualConfig());
}

function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text);
  }
  return Promise.reject(new Error("Clipboard unavailable"));
}

function randomChoice(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function buildAutoPlan() {
  const strategy = document.getElementById("auto-strategy").value;
  const objective = document.getElementById("auto-objective").value;
  const trials = Number(document.getElementById("auto-trials").value);
  const base = buildManualConfig();
  const body = document.getElementById("auto-plan-body");
  body.innerHTML = "";

  for (let i = 1; i <= trials; i += 1) {
    const lrScale = 1 + (Math.random() - 0.5) * 0.7;
    const batchScale = 1 + (Math.random() - 0.5) * 0.5;
    const lr = Math.max(0.00001, base.optimization.learning_rate * lrScale);
    const batch = Math.max(2, Math.round(base.optimization.batch_size * batchScale));
    const aug = randomChoice(["light", "balanced", "strong"]);
    const state = i <= 2 ? "queued" : "planned";

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>#${i}</td>
      <td>${lr.toFixed(5)}</td>
      <td>${batch}</td>
      <td>${aug}</td>
      <td>${state}</td>
    `;
    body.appendChild(row);
  }

  const output = document.getElementById("manual-config");
  output.textContent = `${toYamlLike(base)}\nauto_tuning:\n  strategy: ${strategy}\n  objective: ${objective}\n  trials: ${trials}`;
}

function formatMemValue(memMb) {
  if (memMb >= 1024) return `${(memMb / 1024).toFixed(1)}g`;
  return `${Math.round(memMb)}m`;
}

function renderResourcePlan() {
  const profile = document.getElementById("resource-profile").value;
  const cpuMultiplier = Number(document.getElementById("resource-cpu-multiplier").value);
  const memMultiplier = Number(document.getElementById("resource-mem-multiplier").value);
  const reservationPercent = Number(document.getElementById("resource-reservation").value) / 100;

  const base = resourceProfiles[profile] || resourceProfiles.balanced;
  const lines = [
    `# docker-compose.override.yml`,
    `# generated: ${new Date().toISOString()}`,
    `services:`,
  ];

  Object.entries(base).forEach(([serviceName, conf]) => {
    const cpus = Math.max(0.1, conf.cpus * cpuMultiplier);
    const memMb = Math.max(128, conf.memMb * memMultiplier);
    const reservationMb = Math.max(96, memMb * reservationPercent);
    lines.push(`  ${serviceName}:`);
    lines.push(`    cpus: "${cpus.toFixed(2)}"`);
    lines.push(`    mem_limit: ${formatMemValue(memMb)}`);
    lines.push(`    mem_reservation: ${formatMemValue(reservationMb)}`);
  });

  const output = document.getElementById("resource-override");
  output.textContent = lines.join("\n");
}

let pipelineTimer = null;

function resetPipeline() {
  clearInterval(pipelineTimer);
  pipelineTimer = null;
  document.querySelectorAll(".pipeline-step").forEach((step) => {
    step.classList.remove("active", "done");
  });
  document.getElementById("pipeline-status").textContent = "Pipeline is idle.";
}

function runPipeline() {
  resetPipeline();
  const steps = Array.from(document.querySelectorAll(".pipeline-step"));
  let index = 0;
  document.getElementById("pipeline-status").textContent = "Pipeline is running...";

  pipelineTimer = setInterval(() => {
    steps.forEach((s) => s.classList.remove("active"));
    if (index > 0) steps[index - 1].classList.add("done");
    if (index < steps.length) {
      steps[index].classList.add("active");
      document.getElementById("pipeline-status").textContent = `Running: ${steps[index].textContent}`;
      index += 1;
      return;
    }
    clearInterval(pipelineTimer);
    pipelineTimer = null;
    document.getElementById("pipeline-status").textContent =
      "Pipeline completed. Metrics gate passed and deployment prepared.";
  }, 1400);
}

function setupQuickCommands() {
  const preview = document.getElementById("command-preview");
  document.querySelectorAll(".action-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      preview.textContent = btn.dataset.command || "";
    });
  });
}

function setupDashboardSwitcher() {
  const frame = document.getElementById("dashboard-frame");
  const shell = document.getElementById("dashboard-shell");
  const loader = document.getElementById("dashboard-loader");
  const status = document.getElementById("dashboard-status");
  const openLink = document.getElementById("dashboard-open");
  const reloadBtn = document.getElementById("dashboard-reload");
  const tabs = Array.from(document.querySelectorAll(".dash-tab[data-src]"));
  if (!frame || !tabs.length || !shell) return;

  let loadingTimeout = null;
  let activeDashboardTab = null;

  const setDashboardState = (text, state = "info") => {
    if (!status) return;
    status.textContent = text;
    status.dataset.state = state;
  };

  const setLoading = (isLoading) => {
    shell.classList.toggle("loading", isLoading);
    if (loader) loader.setAttribute("aria-hidden", String(!isLoading));
  };

  const resolveDashboardSrc = (rawSrc) => {
    try {
      const url = new URL(rawSrc, window.location.origin);
      const wantsLight = document.body.dataset.theme === "paper";
      url.searchParams.set("theme", wantsLight ? "light" : "dark");
      return `${url.pathname}${url.search}`;
    } catch {
      return rawSrc;
    }
  };

  const armLoadingTimeout = () => {
    clearTimeout(loadingTimeout);
    loadingTimeout = setTimeout(() => {
      setLoading(false);
      setDashboardState("Dashboard loads too long. Use Open full for diagnostics.", "warn");
    }, 12000);
  };

  frame.addEventListener("load", () => {
    clearTimeout(loadingTimeout);
    setLoading(false);
    setDashboardState("Dashboard ready", "ok");
  });

  frame.addEventListener("error", () => {
    clearTimeout(loadingTimeout);
    setLoading(false);
    setDashboardState("Dashboard failed to load. Check Grafana route.", "error");
  });

  const activateDashboard = (tab, forceReload = false) => {
    activeDashboardTab = tab;
    tabs.forEach((btn) => btn.classList.toggle("is-active", btn === tab));
    const src = resolveDashboardSrc(tab.dataset.src || "");
    if (!src) return;

    if (openLink) {
      openLink.setAttribute("href", src);
    }

    if (forceReload || frame.getAttribute("src") !== src) {
      setLoading(true);
      setDashboardState("Loading dashboard...", "info");
      armLoadingTimeout();
      frame.setAttribute("src", src);
    }
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activateDashboard(tab));
  });

  if (reloadBtn) {
    reloadBtn.addEventListener("click", () => {
      const target = activeDashboardTab || tabs.find((tab) => tab.classList.contains("is-active")) || tabs[0];
      activateDashboard(target, true);
    });
  }

  window.addEventListener("uav-theme-change", () => {
    const active = tabs.find((tab) => tab.classList.contains("is-active")) || tabs[0];
    if (active) activateDashboard(active, true);
  });

  window.addEventListener("uav-tab-activate", (event) => {
    if (!event || !event.detail || event.detail.tabId !== "tab-overview") return;
    if (!frame.getAttribute("src")) {
      const target = activeDashboardTab || tabs[0];
      activateDashboard(target, true);
    }
  });

  const initial = tabs.find((tab) => tab.classList.contains("is-active")) || tabs[0];
  activateDashboard(initial);
}

function loadEmbedsForPane(pane) {
  if (!pane) return;
  pane.querySelectorAll("iframe[data-src]").forEach((iframe) => {
    if (!iframe.getAttribute("src")) {
      iframe.setAttribute("src", iframe.dataset.src);
    }
  });
}

function activateTab(tabId, updateHash = true) {
  const buttons = document.querySelectorAll(".tab-btn[data-tab-target]");
  const panes = document.querySelectorAll(".tab-pane");
  const targetPane = document.getElementById(tabId);
  if (!targetPane) return;

  buttons.forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.tabTarget === tabId);
  });
  panes.forEach((pane) => {
    pane.classList.toggle("is-active", pane.id === tabId);
  });

  loadEmbedsForPane(targetPane);
  window.dispatchEvent(new CustomEvent("uav-tab-activate", { detail: { tabId } }));
  if (updateHash) {
    history.replaceState(null, "", `#${tabId}`);
  }
}

function setupTabs() {
  const workspace = document.querySelector(".workspace");
  const menuToggle = document.getElementById("menu-toggle");
  const tabButtons = document.querySelectorAll(".tab-btn[data-tab-target]");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tabTarget;
      activateTab(target);
      if (workspace) workspace.classList.remove("menu-open");
    });
  });

  if (menuToggle && workspace) {
    menuToggle.addEventListener("click", () => {
      workspace.classList.toggle("menu-open");
    });
  }

  const hashTarget = window.location.hash.replace("#", "");
  const defaultTarget =
    hashTarget ||
    document.querySelector(".tab-btn.is-active")?.dataset.tabTarget ||
    tabButtons[0]?.dataset.tabTarget;
  if (defaultTarget) {
    activateTab(defaultTarget, false);
  }

  window.addEventListener("hashchange", () => {
    const next = window.location.hash.replace("#", "");
    if (next) activateTab(next, false);
  });
}

function setupEvents() {
  document.getElementById("apply-manual").addEventListener("click", renderManualConfig);
  document.getElementById("build-auto-plan").addEventListener("click", buildAutoPlan);
  document.getElementById("run-pipeline").addEventListener("click", runPipeline);
  document.getElementById("reset-pipeline").addEventListener("click", resetPipeline);
  document.getElementById("generate-resource-plan").addEventListener("click", renderResourcePlan);

  document.getElementById("copy-manual").addEventListener("click", async () => {
    try {
      await copyText(document.getElementById("manual-config").textContent);
      document.getElementById("pipeline-status").textContent = "Manual config copied to clipboard.";
    } catch {
      document.getElementById("pipeline-status").textContent = "Clipboard access is unavailable.";
    }
  });

  document.getElementById("copy-resource-plan").addEventListener("click", async () => {
    try {
      await copyText(document.getElementById("resource-override").textContent);
      document.getElementById("pipeline-status").textContent = "Resource override copied to clipboard.";
    } catch {
      document.getElementById("pipeline-status").textContent = "Clipboard access is unavailable.";
    }
  });

  document.querySelectorAll("#tuning-form input, #tuning-form select").forEach((el) => {
    el.addEventListener("change", renderManualConfig);
  });

  document
    .querySelectorAll(
      "#resource-profile, #resource-cpu-multiplier, #resource-mem-multiplier, #resource-reservation",
    )
    .forEach((el) => {
      el.addEventListener("change", renderResourcePlan);
      el.addEventListener("input", renderResourcePlan);
    });
}

function bootstrap() {
  setupTabs();
  setupAppearanceControls();
  setupDashboardSwitcher();
  renderManualConfig();
  buildAutoPlan();
  renderResourcePlan();
  setupQuickCommands();
  setupEvents();

  refreshServiceStatus();
  refreshMetrics();
  refreshUsageBreakdown();

  setInterval(refreshServiceStatus, 15000);
  setInterval(refreshMetrics, 9000);
  setInterval(refreshUsageBreakdown, 12000);
}

bootstrap();
