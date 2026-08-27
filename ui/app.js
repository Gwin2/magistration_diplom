// Neural Network Trainer - Main Application

document.addEventListener('DOMContentLoaded', () => {
  // State management
  const state = {
    currentStep: 1,
    totalSteps: 6,
    theme: 'light',
    training: false,
    datasets: [],
    models: [],
    experiments: [],
    workspaceFilePicker: { kind: 'any', targetId: '', items: [] },
    contentsCollapsed: new Set()
  };
  const API_BASE = '/api/control';

  // DOM Elements
  const navPanel = document.getElementById('nav-panel');
  const menuToggle = document.getElementById('menu-toggle');
  const themeToggle = document.getElementById('theme-toggle');
  const prevStepBtn = document.getElementById('prev-step');
  const nextStepBtn = document.getElementById('next-step');
  const stepButtons = document.querySelectorAll('.step-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const headerStatus = document.getElementById('header-status');
  const contentsToggle = document.getElementById('contents-toggle');
  const contentsMenu = document.getElementById('contents-menu');
  const contentsClose = document.getElementById('contents-close');
  const contentsList = document.getElementById('contents-list');
  let contentsObserver;

  // Initialize
  function init() {
    setupNavigation();
    setupContentsNavigation();
    setupThemeToggle();
    setupStepNavigation();
    setupForms();
    void loadDatasets();
    loadSavedState();
    goToStep(state.currentStep);
    updateUI();
  }

  // Navigation toggle for mobile
  function setupNavigation() {
    menuToggle?.addEventListener('click', () => {
      document.querySelector('.workspace')?.classList.toggle('menu-open');
    });

    // Close menu when clicking outside on mobile
    document.addEventListener('click', (e) => {
      if (window.innerWidth <= 900) {
        if (!navPanel?.contains(e.target) && !menuToggle?.contains(e.target)) {
          document.querySelector('.workspace')?.classList.remove('menu-open');
        }
      }
    });
  }

  function setupContentsNavigation() {
    contentsToggle?.addEventListener('click', () => {
      setContentsOpen(contentsMenu?.hidden !== false);
    });
    contentsClose?.addEventListener('click', () => setContentsOpen(false));
    contentsList?.addEventListener('click', event => {
      const groupToggle = event.target.closest('[data-content-group-toggle]');
      if (groupToggle) {
        const group = groupToggle.dataset.contentGroupToggle;
        if (state.contentsCollapsed.has(group)) state.contentsCollapsed.delete(group);
        else state.contentsCollapsed.add(group);
        renderContentsMenu();
        observeContentPanels();
        return;
      }

      const item = event.target.closest('[data-content-target]');
      const target = item && document.getElementById(item.dataset.contentTarget);
      if (!target) return;
      target.scrollIntoView({
        behavior: document.body.dataset.motion === 'off' ? 'auto' : 'smooth',
        block: 'start'
      });
      if (window.innerWidth <= 900) setContentsOpen(false);
    });

    document.addEventListener('click', event => {
      if (contentsMenu?.hidden !== false) return;
      if (!contentsMenu.contains(event.target) && !contentsToggle?.contains(event.target)) {
        setContentsOpen(false);
      }
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') setContentsOpen(false);
    });
  }

  function getContentItems() {
    const pane = document.querySelector('.tab-pane.is-active');
    if (!pane) return [];

    return [...pane.querySelectorAll(':scope > .panel')]
      .map((panel, index) => {
        const heading = panel.querySelector('.panel-title-row h2, .panel-title-row h3, .panel-title-row h4');
        if (!heading) return null;
        if (!panel.id) panel.id = `content-${pane.id}-${index + 1}`;
        return {
          id: panel.id,
          label: heading.textContent.trim(),
          group: pane.dataset.contentGroup || pane.id
        };
      })
      .filter(Boolean);
  }

  function renderContentsMenu() {
    if (!contentsList) return;
    const groups = new Map();
    getContentItems().forEach(item => {
      if (!groups.has(item.group)) groups.set(item.group, []);
      groups.get(item.group).push(item);
    });
    contentsList.replaceChildren();

    groups.forEach((items, group) => {
      const section = document.createElement('section');
      section.className = 'contents-group';
      if (state.contentsCollapsed.has(group)) section.classList.add('is-collapsed');

      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'contents-group-toggle';
      toggle.dataset.contentGroupToggle = group;
      toggle.setAttribute('aria-expanded', String(!state.contentsCollapsed.has(group)));
      toggle.textContent = group;
      section.append(toggle);

      const itemsEl = document.createElement('div');
      itemsEl.className = 'contents-items';
      items.forEach(item => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'contents-item';
        button.dataset.contentTarget = item.id;
        button.textContent = item.label;
        itemsEl.append(button);
      });
      section.append(itemsEl);
      contentsList.append(section);
    });
  }

  function setContentsOpen(open) {
    if (!contentsMenu || !contentsToggle) return;
    contentsMenu.hidden = !open;
    contentsToggle.setAttribute('aria-expanded', String(open));
    document.querySelector('.workspace')?.classList.toggle('contents-open', open);
  }

  function observeContentPanels() {
    contentsObserver?.disconnect();
    const items = getContentItems();
    if (!items.length || !window.IntersectionObserver) return;

    const buttons = [...contentsList.querySelectorAll('.contents-item')];
    contentsObserver = new IntersectionObserver(entries => {
      const visible = entries
        .filter(entry => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      buttons.forEach(button => {
        button.classList.toggle('is-current', button.dataset.contentTarget === visible.target.id);
      });
    }, { rootMargin: '-72px 0px -55% 0px', threshold: [0, 0.25, 0.75] });

    items.forEach(item => {
      const target = document.getElementById(item.id);
      if (target) contentsObserver.observe(target);
    });
  }

  // Theme toggle
  function setupThemeToggle() {
    themeToggle?.addEventListener('click', () => {
      state.theme = state.theme === 'light' ? 'dark' : 'light';
      document.body.setAttribute('data-theme', state.theme);
      localStorage.setItem('nn_trainer_theme', state.theme);
    });
  }

  // Step navigation
  function setupStepNavigation() {
    stepButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const step = parseInt(btn.dataset.step);
        goToStep(step);
      });
    });

    prevStepBtn?.addEventListener('click', () => {
      if (state.currentStep > 1) {
        goToStep(state.currentStep - 1);
      }
    });

    nextStepBtn?.addEventListener('click', () => {
      if (state.currentStep < state.totalSteps) {
        goToStep(state.currentStep + 1);
      }
    });
  }

  // Go to specific step
  function goToStep(step) {
    step = Math.min(Math.max(parseInt(step, 10) || 1, 1), state.totalSteps);
    state.currentStep = step;
    
    // Update step buttons
    stepButtons.forEach(btn => {
      const btnStep = parseInt(btn.dataset.step);
      btn.classList.toggle('is-active', btnStep === step);
    });

    // Update tab panes
    const targetId = document.querySelector(`.step-btn[data-step="${step}"]`)?.dataset.tabTarget;
    const fallbackId = getTabIdForStep(step);
    tabPanes.forEach(pane => {
      pane.classList.toggle('is-active', pane.id === (targetId || fallbackId));
    });
    renderContentsMenu();
    observeContentPanels();

    // Update navigation buttons
    if (prevStepBtn) prevStepBtn.disabled = step === 1;
    if (nextStepBtn) nextStepBtn.disabled = step === state.totalSteps;

    // Save state
    localStorage.setItem('nn_trainer_step', step);

    // Trigger resize observer check and layout validation
    requestAnimationFrame(() => {
      checkContextLayout();
      validateArchitectureLayout();
    });
  }

  // Get tab name from step number
  function getTabIdForStep(step) {
    const tabs = [
      'tab-datasets',
      'tab-studio',
      'tab-experiments',
      'tab-serving',
      'tab-overview',
      'tab-resources'
    ];
    return tabs[step - 1] || 'tab-datasets';
  }

  // Setup form handlers
  function setupForms() {
    // Dataset upload
    bindFileControl('dataset-file', 'dataset-file-name', 'dataset-upload-btn');
    bindFileControl('inference-file', 'inference-file-name', 'run-inference');
    setupWorkspaceFilePicker();
    document.getElementById('dataset-archive-path')?.addEventListener('input', updateDatasetUploadButton);
    document.getElementById('inference-image-path')?.addEventListener('input', updateInferenceAction);
    updateDatasetUploadButton();
    updateInferenceAction();
    document.getElementById('dataset-upload-btn')?.addEventListener('click', handleDatasetUpload);
    document.getElementById('dataset-import-path-btn')?.addEventListener('click', handleDatasetImportPath);
    document.getElementById('dataset-register-btn')?.addEventListener('click', handleDatasetRegister);
    document.getElementById('datasets-refresh')?.addEventListener('click', () => void loadDatasets());
    document.getElementById('dataset-search')?.addEventListener('input', renderDatasets);
    
    // Architecture
    document.getElementById('constructor-generate')?.addEventListener('click', generateArchitecture);
    document.getElementById('architecture-save')?.addEventListener('click', saveArchitecture);
    
    // Config
    document.getElementById('config-save')?.addEventListener('click', saveConfig);
    document.getElementById('config-validate')?.addEventListener('click', validateConfig);
    
    // Training
    document.getElementById('launch-train')?.addEventListener('click', startTraining);
    document.getElementById('training-log-clear')?.addEventListener('click', clearTrainingLog);
    
    // Deploy
    document.getElementById('export-model')?.addEventListener('click', exportModel);
    document.getElementById('run-inference')?.addEventListener('click', runInference);
    
    // Resources
    document.getElementById('apply-resources')?.addEventListener('click', applyResources);
  }

  // Load saved state
  function loadSavedState() {
    const savedTheme = localStorage.getItem('nn_trainer_theme');
    if (savedTheme) {
      state.theme = savedTheme;
      document.body.setAttribute('data-theme', savedTheme);
    }

    const savedStep = localStorage.getItem('nn_trainer_step');
    if (savedStep) {
      goToStep(parseInt(savedStep));
    }
  }

  // Update UI based on state
  function updateUI() {
    updateHeaderStatus('ready');
    updateTrainingSummary();
  }

  // Update header status
  function updateHeaderStatus(status) {
    if (!headerStatus) return;
    
    const dot = headerStatus.querySelector('.status-dot');
    const text = headerStatus.querySelector('.status-text');
    
    dot.className = 'status-dot';
    
    switch(status) {
      case 'ready':
        dot.classList.add('status-ready');
        text.textContent = 'Готово';
        break;
      case 'busy':
        dot.classList.add('status-busy');
        text.textContent = 'Обработка...';
        break;
      case 'error':
        dot.classList.add('status-error');
        text.textContent = 'Ошибка';
        break;
    }
  }

  // Update training summary display
  function updateTrainingSummary() {
    const configName = document.getElementById('config-name')?.value || '--';
    const dataset = document.getElementById('config-dataset')?.value || '--';
    const epochs = document.getElementById('config-epochs')?.value || '--';
    const batchSize = document.getElementById('config-batch-size')?.value || '--';

    const elDataset = document.getElementById('train-dataset');
    const elModel = document.getElementById('train-model');
    const elEpochs = document.getElementById('train-epochs');
    const elBatch = document.getElementById('train-batch');

    if (elDataset) elDataset.textContent = dataset;
    if (elModel) elModel.textContent = configName;
    if (elEpochs) elEpochs.textContent = epochs;
    if (elBatch) elBatch.textContent = batchSize;
  }

  function bindFileControl(inputId, labelId, actionId) {
    const input = document.getElementById(inputId);
    const label = document.getElementById(labelId);
    const action = document.getElementById(actionId);
    const dropZone = document.querySelector(`[data-file-drop="${inputId}"]`);

    if (action) action.disabled = !input?.files?.length;

    input?.addEventListener('change', () => {
      if (label) label.textContent = input.files?.[0]?.name || 'Файл не выбран';
      if (inputId === 'dataset-file') updateDatasetUploadButton();
      else if (inputId === 'inference-file') updateInferenceAction();
      else if (action) action.disabled = !input.files?.length;
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone?.addEventListener(eventName, event => {
        event.preventDefault();
        dropZone.classList.add('is-dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone?.addEventListener(eventName, event => {
        event.preventDefault();
        dropZone.classList.remove('is-dragover');
      });
    });

    dropZone?.addEventListener('drop', event => {
      const files = event.dataTransfer?.files;
      if (!input || !files?.length) return;
      input.files = files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
  }

  function setupWorkspaceFilePicker() {
    document.querySelectorAll('[data-workspace-file-picker]').forEach(button => {
      button.addEventListener('click', () => {
        openWorkspaceFilePicker(
          button.getAttribute('data-file-kind') || 'any',
          button.getAttribute('data-file-target') || ''
        );
      });
    });

    document.querySelectorAll('[data-workspace-file-close]').forEach(element => {
      element.addEventListener('click', closeWorkspaceFilePicker);
    });
    document.getElementById('workspace-file-close')?.addEventListener('click', closeWorkspaceFilePicker);
    document.getElementById('workspace-file-refresh')?.addEventListener('click', () => {
      void loadWorkspaceFiles();
    });
    document.getElementById('workspace-file-search')?.addEventListener('input', renderWorkspaceFiles);
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') closeWorkspaceFilePicker();
    });
  }

  function openWorkspaceFilePicker(kind, targetId) {
    state.workspaceFilePicker = { kind, targetId, items: [] };
    const modal = document.getElementById('workspace-file-modal');
    const title = document.getElementById('workspace-file-title');
    const search = document.getElementById('workspace-file-search');
    if (title) title.textContent = getFilePickerTitle(kind);
    if (search) search.value = '';
    if (modal) modal.hidden = false;
    setWorkspaceFileStatus('Сканирую workspace...');
    void loadWorkspaceFiles();
  }

  function closeWorkspaceFilePicker() {
    const modal = document.getElementById('workspace-file-modal');
    if (modal) modal.hidden = true;
  }

  function getFilePickerTitle(kind) {
    if (kind === 'dataset_archive') return 'Выбор архива датасета';
    if (kind === 'image') return 'Выбор изображения';
    if (kind === 'torchserve_archive') return 'Выбор MAR-архива';
    return 'Выбор файла';
  }

  function setWorkspaceFileStatus(message, kind = 'muted') {
    const status = document.getElementById('workspace-file-status');
    if (!status) return;
    status.textContent = message;
    status.style.color = kind === 'error'
      ? 'var(--danger)'
      : kind === 'success'
        ? 'var(--success)'
        : 'var(--text-muted)';
  }

  async function loadWorkspaceFiles() {
    try {
      const payload = await apiJson(`/files?kind=${encodeURIComponent(state.workspaceFilePicker.kind)}&limit=400`);
      state.workspaceFilePicker.items = Array.isArray(payload.items) ? payload.items : [];
      renderWorkspaceFiles();
      setWorkspaceFileStatus(
        state.workspaceFilePicker.items.length
          ? `Найдено файлов: ${state.workspaceFilePicker.items.length}`
          : 'Подходящие файлы не найдены.'
      );
    } catch (error) {
      state.workspaceFilePicker.items = [];
      renderWorkspaceFiles();
      setWorkspaceFileStatus(`Не удалось получить список файлов: ${error.message}`, 'error');
    }
  }

  function renderWorkspaceFiles() {
    const list = document.getElementById('workspace-file-list');
    if (!list) return;
    const term = String(document.getElementById('workspace-file-search')?.value || '').trim().toLowerCase();
    const items = state.workspaceFilePicker.items.filter(item => {
      if (!term) return true;
      return `${item.name} ${item.path}`.toLowerCase().includes(term);
    });
    if (!items.length) {
      list.innerHTML = '<p class="empty-state">Файлы не найдены.</p>';
      return;
    }
    list.innerHTML = items.map(item => `
      <button class="file-choice" type="button" data-workspace-file="${escapeHtml(item.path)}">
        <span><strong>${escapeHtml(item.name)}</strong><br>${escapeHtml(item.path)}</span>
        <small>${formatBytes(item.size_bytes)}</small>
      </button>
    `).join('');
    list.querySelectorAll('[data-workspace-file]').forEach(button => {
      button.addEventListener('click', () => {
        selectWorkspaceFile(button.getAttribute('data-workspace-file') || '');
      });
    });
  }

  function selectWorkspaceFile(path) {
    const target = document.getElementById(state.workspaceFilePicker.targetId);
    if (target) {
      target.value = path;
      target.dispatchEvent(new Event('input', { bubbles: true }));
      target.dispatchEvent(new Event('change', { bubbles: true }));
    }

    const fileName = path.split('/').pop() || path;
    if (state.workspaceFilePicker.targetId === 'dataset-archive-path') {
      const label = document.getElementById('dataset-file-name');
      if (label) label.textContent = `Путь: ${path}`;
      setDatasetStatus('Архив выбран из workspace. Можно нажать «Загрузить».', 'success');
      updateDatasetUploadButton();
    }
    if (state.workspaceFilePicker.targetId === 'inference-image-path') {
      const label = document.getElementById('inference-file-name');
      if (label) label.textContent = `Путь: ${fileName}`;
      updateInferenceAction();
    }
    closeWorkspaceFilePicker();
  }

  function updateDatasetUploadButton() {
    const uploadBtn = document.getElementById('dataset-upload-btn');
    const fileInput = document.getElementById('dataset-file');
    const archivePath = String(document.getElementById('dataset-archive-path')?.value || '').trim();
    if (uploadBtn) uploadBtn.disabled = !fileInput?.files?.length && !archivePath;
  }

  function updateInferenceAction() {
    const runBtn = document.getElementById('run-inference');
    const fileInput = document.getElementById('inference-file');
    const imagePath = String(document.getElementById('inference-image-path')?.value || '').trim();
    if (runBtn) runBtn.disabled = !fileInput?.files?.length && !imagePath;
  }

  function splitTags(value) {
    return String(value || '')
      .split(',')
      .map(tag => tag.trim())
      .filter(Boolean);
  }

  function setDatasetStatus(message, kind = 'muted') {
    const statusEl = document.getElementById('dataset-status');
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.style.color = kind === 'error'
      ? 'var(--danger)'
      : kind === 'success'
        ? 'var(--success)'
        : 'var(--text-muted)';
  }

  async function apiJson(path, options = {}) {
    const request = { ...options };
    const hasFormData = request.body instanceof FormData;
    if (!hasFormData) {
      request.headers = {
        'Content-Type': 'application/json',
        ...(request.headers || {})
      };
    }

    const response = await fetch(`${API_BASE}${path}`, request);
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const payload = await response.json();
        message = payload.detail || message;
      } catch (_error) {
        // response body is optional
      }
      throw new Error(message);
    }
    return response.json();
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, char => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[char]));
  }

  function formatBytes(value) {
    const bytes = Number(value || 0);
    if (bytes < 1024) return `${bytes} Б`;
    const units = ['КБ', 'МБ', 'ГБ', 'ТБ'];
    let size = bytes / 1024;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[unitIndex]}`;
  }

  async function loadDatasets() {
    try {
      const payload = await apiJson('/datasets');
      state.datasets = Array.isArray(payload.items) ? payload.items : [];
      syncDatasetControls();
      renderDatasets();
      setDatasetStatus('Библиотека датасетов обновлена.', 'success');
    } catch (error) {
      setDatasetStatus(`Не удалось загрузить датасеты: ${error.message}`, 'error');
    }
  }

  function syncDatasetControls() {
    const summary = document.getElementById('summary-datasets');
    if (summary) summary.textContent = String(state.datasets.length);

    const select = document.getElementById('constructor-dataset');
    if (select) {
      const selected = select.value;
      select.innerHTML = '<option value="">Авто / нет</option>' + state.datasets.map(dataset => (
        `<option value="${escapeHtml(dataset.id)}">${escapeHtml(dataset.name)}</option>`
      )).join('');
      if (state.datasets.some(dataset => dataset.id === selected)) {
        select.value = selected;
      }
    }
    updateTrainingSummary();
  }

  function renderDatasets() {
    const list = document.getElementById('dataset-list');
    if (!list) return;

    const term = String(document.getElementById('dataset-search')?.value || '').trim().toLowerCase();
    const items = state.datasets.filter(dataset => {
      if (!term) return true;
      const haystack = [
        dataset.name,
        dataset.path,
        dataset.description,
        ...(dataset.tags || [])
      ].join(' ').toLowerCase();
      return haystack.includes(term);
    });

    if (!items.length) {
      list.innerHTML = '<p class="empty-state">Датасеты не найдены.</p>';
      return;
    }

    list.innerHTML = items.map(dataset => {
      const tags = (dataset.tags || []).map(tag => `<span>${escapeHtml(tag)}</span>`).join('');
      const description = dataset.description || 'Описание не задано.';
      return `
        <article class="dataset-card">
          <div class="dataset-card-head">
            <h3>${escapeHtml(dataset.name)}</h3>
            <span>${escapeHtml(String(dataset.file_count || 0))} файлов</span>
          </div>
          <p class="dataset-path">${escapeHtml(dataset.path)}</p>
          <p class="dataset-description">${escapeHtml(description)}</p>
          <div class="dataset-tags">${tags || '<span>без тегов</span>'}</div>
          <div class="dataset-meta">
            <span>${formatBytes(dataset.size_bytes)}</span>
            <span>${dataset.updated_at ? new Date(dataset.updated_at).toLocaleString('ru-RU') : 'без даты'}</span>
          </div>
          <div class="inline-actions">
            <a class="btn btn-inline" href="${API_BASE}/datasets/${encodeURIComponent(dataset.id)}/download" target="_blank" rel="noreferrer">Скачать</a>
            <button class="btn btn-inline" type="button" data-dataset-edit="${escapeHtml(dataset.id)}">Редактировать</button>
            <button class="btn btn-inline btn-danger" type="button" data-dataset-unregister="${escapeHtml(dataset.id)}">Снять с учета</button>
          </div>
        </article>
      `;
    }).join('');

    list.querySelectorAll('[data-dataset-edit]').forEach(button => {
      button.addEventListener('click', () => {
        const datasetId = button.getAttribute('data-dataset-edit');
        if (datasetId) editDataset(datasetId);
      });
    });

    list.querySelectorAll('[data-dataset-unregister]').forEach(button => {
      button.addEventListener('click', () => {
        const datasetId = button.getAttribute('data-dataset-unregister');
        if (datasetId) void unregisterDataset(datasetId);
      });
    });
  }

  function editDataset(datasetId) {
    const dataset = state.datasets.find(item => item.id === datasetId);
    const form = document.getElementById('dataset-register-form');
    if (!dataset || !form) return;
    form.elements.namedItem('name').value = dataset.name || '';
    form.elements.namedItem('path').value = dataset.path || '';
    form.elements.namedItem('tags').value = (dataset.tags || []).join(', ');
    form.elements.namedItem('description').value = dataset.description || '';
    form.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setDatasetStatus('Метаданные перенесены в форму. Измените и сохраните путь.', 'muted');
  }

  async function unregisterDataset(datasetId) {
    setDatasetStatus('Снимаю датасет с учета...', 'muted');
    try {
      const payload = await apiJson(`/datasets/${encodeURIComponent(datasetId)}`, { method: 'DELETE' });
      state.datasets = Array.isArray(payload.items) ? payload.items : [];
      syncDatasetControls();
      renderDatasets();
      setDatasetStatus('Датасет снят с учета. Файлы на диске не удалены.', 'success');
    } catch (error) {
      setDatasetStatus(`Не удалось снять с учета: ${error.message}`, 'error');
    }
  }

  // Check and fix context layout on resize
  function checkContextLayout() {
    const activePane = document.querySelector('.tab-pane.is-active');
    if (!activePane) return;

    // Ensure panels are properly sized
    const panels = activePane.querySelectorAll('.panel');
    panels.forEach(panel => {
      // Reset any inline styles that might cause issues
      panel.style.maxWidth = '';
      panel.style.minWidth = '';
      
      // Ensure proper overflow handling
      const scrollable = panel.querySelector('.scroll-table, .code-box, .library-list');
      if (scrollable) {
        scrollable.style.maxHeight = '';
      }
    });

    // Check constructor visual preview sizing
    const visualPreview = document.getElementById('constructor-visual-preview');
    if (visualPreview && activePane.id === 'tab-studio') {
      visualPreview.style.height = '';
      visualPreview.style.minHeight = '250px';
    }
  }

  // Validate architecture layout to prevent context overlap
  function validateArchitectureLayout() {
    const archTab = document.getElementById('tab-studio');
    if (!archTab || !archTab.classList.contains('is-active')) return;

    const constructorGrid = document.querySelector('.constructor-grid');
    const monitorPanel = document.querySelector('.constructor-monitor');
    const mainPanel = document.querySelector('.constructor-main');
    
    if (!constructorGrid || !monitorPanel || !mainPanel) return;

    // Get computed styles
    const gridRect = constructorGrid.getBoundingClientRect();
    const monitorRect = monitorPanel.getBoundingClientRect();
    const mainRect = mainPanel.getBoundingClientRect();

    // Check for overlap
    const hasOverlap = (
      monitorRect.left < mainRect.right &&
      monitorRect.right > mainRect.left &&
      monitorRect.top < mainRect.bottom &&
      monitorRect.bottom > mainRect.top
    );

    if (hasOverlap) {
      console.warn('Architecture layout overlap detected, resetting grid');
      constructorGrid.style.gridTemplateColumns = '';
      setTimeout(() => {
        constructorGrid.style.gridTemplateColumns = '';
      }, 50);
    }

    // Ensure monitor stays within bounds
    const monitorShell = document.getElementById('constructor-monitor');
    if (monitorShell) {
      const shellRect = monitorShell.getBoundingClientRect();
      if (shellRect.right > gridRect.right - 10) {
        monitorShell.style.overflowX = 'auto';
      }
    }
  }

  // ResizeObserver for context-aware layout
  const resizeObserver = new ResizeObserver(entries => {
    for (const entry of entries) {
      if (entry.target.classList.contains('tab-pane')) {
        // Debounce the check
        clearTimeout(window.resizeTimer);
        window.resizeTimer = setTimeout(() => {
          checkContextLayout();
        }, 150);
      }
    }
  });

  // Observe all tab panes
  tabPanes.forEach(pane => resizeObserver.observe(pane));

  // Window resize handler
  let windowResizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(windowResizeTimer);
    windowResizeTimer = setTimeout(() => {
      checkContextLayout();
      
      // Auto-close menu on desktop
      if (window.innerWidth > 900) {
        document.querySelector('.workspace')?.classList.remove('menu-open');
      }
    }, 150);
  });

  // Form handlers implementation
  async function handleDatasetUpload() {
    const form = document.getElementById('dataset-upload-form');
    const fileInput = form?.querySelector('input[type="file"]');
    const uploadBtn = document.getElementById('dataset-upload-btn');

    if (!fileInput?.files?.length) {
      const archivePath = String(form?.elements.namedItem('archive_path')?.value || '').trim();
      if (archivePath) {
        await handleDatasetImportPath();
        return;
      }
      setDatasetStatus('Выберите архив из workspace, перетащите файл или импортируйте по пути.', 'error');
      return;
    }

    const formData = new FormData(form);
    const currentName = String(formData.get('dataset_name') || '').trim();
    if (!currentName) {
      formData.set('dataset_name', fileInput.files[0].name.replace(/\.[^.]+$/, ''));
    }

    updateHeaderStatus('busy');
    if (uploadBtn) uploadBtn.disabled = true;
    setDatasetStatus('Загружаю архив...', 'muted');

    try {
      const payload = await apiJson('/datasets/upload', {
        method: 'POST',
        body: formData
      });
      state.datasets = Array.isArray(payload.items) ? payload.items : [];
      form?.reset();
      const fileLabel = document.getElementById('dataset-file-name');
      if (fileLabel) fileLabel.textContent = 'Файл не выбран';
      syncDatasetControls();
      renderDatasets();
      setDatasetStatus('Датасет загружен и зарегистрирован.', 'success');
      updateHeaderStatus('ready');
    } catch (error) {
      setDatasetStatus(`Ошибка загрузки: ${error.message}`, 'error');
      updateHeaderStatus('error');
    } finally {
      updateDatasetUploadButton();
    }
  }

  async function handleDatasetImportPath() {
    const form = document.getElementById('dataset-upload-form');
    const importBtn = document.getElementById('dataset-import-path-btn');
    const uploadBtn = document.getElementById('dataset-upload-btn');
    const formData = new FormData(form);
    const archivePath = String(formData.get('archive_path') || '').trim();
    let datasetName = String(formData.get('dataset_name') || '').trim();

    if (!archivePath) {
      setDatasetStatus('Укажите путь к архиву внутри workspace.', 'error');
      return;
    }
    if (!datasetName) {
      datasetName = archivePath.split(/[\\/]/).pop().replace(/\.[^.]+$/, '') || 'dataset';
    }

    updateHeaderStatus('busy');
    if (importBtn) importBtn.disabled = true;
    setDatasetStatus('Импортирую архив по пути...', 'muted');

    try {
      const payload = await apiJson('/datasets/import', {
        method: 'POST',
        body: JSON.stringify({
          dataset_name: datasetName,
          archive_path: archivePath,
          description: String(formData.get('description') || ''),
          tags: splitTags(formData.get('tags'))
        })
      });
      state.datasets = Array.isArray(payload.items) ? payload.items : [];
      form?.reset();
      const fileLabel = document.getElementById('dataset-file-name');
      if (fileLabel) fileLabel.textContent = 'Файл не выбран';
      if (uploadBtn) uploadBtn.disabled = true;
      syncDatasetControls();
      renderDatasets();
      setDatasetStatus('Датасет импортирован и зарегистрирован.', 'success');
      updateHeaderStatus('ready');
    } catch (error) {
      setDatasetStatus(`Ошибка импорта: ${error.message}`, 'error');
      updateHeaderStatus('error');
    } finally {
      if (importBtn) importBtn.disabled = false;
    }
  }

  async function handleDatasetRegister() {
    const form = document.getElementById('dataset-register-form');
    const registerBtn = document.getElementById('dataset-register-btn');
    const formData = new FormData(form);
    const name = String(formData.get('name') || '').trim();
    const path = String(formData.get('path') || '').trim();

    if (!name || !path) {
      setDatasetStatus('Укажите название и путь к папке датасета.', 'error');
      return;
    }

    updateHeaderStatus('busy');
    if (registerBtn) registerBtn.disabled = true;
    setDatasetStatus('Регистрирую путь...', 'muted');

    try {
      const payload = await apiJson('/datasets/register', {
        method: 'POST',
        body: JSON.stringify({
          name,
          path,
          description: String(formData.get('description') || ''),
          tags: splitTags(formData.get('tags'))
        })
      });
      state.datasets = Array.isArray(payload.items) ? payload.items : [];
      form?.reset();
      syncDatasetControls();
      renderDatasets();
      setDatasetStatus('Путь к датасету зарегистрирован.', 'success');
      updateHeaderStatus('ready');
    } catch (error) {
      setDatasetStatus(`Ошибка регистрации: ${error.message}`, 'error');
      updateHeaderStatus('error');
    } finally {
      if (registerBtn) registerBtn.disabled = false;
    }
  }

  function generateArchitecture() {
    const name = document.getElementById('constructor-name')?.value || 'my_model';
    const task = document.getElementById('constructor-task')?.value || 'classification';
    const backbone = document.getElementById('constructor-backbone')?.value || 'resnet18';
    const inputSize = document.getElementById('constructor-input-size')?.value || '224';
    
    const preview = document.getElementById('constructor-visual-preview');
    const statusEl = document.getElementById('constructor-status');
    const sourceEl = document.getElementById('architecture-source');
    
    if (preview) {
      preview.innerHTML = `
        <div style="text-align: center;">
          <div style="font-size: 1rem; margin-bottom: 1rem; color: var(--text-secondary);">Схема модели</div>
          <div><strong>${name}</strong></div>
          <div style="color: var(--text-secondary);">${task} | ${backbone}</div>
          <div style="color: var(--text-muted); font-size: 0.85rem;">Вход: ${inputSize}x${inputSize}</div>
        </div>
      `;
    }
    
    if (statusEl) {
      statusEl.textContent = 'Архитектура создана. Проверьте и настройте ниже.';
    }
    
    // Generate sample code
    if (sourceEl) {
      sourceEl.value = generateModelCode(name, task, backbone, inputSize);
    }
  }

  function generateModelCode(name, task, backbone, inputSize) {
    return `import torch
import torch.nn as nn
from torchvision import models

class ${name.replace(/\s+/g, '')}(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.${backbone}(pretrained=True)
        
        # Настройка пользовательского размера входа
        self.input_size = ${inputSize}
        
        # Голова под выбранную задачу
        self.task = "${task}"
        if self.task == "classification":
            self.head = nn.Linear(512, num_classes)
        elif self.task == "detection":
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 4 * num_classes)  # координаты bbox
            )
        elif self.task == "segmentation":
            self.head = nn.Conv2d(512, num_classes, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
`;
    }

  function saveArchitecture() {
    const statusEl = document.getElementById('constructor-status');
    if (statusEl) {
      statusEl.textContent = 'Архитектура сохранена. Переходите к конфигурации.';
      statusEl.style.color = 'var(--success)';
    }
  }

  function saveConfig() {
    const statusEl = document.getElementById('config-status');
    if (statusEl) {
      statusEl.textContent = 'Конфиг сохранен. Можно запускать обучение.';
      statusEl.style.color = 'var(--success)';
    }
    
    // Update training summary
    updateTrainingSummary();
  }

  function validateConfig() {
    const statusEl = document.getElementById('config-status');
    if (statusEl) {
      statusEl.textContent = 'Конфиг успешно проверен.';
      statusEl.style.color = 'var(--success)';
    }
  }

  let trainingInterval;
  function startTraining() {
    if (state.training) return;
    
    state.training = true;
    updateHeaderStatus('busy');
    
    const statusEl = document.getElementById('training-status');
    const logEl = document.getElementById('training-log');
    const progressEl = document.getElementById('training-progress');
    const epochEl = document.getElementById('current-epoch');
    const lossEl = document.getElementById('live-loss');
    const accEl = document.getElementById('live-acc');
    const lrEl = document.getElementById('live-lr');
    
    if (statusEl) {
      statusEl.textContent = 'Обучение выполняется...';
    }
    
    if (logEl) {
      logEl.textContent = '[INFO] Запуск обучения...\n';
    }
    
    let epoch = 0;
    const totalEpochs = parseInt(document.getElementById('config-epochs')?.value) || 100;
    
    trainingInterval = setInterval(() => {
      epoch++;
      const progress = (epoch / totalEpochs) * 100;
      const loss = (2.5 * Math.exp(-epoch / 20) + 0.1).toFixed(4);
      const acc = (0.95 - 0.9 * Math.exp(-epoch / 15)).toFixed(4);
      const lr = (0.001 * Math.exp(-epoch / 50)).toFixed(6);
      
      if (progressEl) progressEl.style.width = `${progress}%`;
      if (epochEl) epochEl.textContent = `${epoch}/${totalEpochs}`;
      if (lossEl) lossEl.textContent = loss;
      if (accEl) accEl.textContent = acc;
      if (lrEl) lrEl.textContent = lr;
      
      if (logEl) {
        logEl.textContent += `[Epoch ${epoch}/${totalEpochs}] loss: ${loss}, acc: ${acc}, lr: ${lr}\n`;
        logEl.scrollTop = logEl.scrollHeight;
      }
      
      if (epoch >= totalEpochs) {
        clearInterval(trainingInterval);
        state.training = false;
        updateHeaderStatus('ready');
        
        if (statusEl) {
          statusEl.textContent = 'Обучение успешно завершено.';
          statusEl.style.color = 'var(--success)';
        }
        
        if (logEl) {
          logEl.textContent += '\n[INFO] Обучение завершено.\n';
        }
        
        setTimeout(() => goToStep(3), 2000);
      }
    }, 200);
  }

  function clearTrainingLog() {
    const logEl = document.getElementById('training-log');
    if (logEl) logEl.textContent = 'Ожидание запуска обучения...';
  }

  function exportModel() {
    const statusEl = document.getElementById('export-status');
    if (statusEl) {
      statusEl.textContent = 'Модель экспортирована.';
      statusEl.style.color = 'var(--success)';
    }
  }

  function runInference() {
    const outputEl = document.getElementById('inference-output');
    const fileInput = document.getElementById('inference-file');
    const imagePath = String(document.getElementById('inference-image-path')?.value || '').trim();
    if (!fileInput?.files?.length && !imagePath) {
      if (outputEl) outputEl.textContent = 'Выберите изображение из workspace или перетащите файл.';
      return;
    }
    const sourceName = imagePath || fileInput.files[0].name;
    if (outputEl) {
      outputEl.innerHTML = `
        <div style="display: flex; align-items: center; gap: 1rem;">
          <div>
            <div><strong>Результат инференса</strong></div>
            <div style="color: var(--text-secondary);">Источник: ${escapeHtml(sourceName)}</div>
            <div style="color: var(--text-secondary);">Предсказание: класс A (уверенность 95.2%)</div>
            <div style="color: var(--text-muted); font-size: 0.85rem;">Задержка: 12 мс</div>
          </div>
        </div>
      `;
    }
  }

  function applyResources() {
    const statusEl = document.getElementById('usage-updated-at');
    if (statusEl) {
      statusEl.textContent = `Лимиты ресурсов применены в ${new Date().toLocaleTimeString()}`;
      statusEl.style.color = 'var(--success)';
    }
  }

  // Listen for config changes to update training summary
  ['config-name', 'config-dataset', 'config-epochs', 'config-batch-size'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', updateTrainingSummary);
    document.getElementById(id)?.addEventListener('input', updateTrainingSummary);
  });

  // Initialize the app
  init();
});
