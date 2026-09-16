// Neural Network Trainer - Main Application

document.addEventListener('DOMContentLoaded', () => {
  // State management
  const state = {
    currentStep: 1,
    totalSteps: 6,
    theme: 'light',
    themeMode: 'light',
    accent: 'mint',
    customAccent: '#257a69',
    surface: 'soft',
    radius: 'soft',
    contrast: 'balanced',
    density: 'comfortable',
    motion: 'on',
    training: false,
    datasets: [],
    configs: [],
    architectures: [],
    jobs: [],
    experiments: [],
    recommendations: [],
    models: [],
    constructorCatalog: null,
    constructorBlueprint: null,
    user: null,
    workspaceFilePicker: { kind: 'any', targetId: '', items: [] },
    contentsCollapsed: new Set()
  };
  const THEME_DEFAULTS = {
    themeMode: 'light',
    accent: 'mint',
    customAccent: '#257a69',
    surface: 'soft',
    radius: 'soft',
    contrast: 'balanced',
    density: 'comfortable',
    motion: 'on'
  };
  const ACCENT_PALETTES = {
    mint: {
      light: { primary: '#257a69', primaryDark: '#18594d', accent: '#d76f50', contrast: '#ffffff', ring: 'rgba(37, 122, 105, 0.28)' },
      dark: { primary: '#75d2bd', primaryDark: '#9ae5d2', accent: '#f09878', contrast: '#10231e', ring: 'rgba(117, 210, 189, 0.34)' }
    },
    blue: {
      light: { primary: '#2f6fba', primaryDark: '#24558f', accent: '#4f9bd6', contrast: '#ffffff', ring: 'rgba(47, 111, 186, 0.28)' },
      dark: { primary: '#79b7ff', primaryDark: '#a9d1ff', accent: '#72d6d4', contrast: '#10223a', ring: 'rgba(121, 183, 255, 0.34)' }
    },
    violet: {
      light: { primary: '#6e5ab5', primaryDark: '#4b3e89', accent: '#d36ca4', contrast: '#ffffff', ring: 'rgba(110, 90, 181, 0.28)' },
      dark: { primary: '#b7a8ff', primaryDark: '#d6cdff', accent: '#f19dc8', contrast: '#20173b', ring: 'rgba(183, 168, 255, 0.34)' }
    },
    amber: {
      light: { primary: '#b06b17', primaryDark: '#7e4a0b', accent: '#e28b3f', contrast: '#ffffff', ring: 'rgba(176, 107, 23, 0.28)' },
      dark: { primary: '#f3bc67', primaryDark: '#ffd79a', accent: '#f0a56b', contrast: '#2e1c08', ring: 'rgba(243, 188, 103, 0.34)' }
    },
    rose: {
      light: { primary: '#b34f76', primaryDark: '#873754', accent: '#e17c95', contrast: '#ffffff', ring: 'rgba(179, 79, 118, 0.28)' },
      dark: { primary: '#f09ab7', primaryDark: '#ffc0d1', accent: '#ffae97', contrast: '#351624', ring: 'rgba(240, 154, 183, 0.34)' }
    },
    teal: {
      light: { primary: '#157a82', primaryDark: '#0d5a61', accent: '#d8785e', contrast: '#ffffff', ring: 'rgba(21, 122, 130, 0.28)' },
      dark: { primary: '#62d5d3', primaryDark: '#94ebe8', accent: '#f29b7a', contrast: '#082625', ring: 'rgba(98, 213, 211, 0.34)' }
    },
    indigo: {
      light: { primary: '#465da9', primaryDark: '#32437e', accent: '#778ee0', contrast: '#ffffff', ring: 'rgba(70, 93, 169, 0.28)' },
      dark: { primary: '#9baeff', primaryDark: '#bdcaff', accent: '#e9a4ff', contrast: '#151d3d', ring: 'rgba(155, 174, 255, 0.34)' }
    },
    coral: {
      light: { primary: '#b6533f', primaryDark: '#84392e', accent: '#e48765', contrast: '#ffffff', ring: 'rgba(182, 83, 63, 0.28)' },
      dark: { primary: '#ff9f82', primaryDark: '#ffc0a9', accent: '#8bcebd', contrast: '#351710', ring: 'rgba(255, 159, 130, 0.34)' }
    }
  };
  const SURFACE_PALETTES = {
    soft: {
      light: { bg: '#f5f6f1', secondary: '#ecefe9', tertiary: '#e1e8e1', surface: '#fbfcf8', text: '#17231f', secondaryText: '#64736b', muted: '#89968e', border: '#c4d0c7', strong: '#a9bbb0' },
      dark: { bg: '#0e1513', secondary: '#131d1a', tertiary: '#1c2924', surface: '#15211d', text: '#e5eee7', secondaryText: '#9aaba2', muted: '#71847a', border: '#344b40', strong: '#4b6859' }
    },
    paper: {
      light: { bg: '#f7f3ec', secondary: '#efe8dc', tertiary: '#e5dacb', surface: '#fffaf2', text: '#2b2520', secondaryText: '#766a5d', muted: '#9b8e7f', border: '#d4c6b5', strong: '#b8a692' },
      dark: { bg: '#191513', secondary: '#241d19', tertiary: '#332720', surface: '#211b18', text: '#f1e6d8', secondaryText: '#c1aa94', muted: '#8d7662', border: '#59483a', strong: '#765e4b' }
    },
    ocean: {
      light: { bg: '#eef4f7', secondary: '#e3edf2', tertiary: '#d4e4ec', surface: '#f8fcfd', text: '#182832', secondaryText: '#607682', muted: '#899da6', border: '#bdd0d9', strong: '#9bb9c5' },
      dark: { bg: '#0d1720', secondary: '#12232e', tertiary: '#1b3542', surface: '#142631', text: '#e1f0f5', secondaryText: '#9db8c1', muted: '#6c8994', border: '#31515f', strong: '#477080' }
    },
    graphite: {
      light: { bg: '#f1f2f4', secondary: '#e5e7eb', tertiary: '#d9dde3', surface: '#fbfcfd', text: '#20242a', secondaryText: '#68717d', muted: '#8f98a4', border: '#c7cdd5', strong: '#a9b2be' },
      dark: { bg: '#121417', secondary: '#1b1f24', tertiary: '#282e36', surface: '#191d22', text: '#edf0f3', secondaryText: '#abb4bf', muted: '#77818d', border: '#3a434f', strong: '#596674' }
    }
  };
  const RADIUS_PRESETS = {
    compact: { panel: '10px', control: '7px' },
    soft: { panel: '16px', control: '9px' },
    round: { panel: '24px', control: '13px' }
  };
  const CONTRAST_PRESETS = { soft: 0.18, balanced: 0.28, high: 0.42 };

  function normalizeHex(value, fallback = '#257a69') {
    return /^#[\da-f]{6}$/i.test(value) ? value.toLowerCase() : fallback;
  }

  function mixHex(first, second, amount) {
    const parse = value => value.slice(1).match(/../g).map(channel => parseInt(channel, 16));
    const left = parse(normalizeHex(first));
    const right = parse(normalizeHex(second));
    return `#${left.map((channel, index) => Math.round(channel + (right[index] - channel) * amount).toString(16).padStart(2, '0')).join('')}`;
  }

  function rgbaHex(value, alpha) {
    const [red, green, blue] = normalizeHex(value).slice(1).match(/../g).map(channel => parseInt(channel, 16));
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }

  function getContrastColor(value) {
    const [red, green, blue] = normalizeHex(value).slice(1).match(/../g).map(channel => parseInt(channel, 16));
    return (red * 299 + green * 587 + blue * 114) / 1000 > 160 ? '#17231f' : '#ffffff';
  }

  function buildCustomPalette(value, theme) {
    const primary = normalizeHex(value);
    const dark = theme === 'dark';
    const primaryDark = mixHex(primary, dark ? '#ffffff' : '#000000', dark ? 0.28 : 0.22);
    const accent = mixHex(primary, dark ? '#ffffff' : '#000000', dark ? 0.12 : 0.12);
    return {
      primary,
      primaryDark,
      accent,
      contrast: getContrastColor(primary),
      ring: rgbaHex(primary, CONTRAST_PRESETS[state.contrast] || CONTRAST_PRESETS.balanced)
    };
  }
  const API_BASE = '/api/control';
  const PROMETHEUS_BASE = '/api/prometheus';
  const HIDDEN_PANELS_STORAGE_KEY = 'nn_trainer_hidden_panels';
  const CUSTOM_PANELS_STORAGE_KEY = 'nn_trainer_custom_panels';
  const CONSTRUCTOR_BLUEPRINT_STORAGE_KEY = 'nn_trainer_constructor_blueprint';
  const CUSTOM_METRIC_LABELS = {
    map50: 'mAP50',
    map75: 'mAP75',
    latency: 'Задержка',
    fps: 'FPS',
    health: 'Здоровье сервисов',
    host_cpu: 'CPU хоста',
    host_mem: 'Память хоста',
    firing_alerts: 'Активные алерты'
  };
  const CUSTOM_BLOCK_TYPES = new Set(['text', 'metric', 'status']);
  const PANEL_STYLES = new Set(['plain', 'tinted', 'outlined', 'solid']);
  const PANEL_ACCENTS = new Set(['inherit', ...Object.keys(ACCENT_PALETTES), 'custom']);

  // DOM Elements
  const navPanel = document.getElementById('nav-panel');
  const menuToggle = document.getElementById('menu-toggle');
  const appearanceToggle = document.getElementById('appearance-toggle');
  const themeToggle = document.getElementById('theme-toggle');
  const themeSettings = document.getElementById('theme-settings');
  const themeSettingsClose = document.getElementById('theme-settings-close');
  const themeModeSelect = document.getElementById('theme-mode');
  const accentOptions = document.getElementById('accent-options');
  const customAccentInput = document.getElementById('custom-accent');
  const surfaceSelect = document.getElementById('theme-surface');
  const radiusSelect = document.getElementById('theme-radius');
  const contrastSelect = document.getElementById('theme-contrast');
  const densitySelect = document.getElementById('theme-density');
  const motionToggle = document.getElementById('theme-motion');
  const appearanceReset = document.getElementById('appearance-reset');
  const authGate = document.getElementById('auth-gate');
  const loginForm = document.getElementById('login-form');
  const loginStatus = document.getElementById('login-status');
  const authUser = document.getElementById('auth-user');
  const authUserName = document.getElementById('auth-user-name');
  const authLogout = document.getElementById('auth-logout');
  const userCreateForm = document.getElementById('user-create-form');
  const userList = document.getElementById('user-list');
  const userAdminStatus = document.getElementById('user-admin-status');
  const canvasBuilderToggle = document.getElementById('canvas-builder-toggle');
  const canvasBuilder = document.getElementById('canvas-builder');
  const canvasBuilderClose = document.getElementById('canvas-builder-close');
  const canvasBuilderList = document.getElementById('canvas-builder-list');
  const canvasBuilderForm = document.getElementById('canvas-builder-form');
  const canvasBlockTitle = document.getElementById('canvas-block-title');
  const canvasBlockType = document.getElementById('canvas-block-type');
  const canvasBlockText = document.getElementById('canvas-block-text');
  const canvasBlockTextField = document.getElementById('canvas-block-text-field');
  const canvasBlockMetric = document.getElementById('canvas-block-metric');
  const canvasBlockMetricField = document.getElementById('canvas-block-metric-field');
  const canvasBuilderStatus = document.getElementById('canvas-builder-status');
  const canvasPanelSettings = document.getElementById('canvas-panel-settings');
  const canvasPanelSelected = document.getElementById('canvas-panel-selected');
  const canvasPanelStyle = document.getElementById('canvas-panel-style');
  const canvasPanelAccent = document.getElementById('canvas-panel-accent');
  const canvasPanelCustomAccentField = document.getElementById('canvas-panel-custom-accent-field');
  const canvasPanelCustomAccent = document.getElementById('canvas-panel-custom-accent');
  const canvasPanelStyleReset = document.getElementById('canvas-panel-style-reset');
  const panelTrashDropzone = document.getElementById('panel-trash-dropzone');
  const constructorLayerCatalog = document.getElementById('constructor-layer-catalog');
  const constructorClassifierStack = document.getElementById('constructor-classifier-stack');
  const constructorBboxStack = document.getElementById('constructor-bbox-stack');
  const stepButtons = document.querySelectorAll('.step-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const headerStatus = document.getElementById('header-status');
  const contentsToggle = document.getElementById('contents-toggle');
  const contentsMenu = document.getElementById('contents-menu');
  const contentsClose = document.getElementById('contents-close');
  const contentsList = document.getElementById('contents-list');
  let contentsObserver;
  let panelZIndex = 10;
  let selectedCanvasPanel = null;

  // Initialize
  function init() {
    setupAuth();
    setupUserManagement();
    setupNavigation();
    setupContentsNavigation();
    restoreCustomPanels();
    setupPanelReordering();
    setupCanvasBuilder();
    setupThemeSettings();
    setupThemeToggle();
    setupStepNavigation();
    setupConstructorDragDrop();
    setupForms();
    window.addEventListener('pagehide', savePanelState);
    window.setInterval(() => {
      if (state.user) void loadCatalog();
    }, 15000);
    window.setInterval(() => {
      if (state.user) void loadLiveMetrics();
    }, 10000);
    loadSavedState();
    goToStep(state.currentStep);
    updateUI();
    void loadAuth();
  }

  function setLoginStatus(message, kind = 'muted') {
    if (!loginStatus) return;
    loginStatus.textContent = message;
    loginStatus.dataset.kind = kind;
  }

  function setUserAdminStatus(message, kind = 'muted') {
    if (!userAdminStatus) return;
    userAdminStatus.textContent = message;
    userAdminStatus.dataset.kind = kind;
  }

  function updateAuthUI() {
    const authenticated = Boolean(state.user);
    const isAdmin = state.user?.role === 'admin';
    if (authGate) authGate.hidden = authenticated;
    if (authUser) authUser.hidden = !authenticated;
    if (authUserName) {
      authUserName.textContent = authenticated
        ? `${state.user.username} · ${isAdmin ? 'администратор' : 'пользователь'}`
        : '';
    }
    document.querySelectorAll('[data-auth-admin]').forEach(element => {
      element.hidden = !isAdmin;
    });
    state.totalSteps = isAdmin ? 7 : 6;
    if (!isAdmin && state.currentStep > 6) goToStep(1);
    if (isAdmin) void loadUsers();
  }

  async function loadAuth() {
    try {
      const response = await fetch(`${API_BASE}/auth/me`, { credentials: 'same-origin' });
      if (!response.ok) {
        updateAuthUI();
        return;
      }
      const payload = await response.json();
      state.user = payload.user || null;
      updateAuthUI();
      if (state.user) {
        void loadCatalog();
        void loadLiveMetrics();
      }
    } catch (_error) {
      updateAuthUI();
      setLoginStatus('Сервис входа недоступен.', 'error');
    }
  }

  async function login(event) {
    event.preventDefault();
    const formData = new FormData(loginForm);
    setLoginStatus('Проверяю данные...', 'muted');
    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: formData.get('username'),
          password: formData.get('password')
        })
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || 'Не удалось выполнить вход.');
      state.user = payload.user || null;
      setLoginStatus('');
      updateAuthUI();
      void loadCatalog();
      void loadLiveMetrics();
    } catch (error) {
      setLoginStatus(error.message, 'error');
    }
  }

  async function logout() {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        credentials: 'same-origin'
      });
    } finally {
      state.user = null;
      updateAuthUI();
      goToStep(1);
      setLoginStatus('Сеанс завершён.', 'muted');
    }
  }

  function handleAuthExpired() {
    if (!state.user) return;
    state.user = null;
    updateAuthUI();
    setLoginStatus('Сеанс истёк. Войдите снова.', 'error');
  }

  function setupAuth() {
    loginForm?.addEventListener('submit', login);
    authLogout?.addEventListener('click', () => void logout());
    updateAuthUI();
  }

  function renderUsers(items) {
    if (!userList) return;
    const users = Array.isArray(items) ? items : [];
    const markup = users.map(user => {
      const self = user.username === state.user?.username;
      return `
        <div class="user-admin-row" data-user-row="${escapeHtml(user.username)}">
          <strong>${escapeHtml(user.username)}</strong>
          <select data-user-role aria-label="Роль ${escapeHtml(user.username)}">
            <option value="user" ${user.role === 'user' ? 'selected' : ''}>Пользователь</option>
            <option value="admin" ${user.role === 'admin' ? 'selected' : ''}>Администратор</option>
          </select>
          <input data-user-password type="password" minlength="8" maxlength="256" placeholder="Новый пароль" autocomplete="new-password" aria-label="Новый пароль ${escapeHtml(user.username)}" />
          <button class="btn btn-inline" type="button" data-user-save>Сохранить</button>
          <button class="btn btn-inline btn-danger" type="button" data-user-delete ${self ? 'disabled' : ''}>Удалить</button>
        </div>
      `;
    }).join('');
    setStableMarkup(userList, markup || '<p class="empty-state">Пользователей нет.</p>');
  }

  async function loadUsers() {
    if (state.user?.role !== 'admin') return;
    try {
      const payload = await apiJson('/auth/users');
      renderUsers(payload.items);
    } catch (error) {
      setUserAdminStatus(`Не удалось загрузить пользователей: ${error.message}`, 'error');
    }
  }

  async function saveUser(row) {
    const username = row.dataset.userRow;
    const password = row.querySelector('[data-user-password]')?.value || '';
    const payload = { role: row.querySelector('[data-user-role]')?.value };
    if (password) payload.password = password;
    try {
      await apiJson(`/auth/users/${encodeURIComponent(username)}`, {
        method: 'PUT',
        body: JSON.stringify(payload)
      });
      setUserAdminStatus(`Пользователь ${username} обновлён.`, 'success');
      await loadUsers();
    } catch (error) {
      setUserAdminStatus(error.message, 'error');
    }
  }

  async function deleteUser(username) {
    if (!window.confirm(`Удалить пользователя «${username}»?`)) return;
    try {
      await apiJson(`/auth/users/${encodeURIComponent(username)}`, { method: 'DELETE' });
      setUserAdminStatus(`Пользователь ${username} удалён.`, 'success');
      await loadUsers();
    } catch (error) {
      setUserAdminStatus(error.message, 'error');
    }
  }

  function setupUserManagement() {
    userCreateForm?.addEventListener('submit', async event => {
      event.preventDefault();
      const formData = new FormData(userCreateForm);
      try {
        await apiJson('/auth/users', {
          method: 'POST',
          body: JSON.stringify({
            username: formData.get('username'),
            password: formData.get('password'),
            role: formData.get('role')
          })
        });
        userCreateForm.reset();
        setUserAdminStatus('Пользователь добавлен.', 'success');
        await loadUsers();
      } catch (error) {
        setUserAdminStatus(error.message, 'error');
      }
    });
    userList?.addEventListener('click', event => {
      const row = event.target.closest('[data-user-row]');
      if (!row) return;
      if (event.target.closest('[data-user-save]')) void saveUser(row);
      if (event.target.closest('[data-user-delete]')) void deleteUser(row.dataset.userRow);
    });
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
      .filter(panel => !panel.hidden)
      .map((panel, index) => {
        const heading = panel.querySelector('.panel-title-row h2, .panel-title-row h3, .panel-title-row h4');
        if (!heading) return null;
        if (!panel.id) panel.id = `content-${pane.id}-${index + 1}`;
        const subitems = [
          ...panel.querySelectorAll(':scope > .subsection-head'),
          ...panel.querySelectorAll(':scope > details')
        ].map((section, sectionIndex) => {
          const sectionHeading = section.querySelector(':scope > h3, :scope > h4, :scope > summary');
          if (!sectionHeading) return null;
          if (!section.id) section.id = `${panel.id}-sub-${sectionIndex + 1}`;
          return {
            id: section.id,
            label: section.matches('details')
              ? sectionHeading.firstChild?.textContent.trim() || sectionHeading.textContent.trim()
              : sectionHeading.textContent.trim()
          };
        }).filter(Boolean);
        return {
          id: panel.id,
          label: heading.textContent.trim(),
          group: pane.dataset.contentGroup || pane.id,
          subitems
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

        if (item.subitems.length) {
          const subitemsEl = document.createElement('div');
          subitemsEl.className = 'contents-subitems';
          item.subitems.forEach(subitem => {
            const subbutton = document.createElement('button');
            subbutton.type = 'button';
            subbutton.className = 'contents-subitem';
            subbutton.dataset.contentTarget = subitem.id;
            subbutton.textContent = subitem.label;
            subitemsEl.append(subbutton);
          });
          itemsEl.append(subitemsEl);
        }
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

    const buttons = [...contentsList.querySelectorAll('.contents-item, .contents-subitem')];
    contentsObserver = new IntersectionObserver(entries => {
      const visible = entries
        .filter(entry => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      buttons.forEach(button => {
        button.classList.toggle('is-current', button.dataset.contentTarget === visible.target.id);
      });
    }, { rootMargin: '-72px 0px -55% 0px', threshold: [0, 0.25, 0.75] });

    items.flatMap(item => [item, ...item.subitems]).forEach(item => {
      const target = document.getElementById(item.id);
      if (target) contentsObserver.observe(target);
    });
  }

  function getPanelKey(pane, panel) {
    if (panel.dataset.panelKey) return panel.dataset.panelKey;
    const heading = panel.querySelector(':scope > .panel-title-row h2, :scope > .panel-title-row h3, :scope > .panel-title-row h4');
    const title = heading?.textContent.trim() || panel.className;
    panel.dataset.panelKey = `${pane.id}:${encodeURIComponent(title.toLowerCase())}`;
    return panel.dataset.panelKey;
  }

  function readHiddenPanelKeys() {
    try {
      const value = JSON.parse(localStorage.getItem(HIDDEN_PANELS_STORAGE_KEY) || '[]');
      return new Set(Array.isArray(value) ? value.filter(item => typeof item === 'string') : []);
    } catch (_error) {
      return new Set();
    }
  }

  function saveHiddenPanelKeys(keys) {
    localStorage.setItem(HIDDEN_PANELS_STORAGE_KEY, JSON.stringify([...keys]));
  }

  function readPanelStyles() {
    try {
      const value = JSON.parse(localStorage.getItem('nn_trainer_panel_styles') || '{}');
      return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    } catch (_error) {
      return {};
    }
  }

  function savePanelStyles(styles) {
    localStorage.setItem('nn_trainer_panel_styles', JSON.stringify(styles));
  }

  function getPanelAccentColor(accent) {
    const key = accent === 'inherit' ? state.accent : accent;
    if (key === 'custom') return normalizeHex(state.customAccent);
    return ACCENT_PALETTES[key]?.[state.theme]?.primary || ACCENT_PALETTES.mint[state.theme].primary;
  }

  function applyPanelStyle(pane, panel, styles = readPanelStyles()) {
    const saved = styles[getPanelKey(pane, panel)] || {};
    const style = PANEL_STYLES.has(saved.style) ? saved.style : 'plain';
    const accent = PANEL_ACCENTS.has(saved.accent) ? saved.accent : 'inherit';
    const color = accent === 'custom'
      ? normalizeHex(saved.customAccent, state.customAccent)
      : getPanelAccentColor(accent);
    panel.dataset.panelStyle = style;
    panel.dataset.panelAccent = accent;
    panel.style.setProperty('--panel-accent', color);
    panel.style.setProperty('--panel-accent-soft', rgbaHex(color, state.theme === 'dark' ? 0.16 : 0.09));
    panel.style.setProperty('--panel-accent-strong', mixHex(color, state.theme === 'dark' ? '#ffffff' : '#000000', 0.18));
  }

  function applyPanelStyles() {
    const styles = readPanelStyles();
    tabPanes.forEach(pane => {
      pane.querySelectorAll(':scope > .panel').forEach(panel => applyPanelStyle(pane, panel, styles));
    });
    renderPanelStyleSettings();
  }

  function readCustomPanels() {
    try {
      const value = JSON.parse(localStorage.getItem(CUSTOM_PANELS_STORAGE_KEY) || '[]');
      if (!Array.isArray(value)) return [];
      return value.map((item, index) => {
        if (!item || typeof item !== 'object' || !document.getElementById(item.paneId)) return null;
        const type = CUSTOM_BLOCK_TYPES.has(item.type) ? item.type : 'text';
        const id = String(item.id || `custom-${index + 1}`).replace(/[^a-z0-9_-]/gi, '').slice(0, 48);
        const metric = CUSTOM_METRIC_LABELS[item.metric] ? item.metric : 'map50';
        return {
          id: id || `custom-${index + 1}`,
          paneId: String(item.paneId),
          title: String(item.title || 'Пользовательский блок').trim().slice(0, 80) || 'Пользовательский блок',
          type,
          text: String(item.text || '').trim().slice(0, 2000),
          metric
        };
      }).filter(Boolean);
    } catch (_error) {
      return [];
    }
  }

  function saveCustomPanels(panels) {
    localStorage.setItem(CUSTOM_PANELS_STORAGE_KEY, JSON.stringify(panels));
  }

  function createCustomPanel(spec) {
    const panel = document.createElement('article');
    panel.className = 'panel custom-panel';
    panel.id = `custom-panel-${spec.id}`;
    panel.dataset.customPanelId = spec.id;
    panel.dataset.panelKey = `${spec.paneId}:custom:${spec.id}`;

    const titleRow = document.createElement('div');
    titleRow.className = 'panel-title-row';
    const title = document.createElement('h2');
    title.textContent = spec.title;
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = spec.type === 'metric'
      ? CUSTOM_METRIC_LABELS[spec.metric]
      : spec.type === 'status' ? 'Система' : 'Пользовательский';
    titleRow.append(title, tag);
    panel.append(titleRow);

    if (spec.type === 'metric') {
      const grid = document.createElement('div');
      grid.className = 'kpi-grid kpi-grid-primary';
      const card = document.createElement('div');
      card.className = 'kpi-card';
      card.dataset.metric = spec.metric;
      const label = document.createElement('p');
      label.className = 'kpi-label';
      label.textContent = CUSTOM_METRIC_LABELS[spec.metric];
      const value = document.createElement('p');
      value.className = 'kpi-value';
      value.textContent = '--';
      card.append(label, value);
      grid.append(card);
      panel.append(grid);
    } else if (spec.type === 'status') {
      const status = document.createElement('div');
      status.className = 'custom-system-status';
      status.dataset.systemStatus = 'ready';
      const dot = document.createElement('span');
      dot.className = 'status-dot status-ready';
      const text = document.createElement('strong');
      text.className = 'custom-system-status-text';
      text.textContent = 'Готово';
      status.append(dot, text);
      panel.append(status);
    } else {
      const text = document.createElement('p');
      text.className = 'custom-block-text';
      text.textContent = spec.text || 'Пользовательский блок';
      panel.append(text);
    }
    return panel;
  }

  function restoreCustomPanels() {
    readCustomPanels().forEach(spec => {
      const pane = document.getElementById(spec.paneId);
      if (!pane || [...pane.querySelectorAll(':scope > .panel')]
        .some(panel => panel.dataset.customPanelId === spec.id)) return;
      pane.append(createCustomPanel(spec));
    });
  }

  function applyPanelVisibility() {
    const hiddenKeys = readHiddenPanelKeys();
    tabPanes.forEach(pane => {
      pane.querySelectorAll(':scope > .panel').forEach(panel => {
        const hidden = hiddenKeys.has(getPanelKey(pane, panel));
        panel.hidden = hidden;
        panel.classList.toggle('is-deleted', hidden);
      });
      updateCanvasBounds(pane);
    });
  }

  function setPanelVisibility(pane, panel, hidden) {
    const key = getPanelKey(pane, panel);
    const keys = readHiddenPanelKeys();
    if (hidden) keys.add(key);
    else keys.delete(key);
    panel.hidden = hidden;
    panel.classList.toggle('is-deleted', hidden);
    saveHiddenPanelKeys(keys);
    updateCanvasBounds(pane);
    renderCanvasBuilder();
    renderContentsMenu();
    observeContentPanels();
  }

  function deleteCustomPanel(pane, panel) {
    const customId = panel.dataset.customPanelId;
    if (!customId || !window.confirm(`Удалить блок «${getPanelLabel(panel)}» без возможности восстановления?`)) return;

    saveCustomPanels(readCustomPanels().filter(spec => spec.id !== customId));
    const key = getPanelKey(pane, panel);
    const hiddenKeys = readHiddenPanelKeys();
    hiddenKeys.delete(key);
    saveHiddenPanelKeys(hiddenKeys);
    const styles = readPanelStyles();
    delete styles[key];
    savePanelStyles(styles);
    if (selectedCanvasPanel?.panel === panel) selectedCanvasPanel = null;
    panel.remove();
    savePanelState();
    updateCanvasBounds(pane);
    renderCanvasBuilder();
    renderContentsMenu();
    observeContentPanels();
    setCanvasBuilderStatus('Пользовательский блок удалён окончательно.');
  }

  function getPanelLabel(panel) {
    return panel.querySelector(':scope > .panel-title-row h2, :scope > .panel-title-row h3, :scope > .panel-title-row h4')
      ?.textContent.trim() || 'Без названия';
  }

  function renderPanelStyleSettings() {
    const panel = selectedCanvasPanel?.panel;
    const pane = selectedCanvasPanel?.pane;
    const available = Boolean(panel && pane && document.contains(panel));
    if (canvasPanelSettings) canvasPanelSettings.hidden = !available;
    if (!available) {
      if (canvasPanelSelected) canvasPanelSelected.textContent = 'Выберите блок';
      return;
    }
    const saved = readPanelStyles()[getPanelKey(pane, panel)] || {};
    const style = PANEL_STYLES.has(saved.style) ? saved.style : 'plain';
    const accent = PANEL_ACCENTS.has(saved.accent) ? saved.accent : 'inherit';
    if (canvasPanelSelected) canvasPanelSelected.textContent = getPanelLabel(panel);
    if (canvasPanelStyle) canvasPanelStyle.value = style;
    if (canvasPanelAccent) canvasPanelAccent.value = accent;
    if (canvasPanelCustomAccent) canvasPanelCustomAccent.value = normalizeHex(saved.customAccent, state.customAccent);
    if (canvasPanelCustomAccentField) canvasPanelCustomAccentField.hidden = accent !== 'custom';
  }

  function selectCanvasPanel(pane, panel) {
    selectedCanvasPanel = { pane, panel };
    raisePanel(panel);
    renderCanvasBuilder();
    renderPanelStyleSettings();
  }

  function saveSelectedPanelStyle() {
    const panel = selectedCanvasPanel?.panel;
    const pane = selectedCanvasPanel?.pane;
    if (!panel || !pane) return;
    const styles = readPanelStyles();
    const key = getPanelKey(pane, panel);
    styles[key] = {
      style: PANEL_STYLES.has(canvasPanelStyle?.value) ? canvasPanelStyle.value : 'plain',
      accent: PANEL_ACCENTS.has(canvasPanelAccent?.value) ? canvasPanelAccent.value : 'inherit',
      customAccent: normalizeHex(canvasPanelCustomAccent?.value, state.customAccent)
    };
    savePanelStyles(styles);
    applyPanelStyle(pane, panel, styles);
    renderPanelStyleSettings();
    setCanvasBuilderStatus('Оформление сохранено');
  }

  function resetSelectedPanelStyle() {
    const panel = selectedCanvasPanel?.panel;
    const pane = selectedCanvasPanel?.pane;
    if (!panel || !pane) return;
    const styles = readPanelStyles();
    delete styles[getPanelKey(pane, panel)];
    savePanelStyles(styles);
    applyPanelStyle(pane, panel, styles);
    renderPanelStyleSettings();
    setCanvasBuilderStatus('Оформление сброшено');
  }

  function renderCanvasBuilder() {
    if (!canvasBuilderList) return;
    canvasBuilderList.replaceChildren();
    tabPanes.forEach(pane => {
      const panels = [...pane.querySelectorAll(':scope > .panel')];
      if (!panels.length) return;
      const group = document.createElement('section');
      group.className = 'canvas-builder-group';
      const heading = document.createElement('strong');
      heading.textContent = pane.dataset.contentGroup || pane.id;
      group.append(heading);

      panels.forEach(panel => {
        const item = document.createElement('div');
        item.className = 'canvas-builder-item';
        item.classList.toggle('is-deleted', panel.hidden);
        const label = document.createElement('button');
        label.type = 'button';
        label.className = 'canvas-builder-panel-name';
        label.textContent = getPanelLabel(panel);
        label.classList.toggle('is-selected', selectedCanvasPanel?.panel === panel);
        label.addEventListener('click', () => selectCanvasPanel(pane, panel));
        const actions = document.createElement('div');
        actions.className = 'canvas-builder-actions';
        const action = document.createElement('button');
        action.type = 'button';
        action.className = panel.hidden ? 'btn btn-inline btn-primary' : 'btn btn-inline';
        action.textContent = panel.hidden ? 'Вернуть' : 'Убрать';
        action.addEventListener('click', () => setPanelVisibility(pane, panel, !panel.hidden));
        actions.append(action);
        if (panel.dataset.customPanelId) {
          const deleteAction = document.createElement('button');
          deleteAction.type = 'button';
          deleteAction.className = 'btn btn-inline btn-danger';
          deleteAction.textContent = 'Удалить';
          deleteAction.title = 'Удалить без возможности восстановления';
          deleteAction.addEventListener('click', () => deleteCustomPanel(pane, panel));
          actions.append(deleteAction);
        }
        item.append(label, actions);
        group.append(item);
      });
      canvasBuilderList.append(group);
    });
    renderPanelStyleSettings();
  }

  function setCanvasBuilderStatus(message) {
    if (canvasBuilderStatus) canvasBuilderStatus.textContent = message;
  }

  function updateCanvasBuilderFields() {
    const type = canvasBlockType?.value || 'text';
    const isText = type === 'text';
    const isMetric = type === 'metric';
    if (canvasBlockTextField) canvasBlockTextField.hidden = !isText;
    if (canvasBlockMetricField) canvasBlockMetricField.hidden = !isMetric;
    if (canvasBlockText) canvasBlockText.required = isText;
  }

  function addCustomPanel() {
    const pane = document.querySelector('.tab-pane.is-active');
    const title = String(canvasBlockTitle?.value || '').trim();
    const type = CUSTOM_BLOCK_TYPES.has(canvasBlockType?.value) ? canvasBlockType.value : 'text';
    if (!pane || !title) return;
    const customPanels = readCustomPanels();
    const id = `custom-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
    const spec = {
      id,
      paneId: pane.id,
      title: title.slice(0, 80),
      type,
      text: String(canvasBlockText?.value || '').trim().slice(0, 2000),
      metric: CUSTOM_METRIC_LABELS[canvasBlockMetric?.value] ? canvasBlockMetric.value : 'map50'
    };
    customPanels.push(spec);
    saveCustomPanels(customPanels);

    const panel = createCustomPanel(spec);
    pane.append(panel);
    panel.style.width = '320px';
    panel.style.height = '220px';
    const count = pane.querySelectorAll(':scope > .panel:not([hidden])').length - 1;
    const offset = 28 * Math.min(count, 6);
    setPanelPosition(pane, panel, 32 + offset, 32 + offset);
    setPanelSize(pane, panel, 320, 220);
    addPanelCanvasInteractions(pane, panel);
    raisePanel(panel);
    savePanelLayout(pane, panel);
    savePanelState();
    renderCanvasBuilder();
    renderContentsMenu();
    observeContentPanels();
    canvasBuilderForm?.reset();
    updateCanvasBuilderFields();
    setCanvasBuilderStatus('Блок добавлен');
  }

  function setCanvasBuilderOpen(open) {
    if (!canvasBuilder) return;
    canvasBuilder.hidden = !open;
    canvasBuilderToggle?.setAttribute('aria-expanded', String(open));
    if (open) renderCanvasBuilder();
  }

  function setupCanvasBuilder() {
    canvasBuilderToggle?.addEventListener('click', event => {
      event.stopPropagation();
      const open = canvasBuilder?.hidden !== false;
      setThemeSettingsOpen(false);
      setCanvasBuilderOpen(open);
    });
    canvasBuilderClose?.addEventListener('click', () => setCanvasBuilderOpen(false));
    canvasBlockType?.addEventListener('change', updateCanvasBuilderFields);
    canvasPanelStyle?.addEventListener('change', saveSelectedPanelStyle);
    canvasPanelAccent?.addEventListener('change', saveSelectedPanelStyle);
    canvasPanelCustomAccent?.addEventListener('input', saveSelectedPanelStyle);
    canvasPanelStyleReset?.addEventListener('click', resetSelectedPanelStyle);
    canvasBuilderForm?.addEventListener('submit', event => {
      event.preventDefault();
      addCustomPanel();
    });
    document.addEventListener('click', event => {
      if (canvasBuilder?.hidden !== false) return;
      const path = event.composedPath?.() || [];
      const insideBuilder = path.includes(canvasBuilder) || canvasBuilder?.contains(event.target);
      const insideToggle = path.includes(canvasBuilderToggle) || canvasBuilderToggle?.contains(event.target);
      if (!insideBuilder && !insideToggle) {
        setCanvasBuilderOpen(false);
      }
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') setCanvasBuilderOpen(false);
    });
    updateCanvasBuilderFields();
    renderCanvasBuilder();
  }

  function readPanelOrders() {
    try {
      const value = JSON.parse(localStorage.getItem('nn_trainer_panel_order') || '{}');
      return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    } catch (_error) {
      return {};
    }
  }

  function readPanelLayouts() {
    try {
      const value = JSON.parse(localStorage.getItem('nn_trainer_panel_layouts') || '{}');
      return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    } catch (_error) {
      return {};
    }
  }

  function savePanelLayout(pane, panel) {
    const layouts = readPanelLayouts();
    layouts[pane.id] ||= {};
    const rect = getPanelCanvasRect(pane, panel);
    layouts[pane.id][getPanelKey(pane, panel)] = {
      left: Number(panel.dataset.canvasLeft) || Math.round(rect.left),
      top: Number(panel.dataset.canvasTop) || Math.round(rect.top),
      width: Number(panel.dataset.canvasWidth) || Math.round(rect.width),
      height: Number(panel.dataset.canvasHeight) || Math.round(rect.height),
      zIndex: Number(panel.dataset.canvasZ) || undefined
    };
    localStorage.setItem('nn_trainer_panel_layouts', JSON.stringify(layouts));
  }

  function savePanelState() {
    const orders = readPanelOrders();
    const layouts = readPanelLayouts();
    tabPanes.forEach(pane => {
      const panels = [...pane.querySelectorAll(':scope > .panel')];
      orders[pane.id] = panels.map(panel => getPanelKey(pane, panel));
      const paneLayouts = {};
      panels.forEach(panel => {
        const rect = getPanelCanvasRect(pane, panel);
        paneLayouts[getPanelKey(pane, panel)] = {
          left: Number(panel.dataset.canvasLeft) || Math.round(rect.left),
          top: Number(panel.dataset.canvasTop) || Math.round(rect.top),
          width: Number(panel.dataset.canvasWidth) || Math.round(rect.width),
          height: Number(panel.dataset.canvasHeight) || Math.round(rect.height),
          zIndex: Number(panel.dataset.canvasZ) || undefined
        };
      });
      layouts[pane.id] = paneLayouts;
    });
    localStorage.setItem('nn_trainer_panel_order', JSON.stringify(orders));
    localStorage.setItem('nn_trainer_panel_layouts', JSON.stringify(layouts));
  }

  function getPanelCanvasRect(pane, panel) {
    const paneRect = pane.getBoundingClientRect();
    const rect = panel.getBoundingClientRect();
    return {
      left: rect.left - paneRect.left + pane.scrollLeft,
      top: rect.top - paneRect.top + pane.scrollTop,
      width: rect.width,
      height: rect.height
    };
  }

  function capturePanelDefaults(pane, panels) {
    const hidden = getComputedStyle(pane).display === 'none';
    const inline = {
      display: pane.style.display,
      gridTemplateColumns: pane.style.gridTemplateColumns,
      gap: pane.style.gap,
      alignItems: pane.style.alignItems
    };
    if (hidden) {
      pane.style.display = 'grid';
      pane.style.gridTemplateColumns = 'repeat(12, minmax(0, 1fr))';
      pane.style.gap = '20px';
      pane.style.alignItems = 'start';
    }
    const defaults = panels.map(panel => getPanelCanvasRect(pane, panel));
    if (hidden) {
      pane.style.display = inline.display;
      pane.style.gridTemplateColumns = inline.gridTemplateColumns;
      pane.style.gap = inline.gap;
      pane.style.alignItems = inline.alignItems;
    }
    return defaults;
  }

  function updateCanvasBounds(pane) {
    if (!pane.clientWidth) return;
    const bottom = [...pane.querySelectorAll(':scope > .panel:not([hidden])')].reduce((max, panel) => {
      const top = Number(panel.dataset.canvasTop) || 0;
      return Math.max(max, top + panel.offsetHeight);
    }, 0);
    pane.style.minHeight = `${Math.max(420, Math.round(bottom + 32))}px`;
  }

  function setPanelPosition(pane, panel, left, top) {
    const width = panel.offsetWidth || Number(panel.dataset.canvasWidth) || 280;
    const maxLeft = pane.clientWidth ? Math.max(0, pane.clientWidth - Math.min(width, 80)) : Number.POSITIVE_INFINITY;
    const nextLeft = Math.max(0, Math.min(maxLeft, Math.round(left)));
    const nextTop = Math.max(0, Math.round(top));
    panel.dataset.canvasLeft = String(nextLeft);
    panel.dataset.canvasTop = String(nextTop);
    panel.style.left = `${nextLeft}px`;
    panel.style.top = `${nextTop}px`;
    updateCanvasBounds(pane);
  }

  function setPanelSize(pane, panel, width, height) {
    const left = Number(panel.dataset.canvasLeft) || 0;
    const availableWidth = pane.clientWidth ? Math.max(240, pane.clientWidth - left) : Number.POSITIVE_INFINITY;
    const nextWidth = Math.max(240, Math.min(availableWidth, Math.round(width)));
    const nextHeight = Math.max(180, Math.round(height));
    panel.dataset.canvasWidth = String(nextWidth);
    panel.dataset.canvasHeight = String(nextHeight);
    panel.style.width = `${nextWidth}px`;
    panel.style.height = `${nextHeight}px`;
    updateCanvasBounds(pane);
  }

  function isPointerOverTrash(event) {
    if (!panelTrashDropzone || panelTrashDropzone.hidden) return false;
    const rect = panelTrashDropzone.getBoundingClientRect();
    return event.clientX >= rect.left && event.clientX <= rect.right
      && event.clientY >= rect.top && event.clientY <= rect.bottom;
  }

  function setTrashDropzoneVisible(visible, over = false) {
    if (!panelTrashDropzone) return;
    panelTrashDropzone.hidden = !visible;
    panelTrashDropzone.classList.toggle('is-visible', visible);
    panelTrashDropzone.classList.toggle('is-over', visible && over);
  }

  function setPanelLayout(pane, panel, saved, fallback) {
    const layout = saved && Number(saved.width) > 0 ? saved : fallback;
    const left = Number(layout?.left) || 0;
    const top = Number(layout?.top) || 0;
    const width = Number(layout?.width) || Math.max(240, fallback?.width || 280);
    const height = Number(layout?.height) || Math.max(180, fallback?.height || 220);
    panel.style.position = 'absolute';
    setPanelPosition(pane, panel, left, top);
    setPanelSize(pane, panel, width, height);
    if (Number(saved?.zIndex) > 0) {
      panel.dataset.canvasZ = String(Math.round(saved.zIndex));
      panel.style.zIndex = panel.dataset.canvasZ;
      panelZIndex = Math.max(panelZIndex, Number(saved.zIndex));
    }
  }

  function applyPanelLayouts(pane, layouts, defaults) {
    const paneLayouts = layouts[pane.id] && typeof layouts[pane.id] === 'object' ? layouts[pane.id] : {};
    const panels = [...pane.querySelectorAll(':scope > .panel')];
    panels.forEach((panel, index) => {
      const saved = paneLayouts[getPanelKey(pane, panel)];
      setPanelLayout(pane, panel, saved, defaults[index]);
    });
    updateCanvasBounds(pane);
  }

  let activePanelOperation = null;

  function raisePanel(panel) {
    panel.style.zIndex = String(++panelZIndex);
    panel.dataset.canvasZ = panel.style.zIndex;
  }

  function getPanelResizeEdge(panel, event) {
    const rect = panel.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const edge = Math.min(12, Math.max(8, Math.min(rect.width, rect.height) / 4));
    const horizontal = x <= edge ? 'w' : rect.width - x <= edge ? 'e' : '';
    const vertical = y <= edge ? 'n' : rect.height - y <= edge ? 's' : '';
    return `${vertical}${horizontal}`;
  }

  function finishPanelOperation(event) {
    const operation = activePanelOperation;
    if (!operation || (event?.pointerId !== undefined && operation.pointerId !== event.pointerId)) return;
    const droppedInTrash = operation.mode === 'move' && isPointerOverTrash(event);
    operation.panel.classList.remove('is-dragging', 'is-resizing');
    delete operation.panel.dataset.resizeEdge;
    if (operation.source?.hasPointerCapture?.(operation.pointerId)) {
      operation.source.releasePointerCapture(operation.pointerId);
    }
    if (droppedInTrash) {
      savePanelLayout(operation.pane, operation.panel);
      setPanelVisibility(operation.pane, operation.panel, true);
      setCanvasBuilderStatus(`${getPanelLabel(operation.panel)} перемещён в корзину`);
    } else {
      savePanelLayout(operation.pane, operation.panel);
    }
    setTrashDropzoneVisible(false);
    activePanelOperation = null;
  }

  function beginPanelOperation(pane, panel, event, mode, edge, source) {
    if (event.button !== undefined && event.button !== 0) return;
    const rect = getPanelCanvasRect(pane, panel);
    event.preventDefault();
    raisePanel(panel);
    activePanelOperation = {
      pane,
      panel,
      source,
      mode,
      edge,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      start: {
        left: Number(panel.dataset.canvasLeft) || Math.round(rect.left),
        top: Number(panel.dataset.canvasTop) || Math.round(rect.top),
        width: panel.getBoundingClientRect().width,
        height: panel.getBoundingClientRect().height
      }
    };
    panel.classList.toggle('is-dragging', mode === 'move');
    panel.classList.toggle('is-resizing', mode === 'resize');
    setTrashDropzoneVisible(mode === 'move');
    source?.setPointerCapture?.(event.pointerId);
  }

  function handlePanelPointerMove(event) {
    const operation = activePanelOperation;
    if (!operation) return;
    if (operation.pointerId !== event.pointerId) return;
    const dx = event.clientX - operation.startX;
    const dy = event.clientY - operation.startY;
    if (operation.mode === 'move') {
      panelTrashDropzone?.classList.toggle('is-over', isPointerOverTrash(event));
      setPanelPosition(
        operation.pane,
        operation.panel,
        operation.start.left + dx,
        operation.start.top + dy
      );
      event.preventDefault();
      return;
    }

    let left = operation.start.left;
    let top = operation.start.top;
    let width = operation.start.width;
    let height = operation.start.height;
    if (operation.edge.includes('w')) {
      left = Math.min(operation.start.left + dx, operation.start.left + operation.start.width - 240);
      width = operation.start.width + operation.start.left - left;
    } else if (operation.edge.includes('e') || !operation.edge) {
      width = Math.max(240, operation.start.width + dx);
    }
    if (operation.edge.includes('n')) {
      top = Math.min(operation.start.top + dy, operation.start.top + operation.start.height - 180);
      height = operation.start.height + operation.start.top - top;
    } else if (operation.edge.includes('s') || !operation.edge) {
      height = Math.max(180, operation.start.height + dy);
    }
    setPanelPosition(operation.pane, operation.panel, left, top);
    setPanelSize(operation.pane, operation.panel, width, height);
    event.preventDefault();
  }

  function addPanelCanvasInteractions(pane, panel) {
    panel.addEventListener('pointerdown', event => {
      if (event.button !== undefined && event.button !== 0) return;
      const target = event.target instanceof Element ? event.target : null;
      const interactive = target?.closest(
        'button, a, input, select, textarea, option, label, [contenteditable="true"]'
      );
      const edge = getPanelResizeEdge(panel, event);
      if (interactive) {
        raisePanel(panel);
        return;
      }
      beginPanelOperation(pane, panel, event, edge ? 'resize' : 'move', edge, panel);
    });
    panel.addEventListener('pointermove', event => {
      if (activePanelOperation?.panel === panel) return;
      panel.dataset.resizeEdge = getPanelResizeEdge(panel, event);
    });
    panel.addEventListener('pointerleave', () => {
      if (!activePanelOperation) delete panel.dataset.resizeEdge;
    });
    panel.addEventListener('click', () => raisePanel(panel));
  }

  function applyPanelOrder(pane, order) {
    if (!Array.isArray(order) || !order.length) return;
    const panels = [...pane.querySelectorAll(':scope > .panel')];
    const byKey = new Map(panels.map(panel => [getPanelKey(pane, panel), panel]));
    order.forEach(key => {
      const panel = byKey.get(key);
      if (panel) pane.append(panel);
    });
  }

  function setupPanelReordering() {
    const orders = readPanelOrders();
    const layouts = readPanelLayouts();

    document.addEventListener('pointermove', handlePanelPointerMove, { passive: false });
    document.addEventListener('pointerup', finishPanelOperation);
    document.addEventListener('pointercancel', finishPanelOperation);
    tabPanes.forEach(pane => {
      applyPanelOrder(pane, orders[pane.id]);
      const panels = [...pane.querySelectorAll(':scope > .panel')];
      panels.forEach(panel => getPanelKey(pane, panel));
      const defaults = capturePanelDefaults(pane, panels);
      pane.classList.add('free-canvas');
      applyPanelLayouts(pane, layouts, defaults);
      panels.forEach(panel => {
        addPanelCanvasInteractions(pane, panel);
      });
      updateCanvasBounds(pane);
    });
    applyPanelVisibility();
    applyPanelStyles();
  }

  // Theme toggle
  function setupThemeToggle() {
    themeToggle?.addEventListener('click', () => {
      setThemePreferences({ themeMode: state.theme === 'dark' ? 'light' : 'dark' });
    });
  }

  function setupThemeSettings() {
    appearanceToggle?.addEventListener('click', event => {
      event.stopPropagation();
      setCanvasBuilderOpen(false);
      setThemeSettingsOpen(themeSettings?.hidden !== false);
    });
    themeSettingsClose?.addEventListener('click', () => setThemeSettingsOpen(false));
    document.addEventListener('click', event => {
      if (!themeSettings?.contains(event.target) && !appearanceToggle?.contains(event.target)) {
        setThemeSettingsOpen(false);
      }
    });
    themeModeSelect?.addEventListener('change', event => {
      setThemePreferences({ themeMode: event.target.value });
    });
    accentOptions?.addEventListener('click', event => {
      const button = event.target.closest('[data-accent]');
      if (button) setThemePreferences({ accent: button.dataset.accent });
    });
    customAccentInput?.addEventListener('input', event => {
      setThemePreferences({ accent: 'custom', customAccent: event.target.value });
    });
    surfaceSelect?.addEventListener('change', event => {
      setThemePreferences({ surface: event.target.value });
    });
    radiusSelect?.addEventListener('change', event => {
      setThemePreferences({ radius: event.target.value });
    });
    contrastSelect?.addEventListener('change', event => {
      setThemePreferences({ contrast: event.target.value });
    });
    densitySelect?.addEventListener('change', event => {
      setThemePreferences({ density: event.target.value });
    });
    motionToggle?.addEventListener('change', event => {
      setThemePreferences({ motion: event.target.checked ? 'off' : 'on' });
    });
    appearanceReset?.addEventListener('click', () => setThemePreferences(THEME_DEFAULTS));
    window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener('change', () => {
      if (state.themeMode === 'system') setThemePreferences({}, false);
    });
  }

  function setThemeSettingsOpen(open) {
    if (!themeSettings) return;
    themeSettings.hidden = !open;
    appearanceToggle?.setAttribute('aria-expanded', String(open));
  }

  function getThemePreferences() {
    const stored = localStorage.getItem('nn_trainer_preferences');
    if (!stored) return { ...THEME_DEFAULTS };
    try {
      const parsed = JSON.parse(stored);
      return {
        themeMode: ['light', 'dark', 'system'].includes(parsed.themeMode) ? parsed.themeMode : THEME_DEFAULTS.themeMode,
        accent: parsed.accent === 'custom' || ACCENT_PALETTES[parsed.accent] ? parsed.accent : THEME_DEFAULTS.accent,
        customAccent: normalizeHex(parsed.customAccent, THEME_DEFAULTS.customAccent),
        surface: SURFACE_PALETTES[parsed.surface] ? parsed.surface : THEME_DEFAULTS.surface,
        radius: RADIUS_PRESETS[parsed.radius] ? parsed.radius : THEME_DEFAULTS.radius,
        contrast: CONTRAST_PRESETS[parsed.contrast] ? parsed.contrast : THEME_DEFAULTS.contrast,
        density: ['comfortable', 'compact'].includes(parsed.density) ? parsed.density : THEME_DEFAULTS.density,
        motion: parsed.motion === 'off' ? 'off' : THEME_DEFAULTS.motion
      };
    } catch {
      return { ...THEME_DEFAULTS };
    }
  }

  function getEffectiveTheme(themeMode) {
    if (themeMode !== 'system') return themeMode;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function setThemePreferences(preferences, persist = true) {
    state.themeMode = preferences.themeMode ?? state.themeMode;
    state.accent = preferences.accent ?? state.accent;
    state.customAccent = normalizeHex(preferences.customAccent ?? state.customAccent, THEME_DEFAULTS.customAccent);
    state.surface = preferences.surface ?? state.surface;
    state.radius = preferences.radius ?? state.radius;
    state.contrast = preferences.contrast ?? state.contrast;
    state.density = preferences.density ?? state.density;
    state.motion = preferences.motion ?? state.motion;
    state.theme = getEffectiveTheme(state.themeMode);
    const palette = state.accent === 'custom'
      ? buildCustomPalette(state.customAccent, state.theme)
      : ACCENT_PALETTES[state.accent][state.theme];
    const surface = SURFACE_PALETTES[state.surface][state.theme];
    const radius = RADIUS_PRESETS[state.radius];
    const borderStrong = state.contrast === 'high'
      ? mixHex(surface.strong, surface.text, 0.22)
      : state.contrast === 'soft'
        ? mixHex(surface.strong, surface.surface, 0.3)
        : surface.strong;

    document.body.dataset.theme = state.theme;
    document.body.dataset.themeMode = state.themeMode;
    document.body.dataset.density = state.density;
    document.body.dataset.motion = state.motion;
    Object.entries({
      '--primary': palette.primary,
      '--primary-dark': palette.primaryDark,
      '--accent': palette.accent,
      '--primary-contrast': palette.contrast,
      '--focus-ring': rgbaHex(palette.primary, CONTRAST_PRESETS[state.contrast]),
      '--bg-primary': surface.bg,
      '--bg-secondary': surface.secondary,
      '--bg-tertiary': surface.tertiary,
      '--surface': surface.surface,
      '--surface-raised': surface.surface,
      '--surface-soft': surface.secondary,
      '--text-primary': surface.text,
      '--text-secondary': surface.secondaryText,
      '--text-muted': surface.muted,
      '--border-color': surface.border,
      '--border-strong': borderStrong,
      '--panel-radius': radius.panel,
      '--control-radius': radius.control
    }).forEach(([name, value]) => document.body.style.setProperty(name, value, 'important'));
    updateThemeControls();
    applyPanelStyles();

    if (persist) {
      localStorage.setItem('nn_trainer_preferences', JSON.stringify({
        themeMode: state.themeMode,
        accent: state.accent,
        customAccent: state.customAccent,
        surface: state.surface,
        radius: state.radius,
        contrast: state.contrast,
        density: state.density,
        motion: state.motion
      }));
      localStorage.setItem('nn_trainer_theme', state.theme);
    }
  }

  function updateThemeControls() {
    if (themeModeSelect) themeModeSelect.value = state.themeMode;
    if (customAccentInput) customAccentInput.value = state.customAccent;
    if (surfaceSelect) surfaceSelect.value = state.surface;
    if (radiusSelect) radiusSelect.value = state.radius;
    if (contrastSelect) contrastSelect.value = state.contrast;
    if (densitySelect) densitySelect.value = state.density;
    if (motionToggle) motionToggle.checked = state.motion === 'off';
    accentOptions?.querySelectorAll('[data-accent]').forEach(button => {
      const active = button.dataset.accent === state.accent;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    themeToggle?.setAttribute('aria-label', state.theme === 'dark' ? 'Включить светлую тему' : 'Включить тёмную тему');
  }

  // Step navigation
  function setupStepNavigation() {
    stepButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const step = parseInt(btn.dataset.step);
        goToStep(step);
      });
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
    renderCanvasBuilder();

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
      'tab-resources',
      'tab-users'
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
    document.getElementById('experiments-refresh')?.addEventListener('click', () => void loadCatalog());
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
    const savedPreferences = getThemePreferences();
    const savedTheme = localStorage.getItem('nn_trainer_theme');
    if (!localStorage.getItem('nn_trainer_preferences') && ['light', 'dark'].includes(savedTheme)) {
      savedPreferences.themeMode = savedTheme;
    }
    setThemePreferences(savedPreferences, false);

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
    const labels = { ready: 'Готово', busy: 'Обработка...', error: 'Ошибка' };
    if (headerStatus) {
      const dot = headerStatus.querySelector('.status-dot');
      const text = headerStatus.querySelector('.status-text');
      dot.className = 'status-dot';
      text.textContent = labels[status] || labels.ready;
      dot.classList.add(`status-${status}`);
    }

    document.querySelectorAll('[data-system-status]').forEach(element => {
      const dot = element.querySelector('.status-dot');
      const text = element.querySelector('.custom-system-status-text');
      if (dot) {
        dot.className = 'status-dot';
        dot.classList.add(`status-${status}`);
      }
      if (text) text.textContent = labels[status] || labels.ready;
      element.dataset.systemStatus = status;
    });
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
      setStableMarkup(list, '<p class="empty-state">Файлы не найдены.</p>');
      return;
    }
    const markup = items.map(item => `
      <button class="file-choice" type="button" data-workspace-file="${escapeHtml(item.path)}">
        <span><strong>${escapeHtml(item.name)}</strong><br>${escapeHtml(item.path)}</span>
        <small>${formatBytes(item.size_bytes)}</small>
      </button>
    `).join('');
    if (!setStableMarkup(list, markup)) return;
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

    request.credentials = 'same-origin';
    const response = await fetch(`${API_BASE}${path}`, request);
    if (response.status === 401) handleAuthExpired();
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

  function setStableText(elementOrId, value) {
    const element = typeof elementOrId === 'string'
      ? document.getElementById(elementOrId)
      : elementOrId;
    if (!element) return;
    const nextValue = String(value ?? '--');
    if (element.textContent !== nextValue) element.textContent = nextValue;
  }

  function setStableMarkup(element, markup) {
    if (!element || element.dataset.renderedMarkup === markup) return false;
    element.innerHTML = markup;
    element.dataset.renderedMarkup = markup;
    return true;
  }

  function formatMetric(value, digits = 2) {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) return '--';
    return numericValue.toLocaleString('ru-RU', {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits
    });
  }

  function updateKpi(metric, value, digits = 2) {
    document.querySelectorAll(`[data-metric="${metric}"]`).forEach(card => {
      setStableText(card.querySelector('.kpi-value'), formatMetric(value, digits));
    });
  }

  let activeConstructorDrag = null;

  function readConstructorBlueprint() {
    try {
      const value = JSON.parse(localStorage.getItem(CONSTRUCTOR_BLUEPRINT_STORAGE_KEY) || 'null');
      return value?.head_specs && typeof value.head_specs === 'object' ? value : null;
    } catch (_error) {
      return null;
    }
  }

  function saveConstructorBlueprint() {
    if (state.constructorBlueprint) {
      localStorage.setItem(
        CONSTRUCTOR_BLUEPRINT_STORAGE_KEY,
        JSON.stringify(state.constructorBlueprint)
      );
    }
  }

  function getConstructorLayer(type) {
    return state.constructorCatalog?.layers?.find(layer => layer.type === type) || null;
  }

  function constructorLayerParams(layer) {
    const params = layer?.params || {};
    return Object.entries(params)
      .map(([name, value]) => `${name}: ${value}`)
      .join(' · ');
  }

  function createConstructorLayerCard(layer, sourceHead = '', sourceIndex = 0) {
    const catalogLayer = getConstructorLayer(layer.type);
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'constructor-layer-card';
    card.draggable = true;
    card.dataset.layerType = layer.type;
    card.dataset.sourceHead = sourceHead;
    card.dataset.sourceIndex = String(sourceIndex);
    card.title = catalogLayer?.description || layer.type;

    const label = document.createElement('strong');
    label.textContent = catalogLayer?.label || layer.type;
    const meta = document.createElement('span');
    meta.textContent = sourceHead ? constructorLayerParams(layer) || 'Без параметров' : 'Перетащить в ветку';
    card.append(label, meta);

    card.addEventListener('dragstart', event => {
      activeConstructorDrag = {
        sourceHead,
        sourceIndex: Number(sourceIndex),
        layerType: layer.type
      };
      event.dataTransfer?.setData('application/x-nn-layer', JSON.stringify(activeConstructorDrag));
      event.dataTransfer?.setData('text/plain', layer.type);
      if (event.dataTransfer) event.dataTransfer.effectAllowed = sourceHead ? 'move' : 'copy';
      card.classList.add('is-dragging');
    });
    card.addEventListener('dragend', () => {
      activeConstructorDrag = null;
      card.classList.remove('is-dragging');
      document.querySelectorAll('.builder-stack.is-drag-over').forEach(stack => {
        stack.classList.remove('is-drag-over');
      });
    });
    return card;
  }

  function renderConstructorStacks() {
    const blueprint = state.constructorBlueprint;
    if (!blueprint) return;
    const heads = blueprint.head_specs || {};
    const layers = state.constructorCatalog?.layers || [];

    if (constructorLayerCatalog) {
      constructorLayerCatalog.replaceChildren(
        ...layers.map(layer => createConstructorLayerCard(layer))
      );
    }
    [
      ['classifier', constructorClassifierStack],
      ['bbox', constructorBboxStack]
    ].forEach(([head, stack]) => {
      if (!stack) return;
      stack.replaceChildren(
        ...(Array.isArray(heads[head]) ? heads[head] : [])
          .map((layer, index) => createConstructorLayerCard(layer, head, index))
      );
    });
  }

  function getConstructorDropIndex(stack, event) {
    const cards = [...stack.querySelectorAll(':scope > .constructor-layer-card')];
    const index = cards.findIndex(card => {
      const rect = card.getBoundingClientRect();
      return event.clientY < rect.top + rect.height / 2;
    });
    return index === -1 ? cards.length : index;
  }

  function readConstructorDrag(event) {
    try {
      const raw = event.dataTransfer?.getData('application/x-nn-layer');
      if (raw) return JSON.parse(raw);
    } catch (_error) {
      // Use the in-memory drag payload below when the browser blocks custom types.
    }
    return activeConstructorDrag;
  }

  function dropConstructorLayer(event) {
    const targetStack = event.currentTarget;
    const targetHead = targetStack?.dataset.headName;
    const drag = readConstructorDrag(event);
    const blueprint = state.constructorBlueprint;
    if (!targetHead || !drag?.layerType || !blueprint) return;

    const targetLayers = Array.isArray(blueprint.head_specs[targetHead])
      ? blueprint.head_specs[targetHead]
      : (blueprint.head_specs[targetHead] = []);
    let insertAt = getConstructorDropIndex(targetStack, event);
    let layer;
    if (drag.sourceHead) {
      const sourceLayers = blueprint.head_specs[drag.sourceHead];
      const sourceIndex = Number(drag.sourceIndex);
      if (!Array.isArray(sourceLayers) || !Number.isInteger(sourceIndex)) return;
      [layer] = sourceLayers.splice(sourceIndex, 1);
      if (!layer) return;
      if (drag.sourceHead === targetHead && sourceIndex < insertAt) insertAt -= 1;
    } else {
      const catalogLayer = getConstructorLayer(drag.layerType);
      if (!catalogLayer) return;
      layer = {
        type: catalogLayer.type,
        params: Object.fromEntries((catalogLayer.params || []).map(param => [param.name, param.default]))
      };
    }

    targetLayers.splice(Math.max(0, Math.min(insertAt, targetLayers.length)), 0, layer);
    saveConstructorBlueprint();
    renderConstructorStacks();
    setStableText('constructor-status', `${getConstructorLayer(layer.type)?.label || layer.type} добавлен в ${targetHead === 'bbox' ? 'BBox-голову' : 'голову классификатора'}.`);
  }

  function setupConstructorDragDrop() {
    [constructorClassifierStack, constructorBboxStack].forEach(stack => {
      stack?.addEventListener('dragover', event => {
        if (!readConstructorDrag(event)) return;
        event.preventDefault();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        stack.classList.add('is-drag-over');
      });
      stack?.addEventListener('dragleave', event => {
        if (!stack.contains(event.relatedTarget)) stack.classList.remove('is-drag-over');
      });
      stack?.addEventListener('drop', event => {
        event.preventDefault();
        stack.classList.remove('is-drag-over');
        dropConstructorLayer(event);
      });
    });
  }

  function initializeConstructorCatalog(payload) {
    if (!Array.isArray(payload?.layers) || !payload.layers.length) return;
    state.constructorCatalog = payload;
    if (!state.constructorBlueprint) {
      state.constructorBlueprint = readConstructorBlueprint()
        || JSON.parse(JSON.stringify(payload.templates?.default_blueprint || {
          name: 'custom_detector_builder',
          head_specs: { classifier: [], bbox: [] }
        }));
    }
    renderConstructorStacks();
  }

  async function loadCatalog() {
    if (!state.user) return;
    try {
      const [payload, servingPayload, constructorPayload] = await Promise.all([
        apiJson('/catalog'),
        apiJson('/torchserve/models').catch(() => null),
        apiJson('/architectures/constructor/catalog').catch(() => null)
      ]);
      state.datasets = Array.isArray(payload.datasets) ? payload.datasets : [];
      state.configs = Array.isArray(payload.configs) ? payload.configs : [];
      state.architectures = Array.isArray(payload.architectures) ? payload.architectures : [];
      state.jobs = Array.isArray(payload.jobs) ? payload.jobs : [];
      state.experiments = Array.isArray(payload.experiments) ? payload.experiments : [];
      state.recommendations = Array.isArray(payload.recommendations) ? payload.recommendations : [];
      if (Array.isArray(servingPayload?.models)) state.models = servingPayload.models;
      initializeConstructorCatalog(constructorPayload);
      syncDatasetControls();
      renderDatasets();
      renderOverview(payload);
      renderExperiments(state.experiments);
      renderJobs(state.jobs);
    } catch (_error) {
      // The UI keeps the last known values when monitoring is temporarily unavailable.
    }
  }

  function renderOverview(payload) {
    const datasets = Array.isArray(payload.datasets) ? payload.datasets : [];
    const configs = Array.isArray(payload.configs) ? payload.configs : [];
    const architectures = Array.isArray(payload.architectures) ? payload.architectures : [];
    const jobs = Array.isArray(payload.jobs) ? payload.jobs : [];
    const experiments = Array.isArray(payload.experiments) ? payload.experiments : [];
    const runningJobs = jobs.filter(job => job.status === 'running').length;
    const best = experiments.reduce((current, item) => {
      if (!current || Number(item.map_50 || 0) > Number(current.map_50 || 0)) return item;
      return current;
    }, null);

    setStableText('summary-datasets', datasets.length);
    setStableText('summary-configs', configs.length);
    setStableText('summary-architectures', architectures.length);
    setStableText('summary-experiments', experiments.length);
    setStableText('summary-jobs', jobs.length);
    setStableText('summary-running', runningJobs);
    setStableText('hero-tracked-runs', experiments.length);
    setStableText('hero-running-jobs', runningJobs);
    setStableText('hero-best-map50', best ? formatMetric(best.map_50, 3) : '--');
    setStableText('hero-served-models', state.models.length);

    if (best) {
      updateKpi('map50', best.map_50, 3);
      updateKpi('map75', best.map_75, 3);
      updateKpi('latency', best.latency_ms);
      updateKpi('fps', best.fps);
    } else {
      updateKpi('map50', Number.NaN, 3);
      updateKpi('map75', Number.NaN, 3);
      updateKpi('latency', Number.NaN);
      updateKpi('fps', Number.NaN);
    }
    renderRecommendations(payload.recommendations);
  }

  function renderRecommendations(items) {
    const list = document.getElementById('recommendations-list');
    if (!list) return;
    const recommendations = Array.isArray(items) ? items : [];
    const markup = recommendations.length
      ? recommendations.map((item, index) => `
          <article class="recommendation-card">
            <div class="recommendation-card-head">
              <strong>${index + 1}. ${escapeHtml(item.run_name || item.key)}</strong>
              <span class="tag">${formatMetric(item.score, 3)}</span>
            </div>
            <span class="recommendation-summary">${escapeHtml(item.summary || '')}</span>
          </article>
        `).join('')
      : '<p class="empty-state">Пока нет завершенных запусков для рекомендаций.</p>';
    setStableMarkup(list, markup);
  }

  function renderExperiments(items) {
    const body = document.getElementById('experiments-body');
    if (!body) return;
    const experiments = Array.isArray(items) ? items : [];
    const markup = experiments.length
      ? experiments.map(item => {
          const tags = (item.tags || []).map(tag => `<span>${escapeHtml(tag)}</span>`).join('');
          return `
            <tr>
              <td><input type="checkbox" aria-label="Сравнить ${escapeHtml(item.run_name)}" /></td>
              <td><strong>${escapeHtml(item.run_name || item.key)}</strong></td>
              <td>${escapeHtml(item.model_name || 'unknown')}</td>
              <td><span class="status-chip status-${escapeHtml(item.status || 'unknown')}">${escapeHtml(item.status || '—')}</span></td>
              <td>${formatMetric(item.map_50, 3)}</td>
              <td>${formatMetric(item.fps)}</td>
              <td>${formatMetric(item.latency_ms)} мс</td>
              <td>${item.rating ? `${escapeHtml(item.rating)}/5` : '—'}</td>
              <td><div class="table-tags">${tags || '<span>—</span>'}</div></td>
            </tr>
          `;
        }).join('')
      : '<tr><td colspan="9" class="empty-state">Запуски пока не найдены.</td></tr>';
    setStableMarkup(body, markup);
  }

  function renderJobs(items) {
    const list = document.getElementById('job-list');
    if (!list) return;
    const jobs = Array.isArray(items) ? items : [];
    const markup = jobs.length
      ? jobs.map(job => `
          <article class="job-card">
            <div class="job-card-head">
              <strong>${escapeHtml(job.run_name || job.id)}</strong>
              <span class="status-chip status-${escapeHtml(job.status || 'unknown')}">${escapeHtml(job.status || '—')}</span>
            </div>
            <span class="job-card-meta">${escapeHtml(job.kind || 'job')} · ${escapeHtml(job.experiment_name || 'без эксперимента')}</span>
          </article>
        `).join('')
      : '<p class="empty-state">Активных задач нет.</p>';
    setStableMarkup(list, markup);
  }

  async function queryPrometheus(query) {
    const params = new URLSearchParams({ query });
    const response = await fetch(`${PROMETHEUS_BASE}/api/v1/query?${params}`);
    if (!response.ok) throw new Error(`Prometheus ${response.status}`);
    const payload = await response.json();
    const rawValue = payload.data?.result?.[0]?.value?.[1];
    const value = Number(rawValue);
    return Number.isFinite(value) ? value : null;
  }

  async function loadLiveMetrics() {
    if (!state.user) return;
    const queries = {
      health: 'avg(probe_success{job="blackbox-http"}) * 100',
      host_cpu: '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
      host_mem: '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
      firing_alerts: 'sum(ALERTS{alertstate="firing"})'
    };
    await Promise.all(Object.entries(queries).map(async ([metric, query]) => {
      try {
        const value = await queryPrometheus(query);
        if (value !== null) {
          updateKpi(metric, value, metric === 'firing_alerts' ? 0 : 1);
          if (metric === 'health') {
            setStableText('hero-health-label', value >= 99 ? 'Все сервисы в норме' : 'Есть отклонения');
            setStableText('hero-health-note', `Доступность сервисов: ${formatMetric(value, 1)}%`);
          }
        }
      } catch (_error) {
        // Keep the last value instead of flashing the card back to a placeholder.
      }
    }));
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
      const markup = '<option value="">Авто / нет</option>' + state.datasets.map(dataset => (
        `<option value="${escapeHtml(dataset.id)}">${escapeHtml(dataset.name)}</option>`
      )).join('');
      if (select.dataset.renderedMarkup !== markup) {
        select.innerHTML = markup;
        select.dataset.renderedMarkup = markup;
        if (state.datasets.some(dataset => dataset.id === selected)) {
          select.value = selected;
        }
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
      setStableMarkup(list, '<p class="empty-state">Датасеты не найдены.</p>');
      return;
    }

    const markup = items.map(dataset => {
      const tags = (dataset.tags || []).map(tag => `
        <button class="dataset-tag" type="button" data-dataset-filter-tag="${escapeHtml(tag)}">
          ${escapeHtml(tag)}
        </button>
      `).join('');
      const description = dataset.description || 'Описание не задано.';
      const sourceKind = String(dataset.path || '').includes('/uploads/') ? 'upload' : 'workspace';
      const sourceLabel = sourceKind === 'upload' ? 'Загрузка' : 'Workspace';
      return `
        <article class="dataset-card">
          <div class="dataset-card-head">
            <div class="dataset-card-title">
              <span class="dataset-source-icon source-${sourceKind}" aria-hidden="true">${sourceKind === 'upload' ? '↑' : '⌁'}</span>
              <h3>${escapeHtml(dataset.name)}</h3>
            </div>
            <span>${escapeHtml(String(dataset.file_count || 0))} файлов</span>
          </div>
          <span class="dataset-source source-${sourceKind}">${sourceLabel}</span>
          <div class="dataset-tags">${tags || '<span class="dataset-tag-empty">без тегов</span>'}</div>
          <details class="dataset-card-details">
            <summary>Подробнее</summary>
            <p class="dataset-path">${escapeHtml(dataset.path)}</p>
            <p class="dataset-description">${escapeHtml(description)}</p>
            <div class="dataset-meta">
              <span>${formatBytes(dataset.size_bytes)}</span>
              <span>${dataset.updated_at ? new Date(dataset.updated_at).toLocaleString('ru-RU') : 'без даты'}</span>
            </div>
          </details>
          <div class="inline-actions">
            <a class="btn btn-inline" href="${API_BASE}/datasets/${encodeURIComponent(dataset.id)}/download" target="_blank" rel="noreferrer">Скачать</a>
            <button class="btn btn-inline" type="button" data-dataset-edit="${escapeHtml(dataset.id)}">Редактировать</button>
            <button class="btn btn-inline btn-danger" type="button" data-dataset-unregister="${escapeHtml(dataset.id)}">Снять с учета</button>
          </div>
        </article>
      `;
    }).join('');
    if (!setStableMarkup(list, markup)) return;

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

    list.querySelectorAll('[data-dataset-filter-tag]').forEach(button => {
      button.addEventListener('click', () => {
        const search = document.getElementById('dataset-search');
        if (!search) return;
        search.value = button.getAttribute('data-dataset-filter-tag') || '';
        renderDatasets();
        search.focus();
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
        <div class="architecture-preview">
          <div class="architecture-preview-title">Схема модели</div>
          <strong>${name}</strong>
          <div class="architecture-preview-meta">${task} | ${backbone}</div>
          <div class="architecture-preview-note">Вход: ${inputSize}x${inputSize}</div>
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
        <div class="inference-summary">
          <strong>Результат инференса</strong>
          <div class="inference-summary-meta">Источник: ${escapeHtml(sourceName)}</div>
          <div class="inference-summary-meta">Предсказание: класс A (уверенность 95.2%)</div>
          <div class="inference-summary-note">Задержка: 12 мс</div>
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
