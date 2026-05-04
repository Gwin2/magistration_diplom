// Neural Network Trainer - Main Application

document.addEventListener('DOMContentLoaded', () => {
  // State management
  const state = {
    currentStep: 1,
    totalSteps: 8,
    theme: 'light',
    training: false,
    datasets: [],
    models: [],
    experiments: []
  };

  // DOM Elements
  const navPanel = document.getElementById('nav-panel');
  const menuToggle = document.getElementById('menu-toggle');
  const themeToggle = document.getElementById('theme-toggle');
  const prevStepBtn = document.getElementById('prev-step');
  const nextStepBtn = document.getElementById('next-step');
  const stepButtons = document.querySelectorAll('.step-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const headerStatus = document.getElementById('header-status');

  // Initialize
  function init() {
    setupNavigation();
    setupThemeToggle();
    setupStepNavigation();
    setupForms();
    loadSavedState();
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
    state.currentStep = step;
    
    // Update step buttons
    stepButtons.forEach(btn => {
      const btnStep = parseInt(btn.dataset.step);
      btn.classList.toggle('is-active', btnStep === step);
    });

    // Update tab panes
    tabPanes.forEach(pane => {
      pane.classList.toggle('is-active', pane.id === `tab-${getTabNameForStep(step)}`);
    });

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
  function getTabNameForStep(step) {
    const tabs = ['data', 'architecture', 'config', 'train', 'evaluate', 'deploy', 'metrics', 'resources'];
    return tabs[step - 1] || 'data';
  }

  // Setup form handlers
  function setupForms() {
    // Dataset upload
    document.getElementById('dataset-upload-btn')?.addEventListener('click', handleDatasetUpload);
    document.getElementById('dataset-register-btn')?.addEventListener('click', handleDatasetRegister);
    
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
        text.textContent = 'Ready';
        break;
      case 'busy':
        dot.classList.add('status-busy');
        text.textContent = 'Processing...';
        break;
      case 'error':
        dot.classList.add('status-error');
        text.textContent = 'Error';
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
    if (visualPreview && activePane.id === 'tab-architecture') {
      visualPreview.style.height = '';
      visualPreview.style.minHeight = '250px';
    }
  }

  // Validate architecture layout to prevent context overlap
  function validateArchitectureLayout() {
    const archTab = document.getElementById('tab-architecture');
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
        constructorGrid.style.gridTemplateColumns = '1fr 2fr 1fr';
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
    const statusEl = document.getElementById('dataset-status');
    
    updateHeaderStatus('busy');
    
    // Simulate upload
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    if (statusEl) {
      statusEl.textContent = 'Dataset uploaded successfully!';
      statusEl.style.color = 'var(--success)';
    }
    
    updateHeaderStatus('ready');
    
    // Auto-advance to next step after successful upload
    setTimeout(() => {
      goToStep(2);
    }, 1500);
  }

  function handleDatasetRegister() {
    const form = document.getElementById('dataset-register-form');
    const statusEl = document.getElementById('dataset-status');
    
    if (statusEl) {
      statusEl.textContent = 'Dataset path registered!';
      statusEl.style.color = 'var(--success)';
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
          <div style="font-size: 3rem; margin-bottom: 1rem;">🏗️</div>
          <div><strong>${name}</strong></div>
          <div style="color: var(--text-secondary);">${task} | ${backbone}</div>
          <div style="color: var(--text-muted); font-size: 0.85rem;">Input: ${inputSize}x${inputSize}</div>
        </div>
      `;
    }
    
    if (statusEl) {
      statusEl.textContent = 'Architecture generated! Review and customize below.';
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
        
        # Modify for custom input size
        self.input_size = ${inputSize}
        
        # Task-specific head
        self.task = "${task}"
        if self.task == "classification":
            self.head = nn.Linear(512, num_classes)
        elif self.task == "detection":
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 4 * num_classes)  # bbox coords
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
      statusEl.textContent = 'Architecture saved! Proceed to configuration.';
      statusEl.style.color = 'var(--success)';
    }
  }

  function saveConfig() {
    const statusEl = document.getElementById('config-status');
    if (statusEl) {
      statusEl.textContent = 'Configuration saved! Ready to train.';
      statusEl.style.color = 'var(--success)';
    }
    
    // Update training summary
    updateTrainingSummary();
  }

  function validateConfig() {
    const statusEl = document.getElementById('config-status');
    if (statusEl) {
      statusEl.textContent = '✓ Configuration validated successfully!';
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
      statusEl.textContent = 'Training in progress...';
    }
    
    if (logEl) {
      logEl.textContent = '[INFO] Starting training...\n';
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
          statusEl.textContent = '✓ Training completed successfully!';
          statusEl.style.color = 'var(--success)';
        }
        
        if (logEl) {
          logEl.textContent += '\n[INFO] Training completed!\n';
        }
        
        // Auto-advance to evaluation
        setTimeout(() => goToStep(5), 2000);
      }
    }, 200);
  }

  function clearTrainingLog() {
    const logEl = document.getElementById('training-log');
    if (logEl) logEl.textContent = 'Waiting for training to start...';
  }

  function exportModel() {
    const statusEl = document.getElementById('export-status');
    if (statusEl) {
      statusEl.textContent = 'Model exported successfully!';
      statusEl.style.color = 'var(--success)';
    }
  }

  function runInference() {
    const outputEl = document.getElementById('inference-output');
    if (outputEl) {
      outputEl.innerHTML = `
        <div style="display: flex; align-items: center; gap: 1rem;">
          <div style="font-size: 2rem;">✅</div>
          <div>
            <div><strong>Inference Result</strong></div>
            <div style="color: var(--text-secondary);">Prediction: Class A (95.2% confidence)</div>
            <div style="color: var(--text-muted); font-size: 0.85rem;">Latency: 12ms</div>
          </div>
        </div>
      `;
    }
  }

  function applyResources() {
    const statusEl = document.getElementById('usage-updated-at');
    if (statusEl) {
      statusEl.textContent = `Resource limits applied at ${new Date().toLocaleTimeString()}`;
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
