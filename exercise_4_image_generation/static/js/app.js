/**
 * Lumina AI Studio — Exercise 4 Frontend Application Logic
 * Interactive studio controller for Stable Diffusion v1.5 text-to-image synthesis.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const form = document.getElementById('generation-form');
  const promptInput = document.getElementById('prompt-input');
  const charCounter = document.getElementById('char-counter');
  const btnSurpriseMe = document.getElementById('btn-surprise-me');
  const btnClearPrompt = document.getElementById('btn-clear-prompt');
  const promptChipsContainer = document.getElementById('prompt-chips');

  const styleCards = document.querySelectorAll('.style-card');
  const selectedStyleInput = document.getElementById('selected-style-input');
  const selectedStyleName = document.getElementById('selected-style-name');

  const ratioButtons = document.querySelectorAll('.ratio-pill');
  const selectedRatioInput = document.getElementById('selected-ratio-input');

  const accordionToggle = document.getElementById('btn-toggle-advanced');
  const accordionItem = document.getElementById('accordion-advanced');

  const negPromptInput = document.getElementById('negative-prompt-input');
  const negTags = document.querySelectorAll('.neg-tag');

  const batchSlider = document.getElementById('batch-slider');
  const batchVal = document.getElementById('batch-val');
  const stepsSlider = document.getElementById('steps-slider');
  const stepsVal = document.getElementById('steps-val');
  const cfgSlider = document.getElementById('cfg-slider');
  const cfgVal = document.getElementById('cfg-val');

  const seedInput = document.getElementById('seed-input');
  const btnRandomSeed = document.getElementById('btn-random-seed');
  const btnLockSeed = document.getElementById('btn-lock-seed');

  const btnGenerate = document.getElementById('btn-generate');
  const progressContainer = document.getElementById('progress-container');
  const progressFill = document.getElementById('progress-fill');
  const progressLabel = document.getElementById('progress-label');
  const progressTimer = document.getElementById('progress-timer');
  const progressStepHint = document.getElementById('progress-step-hint');

  const statusText = document.getElementById('status-text');
  const emptyState = document.getElementById('empty-state');
  const galleryGrid = document.getElementById('gallery-grid');
  const viewportActions = document.getElementById('viewport-actions');
  const btnDownloadZip = document.getElementById('btn-download-zip');

  const telemetryDot = document.getElementById('telemetry-dot');
  const telemetryLabel = document.getElementById('telemetry-label');
  const historyBadge = document.getElementById('history-badge');

  // Modal Elements
  const imageModal = document.getElementById('image-modal');
  const btnModalClose = document.getElementById('btn-modal-close');
  const modalImg = document.getElementById('modal-img');
  const modalPromptText = document.getElementById('modal-prompt-text');
  const modalNegText = document.getElementById('modal-neg-text');
  const modalTimeBadge = document.getElementById('modal-time-badge');
  const modalSpecStyle = document.getElementById('modal-spec-style');
  const modalSpecSeed = document.getElementById('modal-spec-seed');
  const modalSpecSteps = document.getElementById('modal-spec-steps');
  const modalSpecCfg = document.getElementById('modal-spec-cfg');
  const modalSpecDims = document.getElementById('modal-spec-dims');
  const modalSpecEngine = document.getElementById('modal-spec-engine');
  const modalDownloadBtn = document.getElementById('modal-download-btn');
  const modalRemixBtn = document.getElementById('modal-remix-btn');
  const btnCopyPrompt = document.getElementById('btn-copy-prompt');

  // Drawer Elements
  const historyDrawerBackdrop = document.getElementById('history-drawer-backdrop');
  const btnOpenHistory = document.getElementById('btn-open-history');
  const btnCloseHistory = document.getElementById('btn-close-history');
  const historyList = document.getElementById('history-list');
  const drawerCountLabel = document.getElementById('drawer-count-label');

  // Docs Modal
  const docsModal = document.getElementById('docs-modal');
  const btnOpenDocs = document.getElementById('btn-open-docs');
  const btnDocsClose = document.getElementById('btn-docs-close');

  const toastContainer = document.getElementById('toast-container');

  // App State
  let currentMetadata = null;
  let timerInterval = null;
  let simulatedProgressInterval = null;
  let seedLocked = false;
  let curatedPrompts = [];

  // ==========================================
  // Initialization & System Telemetry
  // ==========================================
  async function initApp() {
    try {
      const res = await fetch('/api/system-info');
      const data = await res.json();
      if (data.system) {
        updateTelemetry(data.system);
      }
    } catch (e) {
      console.warn('Telemetry fetch error:', e);
      telemetryLabel.textContent = 'Engine Ready';
    }

    try {
      const res = await fetch('/api/presets');
      const data = await res.json();
      if (data.prompts && data.prompts.length > 0) {
        curatedPrompts = data.prompts;
        renderPromptChips(data.prompts);
      }
    } catch (e) {
      console.warn('Presets fetch error:', e);
    }

    loadHistoryCount();
  }

  function updateTelemetry(sys) {
    if (sys.cuda_available) {
      telemetryDot.className = 'pulse-dot active';
      telemetryLabel.textContent = sys.low_vram_warning
        ? `${sys.device} (Low VRAM Mode)`
        : `${sys.device}`;
    } else {
      telemetryDot.className = 'pulse-dot warning';
      telemetryLabel.textContent = sys.device;
    }
    historyBadge.textContent = sys.total_generated || 0;
  }

  function renderPromptChips(prompts) {
    promptChipsContainer.innerHTML = '';
    prompts.slice(0, 5).forEach((p) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'prompt-chip';
      chip.textContent = p;
      chip.title = p;
      chip.addEventListener('click', () => {
        promptInput.value = p;
        updateCharCounter();
        promptInput.focus();
      });
      promptChipsContainer.appendChild(chip);
    });
  }

  // ==========================================
  // Prompt Controls
  // ==========================================
  function updateCharCounter() {
    charCounter.textContent = `${promptInput.value.length} / 1000`;
  }
  promptInput.addEventListener('input', updateCharCounter);

  btnSurpriseMe.addEventListener('click', () => {
    if (curatedPrompts.length > 0) {
      const randomPrompt = curatedPrompts[Math.floor(Math.random() * curatedPrompts.length)];
      promptInput.value = randomPrompt;
      updateCharCounter();
      showToast('New inspiration prompt applied!', 'info');
    }
  });

  btnClearPrompt.addEventListener('click', () => {
    promptInput.value = '';
    updateCharCounter();
    promptInput.focus();
  });

  // ==========================================
  // Style Card Selection
  // ==========================================
  styleCards.forEach((card) => {
    card.addEventListener('click', () => {
      styleCards.forEach((c) => c.classList.remove('active'));
      card.classList.add('active');
      const styleName = card.dataset.style;
      selectedStyleInput.value = styleName;
      selectedStyleName.textContent = styleName;
    });
  });

  // ==========================================
  // Aspect Ratio Selection
  // ==========================================
  ratioButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      ratioButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      selectedRatioInput.value = btn.dataset.ratio;
    });
  });

  // ==========================================
  // Accordion & Advanced Tuning
  // ==========================================
  accordionToggle.addEventListener('click', () => {
    accordionItem.classList.toggle('open');
  });

  negTags.forEach((tag) => {
    tag.addEventListener('click', () => {
      const textToAdd = tag.dataset.text;
      const current = negPromptInput.value.trim();
      if (!current.includes(textToAdd)) {
        negPromptInput.value = current ? `${current}, ${textToAdd}` : textToAdd;
        showToast(`Added: ${textToAdd}`, 'info');
      }
    });
  });

  // Sliders value feedback
  batchSlider.addEventListener('input', (e) => { batchVal.textContent = e.target.value; });
  stepsSlider.addEventListener('input', (e) => { stepsVal.textContent = e.target.value; });
  cfgSlider.addEventListener('input', (e) => { cfgVal.textContent = parseFloat(e.target.value).toFixed(1); });

  // Seed controls
  btnRandomSeed.addEventListener('click', () => {
    const randomSeed = Math.floor(Math.random() * 2147483647);
    seedInput.value = randomSeed;
    showToast(`Generated seed: ${randomSeed}`, 'info');
  });

  btnLockSeed.addEventListener('click', () => {
    seedLocked = !seedLocked;
    btnLockSeed.classList.toggle('locked', seedLocked);
    showToast(seedLocked ? 'Seed locked' : 'Seed unlocked', 'info');
  });

  // ==========================================
  // Generation Process
  // ==========================================
  btnGenerate.addEventListener('click', handleGenerate);

  // Keyboard shortcut Ctrl+Enter
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleGenerate();
    }
  });

  async function handleGenerate() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      showToast('Please enter a descriptive prompt.', 'error');
      promptInput.focus();
      return;
    }

    const style = selectedStyleInput.value;
    const aspectRatio = selectedRatioInput.value;
    const negativePrompt = negPromptInput.value.trim();
    const numImages = parseInt(batchSlider.value, 10);
    const steps = parseInt(stepsSlider.value, 10);
    const guidance = parseFloat(cfgSlider.value);

    let seed = null;
    const seedVal = parseInt(seedInput.value, 10);
    if (!isNaN(seedVal) && seedVal >= 0) {
      seed = seedVal;
    }

    const selectedEngine = document.querySelector('input[name="engine"]:checked')?.value || 'auto';

    // Start UI Progress State
    startProgressUI(steps);

    const payload = {
      prompt,
      style,
      aspect_ratio: aspectRatio,
      negative_prompt: negativePrompt,
      num_images: numImages,
      num_inference_steps: steps,
      guidance_scale: guidance,
      seed,
      engine: selectedEngine,
    };

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await response.json();
      stopProgressUI();

      if (result.success && result.images && result.images.length > 0) {
        renderGallery(result.images, result.zip_url, result.elapsed_seconds);
        showToast(`Synthesized ${result.images.length} image(s) in ${result.elapsed_seconds}s!`, 'success');

        // Update history count
        loadHistoryCount();

        // Update seed input if was random
        if (!seedLocked && result.images[0].seed) {
          seedInput.value = result.images[0].seed;
        }
      } else {
        const errMessage = result.error || 'Generation failed. Please try again.';
        showToast(errMessage, 'error');
        statusText.textContent = `Error: ${errMessage}`;
      }
    } catch (err) {
      stopProgressUI();
      console.error('Fetch error:', err);
      showToast(`Network error: ${err.message}`, 'error');
      statusText.textContent = 'Connection error';
    }
  }

  function startProgressUI(steps) {
    btnGenerate.disabled = true;
    btnGenerate.querySelector('.btn-text span').textContent = 'Synthesizing...';
    progressContainer.style.display = 'block';
    progressFill.style.width = '10%';
    progressLabel.textContent = 'Preparing diffusion model...';
    statusText.textContent = 'Processing request...';

    const startTime = Date.now();
    timerInterval = setInterval(() => {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      progressTimer.textContent = `${elapsed}s`;
    }, 100);

    // Dynamic simulated step feedback
    let currentPct = 10;
    const totalSteps = steps || 25;
    let stepNum = 1;
    simulatedProgressInterval = setInterval(() => {
      if (currentPct < 90) {
        currentPct += 4;
        progressFill.style.width = `${currentPct}%`;
        stepNum = Math.min(totalSteps, Math.floor((currentPct / 100) * totalSteps) + 1);
        progressStepHint.textContent = `Denoising step ${stepNum} of ${totalSteps} (Latent UNet space)`;
        if (currentPct > 30) progressLabel.textContent = 'Iterative reverse diffusion in progress...';
        if (currentPct > 70) progressLabel.textContent = 'Decoding latents with VAE Decoder...';
      }
    }, 450);
  }

  function stopProgressUI() {
    clearInterval(timerInterval);
    clearInterval(simulatedProgressInterval);
    progressFill.style.width = '100%';
    progressLabel.textContent = 'Done!';

    setTimeout(() => {
      progressContainer.style.display = 'none';
      btnGenerate.disabled = false;
      btnGenerate.querySelector('.btn-text span').textContent = 'Generate Masterpiece';
    }, 600);
  }

  // ==========================================
  // Gallery Rendering & Lightbox
  // ==========================================
  function renderGallery(images, zipUrl, elapsed) {
    emptyState.style.display = 'none';
    galleryGrid.style.display = 'grid';
    galleryGrid.innerHTML = '';

    statusText.textContent = `Completed in ${elapsed}s · ${images.length} output(s)`;

    if (zipUrl) {
      btnDownloadZip.href = zipUrl;
      viewportActions.style.display = 'flex';
    }

    images.forEach((imgMeta, index) => {
      const card = document.createElement('div');
      card.className = 'image-card';
      card.id = `img-card-${index}`;

      card.innerHTML = `
        <div class="img-aspect-box">
          <img src="${imgMeta.filepath}" alt="${imgMeta.prompt}" loading="lazy">
          <div class="card-overlay">
            <div class="overlay-top">
              <button class="overlay-btn btn-zoom" title="Inspect Fullscreen Lightbox">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/>
                  <line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>
                </svg>
              </button>
              <a href="${imgMeta.filepath}" download="${imgMeta.filename}" class="overlay-btn" title="Download Image">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
              </a>
            </div>
            <div class="overlay-bottom">
              <span class="card-meta-pill">${imgMeta.style}</span>
              <span class="card-meta-pill">Seed: ${imgMeta.seed}</span>
            </div>
          </div>
        </div>
        <div class="card-footer">
          <p class="card-prompt-preview" title="${imgMeta.prompt}">${imgMeta.prompt}</p>
          <div class="card-details-row">
            <span>${imgMeta.dimensions}</span>
            <span>${imgMeta.steps} steps · CFG ${imgMeta.guidance}</span>
          </div>
        </div>
      `;

      // Click to open Lightbox
      card.querySelector('.img-aspect-box').addEventListener('click', (e) => {
        if (!e.target.closest('a')) {
          openLightbox(imgMeta);
        }
      });

      galleryGrid.appendChild(card);
    });
  }

  function openLightbox(meta) {
    currentMetadata = meta;
    modalImg.src = meta.filepath;
    modalPromptText.textContent = meta.prompt;
    modalNegText.textContent = meta.negative_prompt || 'None specified';
    modalTimeBadge.textContent = `${meta.generation_time_sec}s`;

    modalSpecStyle.textContent = meta.style;
    modalSpecSeed.textContent = meta.seed;
    modalSpecSteps.textContent = meta.steps;
    modalSpecCfg.textContent = meta.guidance;
    modalSpecDims.textContent = meta.dimensions;
    modalSpecEngine.textContent = meta.engine;

    modalDownloadBtn.href = meta.filepath;
    modalDownloadBtn.download = meta.filename;

    imageModal.style.display = 'flex';
  }

  btnModalClose.addEventListener('click', () => { imageModal.style.display = 'none'; });
  imageModal.addEventListener('click', (e) => {
    if (e.target === imageModal) imageModal.style.display = 'none';
  });

  btnCopyPrompt.addEventListener('click', () => {
    if (currentMetadata && currentMetadata.prompt) {
      navigator.clipboard.writeText(currentMetadata.prompt);
      showToast('Prompt copied to clipboard!', 'success');
    }
  });

  modalRemixBtn.addEventListener('click', () => {
    if (!currentMetadata) return;
    promptInput.value = currentMetadata.prompt;
    updateCharCounter();

    // Select style
    const styleCard = Array.from(styleCards).find((c) => c.dataset.style === currentMetadata.style);
    if (styleCard) styleCard.click();

    // Set parameters
    stepsSlider.value = currentMetadata.steps;
    stepsVal.textContent = currentMetadata.steps;

    cfgSlider.value = currentMetadata.guidance;
    cfgVal.textContent = parseFloat(currentMetadata.guidance).toFixed(1);

    seedInput.value = currentMetadata.seed;
    seedLocked = true;
    btnLockSeed.classList.add('locked');

    imageModal.style.display = 'none';
    showToast('Parameters loaded into Creation Studio!', 'info');
  });

  // ==========================================
  // History Drawer
  // ==========================================
  btnOpenHistory.addEventListener('click', async () => {
    historyDrawerBackdrop.style.display = 'block';
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      renderHistoryList(data.history || []);
    } catch (e) {
      console.warn('History fetch error:', e);
    }
  });

  btnCloseHistory.addEventListener('click', () => { historyDrawerBackdrop.style.display = 'none'; });
  historyDrawerBackdrop.addEventListener('click', (e) => {
    if (e.target === historyDrawerBackdrop) historyDrawerBackdrop.style.display = 'none';
  });

  function renderHistoryList(items) {
    historyList.innerHTML = '';
    drawerCountLabel.textContent = `${items.length} items`;

    if (items.length === 0) {
      historyList.innerHTML = '<p style="color:var(--text-dim);text-align:center;padding:30px;">No generations yet.</p>';
      return;
    }

    items.forEach((item) => {
      const card = document.createElement('div');
      card.className = 'history-card';
      card.innerHTML = `
        <img class="history-thumb" src="${item.filepath}" alt="${item.prompt}" loading="lazy">
        <div class="history-details">
          <p class="history-prompt">${item.prompt}</p>
          <div class="history-tags">
            <span>${item.style}</span>
            <span>Seed: ${item.seed}</span>
          </div>
        </div>
      `;

      card.addEventListener('click', () => {
        historyDrawerBackdrop.style.display = 'none';
        openLightbox(item);
      });

      historyList.appendChild(card);
    });
  }

  async function loadHistoryCount() {
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      historyBadge.textContent = data.count || 0;
    } catch (e) {
      // ignore
    }
  }

  // ==========================================
  // Architecture Documentation Modal
  // ==========================================
  btnOpenDocs.addEventListener('click', () => { docsModal.style.display = 'flex'; });
  btnDocsClose.addEventListener('click', () => { docsModal.style.display = 'none'; });
  docsModal.addEventListener('click', (e) => {
    if (e.target === docsModal) docsModal.style.display = 'none';
  });

  // ESC key closes all modals
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      imageModal.style.display = 'none';
      historyDrawerBackdrop.style.display = 'none';
      docsModal.style.display = 'none';
    }
  });

  // ==========================================
  // Toast Notifications Helper
  // ==========================================
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    let icon = 'ℹ️';
    if (type === 'success') icon = '✓';
    if (type === 'error') icon = '✕';

    toast.innerHTML = `<span style="font-weight:bold;">${icon}</span> <span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = '0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // Start initialization
  initApp();
});
