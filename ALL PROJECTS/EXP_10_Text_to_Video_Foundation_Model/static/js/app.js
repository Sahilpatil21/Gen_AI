/**
 * app.js
 * ======
 * Modern Cinema Video Studio Client (Experiment 10)
 * Handles interactive directorial controls, filmstrip frame scrubber, video playback,
 * asynchronous API requests, and toast alerts.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const promptInput = document.getElementById("prompt-input");
  const charCounter = document.getElementById("char-counter");
  const btnSurpriseMe = document.getElementById("btn-surprise-me");
  const btnClearPrompt = document.getElementById("btn-clear-prompt");
  const sampleChips = document.querySelectorAll(".scene-chip");

  const styleCards = document.querySelectorAll(".style-card");
  const selectedStyleInput = document.getElementById("selected-style-input");
  const selectedStyleName = document.getElementById("selected-style-name");

  const cameraSelect = document.getElementById("camera-select");
  const lightingSelect = document.getElementById("lighting-select");

  const ratioPills = document.querySelectorAll(".ratio-pill");
  const selectedResInput = document.getElementById("selected-res-input");

  const btnToggleAdvanced = document.getElementById("btn-toggle-advanced");
  const advancedAccordion = document.getElementById("accordion-advanced");

  const framesSlider = document.getElementById("frames-slider");
  const framesVal = document.getElementById("frames-val");
  const fpsSlider = document.getElementById("fps-slider");
  const fpsVal = document.getElementById("fps-val");
  const motionSlider = document.getElementById("motion-slider");
  const motionVal = document.getElementById("motion-val");

  const seedInput = document.getElementById("seed-input");
  const btnRandomSeed = document.getElementById("btn-random-seed");
  const btnLockSeed = document.getElementById("btn-lock-seed");

  const btnGenerate = document.getElementById("btn-generate");
  const statusText = document.getElementById("status-text");

  const progressContainer = document.getElementById("progress-container");
  const progressLabel = document.getElementById("progress-label");
  const progressTimer = document.getElementById("progress-timer");
  const progressFill = document.getElementById("progress-fill");
  const progressStepHint = document.getElementById("progress-step-hint");

  const playerWrapper = document.getElementById("player-wrapper");
  const mainVideoPlayer = document.getElementById("main-video-player");
  const filmstripContainer = document.getElementById("filmstrip-container");
  const filmstripTrack = document.getElementById("filmstrip-track");
  const filmstripCount = document.getElementById("filmstrip-count");
  const videoMetaCard = document.getElementById("video-meta-card");
  const emptyVideoState = document.getElementById("empty-video-state");
  const viewportActions = document.getElementById("viewport-actions");
  const btnDownloadMp4 = document.getElementById("btn-download-mp4");
  const btnDownloadGif = document.getElementById("btn-download-gif");

  // Meta Elements
  const metaTitle = document.getElementById("meta-title");
  const metaEngineTag = document.getElementById("meta-engine-tag");
  const metaPrompt = document.getElementById("meta-prompt");
  const metaCam = document.getElementById("meta-cam");
  const metaLight = document.getElementById("meta-light");
  const metaFrames = document.getElementById("meta-frames");
  const metaDuration = document.getElementById("meta-duration");
  const metaRes = document.getElementById("meta-res");
  const metaTime = document.getElementById("meta-time");

  // Telemetry & Modals
  const telemetryDot = document.getElementById("telemetry-dot");
  const telemetryLabel = document.getElementById("telemetry-label");
  const btnOpenHistory = document.getElementById("btn-open-history");
  const btnCloseHistory = document.getElementById("btn-close-history");
  const historyDrawerBackdrop = document.getElementById("history-drawer-backdrop");
  const historyDrawer = document.getElementById("history-drawer");
  const historyList = document.getElementById("history-list");
  const historyBadge = document.getElementById("history-badge");
  const drawerCountLabel = document.getElementById("drawer-count-label");

  const btnOpenDocs = document.getElementById("btn-open-docs");
  const btnDocsClose = document.getElementById("btn-docs-close");
  const docsModal = document.getElementById("docs-modal");
  const toastContainer = document.getElementById("toast-container");

  // State
  let isGenerating = false;
  let timerInterval = null;
  let currentStartTime = 0;
  let isSeedLocked = false;

  const surprisePrompts = [
    "A small boat sailing across a calm lake at sunrise, gentle water movement, cinematic wide shot",
    "Futuristic cyberpunk hovercraft speeding through neon rain-slicked skyscraper canyons, reflections in puddles",
    "Majestic snow leopard stalking across jagged Himalayan cliff tops in swirling snowstorm, slow motion",
    "Time-lapse of bioluminescent alien forest mushrooms pulsing glowing emerald spores under twin moons",
    "Vintage steam locomotive cutting through autumn misty pine forest, billowing white smoke, golden morning rays",
    "Deep ocean coral reef with shimmering schools of tropical fish and manta ray gliding gracefully past"
  ];

  // 1. Prompt Input & Character Counter
  function updateCharCounter() {
    const len = promptInput.value.length;
    charCounter.textContent = `${len} / 500`;
  }
  promptInput.addEventListener("input", updateCharCounter);
  updateCharCounter();

  btnClearPrompt.addEventListener("click", () => {
    promptInput.value = "";
    updateCharCounter();
    promptInput.focus();
  });

  btnSurpriseMe.addEventListener("click", () => {
    const rand = surprisePrompts[Math.floor(Math.random() * surprisePrompts.length)];
    promptInput.value = rand;
    updateCharCounter();
    showToast("Generated surprise cinematic prompt!", "info");
  });

  sampleChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      promptInput.value = chip.dataset.prompt;
      updateCharCounter();
      showToast("Applied lab manual sample prompt", "info");
    });
  });

  // 2. Style Selection
  styleCards.forEach((card) => {
    card.addEventListener("click", () => {
      styleCards.forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
      const styleName = card.dataset.style;
      selectedStyleInput.value = styleName;
      selectedStyleName.textContent = styleName;
    });
  });

  // 3. Aspect Ratio Selection
  ratioPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      ratioPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      selectedResInput.value = pill.dataset.res;
    });
  });

  // 4. Sliders
  function updateDurationReadout() {
    const duration = Number(framesSlider.value) / Number(fpsSlider.value);
    framesVal.textContent = `${framesSlider.value} Frames · ${duration.toFixed(1)}s`;
  }

  framesSlider.addEventListener("input", updateDurationReadout);

  fpsSlider.addEventListener("input", (e) => {
    fpsVal.textContent = `${e.target.value} FPS`;
    updateDurationReadout();
  });

  motionSlider.addEventListener("input", (e) => {
    motionVal.textContent = parseFloat(e.target.value).toFixed(1);
  });

  // 5. Seed Controls
  btnRandomSeed.addEventListener("click", () => {
    if (!isSeedLocked) {
      seedInput.value = Math.floor(Math.random() * 1000000);
      showToast(`Set randomized seed: ${seedInput.value}`, "info");
    }
  });

  btnLockSeed.addEventListener("click", () => {
    isSeedLocked = !isSeedLocked;
    btnLockSeed.classList.toggle("locked", isSeedLocked);
    seedInput.readOnly = isSeedLocked;
    showToast(isSeedLocked ? "Seed locked" : "Seed unlocked", "info");
  });

  // 6. Accordion
  btnToggleAdvanced.addEventListener("click", () => {
    advancedAccordion.classList.toggle("open");
  });

  // 7. System Telemetry Fetch
  async function fetchSystemInfo() {
    try {
      const res = await fetch("/api/system-info");
      const data = await res.json();
      if (data.system) {
        const sys = data.system;
        const gpuName = sys.gpu_name || "CPU Native";
        const vram = sys.vram_mb ? `${Math.round(sys.vram_mb)}MB` : "RAM";
        telemetryLabel.textContent = `${gpuName} (${vram}) · ${sys.backend}`;
      }
    } catch (err) {
      console.warn("System info fetch failed", err);
      telemetryLabel.textContent = "Video Foundation Engine Ready";
    }
  }
  fetchSystemInfo();

  // 8. Video Generation Submission
  document.getElementById("video-form").addEventListener("submit", (e) => {
    e.preventDefault();
    triggerGeneration();
  });

  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      triggerGeneration();
    }
  });

  async function triggerGeneration() {
    if (isGenerating) return;

    const prompt = promptInput.value.trim();
    if (!prompt) {
      showToast("Please enter a video prompt to direct.", "warning");
      promptInput.focus();
      return;
    }

    const style = selectedStyleInput.value;
    const cameraMotion = cameraSelect.value;
    const lighting = lightingSelect.value;
    const resolution = selectedResInput.value;
    const numFrames = parseInt(framesSlider.value, 10);
    const fps = parseInt(fpsSlider.value, 10);
    const motionIntensity = parseFloat(motionSlider.value);

    let seedVal = seedInput.value.trim();
    let seed = seedVal === "" || seedVal === "-1" ? null : parseInt(seedVal, 10);

    const engineInput = document.querySelector('input[name="engine"]:checked');
    const engine = engineInput ? engineInput.value : "auto";

    // UI Loading state
    startLoadingState(numFrames);

    const payload = {
      prompt,
      style,
      camera_motion: cameraMotion,
      lighting,
      resolution,
      num_frames: numFrames,
      fps,
      motion_intensity: motionIntensity,
      seed,
      engine
    };

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const result = await res.json();

      if (!res.ok || !result.success) {
        throw new Error(result.error || "Failed to generate video sequence.");
      }

      renderGeneratedVideo(result.video);
      showToast("Cinema video sequence generated successfully!", "success");
      loadHistory();
    } catch (err) {
      console.error(err);
      showToast(`Error: ${err.message}`, "error");
      statusText.textContent = "Error occurred during video synthesis.";
    } finally {
      stopLoadingState();
    }
  }

  function startLoadingState(numFrames) {
    isGenerating = true;
    btnGenerate.disabled = true;
    btnGenerate.classList.add("loading");
    statusText.textContent = "Synthesizing video sequence...";

    progressContainer.style.display = "block";
    progressFill.style.width = "5%";
    currentStartTime = Date.now();
    progressTimer.textContent = "0.0s";

    const steps = [
      "Tokenizing natural-language prompt with text encoder...",
      "Setting up 3D spatio-temporal latent lattice...",
      "Generating smooth camera trajectory & kinematics...",
      "Synthesizing temporal frame coherence...",
      "Decoding latent frames with Temporal VAE...",
      "Encoding frames into H.264 MP4 container with OpenCV..."
    ];

    let stepIndex = 0;
    progressStepHint.textContent = steps[0];

    timerInterval = setInterval(() => {
      const elapsed = ((Date.now() - currentStartTime) / 1000).toFixed(1);
      progressTimer.textContent = `${elapsed}s`;

      const progress = Math.min(95, Math.floor(elapsed * 18));
      progressFill.style.width = `${progress}%`;

      if (elapsed > (stepIndex + 1) * 0.8 && stepIndex < steps.length - 1) {
        stepIndex++;
        progressStepHint.textContent = steps[stepIndex];
      }
    }, 100);
  }

  function stopLoadingState() {
    isGenerating = false;
    btnGenerate.disabled = false;
    btnGenerate.classList.remove("loading");
    clearInterval(timerInterval);
    progressFill.style.width = "100%";

    setTimeout(() => {
      progressContainer.style.display = "none";
    }, 600);
  }

  // 9. Render Video Result
  function renderGeneratedVideo(video) {
    emptyVideoState.style.display = "none";
    playerWrapper.style.display = "block";
    filmstripContainer.style.display = "block";
    videoMetaCard.style.display = "block";
    viewportActions.style.display = "flex";

    // Main Video Player
    mainVideoPlayer.src = video.video_url;
    mainVideoPlayer.load();
    mainVideoPlayer.play().catch(() => {});

    // Downloads
    btnDownloadMp4.href = video.video_url;
    btnDownloadMp4.download = `${video.video_id}.mp4`;

    if (video.gif_url) {
      btnDownloadGif.style.display = "inline-flex";
      btnDownloadGif.href = video.gif_url;
      btnDownloadGif.download = `${video.video_id}.gif`;
    } else {
      btnDownloadGif.style.display = "none";
    }

    // Populate Meta Card
    metaTitle.textContent = `${video.style} · Sequence`;
    metaEngineTag.textContent = video.engine || "Foundation Engine";
    metaPrompt.textContent = `"${video.prompt}"`;
    metaCam.textContent = video.camera_motion || "Panoramic";
    metaLight.textContent = video.lighting || "Cinematic";
    metaFrames.textContent = `${video.num_frames} @ ${video.fps} FPS`;
    metaDuration.textContent = `${video.duration_seconds.toFixed(2)}s`;
    metaRes.textContent = video.resolution || "512x288";
    metaTime.textContent = `${video.generation_time.toFixed(2)}s`;

    statusText.textContent = `Footage ready · ${video.num_frames} frames (${video.duration_seconds.toFixed(2)}s)`;

    // Filmstrip Scrubber
    renderFilmstrip(video.frame_urls, video.duration_seconds, video.num_frames);
  }

  function renderFilmstrip(frameUrls, duration, numFrames) {
    filmstripTrack.innerHTML = "";
    filmstripCount.textContent = `${numFrames} Frames`;

    if (!frameUrls || frameUrls.length === 0) return;

    frameUrls.forEach((url, idx) => {
      const frameItem = document.createElement("div");
      frameItem.className = "filmstrip-frame";
      frameItem.dataset.frameIndex = idx;

      const img = document.createElement("img");
      img.src = url;
      img.alt = `Frame ${idx + 1}`;
      img.loading = "lazy";

      const frameLabel = document.createElement("span");
      frameLabel.className = "frame-num";
      frameLabel.textContent = `#${idx + 1}`;

      frameItem.appendChild(img);
      frameItem.appendChild(frameLabel);

      // Clicking frame seeks the video player to that exact time
      frameItem.addEventListener("click", () => {
        document.querySelectorAll(".filmstrip-frame").forEach(f => f.classList.remove("active-frame"));
        frameItem.classList.add("active-frame");

        const targetTime = (idx / (numFrames - 1 || 1)) * duration;
        mainVideoPlayer.currentTime = targetTime;
        mainVideoPlayer.pause();
      });

      filmstripTrack.appendChild(frameItem);
    });
  }

  // Sync filmstrip active frame during video playback
  mainVideoPlayer.addEventListener("timeupdate", () => {
    const curTime = mainVideoPlayer.currentTime;
    const dur = mainVideoPlayer.duration || 1;
    const frames = document.querySelectorAll(".filmstrip-frame");
    if (!frames.length) return;

    const frameIdx = Math.min(frames.length - 1, Math.floor((curTime / dur) * frames.length));
    frames.forEach((f, i) => {
      f.classList.toggle("active-frame", i === frameIdx);
    });
  });

  // 10. History / Cinema Vault
  async function loadHistory() {
    try {
      const res = await fetch("/api/history");
      const data = await res.json();
      const items = data.history || [];

      historyBadge.textContent = items.length;
      drawerCountLabel.textContent = `${items.length} videos`;
      renderHistoryItems(items);
    } catch (err) {
      console.warn("History fetch failed:", err);
    }
  }

  function renderHistoryItems(items) {
    historyList.innerHTML = "";
    if (items.length === 0) {
      historyList.innerHTML = `
        <div class="empty-history">
          <p>No video clips generated yet. Direct your first scene!</p>
        </div>
      `;
      return;
    }

    items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "history-card";

      const thumbUrl = item.frame_urls && item.frame_urls.length > 0 ? item.frame_urls[0] : item.gif_url || "";

      card.innerHTML = `
        <div class="history-thumb-wrapper">
          ${thumbUrl ? `<img src="${thumbUrl}" alt="Thumbnail" class="history-thumb" />` : '<div class="history-placeholder">MP4</div>'}
          <span class="history-duration">${item.duration_seconds.toFixed(1)}s</span>
        </div>
        <div class="history-info">
          <p class="history-prompt" title="${item.prompt}">${item.prompt}</p>
          <div class="history-meta-row">
            <span class="history-tag">${item.style}</span>
            <span class="history-specs">${item.num_frames} frames · ${item.resolution}</span>
          </div>
        </div>
      `;

      card.addEventListener("click", () => {
        renderGeneratedVideo(item);
        closeHistoryDrawer();
        showToast("Loaded video clip from vault", "info");
      });

      historyList.appendChild(card);
    });
  }

  function openHistoryDrawer() {
    loadHistory();
    historyDrawerBackdrop.style.display = "block";
    setTimeout(() => historyDrawer.classList.add("open"), 10);
  }

  function closeHistoryDrawer() {
    historyDrawer.classList.remove("open");
    setTimeout(() => {
      historyDrawerBackdrop.style.display = "none";
    }, 250);
  }

  btnOpenHistory.addEventListener("click", openHistoryDrawer);
  btnCloseHistory.addEventListener("click", closeHistoryDrawer);
  historyDrawerBackdrop.addEventListener("click", (e) => {
    if (e.target === historyDrawerBackdrop) closeHistoryDrawer();
  });

  // 11. Theory & Documentation Modal
  btnOpenDocs.addEventListener("click", () => {
    docsModal.style.display = "flex";
  });

  btnDocsClose.addEventListener("click", () => {
    docsModal.style.display = "none";
  });

  docsModal.addEventListener("click", (e) => {
    if (e.target === docsModal) docsModal.style.display = "none";
  });

  // 12. Toast System
  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    const iconMap = {
      info: "ℹ️",
      success: "✓",
      warning: "⚠️",
      error: "✕"
    };

    toast.innerHTML = `
      <span class="toast-icon">${iconMap[type] || "•"}</span>
      <span class="toast-msg">${message}</span>
    `;

    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("visible");
    }, 10);

    setTimeout(() => {
      toast.classList.remove("visible");
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // Initial History Load
  loadHistory();
});
