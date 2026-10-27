/**
 * app.js
 * ======
 * Client controller for Experiment 6 (DCGAN CelebA Face Generator).
 */

document.addEventListener("DOMContentLoaded", () => {
  // Tabs
  const tabBtnSynthesis = document.getElementById("tab-btn-synthesis");
  const tabBtnMorph = document.getElementById("tab-btn-morph");
  const paneSynthesis = document.getElementById("pane-synthesis");
  const paneMorph = document.getElementById("pane-morph");
  const viewportTitle = document.getElementById("viewport-title");
  const faceGrid = document.getElementById("face-grid");
  const morphView = document.getElementById("morph-view");
  const emptyState = document.getElementById("empty-state");
  const viewportActions = document.getElementById("viewport-actions");
  const statusText = document.getElementById("status-text");
  const progressContainer = document.getElementById("progress-container");

  // Synthesis Controls
  const countPills = document.querySelectorAll(".count-pill");
  const countInput = document.getElementById("count-input");
  const resPills = document.querySelectorAll(".res-pill");
  const scaleInput = document.getElementById("scale-input");
  const tempSlider = document.getElementById("temp-slider");
  const tempVal = document.getElementById("temp-val");
  const seedInput = document.getElementById("seed-input");
  const btnRandomSeed = document.getElementById("btn-random-seed");
  const btnLockSeed = document.getElementById("btn-lock-seed");
  const btnGenerate = document.getElementById("btn-generate");
  const faceForm = document.getElementById("face-form");
  const btnDownloadAll = document.getElementById("btn-download-all");

  // Morph Controls
  const morphForm = document.getElementById("morph-form");
  const morphSeedA = document.getElementById("morph-seed-a");
  const morphSeedB = document.getElementById("morph-seed-b");
  const btnDiceA = document.getElementById("btn-dice-a");
  const btnDiceB = document.getElementById("btn-dice-b");
  const morphStepsSlider = document.getElementById("morph-steps-slider");
  const morphStepsVal = document.getElementById("morph-steps-val");
  const morphFpsSlider = document.getElementById("morph-fps-slider");
  const morphFpsVal = document.getElementById("morph-fps-val");
  const btnGenerateMorph = document.getElementById("btn-generate-morph");
  const morphImg = document.getElementById("morph-img");
  const morphStepsBadge = document.getElementById("morph-steps-badge");
  const btnDownloadGif = document.getElementById("btn-download-gif");
  const btnDownloadWebp = document.getElementById("btn-download-webp");
  const morphFramesStrip = document.getElementById("morph-frames-strip");

  // History & Docs
  const btnOpenHistory = document.getElementById("btn-open-history");
  const btnCloseHistory = document.getElementById("btn-close-history");
  const historyBackdrop = document.getElementById("history-backdrop");
  const historyList = document.getElementById("history-list");
  const historyBadge = document.getElementById("history-badge");
  const drawerCount = document.getElementById("drawer-count");
  const btnOpenDocs = document.getElementById("btn-open-docs");
  const btnCloseDocs = document.getElementById("btn-close-docs");
  const docsModal = document.getElementById("docs-modal");
  const telemetryLabel = document.getElementById("telemetry-label");
  const toastContainer = document.getElementById("toast-container");

  let isGenerating = false;
  let isSeedLocked = false;
  let activeFaces = [];

  // 1. Tab Switching
  function switchTab(tab) {
    if (tab === "synthesis") {
      tabBtnSynthesis.classList.add("active");
      tabBtnMorph.classList.remove("active");
      paneSynthesis.classList.add("active");
      paneMorph.classList.remove("active");
      viewportTitle.textContent = "Synthetic Face Gallery";
      morphView.style.display = "none";
      if (activeFaces.length > 0) {
        faceGrid.style.display = "grid";
        emptyState.style.display = "none";
        viewportActions.style.display = "flex";
      } else {
        emptyState.style.display = "block";
        viewportActions.style.display = "none";
      }
    } else {
      tabBtnMorph.classList.add("active");
      tabBtnSynthesis.classList.remove("active");
      paneMorph.classList.add("active");
      paneSynthesis.classList.remove("active");
      viewportTitle.textContent = "Latent Geodesic Trajectory (Identity Morphing)";
      faceGrid.style.display = "none";
      viewportActions.style.display = "none";
      if (morphImg.src && morphImg.src !== window.location.href) {
        morphView.style.display = "block";
        emptyState.style.display = "none";
      } else {
        emptyState.style.display = "block";
      }
    }
  }

  tabBtnSynthesis.addEventListener("click", () => switchTab("synthesis"));
  tabBtnMorph.addEventListener("click", () => switchTab("morph"));

  // 2. Count & Scale Selectors
  countPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      countPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      countInput.value = pill.dataset.count;
    });
  });

  resPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      resPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      scaleInput.value = pill.dataset.scale;
    });
  });

  // 3. Sliders
  tempSlider.addEventListener("input", (e) => {
    tempVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  morphStepsSlider.addEventListener("input", (e) => {
    morphStepsVal.textContent = `${e.target.value} Steps`;
  });

  morphFpsSlider.addEventListener("input", (e) => {
    morphFpsVal.textContent = `${e.target.value} FPS`;
  });

  // 4. Seed Rollers
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
    showToast(isSeedLocked ? "Seed locked for reproduction" : "Seed unlocked", "info");
  });

  btnDiceA.addEventListener("click", () => {
    morphSeedA.value = Math.floor(Math.random() * 1000000);
  });

  btnDiceB.addEventListener("click", () => {
    morphSeedB.value = Math.floor(Math.random() * 1000000);
  });

  // 5. System Telemetry Fetch
  async function fetchSystemInfo() {
    try {
      const res = await fetch("/api/system-info");
      const data = await res.json();
      if (data.system) {
        telemetryLabel.textContent = `${data.system.architecture} · ${data.system.device}`;
      }
    } catch (err) {
      console.warn("System info fetch failed:", err);
    }
  }
  fetchSystemInfo();

  // 6. Generate Faces
  faceForm.addEventListener("submit", (e) => {
    e.preventDefault();
    triggerFaceGeneration();
  });

  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      triggerFaceGeneration();
    }
  });

  async function triggerFaceGeneration() {
    if (isGenerating) return;

    const count = parseInt(countInput.value, 10);
    const scale = parseInt(scaleInput.value, 10);
    const temperature = parseFloat(tempSlider.value);
    let seedVal = seedInput.value.trim();
    let seed = seedVal === "" || seedVal === "-1" ? null : parseInt(seedVal, 10);

    isGenerating = true;
    btnGenerate.disabled = true;
    btnGenerate.classList.add("loading");
    statusText.textContent = `Sampling ${count} latent vectors...`;
    progressContainer.style.display = "block";

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ count, scale, temperature, seed }),
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || "Failed to generate faces.");
      }

      activeFaces = data.faces;
      renderFaceGrid(data.faces);
      statusText.textContent = `Synthesized ${data.count} faces in ${data.elapsed_seconds}s`;
      showToast(`Generated ${data.count} celebrity faces!`, "success");
      loadHistory();
    } catch (err) {
      console.error(err);
      showToast(`Error: ${err.message}`, "error");
      statusText.textContent = "Error occurred during synthesis.";
    } finally {
      isGenerating = false;
      btnGenerate.disabled = false;
      btnGenerate.classList.remove("loading");
      progressContainer.style.display = "none";
    }
  }

  function renderFaceGrid(faces) {
    emptyState.style.display = "none";
    faceGrid.style.display = "grid";
    viewportActions.style.display = "flex";
    faceGrid.innerHTML = "";

    faces.forEach((face, idx) => {
      const card = document.createElement("div");
      card.className = "face-card";

      card.innerHTML = `
        <div class="face-card-img-wrapper">
          <img src="${face.filepath}" alt="Face #${idx + 1}" loading="lazy" />
          <div class="face-card-overlay">
            <a href="${face.filepath}" download="${face.filename}" class="overlay-btn" title="Download High-Res PNG">Download</a>
            <button type="button" class="overlay-btn btn-use-seed" data-seed="${face.seed}" title="Use this seed for morphing">Morph This</button>
          </div>
        </div>
        <div class="face-card-info">
          <span class="face-card-seed">Seed #${face.seed}</span>
          <span class="face-card-tag">${face.resolution}</span>
        </div>
      `;

      const morphBtn = card.querySelector(".btn-use-seed");
      morphBtn.addEventListener("click", () => {
        morphSeedA.value = face.seed;
        switchTab("morph");
        showToast(`Set Identity A to Seed #${face.seed}`, "info");
      });

      faceGrid.appendChild(card);
    });
  }

  // 7. Latent Morphing
  morphForm.addEventListener("submit", (e) => {
    e.preventDefault();
    triggerMorphGeneration();
  });

  async function triggerMorphGeneration() {
    if (isGenerating) return;

    const seedA = parseInt(morphSeedA.value, 10);
    const seedB = parseInt(morphSeedB.value, 10);
    const steps = parseInt(morphStepsSlider.value, 10);
    const fps = parseInt(morphFpsSlider.value, 10);
    const scale = parseInt(scaleInput.value, 10);

    isGenerating = true;
    btnGenerateMorph.disabled = true;
    btnGenerateMorph.classList.add("loading");
    statusText.textContent = `Computing geodesic trajectory ($z_{${seedA}} \\to z_{${seedB}}$)...`;
    progressContainer.style.display = "block";

    try {
      const res = await fetch("/api/morph", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed_a: seedA, seed_b: seedB, steps, fps, scale }),
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || "Failed to compute latent morphing.");
      }

      renderMorphResult(data.morph);
      statusText.textContent = `Morphed ${data.morph.steps} steps (${data.morph.duration_sec}s) in ${data.elapsed_seconds}s`;
      showToast("Latent space morphing generated successfully!", "success");
      loadHistory();
    } catch (err) {
      console.error(err);
      showToast(`Error: ${err.message}`, "error");
      statusText.textContent = "Error occurred during morphing.";
    } finally {
      isGenerating = false;
      btnGenerateMorph.disabled = false;
      btnGenerateMorph.classList.remove("loading");
      progressContainer.style.display = "none";
    }
  }

  function renderMorphResult(morph) {
    emptyState.style.display = "none";
    faceGrid.style.display = "none";
    morphView.style.display = "block";

    // Media
    morphImg.src = morph.gif_filepath;
    morphStepsBadge.textContent = `${morph.steps} Steps @ ${morph.fps} FPS (${morph.duration_sec}s)`;
    btnDownloadGif.href = morph.gif_filepath;
    btnDownloadGif.download = `${morph.id}.gif`;
    btnDownloadWebp.href = morph.webp_filepath;
    btnDownloadWebp.download = `${morph.id}.webp`;

    // Filmstrip Track
    morphFramesStrip.innerHTML = "";
    morph.frame_filepaths.forEach((path, idx) => {
      const thumb = document.createElement("div");
      thumb.className = "morph-frame-thumb";
      thumb.title = `Step ${idx + 1} of ${morph.steps}`;
      thumb.innerHTML = `<img src="${path}" alt="Step ${idx + 1}" />`;
      thumb.addEventListener("click", () => {
        morphImg.src = path;
      });
      morphFramesStrip.appendChild(thumb);
    });
  }

  // 8. Download All Active Batch
  btnDownloadAll.addEventListener("click", () => {
    if (!activeFaces.length) return;
    activeFaces.forEach((face, idx) => {
      setTimeout(() => {
        const a = document.createElement("a");
        a.href = face.filepath;
        a.download = face.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }, idx * 150);
    });
    showToast(`Downloading ${activeFaces.length} portraits...`, "info");
  });

  // 9. History Vault
  async function loadHistory() {
    try {
      const res = await fetch("/api/history");
      const data = await res.json();
      const items = data.history || [];

      historyBadge.textContent = items.length;
      drawerCount.textContent = `${items.length} faces`;
      historyList.innerHTML = "";

      items.slice().reverse().forEach((item) => {
        const card = document.createElement("div");
        card.className = "face-card";
        card.innerHTML = `
          <div class="face-card-img-wrapper">
            <img src="${item.filepath}" alt="Face #${item.seed}" loading="lazy" />
          </div>
          <div class="face-card-info">
            <span class="face-card-seed">#${item.seed}</span>
            <a href="${item.filepath}" download="${item.filename}" class="overlay-btn">Save</a>
          </div>
        `;
        historyList.appendChild(card);
      });
    } catch (err) {
      console.warn("Failed to load history:", err);
    }
  }

  btnOpenHistory.addEventListener("click", () => {
    loadHistory();
    historyBackdrop.style.display = "flex";
  });

  btnCloseHistory.addEventListener("click", () => {
    historyBackdrop.style.display = "none";
  });

  historyBackdrop.addEventListener("click", (e) => {
    if (e.target === historyBackdrop) historyBackdrop.style.display = "none";
  });

  // 10. Docs / Theory Modal
  btnOpenDocs.addEventListener("click", () => {
    docsModal.style.display = "flex";
  });

  btnCloseDocs.addEventListener("click", () => {
    docsModal.style.display = "none";
  });

  docsModal.addEventListener("click", (e) => {
    if (e.target === docsModal) docsModal.style.display = "none";
  });

  // 11. Toast System
  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);

    setTimeout(() => toast.classList.add("visible"), 10);
    setTimeout(() => {
      toast.classList.remove("visible");
      setTimeout(() => toast.remove(), 250);
    }, 3500);
  }

  // Load history count on init
  loadHistory();
});
