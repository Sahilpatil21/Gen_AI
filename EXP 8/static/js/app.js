/**
 * Sonora AI Music Studio — Experiment 8 Frontend Client Logic
 * Handles real-time Web Audio API visualizer, custom audio player, and composition pipeline.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const form = document.getElementById('music-form');
  const promptInput = document.getElementById('prompt-input');
  const charCounter = document.getElementById('char-counter');
  const btnSurpriseMe = document.getElementById('btn-surprise-me');
  const btnClearPrompt = document.getElementById('btn-clear-prompt');
  const genreChips = document.querySelectorAll('.genre-chip');

  const genreCards = document.querySelectorAll('.genre-card');
  const selectedGenreInput = document.getElementById('selected-genre-input');
  const selectedGenreName = document.getElementById('selected-genre-name');

  const durationSlider = document.getElementById('duration-slider');
  const durationVal = document.getElementById('duration-val');

  const accordionToggle = document.getElementById('btn-toggle-advanced');
  const accordionItem = document.getElementById('accordion-advanced');

  const tempSlider = document.getElementById('temp-slider');
  const tempVal = document.getElementById('temp-val');
  const topkSlider = document.getElementById('topk-slider');
  const topkVal = document.getElementById('topk-val');
  const guidanceSlider = document.getElementById('guidance-slider');
  const guidanceVal = document.getElementById('guidance-val');

  const seedInput = document.getElementById('seed-input');
  const btnRandomSeed = document.getElementById('btn-random-seed');
  const btnLockSeed = document.getElementById('btn-lock-seed');

  const btnCompose = document.getElementById('btn-compose');
  const progressContainer = document.getElementById('progress-container');
  const progressFill = document.getElementById('progress-fill');
  const progressLabel = document.getElementById('progress-label');
  const progressTimer = document.getElementById('progress-timer');
  const progressStepHint = document.getElementById('progress-step-hint');

  const statusText = document.getElementById('status-text');
  const viewportActions = document.getElementById('viewport-actions');
  const btnDownloadWav = document.getElementById('btn-download-wav');
  const emptyAudioState = document.getElementById('empty-audio-state');
  const playerControls = document.getElementById('player-controls');

  // Player Elements
  const audioElement = document.getElementById('audio-element');
  const btnPlayPause = document.getElementById('btn-play-pause');
  const playIcon = document.getElementById('play-icon');
  const pauseIcon = document.getElementById('pause-icon');
  const btnReplay = document.getElementById('btn-replay');
  const scrubBar = document.getElementById('scrub-bar');
  const scrubFill = document.getElementById('scrub-fill');
  const scrubHandle = document.getElementById('scrub-handle');
  const currentTimeEl = document.getElementById('current-time');
  const totalDurationEl = document.getElementById('total-duration');
  const volumeSlider = document.getElementById('volume-slider');
  const playerTrackTitle = document.getElementById('player-track-title');
  const playerTrackMeta = document.getElementById('player-track-meta');

  // Visualizer Canvas & Disc
  const canvas = document.getElementById('waveform-canvas');
  const ctx = canvas.getContext('2d');
  const discSpin = document.getElementById('disc-spin');
  const visualizerWrapper = document.querySelector('.visualizer-wrapper');

  // Telemetry & Badges
  const telemetryDot = document.getElementById('telemetry-dot');
  const telemetryLabel = document.getElementById('telemetry-label');
  const historyBadge = document.getElementById('history-badge');

  // Drawer & Modals
  const historyDrawerBackdrop = document.getElementById('history-drawer-backdrop');
  const btnOpenHistory = document.getElementById('btn-open-history');
  const btnCloseHistory = document.getElementById('btn-close-history');
  const historyList = document.getElementById('history-list');
  const drawerCountLabel = document.getElementById('drawer-count-label');

  const docsModal = document.getElementById('docs-modal');
  const btnOpenDocs = document.getElementById('btn-open-docs');
  const btnDocsClose = document.getElementById('btn-docs-close');
  const toastContainer = document.getElementById('toast-container');

  // Audio Context & State
  let audioCtx = null;
  let analyser = null;
  let audioSource = null;
  let isAudioCtxConnected = false;
  let animFrameId = null;
  let timerInterval = null;
  let progressStepInterval = null;
  let seedLocked = false;
  let isScrubbing = false;

  const CURATED_PROMPTS = [
    { text: "An upbeat lo-fi hip-hop track with a relaxed piano melody, vinyl crackle, and chill drums", genre: "Lo-Fi Hip-Hop" },
    { text: "Driving 80s cyberpunk synthwave with energetic arpeggiated bass, neon leads, and retro drums", genre: "Cyberpunk Synthwave" },
    { text: "Epic cinematic soundtrack with marching strings, heroic brass fanfare, and thunderous drums", genre: "Cinematic Orchestral" },
    { text: "Smooth late night jazz with walking upright bass, gentle brush snare, and mellow piano chords", genre: "Smooth Jazz" },
    { text: "Retro 8-bit arcade video game melody with playful fast square waves and cheerful chiptune beat", genre: "8-Bit Chiptune" },
    { text: "Deep tranquil ambient soundscape with evolving harmonic pads, cosmic space, and soothing chimes", genre: "Ambient Meditation" },
    { text: "Warm uplifting acoustic guitar fingerpicking with joyful rhythm and natural wooden resonance", genre: "Acoustic Folk" },
    { text: "High energy festival club dance track with infectious drop, punchy bass, and synth stabs", genre: "EDM / Dance" },
    { text: "Emotional solo classical piano piece with expressive dynamic phrasing and delicate reverb", genre: "Melancholic Piano" }
  ];

  // ==========================================
  // Initialization
  // ==========================================
  async function init() {
    updateCharCounter();
    drawAmbientWaveform();

    try {
      const res = await fetch('/api/system-info');
      const data = await res.json();
      if (data.system) {
        telemetryLabel.textContent = `${data.system.device} · EnCodec 32 kHz`;
        historyBadge.textContent = data.system.total_tracks || 0;
      }
    } catch (e) {
      telemetryLabel.textContent = 'EnCodec 32 kHz · Ready';
    }

    loadHistoryCount();
  }

  // ==========================================
  // Prompt & Genre Controls
  // ==========================================
  function updateCharCounter() {
    charCounter.textContent = `${promptInput.value.length} / 500`;
  }
  promptInput.addEventListener('input', updateCharCounter);

  btnSurpriseMe.addEventListener('click', () => {
    const item = CURATED_PROMPTS[Math.floor(Math.random() * CURATED_PROMPTS.length)];
    promptInput.value = item.text;
    updateCharCounter();

    // Select matching genre card
    const card = Array.from(genreCards).find((c) => c.dataset.genre === item.genre);
    if (card) card.click();

    showToast(`Loaded: ${item.genre} prompt`, 'info');
  });

  btnClearPrompt.addEventListener('click', () => {
    promptInput.value = '';
    updateCharCounter();
    promptInput.focus();
  });

  genreChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      genreChips.forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      const targetGenre = chip.dataset.genre;
      const card = Array.from(genreCards).find((c) => c.dataset.genre === targetGenre);
      if (card) card.click();
    });
  });

  genreCards.forEach((card) => {
    card.addEventListener('click', () => {
      genreCards.forEach((c) => c.classList.remove('active'));
      card.classList.add('active');
      const gName = card.dataset.genre;
      selectedGenreInput.value = gName;
      selectedGenreName.textContent = gName;

      // Update genre chip highlight
      genreChips.forEach((ch) => {
        ch.classList.toggle('active', ch.dataset.genre === gName);
      });
    });
  });

  durationSlider.addEventListener('input', (e) => {
    durationVal.textContent = `${e.target.value}s`;
  });

  // Accordion Toggle
  accordionToggle.addEventListener('click', () => {
    accordionItem.classList.toggle('open');
  });

  tempSlider.addEventListener('input', (e) => { tempVal.textContent = parseFloat(e.target.value).toFixed(1); });
  topkSlider.addEventListener('input', (e) => { topkVal.textContent = e.target.value; });
  guidanceSlider.addEventListener('input', (e) => { guidanceVal.textContent = parseFloat(e.target.value).toFixed(1); });

  btnRandomSeed.addEventListener('click', () => {
    const rSeed = Math.floor(Math.random() * 2147483647);
    seedInput.value = rSeed;
    showToast(`Generated seed: ${rSeed}`, 'info');
  });

  btnLockSeed.addEventListener('click', () => {
    seedLocked = !seedLocked;
    btnLockSeed.classList.toggle('locked', seedLocked);
    showToast(seedLocked ? 'Seed locked' : 'Seed unlocked', 'info');
  });

  // ==========================================
  // Music Composition Dispatcher
  // ==========================================
  btnCompose.addEventListener('click', handleCompose);

  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleCompose();
    }
  });

  async function handleCompose() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      showToast('Please enter a music prompt description.', 'error');
      promptInput.focus();
      return;
    }

    const genre = selectedGenreInput.value;
    const duration = parseInt(durationSlider.value, 10);
    const temperature = parseFloat(tempSlider.value);
    const topK = parseInt(topkSlider.value, 10);
    const guidance = parseFloat(guidanceSlider.value);

    let seed = null;
    const seedVal = parseInt(seedInput.value, 10);
    if (!isNaN(seedVal) && seedVal >= 0) {
      seed = seedVal;
    }

    const selectedEngine = document.querySelector('input[name="engine"]:checked')?.value || 'auto';

    startProgressUI(duration);

    const payload = {
      prompt,
      genre,
      duration_seconds: duration,
      temperature,
      top_k: topK,
      guidance_scale: guidance,
      seed,
      engine: selectedEngine,
    };

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      stopProgressUI();

      if (data.success && data.track) {
        loadTrackIntoPlayer(data.track);
        showToast(`Composed '${data.track.title}' in ${data.elapsed_seconds}s!`, 'success');
        loadHistoryCount();

        if (!seedLocked && seedInput.value === '-1') {
          // Keep -1
        }
      } else {
        const errMsg = data.error || 'Failed to compose track.';
        showToast(errMsg, 'error');
        statusText.textContent = `Error: ${errMsg}`;
      }
    } catch (err) {
      stopProgressUI();
      console.error('Fetch error:', err);
      showToast(`Network error: ${err.message}`, 'error');
      statusText.textContent = 'Connection error';
    }
  }

  function startProgressUI(duration) {
    btnCompose.disabled = true;
    btnCompose.querySelector('.btn-text span').textContent = 'Composing...';
    progressContainer.style.display = 'block';
    progressFill.style.width = '12%';
    progressLabel.textContent = 'Tokenizing text prompt via T5...';
    statusText.textContent = 'Synthesizing musical tokens...';

    const startTime = Date.now();
    timerInterval = setInterval(() => {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      progressTimer.textContent = `${elapsed}s`;
    }, 100);

    let pct = 15;
    progressStepInterval = setInterval(() => {
      if (pct < 90) {
        pct += 6;
        progressFill.style.width = `${pct}%`;
        if (pct > 30 && pct <= 65) {
          progressLabel.textContent = 'Autoregressively generating EnCodec acoustic codebooks...';
          progressStepHint.textContent = 'Predicting multi-stream tokens @ 50 Hz';
        } else if (pct > 65) {
          progressLabel.textContent = 'Decoding 32 kHz high-fidelity stereo waveform...';
          progressStepHint.textContent = 'Synthesizing audio output';
        }
      }
    }, 380);
  }

  function stopProgressUI() {
    clearInterval(timerInterval);
    clearInterval(progressStepInterval);
    progressFill.style.width = '100%';
    progressLabel.textContent = 'Complete!';

    setTimeout(() => {
      progressContainer.style.display = 'none';
      btnCompose.disabled = false;
      btnCompose.querySelector('.btn-text span').textContent = 'Compose Novel Music';
    }, 500);
  }

  // ==========================================
  // Track Player & Web Audio API Visualizer
  // ==========================================
  function setupWebAudio() {
    if (isAudioCtxConnected) return;

    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContext();
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;

      audioSource = audioCtx.createMediaElementSource(audioElement);
      audioSource.connect(analyser);
      analyser.connect(audioCtx.destination);
      isAudioCtxConnected = true;
    } catch (e) {
      console.warn('Web Audio API setup notice:', e);
    }
  }

  function loadTrackIntoPlayer(track) {
    emptyAudioState.style.display = 'none';
    playerControls.style.display = 'block';
    viewportActions.style.display = 'flex';

    btnDownloadWav.href = track.filepath;
    btnDownloadWav.download = track.filename;

    playerTrackTitle.textContent = track.title;
    playerTrackMeta.textContent = `${track.musical_key} · ${track.bpm} BPM · ${track.duration_seconds}s · 32 kHz Master`;

    statusText.textContent = `Ready · ${track.title} (${track.genre})`;

    audioElement.src = track.filepath;
    audioElement.load();

    // Auto-play on user action
    audioElement.play().then(() => {
      updatePlayPauseState(true);
      setupWebAudio();
      if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
      }
      startVisualizerLoop();
    }).catch(() => {
      updatePlayPauseState(false);
    });
  }

  function updatePlayPauseState(isPlaying) {
    playIcon.style.display = isPlaying ? 'none' : 'block';
    pauseIcon.style.display = isPlaying ? 'block' : 'none';
    discSpin.classList.toggle('playing', isPlaying);
    visualizerWrapper.classList.toggle('active-playback', isPlaying);
  }

  btnPlayPause.addEventListener('click', () => {
    setupWebAudio();
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }

    if (audioElement.paused) {
      audioElement.play();
      updatePlayPauseState(true);
      startVisualizerLoop();
    } else {
      audioElement.pause();
      updatePlayPauseState(false);
    }
  });

  btnReplay.addEventListener('click', () => {
    audioElement.currentTime = 0;
    audioElement.play();
    updatePlayPauseState(true);
  });

  // Time formatting helper (0:00)
  function formatTime(sec) {
    if (isNaN(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  }

  audioElement.addEventListener('timeupdate', () => {
    if (isScrubbing) return;
    const cur = audioElement.currentTime;
    const dur = audioElement.duration || 1;
    const pct = (cur / dur) * 100;

    scrubFill.style.width = `${pct}%`;
    scrubHandle.style.left = `${pct}%`;
    currentTimeEl.textContent = formatTime(cur);
  });

  audioElement.addEventListener('loadedmetadata', () => {
    totalDurationEl.textContent = formatTime(audioElement.duration);
  });

  audioElement.addEventListener('ended', () => {
    updatePlayPauseState(false);
  });

  // Scrub Bar interaction
  function seekTo(e) {
    const rect = scrubBar.getBoundingClientRect();
    const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    scrubFill.style.width = `${pos * 100}%`;
    scrubHandle.style.left = `${pos * 100}%`;
    if (audioElement.duration) {
      audioElement.currentTime = pos * audioElement.duration;
    }
  }

  scrubBar.addEventListener('mousedown', (e) => {
    isScrubbing = true;
    seekTo(e);
  });
  window.addEventListener('mousemove', (e) => {
    if (isScrubbing) seekTo(e);
  });
  window.addEventListener('mouseup', () => {
    isScrubbing = false;
  });

  volumeSlider.addEventListener('input', (e) => {
    audioElement.volume = parseFloat(e.target.value);
  });

  // ==========================================
  // Visualizer Animation Loops
  // ==========================================
  function startVisualizerLoop() {
    if (animFrameId) cancelAnimationFrame(animFrameId);

    function draw() {
      animFrameId = requestAnimationFrame(draw);

      if (!analyser || audioElement.paused) {
        drawAmbientWaveform();
        return;
      }

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const w = canvas.width;
      const h = canvas.height;
      const barCount = 64;
      const barWidth = (w / barCount) * 0.75;
      const gap = (w / barCount) * 0.25;

      for (let i = 0; i < barCount; i++) {
        const binIndex = Math.floor((i / barCount) * (bufferLength * 0.7));
        const val = dataArray[binIndex] || 0;
        const barHeight = (val / 255) * (h * 0.85);

        const x = i * (barWidth + gap) + gap / 2;
        const y = (h - barHeight) / 2;

        const grad = ctx.createLinearGradient(0, y, 0, y + barHeight);
        grad.addColorStop(0, '#06b6d4');
        grad.addColorStop(0.5, '#a855f7');
        grad.addColorStop(1, '#ec4899');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, Math.max(4, barHeight), 4);
        ctx.fill();
      }
    }

    draw();
  }

  let waveOffset = 0;
  function drawAmbientWaveform() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    waveOffset += 0.035;

    ctx.lineWidth = 2.5;

    // Cyan glowing sine wave
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.45)';
    ctx.beginPath();
    for (let x = 0; x < w; x += 4) {
      const y = h / 2 + Math.sin(x * 0.015 + waveOffset) * 22;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Purple harmonizing sine wave
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.35)';
    ctx.beginPath();
    for (let x = 0; x < w; x += 4) {
      const y = h / 2 + Math.sin(x * 0.02 - waveOffset * 0.8) * 16;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // ==========================================
  // History Library Drawer
  // ==========================================
  btnOpenHistory.addEventListener('click', async () => {
    historyDrawerBackdrop.style.display = 'block';
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      renderHistory(data.history || []);
    } catch (e) {
      console.warn('History fetch error:', e);
    }
  });

  btnCloseHistory.addEventListener('click', () => { historyDrawerBackdrop.style.display = 'none'; });
  historyDrawerBackdrop.addEventListener('click', (e) => {
    if (e.target === historyDrawerBackdrop) historyDrawerBackdrop.style.display = 'none';
  });

  function renderHistory(items) {
    historyList.innerHTML = '';
    drawerCountLabel.textContent = `${items.length} tracks`;

    if (items.length === 0) {
      historyList.innerHTML = '<p style="color:var(--text-dim);text-align:center;padding:30px;">No tracks composed yet.</p>';
      return;
    }

    items.forEach((track) => {
      const card = document.createElement('div');
      card.className = 'track-history-card';
      card.innerHTML = `
        <div class="history-info">
          <h5>${track.title}</h5>
          <p>${track.genre} · ${track.duration_seconds}s · ${track.bpm} BPM</p>
        </div>
        <div class="history-actions">
          <button class="history-btn btn-play-history" title="Play Track">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
          </button>
          <a href="${track.filepath}" download="${track.filename}" class="history-btn" title="Download WAV">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
          </a>
        </div>
      `;

      card.querySelector('.btn-play-history').addEventListener('click', () => {
        historyDrawerBackdrop.style.display = 'none';
        loadTrackIntoPlayer(track);
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

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      historyDrawerBackdrop.style.display = 'none';
      docsModal.style.display = 'none';
    }
  });

  // Toast Helper
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

  // Run initialization
  init();
});
