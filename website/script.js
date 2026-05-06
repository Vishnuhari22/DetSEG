/* ============================================
   DetSEG — Interactivity, Upload & Analysis
   ============================================ */

/* ============================================
   AUDIO ALERT SYSTEM (Web Audio API)
   Clinical dual-tone beep for polyp detection
   ============================================ */
class AudioAlertSystem {
  constructor() {
    this.audioCtx = null;
    this.enabled = true;
    this.volume = 0.7;
    this.pendingTimeouts = [];
  }

  _ensureContext() {
    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  playBeep() {
    if (!this.enabled) return;
    this._ensureContext();
    const ctx = this.audioCtx;
    const now = ctx.currentTime;

    // Gain envelope
    const gainNode = ctx.createGain();
    gainNode.gain.setValueAtTime(0, now);
    gainNode.gain.linearRampToValueAtTime(this.volume * 0.3, now + 0.01);
    gainNode.gain.setValueAtTime(this.volume * 0.3, now + 0.08);
    gainNode.gain.linearRampToValueAtTime(0, now + 0.2);
    gainNode.connect(ctx.destination);

    // Tone 1: 880Hz (high, attention-getting)
    const osc1 = ctx.createOscillator();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(880, now);
    osc1.frequency.linearRampToValueAtTime(440, now + 0.2);
    osc1.connect(gainNode);
    osc1.start(now);
    osc1.stop(now + 0.2);

    // Tone 2: harmonic layer for clinical character
    const osc2 = ctx.createOscillator();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(1320, now);
    osc2.frequency.linearRampToValueAtTime(660, now + 0.2);
    const gain2 = ctx.createGain();
    gain2.gain.setValueAtTime(0, now);
    gain2.gain.linearRampToValueAtTime(this.volume * 0.1, now + 0.01);
    gain2.gain.linearRampToValueAtTime(0, now + 0.2);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(now);
    osc2.stop(now + 0.2);
  }

  playAlertSequence(alertTimestamps) {
    this.stop();
    if (!this.enabled || !alertTimestamps || alertTimestamps.length === 0) return;

    // Schedule beeps relative to now, spread out over time
    const baseDelay = 500; // small initial delay (ms)
    const spacing = 1200;  // ms between beeps in playback

    alertTimestamps.forEach((alert, i) => {
      const delay = baseDelay + i * spacing;
      const tid = setTimeout(() => this.playBeep(), delay);
      this.pendingTimeouts.push(tid);
    });
  }

  /**
   * Synchronize beep alerts to video playback.
   * Fires beeps when video.currentTime crosses each alert timestamp.
   */
  syncToVideo(videoElement, alertTimestamps) {
    this.stop();
    if (!alertTimestamps || alertTimestamps.length === 0) return;

    const firedSet = new Set();
    const tolerance = 0.5; // seconds — fire if within this window

    const onTimeUpdate = () => {
      if (!this.enabled) return;
      const t = videoElement.currentTime;
      alertTimestamps.forEach((alert, i) => {
        if (!firedSet.has(i) && Math.abs(t - alert.time_sec) < tolerance) {
          firedSet.add(i);
          this.playBeep();
        }
      });
    };

    // Reset fired alerts when user seeks
    const onSeeked = () => {
      const t = videoElement.currentTime;
      firedSet.clear();
      // Re-mark alerts that are already in the past
      alertTimestamps.forEach((alert, i) => {
        if (alert.time_sec < t - tolerance) {
          firedSet.add(i);
        }
      });
    };

    videoElement.addEventListener('timeupdate', onTimeUpdate);
    videoElement.addEventListener('seeked', onSeeked);

    // Store cleanup function
    this._videoCleanup = () => {
      videoElement.removeEventListener('timeupdate', onTimeUpdate);
      videoElement.removeEventListener('seeked', onSeeked);
      firedSet.clear();
      this._videoCleanup = null;
    };
  }

  stop() {
    this.pendingTimeouts.forEach(tid => clearTimeout(tid));
    this.pendingTimeouts = [];
    if (this._videoCleanup) this._videoCleanup();
  }

  setVolume(v) { this.volume = Math.max(0, Math.min(1, v)); }
  setEnabled(on) { this.enabled = on; }
}


document.addEventListener('DOMContentLoaded', () => {

  // ==============================
  // SCROLL REVEAL
  // ==============================
  const revealElements = document.querySelectorAll('.reveal');
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const siblings = entry.target.parentElement.querySelectorAll('.reveal');
        const idx = Array.from(siblings).indexOf(entry.target);
        setTimeout(() => entry.target.classList.add('visible'), idx * 100);
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });
  revealElements.forEach(el => revealObserver.observe(el));


  // ==============================
  // NAVBAR
  // ==============================
  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.pageYOffset > 60);
  }, { passive: true });


  // ==============================
  // MOBILE NAV
  // ==============================
  const navToggle = document.getElementById('navToggle');
  const navLinks = document.getElementById('navLinks');

  navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
    const spans = navToggle.querySelectorAll('span');
    if (navLinks.classList.contains('open')) {
      spans[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
      spans[1].style.opacity = '0';
      spans[2].style.transform = 'rotate(-45deg) translate(5px, -5px)';
    } else {
      spans.forEach(s => { s.style.transform = ''; s.style.opacity = ''; });
    }
  });

  navLinks.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('open');
      navToggle.querySelectorAll('span').forEach(s => { s.style.transform = ''; s.style.opacity = ''; });
    });
  });


  // ==============================
  // ANIMATED COUNTERS
  // ==============================
  const statNumbers = document.querySelectorAll('.stat-number[data-target]');
  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        counterObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });
  statNumbers.forEach(el => counterObserver.observe(el));

  function animateCounter(el) {
    const target = parseInt(el.dataset.target);
    const prefix = el.dataset.prefix || '';
    const suffix = el.dataset.suffix || '';
    const duration = 2000;
    const start = performance.now();
    function update(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = prefix + Math.round(eased * target).toLocaleString() + suffix;
      if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
  }


  // ==============================
  // FAQ ACCORDION
  // ==============================
  document.querySelectorAll('.faq-item').forEach(item => {
    item.querySelector('.faq-question').addEventListener('click', () => {
      const isActive = item.classList.contains('active');
      document.querySelectorAll('.faq-item').forEach(o => o.classList.remove('active'));
      if (!isActive) item.classList.add('active');
    });
  });


  // ==============================
  // SMOOTH SCROLL
  // ==============================
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const id = anchor.getAttribute('href');
      if (id === '#') return;
      const el = document.querySelector(id);
      if (el) {
        e.preventDefault();
        window.scrollTo({ top: el.offsetTop - 80, behavior: 'smooth' });
      }
    });
  });


  // ============================================================
  // ANALYSIS SECTION — File Upload, Settings, API Calls
  // ============================================================

  let currentMode = 'image'; // 'image' or 'video'
  let selectedFile = null;
  const alertSystem = new AudioAlertSystem();

  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const filePreview = document.getElementById('filePreview');
  const previewImage = document.getElementById('previewImage');
  const previewVideo = document.getElementById('previewVideo');
  const removeFileBtn = document.getElementById('removeFile');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const uploadHint = document.getElementById('uploadHint');
  const skipFramesGroup = document.getElementById('skipFramesGroup');
  const alertSettingsGroup = document.getElementById('alertSettingsGroup');
  const resultsPlaceholder = document.getElementById('resultsPlaceholder');
  const resultsContent = document.getElementById('resultsContent');
  const detectionResult = document.getElementById('detectionResult');
  const segmentationResult = document.getElementById('segmentationResult');
  const videoResult = document.getElementById('videoResult');
  const resultTabs = document.querySelector('.result-tabs');
  const resultStats = document.getElementById('resultStats');
  const serverStatus = document.getElementById('serverStatus');
  const serverStatusText = document.getElementById('serverStatusText');

  // Sliders
  const sliders = {
    conf: { input: document.getElementById('confSlider'), display: document.getElementById('confValue') },
    iou: { input: document.getElementById('iouSlider'), display: document.getElementById('iouValue') },
    opacity: { input: document.getElementById('opacitySlider'), display: document.getElementById('opacityValue') },
    skip: { input: document.getElementById('skipSlider'), display: document.getElementById('skipValue') }
  };

  // Bind slider value displays
  Object.values(sliders).forEach(s => {
    s.input.addEventListener('input', () => {
      s.display.textContent = parseFloat(s.input.value).toFixed(s.input.step === '1' ? 0 : 2);
    });
  });

  // Alert setting controls
  const audioAlertToggle = document.getElementById('audioAlertToggle');
  const alertCooldownSlider = document.getElementById('alertCooldownSlider');
  const alertCooldownValue = document.getElementById('alertCooldownValue');
  const alertVolumeSlider = document.getElementById('alertVolumeSlider');
  const alertVolumeValue = document.getElementById('alertVolumeValue');

  audioAlertToggle.addEventListener('change', () => {
    alertSystem.setEnabled(audioAlertToggle.checked);
  });
  alertCooldownSlider.addEventListener('input', () => {
    alertCooldownValue.textContent = alertCooldownSlider.value + 's';
  });
  alertVolumeSlider.addEventListener('input', () => {
    const vol = parseInt(alertVolumeSlider.value);
    alertVolumeValue.textContent = vol + '%';
    alertSystem.setVolume(vol / 100);
  });


  // --- Mode Tabs ---
  document.querySelectorAll('.analysis-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.analysis-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentMode = tab.dataset.mode;

      if (currentMode === 'video') {
        fileInput.accept = 'video/*';
        uploadHint.textContent = 'Supports MP4, AVI';
        skipFramesGroup.style.display = '';
        alertSettingsGroup.style.display = '';
        analyzeBtn.querySelector('.btn-analyze-text').textContent = '🎬 Analyze Video';
      } else {
        fileInput.accept = 'image/*';
        uploadHint.textContent = 'Supports JPG, PNG, JPEG';
        skipFramesGroup.style.display = 'none';
        alertSettingsGroup.style.display = 'none';
        analyzeBtn.querySelector('.btn-analyze-text').textContent = '🚀 Analyze Image';
      }

      // Clear current file
      clearFile();
    });
  });

  // Capability card links switch tab
  document.querySelectorAll('.card-link[data-tab]').forEach(link => {
    link.addEventListener('click', () => {
      const tab = link.dataset.tab;
      document.querySelector(`.analysis-tab[data-mode="${tab}"]`)?.click();
    });
  });


  // --- Result Tabs ---
  document.querySelectorAll('.result-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.result-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      if (tab.dataset.result === 'detection') {
        detectionResult.style.display = '';
        detectionResult.classList.add('active');
        segmentationResult.style.display = 'none';
        segmentationResult.classList.remove('active');
      } else {
        detectionResult.style.display = 'none';
        detectionResult.classList.remove('active');
        segmentationResult.style.display = '';
        segmentationResult.classList.add('active');
      }
    });
  });


  // --- Upload Zone ---
  uploadZone.addEventListener('click', () => fileInput.click());

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  removeFileBtn.addEventListener('click', clearFile);


  function handleFile(file) {
    selectedFile = file;

    // Show preview
    uploadZone.style.display = 'none';
    filePreview.style.display = 'flex';

    if (file.type.startsWith('image/')) {
      previewImage.style.display = '';
      previewVideo.style.display = 'none';
      const reader = new FileReader();
      reader.onload = (e) => { previewImage.src = e.target.result; };
      reader.readAsDataURL(file);
    } else if (file.type.startsWith('video/')) {
      previewImage.style.display = 'none';
      previewVideo.style.display = '';
      previewVideo.src = URL.createObjectURL(file);
    }

    analyzeBtn.disabled = false;
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    uploadZone.style.display = '';
    filePreview.style.display = 'none';
    previewImage.src = '';
    previewVideo.src = '';
    analyzeBtn.disabled = true;
    resultsContent.style.display = 'none';
    resultsPlaceholder.style.display = '';
    alertSystem.stop();

    // Reset video player
    videoResult.pause();
    videoResult.removeAttribute('src');
    videoResult.style.display = 'none';
    detectionResult.style.display = '';
    if (resultTabs) resultTabs.style.display = '';
  }


  // --- Analyze Button ---
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Show loading state
    const textEl = analyzeBtn.querySelector('.btn-analyze-text');
    const loadEl = analyzeBtn.querySelector('.btn-analyze-loading');
    textEl.style.display = 'none';
    loadEl.style.display = '';
    analyzeBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append('confidence', sliders.conf.input.value);
      formData.append('iou', sliders.iou.input.value);
      formData.append('opacity', sliders.opacity.input.value);
      formData.append('deep_scan', document.getElementById('deepScanToggle').checked ? 'true' : 'false');

      let endpoint, fileKey;
      if (currentMode === 'image') {
        endpoint = '/api/analyze-image';
        fileKey = 'image';
      } else {
        endpoint = '/api/analyze-video';
        fileKey = 'video';
        formData.append('skip_frames', sliders.skip.input.value);
        formData.append('alert_cooldown', alertCooldownSlider.value);
      }
      formData.append(fileKey, selectedFile);

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Server error');
      }

      const data = await response.json();
      displayResults(data);

    } catch (error) {
      alert('Analysis failed: ' + error.message + '\n\nMake sure the server is running: python website/server.py');
    } finally {
      textEl.style.display = '';
      loadEl.style.display = 'none';
      analyzeBtn.disabled = false;
    }
  });


  function displayResults(data) {
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = '';

    // Video playback mode
    if (data.video_id) {
      // Hide image results and tabs, show video player
      detectionResult.style.display = 'none';
      segmentationResult.style.display = 'none';
      if (resultTabs) resultTabs.style.display = 'none';

      videoResult.style.display = '';
      videoResult.src = '/api/video/' + data.video_id;
      videoResult.load();
      videoResult.play().catch(() => { }); // autoplay may be blocked

      // Sync beep alerts to video playback
      if (data.stats && data.stats.alert_timestamps && data.stats.alert_timestamps.length > 0) {
        alertSystem.syncToVideo(videoResult, data.stats.alert_timestamps);
      }
    } else {
      // Image results
      videoResult.style.display = 'none';
      detectionResult.style.display = '';
      if (resultTabs) resultTabs.style.display = '';

      if (data.detection_image) {
        detectionResult.src = 'data:image/jpeg;base64,' + data.detection_image;
      }
      if (data.segmentation_image) {
        segmentationResult.src = 'data:image/jpeg;base64,' + data.segmentation_image;
      } else {
        segmentationResult.src = '';
      }
    }

    // Stats
    const stats = data.stats;
    let statsHTML = '<div class="stats-cards">';

    // Deep Scan badge
    if (stats.deep_scan) {
      statsHTML += `
        <div class="mini-stat deep-scan-badge">
          <div class="mini-stat-value">🔬</div>
          <div class="mini-stat-label">Deep Scan — Enhanced Models</div>
        </div>
      `;
    }

    // Models used badge
    if (stats.models_used) {
      statsHTML += `
        <div class="mini-stat">
          <div class="mini-stat-value">🧠</div>
          <div class="mini-stat-label">${stats.models_used}</div>
        </div>
      `;
    }

    if (currentMode === 'image') {
      statsHTML += `
        <div class="mini-stat">
          <div class="mini-stat-value">${stats.polyps_found}</div>
          <div class="mini-stat-label">Polyps Found</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-value">${stats.processing_time}s</div>
          <div class="mini-stat-label">Processing Time</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-value">${stats.device?.toUpperCase()}</div>
          <div class="mini-stat-label">Device</div>
        </div>
      `;
      if (stats.mask_coverage !== undefined) {
        statsHTML += `
          <div class="mini-stat">
            <div class="mini-stat-value">${stats.mask_coverage}%</div>
            <div class="mini-stat-label">Mask Coverage</div>
          </div>
        `;
      }

      // Detection details
      if (stats.detections && stats.detections.length > 0) {
        statsHTML += '</div><div class="detection-details"><h4>Detection Details</h4><ul>';
        stats.detections.forEach((d, i) => {
          statsHTML += `<li>Polyp #${i + 1}: <strong>${(d.confidence * 100).toFixed(0)}%</strong> confidence at [${d.x1}, ${d.y1}, ${d.x2}, ${d.y2}]</li>`;
        });
        statsHTML += '</ul></div>';
      } else {
        statsHTML += '</div>';
      }
    } else {
      // Video stats
      statsHTML += `
        <div class="mini-stat">
          <div class="mini-stat-value">${stats.total_frames}</div>
          <div class="mini-stat-label">Frames Processed</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-value">${stats.frames_with_polyps}</div>
          <div class="mini-stat-label">Frames with Polyps</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-value">${stats.detection_rate}%</div>
          <div class="mini-stat-label">Detection Rate</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-value">${stats.processing_time}s</div>
          <div class="mini-stat-label">Processing Time</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-value">${stats.fps_achieved} FPS</div>
          <div class="mini-stat-label">Speed</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-value">${stats.avg_confidence}%</div>
          <div class="mini-stat-label">Avg Confidence</div>
        </div>
      `;
      // Alert count stat
      if (stats.alert_count !== undefined && stats.alert_count > 0) {
        statsHTML += `
          <div class="mini-stat alert-stat">
            <div class="mini-stat-value">🔔 ${stats.alert_count}</div>
            <div class="mini-stat-label">Alerts Triggered</div>
          </div>
        `;
      }
      statsHTML += '</div>';
    }

    resultStats.innerHTML = statsHTML;

    // Switch to detection tab
    document.querySelector('.result-tab[data-result="detection"]')?.click();
  }


  // --- Server Health Check ---
  async function checkServer() {
    try {
      const res = await fetch('/api/health', { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        const indicator = serverStatus.querySelector('.status-indicator');
        indicator.classList.remove('offline');
        indicator.classList.add('online');
        const modelsCount = (data.yolo_loaded ? 1 : 0) + (data.unet_loaded ? 1 : 0) +
          (data.yolo_small_polyp_loaded ? 1 : 0) + (data.unet_upgraded_loaded ? 1 : 0);
        const deepScanText = data.deep_scan_available ? ' | Deep Scan ✓' : '';
        serverStatusText.textContent = `Server online — ${data.device?.toUpperCase()} | ${modelsCount} models loaded${deepScanText}`;
        serverStatus.classList.add('connected');
      }
    } catch {
      const indicator = serverStatus.querySelector('.status-indicator');
      indicator.classList.remove('online');
      indicator.classList.add('offline');
      serverStatusText.textContent = 'Server offline — Run "python website/server.py" to enable AI analysis';
      serverStatus.classList.remove('connected');
    }
  }

  // Check on load and every 10s
  checkServer();
  setInterval(checkServer, 10000);

});
