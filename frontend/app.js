/**
 * AI Animal Guard — Multi-Camera Dashboard (Sprint 4)
 * ======================================================
 * - Fetches /cameras on load to discover active cameras
 * - Builds a camera tile per camera dynamically
 * - Opens one WebSocket per camera (/ws/{cam_id})
 * - Streams MJPEG per camera (/stream/{cam_id})
 * - Global status bar reflects the "worst" status across all cameras
 * - Alert toast fires for any camera alert
 * - Per-camera state: status, detections, fps
 * - Auto-reconnects WebSockets with exponential backoff
 */

'use strict';

// ── Global state ──────────────────────────────────────────────────────────────
const appState = {
  cameras: [],        // [{id, label, index, ...}]
  channels: {},       // {cam_id: {ws, reconnectDelay, lastEvent, status}}
  config: null,
  toastTimeout: null,
  alertHistory: [],
  overallStatus: 'clear',
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const els = {
  connBadge:    $('conn-badge'),
  connLabel:    $('conn-label'),
  deviceLabel:  $('device-label'),
  statusBar:    $('status-bar'),
  statusIndicator: $('status-indicator'),
  statusIcon:   $('status-icon'),
  statusLabel:  $('status-label'),
  statusSub:    $('status-sub'),
  statAiFps:    $('stat-ai-fps'),
  statCamFps:   $('stat-cam-fps'),
  confirmFill:  $('confirm-fill'),
  confirmRatio: $('confirm-ratio'),
  cooldownPill: $('cooldown-pill'),
  statCooldown: $('stat-cooldown'),
  historyList:  $('history-list'),
  alertToast:   $('alert-toast'),
  toastTitle:   $('toast-title'),
  toastBody:    $('toast-body'),
  sysDevice:    $('sys-device'),
  sysModel:     $('sys-model'),
  sysTargets:   $('sys-targets'),
  sysLatency:   $('sys-latency'),
  sysCamera:    $('sys-camera'),
  cameraGrid:   $('camera-grid'),
  camCountLabel: $('cam-count-label'),
};

// ── Status config map ─────────────────────────────────────────────────────────
const STATUS_MAP = {
  clear: {
    barClass: 'status-clear', indClass: 'ind-clear', lblClass: 'lbl-clear',
    icon: '🟢', label: 'ALL CLEAR', sub: 'No threats detected on any camera',
    fillColor: 'var(--clr-clear)',
  },
  evaluating: {
    barClass: 'status-evaluating', indClass: 'ind-evaluating', lblClass: 'lbl-evaluating',
    icon: '🟡', label: 'EVALUATING', sub: 'Possible threat — confirming…',
    fillColor: 'var(--clr-eval)',
  },
  alert: {
    barClass: 'status-alert', indClass: 'ind-alert', lblClass: 'lbl-alert',
    icon: '🔴', label: '⚠ ALERT', sub: 'Threat confirmed! Alert fired.',
    fillColor: 'var(--clr-alert)',
  },
  cooldown: {
    barClass: 'status-cooldown', indClass: 'ind-cooldown', lblClass: 'lbl-cooldown',
    icon: '⏳', label: 'COOLDOWN', sub: 'Alert fired — cooling down…',
    fillColor: 'var(--clr-cooldown)',
  },
  error: {
    barClass: 'status-alert', indClass: 'ind-alert', lblClass: 'lbl-alert',
    icon: '✖', label: 'ERROR', sub: 'Camera disconnected or failed.',
    fillColor: 'var(--clr-alert)',
  },
  starting: {
    barClass: 'status-clear', indClass: 'ind-clear', lblClass: 'lbl-clear',
    icon: '⏳', label: 'STARTING', sub: 'Initializing cameras…',
    fillColor: 'var(--clr-accent)',
  },
};

const STATUS_PRIORITY = ['alert', 'error', 'evaluating', 'cooldown', 'starting', 'clear'];

// ── Compute overall status from all channels ──────────────────────────────────
function computeOverallStatus() {
  const statuses = Object.values(appState.channels).map(c => c.status || 'clear');
  for (const s of STATUS_PRIORITY) {
    if (statuses.includes(s)) return s;
  }
  return 'clear';
}

// ── Build camera tiles ────────────────────────────────────────────────────────
function buildCameraTiles(cameras) {
  const grid = els.cameraGrid;
  grid.innerHTML = '';

  if (cameras.length > 1) grid.classList.add('multi');
  else grid.classList.remove('multi');

  for (const cam of cameras) {
    const tile = document.createElement('div');
    tile.className = 'camera-tile card';
    tile.id = `tile-${cam.id}`;
    tile.innerHTML = `
      <div class="tile-header">
        <span class="tile-label">${cam.label}</span>
        <span class="tile-status-badge badge-starting" id="badge-${cam.id}">Starting…</span>
      </div>
      <div class="tile-video-wrap">
        <img class="tile-video-feed" id="feed-${cam.id}"
             src="/stream/${cam.id}"
             alt="Live feed: ${cam.label}"
             onerror="handleVideoError('${cam.id}')" />
        <div class="tile-live-badge">
          <div class="live-dot"></div>LIVE
        </div>
      </div>
      <div class="tile-detections" id="dets-${cam.id}">
        <div class="tile-no-det">Waiting for detections…</div>
      </div>
      <div class="tile-footer">
        <span class="tile-fps" id="fps-${cam.id}">AI: — · CAM: —</span>
        <span class="card-badge txt-mono" id="detcount-${cam.id}">0 objects</span>
      </div>`;
    grid.appendChild(tile);
  }

  // Update camera count badge
  const active = cameras.filter(c => c.camera_ok || c.is_running).length;
  els.camCountLabel.textContent = `${cameras.length} camera${cameras.length !== 1 ? 's' : ''}`;
  els.sysCamera.textContent = `${active}/${cameras.length} active`;
}

// ── Update a camera tile from a DetectionEvent ────────────────────────────────
function updateTile(camId, evt) {
  const tile   = $(`tile-${camId}`);
  const badge  = $(`badge-${camId}`);
  const detsEl = $(`dets-${camId}`);
  const fpsEl  = $(`fps-${camId}`);
  const cntEl  = $(`detcount-${camId}`);
  if (!tile) return;

  const status = evt.status || 'clear';

  // Tile border class
  tile.className = `camera-tile card tile-${status}`;

  // Status badge
  badge.className = `tile-status-badge badge-${status}`;
  const BADGE_LABELS = {
    clear: 'Clear', evaluating: 'Evaluating', alert: 'ALERT!',
    cooldown: 'Cooldown', error: 'Error', starting: 'Starting…',
  };
  badge.textContent = BADGE_LABELS[status] || status;

  // FPS
  const aiFps = evt.inference_ms > 0 ? (1000 / evt.inference_ms).toFixed(0) : '—';
  if (fpsEl) fpsEl.textContent = `AI: ${aiFps}fps · CAM: ${(evt.capture_fps || 0).toFixed(0)}fps`;

  // Detection count
  const count = (evt.detections || []).length;
  if (cntEl) cntEl.textContent = `${count} object${count !== 1 ? 's' : ''}`;

  // Detections list
  if (detsEl) {
    if (!count) {
      detsEl.innerHTML = `<div class="tile-no-det">No objects in frame</div>`;
    } else {
      detsEl.innerHTML = evt.detections.map(d => {
        const confPct = Math.round(d.confidence * 100);
        const lblClass = `lbl-${d.label}`;
        return `
          <div class="tile-det-row">
            <span class="tile-det-name">${d.class_name}</span>
            <span class="tile-det-conf">${confPct}%</span>
            <span class="tile-det-label ${lblClass}">${d.label}</span>
          </div>`;
      }).join('');
    }
  }
}

// ── Update global status bar ──────────────────────────────────────────────────
function updateGlobalStatusBar() {
  const overall = computeOverallStatus();
  appState.overallStatus = overall;
  const st = STATUS_MAP[overall] || STATUS_MAP.clear;

  els.statusBar.className = `status-bar ${st.barClass}`;
  els.statusIndicator.className = `status-indicator ${st.indClass}`;
  els.statusLabel.className = `status-label ${st.lblClass}`;
  els.statusIcon.textContent = st.icon;
  els.statusLabel.textContent = st.label;
  els.statusSub.textContent = st.sub;

  // Find the "loudest" channel event for stats display
  const events = Object.values(appState.channels)
    .map(c => c.lastEvent)
    .filter(Boolean);

  if (!events.length) return;

  // Pick the event from the alerting/evaluating camera, or first
  const priorityEvt = events.find(e => e.status === 'alert')
    || events.find(e => e.status === 'evaluating')
    || events[0];

  const aiFps = priorityEvt.inference_ms > 0
    ? (1000 / priorityEvt.inference_ms).toFixed(1) : '—';
  els.statAiFps.textContent = aiFps;
  els.statCamFps.textContent = priorityEvt.capture_fps > 0
    ? `${priorityEvt.capture_fps.toFixed(0)}` : '—';

  const [hits, total] = (priorityEvt.confirmation_ratio || '0/5').split('/').map(Number);
  const pct = total > 0 ? (hits / total) * 100 : 0;
  els.confirmFill.style.width = `${pct}%`;
  els.confirmFill.style.background = st.fillColor;
  els.confirmRatio.textContent = priorityEvt.confirmation_ratio;

  if (overall === 'cooldown') {
    els.cooldownPill.style.display = 'flex';
    els.statCooldown.textContent = `${priorityEvt.cooldown_remaining.toFixed(0)}s`;
  } else {
    els.cooldownPill.style.display = 'none';
  }
}

// ── Handle detection event from a specific camera ─────────────────────────────
function handleDetectionEvent(camId, evt) {
  const ch = appState.channels[camId];
  if (!ch) return;

  ch.lastEvent = evt;
  ch.status    = evt.status;

  updateTile(camId, evt);
  updateGlobalStatusBar();

  // Update latency in system panel with most recent inference
  if (evt.inference_ms) {
    els.sysLatency.textContent = `${evt.inference_ms.toFixed(0)}ms`;
  }

  if (evt.alert_fired) {
    fetchAllAlerts();
    const label = appState.cameras.find(c => c.id === camId)?.label || camId;
    showToast(evt.alert_class, evt.alert_confidence, label);
  }
}

// ── WebSocket management ──────────────────────────────────────────────────────
function connectCamera(cam) {
  const camId = cam.id;
  if (!appState.channels[camId]) {
    appState.channels[camId] = { ws: null, reconnectDelay: 1000, lastEvent: null, status: 'starting' };
  }
  const ch = appState.channels[camId];

  const wsUrl = `ws://${location.host}/ws/${camId}`;
  const ws = new WebSocket(wsUrl);
  ch.ws = ws;

  ws.addEventListener('open', () => {
    ch.reconnectDelay = 1000;
    updateConnectedCount();
    console.info(`[WS:${camId}] Connected`);
  });

  ws.addEventListener('message', evt => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === 'detection') handleDetectionEvent(camId, msg);
    } catch (e) {
      console.warn(`[WS:${camId}] Bad message:`, evt.data);
    }
  });

  ws.addEventListener('close', () => {
    updateConnectedCount();
    console.warn(`[WS:${camId}] Disconnected. Retry in ${ch.reconnectDelay}ms`);
    setTimeout(() => connectCamera(cam), ch.reconnectDelay);
    ch.reconnectDelay = Math.min(ch.reconnectDelay * 1.5, 15000);
  });

  ws.addEventListener('error', () => ws.close());
}

function updateConnectedCount() {
  const connected = Object.values(appState.channels)
    .filter(c => c.ws && c.ws.readyState === WebSocket.OPEN).length;
  const total = appState.cameras.length;
  const ok = connected === total && total > 0;
  els.connBadge.className = `conn-badge ${ok ? 'connected' : 'disconnected'}`;
  els.connLabel.textContent = ok ? 'Live' : connected > 0 ? `${connected}/${total} live` : 'Reconnecting…';
}

// ── Fetch cameras from server ─────────────────────────────────────────────────
async function fetchCameras() {
  try {
    const res = await fetch('/cameras');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cameras = await res.json();
    appState.cameras = cameras;
    buildCameraTiles(cameras);

    // Open one WebSocket per camera
    for (const cam of cameras) {
      connectCamera(cam);
    }
    console.info(`[cameras] ${cameras.length} camera(s) found.`);
  } catch (e) {
    console.error('[cameras] Failed to fetch camera list:', e);
    els.cameraGrid.innerHTML = `
      <div class="camera-loading">
        <div style="font-size:2rem">⚠️</div>
        <span style="color:#ff6060">Cannot connect to server.<br/>Is the backend running?</span>
      </div>`;
  }
}

// ── Alert history ─────────────────────────────────────────────────────────────
async function fetchAllAlerts() {
  try {
    // Fetch alerts for first camera (or aggregate in future)
    const res = await fetch('/alerts');
    const data = await res.json();
    appState.alertHistory = data;
    renderAlerts(data);
  } catch (e) {
    console.warn('[alerts] fetch failed:', e);
  }
}

function renderAlerts(alerts) {
  if (!alerts.length) {
    els.historyList.innerHTML = `
      <div class="no-alerts">
        <div class="no-alerts-icon">🛡️</div>
        <div>No alerts fired this session</div>
      </div>`;
    return;
  }

  const items = alerts.slice(0, 50).map(a => {
    const ts = formatTimestamp(a.timestamp);
    const confPct = Math.round(a.confidence * 100);
    const thumb = a.screenshot_url
      ? `<img class="alert-thumb" src="${a.screenshot_url}" alt="${a.class_name}" loading="lazy" />`
      : `<div class="alert-thumb-placeholder">📸</div>`;
    return `
      <div class="alert-item">
        ${thumb}
        <div class="alert-info">
          <div class="alert-class">${a.class_name}</div>
          <div class="alert-meta">${ts}</div>
        </div>
        <div class="alert-conf">${confPct}%</div>
      </div>`;
  }).join('');

  els.historyList.innerHTML = items;
}

function formatTimestamp(ts) {
  if (!ts || ts.length < 8) return ts;
  try {
    const y = ts.slice(0,4), mo = ts.slice(4,6), d = ts.slice(6,8);
    const h = ts.slice(9,11), mi = ts.slice(11,13), s = ts.slice(13,15);
    return `${y}-${mo}-${d} ${h}:${mi}:${s}`;
  } catch {
    return ts;
  }
}

async function clearAlerts() {
  try {
    await fetch('/alerts', { method: 'DELETE' });
    appState.alertHistory = [];
    renderAlerts([]);
  } catch (e) {
    console.warn('[alerts] clear failed:', e);
  }
}
window.clearAlerts = clearAlerts;

// ── Toast notification ────────────────────────────────────────────────────────
function showToast(className, confidence, cameraLabel) {
  if (appState.toastTimeout) clearTimeout(appState.toastTimeout);
  const confPct = confidence ? Math.round(confidence * 100) : 0;
  els.toastTitle.textContent = `🚨 ${(className || 'ANIMAL').toUpperCase()} DETECTED`;
  els.toastBody.textContent = `Camera: ${cameraLabel || '?'} · Confidence: ${confPct}% · Alert fired`;
  els.alertToast.classList.add('show');
  appState.toastTimeout = setTimeout(() => {
    els.alertToast.classList.remove('show');
  }, 6000);
}

// ── Config & system info ──────────────────────────────────────────────────────
async function fetchConfig() {
  try {
    const res = await fetch('/config');
    const cfg = await res.json();
    appState.config = cfg;
    els.sysModel.textContent = cfg.model?.name || '—';
    els.sysTargets.textContent = (cfg.classes?.target_classes || []).join(', ') || '—';

    // Populate settings form
    if ($('input-targets')) {
      $('input-targets').value = (cfg.classes?.target_classes || []).join(', ');
      $('input-pets').value    = (cfg.classes?.pet_classes || []).join(', ');
      $('input-cooldown').value = cfg.alert?.cooldown_seconds || 30;
      $('input-sound').checked  = cfg.alert?.sound_enabled ?? true;
    }
  } catch (e) {
    console.warn('[config] fetch failed:', e);
  }
}

async function saveSettings() {
  const btn = $('btn-save-settings');
  const oldText = btn.textContent;
  btn.textContent = 'Saving...';
  btn.disabled = true;

  try {
    const targets  = $('input-targets').value.split(',').map(s => s.trim()).filter(Boolean);
    const pets     = $('input-pets').value.split(',').map(s => s.trim()).filter(Boolean);
    const cooldown = parseInt($('input-cooldown').value, 10);
    const sound    = $('input-sound').checked;

    const res = await fetch('/config', {
      method:  'PUT',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        target_classes:   targets,
        pet_classes:      pets,
        cooldown_seconds: isNaN(cooldown) ? 30 : cooldown,
        sound_enabled:    sound,
      }),
    });

    if (res.ok) {
      await fetchConfig();
      btn.textContent = 'Saved!';
      btn.style.color = 'var(--clr-clear)';
      btn.style.borderColor = 'var(--clr-clear)';
    } else {
      console.warn('[config] save failed', await res.text());
      btn.textContent = 'Error';
    }
  } catch (e) {
    console.warn('[config] save error:', e);
    btn.textContent = 'Error';
  }

  setTimeout(() => {
    btn.textContent = oldText;
    btn.disabled = false;
    btn.style.color = '';
    btn.style.borderColor = '';
  }, 2000);
}
window.saveSettings = saveSettings;

// ── Fetch status for device info ──────────────────────────────────────────────
async function fetchStatus() {
  try {
    const res = await fetch('/status');
    const data = await res.json();
    if (data.cameras && data.cameras.length) {
      const first = data.cameras[0];
      els.sysDevice.textContent = first.device || '—';
      els.deviceLabel.textContent = first.device || '—';
      els.sysCamera.textContent = `${data.cameras.filter(c => c.camera_ok).length}/${data.camera_count} active`;
    }
  } catch (e) {
    console.warn('[status] fetch failed:', e);
  }
}

// ── Video error handling ──────────────────────────────────────────────────────
function handleVideoError(camId) {
  console.warn(`[video:${camId}] Stream failed, retrying…`);
  setTimeout(() => {
    const feed = $(`feed-${camId}`);
    if (feed) feed.src = `/stream/${camId}?t=${Date.now()}`;
  }, 2000);
}
window.handleVideoError = handleVideoError;

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
  await Promise.all([fetchConfig(), fetchStatus(), fetchAllAlerts()]);
  await fetchCameras();

  // Periodic refresh
  setInterval(fetchAllAlerts, 30_000);
  setInterval(fetchStatus, 10_000);
})();
