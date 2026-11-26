// CrowdPulse dashboard: asks the local server for the latest state about twice a second and draws it.
"use strict";

const $ = (id) => document.getElementById(id);
const LEVELS = ["green", "amber", "red"];
const LINE_COLOURS = ["#1d4ed8", "#7c3aed", "#0e7490", "#be185d", "#4d7c0f", "#c2410c"];
const POLL_MS = 600;

const SIMS = ["sim:stadium", "sim:stadium-thermal"];

let sources = [];
let lastFrame = 0;
let lastAlert = 0;
let alertCount = 0;
let audio = null;
let simShown = "";
let synced = false;

function clock(seconds) {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const s = Math.floor(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

// ---- picking what to watch

async function loadSources() {
  const data = await api("/api/sources");
  sources = data.sources;
  const source = $("source");
  source.replaceChildren(...sources.map((s) => { const o = el("option", "", s.label); o.value = s.id; return o; }));
  const zones = $("zones");
  const none = el("option", "", "None (head count only)");
  none.value = "";
  zones.replaceChildren(none, ...data.zones.map((z) => { const o = el("option", "", z); o.value = z; return o; }));
  source.value = sources[0].id;                     // the simulation: the one source that shows warnings
  matchZones();
}

function matchZones() {
  const chosen = sources.find((s) => s.id === $("source").value);
  $("zones").value = (chosen && chosen.zones) || "";
  const isSim = SIMS.includes($("source").value);   // the simulation has its own zones, camera and timing
  $("zones").disabled = isSim;
  $("every").disabled = isSim;
  $("camera").disabled = isSim;
}

// ---- simulation controls

function renderSim(info) {
  const card = $("sim-card");
  card.hidden = !info;
  if (!info) { simShown = ""; return; }
  const key = JSON.stringify([info.gates, info.holding, info.stewards]);
  if (key !== simShown) {                           // rebuild buttons only when something changed
    simShown = key;
    $("gate-buttons").replaceChildren(...info.gates.map((g) => {
      const b = el("button", g.open ? "gate-open" : "gate-closed", `${g.name}: ${g.open ? "open" : "closed"}`);
      b.type = "button";
      b.title = g.open ? `Close ${g.name}` : `Open ${g.name}`;
      b.addEventListener("click", () => act(g.open ? "close_gate" : "open_gate", g.number));
      return b;
    }));
    $("hold").setAttribute("aria-pressed", String(info.holding));
    $("hold").textContent = info.holding ? "Release arrivals" : "Hold arrivals at the cordon";
    $("stewards").setAttribute("aria-pressed", String(info.stewards));
    $("stewards").textContent = info.stewards ? "Stewards redirecting (stop)" : "Send stewards to redirect";
  }
  const rows = [["Arriving now", `${info.arrivals_per_min} a minute`], ["On the plaza", info.on_plaza],
    ["Through the gates", info.entered], ["Held at the cordon", info.held]];
  $("sim-stats").replaceChildren(...rows.flatMap(([k, v]) => [el("dt", "", k), el("dd", "", String(v))]));
}

function act(action, gate) {
  api("/api/action", { action, gate: gate ?? null }).catch((err) => { $("status").textContent = err.message; });
}

// ---- sound and voice for alerts

function unlockAudio() {
  if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
  if (audio.state === "suspended") audio.resume();
}

function beep(level) {
  if (!$("sound").checked || !audio) return;
  const pattern = level === "red" ? [880, 0, 880, 0, 880] : [660];
  let at = audio.currentTime;
  for (const freq of pattern) {
    if (freq) {
      const osc = audio.createOscillator();
      const gain = audio.createGain();
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, at);
      gain.gain.exponentialRampToValueAtTime(0.25, at + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.22);
      osc.connect(gain).connect(audio.destination);
      osc.start(at);
      osc.stop(at + 0.25);
    }
    at += 0.15;
  }
}

function speak(alert) {
  if (!$("voice").checked || !window.speechSynthesis) return;
  speechSynthesis.speak(new SpeechSynthesisUtterance(`${alert.message} ${alert.action}`));
}

// ---- drawing the state

function renderZones(record) {
  const list = $("zone-list");
  if (!record || !record.zones.length) {
    const p = el("p", "muted", record ? `About ${Math.round(record.people)} people in view. Add a zones file to get warnings per area.` :
      "Zones appear here once the analysis starts.");
    list.replaceChildren(p);
    return;
  }
  list.replaceChildren(...record.zones.map((z) => {
    const card = el("div", `zone ${z.level}`);
    card.append(el("div", "name", z.name), el("span", `pill ${z.level}`, z.level.toUpperCase()));
    const big = el("div", "big", z.density.toFixed(1));
    big.append(el("small", "", "people/m²"));
    card.append(big, el("div", ""));
    const meta = el("div", "meta");
    const flow = z.inflow_per_min >= 1 ? `+${Math.round(z.inflow_per_min)} people/min walking in` :
      z.inflow_per_min <= -1 ? `${Math.round(-z.inflow_per_min)} people/min leaving` : "steady";
    meta.append(`${Math.round(z.people)} people · ${flow}`);
    if (z.in_30s !== null && z.in_30s !== undefined) {
      const later = `in 30 s: ${z.in_30s.toFixed(1)}/m²`;
      const worse = levelOf(z.in_30s) > levelOf(z.density);      // about to cross into amber or red
      meta.append(" · ", worse ? el("b", "", `${later}, ${LEVELS[levelOf(z.in_30s)]}`) : later);
    }
    if (z.minutes_to_critical !== null) {
      meta.append(" · ");
      const m = z.minutes_to_critical;
      meta.append(el("b", "", m === 0 ? "crush-risk density now" : m < 1 ? "critical in under a minute" : `critical in about ${Math.round(m)} min`));
    }
    card.append(meta);
    return card;
  }));
}

function renderAlerts(alerts) {
  if (!alerts.length) return;
  const list = $("alerts");
  if (alertCount === 0) list.replaceChildren();
  for (const a of alerts) {
    const item = el("li", `alert ${a.level}`);
    item.append(el("div", "when", `${clock(a.t)} · ${a.zone} · ${a.level.toUpperCase()}`), el("div", "what", a.message), el("div", "do", a.action));
    list.prepend(item);
    alertCount += 1;
    if (a.level !== "green") { beep(a.level); speak(a); }
  }
  $("alert-count").textContent = `(${alertCount})`;
}

const levelOf = (density) => (density >= 4 ? 2 : density >= 2 ? 1 : 0);   // index into LEVELS, as in risk.py

function renderChart(history, record) {
  const canvas = $("chart");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = 220;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const names = history.length ? Object.keys(history[history.length - 1].zones) : [];
  const legend = $("legend");
  legend.replaceChildren(...names.map((n, i) => { const s = el("span", "", n); s.style.setProperty("--c", LINE_COLOURS[i % LINE_COLOURS.length]); return s; }));

  const left = 34, right = 8, top = 8, bottom = 24;
  const w = width - left - right, h = height - top - bottom;
  const ahead = {};                                 // each zone's forecast density in 30 s, if there is one
  for (const z of record ? record.zones : []) if (z.in_30s !== null && z.in_30s !== undefined) ahead[z.name] = z.in_30s;
  const peak = Math.max(6, ...history.flatMap((r) => Object.values(r.zones)), ...Object.values(ahead));
  const y = (d) => top + h - (d / peak) * h;
  // bands: free movement, crowded, dangerous
  for (const [from, to, colour] of [[0, 2, "#e6f4ea"], [2, 4, "#fef3c7"], [4, peak, "#fde8e8"]]) {
    ctx.fillStyle = colour;
    ctx.fillRect(left, y(to), w, y(from) - y(to));
  }
  ctx.fillStyle = "#5b6475"; ctx.font = "12px Segoe UI, system-ui, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let d = 0; d <= peak; d += 2) ctx.fillText(String(d), left - 6, y(d));
  if (history.length < 2) {
    ctx.textAlign = "center";
    ctx.fillText(names.length ? "Collecting readings…" : "The chart fills in as frames are analysed.", left + w / 2, top + h / 2);
    return;
  }
  const t0 = history[0].t, t1 = history[history.length - 1].t;
  const forecastShown = Object.keys(ahead).length > 0;
  const end = forecastShown ? t1 + 30 : t1;
  const x = (t) => left + ((t - t0) / Math.max(end - t0, 1)) * w;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText(clock(t0), left + 14, top + h + 6);
  ctx.fillText(forecastShown ? `now ${clock(t1)}` : clock(t1), forecastShown ? x(t1) : left + w - 14, top + h + 6);
  if (forecastShown) {                              // the next 30 s, shaded, with each zone's forecast dashed
    ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
    ctx.fillRect(x(t1), top, x(end) - x(t1), h);
    ctx.fillStyle = "#5b6475";
    ctx.fillText("+30 s", left + w - 16, top + h + 6);
  }
  names.forEach((name, i) => {
    ctx.strokeStyle = LINE_COLOURS[i % LINE_COLOURS.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((r, j) => { const p = [x(r.t), y(r.zones[name] ?? 0)]; j ? ctx.lineTo(...p) : ctx.moveTo(...p); });
    ctx.stroke();
    if (name in ahead) {
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(x(t1), y(history[history.length - 1].zones[name] ?? 0));
      ctx.lineTo(x(end), y(ahead[name]));
      ctx.stroke();
      ctx.setLineDash([]);
    }
  });
}

function renderStatus(state) {
  const zones = state.record ? state.record.zones : [];
  const worst = zones.reduce((acc, z) => (LEVELS.indexOf(z.level) > LEVELS.indexOf(acc) ? z.level : acc), "green");
  const overall = $("overall");
  if (state.error) {
    overall.className = "pill red"; overall.textContent = "Error";
    $("status").textContent = `Stopped: ${state.error}`;
  } else if (state.running && !state.record) {
    overall.className = "pill"; overall.textContent = "Starting";
    $("status").textContent = `Opening ${state.source} and loading the crowd model…`;
  } else if (state.record) {
    overall.className = zones.length ? `pill ${worst}` : "pill";
    overall.textContent = zones.length ? worst.toUpperCase() : state.running ? "Running" : "Stopped";
    const isSim = SIMS.includes(state.source);
    const speed = state.seconds_per_frame && !isSim ? ` · ${state.seconds_per_frame.toFixed(1)} s per frame on this laptop` : "";
    const where = isSim ? `Simulation${state.source === SIMS[1] ? ", thermal view" : ""}, 5× speed` : state.source === "webcam" ? "Live webcam" : state.source;
    const done = state.finished ? (isSim ? " · scenario finished" : " · end of video") : !state.running ? " · stopped" : "";
    const people = state.source === SIMS[0] ? `${Math.round(state.record.people)} people on the plaza` : `about ${Math.round(state.record.people)} people`;
    $("status").textContent = `${where} · ${clock(state.record.t)} · ${people}${speed}${done}`;
  }
  $("start").disabled = state.running;
  $("stop").disabled = !state.running;
  $("report").disabled = !state.frame_id;
}

async function poll() {
  try {
    const state = await api(`/api/state?since=${lastAlert}`);
    if (!synced && sources.length) {                // after a reload, show what is already running
      synced = true;
      if (state.running && sources.some((s) => s.id === state.source)) {
        $("source").value = state.source;
        matchZones();
        if (state.zones) $("zones").value = state.zones;
      }
    }
    if (state.frame_id && state.frame_id !== lastFrame) {
      lastFrame = state.frame_id;
      $("frame").src = `/frame.jpg?n=${lastFrame}`;
      $("frame").hidden = false;
      $("empty").hidden = true;
    }
    renderStatus(state);
    renderSim(state.running || state.finished ? state.sim : null);
    renderZones(state.record);
    renderAlerts(state.alerts);
    if (state.alerts.length) lastAlert = state.alerts[state.alerts.length - 1].id;
    renderChart(state.history, state.record);
  } catch (err) {
    $("overall").className = "pill red"; $("overall").textContent = "Offline";
    $("status").textContent = "Cannot reach the CrowdPulse server. Is scripts/dashboard.py still running?";
  }
  setTimeout(poll, POLL_MS);
}

// ---- buttons

$("source").addEventListener("change", matchZones);
$("controls").addEventListener("submit", async (event) => {
  event.preventDefault();
  unlockAudio();                                   // browsers only allow sound after a click
  lastFrame = 0; lastAlert = 0; alertCount = 0;
  $("alerts").replaceChildren(el("li", "muted", "No alerts yet."));
  $("alert-count").textContent = "";
  $("frame").hidden = true; $("empty").hidden = false;
  try {
    await api("/api/start", { source: $("source").value, zones: $("zones").value || null, every: Number($("every").value),
                              thermal: $("camera").value === "thermal" });
  } catch (err) {
    $("status").textContent = `Could not start: ${err.message}`;
  }
});
$("stop").addEventListener("click", () => api("/api/stop", {}).catch(() => {}));
$("report").addEventListener("click", () => { window.location.href = "/api/report"; });
$("hold").addEventListener("click", () => act("hold_arrivals"));
$("stewards").addEventListener("click", () => act("stewards"));

loadSources().catch(() => { $("status").textContent = "Cannot reach the CrowdPulse server."; });
poll();
