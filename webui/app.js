/* Video Upscaler — frontend logic. Vanilla JS, no dependencies. */
"use strict";

const TOKEN = new URLSearchParams(location.search).get("token") || "";
const qs = (s) => "?token=" + encodeURIComponent(TOKEN);
const $ = (id) => document.getElementById(id);

const state = { jobs: [], models: [], info: {} };

/* ── API helpers ─────────────────────────────────────────── */
async function api(path, opts = {}) {
  const sep = path.includes("?") ? "&" : "?";
  const url = path + (TOKEN ? sep + "token=" + encodeURIComponent(TOKEN) : "");
  const res = await fetch(url, opts);
  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(res.status + " " + t.slice(0, 200));
  }
  return res.status === 204 ? null : res.json();
}
const post = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify(body || {}) });

/* ── settings ────────────────────────────────────────────── */
function gatherSettings() {
  let target = document.querySelector("#target-seg button.active").dataset.v;
  if (target === "custom") target = $("target-custom").value.trim() || "4k";
  const model = $("model-custom").value.trim() || $("model").value;
  return {
    target,
    container: $("container").value,
    codec: $("codec").value,
    output_dir: $("output").value.trim() || null,
    model,
    qp: $("qp").value,
    tile: parseInt($("tile").value) || 0,
    fp16: $("fp16").checked,
    interpolate: $("interpolate").checked,
    fps: $("interpolate").checked ? parseFloat($("fps").value) : null,
  };
}

/* ── chips / info ────────────────────────────────────────── */
async function loadInfo() {
  const info = await api("/api/info");
  state.info = info;
  setChip("chip-gpu", info.gpu && info.gpu !== "unknown", "GPU: " + info.gpu);
  setChip("chip-enc", /GPU/.test(info.encoder), info.encoder);
  setChip("chip-torch", info.torch, info.torch ? "torch + CUDA ready" : "torch not ready");
}
function setChip(id, good, text) {
  const el = $(id);
  el.className = "chip " + (good ? "ok" : "bad");
  el.innerHTML = '<span class="dot"></span> ' + text;
}
async function loadModels() {
  const { builtins } = await api("/api/models");
  state.models = builtins;
  const sel = $("model");
  sel.innerHTML = "";
  builtins.forEach((m) => {
    const o = document.createElement("option");
    o.value = m.name;
    o.textContent = m.name + " (x" + m.scale + ")" + (m.cached ? " ✓" : "");
    sel.appendChild(o);
  });
  updateModelNote();
}
function updateModelNote() {
  const m = state.models.find((x) => x.name === $("model").value);
  $("model-note").textContent = m ? m.note + (m.cached ? "" : " — downloads on first use") : "";
}

function applyPreset(name) {
  const p = (state.info && state.info.presets || {})[name];
  if (!p) return;
  $("model").value = p.model; updateModelNote();
  $("qp").value = p.qp;
  $("codec").value = p.codec;
  document.querySelectorAll("#preset-seg button").forEach(
    (b) => b.classList.toggle("active", b.dataset.v === name));
  $("preset-note").textContent = p.label;
}

/* ── queue rendering ─────────────────────────────────────── */
function renderQueue() {
  const box = $("queue");
  $("queue-empty").style.display = state.jobs.length ? "none" : "block";
  $("queue-count").textContent = state.jobs.length
    ? state.jobs.length + " job" + (state.jobs.length > 1 ? "s" : "") : "";
  // remove old job nodes (keep the empty placeholder)
  [...box.querySelectorAll(".job")].forEach((n) => n.remove());
  state.jobs.forEach((j) => box.appendChild(jobNode(j)));
}
function fmtEta(s) {
  if (s == null) return "--";
  s = Math.max(0, Math.round(s));
  if (s >= 3600) return `${(s/3600)|0}:${String(((s%3600)/60)|0).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`;
  return `${(s/60)|0}:${String(s%60).padStart(2,"0")}`;
}
function jobNode(j) {
  const el = document.createElement("div");
  el.className = "job " + (j.status === "running" ? "running" : "");
  const pct = Math.round((j.progress || 0) * 100);
  const dim = j.meta && j.meta.width ? `${j.meta.width}×${j.meta.height}` : "";
  const tgt = (j.settings && j.settings.target) || "";
  const running = j.status === "running";
  const showBar = ["running", "paused", "done"].includes(j.status) || pct > 0;
  const indet = running && !j.total_frames;
  el.innerHTML = `
    <div class="title">
      <div style="min-width:0">
        <div class="name" title="${esc(j.src)}">${esc(j.name)}</div>
        <div class="meta">${dim ? dim + " → " + esc(tgt) + " · " : ""}${esc((j.settings&&j.settings.model)||"")}${
          j.error ? ' · <span style="color:var(--danger)">' + esc(j.error) + "</span>" : ""}</div>
      </div>
    </div>
    <div class="actions">
      <span class="status ${j.status}">${j.status}</span>
      ${ctlButtons(j)}
    </div>
    ${showBar ? `<div class="progress">
      <div class="pbar ${indet ? "indet" : ""}"><span style="width:${pct}%"></span></div>
      <div class="pline">
        <span>${running ? `seg ${j.segment}/${j.total_segments} · ${j.fps||0} fps` : capitalize(j.status)}</span>
        <span>${running ? pct + "% · ETA " + fmtEta(j.eta) : (j.status==="done"?"done":"")}</span>
      </div></div>` : ""}`;
  el.querySelectorAll("[data-act]").forEach((b) =>
    b.addEventListener("click", () => control(j.id, b.dataset.act)));
  return el;
}
function ctlButtons(j) {
  const b = (act, label, title) =>
    `<button class="btn ghost sm icon" data-act="${act}" title="${title}">${label}</button>`;
  if (j.status === "running") return b("pause", "⏸", "Pause");
  if (j.status === "paused") return b("resume", "▶", "Resume") + b("cancel", "✕", "Cancel");
  if (j.status === "queued") return b("remove", "✕", "Remove from queue");
  if (["failed", "cancelled"].includes(j.status)) return b("retry", "↻", "Retry") + b("remove", "🗑", "Remove");
  if (j.status === "done") return b("remove", "🗑", "Remove from list");
  return "";
}
async function control(id, act) {
  try {
    await post(`/api/jobs/${id}/${act}`);
    if (act === "remove") { state.jobs = state.jobs.filter((j) => j.id !== id); renderQueue(); }
  } catch (e) { toast("Action failed", e.message, "bad"); }
}
async function clearFinished() {
  try {
    const r = await post("/api/clear-finished");
    state.jobs = r.jobs; renderQueue();
  } catch (e) { toast("Clear failed", e.message, "bad"); }
}
function updateStartBtn() {
  const b = $("start-btn"), s = $("stop-btn");
  if (!b) return;
  const active = state.jobs.some((j) => ["queued", "running", "paused"].includes(j.status));
  b.textContent = state.running ? "⏸ Pause queue" : "▶ Start";
  b.classList.toggle("primary", !state.running);
  b.classList.toggle("ghost", state.running);
  b.disabled = !state.jobs.length && !state.running;
  if (s) s.style.display = state.running ? "" : "none";   // Stop only while running
}
async function queueAction(action) {
  try {
    const r = await post(`/api/queue/${action}`);
    state.running = r.running; state.jobs = r.jobs; renderQueue(); updateStartBtn();
  } catch (e) { toast("Queue action failed", e.message, "bad"); }
}

/* ── websocket ───────────────────────────────────────────── */
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws${TOKEN ? "?token=" + encodeURIComponent(TOKEN) : ""}`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") { state.jobs = msg.jobs; if (msg.running !== undefined) state.running = msg.running; renderQueue(); updateStartBtn(); }
    else if (msg.type === "job") { upsertJob(msg.job); }
  };
  ws.onclose = () => setTimeout(connectWS, 1500);
  ws.onerror = () => ws.close();
}
function upsertJob(job) {
  const i = state.jobs.findIndex((j) => j.id === job.id);
  const prev = i >= 0 ? state.jobs[i] : null;
  if (i >= 0) state.jobs[i] = job; else state.jobs.push(job);
  renderQueue();
  if (prev && prev.status !== job.status) {
    if (job.status === "done") toast("Upscale complete", job.name, "ok");
    if (job.status === "failed") toast("Upscale failed", job.name + " — " + (job.error||""), "bad");
  }
}

/* ── file browser modal ──────────────────────────────────── */
let curPath = "ROOT";
let browserMode = "files";  // files | output | images | ph-output
let browserKind = "video";  // video | image (what files to list)
const selected = new Set();
function openBrowser(mode) {
  browserMode = ["output", "images", "ph-output"].includes(mode) ? mode : "files";
  browserKind = browserMode === "images" ? "image" : "video";
  selected.clear();
  const out = browserMode === "output" || browserMode === "ph-output";
  $("browser-title").textContent = out ? "📁 Choose an output folder"
    : (browserKind === "image" ? "🖼 Choose photos" : "📁 Choose videos");
  $("add-folder-btn").style.display = out ? "none" : "";
  $("add-selected-btn").style.display = out ? "none" : "";
  $("sel-count").style.display = out ? "none" : "";
  $("use-folder-btn").style.display = out ? "" : "none";
  $("overlay").classList.add("show");
  $("browser-modal").classList.add("show");
  navigate("ROOT");
}
function closeBrowser() {
  $("overlay").classList.remove("show");
  $("browser-modal").classList.remove("show");
}
async function navigate(path) {
  let data;
  try { data = await api("/api/browse?path=" + encodeURIComponent(path) + "&kind=" + browserKind); }
  catch (e) { toast("Browse failed", e.message, "bad"); return; }
  curPath = data.path;
  if (browserMode === "output" || browserMode === "ph-output") {
    const ub = $("use-folder-btn");
    ub.textContent = curPath === "ROOT" ? "Pick a folder…" : "Use this folder";
    ub.disabled = curPath === "ROOT";
  }
  renderBreadcrumb(data);
  const box = $("browser");
  box.innerHTML = "";
  if (data.parent) box.appendChild(entryNode({ name: "..", path: data.parent, kind: "up" }));
  data.entries.forEach((e) => box.appendChild(entryNode(e)));
  updateSelCount();
}
function renderBreadcrumb(data) {
  const bc = $("breadcrumb");
  bc.innerHTML = "";
  const add = (label, path) => {
    const s = document.createElement("span");
    s.className = "crumb"; s.textContent = label;
    s.onclick = () => navigate(path); bc.appendChild(s);
  };
  add("💻 This PC", "ROOT");
  if (data.path && data.path !== "ROOT") {
    const sep = data.path.includes("\\") ? "\\" : "/";
    const parts = data.path.split(/[\\/]/).filter(Boolean);
    let acc = "";
    parts.forEach((p, i) => {
      acc += (i === 0 && sep === "\\" ? p + "\\" : (sep === "/" && i === 0 ? "/" : p + sep));
      bc.appendChild(document.createTextNode(" › "));
      add(p, acc);
    });
  }
}
function entryNode(e) {
  const el = document.createElement("div");
  el.className = "entry " + e.kind + (selected.has(e.path) ? " sel" : "");
  const ico = e.kind === "file" ? "🎞" : e.kind === "drive" ? "💽" : e.kind === "up" ? "↩" : "📁";
  el.innerHTML = `<span class="ico">${ico}</span><span class="name">${esc(e.name)}</span>` +
    (e.size ? `<span class="size">${humanSize(e.size)}</span>` : "") +
    (e.kind === "file" ? `<span class="check">${selected.has(e.path) ? "✓" : ""}</span>` : "");
  el.onclick = () => {
    if (e.kind === "file") {
      if (selected.has(e.path)) selected.delete(e.path); else selected.add(e.path);
      el.classList.toggle("sel");
      el.querySelector(".check").textContent = selected.has(e.path) ? "✓" : "";
      updateSelCount();
      $("preview-file").value = e.path;   // last-clicked file feeds the preview
    } else {
      navigate(e.path);
    }
  };
  return el;
}
function updateSelCount() {
  $("sel-count").textContent = selected.size + " selected";
  $("add-selected-btn").disabled = selected.size === 0;
}
async function addPaths(paths) {
  if (!paths.length) return;
  try {
    const r = await post("/api/jobs", { paths, settings: gatherSettings() });
    toast("Queued", r.added.length + " file(s) added", "ok");
    closeBrowser();
  } catch (e) { toast("Add failed", e.message, "bad"); }
}

/* ── model benchmark ─────────────────────────────────────── */
async function runBenchmark() {
  const photos = $("view-photos").style.display !== "none";   // which tab is active
  const btn = photos ? $("ph-bench") : $("bench-btn");
  const box = photos ? $("ph-bench-results") : $("bench-results");
  const modelSel = photos ? "ph-model" : "model";
  const noteFn = photos ? updatePhModelNote : updateModelNote;
  // benchmark the practical (fast) models by default — heavy ones are labelled slow
  const models = state.models.filter((m) => !/x4plus/.test(m.name)).map((m) => m.name);
  btn.disabled = true; btn.textContent = "⏳ Benchmarking…";
  box.style.display = "block";
  box.innerHTML = `<div class="bhead">Measuring ${models.length} models at 1080p→4K on your GPU… (10–30s; GPU load affects results)</div>`;
  try {
    const r = await post("/api/benchmark", { models, res: "1920x1080" });
    const ok = r.results.filter((x) => x.fps);
    const max = Math.max(1, ...ok.map((x) => x.fps));
    box.innerHTML = `<div class="bhead">1080p→4K on this GPU · click a row to select · fastest first</div>` +
      r.results.map((x, i) => x.error
        ? `<div class="brow err"><span class="bname">${esc(x.name)} — ${esc(x.error)}</span></div>`
        : `<div class="brow ${i === 0 ? "fastest" : ""}" data-m="${esc(x.name)}">
             <span class="bname">${esc(x.name)} <span style="color:var(--text-faint)">x${x.scale}</span></span>
             <span class="bmeter"><span style="width:${Math.round(100*x.fps/max)}%"></span></span>
             <span class="bfps">${x.fps} fps</span>
           </div>`).join("");
    box.querySelectorAll(".brow[data-m]").forEach((el) => el.onclick = () => {
      $(modelSel).value = el.dataset.m; noteFn();
    });
  } catch (e) {
    box.innerHTML = `<div class="bhead" style="color:var(--danger)">Benchmark failed: ${esc(e.message)}</div>`;
  } finally { btn.disabled = false; btn.textContent = "⚡ Benchmark"; }
}

/* ── preview + comparison slider ─────────────────────────── */
async function runPreview() {
  const file = $("preview-file").value.trim();
  if (!file) { toast("No file", "Pick a file first (＋ Add, then click one).", "bad"); return; }
  const btn = $("preview-btn");
  btn.disabled = true; btn.textContent = "Rendering…";
  try {
    const r = await post("/api/preview", { path: file, timestamp: parseFloat($("preview-ts").value) || 0, settings: gatherSettings() });
    $("preview-ph").style.display = "none";
    $("img-before").src = r.before; $("img-after").src = r.after;
    ["img-before", "after-wrap", "divider", "knob", "lbl-b", "lbl-a"].forEach((id) => $(id).style.display = "");
    setCompare(50);
  } catch (e) {
    toast("Preview failed", e.message, "bad");
  } finally { btn.disabled = false; btn.textContent = "Preview"; }
}
function sizeAfter() { $("img-after").style.width = $("compare").clientWidth + "px"; }
function setCompare(pct) {
  pct = Math.max(0, Math.min(100, pct));
  $("after").style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  $("divider").style.left = pct + "%";
  $("knob").style.left = pct + "%";
}
(function compareDrag() {
  let dragging = false;
  const c = $("compare");
  const move = (clientX) => {
    const r = c.getBoundingClientRect();
    setCompare(((clientX - r.left) / r.width) * 100);
  };
  const down = (e) => { dragging = true; move((e.touches ? e.touches[0] : e).clientX); e.preventDefault(); };
  const mv = (e) => { if (dragging) move((e.touches ? e.touches[0] : e).clientX); };
  const up = () => { dragging = false; };
  window.addEventListener("DOMContentLoaded", () => {
    c.addEventListener("mousedown", down); c.addEventListener("touchstart", down, { passive: false });
    window.addEventListener("mousemove", mv); window.addEventListener("touchmove", mv);
    window.addEventListener("mouseup", up); window.addEventListener("touchend", up);
    window.addEventListener("resize", sizeAfter);
  });
})();

/* ── toasts ──────────────────────────────────────────────── */
function toast(title, detail, kind) {
  const el = document.createElement("div");
  el.className = "toast " + (kind || "");
  el.innerHTML = `<div class="t">${esc(title)}</div>${detail ? `<div class="d">${esc(detail)}</div>` : ""}`;
  $("toasts").appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = ".4s"; setTimeout(() => el.remove(), 400); }, 4200);
}

/* ── misc ────────────────────────────────────────────────── */
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const capitalize = (s) => s ? s[0].toUpperCase() + s.slice(1) : s;
function humanSize(n) {
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i === 0 ? n : n.toFixed(1)) + u[i];
}

/* ── photos tab ──────────────────────────────────────────── */
state.photos = []; state.phRunning = false; state.photosInit = false;
let phPoll = null;
const _phPrevCache = {};

function switchTab(tab) {
  document.querySelectorAll("#tabs .tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $("view-video").style.display = tab === "video" ? "" : "none";
  $("view-photos").style.display = tab === "photos" ? "" : "none";
  if (tab === "photos" && !state.photosInit) initPhotos();
}
function initPhotos() {
  state.photosInit = true;
  const sel = $("ph-model"); sel.innerHTML = "";
  state.models.forEach((m) => {
    const o = document.createElement("option");
    o.value = m.name; o.textContent = m.name + " (x" + m.scale + ")" + (m.cached ? " ✓" : "");
    sel.appendChild(o);
  });
  if (state.models.find((m) => m.name === "realesrgan-x4plus")) sel.value = "realesrgan-x4plus";
  updatePhModelNote();
  loadPhotoState();
}
function updatePhModelNote() {
  const m = state.models.find((x) => x.name === $("ph-model").value);
  $("ph-model-note").textContent = m ? m.note + (m.cached ? "" : " — downloads on first use") : "";
}
function phSettings() {
  return {
    model: $("ph-model-custom").value.trim() || $("ph-model").value,
    output_dir: $("ph-output").value.trim() || null,
    outscale: parseInt($("ph-scale").value) || null,
    format: $("ph-format").value || null, tile: 0, fp16: true,
  };
}
async function addPhotos(paths) {
  if (!paths.length) return;
  try {
    const r = await post("/api/photos/add", { paths, settings: phSettings() });
    state.photos = r.items; renderPhotos(); phButtons(); closeBrowser();
    toast("Added", r.added.length + " photo(s) — press Upscale", "ok");
  } catch (e) { toast("Add failed", e.message, "bad"); }
}
async function photoAction(action) {
  try {
    const r = await post("/api/photos/" + action);
    state.phRunning = r.running; state.photos = r.items; renderPhotos(); phButtons();
    if (r.running) startPhPoll();
  } catch (e) { toast("Action failed", e.message, "bad"); }
}
async function loadPhotoState() {
  try {
    const s = await api("/api/photos/state");
    state.photos = s.items; state.phRunning = s.running; renderPhotos(); phButtons();
    if (s.running) startPhPoll();
  } catch (e) {}
}
function startPhPoll() {
  if (phPoll) return;
  phPoll = setInterval(async () => {
    try {
      const s = await api("/api/photos/state");
      state.photos = s.items; state.phRunning = s.running; renderPhotos(); phButtons();
      if (!s.running) { clearInterval(phPoll); phPoll = null; }
    } catch (e) {}
  }, 1500);
}
function phButtons() {
  $("ph-start").textContent = state.phRunning ? "⏸ Stop" : "▶ Upscale";
  $("ph-start").disabled = !state.phRunning && !state.photos.some((i) => ["queued", "running"].includes(i.status));
}
function renderPhotos() {
  const grid = $("photo-grid");
  $("ph-empty").style.display = state.photos.length ? "none" : "block";
  $("ph-count").textContent = state.photos.length ? state.photos.length + " image" + (state.photos.length > 1 ? "s" : "") : "";
  [...grid.querySelectorAll(".pcard")].forEach((n) => n.remove());
  state.photos.forEach((it) => grid.appendChild(pcardNode(it)));
}
function pcardNode(it) {
  const el = document.createElement("div");
  el.className = "pcard " + it.status;
  const icon = it.status === "running" ? "⏳" : it.status === "failed" ? "⚠" : "🖼";
  el.innerHTML = (it.has_preview ? `<img class="thumb">` : `<div class="thumb placeholder">${icon}</div>`) +
    `<div class="cap"><span class="nm" title="${esc(it.error || it.name)}">${esc(it.name)}</span><span class="st">${it.status}</span></div>`;
  if (it.has_preview) {
    const img = el.querySelector("img.thumb");
    fetchPhPreview(it.id).then((p) => { if (p) img.src = p.after; });
    img.onclick = () => openPhCompare(it.id);
  }
  return el;
}
async function fetchPhPreview(id) {
  if (_phPrevCache[id]) return _phPrevCache[id];
  try { const p = await api("/api/photos/" + id + "/preview"); _phPrevCache[id] = p; return p; }
  catch (e) { return null; }
}
/* zoom/pan viewer: Fit/1×/2×/4×/8× (1× = 100% actual pixels), scroll-zoom, drag-pan,
   before/after toggle (source is shown in the upscaled box for a same-framing compare). */
const phView = { natW: 0, natH: 0, scale: 1, fit: 1, panX: 0, panY: 0, beforeUrl: "", afterUrl: "", src: "after" };
function phApply() {
  const img = $("ph-img");
  img.style.width = phView.natW + "px"; img.style.height = phView.natH + "px";
  img.style.transform = `translate(${phView.panX}px, ${phView.panY}px) scale(${phView.scale})`;
  $("ph-zoomlbl").textContent = Math.round(phView.scale * 100) + "% · scroll = zoom · drag = pan";
}
function phFitScale() {
  const vp = $("ph-viewport");
  if (!phView.natW || !vp.clientWidth) return 1;
  return Math.min(vp.clientWidth / phView.natW, vp.clientHeight / phView.natH);
}
function phCenter() {
  const vp = $("ph-viewport");
  phView.panX = (vp.clientWidth - phView.natW * phView.scale) / 2;
  phView.panY = (vp.clientHeight - phView.natH * phView.scale) / 2;
}
function phSetZoom(mode) {
  phView.fit = phFitScale();
  phView.scale = mode === "fit" ? phView.fit : parseFloat(mode);
  phCenter(); phApply();
  document.querySelectorAll("#ph-zoombar button").forEach((b) => b.classList.toggle("active", b.dataset.z === mode));
}
async function openPhCompare(id) {
  const it = state.photos.find((x) => x.id === id);
  $("ph-modal-title").textContent = it ? it.name : id;
  const tk = TOKEN ? "&token=" + encodeURIComponent(TOKEN) : "";
  phView.afterUrl = `/api/photos/${id}/image?which=after${tk}`;
  phView.beforeUrl = `/api/photos/${id}/image?which=before${tk}`;
  phView.src = "after";
  $("ph-view-toggle").textContent = "Showing: Upscaled";
  const img = $("ph-img");
  img.onload = () => {                         // size from the full-res UPSCALED image
    img.onload = null;
    phView.natW = img.naturalWidth; phView.natH = img.naturalHeight;
    phSetZoom("fit");
  };
  img.src = phView.afterUrl;                   // full file, not the thumbnail
  $("ph-overlay").classList.add("show"); $("ph-modal").classList.add("show");
}
function closePhCompare() { $("ph-overlay").classList.remove("show"); $("ph-modal").classList.remove("show"); }
function phToggleSrc() {                        // swap src, keep zoom/pan + the upscaled box
  phView.src = phView.src === "after" ? "before" : "after";
  $("ph-img").src = phView.src === "after" ? phView.afterUrl : phView.beforeUrl;
  $("ph-view-toggle").textContent = "Showing: " + (phView.src === "after" ? "Upscaled" : "Source");
}
(function phPanZoom() {
  const shown = () => $("ph-modal").classList.contains("show");
  const vp = () => $("ph-viewport");
  let drag = false, sx = 0, sy = 0, px = 0, py = 0;
  document.addEventListener("mousedown", (e) => {
    if (shown() && vp().contains(e.target)) {
      drag = true; sx = e.clientX; sy = e.clientY; px = phView.panX; py = phView.panY;
      vp().classList.add("grabbing"); e.preventDefault();
    }
  });
  document.addEventListener("mousemove", (e) => {
    if (drag) { phView.panX = px + (e.clientX - sx); phView.panY = py + (e.clientY - sy); phApply(); }
  });
  document.addEventListener("mouseup", () => { drag = false; const v = vp(); if (v) v.classList.remove("grabbing"); });
  document.addEventListener("wheel", (e) => {
    if (!shown() || !vp().contains(e.target)) return;
    e.preventDefault();
    const r = vp().getBoundingClientRect(), cx = e.clientX - r.left, cy = e.clientY - r.top;
    const ns = Math.max(phView.fit * 0.5, Math.min(16, phView.scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    phView.panX = cx - (cx - phView.panX) * (ns / phView.scale);
    phView.panY = cy - (cy - phView.panY) * (ns / phView.scale);
    phView.scale = ns; phApply();
    document.querySelectorAll("#ph-zoombar button").forEach((b) => b.classList.remove("active"));
  }, { passive: false });
  // recompute fit when the resizable modal changes size
  if (window.ResizeObserver) new ResizeObserver(() => {
    if (!shown() || !phView.natW) return;
    const wasFit = Math.abs(phView.scale - phView.fit) < 1e-4;
    phView.fit = phFitScale();
    if (wasFit) { phView.scale = phView.fit; phCenter(); }
    phApply();
  }).observe(document.getElementById("ph-viewport"));
})();

/* ── wire up ─────────────────────────────────────────────── */
function wire() {
  $("add-btn").onclick = openBrowser;
  $("clear-btn").onclick = clearFinished;
  $("start-btn").onclick = () => queueAction(state.running ? "pause" : "start");
  $("stop-btn").onclick = () => queueAction("stop");
  $("browser-close").onclick = closeBrowser;
  $("overlay").onclick = closeBrowser;
  const addFromBrowser = (paths) => (browserMode === "images" ? addPhotos(paths) : addPaths(paths));
  $("add-selected-btn").onclick = () => addFromBrowser([...selected]);
  $("add-folder-btn").onclick = () => addFromBrowser([curPath]);
  $("output-browse").onclick = () => openBrowser("output");
  $("use-folder-btn").onclick = () => {
    if (curPath && curPath !== "ROOT") {
      $(browserMode === "ph-output" ? "ph-output" : "output").value = curPath;
    }
    closeBrowser();
  };
  // tabs
  document.querySelectorAll("#tabs .tab").forEach((b) => b.onclick = () => switchTab(b.dataset.tab));
  // photos
  $("ph-add").onclick = () => openBrowser("images");
  $("ph-start").onclick = () => photoAction(state.phRunning ? "stop" : "start");
  $("ph-clear").onclick = () => photoAction("clear");
  $("ph-output-browse").onclick = () => openBrowser("ph-output");
  $("ph-model").onchange = updatePhModelNote;
  $("ph-bench").onclick = runBenchmark;
  $("ph-modal-close").onclick = closePhCompare;
  $("ph-overlay").onclick = closePhCompare;
  document.querySelectorAll("#ph-zoombar button").forEach((b) => b.onclick = () => phSetZoom(b.dataset.z));
  $("ph-view-toggle").onclick = phToggleSrc;
  $("preview-btn").onclick = runPreview;
  $("model").onchange = updateModelNote;
  $("bench-btn").onclick = runBenchmark;
  document.querySelectorAll("#preset-seg button").forEach(
    (b) => b.onclick = () => applyPreset(b.dataset.v));
  $("interpolate").onchange = (e) => $("fps-field").style.display = e.target.checked ? "" : "none";
  document.querySelectorAll("#target-seg button").forEach((b) =>
    b.onclick = () => {
      document.querySelectorAll("#target-seg button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      $("target-custom").style.display = b.dataset.v === "custom" ? "" : "none";
    });
  initLivePreview();
}

/* ── live preview (WebSocket comparison) ─────────────────── */
let liveWS = null;
let liveDragging = false;
let liveComparePct = 50;

function initLivePreview() {
  // Watch/Pause button inside popup
  $("live-toggle").onclick = () => {
    if (liveWS) stopLivePreview(); else startLivePreview();
  };

  // Topbar button shows/hides popup
  $("live-popup-toggle").onclick = () => {
    const p = $("live-popup");
    p.style.display = p.style.display === "none" ? "flex" : "none";
  };

  // Close button hides popup
  $("live-popup-close").onclick = () => { $("live-popup").style.display = "none"; };

  // Drag popup by header
  const popup = $("live-popup");
  const head = $("live-popup-head");
  let dragOff = null;
  head.addEventListener("mousedown", (e) => {
    if (e.target.closest("button")) return;
    const r = popup.getBoundingClientRect();
    dragOff = { x: e.clientX - r.left, y: e.clientY - r.top };
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragOff) return;
    popup.style.left = Math.max(0, e.clientX - dragOff.x) + "px";
    popup.style.top = Math.max(0, e.clientY - dragOff.y) + "px";
    popup.style.right = "auto";
  });
  window.addEventListener("mouseup", () => { dragOff = null; });

  // Comparison slider drag
  const c = $("live-compare");
  const move = (clientX) => {
    const r = c.getBoundingClientRect();
    liveComparePct = Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100));
    setLiveCompare(liveComparePct);
  };
  const down = (e) => { liveDragging = true; move((e.touches ? e.touches[0] : e).clientX); e.preventDefault(); };
  const mv = (e) => { if (liveDragging) move((e.touches ? e.touches[0] : e).clientX); };
  const up = () => { liveDragging = false; };
  c.addEventListener("mousedown", down); c.addEventListener("touchstart", down, { passive: false });
  window.addEventListener("mousemove", mv); window.addEventListener("touchmove", mv);
  window.addEventListener("mouseup", up); window.addEventListener("touchend", up);
}

function setLiveCompare(pct) {
  pct = Math.max(0, Math.min(100, pct));
  $("live-after").style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  $("live-divider").style.left = pct + "%";
  $("live-knob").style.left = pct + "%";
}

function startLivePreview() {
  if (liveWS) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws/preview${TOKEN ? "?token=" + encodeURIComponent(TOKEN) : ""}`;
  liveWS = new WebSocket(url);

  $("live-toggle").textContent = "⏸ Pause";
  $("live-status").textContent = "connecting…";

  liveWS.onopen = () => {
    $("live-status").textContent = "watching";
  };

  liveWS.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type !== "preview") return;

    $("live-ph").style.display = "none";
    $("live-before").style.display = "";
    $("live-after-wrap").style.display = "";
    $("live-divider").style.display = "";
    $("live-knob").style.display = "";
    $("live-lbl-b").style.display = "";
    $("live-lbl-a").style.display = "";

    $("live-before").src = msg.src;
    $("live-after").src = msg.upscaled;
    $("live-fps").textContent = msg.fps ? msg.fps.toFixed(1) + " fps" : "";
    $("live-progress").textContent = msg.progress != null ? (msg.progress * 100).toFixed(1) + "%" : "";
    $("live-name").textContent = msg.name || "";
    $("live-status").textContent = "● live";
  };

  liveWS.onclose = () => {
    if (liveWS) {
      liveWS = null;
      // Auto-reconnect after a delay if the job might still be running
      setTimeout(() => {
        if (!liveWS && state.jobs.some(j => j.status === "running")) {
          startLivePreview();
        }
      }, 2000);
    }
  };

  liveWS.onerror = () => {
    if (liveWS) liveWS.close();
  };
}

function stopLivePreview() {
  if (liveWS) {
    liveWS.close();
    liveWS = null;
  }
  $("live-toggle").textContent = "▶ Watch";
  $("live-status").textContent = "stopped";
}

async function boot() {
  wire();
  try { await loadInfo(); } catch (e) { toast("Backend error", e.message, "bad"); }
  try { await loadModels(); } catch (e) { /* torch-less still ok */ }
  try { const s = await api("/api/state"); state.jobs = s.jobs; state.running = s.running; renderQueue(); updateStartBtn(); } catch (e) {}
  connectWS();
}
document.addEventListener("DOMContentLoaded", boot);
