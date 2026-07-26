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

/* ── websocket ───────────────────────────────────────────── */
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws${TOKEN ? "?token=" + encodeURIComponent(TOKEN) : ""}`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") { state.jobs = msg.jobs; renderQueue(); }
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
const selected = new Set();
function openBrowser() {
  selected.clear();
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
  try { data = await api("/api/browse?path=" + encodeURIComponent(path)); }
  catch (e) { toast("Browse failed", e.message, "bad"); return; }
  curPath = data.path;
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
    const parts = data.path.split(/[\\/]/).filter(Boolean);
    let acc = "";
    parts.forEach((p, i) => {
      acc += (i === 0 ? p + "\\" : p + "\\");
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
    sizeAfter(); setCompare(50);
  } catch (e) {
    toast("Preview failed", e.message, "bad");
  } finally { btn.disabled = false; btn.textContent = "Preview"; }
}
function sizeAfter() { $("img-after").style.width = $("compare").clientWidth + "px"; }
function setCompare(pct) {
  pct = Math.max(0, Math.min(100, pct));
  $("after-wrap").style.width = pct + "%";
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

/* ── wire up ─────────────────────────────────────────────── */
function wire() {
  $("add-btn").onclick = openBrowser;
  $("clear-btn").onclick = clearFinished;
  $("browser-close").onclick = closeBrowser;
  $("overlay").onclick = closeBrowser;
  $("add-selected-btn").onclick = () => addPaths([...selected]);
  $("add-folder-btn").onclick = () => addPaths([curPath]);
  $("preview-btn").onclick = runPreview;
  $("model").onchange = updateModelNote;
  $("interpolate").onchange = (e) => $("fps-field").style.display = e.target.checked ? "" : "none";
  document.querySelectorAll("#target-seg button").forEach((b) =>
    b.onclick = () => {
      document.querySelectorAll("#target-seg button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      $("target-custom").style.display = b.dataset.v === "custom" ? "" : "none";
    });
}

async function boot() {
  wire();
  try { await loadInfo(); } catch (e) { toast("Backend error", e.message, "bad"); }
  try { await loadModels(); } catch (e) { /* torch-less still ok */ }
  try { state.jobs = (await api("/api/state")).jobs; renderQueue(); } catch (e) {}
  connectWS();
}
document.addEventListener("DOMContentLoaded", boot);
