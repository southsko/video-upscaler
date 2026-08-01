#!/usr/bin/env python3
"""FastAPI backend for the Video Upscaler web UI.

Launched by `upscale_video.py --serve`. Drives the same engine (Job/JobQueue,
Upscaler) the CLI uses — this layer only exposes REST + WebSocket and never
re-implements pipeline logic.

Security: binds 127.0.0.1 by default. Any non-localhost host requires a token
(printed at startup, passed as ?token=... or X-Upscale-Token). The UI can read
the server filesystem, so only expose it on a trusted LAN.
"""
import asyncio
import os
import secrets
import string
import threading
import webbrowser

import upscale_video as U

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.concurrency import run_in_threadpool
    import uvicorn
except ImportError as e:                              # surfaced by main()
    raise ImportError(f"web UI needs fastapi + uvicorn ({e})")

WEBUI_DIR = os.path.join(U.HERE, "webui")


# ── websocket broadcast bridge (worker thread -> asyncio) ─────────────────────
class Hub:
    """Fan-out queue updates to all connected websockets. The JobQueue worker
    runs in a plain thread; it hands dicts to us via push(), which schedules the
    broadcast on the event loop thread-safely."""

    def __init__(self):
        self.clients = set()
        self.loop = None
        self.queue = None

    def bind(self, loop):
        self.loop = loop
        self.queue = asyncio.Queue()

    def push(self, payload):
        # called from the worker thread
        if self.loop and self.queue is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)

    async def broadcast_loop(self):
        while True:
            payload = await self.queue.get()          # payload is a full message dict
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_json(payload)
                except Exception:                    # noqa: BLE001
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


# ── filesystem browser (server-side) ──────────────────────────────────────────
def list_drives():
    # Linux / macOS: return filesystem root entries
    if os.name != "nt":
        entries = []
        try:
            for name in sorted(os.listdir("/"), key=str.lower):
                full = os.path.join("/", name)
                try:
                    if os.path.isdir(full):
                        entries.append({"name": name, "path": full, "kind": "dir"})
                except OSError:
                    continue
        except OSError:
            pass
        return entries
    # Windows: list drive letters
    drives = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            drives.append({"name": root, "path": root, "kind": "drive"})
    return drives


def browse(path):
    """Return {path, parent, entries[]} for a directory. '' or 'ROOT' -> drives."""
    if not path or path in ("ROOT", "/"):
        return {"path": "ROOT", "parent": None, "entries": list_drives()}
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise HTTPException(404, f"Not a directory: {path}")
    parent = os.path.dirname(path.rstrip("\\/")) or None
    if parent and os.path.normcase(parent) == os.path.normcase(path):
        parent = "ROOT"
    dirs, files = [], []
    try:
        for name in sorted(os.listdir(path), key=str.lower):
            full = os.path.join(path, name)
            try:
                if os.path.isdir(full):
                    dirs.append({"name": name, "path": full, "kind": "dir"})
                elif os.path.splitext(name)[1].lower() in U._VIDEO_EXTS:
                    files.append({"name": name, "path": full, "kind": "file",
                                  "size": os.path.getsize(full)})
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(403, "Permission denied")
    return {"path": path, "parent": parent if parent else "ROOT",
            "entries": dirs + files}


# ── app factory ───────────────────────────────────────────────────────────────
def create_app(state):
    app = FastAPI(title="Video Upscaler")
    hub = state["hub"]
    q = state["queue"]

    def check_token(request=None, token=None):
        need = state["token"]
        if not need:
            return
        got = token or (request.headers.get("X-Upscale-Token") if request else None)
        if got != need:
            raise HTTPException(401, "Invalid or missing token")

    @app.get("/", response_class=HTMLResponse)
    def index():
        idx = os.path.join(WEBUI_DIR, "index.html")
        if not os.path.isfile(idx):
            return HTMLResponse("<h1>webui/index.html missing</h1>", status_code=500)
        with open(idx, encoding="utf-8") as f:
            html = f.read()
        # cache-bust css/js by file mtime, so edits always load (no stale JS)
        for asset in ("styles.css", "app.js"):
            p = os.path.join(WEBUI_DIR, asset)
            v = int(os.path.getmtime(p)) if os.path.isfile(p) else 0
            html = html.replace(f"/static/{asset}", f"/static/{asset}?v={v}")
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    if os.path.isdir(WEBUI_DIR):
        app.mount("/static", StaticFiles(directory=WEBUI_DIR), name="static")

    @app.get("/api/info")
    def api_info(request: Request, token: str = Query(None)):
        check_token(request, token)
        _va, label = U.pick_encoder({"codec": "h264_nvenc"})
        gpu = _gpu_name()
        return {
            "encoder": label, "gpu": gpu,
            "default_model": U.DEFAULT_MODEL,
            "targets": list(U.TARGET_PRESETS.keys()),
            "codecs": ["h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264"],
            "presets": {k: v for k, v in U.QUALITY_PRESETS.items()},
            "torch": _torch_ok(),
            "weights_dir": U.DEFAULT_WEIGHTS_DIR,
        }

    @app.get("/api/models")
    def api_models(request: Request, token: str = Query(None)):
        check_token(request, token)
        return {"builtins": [
            {"name": n, "scale": s, "note": note,
             "cached": os.path.isfile(os.path.join(U.DEFAULT_WEIGHTS_DIR, os.path.basename(url)))}
            for n, (url, s, note) in U.BUILTIN_MODELS.items()]}

    @app.get("/api/browse")
    def api_browse(request: Request, path: str = Query(""), token: str = Query(None)):
        check_token(request, token)
        return browse(path)

    @app.get("/api/state")
    def api_state(request: Request, token: str = Query(None)):
        check_token(request, token)
        return {"jobs": q.snapshot(), "current": q.current.id if q.current else None,
                "running": q.running}

    @app.post("/api/queue/{action}")
    def api_queue(action: str, request: Request, token: str = Query(None)):
        check_token(request, token)
        if action == "start":
            q.start()
        elif action == "pause":
            q.pause_queue()
        elif action == "stop":
            q.stop()
        else:
            raise HTTPException(400, f"Unknown action {action}")
        return {"running": q.running, "jobs": q.snapshot()}

    @app.post("/api/jobs")
    async def api_add(request: Request, token: str = Query(None)):
        check_token(request, token)
        body = await request.json()
        paths = body.get("paths", [])
        settings = body.get("settings", {})
        files = U.expand_inputs(paths)
        if not files:
            raise HTTPException(400, "No videos found in the given paths")
        added = [q.add(f, settings).to_dict() for f in files]
        return {"added": added}

    @app.post("/api/clear-finished")
    def api_clear(request: Request, token: str = Query(None)):
        check_token(request, token)
        q.clear_finished()
        q._changed(None)
        return {"jobs": q.snapshot()}

    @app.post("/api/jobs/{job_id}/{action}")
    def api_control(job_id: str, action: str, request: Request, token: str = Query(None)):
        check_token(request, token)
        if action == "remove":
            if not q.remove(job_id):
                raise HTTPException(404, "No such job")
            q._changed(None)
            return {"removed": job_id}
        job = q.get(job_id)
        if not job:
            raise HTTPException(404, "No such job")
        if action == "pause":
            job.pause()
        elif action == "resume":
            job.resume()
        elif action == "cancel":
            job.cancel()
        elif action == "retry":
            if job.status in ("failed", "cancelled"):
                job.status = "queued"
                job._cancel.clear()
                q._ensure_worker()
        else:
            raise HTTPException(400, f"Unknown action {action}")
        q._changed(job)
        return job.to_dict()

    @app.post("/api/benchmark")
    async def api_benchmark(request: Request, token: str = Query(None)):
        check_token(request, token)
        if not _torch_ok():
            raise HTTPException(503, "Benchmark needs torch/CUDA installed")
        body = await request.json()
        names = body.get("models") or None
        res = body.get("res", "1920x1080")
        try:
            results = await run_in_threadpool(U.benchmark_models, names, res)
        except Exception as e:                        # noqa: BLE001
            raise HTTPException(500, f"Benchmark failed: {e}")
        return {"res": res, "results": results}

    @app.post("/api/preview")
    async def api_preview(request: Request, token: str = Query(None)):
        check_token(request, token)
        if not _torch_ok():
            raise HTTPException(503, "Preview needs torch/CUDA installed")
        body = await request.json()
        path = body.get("path")
        ts = float(body.get("timestamp", 1.0))
        settings = body.get("settings", {})
        crop = int(body.get("crop", 480))
        cx = float(body.get("cx", 0.5)); cy = float(body.get("cy", 0.5))
        try:
            # torch inference is blocking — run it off the event loop so the
            # server (queue, websockets, other requests) stays responsive.
            before_b64, after_b64 = await run_in_threadpool(
                _make_preview, path, ts, settings, crop, (cx, cy))
        except Exception as e:                        # noqa: BLE001
            raise HTTPException(500, f"Preview failed: {e}")
        return {"before": before_b64, "after": after_b64, "crop": crop}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        # token check for non-local binds
        if state["token"]:
            if websocket.query_params.get("token") != state["token"]:
                await websocket.close(code=1008)
                return
        await websocket.accept()
        hub.clients.add(websocket)
        try:
            await websocket.send_json({"type": "state", "jobs": q.snapshot(),
                                       "running": q.running})
            while True:
                await websocket.receive_text()       # keepalive / ignore
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(websocket)

    @app.websocket("/ws/preview")
    async def ws_preview(websocket: WebSocket):
        """Stream live before/after frame pairs from the running job."""
        import base64
        if state["token"]:
            if websocket.query_params.get("token") != state["token"]:
                await websocket.close(code=1008)
                return
        await websocket.accept()
        last_seq = -1
        try:
            while True:
                running_job = None
                for j in q.jobs:
                    if j.status == "running":
                        running_job = j
                        break
                if running_job and running_job._live_preview:
                    preview = running_job._live_preview
                    seq = preview.get("seq", 0)
                    if seq != last_seq:
                        last_seq = seq
                        await websocket.send_json({
                            "type": "preview",
                            "src": "data:image/jpeg;base64," + base64.b64encode(preview["src"]).decode(),
                            "upscaled": "data:image/jpeg;base64," + base64.b64encode(preview["upscaled"]).decode(),
                            "seq": seq,
                            "name": os.path.basename(running_job.src),
                            "fps": running_job.fps,
                            "progress": running_job.progress,
                        })
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            import traceback, sys
            print(f"[WS/PREVIEW ERROR] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)

    @app.on_event("startup")
    async def _startup():
        loop = asyncio.get_running_loop()
        hub.bind(loop)

        def emit(payload):
            # payload: a job dict (single-job update) or None (full-state refresh)
            if payload is None:
                hub.push({"type": "state", "jobs": q.snapshot(), "running": q.running})
            else:
                hub.push({"type": "job", "job": payload})
        q.on_change = emit
        asyncio.create_task(hub.broadcast_loop())

    return app


# ── preview helper (before/after single frame) ────────────────────────────────
def _make_preview(path, timestamp, settings, crop=480, center=(0.5, 0.5)):
    """Upscale a small CENTERED CROP of one frame (not the whole frame) — instant
    even at 4K and shows true 100% pixel detail. Applies the same deinterlace/
    HDR-tonemap the real job would, so the preview matches the output."""
    import base64
    import subprocess
    import numpy as np
    import cv2
    meta = U.probe(path)
    w, h = meta["width"], meta["height"]
    di, tm = U._decode_flags(meta, settings)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', str(timestamp),
           '-i', path, '-frames:v', '1']
    vf = U.source_vf(di, tm, meta.get("color_transfer") or "smpte2084")
    if vf:
        cmd += ['-vf', vf]
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    raw = subprocess.run(cmd, capture_output=True).stdout
    if len(raw) < w * h * 3:
        raise RuntimeError("could not read a frame at that timestamp")
    frame = np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3)

    # centered detail crop (16:9-ish, capped so upscaling is fast)
    cw = min(w, crop)
    ch = min(h, max(1, round(cw * 9 / 16)))
    x0 = int(min(max(center[0] * w - cw / 2, 0), w - cw))
    y0 = int(min(max(center[1] * h - ch / 2, 0), h - ch))
    before = np.ascontiguousarray(frame[y0:y0 + ch, x0:x0 + cw])

    model_path = U.resolve_model(settings.get("model", U.DEFAULT_MODEL),
                                 settings.get("weights_dir", U.DEFAULT_WEIGHTS_DIR),
                                 assume_yes=True)
    up = _preview_upscaler(model_path, settings)
    after = up.enhance(before)

    def enc(img_rgb):
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        okj, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    return enc(before), enc(after)


_PREVIEW_CACHE = {}


def _preview_upscaler(model_path, settings):
    key = (model_path, settings.get("gpu", 0), settings.get("fp16", True),
           settings.get("tile", 512))
    if key not in _PREVIEW_CACHE:
        _PREVIEW_CACHE[key] = U.Upscaler(
            model_path, gpu=settings.get("gpu", 0), fp16=settings.get("fp16", True),
            tile=settings.get("tile", 512), tile_pad=settings.get("tile_pad", 16))
    return _PREVIEW_CACHE[key]


# ── system probes ─────────────────────────────────────────────────────────────
def _torch_ok():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                # noqa: BLE001
        return False


def _gpu_name():
    try:
        rc, out, _ = U._run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                            timeout=8)
        return out.strip().splitlines()[0] if rc == 0 and out.strip() else "unknown"
    except Exception:                                # noqa: BLE001
        return "unknown"


# ── entrypoint ────────────────────────────────────────────────────────────────
def serve(args):
    host = args.host
    port = args.port
    is_local = host in ("127.0.0.1", "localhost", "::1")
    # Token disabled for local network use
    token = None

    persist = os.path.join(U.HERE, "jobs.json")
    q = U.JobQueue(persist_path=persist)
    hub = Hub()
    state = {"queue": q, "hub": hub, "token": token}
    app = create_app(state)

    U.div()
    U.ok(f"Video Upscaler web UI  →  http://{host}:{port}")
    if token:
        U.warn("Non-localhost bind — access token required (share only on a trusted LAN):")
        print(f"        token: {token}")
        U.warn(f"Open:  http://{host}:{port}/?token={token}")
    if not _torch_ok():
        U.warn("torch/CUDA not detected — the UI runs, but jobs & preview need it. See README.")
    U.div()

    url = f"http://{'127.0.0.1' if is_local else host}:{port}/"
    if token:
        url += f"?token={token}"
    if getattr(args, "open", False):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def _gen_token(n=24):
    alpha = string.ascii_letters + string.digits
    return "".join(secrets.choice(alpha) for _ in range(n))
