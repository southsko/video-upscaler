# Video Upscaler — Developer Handoff

Self-contained context so a fresh session (any model) can continue without prior history.

## 0. North star (long-term direction — decided with the owner)
Target hardware is **24 GB VRAM** (owner is NOT upgrading beyond that; the 3080/10GB is the dev box).
Quality ladder, all riding ONE multi-frame pipe:
1. **Single-image SR** (spandrel: Real-ESRGAN/SPAN/Compact) — fast everyday workhorse. ✅ done.
2. **Temporal VSR** (ccrestoration: AnimeSR/BasicVSR/IconVSR/EDVR) — multi-frame, less flicker. ✅ done
   (`--vsr`). The realistic quality ceiling on 24 GB at sane speed.
3. **Diffusion video restoration** — the true ceiling. **Target model: SeedVR2** (ICLR2026,
   github.com/IceClear/SeedVR2, code at github.com/ByteDance-Seed/SeedVR) — *one-step* diffusion VSR,
   3B/7B with FP8/GGUF quant to fit 24 GB. **Integration path is BUILT (`--external-cmd`)** but
   SeedVR2 itself is NOT installed/verified here (it can't: needs xformers/flash-attn — no py3.14/
   Windows wheels — and 24 GB). See below.

### Tier 3 / SeedVR2 integration — how it actually plugs in
SeedVR2 can't live in our py3.14 venv (heavy deps). So the design is **orchestration, not in-process**:
`--external-cmd` runs ANY external video-to-video upscaler **per segment in its own env/GPU**, and our
tool wraps it with split/resume/concat/audio-mux. The plumbing is built and verified with an ffmpeg
stand-in (`_run_external_segment` / `_external_args`). To finish on the 24 GB box:
1. Install SeedVR2 in ITS OWN environment (clone ByteDance-Seed/SeedVR, its deps, download 3B/FP8 weights).
   Linux strongly preferred (xformers/flash-attn wheels exist there for a supported Python).
2. Wrap its inference as a video-in/video-out command, then:
   ```
   python upscale_video.py "movie.mkv" --target 4k --segment-seconds 60 \
     --external-cmd "python /opt/SeedVR2/infer_video.py --input {input} --output {output} --res {target} --model 3b-fp8"
   ```
   Placeholders: `{input} {output} {width} {height} {target}`. Our tool handles everything around it.
3. Tune `--segment-seconds` down (e.g. 30–60) since diffusion is slow and you want frequent resume points.

The architecture bet paid off: **the multi-frame window (`_vsr_stream`) AND the external per-segment
seam are both built** — temporal VSR uses the window now; a diffusion model plugs in via `--external-cmd`
today (or in-process later via the same `enhance_clip`/`is_temporal` protocol if someone packages it).

## 1. What this project is
A free, self-hosted **AI video upscaler**: streams frames `ffmpeg decode → GPU super-resolution →
ffmpeg NVENC encode` with **no PNG files touching disk**. Open models via `spandrel` (Real-ESRGAN
etc.), optional RIFE frame interpolation via `ccvfi`. CLI **and** a polished local web UI share one
engine. (Do **not** reference "Topaz" anywhere — deliberately scrubbed.)

- **Repo:** https://github.com/southsko/video-upscaler (public). Auth is **SSH** (`git@github.com:...`),
  key already installed; `git push` works with no prompts. **`git push --force` is blocked by the
  local safety classifier** — the human must run force-pushes manually.
- **Local path:** `C:\Users\Joey\video-upscaler`
- **Owner:** GitHub `southsko`, email `joeygrant@gmail.com`.

## 2. Environment (verified working)
- Windows 11, **Python 3.14.6** (python.org, NOT the Store build), venv at `.venv`.
- **torch 2.11.0+cu128** (CUDA), spandrel 0.4.2, ccvfi 0.0.3, opencv 5.0, numpy 2.4, fastapi, uvicorn.
- ffmpeg/ffprobe 8.1 on PATH (full build, NVENC). **NVIDIA RTX 3080, 10 GB**, driver 595.79.
- Run everything through the venv: `.\.venv\Scripts\python.exe upscale_video.py ...`
- torch CUDA wheels for py3.14 only exist on the **cu128/cu126** indexes (not cu124). `--setup`
  reproduces the env.
- **Ampere can't AV1-encode** — use h264_nvenc/hevc_nvenc.

## 3. Files
- `upscale_video.py` — engine + CLI (everything: probe, model loading, streaming pipe, encoder
  detection, Job/JobQueue, dedup, ETA, dry-run, argparse). Heavy deps lazy-imported so
  `--help/--dry-run/--list-models` work without torch.
- `server.py` — FastAPI backend (`--serve`): file browser, queue endpoints, WebSocket progress,
  before/after preview, model list. Binds 127.0.0.1; `--host 0.0.0.0` requires a printed token.
- `webui/` — `index.html`, `styles.css`, `app.js` (vanilla, no build step, no external CDNs).
- `requirements.txt`, `README.md`, `.gitignore` (ignores `.venv`, `models/`, `__pycache__`, `jobs.json`).

## 4. Architecture / key symbols (in `upscale_video.py`)
- `probe(path)` → dict {width,height,fps,duration,nb_frames,pix_fmt,vcodec,has_audio,has_subs}.
- `class Upscaler` — loads spandrel model; `enhance(rgb_uint8)→rgb_uint8` with tiling + fp16.
  `_auto_tile()` picks tile from free VRAM (512 on the 3080). `--tile 0` = auto (default).
- `class Interpolator` — RIFE via `ccvfi.AutoModel`; `interpolate(a,b,t)` at fractional timestep.
- `class _DedupEnhance` — reuses last upscaled frame when the source frame is byte-identical (lossless).
- `_resample(frames, fps_in, fps_out, interp)` — generator for arbitrary-ratio interpolation.
- `_process_segment(...)` — the threaded reader→GPU→writer pipe for one segment.
- `class Job` / `class JobQueue` — shared by CLI and web; queue persists to `jobs.json`, restores on
  start. `_scratch_dir(job)` is keyed by **hash(src+settings)** so resume survives restarts.
- `run_job(job, upscaler, on_progress, interpolator)` — full pipeline: probe → split (lossless
  segments) → per-segment upscale (skip finished) → concat → mux original audio/subs + metadata.
- `pick_encoder(opts)` / `build_video_args(opts)` — NVENC detection + high-quality defaults
  (constqp qp18, p7, hq, spatial_aq). Final `scale=...:flags=lanczos,pad=...` reaches exact target.
- `estimate_and_report(...)`, `benchmark_fps(...)`, `sample_frame(...)` — upfront ETA.

Pipeline: split source losslessly at keyframes into `src_%04d`; each segment decodes to rgb24
rawvideo → GPU (interpolate?→upscale) → rawvideo into an NVENC ffmpeg that does the final lanczos
scale+pad to the exact target → `up_%04d.mkv`; concat (`-c copy`) → mux audio/subs from the original.

## 5. Current state — DONE & verified on the RTX 3080
Upscale (480p→4K, 1080p→4K on real film), audio/subs preserved, videoai metadata; segmentation +
concat; resume across restarts; RIFE interpolation (2× and non-integer, e.g. 30→45); web UI (file
browser, live queue, before/after slider, persistence, clear/remove); auto-tile; upfront ETA;
lossless dup-frame dedup. All committed & pushed.

## 6. Key measured findings (RTX 3080) — IMPORTANT
- Pipeline is **GPU-compute-bound and already well-overlapped** (~30 fps end-to-end at 480p→4K =
  inference ceiling). **Batching and `torch.compile` gave NO gain** — do not re-add them.
- **Model choice dominates — and it's about ARCHITECTURE SIZE, not upscale factor.** The backbone
  runs at INPUT resolution, so ×2 vs ×4 of the same-size net cost ~the same (measured: `2x-nomos-span`
  ×2 = 2.8fps vs `animevideov3` ×4 = 6.5fps at 1080p; the ×2 is *slower* because it's a bigger net).
  My earlier "×2 = 4× faster" theory was WRONG — disproven by measurement. Don't repeat it.
- 1080p→4K throughput: `realesr-animevideov3` (tiny SRVGGNet, anime) ~6.5fps; `2x-nomos-span` (SPAN,
  live-action) ~2.8fps (~17h/movie — practical); `realesrgan-x4plus` (heavy RRDBNet) ~0.25fps (~8
  days — impractical). **Fast small backbones (SRVGGNet/SPAN) are the only viable choice for video.**
- Added fast SPAN/Compact builtins (`2x-nomos-span`, `2x-parimg-compact`, HF-backed via `hf:` prefix
  in BUILTIN_MODELS) so live-action film is finally practical.
- **Exact-match dedup barely triggers on lossy-encoded video** (decoded "duplicate" frames aren't
  byte-identical). Needs a tolerance threshold to help real animation.

## 7. Gotchas (things that bit us — keep in mind)
- **ccrestoration VSR is fp32-only** — its VSR inference doesn't cast inputs to half, so fp16 crashes
  ("Input type float / bias type Half"). `VSRUpscaler` forces fp32 (fine on 24 GB). ccvfi (RIFE) fp16 is fine.
- spandrel's model `__call__` returns an **inference-mode tensor** → do **non-inplace** post-processing
  (no `.clamp_()`/`.mul_()`). Both `Upscaler.enhance` and `Interpolator.interpolate` were fixed;
  keep any new tensor code non-inplace or `.clone()` first.
- **Windows console is cp1252** — module top reconfigures stdout/stderr to UTF-8; keep that.
- Interpolation runs **per segment**, so one bridging frame is dropped at each segment boundary
  (negligible; documented).
- LF→CRLF git warnings are harmless.
- The safety classifier blocks `git push --force` and can transiently block other Bash calls (retry).

## 8. How to run / test
```powershell
# CLI
.\.venv\Scripts\python.exe upscale_video.py "<file-or-folder>" --target 4k
.\.venv\Scripts\python.exe upscale_video.py <file> --dry-run        # prints plan+commands
.\.venv\Scripts\python.exe upscale_video.py --serve --open          # web UI (http://127.0.0.1:8848)
.\.venv\Scripts\python.exe upscale_video.py --list-models
```
- **Real test sources:** the remote share **`Z:\movies`** has 1080p BluRay remuxes. Test clip used:
  `The Game (1997)` (1920×1080 h264 8-bit SDR 23.976fps). Extract a short clip with
  `ffmpeg -ss 0:30:00 -t 20 -i <movie> -c copy clip.mkv` rather than upscaling a whole film.
  Some films are 2160p HDR (good for the 10-bit/HDR TODO).
- Scratch/tmp lives under `%TEMP%\upscale_scratch`. Downloaded model weights cache in `models/`.

## 9. TODO — prioritized roadmap (🔴 high / 🟡 med / ⚪ low)
**Speed/efficiency**
1. ✅ DONE — added fast SPAN/Compact builtins (`2x-nomos-span` etc.); live-action now practical.
2. 🟡 **Content-aware model auto-select** (`--model auto`): the hard part is detecting anime vs
   live-action (can't be done from scale alone — my scale-based idea was wrong). Consider a cheap
   heuristic (edge/palette stats) or just a UI hint. Lower priority than I first thought.
3. ✅ DONE — models labelled by speed/content in `--list-models` and notes.
4. 🟡 `--dedup-threshold` for near-duplicate frames (exact-match barely triggers on lossy video).
5. ⚪ Surface the ETA in the web UI (currently CLI only).

**Real-media correctness**
6. 🔴 **VFR**: detect variable frame rate, preserve timestamps (avoid A/V drift from `-r` re-timing).
7. 🔴 **10-bit/HDR**: detect and keep a p010/main10 path or tonemap+warn (today HDR flattens to 8-bit).
8. 🟡 **Deinterlace** (`yadif`) option for interlaced sources.
9. 🟡 Show pix_fmt/bit-depth/HDR/VFR/interlaced in `--dry-run` and the UI.

**UX**
10. 🔴 **Quality presets** Fast/Balanced/Best (bundle model+qp+codec+tile) in CLI + UI.
11. 🟡 **Zoom-crop preview** — upscale a small centered crop, not the whole frame (instant at 4K).
12. 🟡 **Model manager** UI (cached/download/add HF-OpenModelDB ids; more curated models).
13. ⚪ Per-model denoise for `realesr-general-x4v3`; ⚪ cross-platform file browser (Linux `/` roots).

**Output quality**
14. 🟡 10-bit encode option (hevc main10/p010) to cut banding on upscaled gradients.
15. ⚪ Compare-with-lanczos toggle.

**Foundation**
16. ✅ DONE — `tests/test_engine.py`, 24 pure-logic tests (no GPU). Run `python -m pytest`.
17. ✅ DONE — `pyproject.toml` + `upscale-video` console entry point (`pip install -e ".[dev]"`).
18. ⚪ CI smoke test (GitHub Action running pytest + `--dry-run` on push).

## 10. Conventions
- House style: single-file engine, colorama `info/warn/err/ok/div`, argparse, collision-aware output.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` (adjust for whoever runs it).
- Keep it dependency-light where possible; heavy ML deps lazy-imported.
- **No "Topaz" references** in code, docs, or commit messages.
