# Video Upscaler — Developer Handoff

> **▶ RESUMING? START HERE.** Read ONLY this file (it's self-contained), then ask the user for the
> task. Do NOT re-explore the codebase or re-read every file — that wastes the user's usage. Run
> `python -m pytest` (fast, no GPU) only if you changed engine logic. `git push` works over SSH;
> **`git push --force` is blocked** — you'll be handed the command. Update+commit this file when done.
>
> **CHEAP RESUME (paste this to start a session):** *"Read handoff.md. Then do: <one task>. Don't
> re-explore."* Keep the task specific. The whole project is already built & working — sessions are
> for small changes, not re-derivation.
>
> **HARDWARE / COMPATIBILITY (a recurring question):** ONE codebase runs on ANY NVIDIA card. It is
> NOT multi-GPU or networked — it adapts to whatever card it's on. `--tile 0` (default) auto-sizes
> the tile from that card's free VRAM (~512 on a 10 GB 3080, ~1024 on a 24 GB 3090). If a model/tile
> is too big it **catches CUDA OOM and auto-shrinks the tile and retries** (Upscaler.enhance), so the
> same model just runs slower on a smaller card instead of crashing. Bigger card = heavier models
> (DAT/SwinIR/x4plus) + diffusion (SUPIR/SD, 24 GB). Nothing to configure per machine.

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
tool wraps it with split/resume/concat/audio-mux. The plumbing is BUILT + VERIFIED with an ffmpeg
stand-in (`_run_external_segment` / `_external_args`, run_job external branch, `--external-cmd` CLI).

### ✅ Done (our side, verified on the 10 GB box)
- Per-segment external orchestration, placeholder substitution ({input}{output}{width}{height}{target}),
  split → resume → concat → audio/sub mux around the external command. Unit-tested + stand-in verified.

### ⬜ What's LEFT to finish tier 3 (do on the 24 GB box — needs SeedVR2 + big VRAM)
1. **Install SeedVR2 in its OWN environment** (do NOT add to our venv): clone `ByteDance-Seed/SeedVR`,
   create a separate conda/venv, install its deps (torch + diffusers + xformers + flash-attn, maybe
   apex). **Linux strongly preferred** — xformers/flash-attn have wheels there for a supported Python;
   on Windows/py3.14 they don't. The Z:\ server (has `stable-diffusion`/`tdarr` already) is the natural home.
2. **Download weights** — SeedVR2 3B (FP8/GGUF quant to fit 24 GB) from its HuggingFace; note the path.
3. **Find SeedVR2's real inference interface** (I could NOT verify it — the repo page was sparse). Read
   its `infer*.py`/README: does it take a **video file** or a **folder of frames**? What are the exact
   flags (input, output, resolution, model path, tile/vram, steps)?
4. **If it's frames-folder-based** (likely), write a tiny wrapper script `seedvr2_vidwrap.sh/py` that:
   `ffmpeg extract {input} → PNG dir` → run SeedVR2 on the dir → `ffmpeg reassemble → {output}` (video-only,
   at {target}). Then point `--external-cmd` at that wrapper. If it's already video-in/video-out, skip this.
5. **VRAM-tune on 24 GB**: pick FP8/quant, set SeedVR2's internal tile/chunk so 4K fits; SeedVR2 has its
   own temporal window — our `--segment-seconds` just bounds resume granularity. Use short segments
   (30–60s) because diffusion is slow and you want frequent resume points.
6. **Verify**: output is {target} res, plays, audio preserved (we mux it), duration matches; check the
   segment-boundary seams are acceptable (SeedVR2 resets temporal state per segment, like our other tiers).
7. **(Optional) polish**: add a `--diffusion` convenience alias that fills in the `--external-cmd` for a
   known SeedVR2 install path; expose `external_cmd` in the web UI settings (currently CLI-only — the
   JobQueue already carries settings, the frontend just lacks a field); a `models/` note for the weights.
8. **(Optional, better) in-process** later: if anyone packages SeedVR2 as a pip lib, wrap it as a class
   with `is_temporal=True` + `enhance_clip(frames)->frames` and it drops into `_vsr_stream` directly —
   same seam as `VSRUpscaler`. External-cmd is the pragmatic path for now.

The architecture bet paid off: **the multi-frame window (`_vsr_stream`) AND the external per-segment
seam are both built** — temporal VSR uses the window now; a diffusion model plugs in via `--external-cmd`
today (or in-process later via the same `enhance_clip`/`is_temporal` protocol).

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

### Runs on BOTH the owner's machines (10 GB workstation + 24 GB server)
Same code, same commands — it is **GPU-adaptive** (auto-tile reads free VRAM: ~512 on 10 GB, ~1024 on
24 GB; auto-detects CUDA). Nothing in the normal pipeline requires a big card.
- **10 GB (RTX 3080, dev/workstation):** Tier 1 (single-image, default) and Tier 2 (`--vsr`) both run
  here — all testing was done on this card. Tier 3 diffusion is NOT usable (VRAM) — just don't pass
  `--external-cmd`. For `--vsr` on 10 GB keep `--vsr-window` modest (8–16) to avoid OOM (it's fp32).
- **24 GB (server):** all three tiers, incl. diffusion via `--external-cmd`. Bigger `--vsr-window` OK.
- The tiers are independent: diffusion runs as a separate external process, so it never affects or
  burdens the normal single-image/VSR pipeline. Nothing to configure differently per machine except
  whether you add `--external-cmd`.

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
**Engine:** upscale (480p→4K, 1080p→4K on real film), audio/subs preserved, metadata; segmentation +
concat; resume across restarts; RIFE interpolation (2× and non-integer); temporal VSR (`--vsr`, ccrestoration
AnimeSR/BasicVSR/IconVSR/EDVR, fp32); external/diffusion seam (`--external-cmd`, SeedVR2 path); auto-tile;
upfront ETA; lossless dup-frame dedup; per-model `--benchmark`; quality presets (`--quality`).
**Real-media (auto-detected):** HDR/10-bit → tonemap to SDR; interlaced → yadif; VFR → CFR-at-avg + warn;
**anamorphic → un-squeeze to display aspect** (DVD/VOB/3gp not distorted). 30+ input extensions.
**Web UI:** file browser, **Start/Pause/Stop queue control (no auto-run on add)**, live queue + WebSocket
progress, before/after **zoom-crop** preview, ⚡ Benchmark button, **quality-preset buttons**, **output-folder
Browse button**, model dropdown, persistence, clear/remove, cache-busted assets, localhost+token security.
**Foundation:** 31 pytest tests (pure logic), pyproject `upscale-video` entry point. All committed & pushed.

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

## 9. TODO — what's LEFT (🔴 high / 🟡 med / ⚪ low)  [much of the original roadmap is now DONE — see §5]
**Biggest remaining item**
- 🔴 **Finish tier-3 SeedVR2 diffusion** on the 24 GB box (see §0 + the "What's LEFT to finish tier 3"
  checklist above) — the `--external-cmd` seam is built; SeedVR2 itself needs installing + wiring there.

**Web UI polish (CLI-only features not yet in the dashboard)**
- 🟡 Expose **`--vsr` temporal toggle**, **`--external-cmd`**, and **upfront ETA** in the web UI.
- 🟡 **Model manager** panel (browse/download/see-cached, add HF/OpenModelDB ids with progress).
- 🟡 Show detected **source flags** (HDR/interlaced/VFR/anamorphic badges) when a file is picked.

**Engine niceties**
- 🟡 **Format robustness** (from the formats plan): split to a robust intermediate (.mkv/.ts) for exotic
  containers that don't `-c copy` segment cleanly; **audio/sub mux fallback** (re-encode to AAC when copy
  fails, e.g. RealAudio). Anamorphic + extensions are already done.
- 🟡 `--dedup-threshold` for near-duplicate frames (exact-match barely triggers on lossy video).
- ⚪ Content-aware `--model auto` (needs anime-vs-live detection — hard; a UI hint may be enough).
- ⚪ 10-bit HEVC encode option (p010) to cut banding; ⚪ temporal-VSR fp16 (upstream ccrestoration bug).

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
