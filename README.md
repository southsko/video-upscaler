# 🔍 Video Upscaler

> Free, self-hosted **AI video upscaling** with open models and a polished local web UI.

![Python](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c)
![ffmpeg](https://img.shields.io/badge/ffmpeg-NVENC-007808)
![platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

Frames stream straight through the GPU — `decode → GPU → encode` with **no PNG files ever hitting
disk** — using open super-resolution models (Real-ESRGAN and any
[`spandrel`](https://github.com/chaiNNer-org/spandrel)-loadable checkpoint: ESRGAN, SwinIR, HAT, DAT,
SPAN, …, or a HuggingFace id). Optional RIFE frame interpolation. Drive it from the command line or a
local web dashboard.

Companion to [`pi_convert.py`](https://github.com/southsko/pi-tv) (downscaler) and
[`drone-footage-merger`](https://github.com/southsko/drone-footage-merger) (joiner) — same house style.

---

## Highlights

| | |
|---|---|
| 🧠 **Any model** | Real-ESRGAN, SwinIR, HAT, DAT, SPAN, Compact, … from a builtin name, a local `.pth`/`.safetensors`, or a HuggingFace / OpenModelDB id |
| ⚡ **Efficient** | GPU-resident raw-video pipe, threaded to keep the GPU saturated — **no intermediate PNGs on disk** |
| ⏯️ **Robust resume** | Segmented processing; an interrupted run skips the segments it already finished, even across restarts |
| 🎞️ **Frame interpolation** | Optional RIFE fps boost at *any* ratio (24/25/30 → 48/60…), via fractional timesteps |
| 📺 **4K by default** | Any custom `WxH`; tuned NVENC encode (H.264 / HEVC); auto tile-sizing from free VRAM |
| 🖥️ **Polished web UI** | Server-side file browser, live queue with WebSocket progress, before/after comparison slider, persistent queue |
| 🔌 **Headless friendly** | CLI and web UI share one engine; batch whole folders over SSH |

---

## How it works

Per file, everything streams through memory — the only thing on disk is the finished output (and,
optionally, keyframe-aligned source segments so runs can resume):

```
ffprobe ─► {res, fps, duration, audio/subs}
   │  split losslessly into resumable segments (keyframe-aligned)
   ▼
 ffmpeg DECODE ──rgb24 rawvideo──►┐
                                  │  threaded pipeline (GPU stays saturated)
        reader → [RIFE interpolate?] → Real-ESRGAN upscale (tiled, fp16) → writer
                                  │
                                  └──rgb24 rawvideo──► ffmpeg ENCODE
                                       (lanczos scale+pad to exact target, NVENC)
   │  concat segments (-c copy)
   ▼
 mux original audio + subtitles + metadata ─► <name>_upscaled.mkv
```

Real-ESRGAN is native ×2/×4; the exact target size is reached with a final lanczos `scale`+`pad`,
so any output resolution works.

---

## Setup

**Verified** on Windows 11, **Python 3.14.6**, **torch 2.11.0+cu128**, spandrel 0.4.2, ccvfi 0.0.3,
opencv 5.0, ffmpeg 8.1, NVIDIA RTX 3080 (driver 595.79). The whole stack has Python 3.14 wheels — no
separate Python needed.

### Quick setup

```powershell
python upscale_video.py --setup       # creates .venv, installs CUDA torch + everything
```

Then run through the venv, e.g. `.\.venv\Scripts\python.exe upscale_video.py ...`.

### Manual setup (equivalent)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1) PyTorch CUDA build FIRST (the default `pip install torch` is CPU-only!)
#    cu128 matches modern NVIDIA drivers; older drivers can use cu126/cu124.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) everything else
pip install -r requirements.txt

# 3) verify the GPU is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Also needs **`ffmpeg`** + **`ffprobe`** on `PATH` (a full build, for NVENC).

### Docker (recommended for servers)

One command — pulls CUDA, PyTorch, ffmpeg, and all dependencies into an isolated container with GPU passthrough.

**Requirements:** Docker, an NVIDIA GPU, and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

```bash
git clone https://github.com/southsko/video-upscaler.git
cd video-upscaler
docker compose up -d --build
```

The UI is at **http://your-server:8848**. Edit `docker-compose.yml` to change the media mount path:

```yaml
volumes:
  - /path/to/your/media:/media:ro   # ← change this
```

**What's included:**
- NVIDIA GPU passthrough (NVENC encoding + CUDA inference)
- Media mounted read-only at `/media`
- Models persisted in `./models` (survives rebuilds)
- Logs in `./logs/` with rotation (50MB × 5 files)
- Auto-restart on crash (`unless-stopped`)
- Memory (32GB) and CPU (8 cores) limits — adjust in `docker-compose.yml`

**Useful commands:**
```bash
docker compose logs -f          # follow logs
docker compose restart          # restart
docker compose down             # stop
docker compose up -d --build    # rebuild after updates
```

**Gotchas**
- If `torch.cuda.is_available()` is `False`, you installed the CPU wheel — reinstall from the cu128 index.
- Pick the CUDA index for your driver: newest → `cu128`, then `cu126`, `cu124`. Very new Python
  versions only appear on the newer indexes.
- Avoid the Windows **Store** Python (under `WindowsApps`) — sandboxed and flaky for native packages.

---

## Usage

### CLI

```powershell
# Single file, 480p cartoon → 4K with the anime-video model (default)
python upscale_video.py "D:\show\s01e02.mp4"

# A whole folder (recursive), to 1080p, HEVC
python upscale_video.py "D:\show\Season 1" --target 1080p --codec hevc_nvenc

# A specific HuggingFace / community model, exact custom size
python upscale_video.py clip.mkv --model 4x-UltraSharp --target 2560x1440

# Upscale AND boost frame rate to 60fps (RIFE interpolation)
python upscale_video.py clip.mkv --interpolate --fps 60

# Temporal VSR (multi-frame, less flicker than single-image) — great for anime
python upscale_video.py clip.mkv --vsr --vsr-model AnimeSR_v2_4x --vsr-window 16

# Orchestrate an external upscaler per segment (its own env/GPU) — e.g. a heavy
# diffusion model like SeedVR2. We handle split/resume/concat/audio-mux around it.
python upscale_video.py clip.mkv --target 4k --segment-seconds 60 \
  --external-cmd "python /opt/SeedVR2/infer.py --input {input} --output {output} --res {target}"

# Compare model speeds on YOUR GPU (run on an idle GPU for accurate numbers)
python upscale_video.py clip.mkv --benchmark

# See exactly what it would run, without running it
python upscale_video.py clip.mkv --dry-run
```

`--list-models` prints the builtin shortlist. `python upscale_video.py --help` shows the full option
surface (target, model, tile/fp16/gpu, codec/qp/preset, segmentation/resume, interpolation, …).

### Web UI

```powershell
python upscale_video.py --serve --open
```

Opens a local dashboard (default `http://127.0.0.1:8848`): browse the server's drives, queue files,
watch live per-job progress, and use the before/after slider to tune settings **before** committing a
long run.

> **🔒 Security:** the UI can read the server's filesystem, so it binds to **localhost only** by
> default. To reach it from another device use `--host 0.0.0.0`, which prints a required access
> **token** — only expose it on a trusted LAN.

---

## Performance

AI upscaling is heavy — you're generating 4× the pixels with a neural net — so **model choice
dominates everything**. Measured on an RTX 3080:

| Source → 4K | Model | Arch | Throughput | ~2-hour movie |
|---|---|---|---|---|
| 480p → 4K | `realesr-animevideov3` | tiny SRVGGNet | ~27 fps | ~real-time |
| 1080p → 4K | `realesr-animevideov3` (anime) | tiny SRVGGNet | ~6.5 fps | ~7 hours |
| 1080p → 4K | `2x-nomos-span` (live-action) | SPAN | ~2.8 fps | **~17 hours** |
| 1080p → 4K | `realesrgan-x4plus` (heavy) | RRDBNet | ~0.25 fps | **~8 days** ⚠️ |

Takeaways:
- **Speed is set by the model's *architecture size*, not the upscale factor.** The backbone runs at
  the *input* resolution, so a ×2 and a ×4 model of the same size cost about the same; the output
  scale only affects the tiny final upsample. Small backbones (SRVGGNet / SPAN) are the only
  practical choice for full-length video.
- **Pick by content:** `realesr-animevideov3` for anime/cartoons (fastest), **`2x-nomos-span` for
  live-action film** (fast SPAN, native 4K from 1080p — the practical general choice). Heavy RRDBNet
  models (`x4plus`) are lovely for stills but impractical for video (~8 days/movie).
- The tool **benchmarks the first file and prints an ETA before starting** (disable with `--no-eta`),
  so a multi-hour — or multi-*day* — job is never a surprise.
- The pipeline is already GPU-compute-bound and well-overlapped; batching and `torch.compile` were
  measured to give no gain on these models, so they aren't used.

## Notes & caveats

- **GPU / tiling:** `--tile 0` (default) auto-sizes the tile from free VRAM (512 on a 10 GB card);
  set `--tile N` to override. Ampere cards decode but **can't encode AV1** — `av1_nvenc` needs an
  RTX 40-series; default is `h264_nvenc` (or `hevc_nvenc`).
- **Resume across restarts:** the scratch dir is keyed by a hash of source + key settings (not the
  run's job id), so re-running an interrupted job skips finished segments.
- **Persistent web queue:** jobs are saved to `jobs.json` and restored on restart — unfinished jobs
  re-queue (and resume), done/failed stay as history until you *Clear finished*; jobs whose source
  has moved are dropped.
- **Real-media handling (auto-detected from the source):**
  - **HDR / 10-bit** (PQ/HLG, bt2020) → tonemapped to SDR before upscaling, since the AI models are
    SDR-domain (output is SDR bt709). Disable with `--no-tonemap` (HDR will then look washed out).
  - **Interlaced** → deinterlaced (yadif). Control with `--deinterlace auto|on|off` (default auto).
  - **VFR** (variable frame rate) → output is constant-fps at the average rate; audio stays aligned
    at the ends. `--dry-run` prints all detected flags (bit depth, HDR, interlaced, VFR).
- **Models** download on first use (with a confirmation) into `--weights-dir` and are cached.
- **Interpolation** (`--interpolate`) uses RIFE via [`ccvfi`](https://pypi.org/project/ccvfi/)
  (weights auto-download). `--interp-order pre` (default) interpolates at source res then upscales
  (fast); `post` upscales then interpolates at target res (slower, sharper motion). It runs
  independently per segment, so one bridging frame is dropped at each segment boundary (negligible).
- Uses **spandrel** instead of `realesrgan`/`basicsr` — broader model support, and it sidesteps the
  well-known `basicsr` crash importing the removed `torchvision.transforms.functional_tensor`.

---

## Development

```powershell
pip install -e ".[dev]"     # editable install + pytest; gives an `upscale-video` command
python -m pytest            # 24 pure-logic tests (no GPU needed): target math, RIFE resample
                            # frame-counts, scratch keying, encoder args, dedup, output naming
```

## Built on

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) & the ESRGAN community models (BSD-3)
- [spandrel](https://github.com/chaiNNer-org/spandrel) — architecture-detecting model loader
- [ccvfi](https://github.com/EutropicAI/ccvfi) / [RIFE](https://github.com/hzwer/Practical-RIFE) — frame interpolation
- [FFmpeg](https://ffmpeg.org/) — decode, NVENC encode, mux
- [FastAPI](https://fastapi.tiangolo.com/) + [PyTorch](https://pytorch.org/)

---

## License

The tool itself is provided as-is for personal use. **Individual models carry their own licenses**
(many permissive, some non-commercial) — check the terms of any model you download.
