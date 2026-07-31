#!/usr/bin/env python3
"""Video Upscaler — a free, open-model AI video upscaler.

Efficient by design: frames stream
``ffmpeg decode -> GPU inference -> ffmpeg NVENC`` with NO PNG files hitting
disk. Upscaling is done by open super-resolution models (Real-ESRGAN and any
other checkpoint spandrel can load: ESRGAN, SwinIR, HAT, DAT, SPAN, ...),
loaded from a builtin name, a local .pth/.safetensors, or a HuggingFace id.

Companion to pi_convert.py (downscaler) and merge_videos.py (joiner) — same
house style (colorama logging, tiered NVENC detection, collision-aware output,
folder recursion + interactive browser).

    python upscale_video.py "D:\\show\\s01e02.mp4"            # 480p -> 4K
    python upscale_video.py "D:\\show\\Season 1" --target 1080p --codec hevc_nvenc
    python upscale_video.py clip.mkv --model 4x-UltraSharp --dry-run
    python upscale_video.py --serve --open                    # polished web UI

Heavy deps (torch, spandrel, numpy, opencv) are imported lazily, so --help,
--dry-run, --list-models, probing and encoder detection work without them.
See README.md for setup (venv + CUDA torch; note the Python 3.14/3.12 caveat).
"""
import argparse
import glob
import hashlib
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

# Windows consoles / redirected pipes often default to cp1252, which can't encode
# the arrows/box characters we print — force UTF-8 so logging never crashes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _setup_logging():
    """Configure logging to both console and file."""
    log_dir = os.environ.get("LOG_DIR", "/app/logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("video-upscaler")
    logger.setLevel(logging.DEBUG)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(ch)
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "upscaler.log"), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(fh)
    return logger

log = _setup_logging()

# ── colorama (optional, nicer colours) ────────────────────────────────────────
try:
    from colorama import init as _color_init, Fore, Back, Style
    _color_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

    class _D:
        def __getattr__(self, _):
            return ''
    Fore = Back = Style = _D()


def info(m):  print(f"{Fore.CYAN}{Style.BRIGHT}[INFO]{Style.RESET_ALL}  {m}")
def warn(m):  print(f"{Fore.YELLOW}{Style.BRIGHT}[WARN]{Style.RESET_ALL}  {m}")
def err(m):   print(f"{Fore.RED}{Style.BRIGHT}[ERR] {Style.RESET_ALL}  {m}")
def ok(m):    print(f"{Fore.GREEN}{Style.BRIGHT}[OK]  {Style.RESET_ALL}  {m}")
def div():    print(f"{Fore.BLUE}{Style.BRIGHT}{'-' * 62}{Style.RESET_ALL}")


# ── constants / config (env-overridable) ──────────────────────────────────────
# Extensions used for FOLDER scans + the web file browser. (Files passed directly
# by path are attempted regardless of extension — ffmpeg decodes far more than this.)
_VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv", ".wmv", ".ts",
    ".mpg", ".mpeg", ".mpe", ".m2ts", ".mts", ".m2v", ".m1v", ".mpv",
    ".3gp", ".3g2", ".ogv", ".ogm", ".vob", ".divx", ".asf", ".rm", ".rmvb",
    ".mxf", ".y4m", ".f4v", ".dv", ".nut", ".qt",
}
VIDEO_EXTENSIONS = [f"*{e}" for e in sorted(_VIDEO_EXTS)]   # glob patterns (case-insensitive FS)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_DIR = os.environ.get("UPSCALE_WEIGHTS_DIR", os.path.join(HERE, "models"))

TARGET_PRESETS = {
    "4k": (3840, 2160), "2160p": (3840, 2160), "uhd": (3840, 2160),
    "1440p": (2560, 1440), "2k": (2560, 1440),
    "1080p": (1920, 1080), "fhd": (1920, 1080),
    "720p": (1280, 720),
}

# Builtin models: name -> (download url, native scale, note). Anything spandrel
# can load also works via a local path or a HuggingFace id, so this is just a
# curated shortcut list (printed by --list-models).
BUILTIN_MODELS = {
    # name: (source, native_scale, note).  source = direct URL or "hf:repo[/id][:file]".
    # FAST models (small SRVGGNet/SPAN backbones) — the only practical ones for full video.
    "realesr-animevideov3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        4, "Anime/cartoon VIDEO — FASTEST (~6.5fps @1080p→4K on a 3080). Default."),
    "2x-nomos-span": (
        "hf:Phips/2xNomosUni_span_multijpg_ldl",
        2, "General / LIVE-ACTION (SPAN, fast, ~2.8fps). Best practical film upscaler; native 4K from 1080p."),
    "2x-parimg-compact": (
        "hf:Phips/2xParimgCompact",
        2, "Photo / live-action (Compact, fast). Native 4K from 1080p."),
    "realesr-general-x4v3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        4, "General purpose, x4 (compact, moderate speed)."),
    # HEAVY models (RRDBNet) — great for stills, IMPRACTICAL for full video (~0.25fps @1080p→4K).
    "realesrgan-x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        4, "General/live-action, HIGH QUALITY but VERY SLOW (~0.25fps @1080p→4K ≈ 8 days/movie). Stills only."),
    "realesrgan-x4plus-anime": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        4, "Anime stills, x4 (lighter 6-block RRDBNet)."),
}
DEFAULT_MODEL = os.environ.get("UPSCALE_MODEL", "realesr-animevideov3")

# High-quality NVENC defaults (constqp, spatial AQ, p7/hq).
ENC_DEFAULTS = {
    "codec": "h264_nvenc", "qp": "18", "preset": "p7", "tune": "hq",
    "pix_fmt": "yuv420p", "rc_lookahead": "20", "aq_strength": "15",
    "gop": "30", "bf": "0",
}

# One-click quality/speed bundles (model + encode). Labels shown in --list-presets
# and the web UI so the model/speed tradeoff is explicit (no content guessing).
QUALITY_PRESETS = {
    "fast":     {"model": "realesr-animevideov3", "qp": "19", "codec": "h264_nvenc",
                 "label": "Fastest — anime/cartoon model, H.264"},
    "balanced": {"model": "2x-nomos-span", "qp": "17", "codec": "h264_nvenc",
                 "label": "Balanced — fast live-action SPAN model, H.264 (recommended)"},
    "best":     {"model": "2x-nomos-span", "qp": "14", "codec": "hevc_nvenc",
                 "label": "Best — same fast model, higher-quality HEVC encode"},
    "max":      {"model": "realesrgan-x4plus", "qp": "13", "codec": "hevc_nvenc",
                 "label": "Max detail — heavy model, VERY slow (~days/movie)"},
}

DEFAULT_PORT = int(os.environ.get("UPSCALE_PORT", "8848"))
DEFAULT_SEGMENT_SECONDS = int(os.environ.get("UPSCALE_SEGMENT_SECONDS", "300"))


# ── small utils ───────────────────────────────────────────────────────────────
def _fmt(secs):
    secs = int(max(0, secs))
    if secs >= 3600:
        return "%d:%02d:%02d" % (secs // 3600, (secs % 3600) // 60, secs % 60)
    return "%d:%02d" % (secs // 60, secs % 60)


def _bar(frac, width=22):
    frac = max(0.0, min(1.0, frac))
    fill = int(frac * width)
    return '[' + '#' * fill + '-' * (width - fill) + ']'


def _human_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0


def _run(cmd, timeout=None):
    """Run, return (returncode, stdout, stderr) as text."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def _quote(cmd):
    """Render an argv list as a copy-pasteable command line (for --dry-run)."""
    out = []
    for a in cmd:
        a = str(a)
        out.append(f'"{a}"' if (' ' in a or ':' in a[2:] or ',' in a) else a)
    return " ".join(out)


# ── ffprobe metadata ──────────────────────────────────────────────────────────
def probe(path):
    """Return a dict of stream/format info, or raise RuntimeError."""
    rc, out, errtxt = _run([
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', path])
    if rc != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {errtxt.strip()[:200]}")
    data = json.loads(out)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError(f"No video stream in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])
    has_subs = any(s.get("codec_type") == "subtitle" for s in data["streams"])

    # frame rate can be "30000/1001"
    def _rat(s, default=0.0):
        try:
            if s and "/" in s:
                a, b = s.split("/")
                return float(a) / float(b) if float(b) else default
            return float(s)
        except (ValueError, ZeroDivisionError, TypeError):
            return default

    fps = _rat(v.get("avg_frame_rate")) or _rat(v.get("r_frame_rate")) or 0.0
    dur = 0.0
    for src in (v.get("duration"), data.get("format", {}).get("duration")):
        try:
            dur = float(src)
            if dur > 0:
                break
        except (TypeError, ValueError):
            continue
    nb = v.get("nb_frames")
    try:
        nb_frames = int(nb)
    except (TypeError, ValueError):
        nb_frames = int(round(fps * dur)) if (fps and dur) else 0

    pix_fmt = v.get("pix_fmt", "yuv420p")
    bit_depth = 10 if any(t in pix_fmt for t in ("10le", "10be", "p010", "p210")) \
        else (12 if "12" in pix_fmt else 8)
    trc = (v.get("color_transfer") or "").lower()
    prim = (v.get("color_primaries") or "").lower()
    is_hdr = trc in ("smpte2084", "arib-std-b67") or prim == "bt2020"
    field_order = (v.get("field_order") or "progressive").lower()
    is_interlaced = field_order in ("tt", "bb", "tb", "bt")
    r_fps = _rat(v.get("r_frame_rate"))
    is_vfr = bool(fps and r_fps and abs(r_fps - fps) / fps > 0.02)

    width, height = int(v.get("width", 0)), int(v.get("height", 0))
    disp_w, disp_h = width, height          # display (square-pixel) dimensions
    try:                                    # anamorphic (non-square pixels): DVD/VOB/3gp
        sn, sd = (v.get("sample_aspect_ratio") or "1:1").split(":")
        sn, sd = int(sn), int(sd)
        if sn > 0 and sd > 0 and sn != sd:
            disp_w = int(round(width * sn / sd))   # correct horizontal stretch
            disp_w -= disp_w % 2                     # keep even for encoders
    except (ValueError, AttributeError, TypeError):
        pass
    is_anamorphic = (disp_w, disp_h) != (width, height)
    return {
        "path": path,
        "width": width,
        "height": height,
        "disp_w": disp_w,
        "disp_h": disp_h,
        "is_anamorphic": is_anamorphic,
        "fps": fps,
        "duration": dur,
        "nb_frames": nb_frames,
        "pix_fmt": pix_fmt,
        "vcodec": v.get("codec_name", "?"),
        "has_audio": has_audio,
        "has_subs": has_subs,
        "bit_depth": bit_depth,
        "color_transfer": trc,
        "is_hdr": is_hdr,
        "field_order": field_order,
        "is_interlaced": is_interlaced,
        "is_vfr": is_vfr,
    }


def describe_source(meta):
    """Short human summary of source flags for logs / dry-run / UI."""
    tags = [f"{meta['bit_depth']}-bit"]
    if meta.get("is_hdr"):
        tags.append(f"HDR({meta.get('color_transfer') or 'bt2020'})")
    if meta.get("is_interlaced"):
        tags.append(f"interlaced({meta['field_order']})")
    if meta.get("is_vfr"):
        tags.append("VFR")
    if meta.get("is_anamorphic"):
        tags.append(f"anamorphic→{meta['disp_w']}x{meta['disp_h']}")
    return ", ".join(tags)


# ── target / scale math ───────────────────────────────────────────────────────
def parse_target(spec):
    """'4k' | '1080p' | '3840x2160' -> (w, h)."""
    if spec is None:
        return TARGET_PRESETS["4k"]
    key = spec.strip().lower()
    if key in TARGET_PRESETS:
        return TARGET_PRESETS[key]
    for sep in ("x", ":", "X"):
        if sep in key:
            a, b = key.split(sep, 1)
            return int(a), int(b)
    raise ValueError(f"Unrecognised target '{spec}' (try 4k, 1080p, or WxH)")


def model_output_dims(src_w, src_h, scale):
    return src_w * scale, src_h * scale


# ── model layer (lazy torch/spandrel) ─────────────────────────────────────────
def resolve_model(spec, weights_dir=DEFAULT_WEIGHTS_DIR, assume_yes=False,
                  confirm=None):
    """Resolve a --model spec to a local checkpoint path, downloading if needed.

    spec may be: a builtin name, an existing local file, or a HuggingFace id
    ('repo/id' or 'repo/id:filename.safetensors').
    confirm(msg)->bool is called before any download (defaults to a CLI prompt).
    """
    # 1) existing local file
    if os.path.isfile(spec):
        return os.path.abspath(spec)

    os.makedirs(weights_dir, exist_ok=True)

    # 2) builtin name
    if spec in BUILTIN_MODELS:
        url, _scale, _note = BUILTIN_MODELS[spec]
        if url.startswith("hf:"):                    # HF-backed builtin
            return resolve_model(url[3:], weights_dir, assume_yes, confirm)
        fname = os.path.basename(url)
        dest = os.path.join(weights_dir, fname)
        if os.path.isfile(dest):
            return dest
        if not _ask_download(f"model '{spec}' ({fname})", url, assume_yes, confirm):
            raise RuntimeError(f"Download of model '{spec}' declined.")
        _download(url, dest)
        return dest

    # 3) HuggingFace id (optionally repo:file)
    if "/" in spec:
        repo, _, filename = spec.partition(":")
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise RuntimeError("huggingface_hub not installed — pip install huggingface_hub")
        if not filename:
            files = [f for f in list_repo_files(repo)
                     if f.lower().endswith((".safetensors", ".pth", ".pt", ".ckpt"))]
            if not files:
                raise RuntimeError(f"No model weights found in HF repo '{repo}'")
            filename = sorted(files, key=len)[0]
        if not _ask_download(f"HF model '{repo}/{filename}'", f"hf://{repo}/{filename}",
                             assume_yes, confirm):
            raise RuntimeError(f"Download of HF model '{spec}' declined.")
        return hf_hub_download(repo_id=repo, filename=filename, cache_dir=weights_dir)

    raise RuntimeError(
        f"Unknown model '{spec}'. Use a builtin (--list-models), a local file, "
        f"or a HuggingFace id like 'user/repo' or 'user/repo:file.safetensors'.")


def _ask_download(what, url, assume_yes, confirm):
    if assume_yes:
        return True
    if confirm is not None:
        return confirm(f"Download {what} from {url}?")
    try:
        resp = input(f"{Fore.YELLOW}Download {what}?{Style.RESET_ALL}\n  {url}\n  [Y/n] ")
        return resp.strip().lower() in ("", "y", "yes")
    except EOFError:
        return False


def _download(url, dest):
    info(f"Downloading {os.path.basename(dest)} ...")
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    frac = got / total
                    sys.stdout.write(f"\r  {_bar(frac)} {int(frac*100):3d}%  "
                                     f"{_human_size(got)}/{_human_size(total)}")
                    sys.stdout.flush()
    sys.stdout.write("\r" + " " * 60 + "\r")
    os.replace(tmp, dest)
    ok(f"Saved {dest}")


class Upscaler:
    """Loads a spandrel model and upscales RGB uint8 frames with tiling."""

    def __init__(self, model_path, gpu=0, fp16=True, tile=512, tile_pad=16,
                 denoise=None):
        import torch
        from spandrel import ModelLoader, ImageModelDescriptor
        self.torch = torch
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            warn("CUDA not available — running on CPU (very slow). Check the "
                 "torch install (needs the CUDA wheel).")
        desc = ModelLoader(device=self.device).load_from_file(model_path)
        if not isinstance(desc, ImageModelDescriptor):
            raise RuntimeError(f"{os.path.basename(model_path)} is not an image "
                               f"super-resolution model spandrel can drive.")
        desc.to(self.device).eval()
        self.fp16 = bool(fp16 and desc.supports_half and self.device.type == "cuda")
        if self.fp16:
            desc.model.half()
        self.desc = desc
        self.scale = desc.scale
        self.tile_pad = int(tile_pad)
        self.tile = int(tile) if int(tile) > 0 else self._auto_tile()
        self.name = os.path.basename(model_path)

    def _auto_tile(self):
        """Pick a tile size from free VRAM so 4K frames don't OOM."""
        try:
            free, _total = self.torch.cuda.mem_get_info(self.device)
            gb = free / (1024 ** 3)
        except Exception:                            # noqa: BLE001 (CPU / no cuda)
            return 256
        for thr, t in ((20, 1024), (12, 768), (8, 512), (6, 384), (4, 256)):
            if gb >= thr:
                return t
        return 192

    def _forward(self, t):
        with self.torch.no_grad():
            return self.desc(t)

    def _tiled(self, t):
        torch = self.torch
        _, c, h, w = t.shape
        s, tile, pad = self.scale, self.tile, self.tile_pad
        if tile <= 0 or (h <= tile and w <= tile):
            return self._forward(t)
        out = t.new_zeros((1, c, h * s, w * s))
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                ex, ey = min(x + tile, w), min(y + tile, h)
                px0, py0 = max(x - pad, 0), max(y - pad, 0)
                px1, py1 = min(ex + pad, w), min(ey + pad, h)
                tile_out = self._forward(t[:, :, py0:py1, px0:px1])
                # crop the padded border back off, place into the canvas
                cy0, cx0 = (y - py0) * s, (x - px0) * s
                oh, ow = (ey - y) * s, (ex - x) * s
                out[:, :, y * s:ey * s, x * s:ex * s] = \
                    tile_out[:, :, cy0:cy0 + oh, cx0:cx0 + ow]
        return out

    def enhance(self, img):
        """img: HxWx3 RGB uint8 ndarray -> upscaled HxWx3 RGB uint8 ndarray."""
        import numpy as np
        torch = self.torch
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        t = t.unsqueeze(0).to(self.device).float().div_(255.0)
        if self.fp16:
            t = t.half()
        # non-inplace: the no-tile path returns spandrel's inference-mode tensor,
        # which forbids in-place ops (the tiled path returns a fresh tensor).
        out = self._tiled(t).squeeze(0).permute(1, 2, 0).clamp(0, 1)
        return (out * 255).round().to(torch.uint8).cpu().numpy()


# RIFE model names available via ccvfi.ConfigType.
RIFE_MODELS = ["RIFE_IFNet_v426_heavy", "DRBA_IFNet"]
DEFAULT_RIFE_MODEL = os.environ.get("UPSCALE_RIFE_MODEL", "RIFE_IFNet_v426_heavy")


class Interpolator:
    """RIFE frame interpolation via ccvfi. interpolate(a, b, t) synthesises a
    frame at fractional time t in (0,1) between RGB uint8 frames a and b."""

    def __init__(self, device=None, fp16=True, model_name=DEFAULT_RIFE_MODEL,
                 scale=1.0, weights_dir=DEFAULT_WEIGHTS_DIR):
        import torch
        import torch.nn.functional as F
        from ccvfi import AutoModel, ConfigType
        self.torch = torch
        self.F = F
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        cfg = getattr(ConfigType, model_name)
        os.makedirs(weights_dir, exist_ok=True)
        self.model = AutoModel.from_pretrained(cfg, device=self.device,
                                               fp16=self.fp16, model_dir=weights_dir)
        self.scale = float(scale)
        self.name = model_name

    def interpolate(self, a_rgb, b_rgb, t):
        torch, F = self.torch, self.F

        def to_t(rgb):
            import numpy as np
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
            x = x.unsqueeze(0).to(self.device).float().div_(255.0)
            return x.half() if self.fp16 else x

        ta, tb = to_t(a_rgb), to_t(b_rgb)
        _, _, h, w = ta.shape
        ph, pw = (64 - h % 64) % 64, (64 - w % 64) % 64  # IFNet needs mult-of-64
        if ph or pw:
            ta = F.pad(ta, (0, pw, 0, ph), mode="replicate")
            tb = F.pad(tb, (0, pw, 0, ph), mode="replicate")
        inp = torch.stack([ta, tb], dim=1)               # (1, 2, C, H, W)
        with torch.inference_mode():
            out = self.model.inference(inp, timestep=float(t), scale=self.scale)
        out = out[:, :, :h, :w].squeeze(0).permute(1, 2, 0).clamp(0, 1)  # non-inplace
        return (out * 255).round().to(torch.uint8).cpu().numpy()


# Temporal (multi-frame) VSR models via ccrestoration — reduce flicker by using
# neighbouring frames. These are NOT spandrel-loadable; ccrestoration is the loader.
VSR_MODELS = ["AnimeSR_v2_4x", "BasicVSR_REDS_4x", "IconVSR_REDS_4x",
              "EDVR_M_SR_REDS_official_4x"]
DEFAULT_VSR_MODEL = os.environ.get("UPSCALE_VSR_MODEL", "AnimeSR_v2_4x")


class VSRUpscaler:
    """Temporal video super-resolution: processes a *window* of frames together
    for temporal consistency (less flicker) than single-image SR. Duck-types the
    bits of Upscaler that run_job needs (scale/name/device/enhance) plus
    enhance_clip() and is_temporal for the windowed pipe path."""

    is_temporal = True

    def __init__(self, config_name=DEFAULT_VSR_MODEL, gpu=0, fp16=True, tile=128,
                 tile_pad=8, window=16, weights_dir=DEFAULT_WEIGHTS_DIR):
        import torch
        from ccrestoration import AutoModel, ConfigType
        self.torch = torch
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            warn("CUDA not available — temporal VSR on CPU is extremely slow.")
        # ccrestoration's VSR inference doesn't cast inputs to half, so fp16 crashes
        # ("Input type float / bias type Half"). Force fp32 — fine on a 24GB target.
        self.fp16 = False
        cfg = getattr(ConfigType, config_name)
        os.makedirs(weights_dir, exist_ok=True)
        tilearg = (int(tile), int(tile)) if tile and int(tile) > 0 else None
        self.model = AutoModel.from_pretrained(
            cfg, device=self.device, fp16=False, tile=tilearg,
            tile_pad=tile_pad, model_dir=weights_dir)
        self.name = config_name
        self.window = max(2, int(window))
        try:
            self.scale = int(config_name.rsplit("_", 1)[-1].rstrip("xX"))
        except ValueError:
            self.scale = int(getattr(self.model, "scale", 4))

    def enhance_clip(self, rgb_frames):
        """List of HxWx3 RGB uint8 -> list of upscaled HxWx3 RGB uint8.
        ccrestoration works in BGR, so reverse channels around it."""
        import numpy as np
        bgr = [np.ascontiguousarray(f[:, :, ::-1]) for f in rgb_frames]
        out = self.model.inference_image_list(bgr)
        return [np.ascontiguousarray(o[:, :, ::-1]) for o in out]

    def enhance(self, frame):            # single-frame path (ETA/preview)
        return self.enhance_clip([frame])[0]


def _vsr_stream(frames, enhance_clip, window):
    """Feed frames to a temporal VSR model in non-overlapping windows. Frame-exact
    (one output per input); a small discontinuity can occur at each window edge
    (every `window` frames ~1s) — acceptable, like the segment boundaries."""
    buf = []
    for f in frames:
        buf.append(f)
        if len(buf) >= window:
            for o in enhance_clip(buf):
                yield o
            buf = []
    if buf:
        for o in enhance_clip(buf):
            yield o


# ── encoder detection (ported/adapted from pi_convert.py) ─────────────────────
def _encoder_listed(codec):
    try:
        rc, out, _ = _run(['ffmpeg', '-hide_banner', '-encoders'], timeout=15)
        return codec in out
    except (OSError, subprocess.TimeoutExpired):
        return False


def _encoder_works(vargs):
    """(ok, last_error_line) from a tiny real test encode."""
    try:
        rc, _, errtxt = _run(
            ['ffmpeg', '-hide_banner', '-f', 'lavfi', '-i',
             'color=c=black:s=320x240:d=0.3'] + vargs + ['-f', 'null', '-'],
            timeout=25)
        if rc == 0:
            return True, ""
        lines = [ln for ln in errtxt.splitlines() if ln.strip()]
        return False, (lines[-1] if lines else "unknown error")
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)


def build_video_args(opts):
    """NVENC/x264 video-encode args from an options dict (high-quality defaults)."""
    codec = opts.get("codec", ENC_DEFAULTS["codec"])
    qp = str(opts.get("qp", ENC_DEFAULTS["qp"]))
    preset = opts.get("preset", ENC_DEFAULTS["preset"])
    tune = opts.get("tune", ENC_DEFAULTS["tune"])
    pix_fmt = opts.get("pix_fmt", ENC_DEFAULTS["pix_fmt"])
    gop = str(opts.get("gop", ENC_DEFAULTS["gop"]))
    bf = str(opts.get("bf", ENC_DEFAULTS["bf"]))
    lookahead = str(opts.get("rc_lookahead", ENC_DEFAULTS["rc_lookahead"]))
    aq = str(opts.get("aq_strength", ENC_DEFAULTS["aq_strength"]))

    if codec.endswith("_nvenc"):
        profile = "main10" if pix_fmt in ("p010le", "yuv420p10le") and codec == "hevc_nvenc" \
            else ("high" if codec == "h264_nvenc" else "main")
        args = ['-c:v', codec, '-profile:v', profile, '-pix_fmt', pix_fmt,
                '-preset', preset, '-tune', tune, '-rc', 'constqp', '-qp', qp,
                '-b:v', '0', '-g', gop, '-bf', bf, '-rc-lookahead', lookahead,
                '-spatial-aq', '1', '-aq-strength', aq]
    else:  # libx264 CPU fallback
        args = ['-c:v', 'libx264', '-profile:v', 'high', '-preset', 'slow',
                '-crf', qp, '-pix_fmt', pix_fmt, '-g', gop]
    extra = opts.get("extra_enc")
    if extra:
        args += extra.split() if isinstance(extra, str) else list(extra)
    return args


def pick_encoder(opts):
    """Validate the requested codec; fall back to CPU x264 if it can't run.
    Returns (video_args, label)."""
    codec = opts.get("codec", ENC_DEFAULTS["codec"])
    if codec.endswith("_nvenc"):
        if not _encoder_listed(codec):
            warn(f"{codec} not in this ffmpeg build — using CPU libx264.")
            return build_video_args({**opts, "codec": "libx264"}), "CPU (libx264)"
        works, why = _encoder_works(build_video_args(opts))
        if works:
            return build_video_args(opts), f"GPU ({codec})"
        warn(f"{codec} present but test failed → {why}")
        if codec == "av1_nvenc":
            warn("AV1 encode needs an RTX 40-series+; your GPU likely can't. "
                 "Try --codec hevc_nvenc.")
        warn("Falling back to CPU libx264.")
        return build_video_args({**opts, "codec": "libx264"}), "CPU (libx264)"
    return build_video_args(opts), codec


# ── file discovery (ported) ───────────────────────────────────────────────────
def _videos_under(folder):
    """All videos under folder recursively; skips 'sample' files/folders."""
    found = []
    for dp, dn, files in os.walk(folder):
        dn[:] = [d for d in dn if "sample" not in d.lower()]
        for f in files:
            if (os.path.splitext(f)[1].lower() in _VIDEO_EXTS
                    and "sample" not in f.lower()
                    and not f.startswith(".")
                    and "_upscaled" not in f.lower()):
                found.append(os.path.join(dp, f))
    return sorted(found)


def expand_inputs(inputs):
    """Files/folders -> flat de-duplicated video list (folders recurse)."""
    files, seen, out = [], set(), []
    for item in inputs:
        item = os.path.abspath(os.path.expanduser(item))
        if os.path.isdir(item):
            files.extend(_videos_under(item))
        elif os.path.isfile(item):
            files.append(item)
        else:
            warn(f"Not found: {item}")
    for f in files:
        if f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
    return out


def unique_output(path):
    """Append ' (n)' before the extension until the path is free."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{base} ({n}){ext}"):
        n += 1
    return f"{base} ({n}){ext}"


def default_output_path(src, settings):
    d = settings.get("output_dir") or os.path.dirname(src)
    base = os.path.splitext(os.path.basename(src))[0]
    suffix = settings.get("suffix", "_upscaled")
    container = settings.get("container", "mkv")
    return os.path.join(d, f"{base}{suffix}.{container}")


# ── pipeline: commands ────────────────────────────────────────────────────────
def build_decode_cmd(src, hwaccel=False, deinterlace=False, tonemap=False,
                     hdr_transfer="smpte2084", scale_to=None):
    """Decode a whole file to 8-bit rgb24 rawvideo on stdout, with optional
    deinterlace (yadif) and HDR->SDR tonemap (zscale+tonemap) so the SDR-trained
    SR models get correct input. `tonemap` truthy enables it; hdr_transfer selects
    PQ (smpte2084) vs HLG (arib-std-b67). Input colour is declared explicitly so
    it works even if the stream's VUI tags are missing."""
    pre = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    if hwaccel:
        pre += ['-hwaccel', 'cuda']
    cmd = pre + ['-i', src, '-an', '-sn', '-map', '0:v:0']
    vf = source_vf(deinterlace, tonemap, hdr_transfer, scale_to)
    if vf:
        cmd += ['-vf', vf]
    return cmd + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']


def source_vf(deinterlace=False, tonemap=False, hdr_transfer="smpte2084", scale_to=None):
    """The decode-side filter chain (deinterlace + anamorphic un-squeeze + HDR->SDR
    tonemap), or None. Shared by the streaming decode and the preview so they match."""
    filters = []
    if deinterlace:
        filters.append('yadif=deint=interlaced')
    if scale_to:                            # un-squeeze anamorphic to square pixels
        filters.append(f'scale={scale_to[0]}:{scale_to[1]}:flags=lanczos,setsar=1')
    if tonemap:
        tin = "arib-std-b67" if hdr_transfer == "arib-std-b67" else "smpte2084"
        # canonical zscale tonemap: linearize -> float gbr -> bt709 primaries ->
        # tonemap -> bt709 SDR. The float step is required (else "no path between
        # colorspaces"). Input colour declared explicitly for robustness.
        filters.append(
            f'zscale=transferin={tin}:primariesin=bt2020:matrixin=bt2020nc:'
            'transfer=linear:npl=100,format=gbrpf32le,zscale=primaries=bt709,'
            'tonemap=tonemap=hable:desat=0,zscale=transfer=bt709:matrix=bt709:range=tv')
    return ",".join(filters) if filters else None


def build_encode_cmd(up_w, up_h, fps, target_w, target_h, out_path, video_args,
                     pad_color="black", pad=True):
    """Encode rgb24 rawvideo (from stdin) to a video-only segment, scaling to
    the exact target with lanczos + pad."""
    if pad:
        vf = (f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=decrease"
              f":flags=lanczos,pad={target_w}:{target_h}:-1:-1:color={pad_color},setsar=1")
    else:  # crop-to-fill
        vf = (f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=increase"
              f":flags=lanczos,crop={target_w}:{target_h},setsar=1")
    return ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{up_w}x{up_h}',
            '-r', f'{fps:.6f}', '-i', 'pipe:0',
            '-vf', vf] + video_args + ['-an', '-sn', out_path]


def build_split_cmd(src, seg_seconds, pattern):
    """Losslessly split the source at keyframes into <=seg_seconds chunks."""
    return ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', src,
            '-c', 'copy', '-map', '0:v:0', '-f', 'segment',
            '-segment_time', str(seg_seconds), '-reset_timestamps', '1',
            pattern]


def build_concat_cmd(list_file, out_path):
    return ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
            '-safe', '0', '-i', list_file, '-c', 'copy', out_path]


def build_mux_cmd(body, original, out_path, meta, container="mkv"):
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
           '-i', body, '-i', original,
           '-map', '0:v:0', '-map', '1:a?', '-map', '1:s?',
           '-c', 'copy', '-map_metadata', '1', '-metadata', f'videoai={meta}']
    if container == "mp4":
        cmd += ['-movflags', '+faststart']
    cmd += [out_path]
    return cmd


# ── Job / JobQueue (shared by CLI and web UI) ─────────────────────────────────
class Job:
    """A single upscale task + its live status. Both the CLI and the web server
    drive Jobs through run_job()."""
    _counter = 0
    _lock = threading.Lock()

    def __init__(self, src, settings, dst=None):
        with Job._lock:
            Job._counter += 1
            self.id = f"job{Job._counter}"
        self.src = src
        self.settings = dict(settings)
        self.dst = dst or default_output_path(src, settings)
        self.status = "queued"       # queued|running|paused|done|failed|cancelled
        self.error = None
        # progress
        self.total_frames = 0
        self.done_frames = 0
        self.deduped = 0             # frames reused via duplicate detection
        self.segment = 0
        self.total_segments = 0
        self.fps = 0.0
        self.started = None
        self.finished = None
        self.meta = {}
        # control
        self._cancel = threading.Event()
        self._pause = threading.Event()   # set == paused
        # live preview (latest original + upscaled frame pair, JPEG bytes)
        self._live_preview = None
        self._live_preview_seq = 0

    # --- control ---
    def cancel(self):
        self._cancel.set()
        self._pause.clear()

    def pause(self):
        if self.status == "running":
            self._pause.set()
            self.status = "paused"

    def resume(self):
        if self.status == "paused":
            self._pause.clear()
            self.status = "running"

    def _wait_if_paused(self):
        while self._pause.is_set() and not self._cancel.is_set():
            time.sleep(0.1)

    @property
    def progress(self):
        return (self.done_frames / self.total_frames) if self.total_frames else 0.0

    @property
    def eta(self):
        if self.fps > 0 and self.total_frames:
            return (self.total_frames - self.done_frames) / self.fps
        return None

    def to_dict(self):
        return {
            "id": self.id, "src": self.src, "dst": self.dst,
            "name": os.path.basename(self.src),
            "status": self.status, "error": self.error,
            "progress": round(self.progress, 4),
            "done_frames": self.done_frames, "total_frames": self.total_frames,
            "deduped": self.deduped,
            "segment": self.segment, "total_segments": self.total_segments,
            "fps": round(self.fps, 1), "eta": self.eta,
            "settings": self.settings, "meta": self.meta,
        }


def _scratch_dir(job, base=None):
    """Stable per-(source, key-settings) scratch path — NOT keyed by job.id, so
    resume works across restarts and new job ids."""
    base = base or tempfile.gettempdir()
    key = "|".join(str(x) for x in (
        os.path.abspath(job.src), job.settings.get("target"),
        job.settings.get("model"), job.settings.get("codec"),
        job.settings.get("interpolate"), job.settings.get("fps")))
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    name = os.path.splitext(os.path.basename(job.src))[0][:40]
    d = os.path.join(base, "upscale_scratch", f"{name}_{h}")
    os.makedirs(d, exist_ok=True)
    return d


def _resample(frames, fps_in, fps_out, interp):
    """Yield frames at fps_out from a source stream at fps_in, generating the
    in-between frames with interp(a, b, frac) where frac in (0,1). Handles any
    ratio via fractional timesteps (e.g. 24->60), not just integer multiples."""
    it = iter(frames)
    a = next(it, None)
    if a is None:
        return
    step = fps_in / fps_out
    i = 0                       # source index of `a`
    j = 0                       # output index
    b = next(it, None)
    while b is not None:
        while j * step < i + 1 - 1e-9:
            frac = j * step - i
            yield a if frac < 1e-3 else interp(a, b, frac)
            j += 1
        a = b
        i += 1
        b = next(it, None)
    while j * step <= i + 1e-9:  # tail: hold the final frame
        yield a
        j += 1


class _DedupEnhance:
    """Reuse the previous upscaled frame when the incoming source frame is
    byte-identical to the last one. Lossless (identical input -> identical
    output) and a big win on animation, which often holds a frame 2-3x."""

    def __init__(self, enhance):
        import numpy as np
        self._np = np
        self.enhance = enhance
        self.prev = None
        self.prev_out = None
        self.reused = 0

    def __call__(self, frame):
        if (self.prev is not None and frame.shape == self.prev.shape
                and self._np.array_equal(frame, self.prev)):
            self.reused += 1
            return self.prev_out
        self.prev = frame
        self.prev_out = self.enhance(frame)
        return self.prev_out


def _decode_flags(meta, settings):
    """(deinterlace, tonemap) from source flags + settings."""
    di = settings.get("deinterlace", "auto")
    deint = (di == "on") or (di == "auto" and meta.get("is_interlaced"))
    tonemap = bool(meta.get("is_hdr") and settings.get("tonemap", True))
    return deint, tonemap


def _push_live_preview(job, src_frame, up_frame):
    """Encode source + upscaled RGB frames as JPEG and stash on the job."""
    import cv2
    try:
        def encode_jpg(rgb, max_side=640):
            h, w = rgb.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return buf.tobytes()
        job._live_preview = {
            "src": encode_jpg(src_frame),
            "upscaled": encode_jpg(up_frame),
            "seq": job._live_preview_seq,
        }
        job._live_preview_seq += 1
    except Exception:
        pass  # preview is best-effort, never break the pipeline


def _process_segment(job, upscaler, src_seg, info_meta, up_w, up_h, target,
                     video_args, out_seg, on_progress, interpolator=None):
    """Stream one source-segment file: decode -> (interpolate) -> GPU upscale ->
    encode. Threaded so decode-read and encode-write overlap GPU inference."""
    import numpy as np
    # display (square-pixel) dims — the decode un-squeezes anamorphic sources to these
    src_w = info_meta.get("disp_w") or info_meta["width"]
    src_h = info_meta.get("disp_h") or info_meta["height"]
    frame_bytes = src_w * src_h * 3
    fps_in = info_meta["fps"] or 30.0
    interpolating = bool(interpolator)
    fps_out = (float(job.settings.get("fps") or 0) or fps_in * 2) if interpolating else fps_in
    order = job.settings.get("interp_order", "pre")
    tw, th = target
    pad = job.settings.get("pad", True)
    pad_color = job.settings.get("pad_color", "black")
    deint, tonemap = _decode_flags(info_meta, job.settings)
    scale_to = (src_w, src_h) if info_meta.get("is_anamorphic") else None

    dec = subprocess.Popen(
        build_decode_cmd(src_seg, deinterlace=deint, tonemap=tonemap,
                         hdr_transfer=info_meta.get("color_transfer") or "smpte2084",
                         scale_to=scale_to),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(
        build_encode_cmd(up_w, up_h, fps_out, tw, th, out_seg, video_args,
                         pad_color=pad_color, pad=pad),
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    in_q = queue.Queue(maxsize=job.settings.get("queue_size", 6))
    out_q = queue.Queue(maxsize=job.settings.get("queue_size", 6))
    err_box = {}

    def reader():
        try:
            while True:
                buf = dec.stdout.read(frame_bytes)
                if not buf or len(buf) < frame_bytes:
                    break
                # .copy() -> writable array (torch.from_numpy dislikes read-only)
                in_q.put(np.frombuffer(buf, np.uint8).reshape(src_h, src_w, 3).copy())
        except Exception as e:                       # noqa: BLE001
            err_box["reader"] = e
        finally:
            in_q.put(None)

    def writer():
        try:
            while True:
                frame = out_q.get()
                if frame is None:
                    break
                enc.stdin.write(frame.tobytes())
        except Exception as e:                       # noqa: BLE001
            err_box["writer"] = e
        finally:
            try:
                enc.stdin.close()
            except OSError:
                pass

    rt = threading.Thread(target=reader, daemon=True)
    wt = threading.Thread(target=writer, daemon=True)
    rt.start(); wt.start()

    _preview_interval = max(1, int(fps_in / 2))  # ~2 previews per second
    _last_src_frame = [None]

    def src_frames():
        while True:
            f = in_q.get()
            if f is None:
                return
            _last_src_frame[0] = f
            yield f

    dedup = None
    if getattr(upscaler, "is_temporal", False):
        stream = _vsr_stream(src_frames(), upscaler.enhance_clip, upscaler.window)
    else:
        enh = upscaler.enhance
        dedup = _DedupEnhance(enh) if job.settings.get("dedup", True) else None
        if dedup:
            enh = dedup
        if interpolating and abs(fps_out - fps_in) > 1e-6:
            interp = interpolator.interpolate
            if order == "post":
                stream = _resample((enh(f) for f in src_frames()), fps_in, fps_out, interp)
            else:
                stream = (enh(f) for f in _resample(src_frames(), fps_in, fps_out, interp))
        else:
            stream = (enh(f) for f in src_frames())

    import psutil as _psutil
    t0 = time.time()
    seg_frames = 0
    _mem_check_interval = max(1, int(fps_in * 5))
    for out_frame in stream:
        if job._cancel.is_set():
            break
        job._wait_if_paused()
        out_q.put(out_frame)
        seg_frames += 1
        job.done_frames += 1
        elapsed = time.time() - t0
        if elapsed > 0:
            job.fps = seg_frames / elapsed
        # Memory monitoring
        if seg_frames % _mem_check_interval == 0:
            mem = _psutil.virtual_memory()
            if mem.percent > 95:
                log.warning("CRITICAL MEMORY: %s%% used! Pausing 10s...", mem.percent)
                time.sleep(10)
        # Live preview
        if seg_frames % _preview_interval == 0 and _last_src_frame[0] is not None:
            _push_live_preview(job, _last_src_frame[0], out_frame)
        elapsed = time.time() - t0
        if elapsed > 0:
            job.fps = seg_frames / elapsed
        if on_progress and (seg_frames % 5 == 0):
            on_progress(job)
    if dedup:
        job.deduped += dedup.reused
    out_q.put(None)
    rt.join(timeout=5); wt.join(timeout=30)
    dec.terminate()
    enc.wait()
    if err_box:
        raise RuntimeError(f"segment pipe error: {err_box}")
    if enc.returncode not in (0, None) and not job._cancel.is_set():
        tail = (enc.stderr.read().decode('utf-8', 'replace').strip().splitlines()[-2:]
                if enc.stderr else [])
        raise RuntimeError("encoder failed: " + " | ".join(tail))
    return seg_frames


def _external_args(cmd_template, subs):
    """Tokenise a command TEMPLATE then substitute placeholders per token, so a
    substituted path with spaces stays a single argument."""
    import shlex
    out = []
    for tok in shlex.split(cmd_template, posix=True):
        for k, v in subs.items():
            tok = tok.replace("{%s}" % k, v)
        out.append(tok)
    return out


def _run_external_segment(job, src_seg, out_seg, target, cmd_template):
    """Run a user-configured external video-to-video upscaler on one segment.

    This is the integration point for heavy models that can't live in our venv
    (e.g. SeedVR2 diffusion) — they run in THEIR own environment and we just
    orchestrate (segment, resume, concat, mux) around them. The template may use
    {input} {output} {width} {height} {target} placeholders."""
    tw, th = target
    args = _external_args(cmd_template, {
        "input": src_seg, "output": out_seg, "width": str(tw),
        "height": str(th), "target": f"{tw}x{th}"})
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out_seg):
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"external command failed (rc={r.returncode}): "
                           + " | ".join(tail))


def run_job(job, upscaler, on_progress=None, interpolator=None):
    """Full pipeline for one Job. `upscaler` is a ready Upscaler/VSRUpscaler (or
    None when settings['external_cmd'] drives an external per-segment upscaler);
    `interpolator` is an optional ready Interpolator (RIFE) for fps boost."""
    import psutil
    job.status = "running"
    job.started = time.time()
    log.info("Starting job %s: %s", job.id, job.src)
    log.info("Target: %s, Model: %s", job.settings.get('target', '4k'), job.settings.get('model', 'default'))
    log.info("Memory at start: %s%% used, %.1fMB RSS", psutil.virtual_memory().percent, psutil.Process().memory_info().rss / 1024**2)
    try:
        meta = probe(job.src)
        job.meta = {k: meta[k] for k in ("width", "height", "fps", "duration",
                                         "nb_frames", "has_audio", "has_subs",
                                         "bit_depth", "is_hdr", "is_interlaced", "is_vfr",
                                         "is_anamorphic", "disp_w", "disp_h")}
        job.total_frames = meta["nb_frames"] or 1
        if not job.settings.get("external_cmd"):
            _di, _tm = _decode_flags(meta, job.settings)
            if meta.get("is_hdr"):
                if _tm:
                    info(f"  HDR source ({meta.get('color_transfer') or 'bt2020'}) → "
                         f"tonemapping to SDR (AI models are SDR-domain)")
                else:
                    warn("  HDR source but --no-tonemap set — colors will look washed out.")
            if _di:
                info("  interlaced source → deinterlacing (yadif)")
            if meta.get("is_vfr"):
                warn("  variable frame rate → output is constant-fps at the average rate "
                     f"({meta['fps']:.3f}); audio stays aligned at the ends.")
            if meta.get("is_anamorphic"):
                info(f"  anamorphic source ({meta['width']}x{meta['height']}) → "
                     f"un-squeezing to {meta['disp_w']}x{meta['disp_h']} (correct aspect)")
        if interpolator:
            fps_in = meta["fps"] or 30.0
            fps_out = float(job.settings.get("fps") or 0) or fps_in * 2
            job.total_frames = max(1, round(job.total_frames * fps_out / fps_in))
            job.meta["fps_out"] = round(fps_out, 3)
        target = parse_target(job.settings.get("target", "4k"))
        external = job.settings.get("external_cmd")
        if external:
            scale = up_w = up_h = video_args = None
        else:
            scale = upscaler.scale
            sw = meta.get("disp_w") or meta["width"]
            sh = meta.get("disp_h") or meta["height"]
            up_w, up_h = model_output_dims(sw, sh, scale)
            video_args, _label = pick_encoder(job.settings)

        scratch = _scratch_dir(job, job.settings.get("scratch"))
        seg_seconds = job.settings.get("segment_seconds", DEFAULT_SEGMENT_SECONDS)
        do_segment = (job.settings.get("segment", True)
                      and meta["duration"] > seg_seconds * 1.5)

        # 1) split (lossless) or single source
        ext = os.path.splitext(job.src)[1] or ".mkv"
        if do_segment:
            pattern = os.path.join(scratch, f"src_%04d{ext}")
            if not glob.glob(os.path.join(scratch, "src_*")):
                _run(build_split_cmd(job.src, seg_seconds, pattern))
            src_segs = sorted(glob.glob(os.path.join(scratch, f"src_*{ext}")))
        else:
            src_segs = [job.src]
        job.total_segments = len(src_segs)

        # 2) upscale each segment (resume: skip finished)
        up_segs = []
        for i, ss in enumerate(src_segs):
            job.segment = i + 1
            out_seg = os.path.join(scratch, "up_%04d.mkv" % i)
            up_segs.append(out_seg)
            if job.settings.get("resume", True) and os.path.isfile(out_seg) \
                    and _seg_ok(out_seg):
                info(f"  resume: segment {i+1}/{len(src_segs)} already done")
                continue
            if job._cancel.is_set():
                break
            if external:
                _run_external_segment(job, ss, out_seg, target, external)
                job.done_frames = min(job.total_frames,
                                      round(job.total_frames * (i + 1) / len(src_segs)))
                if on_progress:
                    on_progress(job)
            else:
                _process_segment(job, upscaler, ss, meta, up_w, up_h, target,
                                 video_args, out_seg, on_progress, interpolator)
            if job._cancel.is_set():
                break

        if job._cancel.is_set():
            job.status = "cancelled"
            return job

        # 3) concat + 4) mux audio/subs from the original
        body = os.path.join(scratch, "body.mkv")
        if len(up_segs) == 1:
            body = up_segs[0]
        else:
            listf = os.path.join(scratch, "concat.txt")
            with open(listf, "w", encoding="utf-8") as f:
                for s in up_segs:
                    f.write(f"file '{s.replace(chr(39), chr(92) + chr(39))}'\n")
            _run(build_concat_cmd(listf, body))

        dst = unique_output(job.dst) if not job.settings.get("overwrite") else job.dst
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        if external:
            metatag = f"Upscaled with external command to {target[0]}x{target[1]}"
        else:
            metatag = (f"Upscaled with {upscaler.name} to {target[0]}x{target[1]} "
                       f"(x{scale} model + lanczos)")
            if interpolator:
                metatag += (f"; frame-interpolated to {job.meta.get('fps_out')}fps "
                            f"with {interpolator.name}")
        _run(build_mux_cmd(body, job.src, dst,
                           metatag, job.settings.get("container", "mkv")))
        job.dst = dst

        if not job.settings.get("keep_segments"):
            shutil.rmtree(scratch, ignore_errors=True)

        job.status = "done"
        job.finished = time.time()
        log.info("Job %s completed in %.1fs", job.id, job.finished - job.started)
        if on_progress:
            on_progress(job)
        return job
    except Exception as e:                           # noqa: BLE001
        import traceback
        job.status = "failed"
        job.error = str(e)
        job.finished = time.time()
        log.error("Job %s failed after %.1fs: %s", job.id, job.finished - job.started, e)
        log.error("Memory: %s%% used, %.1fMB RSS", psutil.virtual_memory().percent, psutil.Process().memory_info().rss / 1024**2)
        log.error(traceback.format_exc())
        if on_progress:
            on_progress(job)
        return job


def _seg_ok(path):
    """Cheap validity check for a finished segment (non-empty + probes)."""
    try:
        return os.path.getsize(path) > 1024 and probe(path)["nb_frames"] >= 0
    except Exception:                                # noqa: BLE001
        return False


class JobQueue:
    """Serial queue with pause/cancel/persist — used by the web server."""

    def __init__(self, persist_path=None):
        self.jobs = []
        self.persist_path = persist_path
        self._lock = threading.Lock()
        self._worker = None
        self._stop = threading.Event()
        self._upscalers = {}       # cache-key -> Upscaler
        self._interps = {}         # cache-key -> Interpolator
        self.on_change = None      # callback(job_dict or None) for broadcasts
        self.current = None
        self.running = False       # queue does NOT auto-run; user presses Start
        self._load()

    def _load(self):
        """Restore jobs from a previous run (queue survives restarts)."""
        if not self.persist_path or not os.path.isfile(self.persist_path):
            return
        try:
            with open(self.persist_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        for jd in data:
            try:
                if not os.path.isfile(jd.get("src", "")):
                    continue                      # source gone — drop it
                job = Job(jd["src"], jd.get("settings", {}), dst=jd.get("dst"))
                st = jd.get("status")
                # anything unfinished becomes queued again (resume skips done segments)
                job.status = "queued" if st in ("running", "paused", "queued") else st
                job.meta = jd.get("meta", {})
                self.jobs.append(job)
            except (KeyError, TypeError):
                continue
        # restored jobs wait — do NOT auto-run on startup; user presses Start.

    def add(self, src, settings):
        job = Job(src, settings)
        with self._lock:
            self.jobs.append(job)
        self._changed(job)
        if self.running:               # only auto-pick if the queue is started
            self._ensure_worker()
        return job

    def start(self):
        """Begin processing queued jobs."""
        self.running = True
        if self.current and self.current.status == "paused":
            self.current.resume()
        self._ensure_worker()
        self._changed(None)

    def pause_queue(self):
        """Suspend: stop picking up new jobs and pause the current one (resumable)."""
        self.running = False
        if self.current:
            self.current.pause()
        self._changed(None)

    def stop(self):
        """Halt: cancel the current job and stop the queue (not resumable — the
        job's finished segments are kept, so re-running it resumes from there)."""
        self.running = False
        if self.current:
            self.current.cancel()
        self._changed(None)

    def get(self, job_id):
        return next((j for j in self.jobs if j.id == job_id), None)

    def remove(self, job_id):
        job = self.get(job_id)
        if not job:
            return False
        if job.status in ("running", "paused"):
            job.cancel()
        with self._lock:
            self.jobs = [j for j in self.jobs if j.id != job_id]
        self._save()
        return True

    def clear_finished(self):
        with self._lock:
            self.jobs = [j for j in self.jobs
                         if j.status not in ("done", "failed", "cancelled")]
        self._save()

    def _ensure_worker(self):
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def _run_loop(self):
        while not self._stop.is_set():
            if not self.running:
                return
            job = next((j for j in self.jobs if j.status == "queued"), None)
            if job is None:
                return
            self.current = job
            try:
                up = self._get_upscaler(job.settings)
                interp = self._get_interpolator(job.settings, up) \
                    if job.settings.get("interpolate") else None
                run_job(job, up, on_progress=self._changed, interpolator=interp)
            except Exception as e:                       # noqa: BLE001
                job.status = "failed"
                job.error = str(e)
            self.current = None
            self._changed(job)

    def _get_upscaler(self, settings):
        path = resolve_model(settings.get("model", DEFAULT_MODEL),
                             settings.get("weights_dir", DEFAULT_WEIGHTS_DIR),
                             assume_yes=True)
        key = (path, settings.get("gpu", 0), settings.get("fp16", True),
               settings.get("tile", 0))
        if key not in self._upscalers:
            self._upscalers[key] = Upscaler(
                path, gpu=settings.get("gpu", 0), fp16=settings.get("fp16", True),
                tile=settings.get("tile", 0), tile_pad=settings.get("tile_pad", 16))
        return self._upscalers[key]

    def _get_interpolator(self, settings, upscaler):
        name = settings.get("rife_model", DEFAULT_RIFE_MODEL)
        key = (name, settings.get("fp16", True))
        if key not in self._interps:
            self._interps[key] = Interpolator(
                device=upscaler.device, fp16=settings.get("fp16", True),
                model_name=name, weights_dir=settings.get("weights_dir", DEFAULT_WEIGHTS_DIR))
        return self._interps[key]

    def _changed(self, job):
        self._save()
        if self.on_change:
            try:
                self.on_change(job.to_dict() if isinstance(job, Job) else job)
            except Exception:                        # noqa: BLE001
                pass

    def _save(self):
        if not self.persist_path:
            return
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump([j.to_dict() for j in self.jobs], f, indent=2)
        except OSError:
            pass

    def snapshot(self):
        return [j.to_dict() for j in self.jobs]


# ── speed benchmark / upfront ETA ─────────────────────────────────────────────
def sample_frame(path, at=None):
    """Grab one decoded RGB uint8 frame from a file (for benchmarking/preview)."""
    import numpy as np
    meta = probe(path)
    w, h = meta["width"], meta["height"]
    ts = at if at is not None else min(10.0, (meta["duration"] or 20.0) / 2)
    raw = subprocess.run(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', str(ts),
         '-i', path, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
        capture_output=True).stdout
    if len(raw) < w * h * 3:
        return None
    return np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3).copy()


def benchmark_fps(upscaler, frame, n=6, warmup=2):
    """Measured upscale throughput (frames/s) for this model at this resolution."""
    torch = upscaler.torch
    for _ in range(warmup):
        upscaler.enhance(frame)
    if upscaler.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        upscaler.enhance(frame)
    if upscaler.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return n / dt if dt > 0 else 0.0


def estimate_and_report(upscaler, files, settings):
    """Benchmark on the first file and print a per-file + total ETA up front, so a
    multi-hour (or multi-day!) job is never a surprise. Returns measured fps."""
    frame = sample_frame(files[0])
    if frame is None:
        return None
    fps = benchmark_fps(upscaler, frame)
    if fps <= 0:
        return None
    h, w = frame.shape[:2]
    interp_mult = 1.0
    if settings.get("interpolate"):
        fin = probe(files[0]).get("fps") or 30.0
        fout = float(settings.get("fps") or 0) or fin * 2
        interp_mult = fout / fin
    total_frames = 0
    for f in files:
        try:
            total_frames += probe(f)["nb_frames"]
        except Exception:                            # noqa: BLE001
            pass
    total_out = total_frames * interp_mult
    est = total_out / fps if fps else 0
    div()
    info(f"Benchmark: {Fore.WHITE}{Style.BRIGHT}{fps:.1f} fps{Style.RESET_ALL} "
         f"at {w}x{h} with {upscaler.name} (x{upscaler.scale})")
    info(f"Estimated: {Fore.WHITE}{Style.BRIGHT}~{_fmt_long(est)}{Style.RESET_ALL} "
         f"for {len(files)} file(s) (~{int(total_out):,} output frames)")
    if est > 6 * 3600:
        warn(f"That's a long run. Faster options: a compact model "
             f"(realesr-animevideov3 is ~25x faster than x4plus), a lower "
             f"--target, or --interpolate off.")
    return fps


def run_benchmark(args):
    """Benchmark each model on a sample frame and print a comparison table, so you
    can pick by measured speed on THIS GPU. Uses a frame from the given file (at its
    native resolution) if provided, else a synthetic frame at --bench-res."""
    import numpy as np
    import torch

    target = parse_target(args.target)
    frame, label_src, total_frames = None, None, None
    if args.inputs:
        files = expand_inputs(args.inputs)
        if files:
            frame = sample_frame(files[0])
            label_src = os.path.basename(files[0])
            try:
                total_frames = probe(files[0])["nb_frames"]
            except Exception:                        # noqa: BLE001
                pass
    if frame is None:
        try:
            w, h = parse_target(args.bench_res)
        except ValueError:
            w, h = 1920, 1080
        frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        label_src = f"synthetic {w}x{h}"
    h, w = frame.shape[:2]
    if not total_frames:
        total_frames = int(2 * 3600 * 24)           # assume a 2h @24fps movie
        movie_note = "~2h movie"
    else:
        movie_note = f"{label_src}"

    names = (args.bench_models.split(",") if args.bench_models
             else list(BUILTIN_MODELS))
    div()
    info(f"Benchmarking {len(names)} model(s) on a {Fore.WHITE}{Style.BRIGHT}{w}x{h}"
         f"{Style.RESET_ALL} frame ({label_src}) → {target[0]}x{target[1]}")
    warn("Downloads any missing models; heavy models take a few seconds each.")
    print()
    rows = []
    for name in names:
        try:
            path = resolve_model(name, args.weights_dir, assume_yes=args.yes)
            up = Upscaler(path, gpu=args.gpu, fp16=not args.fp32, tile=args.tile)
            for _ in range(4):                       # warmup + let GPU boost clocks ramp
                up.enhance(frame)
            if up.device.type == "cuda":
                torch.cuda.synchronize()
            t = time.time(); up.enhance(frame); t1 = time.time() - t
            n = max(5, min(30, int(2.5 / max(t1, 1e-3))))
            fps = benchmark_fps(up, frame, n=n, warmup=0)
            rows.append([name, up.scale, fps, None])
            del up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:                       # noqa: BLE001
            rows.append([name, None, None, str(e)[:48]])

    rows.sort(key=lambda r: (r[2] is None, -(r[2] or 0)))
    print(f"  {'model':24s} {'scale':>5s} {'fps':>7s} {'ms/fr':>7s}  {'est. ' + movie_note:>16s}")
    print(f"  {'-'*24} {'-'*5} {'-'*7} {'-'*7}  {'-'*16}")
    for name, scale, fps, errtxt in rows:
        star = f" {Fore.CYAN}(default){Style.RESET_ALL}" if name == DEFAULT_MODEL else ""
        if fps is None:
            print(f"  {name:24s}  {Fore.RED}failed: {errtxt}{Style.RESET_ALL}")
            continue
        est = _fmt_long(total_frames / fps) if fps else "--"
        col = Fore.GREEN if fps >= 5 else (Fore.YELLOW if fps >= 1 else Fore.RED)
        print(f"  {name:24s} {('x'+str(scale)):>5s} {col}{fps:7.1f}{Style.RESET_ALL} "
              f"{1000/fps:7.0f}  {est:>16s}{star}")
    div()
    return 0


def benchmark_models(names=None, res="1920x1080", gpu=0, fp16=True, tile=0,
                     weights_dir=DEFAULT_WEIGHTS_DIR):
    """Benchmark models on a synthetic frame; return a sorted list of dicts
    {name, scale, fps, ms, note} (or {name, error}). Used by CLI + web UI."""
    import numpy as np
    import torch
    names = names or list(BUILTIN_MODELS)
    try:
        w, h = parse_target(res)
    except ValueError:
        w, h = 1920, 1080
    frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    out = []
    for name in names:
        try:
            path = resolve_model(name, weights_dir, assume_yes=True)
            up = Upscaler(path, gpu=gpu, fp16=fp16, tile=tile)
            for _ in range(3):
                up.enhance(frame)
            if up.device.type == "cuda":
                torch.cuda.synchronize()
            t = time.time(); up.enhance(frame); t1 = time.time() - t
            n = max(3, min(20, int(2.0 / max(t1, 1e-3))))
            fps = benchmark_fps(up, frame, n=n, warmup=0)
            out.append({"name": name, "scale": up.scale, "fps": round(fps, 1),
                        "ms": round(1000 / fps) if fps else None,
                        "note": BUILTIN_MODELS.get(name, ("", up.scale, ""))[2]})
            del up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:                       # noqa: BLE001
            out.append({"name": name, "error": str(e)[:100]})
    out.sort(key=lambda r: (r.get("fps") is None, -(r.get("fps") or 0)))
    return out


def _fmt_long(secs):
    secs = int(max(0, secs))
    d, r = divmod(secs, 86400)
    h, r = divmod(r, 3600)
    m, _s = divmod(r, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    return f"{m}m {_s}s"


# ── dry-run planner ───────────────────────────────────────────────────────────
def plan_and_print(src, settings):
    """Print the full plan + every ffmpeg command without running anything."""
    meta = probe(src)
    # scale is model-native; without loading torch we look it up for builtins.
    model_spec = settings.get("model", DEFAULT_MODEL)
    scale = BUILTIN_MODELS.get(model_spec, (None, 4, None))[1]
    up_w, up_h = model_output_dims(meta.get("disp_w") or meta["width"],
                                   meta.get("disp_h") or meta["height"], scale)
    target = parse_target(settings.get("target", "4k"))
    video_args, label = pick_encoder(settings)
    seg_seconds = settings.get("segment_seconds", DEFAULT_SEGMENT_SECONDS)
    do_segment = (settings.get("segment", True)
                  and meta["duration"] > seg_seconds * 1.5)
    n_segs = max(1, math.ceil(meta["duration"] / seg_seconds)) if do_segment else 1

    div()
    info(f"DRY RUN — {os.path.basename(src)}")
    print(f"  source     : {meta['width']}x{meta['height']} @ {meta['fps']:.3f}fps, "
          f"{_fmt(meta['duration'])}, {meta['nb_frames']} frames, {meta['vcodec']}")
    print(f"  format     : {describe_source(meta)}")
    _di, _tm = _decode_flags(meta, settings)
    if _di or _tm:
        print(f"  pre-filter : {'deinterlace ' if _di else ''}{'HDR→SDR tonemap' if _tm else ''}".strip())
    print(f"  model      : {model_spec}  (native x{scale})")
    print(f"  model out  : {up_w}x{up_h}  ->  target {target[0]}x{target[1]} (lanczos+pad)")
    print(f"  encoder    : {label}")
    print(f"  segments   : {n_segs} x ~{seg_seconds}s  (resume={settings.get('resume', True)})")
    print(f"  output     : {default_output_path(src, settings)}")
    print()
    ex = "src_0000" + (os.path.splitext(src)[1] or ".mkv")
    if do_segment:
        print(f"{Fore.CYAN}  split :{Style.RESET_ALL} " +
              _quote(build_split_cmd(src, seg_seconds,
                                     os.path.join('<scratch>', 'src_%04d' + (os.path.splitext(src)[1] or '.mkv')))))
    print(f"{Fore.CYAN}  decode:{Style.RESET_ALL} " +
          _quote(build_decode_cmd(os.path.join('<scratch>', ex))))
    print(f"{Fore.CYAN}  encode:{Style.RESET_ALL} " +
          _quote(build_encode_cmd(up_w, up_h, meta['fps'] or 30.0, target[0],
                                  target[1], os.path.join('<scratch>', 'up_0000.mkv'),
                                  video_args, settings.get('pad_color', 'black'),
                                  settings.get('pad', True))))
    print(f"{Fore.CYAN}  mux   :{Style.RESET_ALL} " +
          _quote(build_mux_cmd(os.path.join('<scratch>', 'body.mkv'), src,
                               default_output_path(src, settings), 'Upscaled ...',
                               settings.get('container', 'mkv'))))
    div()


# ── CLI ────────────────────────────────────────────────────────────────────────
def _settings_from_args(a):
    return {
        "model": a.model, "target": a.target, "codec": a.codec, "qp": a.qp,
        "preset": a.preset, "tune": a.tune, "pix_fmt": a.pix_fmt,
        "gop": a.gop, "bf": a.bf, "rc_lookahead": a.rc_lookahead,
        "aq_strength": a.aq_strength, "extra_enc": a.extra_enc,
        "container": a.container, "suffix": a.suffix, "output_dir": a.output,
        "overwrite": a.overwrite, "pad": not a.no_pad, "pad_color": a.pad_color,
        "deinterlace": a.deinterlace, "tonemap": not a.no_tonemap,
        "tile": a.tile, "tile_pad": a.tile_pad, "fp16": not a.fp32, "gpu": a.gpu,
        "denoise": a.denoise, "weights_dir": a.weights_dir,
        "dedup": not a.no_dedup, "external_cmd": a.external_cmd,
        "segment": not a.no_segment, "segment_seconds": a.segment_seconds,
        "resume": not a.no_resume, "keep_segments": a.keep_segments,
        "scratch": a.scratch, "queue_size": a.queue_size,
        "interpolate": a.interpolate, "fps": a.fps, "interp_order": a.interp_order,
        "rife_model": a.rife_model, "assume_yes": a.yes,
    }


def _print_models():
    div()
    info("Builtin models (also: any local .pth/.safetensors or HuggingFace id)")
    for name, (_url, scale, note) in BUILTIN_MODELS.items():
        star = " (default)" if name == DEFAULT_MODEL else ""
        print(f"  {Fore.WHITE}{Style.BRIGHT}{name}{Style.RESET_ALL}  x{scale}{star}")
        print(f"      {note}")
    div()


def build_parser():
    p = argparse.ArgumentParser(
        prog="upscale_video.py",
        description="AI video upscaler (open models, GPU-streaming pipeline). "
                    "Give files/folders to run headless; use --serve for the web UI.")
    p.add_argument("inputs", nargs="*", help="video files and/or folders (recurse)")
    p.add_argument("-o", "--output", metavar="DIR", help="output folder")

    g = p.add_argument_group("target")
    g.add_argument("--target", default="4k", help="4k | 1080p | WxH (default 4k)")
    g.add_argument("--no-pad", action="store_true", help="crop-to-fill instead of pad")
    g.add_argument("--pad-color", default="black")

    g = p.add_argument_group("source handling")
    g.add_argument("--deinterlace", choices=["auto", "on", "off"], default="auto",
                   help="deinterlace interlaced sources (auto = only if detected)")
    g.add_argument("--no-tonemap", action="store_true",
                   help="do NOT tonemap HDR->SDR (HDR sources will look washed out)")

    g = p.add_argument_group("model")
    g.add_argument("--quality", choices=list(QUALITY_PRESETS),
                   help="one-click bundle of model+qp+codec (fast|balanced|best|max)")
    g.add_argument("--model", default=DEFAULT_MODEL,
                   help="builtin name | local file | HF id (see --list-models)")
    g.add_argument("--denoise", type=float, default=None, help="0..1 (models that support it)")
    g.add_argument("--tile", type=int, default=0,
                   help="tile size for VRAM; 0 = auto from free VRAM (default)")
    g.add_argument("--tile-pad", type=int, default=16)
    g.add_argument("--fp32", action="store_true", help="disable fp16 (slower, more VRAM)")
    g.add_argument("--gpu", type=int, default=0)
    g.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    g.add_argument("--list-models", action="store_true")

    g = p.add_argument_group("interpolation")
    g.add_argument("--interpolate", action="store_true", help="RIFE frame interpolation (fps boost)")
    g.add_argument("--fps", type=float, default=None, help="target fps (default 2x source)")
    g.add_argument("--rife-model", default=DEFAULT_RIFE_MODEL, choices=RIFE_MODELS)
    g.add_argument("--interp-order", choices=["pre", "post"], default="pre",
                   help="pre = interpolate then upscale (fast); post = upscale then interpolate")

    g = p.add_argument_group("temporal VSR (multi-frame, less flicker)")
    g.add_argument("--vsr", action="store_true",
                   help="use temporal video super-resolution instead of single-image")
    g.add_argument("--vsr-model", default=DEFAULT_VSR_MODEL, choices=VSR_MODELS,
                   help="AnimeSR_v2_4x (anime) | BasicVSR/IconVSR/EDVR (general)")
    g.add_argument("--vsr-window", type=int, default=16,
                   help="frames per VSR window (more = better temporal, more VRAM)")

    g = p.add_argument_group("external / diffusion (SeedVR2 etc.)")
    g.add_argument("--external-cmd", default=None, metavar="TEMPLATE",
                   help="orchestrate an external video-to-video upscaler per segment "
                        "(its own env/GPU). Placeholders: {input} {output} {width} {height} {target}. "
                        "This is how you plug in a heavy diffusion model like SeedVR2.")

    g = p.add_argument_group("encode")
    g.add_argument("--codec", default=ENC_DEFAULTS["codec"],
                   choices=["h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264"])
    g.add_argument("--qp", default=ENC_DEFAULTS["qp"])
    g.add_argument("--preset", default=ENC_DEFAULTS["preset"])
    g.add_argument("--tune", default=ENC_DEFAULTS["tune"])
    g.add_argument("--pix-fmt", dest="pix_fmt", default=ENC_DEFAULTS["pix_fmt"])
    g.add_argument("--gop", default=ENC_DEFAULTS["gop"])
    g.add_argument("--bf", default=ENC_DEFAULTS["bf"])
    g.add_argument("--rc-lookahead", dest="rc_lookahead", default=ENC_DEFAULTS["rc_lookahead"])
    g.add_argument("--aq-strength", dest="aq_strength", default=ENC_DEFAULTS["aq_strength"])
    g.add_argument("--extra-enc", default=None, help="extra raw ffmpeg encode args")
    g.add_argument("--container", default="mkv", choices=["mkv", "mp4"])
    g.add_argument("--suffix", default="_upscaled")
    g.add_argument("--overwrite", action="store_true")

    g = p.add_argument_group("pipeline")
    g.add_argument("--no-dedup", action="store_true",
                   help="disable duplicate-frame reuse (lossless speedup on animation)")
    g.add_argument("--no-segment", action="store_true", help="single streaming pass")
    g.add_argument("--segment-seconds", type=int, default=DEFAULT_SEGMENT_SECONDS)
    g.add_argument("--no-resume", action="store_true")
    g.add_argument("--keep-segments", action="store_true")
    g.add_argument("--scratch", default=None, help="scratch base dir")
    g.add_argument("--queue-size", type=int, default=6)

    g = p.add_argument_group("misc")
    g.add_argument("--dry-run", action="store_true", help="print plan+commands, run nothing")
    g.add_argument("--benchmark", action="store_true",
                   help="benchmark all models on a sample frame and print a table")
    g.add_argument("--bench-res", default="1920x1080",
                   help="frame size for --benchmark when no input file is given")
    g.add_argument("--bench-models", default=None,
                   help="comma-separated model list for --benchmark (default: all builtins)")
    g.add_argument("--no-eta", action="store_true", help="skip the upfront benchmark/ETA")
    g.add_argument("-y", "--yes", action="store_true", help="auto-confirm downloads")
    g.add_argument("--verbose", action="store_true")
    g.add_argument("--setup", action="store_true",
                   help="create .venv and install CUDA torch + all deps")
    g.add_argument("--cuda", default="cu128",
                   help="pytorch CUDA index for --setup (cu128|cu126|cu124)")

    g = p.add_argument_group("web ui")
    g.add_argument("--serve", action="store_true", help="launch the web UI")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=DEFAULT_PORT)
    g.add_argument("--open", action="store_true", help="open the browser")
    return p


def _cli_progress(job):
    if job.status in ("done", "failed", "cancelled"):
        return
    eta = job.eta
    line = ("  %s  %s %3d%%  seg %d/%d  %.0ffps  ETA %s" % (
        os.path.basename(job.src)[:30], _bar(job.progress), int(job.progress * 100),
        job.segment, job.total_segments, job.fps,
        _fmt(eta) if eta is not None else "--"))
    sys.stdout.write("\r" + line[:110].ljust(110))
    sys.stdout.flush()


def run_cli(args):
    settings = _settings_from_args(args)
    files = expand_inputs(args.inputs)
    if not files:
        warn("No videos found in the given paths.")
        return 1

    if args.dry_run:
        for f in files:
            plan_and_print(f, settings)
        return 0

    # external orchestration path (diffusion/SeedVR2 etc.) — no in-process model
    if args.external_cmd:
        info(f"External upscaler (per-segment): {args.external_cmd[:70]}"
             f"{'...' if len(args.external_cmd) > 70 else ''}")
        n_ok = n_fail = 0
        for i, src in enumerate(files, 1):
            job = Job(src, settings)
            info(f"[{i}/{len(files)}] {os.path.basename(src)} → {os.path.basename(job.dst)}")
            run_job(job, None, on_progress=_cli_progress)
            sys.stdout.write("\r" + " " * 110 + "\r")
            if job.status == "done":
                n_ok += 1
                ok(f"[{i}/{len(files)}] {os.path.basename(job.dst)}")
            else:
                n_fail += 1
                err(f"[{i}/{len(files)}] {job.status}: {job.error}")
        div(); ok(f"{n_ok} done, {n_fail} failed")
        return 0 if n_fail == 0 else 1

    info("Detecting encoder ...")
    _va, label = pick_encoder(settings)
    info(f"Encoder → {label}")
    # resolve + load the model once, reuse across files
    if args.vsr:
        info(f"Loading temporal VSR model {args.vsr_model} ...")
        upscaler = VSRUpscaler(args.vsr_model, gpu=args.gpu, fp16=not args.fp32,
                               tile=(args.tile if args.tile > 0 else 128),
                               window=args.vsr_window, weights_dir=args.weights_dir)
        info(f"VSR ready: {upscaler.name} x{upscaler.scale}, window={upscaler.window}, "
             f"fp16={upscaler.fp16}, device={upscaler.device}")
    else:
        model_path = resolve_model(args.model, args.weights_dir, assume_yes=args.yes)
        info(f"Loading model {os.path.basename(model_path)} ...")
        upscaler = Upscaler(model_path, gpu=args.gpu, fp16=not args.fp32,
                            tile=args.tile, tile_pad=args.tile_pad)
        info(f"Model ready: x{upscaler.scale}, fp16={upscaler.fp16}, "
             f"tile={upscaler.tile}, device={upscaler.device}")

    interpolator = None
    if args.interpolate and args.vsr:
        warn("--interpolate is ignored in --vsr mode (not combined yet).")
    elif args.interpolate:
        info(f"Loading RIFE interpolation model {args.rife_model} ...")
        interpolator = Interpolator(device=upscaler.device, fp16=not args.fp32,
                                    model_name=args.rife_model,
                                    weights_dir=args.weights_dir)
        info(f"Interpolation ready: {interpolator.name}, "
             f"target {args.fps or '2x'} fps, order={args.interp_order}")

    if not args.no_eta:
        try:
            estimate_and_report(upscaler, files, settings)
        except Exception as e:                       # noqa: BLE001
            warn(f"(couldn't estimate time: {e})")

    div()
    print(f"{Fore.CYAN}{Style.BRIGHT}  UPSCALING {len(files)} file(s) → "
          f"{args.target}{Style.RESET_ALL}")
    div()
    n_ok = n_fail = 0
    for i, src in enumerate(files, 1):
        job = Job(src, settings)
        info(f"[{i}/{len(files)}] {os.path.basename(src)} → {os.path.basename(job.dst)}")
        run_job(job, upscaler, on_progress=_cli_progress, interpolator=interpolator)
        sys.stdout.write("\r" + " " * 110 + "\r")
        if job.status == "done":
            n_ok += 1
            saved = (f"  ({job.deduped} dup frames reused, "
                     f"{100*job.deduped/max(1,job.done_frames):.0f}% skipped)"
                     if job.deduped else "")
            ok(f"[{i}/{len(files)}] {os.path.basename(job.dst)}{saved}")
        else:
            n_fail += 1
            err(f"[{i}/{len(files)}] {job.status}: {job.error}")
    div()
    ok(f"{n_ok} done, {n_fail} failed")
    return 0 if n_fail == 0 else 1


def do_setup(args):
    """Create .venv next to this script and install CUDA torch + all deps."""
    venv_dir = os.path.join(HERE, ".venv")
    py = os.path.join(venv_dir, "Scripts", "python.exe") if os.name == "nt" \
        else os.path.join(venv_dir, "bin", "python")
    if not os.path.isfile(py):
        info(f"Creating venv at {venv_dir} ...")
        rc = subprocess.call([sys.executable, "-m", "venv", venv_dir])
        if rc != 0:
            err("venv creation failed."); return 1
    index = f"https://download.pytorch.org/whl/{args.cuda}"
    steps = [
        ([py, "-m", "pip", "install", "--upgrade", "pip"], "upgrade pip"),
        ([py, "-m", "pip", "install", "torch", "torchvision", "--index-url", index],
         f"install CUDA torch ({args.cuda}) — this is a large (~2.5GB) download"),
        ([py, "-m", "pip", "install", "-r", os.path.join(HERE, "requirements.txt")],
         "install the rest of the deps"),
    ]
    for cmd, desc in steps:
        info(desc + " ...")
        if subprocess.call(cmd) != 0:
            err(f"step failed: {desc}"); return 1
    check = subprocess.run(
        [py, "-c", "import torch; print(torch.cuda.is_available(), "
         "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"],
        capture_output=True, text=True)
    div()
    ok(f"Setup complete. torch CUDA → {check.stdout.strip()}")
    info(f"Run with:  {py} upscale_video.py <video>   (or --serve)")
    if "True" not in check.stdout:
        warn("CUDA not available — try a different --cuda index (cu126/cu124) for your driver.")
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.setup:
        return do_setup(args)

    if args.quality:                                 # apply the preset bundle
        preset = QUALITY_PRESETS[args.quality]
        args.model, args.qp, args.codec = preset["model"], preset["qp"], preset["codec"]
        info(f"Quality preset '{args.quality}': {preset['label']}")

    if args.list_models:
        _print_models()
        return 0

    if args.benchmark:
        return run_benchmark(args)

    if args.serve:
        try:
            import server
        except ImportError as e:
            err(f"Web UI deps missing ({e}). pip install fastapi uvicorn")
            return 1
        return server.serve(args)

    if not args.inputs:
        parser.print_help()
        print()
        info("Tip: pass files/folders to upscale, or --serve for the web UI.")
        return 0

    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
