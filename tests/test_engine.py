r"""Pure-logic unit tests for the upscaler engine (no GPU / torch needed).

Run:  .\.venv\Scripts\python.exe -m pytest -q
These cover the parts most likely to break silently on refactor: target math,
the RIFE resample frame-count math, scratch keying (resume), encoder args,
output naming, and duplicate-frame dedup.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import upscale_video as U  # noqa: E402


# ── target parsing ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("spec,expected", [
    ("4k", (3840, 2160)), ("2160p", (3840, 2160)), ("1080p", (1920, 1080)),
    ("720p", (1280, 720)), ("1440p", (2560, 1440)),
    ("1920x1080", (1920, 1080)), ("3840x2160", (3840, 2160)),
    (None, (3840, 2160)),
])
def test_parse_target(spec, expected):
    assert U.parse_target(spec) == expected


def test_parse_target_bad():
    with pytest.raises(ValueError):
        U.parse_target("banana")


def test_model_output_dims():
    assert U.model_output_dims(1920, 1080, 2) == (3840, 2160)
    assert U.model_output_dims(640, 480, 4) == (2560, 1920)


# ── resample (interpolation) frame-count math ────────────────────────────────
def _count(n_in, fps_in, fps_out):
    # interp just returns a marker; we only count how many frames come out
    frames = [("src", i) for i in range(n_in)]
    out = list(U._resample(iter(frames), fps_in, fps_out, lambda a, b, t: ("interp", t)))
    return out


def test_resample_identity():
    out = _count(10, 30, 30)
    assert len(out) == 10
    assert all(f[0] == "src" for f in out)


def test_resample_2x():
    # N source frames at 2x -> 2N-1 (can't interpolate past the last frame)
    out = _count(10, 30, 60)
    assert len(out) == 19
    # every other output is an interpolated frame
    assert out[0][0] == "src" and out[1][0] == "interp"


def test_resample_noninteger():
    # 30 -> 45 fps over 10 source frames ~ 1.5x
    out = _count(10, 30, 45)
    assert 13 <= len(out) <= 15   # ~ (n-1)*1.5 + 1


def test_resample_empty():
    assert list(U._resample(iter([]), 30, 60, lambda *a: a)) == []


# ── temporal VSR windowing ───────────────────────────────────────────────────
def test_vsr_stream_frame_exact():
    frames = [("f", i) for i in range(10)]
    seen_windows = []

    def clip_identity(clip):
        seen_windows.append(len(clip))
        return list(clip)

    out = list(U._vsr_stream(iter(frames), clip_identity, window=4))
    assert out == frames                 # frame-exact, order preserved, none dropped
    assert seen_windows == [4, 4, 2]     # 10 frames -> windows of 4,4,2


# ── scratch dir keying (resume across restarts) ──────────────────────────────
def test_scratch_stable_and_distinct(tmp_path):
    j1 = U.Job("C:/x/movie.mkv", {"target": "4k", "model": "m"})
    j2 = U.Job("C:/x/movie.mkv", {"target": "4k", "model": "m"})   # same -> same dir
    j3 = U.Job("C:/x/movie.mkv", {"target": "1080p", "model": "m"})  # diff settings
    d1 = U._scratch_dir(j1, base=str(tmp_path))
    d2 = U._scratch_dir(j2, base=str(tmp_path))
    d3 = U._scratch_dir(j3, base=str(tmp_path))
    assert d1 == d2                 # stable regardless of job id
    assert d1 != d3                 # settings change -> different scratch


# ── encoder args ─────────────────────────────────────────────────────────────
def test_build_video_args_nvenc():
    args = U.build_video_args({"codec": "h264_nvenc", "qp": "18"})
    assert "h264_nvenc" in args and "-qp" in args and "18" in args
    assert "constqp" in args


def test_build_video_args_cpu():
    args = U.build_video_args({"codec": "libx264", "qp": "20"})
    assert "libx264" in args and "-crf" in args


def test_build_video_args_extra():
    args = U.build_video_args({"codec": "libx264", "extra_enc": "-x264-params foo=1"})
    assert "-x264-params" in args and "foo=1" in args


# ── output naming ────────────────────────────────────────────────────────────
def test_unique_output(tmp_path):
    p = tmp_path / "out.mkv"
    assert U.unique_output(str(p)) == str(p)         # free -> unchanged
    p.write_text("x")
    got = U.unique_output(str(p))
    assert got.endswith("(2).mkv") and got != str(p)


def test_default_output_path():
    got = U.default_output_path("D:/a/clip.mp4", {"suffix": "_up", "container": "mkv"})
    assert got.replace("\\", "/").endswith("a/clip_up.mkv")


# ── duplicate-frame dedup ────────────────────────────────────────────────────
def test_dedup_reuses_identical():
    calls = {"n": 0}

    def fake_enhance(frame):
        calls["n"] += 1
        return frame * 2   # deterministic "upscale"

    d = U._DedupEnhance(fake_enhance)
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.ones((4, 4, 3), np.uint8)
    d(a); d(a); d(a)      # identical -> 1 real call, 2 reused
    d(b)                  # different -> 1 real call
    d(a)                  # changed again -> 1 real call
    assert calls["n"] == 3
    assert d.reused == 2


# ── external command arg substitution ───────────────────────────────────────
def test_external_args_paths_with_spaces():
    subs = {"input": r"C:\my movies\a b.mkv", "output": r"D:\out\up 0.mkv",
            "width": "3840", "height": "2160", "target": "3840x2160"}
    args = U._external_args("tool --in {input} --out {output} --res {target}", subs)
    # {input}/{output} stay single args despite spaces
    assert args == ["tool", "--in", r"C:\my movies\a b.mkv",
                    "--out", r"D:\out\up 0.mkv", "--res", "3840x2160"]


# ── time formatting ──────────────────────────────────────────────────────────
def test_fmt():
    assert U._fmt(65) == "1:05"
    assert U._fmt(3661) == "1:01:01"


def test_fmt_long():
    assert U._fmt_long(90) == "1m 30s"
    assert U._fmt_long(3 * 3600 + 5 * 60) == "3h 5m"
    assert U._fmt_long(2 * 86400 + 3 * 3600) == "2d 3h 0m"


# ── input expansion ──────────────────────────────────────────────────────────
def test_expand_inputs(tmp_path):
    (tmp_path / "a.mp4").write_text("x")
    (tmp_path / "b.mkv").write_text("x")
    (tmp_path / "note.txt").write_text("x")
    (tmp_path / "sample.mp4").write_text("x")   # excluded by 'sample'
    got = U.expand_inputs([str(tmp_path)])
    names = sorted(os.path.basename(p) for p in got)
    assert names == ["a.mp4", "b.mkv"]
