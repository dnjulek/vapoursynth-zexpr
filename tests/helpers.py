"""Shared test helpers and independent NumPy oracles for zexpr."""

from __future__ import annotations

import platform
import sys

import numpy as np
import vapoursynth as vs


def plane_array(frame: vs.VideoFrame, plane: int) -> np.ndarray:
    return np.asarray(frame[plane]).copy()


def clip_planes(clip: vs.VideoNode, n: int = 0) -> list[np.ndarray]:
    frame = clip.get_frame(n)
    return [plane_array(frame, p) for p in range(clip.format.num_planes)]


def make_clip(
    core: vs.Core,
    width: int = 32,
    height: int = 16,
    fmt: vs.PresetVideoFormat = vs.GRAY8,
    length: int = 4,
    seed: int = 0,
) -> vs.VideoNode:
    """Create a deterministic, plane-distinct clip with varied sample values."""
    base = core.std.BlankClip(width=width, height=height, format=fmt, length=length)

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        vf = out.format
        for p in range(vf.num_planes):
            arr = np.asarray(out[p])
            h, w = arr.shape
            frame_rng = np.random.default_rng(seed * 1009 + n * 131 + p * 17)
            if vf.sample_type == vs.INTEGER:
                peak = (1 << vf.bits_per_sample) - 1
                arr[:] = frame_rng.integers(0, peak + 1, size=(h, w), dtype=arr.dtype)
                # Ensure extreme values 0 and peak appear in visible pixels.
                arr[0, 0] = 0
                arr[h - 1, w - 1] = peak
            else:
                lo = -0.5 if (vf.color_family == vs.YUV and p > 0) else 0.0
                hi = 0.5 if (vf.color_family == vs.YUV and p > 0) else 1.0
                vals = frame_rng.uniform(lo, hi, size=(h, w)).astype(arr.dtype)
                vals[0, 0] = lo
                vals[h - 1, w - 1] = hi
                arr[:] = vals
        return out

    return base.std.ModifyFrame(base, fill)


def make_f32_clip(core: vs.Core, rows: np.ndarray, length: int = 1) -> vs.VideoNode:
    """Create a GRAYS clip containing exact 2D float32 values."""
    arr2d = np.ascontiguousarray(rows, dtype=np.float32)
    h, w = arr2d.shape
    base = core.std.BlankClip(width=w, height=h, format=vs.GRAYS, length=length)

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        _ = n
        out = f.copy()
        np.asarray(out[0])[:] = arr2d
        return out

    return base.std.ModifyFrame(base, fill)


def repack(core: vs.Core, clip: vs.VideoNode) -> vs.VideoNode:
    """Copy a clip into freshly allocated frames (eliminating any crop offset/stride)."""
    base = core.std.BlankClip(
        width=clip.width,
        height=clip.height,
        format=clip.format.id,
        length=clip.num_frames,
    )

    def copy_fn(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        _ = n
        src, dst = f[0], f[1].copy()
        for p in range(clip.format.num_planes):
            np.asarray(dst[p])[:] = np.asarray(src[p])
        return dst

    return base.std.ModifyFrame([clip, base], copy_fn)


def assert_same_clip(got: vs.VideoNode, expected: vs.VideoNode, frames: list[int] | None = None) -> None:
    assert got.format.id == expected.format.id
    assert (got.width, got.height, got.num_frames) == (expected.width, expected.height, expected.num_frames)
    idxs = frames if frames is not None else list(range(got.num_frames))
    for n in idxs:
        fg = got.get_frame(n)
        fe = expected.get_frame(n)
        for p in range(got.format.num_planes):
            ag = plane_array(fg, p)
            ae = plane_array(fe, p)
            if np.issubdtype(ag.dtype, np.floating):
                nan_g = np.isnan(ag)
                nan_e = np.isnan(ae)
                assert np.array_equal(nan_g, nan_e), f"NaN mismatch at frame {n} plane {p}"
                mask = ~nan_e
                assert np.array_equal(ag[mask].view(np.uint32) if ag.dtype == np.float32 else ag[mask],
                                      ae[mask].view(np.uint32) if ae.dtype == np.float32 else ae[mask]), (
                    f"float mismatch at frame {n} plane {p}"
                )
            else:
                assert np.array_equal(ag, ae), f"int mismatch at frame {n} plane {p}"


def assert_div255_f32(core: vs.Core, src: vs.VideoNode) -> None:
    """Verify `x 255 /` -> GRAYS against exact IEEE x*(1/255), plus std.Expr off Apple Silicon.
    On macOS arm64 (CI1), std.Expr computes non-IEEE approximate float division."""
    got = core.zexpr.Expr([src], ["x 255 /"], format=vs.GRAYS)
    inv255 = np.float32(1.0) / np.float32(255.0)
    for n in range(src.num_frames):
        fg = got.get_frame(n)
        fs = src.get_frame(n)
        for p in range(src.format.num_planes):
            ag = plane_array(fg, p)
            ae = plane_array(fs, p).astype(np.float32) * inv255
            np.testing.assert_array_equal(ag.view(np.uint32), ae.view(np.uint32))
    if not (sys.platform == "darwin" and platform.machine() == "arm64"):
        assert_same_clip(got, core.std.Expr([src], ["x 255 /"], format=vs.GRAYS))
