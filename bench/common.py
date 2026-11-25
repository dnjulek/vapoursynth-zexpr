"""Shared helper for multi-format vspipe benchmark scripts in `bench/`.

Output index convention (`-o N`):
  0..2  YUV420P8   (0=zexpr, 1=akarin, 2=cranexpr)
  3..5  YUV420P16  (3=zexpr, 4=akarin, 5=cranexpr)
  6..8  YUV420PS   (6=zexpr, 7=akarin, 8=cranexpr)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import vapoursynth as vs

# vspipe evaluates scripts with the script's own directory (`bench/`) as the working directory,
# while `__file__` may remain relative to the caller's directory; locate the repo from cwd.
ROOT = Path.cwd().parent

# Plugin namespace -> (environment override, default library). akarin and cranexpr are normally
# pip-installed (`vapoursynth-akarin`, `vapoursynth-cranexpr`) and autoloaded by VapourSynth; zexpr
# defaults to this checkout's `zig build` output. A relative override is taken from the repo root.
OPTIONAL = {"cranexpr"}  # its outputs (2/5/8) are only registered when it is available

LIBS: dict[str, tuple[str, Path | None]] = {
    "zexpr": (
        "ZEXPR_LIB",
        ROOT / {"win32": "zig-out/bin/zexpr.dll", "darwin": "zig-out/lib/libzexpr.dylib"}.get(sys.platform, "zig-out/lib/libzexpr.so"),
    ),
    "akarin": ("AKARIN_LIB", None),
    "cranexpr": ("CRANEXPR_LIB", None),
}

FORMATS: tuple[tuple[str, vs.PresetVideoFormat], ...] = (
    ("p8", vs.YUV420P8),
    ("p16", vs.YUV420P16),
    ("ps", vs.YUV420PS),
)


def setup(script_globals: dict[str, Any]) -> vs.Core:
    """Load zexpr, akarin, and cranexpr plugins and apply optional `-a threads=N`.

    An autoloaded (pip-installed) plugin is used as is unless its environment override names a
    different library, which is an error: VapourSynth cannot load a second copy. A pip-installed
    zexpr is used with a warning on stderr (`run.sh` shows it once above the table), since it
    shadows any local `zig build` output.
    """
    core = vs.core
    for ns, (env, default) in LIBS.items():
        override = os.environ.get(env)
        lib = (ROOT / override if not Path(override).is_absolute() else Path(override)) if override else default
        if hasattr(core, ns):
            loaded = Path(getattr(core, ns).plugin_path).resolve()
            if override and loaded != lib.resolve():
                raise RuntimeError(f"{ns} is autoloaded from {loaded}, not {lib}; uninstall the pip package or unset {env}")
            if ns == "zexpr" and not override and default is not None and loaded != default.resolve():
                if default.is_file():
                    why = f"the local build {default} is NOT measured (pip uninstall vapoursynth-zexpr to time it)"
                else:
                    why = "no local build yet (zig build -Doptimize=ReleaseFast, then pip uninstall vapoursynth-zexpr to time it)"
                print(f"WARNING: zexpr runs the pip-installed {loaded}; {why}", file=sys.stderr)
            continue
        if lib is None:
            if ns in OPTIONAL:
                continue
            raise RuntimeError(f"{ns} not found: pip install vapoursynth-{ns}, or set {env} to a built library")
        core.std.LoadPlugin(str(lib))
    if "threads" in script_globals:
        core.num_threads = int(script_globals["threads"])
    return core


def _noise(rng: np.random.Generator, vf: vs.VideoFormat, p: int, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    if vf.sample_type == vs.FLOAT:
        vals = rng.random(shape, dtype=np.float32)
        if vf.color_family == vs.YUV and p > 0:
            vals -= np.float32(0.5)
        return vals
    return rng.integers(0, 1 << vf.bits_per_sample, shape, dtype=dtype)


def random_source(
    core: vs.Core,
    fmt: vs.PresetVideoFormat,
    width: int = 1920,
    height: int = 1080,
    length: int = 100000,
    seed_offset: int = 0,
    eager: bool = False,
) -> vs.VideoNode:
    """8 distinct random frames looped and cached, so LUT and branchy paths see real data.

    With eager=True the 8 frames are generated now, during script evaluation, instead of on first
    request: vspipe's timer then measures only the expression, not the source generation (up to
    ~0.3 s for three float clips). The data is identical either way.
    """
    blank = core.std.BlankClip(format=fmt, width=width, height=height, length=8)
    cache: dict[int, list[np.ndarray]] = {}
    if eager:
        probe = blank.get_frame(0)
        for n in range(8):
            rng = np.random.default_rng(n + seed_offset)
            cache[n] = [
                _noise(rng, probe.format, p, np.asarray(probe[p]).shape, np.asarray(probe[p]).dtype)
                for p in range(probe.format.num_planes)
            ]
        del probe

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        vf = out.format
        if n in cache:
            for p, vals in enumerate(cache[n]):
                np.asarray(out[p])[:] = vals
            return out
        rng = np.random.default_rng(n + seed_offset)
        for p in range(vf.num_planes):
            a = np.asarray(out[p])
            a[:] = _noise(rng, vf, p, a.shape, a.dtype)
        return out

    return blank.std.ModifyFrame(blank, fill).std.Loop(times=length // 8)


def run_case(
    script_globals: dict[str, Any],
    expr: str | list[str] | dict[str, str | list[str]],
    *,
    num_clips: int = 1,
    boundary: int | None = None,
    format_out: dict[str, vs.PresetVideoFormat] | None = None,
    props: dict[str, Any] | None = None,
) -> None:
    """Register 0..2 YUV420P8, 3..5 YUV420P16, 6..8 YUV420PS (zexpr/akarin/cranexpr; cranexpr's 2/5/8 only if installed)."""
    core = setup(script_globals)
    # `-a fmt=p8` (as run.sh passes) registers only that format's outputs, with sources generated before
    # vspipe starts timing; without it all nine outputs are registered and sources are generated lazily.
    only = script_globals.get("fmt")
    if only is not None and only not in dict(FORMATS):
        raise ValueError(f"fmt={only!r}: expected one of {', '.join(t for t, _ in FORMATS)}")
    for fmt_idx, (tag, fmt) in enumerate(FORMATS):
        if only is not None and tag != only:
            continue
        clips = [random_source(core, fmt, seed_offset=i * 100, eager=only is not None) for i in range(num_clips)]
        if props:
            clips = [c.std.SetFrameProps(**props) for c in clips]
        fmt_expr = expr[tag] if isinstance(expr, dict) else expr
        kwargs: dict[str, Any] = {}
        if boundary is not None:
            kwargs["boundary"] = boundary
        if format_out is not None and tag in format_out:
            kwargs["format"] = format_out[tag]
        base = fmt_idx * 3
        core.zexpr.Expr(clips, fmt_expr, **kwargs).set_output(base + 0)
        core.akarin.Expr(clips, fmt_expr, **kwargs).set_output(base + 1)
        if hasattr(core, "cranexpr"):
            core.cranexpr.Expr(clips, fmt_expr, **kwargs).set_output(base + 2)
