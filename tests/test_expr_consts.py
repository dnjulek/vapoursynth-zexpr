"""Tests for per-clip, per-frame built-in constants (7 ExprToken names) and variable-format/res Expr."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
import vapoursynth as vs

from helpers import assert_same_clip, make_clip, plane_array

ALL_SEVEN = ("range_min", "range_max", "range_size", "mask_max", "neutral", "plane_min", "plane_max")
FORMAT_ONLY_FIVE = ("range_min", "range_max", "range_size", "mask_max", "neutral")


def jetpack_value(kind: str, vf: vs.VideoFormat, plane: int, is_full: bool) -> int | float:
    """Independent oracle matching vs-jetpack `af85c5f4` `ExprToken.get_value`."""
    is_chroma = vf.color_family == vs.YUV and plane > 0
    if vf.sample_type == vs.FLOAT:
        if kind in ("range_min", "plane_min"):
            return -0.5 if is_chroma else 0.0
        if kind in ("range_max", "plane_max"):
            return 0.5 if is_chroma else 1.0
        if kind in ("range_size", "mask_max"):
            return 1.0
        if kind == "neutral":
            return 0.0
        raise ValueError(kind)

    b = vf.bits_per_sample
    peak = (1 << b) - 1
    shift8 = b - 8
    if kind == "range_min":
        return 0
    if kind in ("range_max", "mask_max"):
        return peak
    if kind == "range_size":
        return 1 << b
    if kind == "neutral":
        return 1 << (b - 1)
    if kind == "plane_min":
        return 0 if is_full else (16 << shift8)
    if kind == "plane_max":
        return peak if is_full else ((240 if is_chroma else 235) << shift8)
    raise ValueError(kind)


def lit_str(val: int | float, is_float_target: bool) -> str:
    """D1 literal spelling: float targets always carry a decimal point; integer targets are integers."""
    if is_float_target:
        s = repr(float(val))
        return s if ("." in s or "e" in s or "E" in s) else f"{s}.0"
    return str(int(val))


def _with_props(core: vs.Core, clip: vs.VideoNode, fn) -> vs.VideoNode:
    def apply(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for k, v in fn(n).items():
                out.props[k] = v
        return out

    return clip.std.ModifyFrame(clip, apply)


def test_expr_phase_a_all_formats_and_opt1_exact(core: vs.Core) -> None:
    """Built-in constants in Expr match their literal twins bit-for-bit across formats, planes, and opt=0/1."""
    formats = (
        vs.GRAY8, vs.GRAY10, vs.GRAY16, vs.GRAYH, vs.GRAYS,
        vs.RGB24, vs.RGB48, vs.RGBH, vs.RGBS,
        vs.YUV420P8, vs.YUV420P10, vs.YUV420P16, vs.YUV420PH, vs.YUV420PS,
        vs.YUV444P8, vs.YUV444PS,
    )
    for fmt in formats:
        src = make_clip(core, width=32, height=8, fmt=fmt, length=1, seed=int(fmt) & 0xFFFF)
        vf = src.format
        is_flt = vf.sample_type == vs.FLOAT
        kinds = ALL_SEVEN if is_flt else FORMAT_ONLY_FIVE

        for opt in (0, 1):
            for kind in kinds:
                lit_exprs = [
                    f"x {lit_str(jetpack_value(kind, vf, p, True), is_flt)} +"
                    for p in range(vf.num_planes)
                ]
                got_bare = core.zexpr.Expr([src], [f"x {kind} +"], opt=opt)
                got_scoped = core.zexpr.Expr([src], [f"x x:{kind} +"], opt=opt)
                got_dst = core.zexpr.Expr([src], [f"x dst:{kind} +"], opt=opt)
                ref = core.zexpr.Expr([src], lit_exprs, opt=opt)
                assert_same_clip(got_bare, ref)
                assert_same_clip(got_scoped, ref)
                assert_same_clip(got_dst, ref)

    # D1 explicit opt=1 typing test: float targets stay f32 even for 0.0 / 1.0, while integer targets stay i32.
    g8 = core.std.BlankClip(width=16, height=2, format=vs.GRAY8, length=1)
    gs = core.std.BlankClip(width=16, height=2, format=vs.GRAYS, length=1)
    assert_same_clip(
        core.zexpr.Expr([gs], ["X 1000000000 * range_max * 3 *"], opt=1),
        core.zexpr.Expr([gs], ["X 1000000000 * 1.0 * 3 *"], opt=1),
    )
    # Confirm float did NOT wrap like integer `1`!
    got_f = plane_array(core.zexpr.Expr([gs], ["X 1000000000 * range_max * 3 *"], opt=1).get_frame(0), 0)
    assert got_f[0, 1] == np.float32(3_000_000_000.0)
    assert_same_clip(
        core.zexpr.Expr([gs], ["X 1000000000 * 3 * range_min +"], opt=1),
        core.zexpr.Expr([gs], ["X 1000000000 * 3 * 0.0 +"], opt=1),
    )
    assert_same_clip(
        core.zexpr.Expr([g8], ["X 1000000000 * range_max * 3 *"], opt=1),
        core.zexpr.Expr([g8], ["X 1000000000 * 255 * 3 *"], opt=1),
    )

    # Multi-clip & dst format= targeting, plus lerp literal equivalence & variable non-collision (`neutral!`/`neutral@`).
    c8 = make_clip(core, width=32, height=8, fmt=vs.GRAY8, length=1, seed=101)
    c16 = make_clip(core, width=32, height=8, fmt=vs.GRAY16, length=1, seed=102)
    cf = make_clip(core, width=32, height=8, fmt=vs.GRAYS, length=1, seed=103)
    assert_same_clip(
        core.zexpr.Expr([c8, c16], ["x range_max / y src1:range_max / + 0.5 * dst:range_max *"], format=vs.GRAY10),
        core.zexpr.Expr([c8, c16], ["x 255 / y 65535 / + 0.5 * 1023 *"], format=vs.GRAY10),
    )
    assert_same_clip(
        core.zexpr.Expr([c8, cf], ["x y:range_max * dst:mask_max *"], format=vs.GRAYS),
        core.zexpr.Expr([c8, cf], ["x 1.0 * 1.0 *"], format=vs.GRAYS),
    )
    assert_same_clip(
        core.zexpr.Expr([c8, c16], ["x y 0.25 lerp"]),
        core.zexpr.Expr([c8, c16], ["x y x:range_max 255 / 0.25 * lerp"]),
    )
    assert_same_clip(
        core.zexpr.Expr([c8], ["x neutral! neutral@ 1 +"]),
        core.zexpr.Expr([c8], ["x 1 +"]),
    )


def test_expr_phase_a_lut_sharing_and_isolation(core: vs.Core) -> None:
    """Costly u16 gamma shares one dimension-free LUT across Y/U/V; float YUV isolates Y vs U/V keys."""
    c16 = make_clip(core, width=32, height=16, fmt=vs.YUV420P16, length=1, seed=201)
    assert_same_clip(
        core.zexpr.Expr([c16], ["x range_max / 2.2 pow range_max *"]),
        core.zexpr.Expr([c16], ["x 65535 / 2.2 pow 65535 *"]),
    )

    cf = make_clip(core, width=16, height=8, fmt=vs.YUV444PS, length=1, seed=202)
    assert_same_clip(
        core.zexpr.Expr([cf], ["x range_min +"]),
        core.zexpr.Expr([cf], ["x 0.0 +", "x -0.5 +", "x -0.5 +"]),
    )


def test_expr_phase_a_rejections_c4(core: vs.Core) -> None:
    """C4: 32-bit int rejects all 7 built-ins; invalid/out-of-scope built-in tokens rejected."""
    g32 = core.std.BlankClip(width=8, height=4, format=vs.GRAY32, length=1)
    for kind in ALL_SEVEN:
        with pytest.raises(vs.Error, match=kind):
            core.zexpr.Expr([g32], [f"x {kind} +"])

    g8 = core.std.BlankClip(width=8, height=4, format=vs.GRAY8, length=1)
    # Invalid/out-of-scope built-in tokens and OOB clips.
    for bad in ("y:neutral", "range_half", "x:range_half", "x:bits", "x:width", "foo:neutral", "dst:"):
        with pytest.raises(vs.Error):
            core.zexpr.Expr([g8], [bad])


def test_expr_phase_b_range_variants(core: vs.Core) -> None:
    """Phase B: Precompiled 2^k (1 <= k <= 2) _Range variants in Expr, incl. E2 float->int dst, D7 k=2, D3 LUT."""
    # 1. Single-clip _Range switching (0, 1, missing, invalid -1, 2, float 1.0) across GRAY8/10/16, YUV420P8/16, RGB24.
    prop_seq = [
        {"_Range": 0},
        {"_Range": 1},
        {},
        {"_Range": -1},
        {"_Range": 2},
        {"_Range": 1.0},
    ]
    for fmt in (vs.GRAY8, vs.GRAY10, vs.GRAY16, vs.YUV420P8, vs.YUV420P16, vs.RGB24):
        base = make_clip(core, width=32, height=8, fmt=fmt, length=len(prop_seq), seed=(int(fmt) & 0xFFFF) + 77)
        src = _with_props(core, base, lambda n: prop_seq[n])
        vf = src.format
        default_full = vf.color_family == vs.RGB
        expected_full = [False, True, default_full, default_full, default_full, default_full]

        for opt in (0, 1):
            out = core.zexpr.Expr([src], ["x plane_min - plane_max plane_min - / range_max *"], opt=opt)
            for n, is_full in enumerate(expected_full):
                lit_exprs = [
                    f"x {jetpack_value('plane_min', vf, p, is_full)} - "
                    f"{jetpack_value('plane_max', vf, p, is_full)} {jetpack_value('plane_min', vf, p, is_full)} - / "
                    f"{jetpack_value('range_max', vf, p, is_full)} *"
                    for p in range(vf.num_planes)
                ]
                ref = core.zexpr.Expr([base[n : n + 1]], lit_exprs, opt=opt)
                assert_same_clip(out[n : n + 1], ref)

    # 2. Float clip carrying `_Range = 0` is still full range!
    gs_base = make_clip(core, width=16, height=4, fmt=vs.GRAYS, length=2, seed=305)
    gs_rng = _with_props(core, gs_base, lambda n: {"_Range": 0 if n == 0 else 1})
    assert_same_clip(
        core.zexpr.Expr([gs_rng], ["x plane_min + plane_max *"]),
        core.zexpr.Expr([gs_base], ["x 0.0 + 1.0 *"]),
    )

    # 3. E2 critical check: float clip 0 + integer `format=` dst with `dst:plane_min`/`dst:plane_max`!
    # Clip 0 is float, so clip 0's shared bit MUST read `_Range` using `dst`'s integer format (not float!).
    yps = make_clip(core, width=16, height=8, fmt=vs.YUV420PS, length=3, seed=306)
    yps_rng = _with_props(core, yps, lambda n: [{"_Range": 0}, {"_Range": 1}, {}][n])
    out_f2i = core.zexpr.Expr([yps_rng], ["x dst:plane_max dst:plane_min - * dst:plane_min +"], format=vs.YUV420P16)
    out_vf = out_f2i.format
    for n, is_full in enumerate((False, True, False)):
        lit_exprs = [
            f"x {jetpack_value('plane_max', out_vf, p, is_full)} {jetpack_value('plane_min', out_vf, p, is_full)} - * "
            f"{jetpack_value('plane_min', out_vf, p, is_full)} +"
            for p in range(3)
        ]
        assert_same_clip(out_f2i[n : n + 1], core.zexpr.Expr([yps[n : n + 1]], lit_exprs, format=vs.YUV420P16))

    # Reverse `format=`: integer clip 0 + float dst (`x:plane_max` switches, `dst:plane_max` stays 1.0).
    g8_base = make_clip(core, width=16, height=4, fmt=vs.GRAY8, length=2, seed=307)
    g8_rng = _with_props(core, g8_base, lambda n: {"_Range": n})
    out_i2f = core.zexpr.Expr([g8_rng], ["x x:plane_max / dst:plane_max *"], format=vs.GRAYS)
    assert_same_clip(out_i2f[0:1], core.zexpr.Expr([g8_base[0:1]], ["x 235 / 1.0 *"], format=vs.GRAYS))
    assert_same_clip(out_i2f[1:2], core.zexpr.Expr([g8_base[1:2]], ["x 255 / 1.0 *"], format=vs.GRAYS))

    # 4. D7 check: `x:plane_min` + `y:plane_min` + `dst:plane_max` uses k = 2 (clip 0 and dst share 1 bit!),
    # exercising all 4 (x_full, y_full) combinations in one clip.
    cx_base = make_clip(core, width=16, height=4, fmt=vs.GRAY8, length=4, seed=308)
    cy_base = make_clip(core, width=16, height=4, fmt=vs.GRAY8, length=4, seed=309)
    cx = _with_props(core, cx_base, lambda n: {"_Range": n & 1})
    cy = _with_props(core, cy_base, lambda n: {"_Range": (n >> 1) & 1})
    out_k2 = core.zexpr.Expr([cx, cy], ["x x:plane_min - y y:plane_min - + 0.5 * dst:plane_max min"])
    for n in range(4):
        xf = bool(n & 1)
        yf = bool((n >> 1) & 1)
        ref = core.zexpr.Expr(
            [cx_base[n : n + 1], cy_base[n : n + 1]],
            [f"x {0 if xf else 16} - y {0 if yf else 16} - + 0.5 * {255 if xf else 235} min"],
        )
        assert_same_clip(out_k2[n : n + 1], ref)

    # 5. D3 LUT sharing: YUV420P16 explicit limited ([4096, 60160, 65535] Y; [4096, 61440, 65535] U/V) vs full.
    y16_base = make_clip(core, width=32, height=16, fmt=vs.YUV420P16, length=2, seed=310)
    y16_rng = _with_props(core, y16_base, lambda n: {"_Range": n})
    out_lut = core.zexpr.Expr([y16_rng], ["x plane_min - plane_max plane_min - / 2.2 pow range_max *"])
    assert_same_clip(
        out_lut[0:1],
        core.zexpr.Expr(
            [y16_base[0:1]],
            [
                "x 4096 - 60160 4096 - / 2.2 pow 65535 *",
                "x 4096 - 61440 4096 - / 2.2 pow 65535 *",
                "x 4096 - 61440 4096 - / 2.2 pow 65535 *",
            ],
        ),
    )
    assert_same_clip(
        out_lut[1:2],
        core.zexpr.Expr([y16_base[1:2]], ["x 0 - 65535 0 - / 2.2 pow 65535 *"]),
    )

    # 6. C12 note: R74+ VapourSynth remaps `_ColorRange` writes (1=limited, 0=full) onto `_Range` (0=limited, 1=full).
    g8_cr = _with_props(core, g8_base, lambda n: {"_ColorRange": 1 if n == 0 else 0})
    out_cr = core.zexpr.Expr([g8_cr], ["plane_max"])
    assert np.all(plane_array(out_cr.get_frame(0), 0) == 235)
    assert np.all(plane_array(out_cr.get_frame(1), 0) == 255)


def test_select_builtins_phase_a(core: vs.Core) -> None:
    """Select: C2 rejects dst:plane_min/max; chroma-aware dst:range_min; per-frame _Range on prop_src; F5 GRAY24."""
    c0_yuv = core.std.BlankClip(width=16, height=8, format=vs.YUV420PS, color=[0.1, -0.1, 0.2], length=2)
    c1_yuv = core.std.BlankClip(width=16, height=8, format=vs.YUV420PS, color=[0.8, 0.3, -0.3], length=2)
    p_gray = core.std.BlankClip(width=8, height=4, format=vs.GRAYS, length=2)

    # C2: dst:plane_min and dst:plane_max rejected in Create.
    for bad in ("dst:plane_min", "dst:plane_max"):
        with pytest.raises(vs.Error):
            core.zexpr.Select([c0_yuv, c1_yuv], [p_gray], [bad])

    # GRAYS prop_src vs YUV420PS clip_src: x:range_min is 0.0 everywhere, dst:range_min is -0.5 on U/V.
    sel_x = core.zexpr.Select([c0_yuv, c1_yuv], [p_gray], ["x:range_min 0 <"])
    sel_dst = core.zexpr.Select([c0_yuv, c1_yuv], [p_gray], ["dst:range_min 0 <"])
    fx = sel_x.get_frame(0)
    fd = sel_dst.get_frame(0)
    assert np.allclose(plane_array(fx, 0), 0.1)
    assert np.allclose(plane_array(fx, 1), -0.1)
    assert np.allclose(plane_array(fx, 2), 0.2)
    assert np.allclose(plane_array(fd, 0), 0.1)  # Y: dst:range_min == 0.0 -> clip 0
    assert np.allclose(plane_array(fd, 1), 0.3)  # U: dst:range_min == -0.5 < 0 -> clip 1
    assert np.allclose(plane_array(fd, 2), -0.3)  # V: dst:range_min == -0.5 < 0 -> clip 1

    # D11: Per-frame `_Range` switching on integer prop_src works in Phase A!
    g0 = core.std.BlankClip(width=8, height=4, format=vs.GRAY8, color=[10], length=3)
    g1 = core.std.BlankClip(width=8, height=4, format=vs.GRAY8, color=[200], length=3)
    p_rng = _with_props(core, g0, lambda n: {"_Range": [0, 1, 0][n]})
    sel_rng = core.zexpr.Select([g0, g1], [p_rng], ["plane_max 255 ="])
    assert [int(plane_array(sel_rng.get_frame(n), 0)[0, 0]) for n in range(3)] == [10, 200, 10]

    # F5: GRAY24 keeps working for ordinary props & unrelated clips, while built-ins targeting GRAY24 reject.
    fmt24 = core.query_video_format(vs.GRAY, vs.INTEGER, 24, 0, 0)
    g24 = core.std.BlankClip(width=8, height=4, format=fmt24.id, length=1)
    p24 = _with_props(core, g24, lambda n: {"Pick": 1})
    assert core.zexpr.Select([g24, g24], [p24], ["x.Pick"]).get_frame(0).format.bits_per_sample == 24
    with pytest.raises(vs.Error):
        core.zexpr.Select([g0, g1], [g24], ["range_max"])
    # Unrelated GRAY24 in prop_src[1] must NOT block x:range_max targeting GRAY8 prop_src[0].
    sel_mix = core.zexpr.Select([g0, g1], [g0, g24], ["x:range_max 255 ="])
    assert int(plane_array(sel_mix.get_frame(0), 0)[0, 0]) == 200


def test_prop_expr_builtins_phase_a(core: vs.Core) -> None:
    """PropExpr: multi-clip, per-frame _Range switching, variable-format, F1 lerp contract, and F5 GRAY24."""
    g8 = core.std.BlankClip(width=8, height=4, format=vs.GRAY8, length=4)
    g16 = core.std.BlankClip(width=8, height=4, format=vs.GRAY16, length=4)
    gs = core.std.BlankClip(width=8, height=4, format=vs.GRAYS, length=4)

    # Multi-clip + per-frame _Range switching (incl. invalid -1, 2, float 1.0 falling back to limited default).
    g8_rng = _with_props(
        core,
        g8,
        lambda n: [
            {"_Range": 0},
            {"_Range": 1},
            {"_Range": -1},
            {"_Range": 1.0},
        ][n],
    )
    out = core.zexpr.PropExpr(
        [g8_rng, g16],
        PMin="plane_min",
        PMax="plane_max",
        XMax="x:range_max",
        YMax="y:range_max",
        DNeu="dst:neutral",
    )
    expected_pmin = [16, 0, 16, 16]
    expected_pmax = [235, 255, 235, 235]
    for n in range(4):
        pr = out.get_frame(n).props
        assert pr["PMin"] == expected_pmin[n]
        assert pr["PMax"] == expected_pmax[n]
        assert pr["XMax"] == 255
        assert pr["YMax"] == 65535
        assert pr["DNeu"] == 128

    # RGB24 carrying explicit _Range = 0 (limited) vs missing (full default).
    rgb = core.std.BlankClip(width=8, height=4, format=vs.RGB24, length=2)
    rgb_r = _with_props(core, rgb, lambda n: {"_Range": 0} if n == 0 else {})
    out_rgb = core.zexpr.PropExpr([rgb_r], PMax="plane_max")
    assert out_rgb.get_frame(0).props["PMax"] == 235
    assert out_rgb.get_frame(1).props["PMax"] == 255

    # Variable-format PropExpr: spliced GRAY8 + GRAY16 + GRAYS.
    var = core.std.Splice([g8[:1], g16[:1], gs[:1]], mismatch=True)
    out_var = core.zexpr.PropExpr([var], RMax="range_max", Neu="neutral")
    assert (out_var.get_frame(0).props["RMax"], out_var.get_frame(0).props["Neu"]) == (255, 128)
    assert (out_var.get_frame(1).props["RMax"], out_var.get_frame(1).props["Neu"]) == (65535, 32768)
    assert (out_var.get_frame(2).props["RMax"], out_var.get_frame(2).props["Neu"]) == (1, 0)

    # F1 scalar-filter lerp contract: prop-slot `neutral` matches loaded prop `x.T` (-inf),
    # while compile-time literal `128` takes `a * (1 - t) + b * t` (finite -3.3909e38).
    p_lerp = _with_props(core, g8[:1], lambda n: {"A": 2.67e36, "B": 0.0, "T": 128})
    out_lerp = core.zexpr.PropExpr(
        [p_lerp],
        FromLit="x.A x.B 128 lerp",
        FromProp="x.A x.B x.T lerp",
        FromBuiltin="x.A x.B neutral lerp",
    )
    pr_l = out_lerp.get_frame(0).props
    assert math.isfinite(pr_l["FromLit"]) and pr_l["FromLit"] < -3.0e38
    assert math.isinf(pr_l["FromProp"]) and pr_l["FromProp"] < 0
    assert math.isinf(pr_l["FromBuiltin"]) and pr_l["FromBuiltin"] < 0

    # F5 GRAY24: ordinary prop expr & unrelated clip succeed; targeting GRAY24 rejects in Create or per-frame.
    fmt24 = core.query_video_format(vs.GRAY, vs.INTEGER, 24, 0, 0)
    g24 = core.std.BlankClip(width=8, height=4, format=fmt24.id, length=1)
    p24 = _with_props(core, g24, lambda n: {"Q": 3})
    assert core.zexpr.PropExpr([p24], R="x.Q 1 +").get_frame(0).props["R"] == 4
    with pytest.raises(vs.Error):
        core.zexpr.PropExpr([g24], R="range_max")
    assert core.zexpr.PropExpr([g8[:1], g24], R="x:range_max").get_frame(0).props["R"] == 255
    var24 = core.std.Splice([g8[:1], g24], mismatch=True)
    out_var24 = core.zexpr.PropExpr([var24], R="range_max")
    assert out_var24.get_frame(0).props["R"] == 255
    with pytest.raises(vs.Error, match=r"R.*range_max"):
        out_var24.get_frame(1)


def test_expr_phase_c_k3_range_dynamic(core: vs.Core) -> None:
    """Phase C: k > 2 distinct integer `_Range` clips fall through cleanly to dynamic mode."""
    for fmt in (vs.GRAY8, vs.YUV420P16):
        cx_base = make_clip(core, width=32, height=8, fmt=fmt, length=8, seed=401)
        cy_base = make_clip(core, width=32, height=8, fmt=fmt, length=8, seed=402)
        cz_base = make_clip(core, width=32, height=8, fmt=fmt, length=8, seed=403)
        cx = _with_props(core, cx_base, lambda n: {"_Range": n & 1})
        cy = _with_props(core, cy_base, lambda n: {"_Range": (n >> 1) & 1})
        cz = _with_props(core, cz_base, lambda n: {"_Range": (n >> 2) & 1})
        vf = cx_base.format

        out = core.zexpr.Expr(
            [cx, cy, cz],
            ["x x:plane_min - y y:plane_min - + z z:plane_max - + dst:neutral +"],
        )
        assert out.format.id == fmt
        for n in range(8):
            xf = bool(n & 1)
            yf = bool((n >> 1) & 1)
            zf = bool((n >> 2) & 1)
            lit_exprs = [
                f"x {jetpack_value('plane_min', vf, p, xf)} - "
                f"y {jetpack_value('plane_min', vf, p, yf)} - + "
                f"z {jetpack_value('plane_max', vf, p, zf)} - + "
                f"{jetpack_value('neutral', vf, p, xf)} +"
                for p in range(vf.num_planes)
            ]
            ref = core.zexpr.Expr(
                [cx_base[n : n + 1], cy_base[n : n + 1], cz_base[n : n + 1]],
                lit_exprs,
            )
            assert_same_clip(out[n : n + 1], ref)


def test_expr_phase_c_variable_format_and_resolution(core: vs.Core) -> None:
    """Phase C: variable-format, variable-resolution, format=, Axis 5 copy planes, F3 GRAYH scalar cache, F4 17x9."""
    # 1. Variable-format splice across YUV420P8/10/16/H/S + GRAY8 + RGB24 with opt=0 and opt=1.
    fmts = (vs.YUV420P8, vs.YUV420P10, vs.YUV420P16, vs.YUV420PH, vs.YUV420PS, vs.GRAY8, vs.RGB24)
    parts = []
    for idx, fmt in enumerate(fmts):
        c = make_clip(core, width=32, height=8, fmt=fmt, length=1, seed=500 + idx)
        parts.append(_with_props(core, c, lambda _n, i=idx: {"_Range": i & 1}))
    var_clip = core.std.Splice(parts, mismatch=True)
    assert var_clip.format.id == vs.NONE
    assert (var_clip.width, var_clip.height) == (32, 8)

    for opt in (0, 1):
        out = core.zexpr.Expr([var_clip], ["x neutral - plane_max + 0.5 *"], opt=opt)
        assert out.format.id == vs.NONE
        assert (out.width, out.height) == (32, 8)
        for idx, part in enumerate(parts):
            vf = part.format
            is_flt = vf.sample_type == vs.FLOAT
            is_full = True if is_flt else bool(idx & 1)
            lit_exprs = [
                f"x {lit_str(jetpack_value('neutral', vf, p, is_full), is_flt)} - "
                f"{lit_str(jetpack_value('plane_max', vf, p, is_full), is_flt)} + 0.5 *"
                for p in range(vf.num_planes)
            ]
            ref = core.zexpr.Expr([part], lit_exprs, opt=opt)
            got_f = out.get_frame(idx)
            ref_f = ref.get_frame(0)
            assert got_f.format.id == ref_f.format.id
            for p in range(vf.num_planes):
                assert np.array_equal(plane_array(got_f, p), plane_array(ref_f, p))

    # 2. Axis 5: Mixed 1-plane GRAY8 + 3-plane YUV420P8 with ["x neutral +", ""] (copy planes on YUV, ignored on GRAY).
    g8 = make_clip(core, width=32, height=8, fmt=vs.GRAY8, length=1, seed=520)
    y8 = make_clip(core, width=32, height=8, fmt=vs.YUV420P8, length=1, seed=521)
    mix = core.std.Splice([g8, y8], mismatch=True)
    out_mix = core.zexpr.Expr([mix], ["x neutral +", ""])
    assert np.array_equal(plane_array(out_mix.get_frame(0), 0), plane_array(core.zexpr.Expr([g8], ["x 128 +"]).get_frame(0), 0))
    fy = out_mix.get_frame(1)
    assert np.array_equal(plane_array(fy, 0), plane_array(core.zexpr.Expr([y8], ["x 128 +", ""]).get_frame(0), 0))
    assert np.array_equal(plane_array(fy, 1), plane_array(y8.get_frame(0), 1))
    assert np.array_equal(plane_array(fy, 2), plane_array(y8.get_frame(0), 2))

    # 3. Variable-format + `format=` override (incl. E2 float->int `dst:plane_max` dynamically) + advertised out_vi.
    y_p8 = _with_props(core, make_clip(core, width=32, height=8, fmt=vs.YUV420P8, length=1, seed=530), lambda _n: {"_Range": 0})
    y_ps = _with_props(core, make_clip(core, width=32, height=8, fmt=vs.YUV420PS, length=1, seed=531), lambda _n: {"_Range": 0})
    y_ps_full = _with_props(core, make_clip(core, width=32, height=8, fmt=vs.YUV420PS, length=1, seed=532), lambda _n: {"_Range": 1})
    var_y = core.std.Splice([y_p8, y_ps, y_ps_full], mismatch=True)
    out_ov = core.zexpr.Expr([var_y], ["x x:range_max / dst:plane_max dst:plane_min - * dst:plane_min +"], format=vs.YUV420P16)
    assert out_ov.format.id == vs.YUV420P16
    assert (out_ov.width, out_ov.height) == (32, 8)
    for idx, (src_c, is_full) in enumerate(((y_p8, False), (y_ps, False), (y_ps_full, True))):
        svf = src_c.format
        dvf = out_ov.format
        sflt = svf.sample_type == vs.FLOAT
        lit_exprs = [
            f"x {lit_str(jetpack_value('range_max', svf, p, is_full), sflt)} / "
            f"{jetpack_value('plane_max', dvf, p, is_full)} {jetpack_value('plane_min', dvf, p, is_full)} - * "
            f"{jetpack_value('plane_min', dvf, p, is_full)} +"
            for p in range(3)
        ]
        ref = core.zexpr.Expr([src_c], lit_exprs, format=vs.YUV420P16)
        assert_same_clip(out_ov[idx : idx + 1], ref)

    # 4. F3: Variable-format GRAY8 + GRAYH with non-LUT spatial expression (exercises cached `.scalar` on SSE2/v2,
    # L0 bypass, and JIT on AVX2/AVX-512).
    gh = make_clip(core, width=33, height=5, fmt=vs.GRAYH, length=1, seed=540)
    g8_33 = make_clip(core, width=33, height=5, fmt=vs.GRAY8, length=1, seed=541)
    var_f3 = core.std.Splice([g8_33, gh, g8_33, gh], mismatch=True)
    out_f3 = core.zexpr.Expr([var_f3], ["x[-1,0] neutral +"])
    ref_g8 = core.zexpr.Expr([g8_33], ["x[-1,0] 128 +"]).get_frame(0)
    ref_gh = core.zexpr.Expr([gh], ["x[-1,0] 0.0 +"]).get_frame(0)
    for idx in (0, 2):
        assert np.array_equal(plane_array(out_f3.get_frame(idx), 0), plane_array(ref_g8, 0))
    for idx in (1, 3):
        assert np.array_equal(plane_array(out_f3.get_frame(idx), 0), plane_array(ref_gh, 0))

    # 5. F4 + cross-clip geometry: 17x9 varformat + format=vs.YUV420P8 rejected in Create;
    # 17x9 varformat + YUV420P8 varsize rejected in Create in BOTH clip orders;
    # varformat GRAY input with format=vs.GRAY16 still accepts ["x", ""] (Axis 5).
    odd_var = core.std.Splice(
        [
            core.std.BlankClip(width=17, height=9, format=vs.GRAY8, length=1),
            core.std.BlankClip(width=17, height=9, format=vs.GRAY16, length=1),
        ],
        mismatch=True,
    )
    with pytest.raises(vs.Error, match="subsampling"):
        core.zexpr.Expr([odd_var], ["x"], format=vs.YUV420P8)

    odd_yuv_varfmt = core.std.BlankClip(width=17, height=9, format=vs.YUV444P8, varformat=True, length=1)
    even_yuv420_varsize = core.std.BlankClip(width=18, height=10, format=vs.YUV420P8, varsize=True, length=1)
    for clips in ([odd_yuv_varfmt, even_yuv420_varsize], [even_yuv420_varsize, odd_yuv_varfmt]):
        with pytest.raises(vs.Error, match="subsampling"):
            core.zexpr.Expr(clips, ["x y +"])

    vf_gray = core.std.BlankClip(width=16, height=8, format=vs.GRAY8, color=[42], varformat=True, length=1)
    out_vf_no_ov = core.zexpr.Expr([vf_gray], ["x", ""])
    out_vf_ov = core.zexpr.Expr([vf_gray], ["x", ""], format=vs.GRAY16)
    assert np.all(plane_array(out_vf_no_ov.get_frame(0), 0) == 42)
    assert out_vf_ov.format.id == vs.GRAY16
    assert np.all(plane_array(out_vf_ov.get_frame(0), 0) == 42)
    # Unused extra plane expression on varformat clip must still be syntax-checked in Create.
    with pytest.raises(vs.Error):
        core.zexpr.Expr([vf_gray], ["x", "x +"], format=vs.GRAY16)
    # Fixed GRAY input still strictly rejects >1 expression.
    with pytest.raises(vs.Error):
        core.zexpr.Expr([g8], ["x", ""], format=vs.GRAY16)

    # Opus Finding 3: variable-size clip 0 paired with a fixed-size clip 1 advertises the fixed size
    # in BOTH input orders (`[var_size, fixed]` and `[fixed, var_size]`).
    vs_gray = core.std.BlankClip(width=16, height=8, format=vs.GRAY8, color=[10], varsize=True, length=1)
    fx_gray = core.std.BlankClip(width=16, height=8, format=vs.GRAY8, color=[20], length=1)
    out_vf_order = core.zexpr.Expr([vs_gray, fx_gray], ["x y +"])
    out_fv_order = core.zexpr.Expr([fx_gray, vs_gray], ["x y +"])
    assert (out_vf_order.width, out_vf_order.height) == (16, 8)
    assert (out_fv_order.width, out_fv_order.height) == (16, 8)
    assert np.all(plane_array(out_vf_order.get_frame(0), 0) == 30)
    assert np.all(plane_array(out_fv_order.get_frame(0), 0) == 30)

    # 6. Variable-resolution multi-clip splice across SIMD boundary widths and spatial heights.
    dims = [(1, 1), (7, 2), (8, 3), (15, 5), (16, 2), (17, 3), (32, 5), (33, 2), (64, 3), (65, 5)]
    res_x = [make_clip(core, width=w, height=h, fmt=vs.GRAY8, length=1, seed=600 + i) for i, (w, h) in enumerate(dims)]
    res_y = [make_clip(core, width=w, height=h, fmt=vs.GRAY16, length=1, seed=700 + i) for i, (w, h) in enumerate(dims)]
    vx = core.std.Splice(res_x, mismatch=True)
    vy = core.std.Splice(res_y, mismatch=True)
    assert (vx.width, vx.height) == (0, 0)
    assert vx.format.id == vs.GRAY8
    out_res = core.zexpr.Expr([vx, vy], ["x[-1,0] y[1,0] 256 / + 0.5 * neutral +"])
    assert (out_res.width, out_res.height) == (0, 0)
    assert out_res.format.id == vs.GRAY8
    for i, (cx_i, cy_i) in enumerate(zip(res_x, res_y)):
        ref_i = core.zexpr.Expr([cx_i, cy_i], ["x[-1,0] y[1,0] 256 / + 0.5 * 128 +"]).get_frame(0)
        got_i = out_res.get_frame(i)
        assert (got_i.width, got_i.height) == (ref_i.width, ref_i.height)
        assert np.array_equal(plane_array(got_i, 0), plane_array(ref_i, 0))


def test_expr_phase_c_dynamic_errors_subprocess(zexpr_lib) -> None:
    """Phase C (C3 + E3): Dynamic per-frame mismatches raise vs.Error cleanly without process abort."""
    script = r"""
import sys
import vapoursynth as vs

core = vs.core
core.std.LoadPlugin(sys.argv[1])

g8_16 = core.std.BlankClip(width=16, height=8, format=vs.GRAY8, length=1)
g8_32 = core.std.BlankClip(width=32, height=8, format=vs.GRAY8, length=1)
y8_16 = core.std.BlankClip(width=16, height=8, format=vs.YUV420P8, length=1)
g32_16 = core.std.BlankClip(width=16, height=8, format=vs.GRAY32, length=1)
fmt24 = core.query_video_format(vs.GRAY, vs.INTEGER, 24, 0, 0)
g24_16 = core.std.BlankClip(width=16, height=8, format=fmt24.id, length=1)

def expect_err(clip, frame_idx, must_contain=()):
    try:
        clip.get_frame(frame_idx)
    except vs.Error as ex:
        msg = str(ex)
        for token in must_contain:
            assert token in msg, f"expected {token!r} in error {msg!r}"
        return
    raise AssertionError(f"expected vs.Error on frame {frame_idx}")

# 1. Cross-clip dimension mismatch on frame 1
c_a = core.std.Splice([g8_16, g8_32], mismatch=True)
c_b = core.std.Splice([g8_16, g8_16], mismatch=True)
out1 = core.zexpr.Expr([c_a, c_b], ["x y +"])
out1.get_frame(0)
expect_err(out1, 1, ("dimensions",))

# 2. Cross-clip plane/subsampling mismatch on frame 1
c_c = core.std.Splice([g8_16, y8_16], mismatch=True)
out2 = core.zexpr.Expr([c_c, c_b], ["x y +"])
out2.get_frame(0)
expect_err(out2, 1, ("plane count",))

# 3. Unsupported 24-bit integer frame arriving in varformat clip
c_24 = core.std.Splice([g8_16, g24_16], mismatch=True)
out3 = core.zexpr.Expr([c_24], ["x 1 +"])
out3.get_frame(0)
expect_err(out3, 1, ("24-bit",))

# 4. format= override mismatch when GRAY8 arrives on YUV420P16 override
c_ov = core.std.Splice([y8_16, g8_16], mismatch=True)
out4 = core.zexpr.Expr([c_ov], ["x"], format=vs.YUV420P16)
out4.get_frame(0)
expect_err(out4, 1, ("color family",))

# 5. E3: GRAY32 frame arriving with built-in constant names both plane and constant
c_32 = core.std.Splice([g8_16, g32_16], mismatch=True)
out5 = core.zexpr.Expr([c_32], ["x neutral +"])
out5.get_frame(0)
expect_err(out5, 1, ("plane 0", "neutral"))

print("ALL_DYNAMIC_ERRORS_CLEAN")
"""
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-B", "-c", script, str(zexpr_lib)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, f"subprocess failed (rc={proc.returncode}):\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert "ALL_DYNAMIC_ERRORS_CLEAN" in proc.stdout


def test_expr_phase_c_parallel_stress(core: vs.Core) -> None:
    """Phase C: Multi-threaded Parallel stress alternating formats and `_Range` over 120 frames."""
    base_clips = [
        make_clip(core, width=32, height=8, fmt=vs.GRAY8, length=1, seed=801),
        make_clip(core, width=32, height=8, fmt=vs.GRAY10, length=1, seed=802),
        make_clip(core, width=32, height=8, fmt=vs.GRAY16, length=1, seed=803),
        make_clip(core, width=32, height=8, fmt=vs.GRAYS, length=1, seed=804),
    ]
    # Precompute expected arrays for each (fmt_idx, is_full) pair.
    expected = {}
    for fi, bc in enumerate(base_clips):
        vf = bc.format
        is_flt = vf.sample_type == vs.FLOAT
        for rng_bit in (0, 1):
            is_full = True if is_flt else bool(rng_bit)
            lit = (
                f"x {lit_str(jetpack_value('plane_min', vf, 0, is_full), is_flt)} - "
                f"{lit_str(jetpack_value('plane_max', vf, 0, is_full), is_flt)} +"
            )
            expected[(fi, rng_bit)] = plane_array(core.zexpr.Expr([bc], [lit]).get_frame(0), 0).copy()

    seq = [
        _with_props(core, base_clips[i % 4], lambda _n, r=(i // 4) & 1: {"_Range": r})
        for i in range(120)
    ]
    var_clip = core.std.Splice(seq, mismatch=True)
    out = core.zexpr.Expr([var_clip], ["x plane_min - plane_max +"])

    for i, frame in enumerate(out.frames(backlog=16)):
        exp = expected[(i % 4, (i // 4) & 1)]
        assert np.array_equal(plane_array(frame, 0), exp), f"mismatch on parallel stress frame {i}"
