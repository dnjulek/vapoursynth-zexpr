"""Phase 3 format and opt=1 tests: u8/u9..16/f16/f32, format= override, opt=1 i32 mode."""

from __future__ import annotations

import numpy as np
import pytest
import vapoursynth as vs

from helpers import assert_div255_f32, assert_same_clip, make_clip, plane_array


@pytest.mark.parametrize("fmt", [vs.GRAYH, vs.YUV444PH, vs.RGBH])
@pytest.mark.parametrize("width", [1, 4, 7, 8, 9, 17, 33])
def test_f16_formats_and_spatial(core: vs.Core, fmt: vs.PresetVideoFormat, width: int) -> None:
    src = make_clip(core, width=width, height=5, fmt=fmt, length=2, seed=505)
    got = core.zexpr.Expr([src], ["x 0.5 * 0.1 +"])
    ref = core.std.Expr([src], ["x 0.5 * 0.1 +"])
    assert_same_clip(got, ref)

    # f16 spatial relative + dynamic gather verified against exact NumPy f16 oracle
    src_p0 = plane_array(src.get_frame(0), 0)
    src_f32 = src_p0.astype(np.float32)
    xs = np.arange(width)
    left_idx = np.clip(xs - 1, 0, width - 1)
    right_idx = np.where(xs + 1 < width, xs + 1, 2 * width - 1 - (xs + 1)) if width > 1 else np.zeros(width, dtype=int)
    expected_rel = ((src_f32[:, left_idx] + src_f32[:, right_idx]) * np.float32(0.5)).astype(np.float16)
    rel_got = plane_array(core.zexpr.Expr([src], ["x[-1,0]:c x[1,0]:m + 0.5 *"]).get_frame(0), 0)
    np.testing.assert_array_equal(rel_got, expected_rel)

    abs_got = plane_array(core.zexpr.Expr([src], ["width 1 - X - Y x[]:c"]).get_frame(0), 0)
    np.testing.assert_array_equal(abs_got, src_p0[:, ::-1])


@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY9, vs.GRAY10, vs.GRAY12, vs.GRAY14, vs.GRAY16])
def test_all_integer_bit_depths_and_format_override(core: vs.Core, fmt: vs.PresetVideoFormat) -> None:
    src = make_clip(core, width=19, height=5, fmt=fmt, length=1, seed=606)
    got = core.zexpr.Expr([src], ["x 2 *"])
    ref = core.std.Expr([src], ["x 2 *"])
    assert_same_clip(got, ref)

    # Cross-format conversion via format= override
    assert_div255_f32(core, src)


def test_opt1_i32_precision_and_f32_promotion(core: vs.Core) -> None:
    src = core.std.BlankClip(width=17, height=3, format=vs.GRAY16, color=[100])

    # In opt=0 (f32), 16777217 rounds to 16777216 so (16777217 + 1) - 16777216 == 0.
    # In opt=1 (i32), 16777217 + 1 - 16777216 == 2!
    r_opt0 = plane_array(core.zexpr.Expr([src], ["x 0 * 16777217 + 1 + 16777216 -"], opt=0).get_frame(0), 0)
    r_opt1 = plane_array(core.zexpr.Expr([src], ["x 0 * 16777217 + 1 + 16777216 -"], opt=1).get_frame(0), 0)
    assert np.all(r_opt0 == 0)
    assert np.all(r_opt1 == 2)

    # Exact 32-bit bitwise operations on >24-bit integer values in opt=1.
    r_bit = plane_array(core.zexpr.Expr([src], ["x 0x01000001 + 0x000000FF bitand"], opt=1).get_frame(0), 0)
    assert np.all(r_bit == ((100 + 0x01000001) & 0xFF))

    # Promotion to f32 on `/`, `sqrt`, float literal still works cleanly in opt=1.
    r_prom = plane_array(core.zexpr.Expr([src], ["x 3 / round"], opt=1).get_frame(0), 0)
    assert np.all(r_prom == 33)

    # X/Y/N stay i32 in opt=1 until promoted.
    r_xy = plane_array(core.zexpr.Expr([src], ["X 100 * Y + 16777217 + 16777216 -"], opt=1).get_frame(0), 0)
    ys, xs = np.mgrid[0:3, 0:17]
    np.testing.assert_array_equal(r_xy, xs * 100 + ys + 1)

    # V3/V4/V7 regressions: opt=1 typing for pow, constant-cond ternary, and float comparison/logic/bitwise.
    hi8 = core.std.BlankClip(width=17, height=2, format=vs.GRAY8, color=[254])
    # V3: `x 4 pow` must evaluate in f32 (clamps to 255), never wrap in i32 (0).
    assert np.all(plane_array(core.zexpr.Expr([hi8], ["x 4 pow"], opt=1).get_frame(0), 0) == 255)
    assert np.all(plane_array(core.zexpr.Expr([hi8], ["x 1 pow 65536 * 65536 *"], opt=1).get_frame(0), 0) == 255)
    # V4: `1 x 0.5 ?` has static f32 type even when constant condition selects `x`.
    assert np.all(plane_array(core.zexpr.Expr([hi8], ["1 x 0.5 ? 65536 * 65536 *"], opt=1).get_frame(0), 0) == 255)
    # V7: comparisons/logic/bitwise always return i32 under opt=1 (wraps 2147483647 + 1 -> INT_MIN -> 0).
    assert np.all(plane_array(core.zexpr.Expr([hi8], ["x 0.5 + 0 > 2147483647 * 1 +"], opt=1).get_frame(0), 0) == 0)
    assert np.all(plane_array(core.zexpr.Expr([hi8], ["x 0.5 + not 0 = 2147483647 * 1 +"], opt=1).get_frame(0), 0) == 0)


@pytest.mark.parametrize("width", [1, 4, 7, 8, 9, 16, 17, 33])
def test_u32_opt1_opt0_spatial_and_wraparound(core: vs.Core, width: int) -> None:
    """32-bit integer clips: opt=1 raw signed i32 (matching akarin), opt=0 float clamp, spatial rel/abs."""
    height = 5
    base = core.std.BlankClip(width=width, height=height, format=vs.GRAY32, length=1)
    samples = np.array(
        [0, 1, 42, 16_777_217, 100_000_000, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFE, 0xFFFF_FFFF],
        dtype=np.uint32,
    )
    grid_x = np.resize(samples, (height, width))
    grid_y = np.resize(np.roll(samples, 3), (height, width))

    def make_u32_clip(arr: np.ndarray) -> vs.VideoNode:
        def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            _ = n
            out = f.copy()
            np.asarray(out[0])[:] = arr
            return out

        return base.std.ModifyFrame(base, fill)

    cx = make_u32_clip(grid_x)
    cy = make_u32_clip(grid_y)
    xi = grid_x.view(np.int32)
    yi = grid_y.view(np.int32)

    # opt=1: exact 32-bit wrapping arithmetic, bitwise, signed comparisons, and negative stores
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([cx], ["x 1 +"], opt=1).get_frame(0), 0),
        (xi + np.int32(1)).view(np.uint32),
    )
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([cx], ["x -10 +"], opt=1).get_frame(0), 0),
        (xi - np.int32(10)).view(np.uint32),
    )
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([cx, cy], ["x y bitxor"], opt=1).get_frame(0), 0),
        (xi ^ yi).view(np.uint32),
    )
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([cx, cy], ["x y > x y ?"], opt=1).get_frame(0), 0),
        np.where(xi > yi, xi, yi).view(np.uint32),
    )

    # opt=1 spatial relative (:c / :m) and dynamic absolute gather (x[]:c)
    xs = np.arange(width)
    left_idx = np.clip(xs - 1, 0, width - 1)
    right_idx = np.where(xs + 1 < width, xs + 1, 2 * width - 1 - (xs + 1)) if width > 1 else np.zeros(width, dtype=int)
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([cx], ["x[-1,0]:c x[1,0]:m +"], opt=1).get_frame(0), 0),
        (xi[:, left_idx] + xi[:, right_idx]).view(np.uint32),
    )
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([cx], ["width 1 - X - Y x[]:c"], opt=1).get_frame(0), 0),
        grid_x[:, ::-1],
    )

    # opt=0: signed i32 -> f32 load, clamp [0, 2147483520] on store (no akarin 1<<32 UB zeroing)
    expected_opt0 = np.clip(np.rint(np.clip(xi.astype(np.float32), 0.0, 2147483520.0)), 0, 2147483520).astype(np.uint32)
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([cx], ["x"], opt=0).get_frame(0), 0),
        expected_opt0,
    )
    # opt=1 with float-promoted root clamps cleanly too
    expected_fprom = np.clip(np.rint(np.clip(xi.astype(np.float32) + np.float32(0.5), 0.0, 2147483520.0)), 0, 2147483520).astype(np.uint32)
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([cx], ["x 0.5 +"], opt=1).get_frame(0), 0),
        expected_fprom,
    )


def test_u32_cross_format_d4_and_multiplane(core: vs.Core) -> None:
    """Cross-format u8/f32 <-> u32 (incl. u8->u32 LUT), D4 edge floats, and multi-plane YUV420P32."""
    # f32 -> GRAY32 D4 store parity: NaN/+inf/>=2^31 -> 2147483520, -inf/negative -> 0
    vals = np.array(
        [[np.nan, np.inf, -np.inf, -10.5, -0.0, 0.0, 0.5, 1.5, 2.5, 2147483520.0, 2147483648.0, 3e9]],
        dtype=np.float32,
    )
    fsrc = core.std.BlankClip(width=vals.shape[1], height=1, format=vs.GRAYS, length=1)

    def fill_f(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        _ = n
        out = f.copy()
        np.asarray(out[0])[:] = vals
        return out

    fsrc = fsrc.std.ModifyFrame(fsrc, fill_f)
    got = plane_array(core.zexpr.Expr([fsrc], ["x"], format=vs.GRAY32).get_frame(0), 0)
    expected = np.array(
        [[2147483520, 2147483520, 0, 0, 0, 0, 0, 2, 2, 2147483520, 2147483520, 2147483520]],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(got, expected)

    # u8 -> GRAY32 (costly expr triggers 1D u8->u32 LUT path in lut.zig)
    c8 = make_clip(core, width=33, height=5, fmt=vs.GRAY8, length=1, seed=777)
    arr8 = plane_array(c8.get_frame(0), 0).astype(np.float32)
    lut_out = plane_array(core.zexpr.Expr([c8], ["x 1 + log 1000000 *"], format=vs.GRAY32).get_frame(0), 0)
    expected_lut = np.rint(np.log(arr8 + np.float32(1.0)) * np.float32(1000000.0)).astype(np.uint32)
    np.testing.assert_allclose(lut_out.astype(np.float64), expected_lut.astype(np.float64), rtol=2e-6, atol=2.0)

    # Multi-plane YUV420P32 with empty-plane copy + per-plane opt=1 expressions
    yuv32_fmt = core.query_video_format(vs.YUV, vs.INTEGER, 32, 1, 1).id
    cyuv = core.std.BlankClip(width=18, height=6, format=yuv32_fmt, color=[100000000, 200000000, 300000000], length=1)
    out_yuv = core.zexpr.Expr([cyuv], ["x 5 +", "", "x -50 +"], opt=1)
    f_yuv = out_yuv.get_frame(0)
    assert np.all(plane_array(f_yuv, 0) == 100000005)
    assert np.all(plane_array(f_yuv, 1) == 200000000)
    assert np.all(plane_array(f_yuv, 2) == 299999950)
