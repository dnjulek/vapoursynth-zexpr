"""Phase 2a vertical-slice tests for zexpr.Expr (u8/u16/f32 core ops, D1-D4/D9-D12/D14-D15)."""

from __future__ import annotations

import numpy as np
import pytest
import vapoursynth as vs

from helpers import assert_same_clip, make_clip, make_f32_clip, plane_array, repack


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65])
@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY10, vs.GRAY16, vs.GRAYS])
def test_simd_boundary_widths_vs_std_expr(core: vs.Core, width: int, fmt: vs.PresetVideoFormat) -> None:
    a = make_clip(core, width=width, height=5, fmt=fmt, length=2, seed=11)
    b = make_clip(core, width=width, height=5, fmt=fmt, length=2, seed=23)
    expr = "x y + 0.5 * x y min - abs"
    got = core.zexpr.Expr([a, b], [expr])
    ref = core.std.Expr([a, b], [expr])
    assert_same_clip(got, ref)


@pytest.mark.parametrize("fmt", [vs.YUV420P8, vs.YUV420P10, vs.YUV420P16, vs.YUV444PS, vs.RGB24, vs.RGB48, vs.RGBS])
def test_multiplane_and_empty_expr(core: vs.Core, fmt: vs.PresetVideoFormat) -> None:
    src = make_clip(core, width=34, height=18, fmt=fmt, length=2, seed=7)
    # Plane 0 processed, plane 1 copied zero-copy, plane 2 different expression.
    exprs = ["x 0.5 *", "", "x 2 *"]
    got = core.zexpr.Expr([src], exprs)
    ref = core.std.Expr([src], exprs)
    assert_same_clip(got, ref)


def test_crop_vs_repack_parity(core: vs.Core) -> None:
    src = make_clip(core, width=64, height=32, fmt=vs.YUV420P16, length=2, seed=99)
    cropped = src.std.Crop(left=2, right=6, top=2, bottom=4)
    got = core.zexpr.Expr([cropped], ["x 100 +", "x 2 /", ""])
    ref = core.std.Expr([repack(core, cropped)], ["x 100 +", "x 2 /", ""])
    assert_same_clip(got, ref)


def test_d1_d2_d3_d4_d9_d10_d12_d15_contracts(core: vs.Core) -> None:
    # Edge float32 values covering negative, -0.3, -0.5, -0.0, +0.0, 0.5, 1.5, 2.5, NaN, +-inf, 1e10.
    vals = np.array(
        [[
            -10.5, -2.5, -1.5, -0.5, -0.3, -0.0, 0.0, 0.5,
            1.5, 2.5, 0.7, 2.0, 1e10, np.nan, np.inf, -np.inf,
        ]],
        dtype=np.float32,
    )
    clip = make_f32_clip(core, vals)

    # D1: leading-zero decimal, hex signed i32 wrap, opt=0 f32 arithmetic.
    assert plane_array(core.zexpr.Expr([clip], ["023"]).get_frame(0), 0)[0, 0] == 23.0
    assert plane_array(core.zexpr.Expr([clip], ["0xFFFFFFFF"]).get_frame(0), 0)[0, 0] == -1.0
    assert plane_array(core.zexpr.Expr([clip], ["0x1.8p2"]).get_frame(0), 0)[0, 0] == 6.0
    assert plane_array(core.zexpr.Expr([clip], ["16777217 1 + 16777216 -"]).get_frame(0), 0)[0, 0] == 0.0

    # D2: constant `-1 sqrt` -> NaN (255 on 8-bit), runtime `x sqrt` on negative & NaN -> 0.
    assert np.isnan(plane_array(core.zexpr.Expr([clip], ["-1 sqrt"]).get_frame(0), 0)[0, 0])
    assert plane_array(core.zexpr.Expr([clip], ["-1 sqrt"], format=vs.GRAY8).get_frame(0), 0)[0, 0] == 255
    rt_sqrt = plane_array(core.zexpr.Expr([clip], ["x sqrt"]).get_frame(0), 0)[0]
    assert rt_sqrt[0] == 0.0  # -10.5 -> 0
    assert rt_sqrt[13] == 0.0  # NaN -> 0

    # D3: unfused `%` on constant & runtime (`0.7 0.1 % == 0`, `1e10 3 % == 0`, `-0 3 % == +0.0`).
    assert plane_array(core.zexpr.Expr([clip], ["0.7 0.1 %"]).get_frame(0), 0)[0, 0] == 0.0
    assert plane_array(core.zexpr.Expr([clip], ["1e10 3 %"]).get_frame(0), 0)[0, 0] == 0.0
    rt_mod = plane_array(core.zexpr.Expr([clip], ["x 3 %"]).get_frame(0), 0)[0]
    assert rt_mod[12] == 0.0  # 1e10 % 3 == 0 (never FMA-fused!)
    assert rt_mod[5].view(np.uint32) == 0x00000000  # -0.0 % 3 == +0.0 (even at SSE2 level 1!)

    # D4: NaN & +inf -> peak, -inf -> 0 for integer outputs.
    u8_out = plane_array(core.zexpr.Expr([clip], ["x"], format=vs.GRAY8).get_frame(0), 0)[0]
    assert (u8_out[13], u8_out[14], u8_out[15]) == (255, 255, 0)
    u10_out = plane_array(core.zexpr.Expr([clip], ["x"], format=vs.GRAY10).get_frame(0), 0)[0]
    assert (u10_out[13], u10_out[14], u10_out[15]) == (1023, 1023, 0)

    # D9: half-to-even rounding preserving -0.0 on -0.3, -0.5, -0.0.
    rnd = plane_array(core.zexpr.Expr([clip], ["x round"]).get_frame(0), 0)[0]
    assert (rnd[1], rnd[2], rnd[7], rnd[8], rnd[9]) == (-2.0, -2.0, 0.0, 2.0, 2.0)
    assert rnd[3].view(np.uint32) == 0x80000000  # round(-0.5) == -0.0
    assert rnd[4].view(np.uint32) == 0x80000000  # round(-0.3) == -0.0
    assert rnd[5].view(np.uint32) == 0x80000000  # round(-0.0) == -0.0

    # D10: bitwise half-to-even (`2.5 0 bitor == 2`).
    assert plane_array(core.zexpr.Expr([clip], ["2.5 0 bitor"]).get_frame(0), 0)[0, 0] == 2.0
    assert plane_array(core.zexpr.Expr([clip], ["x 0 bitor"]).get_frame(0), 0)[0, 9] == 2.0

    # D12 & D15: ordered comparisons, logic, `not`, and `?` on NaN (const and runtime).
    assert plane_array(core.zexpr.Expr([clip], ["0 0 / not"]).get_frame(0), 0)[0, 0] == 0.0
    assert plane_array(core.zexpr.Expr([clip], ["x not"]).get_frame(0), 0)[0, 13] == 0.0
    assert plane_array(core.zexpr.Expr([clip], ["x 0 >"]).get_frame(0), 0)[0, 13] == 0.0
    assert plane_array(core.zexpr.Expr([clip], ["x 0 >="]).get_frame(0), 0)[0, 13] == 0.0
    assert plane_array(core.zexpr.Expr([clip], ["x 1 and"]).get_frame(0), 0)[0, 13] == 0.0
    assert plane_array(core.zexpr.Expr([clip], ["x 0 or"]).get_frame(0), 0)[0, 13] == 0.0
    assert plane_array(core.zexpr.Expr([clip], ["x 10 42 ?"]).get_frame(0), 0)[0, 13] == 42.0


def test_props_coords_vars_sort_and_many_clips(core: vs.Core) -> None:
    base = core.std.BlankClip(width=17, height=5, format=vs.GRAYS, length=3)
    with_props = base.std.SetFrameProps(
        IntP=42,
        FloatP=1.5,
        BinP=b"\xc8rest",  # unsigned byte 0xC8 = 200 (D11!)
        EmptyP=b"",
    )
    out = core.zexpr.Expr(
        [with_props],
        ["x.IntP x.FloatP + x.BinP + X + Y 100 * + N 1000 * + width + height + t! t@"],
    )
    for n in range(3):
        arr = plane_array(out.get_frame(n), 0)
        ys, xs = np.mgrid[0:5, 0:17].astype(np.float32)
        expected = np.float32(42 + 1.5 + 200) + xs + ys * 100.0 + np.float32(n * 1000 + 17 + 5)
        np.testing.assert_array_equal(arr, expected)

    # Empty ptData & missing prop yield NaN in Expr (Y17).
    assert np.isnan(plane_array(core.zexpr.Expr([with_props], ["x.EmptyP"]).get_frame(0), 0)[0, 0])
    assert np.isnan(plane_array(core.zexpr.Expr([with_props], ["x.NoSuchProp"]).get_frame(0), 0)[0, 0])

    # sort9 median (smallest on top: drop4 -> median on top, drop4 rest via swap/drop).
    sorted_clip = core.zexpr.Expr([base], ["9 1 8 2 7 3 6 4 5 sort9 drop4 swap4 drop4"])
    # After sort9: stack bottom->top is [9, 8, 7, 6, 5, 4, 3, 2, 1].
    # drop4 -> [9, 8, 7, 6, 5] (5 is on top!). swap4 -> [5, 8, 7, 6, 9], drop4 -> [5].
    assert plane_array(sorted_clip.get_frame(0), 0)[0, 0] == 5.0

    # > 26 clips (`src30`) and shorter clip frame reuse.
    clips = [core.std.BlankClip(width=8, height=4, format=vs.GRAY8, length=1 if i == 30 else 3, color=[i]) for i in range(31)]
    res = core.zexpr.Expr(clips, ["src0 src30 +"])
    assert res.num_frames == 3
    for n in range(3):
        assert np.all(plane_array(res.get_frame(n), 0) == 30)


def test_validation_rejections(core: vs.Core) -> None:
    g8 = core.std.BlankClip(width=16, height=16, format=vs.GRAY8)
    g24_fmt = core.query_video_format(vs.GRAY, vs.INTEGER, 24, 0, 0).id
    g24 = core.std.BlankClip(width=16, height=16, format=g24_fmt)
    yuv420 = core.std.BlankClip(width=16, height=16, format=vs.YUV420P8)

    # 17..31-bit integer clip formats rejected in Create (only 8..16 and 32-bit int supported).
    with pytest.raises(vs.Error):
        core.zexpr.Expr([g24], ["x"])
    with pytest.raises(vs.Error):
        core.zexpr.Expr([g8], ["x"], format=g24_fmt)

    # D14: format= changing colorFamily or subsampling rejected.
    with pytest.raises(vs.Error):
        core.zexpr.Expr([yuv420], ["x"], format=vs.YUV444P8)
    with pytest.raises(vs.Error):
        core.zexpr.Expr([g8], ["x"], format=vs.RGB24)

    # Invalid opt/boundary/expr count/RPN syntax.
    with pytest.raises(vs.Error):
        core.zexpr.Expr([g8], ["x"], opt=2)
    with pytest.raises(vs.Error):
        core.zexpr.Expr([g8], ["x"], boundary=2)
    with pytest.raises(vs.Error):
        core.zexpr.Expr([g8], ["x", "x"])
    with pytest.raises(vs.Error):
        core.zexpr.Expr([g8], ["foo@"])
    with pytest.raises(vs.Error):
        core.zexpr.Expr([g8], ["y"])
