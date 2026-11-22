"""Phase 4 tests: Optimizer rewrites, LUT (incl. B4 u16 out-of-range), i16 fast path (Y1/Y2), and R3 `%` atomicity."""

from __future__ import annotations

import numpy as np
import pytest
import vapoursynth as vs

from helpers import assert_div255_f32, assert_same_clip, make_clip, make_f32_clip, plane_array


def test_r3_mod_never_fma_fused(core: vs.Core) -> None:
    """R3: `1e10 3 %` must evaluate to 0.0 (not 256.0 from FMA-fusing `x - trunc(x/y)*y`)."""
    vals = np.array([[1.0e10, 1.0e10 + 1.0, 7.5, -7.5, 3.0, 0.0, 99.0, 123456.0]], dtype=np.float32)
    clip = make_f32_clip(core, vals)
    out = core.zexpr.Expr([clip], ["x 3 %"])
    got = plane_array(out.get_frame(0), 0)[0]
    assert got[0] == 0.0, f"R3 regression: 1e10 % 3 returned {got[0]} instead of 0.0"
    # Verify remaining elements against unfused float32 formula `a - trunc(a / b) * b`.
    b = np.float32(3.0)
    expected = vals[0] - np.trunc(vals[0] / b) * b
    np.testing.assert_array_equal(got, expected)


def test_lut_1d_u8_and_2d_u8_exact(core: vs.Core) -> None:
    """1D u8, native u8/i16, and real 2D u8x8 LUT paths match float-input f32 JIT & NumPy bit-for-bit."""
    for w in (1, 7, 16, 33, 64):
        c0 = make_clip(core, width=w, height=5, fmt=vs.GRAY8, length=2, seed=10 + w)
        c1 = make_clip(core, width=w, height=5, fmt=vs.GRAY8, length=2, seed=20 + w)
        # Float copies disable u8, i16, and LUT paths, forcing the general f32 JIT path.
        f0 = core.std.Expr([c0], ["x"], format=vs.GRAYS)
        f1 = core.std.Expr([c1], ["x"], format=vs.GRAYS)

        for expr in (
            "255 x -",
            "x 255 / 2.2 pow 255 *",
            "x 128 > 255 0 ?",
        ):
            got = core.zexpr.Expr([c0], [expr])
            ref = core.zexpr.Expr([f0], [expr], format=vs.GRAY8)
            assert_same_clip(got, ref)

        # Includes both cheap u8/i16 two-clip ops AND an asymmetric heavy expression that hits lut2d_u8.
        for expr in (
            "x y + 2 /",
            "x y - abs",
            "x 0.3 * y 0.7 * +",
            "x 255 / 2.2 pow y 255 / 1.8 pow + 127.5 *",
        ):
            got = core.zexpr.Expr([c0, c1], [expr])
            ref = core.zexpr.Expr([f0, f1], [expr], format=vs.GRAY8)
            assert_same_clip(got, ref)


def test_lut_1d_u16_b4_out_of_range_samples(core: vs.Core) -> None:
    """B4: 10-bit clip carrying synthetic out-of-range u16 samples > 1023 must not crash or read OOB."""
    base = core.std.BlankClip(width=16, height=4, format=vs.GRAY10, length=1)

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        _ = n
        out = f.copy()
        arr = np.asarray(out[0])
        vals = np.array(
            [0, 1, 511, 1023, 1024, 2048, 4096, 16384, 32768, 40000, 65534, 65535, 100, 500, 900, 1022],
            dtype=np.uint16,
        )
        for row in range(arr.shape[0]):
            arr[row, :] = vals
        return out

    clip = base.std.ModifyFrame(base, fill)
    fclip = core.std.Expr([clip], ["x"], format=vs.GRAYS)
    got_lut = core.zexpr.Expr([clip], ["x 1023 / 2.2 pow 1023 *"])
    got_f32 = core.zexpr.Expr([fclip], ["x 1023 / 2.2 pow 1023 *"], format=vs.GRAY10)
    assert_same_clip(got_lut, got_f32)

    arr_out = plane_array(got_lut.get_frame(0), 0)[0]
    # Out-of-range sample 65535 -> (65535/1023)^2.2 * 1023 >> 1023 -> clamped to 10-bit peak 1023!
    assert arr_out[11] == 1023
    assert arr_out[0] == 0


def test_i16_y1_overflow_guard_and_y2_tight_stride(core: vs.Core) -> None:
    """Y1: `x 128 * 127 + 2 /` at x=255 must equal 255 (no i16 rounding-bias overflow).
    Y2: i16 interior fast path on tight stride widths (31, 32, 33, 64, 65) matches independent NumPy oracle."""
    base = core.std.BlankClip(width=32, height=4, format=vs.GRAY8, length=1)

    def fill_255(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        _ = n
        out = f.copy()
        arr = np.asarray(out[0])
        arr[:] = np.arange(32 * 4, dtype=np.uint8).reshape(4, 32)
        arr[0, 0] = 255
        arr[0, 15] = 255
        arr[0, 31] = 255
        return out

    c = base.std.ModifyFrame(base, fill_255)
    # Y1: 255*128 + 127 = 32767, +1 rounding bias would overflow i16 to -32768 -> must be rejected from i16 path!
    y1_out = core.zexpr.Expr([c], ["x[0,0] 128 * 127 + 2 /"])
    y1_arr = plane_array(y1_out.get_frame(0), 0)
    assert y1_arr[0, 0] == 255
    assert y1_arr[0, 15] == 255
    assert y1_arr[0, 31] == 255

    # Y2: 4-tap i16 blur on tight-stride widths vs independent NumPy oracle AND float-input f32 JIT.
    for w in (31, 32, 33, 64, 65):
        src = make_clip(core, width=w, height=5, fmt=vs.GRAY8, length=1, seed=90 + w)
        fsrc = core.std.Expr([src], ["x"], format=vs.GRAYS)
        blur_i16 = core.zexpr.Expr([src], ["x[-1,0] x + x[1,0] + x[0,-1] + 4 /"])
        blur_f32 = core.zexpr.Expr([fsrc], ["x[-1,0] x + x[1,0] + x[0,-1] + 4 /"], format=vs.GRAY8)
        assert_same_clip(blur_i16, blur_f32)

        a = plane_array(src.get_frame(0), 0).astype(np.float32)
        left = np.concatenate([a[:, :1], a[:, :-1]], axis=1)
        right = np.concatenate([a[:, 1:], a[:, -1:]], axis=1)
        up = np.concatenate([a[:1, :], a[:-1, :]], axis=0)
        expected = np.rint((left + a + right + up) * np.float32(0.25)).astype(np.uint8)
        np.testing.assert_array_equal(plane_array(blur_i16.get_frame(0), 0), expected)


def test_v12_wide_reach_and_v13_u8_to_f32(core: vs.Core) -> None:
    """V12: wide horizontal reach (|dx| > 64) matches NumPy oracle across small & wide frames.
    V13: u8 -> f32 cheap conversion (`x 255 /`) matches std.Expr bit-for-bit."""
    for w in (33, 64, 65, 257, 1920):
        src = make_clip(core, width=w, height=3, fmt=vs.GRAY8, length=1, seed=300 + w)
        got = core.zexpr.Expr([src], ["x[-65,0] x[65,0] max x min"])
        a = plane_array(src.get_frame(0), 0)
        xs = np.arange(w)
        left = a[:, np.clip(xs - 65, 0, w - 1)]
        right = a[:, np.clip(xs + 65, 0, w - 1)]
        expected = np.minimum(np.maximum(left, right), a)
        np.testing.assert_array_equal(plane_array(got.get_frame(0), 0), expected)

    src8 = make_clip(core, width=257, height=5, fmt=vs.GRAY8, length=1, seed=313)
    assert_div255_f32(core, src8)


def test_optimizer_rewrites_and_cse(core: vs.Core) -> None:
    """Optimizer CSE, constant folding, algebraic identities, and x/C -> x*(1/C) produce expected outputs."""
    clip = make_clip(core, width=33, height=5, fmt=vs.GRAYS, length=1, seed=77)
    # CSE on repeated subexpression
    got = core.zexpr.Expr([clip], ["x 0.5 * 0.1 + x 0.5 * 0.1 + *"])
    ref = core.zexpr.Expr([clip], ["x 0.5 * 0.1 + dup *"])
    assert_same_clip(got, ref)

    # x / 255 matches exact IEEE x * (1/255) (and std.Expr off Apple Silicon)
    c8 = make_clip(core, width=33, height=5, fmt=vs.GRAY8, length=1, seed=78)
    assert_div255_f32(core, c8)

    # exp(log(x)) and log(exp(x)) rewrites within plan.md §7 bounds
    pos = make_f32_clip(core, np.array([[0.01, 0.5, 1.0, 2.5, 10.0, 100.0, 1e5, 1e10]], dtype=np.float32))
    el = plane_array(core.zexpr.Expr([pos], ["x log exp"]).get_frame(0), 0)[0]
    np.testing.assert_array_equal(el, np.array([0.01, 0.5, 1.0, 2.5, 10.0, 100.0, 1e5, 1e10], dtype=np.float32))
    le = plane_array(core.zexpr.Expr([pos], ["x exp log"]).get_frame(0), 0)[0]
    np.testing.assert_array_equal(le, np.array([0.01, 0.5, 1.0, 2.5, 10.0, 100.0, 1e5, 1e10], dtype=np.float32))


@pytest.mark.parametrize("expr", [
    "x 255 / 2.2 pow 255 *",
    "x[-1,-1] x[0,-1] + x[1,-1] + x[-1,0] + x + x[1,0] + x[-1,1] + x[0,1] + x[1,1] + 9 /",
    "x y + 2 /",
])
def test_fastpaths_match_std_close(core: vs.Core, expr: str) -> None:
    """Fast paths agree closely with std.Expr on normal u8 video."""
    c0 = make_clip(core, width=37, height=7, fmt=vs.GRAY8, length=1, seed=123)
    c1 = make_clip(core, width=37, height=7, fmt=vs.GRAY8, length=1, seed=124)
    if "x[-1" in expr:
        # std.Expr has no spatial access; compare against float-input f32 JIT path instead.
        f0 = core.std.Expr([c0], ["x"], format=vs.GRAYS)
        got = core.zexpr.Expr([c0], [expr])
        ref = core.zexpr.Expr([f0], [expr], format=vs.GRAY8)
        assert_same_clip(got, ref)
    elif "y" in expr:
        got = core.zexpr.Expr([c0, c1], [expr])
        ref = core.std.Expr([c0, c1], [expr])
        assert_same_clip(got, ref)
    else:
        got = core.zexpr.Expr([c0], [expr])
        ref = core.std.Expr([c0], [expr])
        diff = np.max(np.abs(plane_array(got.get_frame(0), 0).astype(np.int32) -
                             plane_array(ref.get_frame(0), 0).astype(np.int32)))
        assert diff <= 1


def test_native_u16_fastpath_and_u10_out_of_range(core: vs.Core) -> None:
    """G6: native u16 interior fast path matches float-input f32 JIT across u16/u10, widths,
    unvalidated raw u10 samples > 1023, and 16-bit -> 10-bit format conversion."""
    u16_exprs = (
        "x y - abs",
        "x y -",
        "x y +",
        "x y + 1023 min",
        "x y - 0 max",
        "65535 x -",
        "x y min 1000 max",
        "x[-1,-1] x[0,-1] max x[1,-1] max x[-1,0] max x max x[1,0] max x[-1,1] max x[0,1] max x[1,1] max x -",
        "x[-1,-1] x[0,-1] x[1,-1] x[-1,0] x x[1,0] x[-1,1] x[0,1] x[1,1] sort9 drop4 swap4 drop4",
        "x[-65,0] x[65,0] max x min",
    )
    for fmt in (vs.GRAY16, vs.GRAY10):
        for w in (1, 15, 16, 17, 32, 33, 65, 193):
            c0 = make_clip(core, width=w, height=5, fmt=fmt, length=1, seed=400 + w)
            c1 = make_clip(core, width=w, height=5, fmt=fmt, length=1, seed=500 + w)
            f0 = core.std.Expr([c0], ["x"], format=vs.GRAYS)
            f1 = core.std.Expr([c1], ["x"], format=vs.GRAYS)
            for expr in u16_exprs:
                got = core.zexpr.Expr([c0, c1], [expr])
                ref = core.zexpr.Expr([f0, f1], [expr], format=fmt)
                assert_same_clip(got, ref)

    # 10-bit input carrying raw u16 samples > 1023 (up to 65535): must not wrap and must clamp to 1023.
    base10 = core.std.BlankClip(width=65, height=4, format=vs.GRAY10, length=1)

    def fill_raw(seed: int):
        def _fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            _ = n
            out = f.copy()
            arr = np.asarray(out[0])
            rng = np.random.default_rng(seed)
            arr[:] = rng.integers(0, 65536, arr.shape, dtype=np.uint16)
            arr[0, 0] = 65535
            arr[0, 1] = 0
            arr[0, 16] = 65535
            arr[0, 32] = 40000
            return out
        return _fill

    r0 = base10.std.ModifyFrame(base10, fill_raw(701))
    r1 = base10.std.ModifyFrame(base10, fill_raw(702))
    fr0 = core.std.Expr([r0], ["x"], format=vs.GRAYS)
    fr1 = core.std.Expr([r1], ["x"], format=vs.GRAYS)
    for expr in u16_exprs:
        got = core.zexpr.Expr([r0, r1], [expr])
        ref = core.zexpr.Expr([fr0, fr1], [expr], format=vs.GRAY10)
        assert_same_clip(got, ref)
        assert int(np.max(plane_array(got.get_frame(0), 0))) <= 1023

    # 16-bit input written to 10-bit format=vs.GRAY10 via native u16 path.
    c16_0 = make_clip(core, width=65, height=5, fmt=vs.GRAY16, length=1, seed=710)
    c16_1 = make_clip(core, width=65, height=5, fmt=vs.GRAY16, length=1, seed=711)
    f16_0 = core.std.Expr([c16_0], ["x"], format=vs.GRAYS)
    f16_1 = core.std.Expr([c16_1], ["x"], format=vs.GRAYS)
    for expr in u16_exprs:
        got = core.zexpr.Expr([c16_0, c16_1], [expr], format=vs.GRAY10)
        ref = core.zexpr.Expr([f16_0, f16_1], [expr], format=vs.GRAY10)
        assert_same_clip(got, ref)
        assert int(np.max(plane_array(got.get_frame(0), 0))) <= 1023


def test_vsjetpack_optimizations_exact(core: vs.Core) -> None:
    """Periodic X%C + hoisted-root u8 pre-pack, fused boolean logic -> select, i16 cmp/select,
    u8 saturating add/sub, and signed diff-limit collapse match float-input f32 / NumPy exact."""
    # deblock_qed mask across edge and SIMD-boundary widths on GRAY8, GRAY10, and GRAY16
    for w in (1, 7, 8, 9, 15, 16, 17, 33, 64):
        yy, xx = np.mgrid[0:19, 0:w]
        mask = ((xx % 8 % 7) > 0) & ((yy % 8 % 7) > 0)

        c8 = core.std.BlankClip(width=w, height=19, format=vs.GRAY8, length=1)
        got8 = plane_array(core.zexpr.Expr([c8], ["X 8 % 7 % Y 8 % 7 % and 0 255 ?"]).get_frame(0), 0)
        np.testing.assert_array_equal(got8, np.where(mask, 0, 255).astype(np.uint8))

        c10 = core.std.BlankClip(width=w, height=19, format=vs.GRAY10, length=1)
        got10 = plane_array(core.zexpr.Expr([c10], ["X 8 % 7 % Y 8 % 7 % and 0 1023 ?"]).get_frame(0), 0)
        np.testing.assert_array_equal(got10, np.where(mask, 0, 1023).astype(np.uint16))

        c16 = core.std.BlankClip(width=w, height=19, format=vs.GRAY16, length=1)
        got16 = plane_array(core.zexpr.Expr([c16], ["X 8 % 7 % Y 8 % 7 % and 0 65535 ?"]).get_frame(0), 0)
        np.testing.assert_array_equal(got16, np.where(mask, 0, 65535).astype(np.uint16))

    # GRAY16 prefilter range mask (stays on SIMD instead of 128 KB LUT, cost < 24)
    c16_pref = make_clip(core, width=65, height=5, fmt=vs.GRAY16, length=1, seed=850)
    f16_pref = core.std.Expr([c16_pref], ["x"], format=vs.GRAYS)
    pref_expr16 = "x 4112 < 65535 x 19275 > 0 65535 x 4112 - 65535 19275 4112 - / * - ? ?"
    assert_same_clip(
        core.zexpr.Expr([c16_pref], [pref_expr16]),
        core.zexpr.Expr([f16_pref], [pref_expr16], format=vs.GRAY16),
    )

    # Native u8/u16 bitwise ops under opt=0 and opt=1, plus u8->u16/f32 LUT & SIMD parity
    for fmt, dtype, mask_c in ((vs.GRAY8, np.uint8, 0xFF), (vs.GRAY16, np.uint16, 0xFFFF)):
        b0 = make_clip(core, width=65, height=5, fmt=fmt, length=1, seed=860 + mask_c)
        b1 = make_clip(core, width=65, height=5, fmt=fmt, length=1, seed=861 + mask_c)
        a0 = plane_array(b0.get_frame(0), 0)
        a1 = plane_array(b1.get_frame(0), 0)
        for opt in (0, 1):
            np.testing.assert_array_equal(
                plane_array(core.zexpr.Expr([b0, b1], ["x y bitxor"], opt=opt).get_frame(0), 0),
                (a0 ^ a1).astype(dtype),
            )
            np.testing.assert_array_equal(
                plane_array(core.zexpr.Expr([b0, b1], ["x y bitand"], opt=opt).get_frame(0), 0),
                (a0 & a1).astype(dtype),
            )
            np.testing.assert_array_equal(
                plane_array(core.zexpr.Expr([b0, b1], ["x y bitor"], opt=opt).get_frame(0), 0),
                (a0 | a1).astype(dtype),
            )
    c8_g = make_clip(core, width=67, height=5, fmt=vs.GRAY8, length=1, seed=870)
    f8_g = core.std.Expr([c8_g], ["x"], format=vs.GRAYS)
    for dst_fmt, expr in (
        (vs.GRAYS, "x 16 < 1 x 75 > 0 1 x 16 - 1 75 16 - / * - ? ?"),
        (vs.GRAY16, "x 255 / 2.2 pow 65535 *"),
    ):
        assert_same_clip(
            core.zexpr.Expr([c8_g], [expr], format=dst_fmt),
            core.zexpr.Expr([f8_g], [expr], format=dst_fmt),
        )

    # sbr, contrasharpening, contrasharpening_dehalo, side_box_blur, hardsub, morpho grad/outer on u8
    for w in (1, 15, 16, 33, 65):
        c0 = make_clip(core, width=w, height=5, fmt=vs.GRAY8, length=1, seed=800 + w)
        c1 = make_clip(core, width=w, height=5, fmt=vs.GRAY8, length=1, seed=810 + w)
        c2 = make_clip(core, width=w, height=5, fmt=vs.GRAY8, length=1, seed=820 + w)
        c3 = make_clip(core, width=w, height=5, fmt=vs.GRAY8, length=1, seed=830 + w)
        f0 = core.std.Expr([c0], ["x"], format=vs.GRAYS)
        f1 = core.std.Expr([c1], ["x"], format=vs.GRAYS)
        f2 = core.std.Expr([c2], ["x"], format=vs.GRAYS)
        f3 = core.std.Expr([c3], ["x"], format=vs.GRAYS)

        for clips_u8, clips_f32, expr in (
            ([c0, c1, c2], [f0, f1, f2], "y z - D1! y 128 - D2! D1@ D2@ xor x x D1@ abs D2@ abs < D1@ D2@ ? - ?"),
            ([c0, c1, c2], [f0, f1, f2], "y 128 - Y! z 128 - Z! Y@ abs Z@ abs < Y@ Z@ ? x +"),
            ([c0, c1, c2], [f0, f1, f2], "x z - abs y z - abs < x y ?"),
            ([c0, c1, c2, c3], [f0, f1, f2, f3], "z a - 2.49 * 1.4 * D1! y x - D2! D1@ D2@ xor x x D1@ abs D2@ abs < D1@ D2@ ? + ?"),
            ([c0, c1], [f0, f1], "x y - 10 < 0 255 ?"),
            ([c0, c1], [f0, f1], "x y -"),
            ([c0, c1], [f0, f1], "x y +"),
            ([c0, c1], [f0, f1], "x y + 255 min"),
            ([c0, c1], [f0, f1], "x y - 0 max"),
            ([c0], [f0], "x[-1,-1] x[0,-1] max x[1,-1] max x[-1,0] max x max x[1,0] max x[-1,1] max x[0,1] max x[1,1] max x[-1,-1] x[0,-1] min x[1,-1] min x[-1,0] min x min x[1,0] min x[-1,1] min x[0,1] min x[1,1] min -"),
        ):
            assert_same_clip(
                core.zexpr.Expr(clips_u8, [expr]),
                core.zexpr.Expr(clips_f32, [expr], format=vs.GRAY8),
            )
