"""Phase 4 tests: Optimizer rewrites, LUT (incl. B4 u16 out-of-range), i16 fast path (Y1/Y2), and R3 `%` atomicity."""

from __future__ import annotations

import numpy as np
import pytest
import vapoursynth as vs

from helpers import (
    LARGE_WIDTHS,
    POINT_HEIGHTS,
    SIMD_WIDTHS,
    SPATIAL_HEIGHTS,
    assert_div255_f32,
    assert_same_clip,
    make_clip,
    make_f32_clip,
    plane_array,
)


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


def test_frameeval_plane_cache_reuse_and_eviction(core: vs.Core) -> None:
    """Bounded LRU plane cache serves FrameEval hits, handles >32-entry eviction, and isolates keys."""
    c8 = make_clip(core, width=32, height=8, fmt=vs.YUV420P8, length=48, seed=910)
    c16 = make_clip(core, width=32, height=8, fmt=vs.GRAY16, length=12, seed=911)

    # 1. Evict past default_capacity (32) while an earlier clip stays alive and is evaluated afterwards.
    early = core.zexpr.Expr([c8], ["x[-1,0] x + 2 /", "x 7 +"])
    churn = [core.zexpr.Expr([c8], [f"x {i} +", f"x {i + 1} +"]) for i in range(40)]
    for i, node in enumerate(churn):
        f = node.get_frame(i)
        src_y = plane_array(c8.get_frame(i), 0).astype(np.int32)
        np.testing.assert_array_equal(plane_array(f, 0), np.clip(src_y + i, 0, 255).astype(np.uint8))
    # Earlier filter's cached planes were evicted from LRU but must stay valid via refcount!
    early_ref = core.zexpr.Expr([c8], ["x[-1,0] x + 2 /", "x 7 +"])
    assert_same_clip(early, early_ref, frames=[0, 1])

    # 2. FrameEval reusing both JIT spatial kernels and 1D u16 LUTs across frames.
    def eval_fn(n: int) -> vs.VideoNode:
        k = (n % 4) + 1
        return core.zexpr.Expr([c8], [f"x[-1,0] x + 2 / {k} +", f"x {k} +"])

    fe = core.std.FrameEval(c8, eval_fn)
    for n in range(12):
        expected = eval_fn(n)
        assert_same_clip(fe[n : n + 1], expected[n : n + 1])

    def eval_lut16(n: int) -> vs.VideoNode:
        g = 1.8 if (n % 2 == 0) else 2.2
        return core.zexpr.Expr([c16], [f"x 65535 / {g} pow 65535 *"])

    fe16 = core.std.FrameEval(c16, eval_lut16)
    for n in range(6):
        assert_same_clip(fe16[n : n + 1], eval_lut16(n)[n : n + 1])

    # 3. Key isolation: same expression with different boundary / opt must never collide.
    clamp_out = core.zexpr.Expr([c8], ["x[-2,0]"], boundary=0)
    mirror_out = core.zexpr.Expr([c8], ["x[-2,0]"], boundary=1)
    assert not np.array_equal(plane_array(clamp_out.get_frame(0), 0), plane_array(mirror_out.get_frame(0), 0))
    opt0_out = core.zexpr.Expr([c16], ["x 0 * 16777217 + 1 + 16777216 -"], opt=0)
    opt1_out = core.zexpr.Expr([c16], ["x 0 * 16777217 + 1 + 16777216 -"], opt=1)
    assert np.all(plane_array(opt0_out.get_frame(0), 0) == 0)
    assert np.all(plane_array(opt1_out.get_frame(0), 0) == 2)


def test_periodic_x_mod_all_formats_and_widths(core: vs.Core) -> None:
    """AR1: Two-phase `X % (2 * lanes)` hoisting matches exact NumPy oracle across all formats,
    all SIMD / two-phase tail widths (1..257 + real-world residues), and YUV420 chroma planes."""
    all_widths = SIMD_WIDTHS + LARGE_WIDTHS
    gray_formats = (
        (vs.GRAY8, np.uint8, 255),
        (vs.GRAY10, np.uint16, 1023),
        (vs.GRAY16, np.uint16, 65535),
        (vs.GRAY32, np.uint32, 2147483520),
        (vs.GRAYH, np.float16, None),
        (vs.GRAYS, np.float32, None),
    )

    def quantize_f32(val: np.ndarray, dtype, peak: int | None) -> np.ndarray:
        if peak is None:
            return val.astype(dtype)
        clamped = np.clip(val, np.float32(0.0), np.float32(peak))
        return np.rint(clamped).astype(dtype)

    for fmt, dtype, peak in gray_formats:
        for h in POINT_HEIGHTS:
            for w in all_widths:
                xs = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
                src = make_clip(core, width=w, height=h, fmt=fmt, length=1, seed=1000 + w * 7 + h)
                src_y = make_clip(core, width=w, height=h, fmt=fmt, length=1, seed=2000 + w * 7 + h)
                raw_x = plane_array(src.get_frame(0), 0)
                raw_y = plane_array(src_y.get_frame(0), 0)
                ax = (raw_x.view(np.int32) if fmt == vs.GRAY32 else raw_x).astype(np.float32)
                ay = (raw_y.view(np.int32) if fmt == vs.GRAY32 else raw_y).astype(np.float32)

                cases = (
                    ("X 4 %", [src], xs % np.float32(4.0)),
                    ("X 8 %", [src], xs % np.float32(8.0)),
                    ("X 16 % 4 *", [src], (xs % np.float32(16.0)) * np.float32(4.0)),
                    ("X 32 % 2 *", [src], (xs % np.float32(32.0)) * np.float32(2.0)),
                    ("X 8 % x +", [src], (xs % np.float32(8.0)) + ax),
                    ("X 16 % x +", [src], (xs % np.float32(16.0)) + ax),
                    ("X 32 % x +", [src], (xs % np.float32(32.0)) + ax),
                    ("X 16 % x * y +", [src, src_y], (xs % np.float32(16.0)) * ax + ay),
                )
                for expr, clips, expected_f32 in cases:
                    got = plane_array(core.zexpr.Expr(clips, [expr]).get_frame(0), 0)
                    exp = quantize_f32(expected_f32, dtype, peak)
                    if expr == "X 16 % x * y +":
                        # Matches exact unfused (L0-L2) or exact FMA (L3/L4/NEON, computed in f64->f32).
                        exp_fma_f32 = (
                            (xs.astype(np.float64) % 16.0) * ax.astype(np.float64) + ay.astype(np.float64)
                        ).astype(np.float32)
                        exp_fma = quantize_f32(exp_fma_f32, dtype, peak)
                        assert np.all((got == exp) | (got == exp_fma)), f"FMA/unfused mismatch on {fmt} w={w} h={h}"
                    else:
                        np.testing.assert_array_equal(got, exp)

    # Float & integer `deblock_qed` mask across spatial heights (including h=2..17) and awkward widths.
    for fmt, dtype, hi_str, hi_val in (
        (vs.GRAY8, np.uint8, "255", 255),
        (vs.GRAY16, np.uint16, "65535", 65535),
        (vs.GRAYH, np.float16, "1", np.float16(1.0)),
        (vs.GRAYS, np.float32, "1", np.float32(1.0)),
    ):
        for h in SPATIAL_HEIGHTS:
            for w in (1, 5, 6, 7, 8, 9, 13, 15, 16, 17, 25, 31, 33, 63, 65, 1366):
                yy, xx = np.mgrid[0:h, 0:w]
                mask8 = ((xx % 8 % 7) > 0) & ((yy % 8 % 7) > 0)
                mask16 = ((xx % 16 % 15) > 0) & ((yy % 8 % 7) > 0)
                c = core.std.BlankClip(width=w, height=h, format=fmt, length=1)
                got8 = plane_array(core.zexpr.Expr([c], [f"X 8 % 7 % Y 8 % 7 % and 0 {hi_str} ?"]).get_frame(0), 0)
                np.testing.assert_array_equal(got8, np.where(mask8, 0, hi_val).astype(dtype))
                got16 = plane_array(core.zexpr.Expr([c], [f"X 16 % 15 % Y 8 % 7 % and 0 {hi_str} ?"]).get_frame(0), 0)
                np.testing.assert_array_equal(got16, np.where(mask16, 0, hi_val).astype(dtype))

    # YUV420 covers half-width chroma planes (e.g. 1366 -> 683, 26 -> 13).
    for w in (10, 14, 26, 30, 62, 1366):
        cyuv = core.std.BlankClip(width=w, height=6, format=vs.YUV420PS, length=1)
        out = core.zexpr.Expr([cyuv], ["X 8 % 7 % Y 8 % 7 % and 0 1 ?"])
        f = out.get_frame(0)
        for p in range(3):
            got = plane_array(f, p)
            ph, pw = got.shape
            yy, xx = np.mgrid[0:ph, 0:pw]
            exp = np.where(((xx % 8 % 7) > 0) & ((yy % 8 % 7) > 0), 0.0, 1.0).astype(np.float32)
            np.testing.assert_array_equal(got, exp)


def test_lut_cache_key_width_height_invariant(core: vs.Core) -> None:
    """AR3/N6: Dimension-free LUT caching shares Y/U/V tables across dimensions and never
    conflates expressions that depend on `width`/`height`."""
    # Positive companion (N6): YUV420P16 gamma builds a dimension-free 1D u16 LUT on Y (32x16)
    # and reuses that exact cached table on subsampled U/V (16x8) and separate odd-width clips.
    gamma_expr = "x 65535 / 2.2 pow 65535 *"
    cyuv_rand = make_clip(core, width=32, height=16, fmt=vs.YUV420P16, length=2, seed=942)
    got_lut = core.zexpr.Expr([cyuv_rand], [gamma_expr])
    # Reference via f32 input (exact for u16 samples, never eligible for lut1d_u16 -> runs SIMD/scalar JIT).
    cyuv_f32 = core.zexpr.Expr([cyuv_rand], ["x"], format=vs.YUV420PS)
    ref_jit = core.zexpr.Expr([cyuv_f32], [gamma_expr], format=vs.YUV420P16)
    assert_same_clip(got_lut, ref_jit)
    # Confirm U and V received distinct plane data (not aliased to Y or to each other).
    f0 = got_lut.get_frame(0)
    assert not np.array_equal(plane_array(f0, 1), plane_array(f0, 2))
    # Separate odd-width GRAY16 clip hitting the same cached dimension-free table.
    cg_odd = make_clip(core, width=47, height=13, fmt=vs.GRAY16, length=1, seed=943)
    cg_odd_f32 = core.zexpr.Expr([cg_odd], ["x"], format=vs.GRAYS)
    assert_same_clip(
        core.zexpr.Expr([cg_odd], [gamma_expr]),
        core.zexpr.Expr([cg_odd_f32], [gamma_expr], format=vs.GRAY16),
    )

    # Negative guard (AR3): expressions referencing `width`/`height` must never share a dimension-free entry.
    expr_w = "x 65535 / 2.2 pow width * 0.5 +"
    expr_h = "x 65535 / 2.2 pow height * 0.5 +"

    # Same YUV420P16 clip: Y plane (32x16) and U/V planes (16x8) must see their own plane dimensions.
    cyuv = core.std.BlankClip(width=32, height=16, format=vs.YUV420P16, color=[40000, 40000, 40000], length=1)
    fy_w = core.zexpr.Expr([cyuv], [expr_w]).get_frame(0)
    fy_h = core.zexpr.Expr([cyuv], [expr_h]).get_frame(0)
    base_val = np.float32(40000.0 / 65535.0) ** np.float32(2.2)
    for p, (pw, ph) in enumerate(((32, 16), (16, 8), (16, 8))):
        exp_w = int(np.rint(base_val * np.float32(pw) + np.float32(0.5)))
        exp_h = int(np.rint(base_val * np.float32(ph) + np.float32(0.5)))
        assert np.all(plane_array(fy_w, p) == exp_w), f"width mismatch on plane {p}"
        assert np.all(plane_array(fy_h, p) == exp_h), f"height mismatch on plane {p}"

    # Separate GRAY16 Creates with different widths/heights must never share a dimension-free LUT entry.
    for w, h in ((32, 8), (64, 16), (128, 32)):
        cg = core.std.BlankClip(width=w, height=h, format=vs.GRAY16, color=[40000], length=1)
        got_w = plane_array(core.zexpr.Expr([cg], [expr_w]).get_frame(0), 0)
        got_h = plane_array(core.zexpr.Expr([cg], [expr_h]).get_frame(0), 0)
        assert np.all(got_w == int(np.rint(base_val * np.float32(w) + np.float32(0.5))))
        assert np.all(got_h == int(np.rint(base_val * np.float32(h) + np.float32(0.5))))


def test_simd_u8_lut_runtime_dims_yuv420_odd_chroma_and_plane_max(core: vs.Core) -> None:
    """Phase 3: Runtime-dimension SIMD u8 LUTs match f32 reference across odd-chroma YUV420P8
    widths, variable-resolution clips, and full vs limited `plane_max` range variants."""
    gamma_u8 = "x 255 / 2.2 pow 255 *"

    # Odd chroma widths (34->17, 62->31, 66->33, 98->49, 130->65) exercising 16/32/64B boundaries.
    for w, h, seed in ((34, 8, 950), (62, 6, 951), (66, 10, 952), (98, 4, 953), (130, 6, 954)):
        cyuv = make_clip(core, width=w, height=h, fmt=vs.YUV420P8, length=2, seed=seed)
        got = core.zexpr.Expr([cyuv], [gamma_u8])
        cyuv_f32 = core.zexpr.Expr([cyuv], ["x"], format=vs.YUV420PS)
        ref = core.zexpr.Expr([cyuv_f32], [gamma_u8], format=vs.YUV420P8)
        assert_same_clip(got, ref)
        f0 = got.get_frame(0)
        assert not np.array_equal(plane_array(f0, 1), plane_array(f0, 2))

    # Variable-resolution clip switching across odd/boundary widths reusing the 0x0 SIMD u8 LUT.
    parts = [
        make_clip(core, width=34, height=8, fmt=vs.YUV420P8, length=1, seed=960),
        make_clip(core, width=64, height=16, fmt=vs.YUV420P8, length=1, seed=961),
        make_clip(core, width=98, height=12, fmt=vs.YUV420P8, length=1, seed=962),
        make_clip(core, width=16, height=4, fmt=vs.YUV420P8, length=1, seed=963),
    ]
    vclip = core.std.Splice(parts, mismatch=True)
    vgot = core.zexpr.Expr([vclip], [gamma_u8])
    for idx, part in enumerate(parts):
        part_f32 = core.zexpr.Expr([part], ["x"], format=vs.YUV420PS)
        part_ref = core.zexpr.Expr([part_f32], [gamma_u8], format=vs.YUV420P8)
        gf = vgot.get_frame(idx)
        rf = part_ref.get_frame(0)
        for p in range(3):
            np.testing.assert_array_equal(plane_array(gf, p), plane_array(rf, p))

    # `plane_max` on YUV420P8: full-range (_Range=1) -> [255, 255, 255], limited (_Range=0) -> [235, 240, 240].
    pmax_expr = "x 255 / 2.2 pow plane_max *"
    c_full = core.std.BlankClip(width=34, height=8, format=vs.YUV420P8, color=[255, 255, 255], length=1)
    c_full = core.std.SetFrameProp(c_full, prop="_Range", intval=1)
    c_lim = core.std.BlankClip(width=34, height=8, format=vs.YUV420P8, color=[255, 255, 255], length=1)
    c_lim = core.std.SetFrameProp(c_lim, prop="_Range", intval=0)

    f_full = core.zexpr.Expr([c_full], [pmax_expr]).get_frame(0)
    f_lim = core.zexpr.Expr([c_lim], [pmax_expr]).get_frame(0)
    for p, exp_val in enumerate((255, 255, 255)):
        assert np.all(plane_array(f_full, p) == exp_val), f"full plane {p} expected {exp_val}"
    for p, exp_val in enumerate((235, 240, 240)):
        assert np.all(plane_array(f_lim, p) == exp_val), f"limited plane {p} expected {exp_val}"
