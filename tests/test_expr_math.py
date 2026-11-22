"""Phase 2b transcendental tests: D5 lowering, accuracy vs f64 libm, and special values."""

from __future__ import annotations

import numpy as np
import pytest
import vapoursynth as vs

from helpers import assert_same_clip, make_clip, make_f32_clip, plane_array


def test_transcendental_accuracy_vs_f64_libm(core: vs.Core) -> None:
    # exp over [-87, 88]
    xs_exp = np.linspace(-87.0, 88.0, 257, dtype=np.float32).reshape(1, -1)
    got_exp = plane_array(core.zexpr.Expr([make_f32_clip(core, xs_exp)], ["x exp"]).get_frame(0), 0)[0]
    ref_exp = np.exp(xs_exp[0].astype(np.float64))
    assert np.max(np.abs(got_exp.astype(np.float64) - ref_exp) / ref_exp) <= 2e-6

    # log over [1e-37, 1e37]
    xs_log = (10.0 ** np.linspace(-37.0, 37.0, 257)).astype(np.float32).reshape(1, -1)
    got_log = plane_array(core.zexpr.Expr([make_f32_clip(core, xs_log)], ["x log"]).get_frame(0), 0)[0]
    ref_log = np.log(xs_log[0].astype(np.float64))
    assert np.max(np.abs(got_log.astype(np.float64) - ref_log) / np.maximum(1.0, np.abs(ref_log))) <= 2e-6

    # sin & cos over |x| <= 1e5
    xs_trig = np.concatenate([
        np.linspace(-20.0, 20.0, 129, dtype=np.float32),
        np.array([-1e5, -12345.67, -1000.5, -0.0, 0.0, 1000.5, 12345.67, 1e5], dtype=np.float32),
    ]).reshape(1, -1)
    c_trig = make_f32_clip(core, xs_trig)
    got_sin = plane_array(core.zexpr.Expr([c_trig], ["x sin"]).get_frame(0), 0)[0]
    got_cos = plane_array(core.zexpr.Expr([c_trig], ["x cos"]).get_frame(0), 0)[0]
    assert np.max(np.abs(got_sin.astype(np.float64) - np.sin(xs_trig[0].astype(np.float64)))) <= 2e-6
    assert np.max(np.abs(got_cos.astype(np.float64) - np.cos(xs_trig[0].astype(np.float64)))) <= 2e-6
    # sin(-0.0) == -0.0 (preserved at every tier including SSE2 level 1, Y4)
    neg0_idx = 129 + 3
    assert got_sin[neg0_idx].view(np.uint32) == 0x80000000

    # tan away from poles
    xs_tan = np.linspace(-1.2, 1.2, 129, dtype=np.float32).reshape(1, -1)
    got_tan = plane_array(core.zexpr.Expr([make_f32_clip(core, xs_tan)], ["x tan"]).get_frame(0), 0)[0]
    ref_tan = np.tan(xs_tan[0].astype(np.float64))
    assert np.max(np.abs(got_tan.astype(np.float64) - ref_tan) / np.maximum(1.0, np.abs(ref_tan))) <= 1e-5

    # atan2 across all 4 quadrants and (0, 0)
    grid = np.array([-3.0, -1.0, -0.25, -0.0, 0.0, 0.25, 1.0, 3.0], dtype=np.float32)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    got_atan2 = plane_array(
        core.zexpr.Expr([make_f32_clip(core, yy), make_f32_clip(core, xx)], ["x y atan2"]).get_frame(0),
        0,
    )
    ref_atan2 = np.arctan2(yy.astype(np.float64), xx.astype(np.float64))
    ref_atan2[(yy == 0.0) & (xx == 0.0)] = 0.0
    assert np.max(np.abs(got_atan2.astype(np.float64) - ref_atan2)) <= 2e-6

    # atan over wide range including Cephes breakpoints, +-0, +-1, +-inf, NaN
    xs_atan = np.concatenate([
        np.linspace(-10.0, 10.0, 257, dtype=np.float32),
        np.array([-1e6, -2.4142136, -1.0, -0.41421356, -0.0, 0.0, 0.41421356, 1.0, 2.4142136, 1e6], dtype=np.float32),
    ]).reshape(1, -1)
    got_atan = plane_array(core.zexpr.Expr([make_f32_clip(core, xs_atan)], ["x atan"]).get_frame(0), 0)[0]
    ref_atan = np.arctan(xs_atan[0].astype(np.float64))
    assert np.max(np.abs(got_atan.astype(np.float64) - ref_atan)) <= 2e-6
    assert got_atan[257 + 4].view(np.uint32) == 0x80000000  # atan(-0.0) == -0.0
    atan_edge = plane_array(
        core.zexpr.Expr([make_f32_clip(core, np.array([[np.inf, -np.inf, np.nan]], dtype=np.float32))], ["x atan"]).get_frame(0),
        0,
    )[0]
    assert abs(float(atan_edge[0]) - np.pi * 0.5) <= 2e-6
    assert abs(float(atan_edge[1]) + np.pi * 0.5) <= 2e-6
    assert np.isnan(atan_edge[2])

    # asin & acos over [-1, 1], +-0, +-0.5, +-1, and |x| > 1 -> NaN
    xs_arc = np.concatenate([
        np.linspace(-1.0, 1.0, 257, dtype=np.float32),
        np.array([-1.0, -0.5, -0.0, 0.0, 0.5, 1.0], dtype=np.float32),
    ]).reshape(1, -1)
    c_arc = make_f32_clip(core, xs_arc)
    got_asin = plane_array(core.zexpr.Expr([c_arc], ["x asin"]).get_frame(0), 0)[0]
    got_acos = plane_array(core.zexpr.Expr([c_arc], ["x acos"]).get_frame(0), 0)[0]
    assert np.max(np.abs(got_asin.astype(np.float64) - np.arcsin(xs_arc[0].astype(np.float64)))) <= 2e-6
    assert np.max(np.abs(got_acos.astype(np.float64) - np.arccos(xs_arc[0].astype(np.float64)))) <= 2e-6
    assert got_asin[257 + 2].view(np.uint32) == 0x80000000  # asin(-0.0) == -0.0
    assert got_acos[257 + 5] == np.float32(0.0)             # acos(1.0) == 0.0
    assert got_acos[257 + 0] == np.float32(np.pi)           # acos(-1.0) == pi

    bad_arc = make_f32_clip(core, np.array([[-1.5, 1.5, np.inf, -np.inf, np.nan]], dtype=np.float32))
    assert np.all(np.isnan(plane_array(core.zexpr.Expr([bad_arc], ["x asin"]).get_frame(0), 0)))
    assert np.all(np.isnan(plane_array(core.zexpr.Expr([bad_arc], ["x acos"]).get_frame(0), 0)))

    # Constant-folded atan/asin/acos
    dummy = make_f32_clip(core, np.array([[0.0]], dtype=np.float32))
    assert abs(float(plane_array(core.zexpr.Expr([dummy], ["1 atan"]).get_frame(0), 0)[0, 0]) - np.pi * 0.25) <= 2e-6
    assert abs(float(plane_array(core.zexpr.Expr([dummy], ["0.5 asin"]).get_frame(0), 0)[0, 0]) - np.pi / 6.0) <= 2e-6
    assert abs(float(plane_array(core.zexpr.Expr([dummy], ["0.5 acos"]).get_frame(0), 0)[0, 0]) - np.pi / 3.0) <= 2e-6


def test_d5_pow_and_special_values(core: vs.Core) -> None:
    vals = np.array([[-2.0, -0.25, 0.0, 0.5, 2.0, np.nan, np.inf, -np.inf]], dtype=np.float32)
    clip = make_f32_clip(core, vals)

    # All-constant powf folding (T2: powf(1.1f, 5.0f) == 1.6105102)
    assert plane_array(core.zexpr.Expr([clip], ["1.1 5 pow"]).get_frame(0), 0)[0, 0] == np.float32(1.6105102)
    assert plane_array(core.zexpr.Expr([clip], ["0 -1 pow"]).get_frame(0), 0)[0, 0] == np.inf
    assert plane_array(core.zexpr.Expr([clip], ["0 0 pow"]).get_frame(0), 0)[0, 0] == 1.0

    # T3: chained power simplifies before expansion (no overflow)
    chained = plane_array(core.zexpr.Expr([clip], ["x 1e20 * 2 pow 0.5 pow"]).get_frame(0), 0)[0]
    assert chained[1] == np.float32(2.5e19)

    # Q1: runtime 0**e == 0 and 1**e == 1
    assert np.all(plane_array(core.zexpr.Expr([clip], ["0 x -1 * pow"]).get_frame(0), 0) == 0.0)
    assert np.all(plane_array(core.zexpr.Expr([clip], ["1 x 0 0 / + pow"]).get_frame(0), 0) == 1.0)

    # Negative-base integer powers
    assert plane_array(core.zexpr.Expr([clip], ["x 3 **"]).get_frame(0), 0)[0, 0] == -8.0
    assert plane_array(core.zexpr.Expr([clip], ["x 4 **"]).get_frame(0), 0)[0, 0] == 16.0

    # Constant vs runtime log & exp special values (D13, U3, W1)
    assert plane_array(core.zexpr.Expr([clip], ["0 log"]).get_frame(0), 0)[0, 0] == -np.inf
    rt_log = plane_array(core.zexpr.Expr([clip], ["x log"]).get_frame(0), 0)[0]
    assert abs(float(rt_log[2]) - (-87.33654)) < 1e-3  # 0.0 -> FLT_MIN clamp
    assert abs(float(rt_log[5]) - (-87.33654)) < 1e-3  # NaN -> FLT_MIN clamp

    rt_exp = plane_array(core.zexpr.Expr([clip], ["x exp"]).get_frame(0), 0)[0]
    assert rt_exp[5] > 2.4e38  # NaN -> ~2.406e38
    assert rt_exp[6] > 2.4e38  # +inf -> ~2.406e38
    assert rt_exp[7] == 0.0    # -inf -> 0.0


def test_exp2_log2_rsqrt_and_hypot(core: vs.Core) -> None:
    """exp2, log2, hardware-refined rsqrt, and hypot accuracy, edge values, and constant folding."""
    # exp2: exact at integers, <= 2e-6 rel error across [-20, 20], underflow -> 0.0
    xs_e2 = np.concatenate([
        np.array([-200.0, -10.0, -1.0, 0.0, 1.0, 3.0, 8.0], dtype=np.float32),
        np.linspace(-20.0, 20.0, 250, dtype=np.float32),
    ]).reshape(1, -1)
    got_e2 = plane_array(core.zexpr.Expr([make_f32_clip(core, xs_e2)], ["x exp2"]).get_frame(0), 0)[0]
    assert got_e2[0] == 0.0
    for i, k in enumerate((-10.0, -1.0, 0.0, 1.0, 3.0, 8.0), start=1):
        assert got_e2[i] == np.float32(2.0 ** k)
    ref_e2 = np.exp2(xs_e2[0, 7:].astype(np.float64))
    assert np.max(np.abs(got_e2[7:].astype(np.float64) - ref_e2) / ref_e2) <= 2e-6

    # log2: exact at powers of 2, <= 2e-6 rel error on positive range, 0 -> FLT_MIN clamp (-126)
    xs_l2 = np.concatenate([
        np.array([0.0, -5.0, 0.25, 0.5, 1.0, 2.0, 8.0, 1024.0], dtype=np.float32),
        np.geomspace(1e-6, 1e6, 250, dtype=np.float32),
    ]).reshape(1, -1)
    got_l2 = plane_array(core.zexpr.Expr([make_f32_clip(core, xs_l2)], ["x log2"]).get_frame(0), 0)[0]
    assert got_l2[0] == np.float32(-126.0)
    assert got_l2[1] == np.float32(-126.0)
    for i, p2 in enumerate((0.25, 0.5, 1.0, 2.0, 8.0, 1024.0), start=2):
        assert got_l2[i] == np.float32(np.log2(p2))
    ref_l2 = np.log2(xs_l2[0, 8:].astype(np.float64))
    assert np.max(np.abs(got_l2[8:].astype(np.float64) - ref_l2) / np.maximum(1.0, np.abs(ref_l2))) <= 2e-6

    # exp2(log2(x)) and log2(exp2(x)) inverse cancellations
    pos = make_f32_clip(core, np.array([[0.25, 1.0, 3.5, 100.0]], dtype=np.float32))
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([pos], ["x log2 exp2"]).get_frame(0), 0),
        np.array([[0.25, 1.0, 3.5, 100.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([pos], ["x exp2 log2"]).get_frame(0), 0),
        np.array([[0.25, 1.0, 3.5, 100.0]], dtype=np.float32),
    )

    # rsqrt: <= 5e-7 rel error across 20 orders of magnitude, x <= 0 -> FLT_MIN, +inf -> FLT_MAX (0 on u8), const-fold exact
    xs_rs = np.concatenate([
        np.array([-1.0, 0.0, np.inf, 0.25, 1.0, 4.0, 16.0], dtype=np.float32),
        np.geomspace(1e-10, 1e10, 250, dtype=np.float32),
    ]).reshape(1, -1)
    got_rs = plane_array(core.zexpr.Expr([make_f32_clip(core, xs_rs)], ["x rsqrt"]).get_frame(0), 0)[0]
    assert np.isfinite(got_rs[0]) and got_rs[0] > 1e18
    assert np.isfinite(got_rs[1]) and got_rs[1] > 1e18
    assert np.isfinite(got_rs[2]) and 0.0 < got_rs[2] < 1e-19
    inf_clip = make_f32_clip(core, np.array([[np.inf, 0.0]], dtype=np.float32))
    got_rs_u8 = plane_array(core.zexpr.Expr([inf_clip], ["x rsqrt 100 *"], format=vs.GRAY8).get_frame(0), 0)[0]
    assert got_rs_u8[0] == 0
    assert got_rs_u8[1] == 255
    ref_rs = 1.0 / np.sqrt(xs_rs[0, 3:].astype(np.float64))
    assert np.max(np.abs(got_rs[3:].astype(np.float64) - ref_rs) / ref_rs) <= 5e-7
    assert plane_array(core.zexpr.Expr([pos], ["4 rsqrt"]).get_frame(0), 0)[0, 0] == np.float32(0.5)
    assert plane_array(core.zexpr.Expr([pos], ["3 exp2"]).get_frame(0), 0)[0, 0] == np.float32(8.0)
    assert plane_array(core.zexpr.Expr([pos], ["8 log2"]).get_frame(0), 0)[0, 0] == np.float32(3.0)
    # E3: constant rsqrt follows the same [FLT_MIN, FLT_MAX] clamp as runtime rsqrt
    for expr, lo, hi in (
        ("0 rsqrt", 9.0e18, 9.5e18),
        ("-1 rsqrt", 9.0e18, 9.5e18),
        ("1 0 / rsqrt", 5.0e-20, 6.0e-20),
    ):
        v = float(plane_array(core.zexpr.Expr([pos], [expr]).get_frame(0), 0)[0, 0])
        assert np.isfinite(v) and lo <= v <= hi

    # E1: constant-folded transcendentals bind to host libm (matching std.Expr across Debug, native, and x86_64_v2)
    c1 = make_f32_clip(core, np.array([[0.0]], dtype=np.float32))
    for expr in (
        "-1.579137 exp",
        "-1.2095927 exp",
        "0.40886822 log",
        "2.6477263 log",
        "0.2698961 sin",
        "0.49449918 sin",
        "2.30829 cos",
        "0.84011394 cos",
    ):
        got_z = plane_array(core.zexpr.Expr([c1], [expr]).get_frame(0), 0)[0, 0]
        ref_s = plane_array(core.std.Expr([c1], [expr]).get_frame(0), 0)[0, 0]
        assert got_z.view(np.uint32) == ref_s.view(np.uint32), f"{expr}: zexpr={got_z!r} std={ref_s!r}"

    # hypot: matches sqrt(a*a + b*b), incl. opt=1
    a_h = np.array([[0.0, 3.0, -5.0, 1.5, 100.0]], dtype=np.float32)
    b_h = np.array([[0.0, 4.0, 12.0, -2.0, 240.0]], dtype=np.float32)
    got_h = plane_array(core.zexpr.Expr([make_f32_clip(core, a_h), make_f32_clip(core, b_h)], ["x y hypot"]).get_frame(0), 0)
    np.testing.assert_allclose(got_h, np.hypot(a_h, b_h), rtol=1e-6, atol=1e-6)
    c16_a = make_clip(core, width=33, height=3, fmt=vs.GRAY16, length=1, seed=91)
    c16_b = make_clip(core, width=33, height=3, fmt=vs.GRAY16, length=1, seed=92)
    assert_same_clip(
        core.zexpr.Expr([c16_a, c16_b], ["x y hypot"], opt=1),
        core.zexpr.Expr([c16_a, c16_b], ["x dup * y dup * + sqrt"]),
    )
