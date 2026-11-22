"""Phase 5 tests: Formal D1-D15 operator and literal contract matrix on constant and runtime inputs."""

from __future__ import annotations

import math

import numpy as np
import pytest
import vapoursynth as vs

from helpers import assert_same_clip, make_clip, make_f32_clip, plane_array


def _eval_f32_single(core: vs.Core, expr: str, x_vals: list[float], y_vals: list[float] | None = None) -> np.ndarray:
    cx = make_f32_clip(core, np.array([x_vals], dtype=np.float32))
    clips = [cx]
    if y_vals is not None:
        clips.append(make_f32_clip(core, np.array([y_vals], dtype=np.float32)))
    out = core.zexpr.Expr(clips, [expr])
    return plane_array(out.get_frame(0), 0)[0]


def test_d1_literals_constant_and_runtime(core: vs.Core) -> None:
    """D1/Y15: Leading-zero decimal is decimal (023 == 23), hex wraps u32->i32, hex floats, -0 sign."""
    vals = _eval_f32_single(core, "023", [0.0])
    assert vals[0] == 23.0

    vals = _eval_f32_single(core, "0xFFFFFFFF", [0.0])
    assert vals[0] == -1.0

    vals = _eval_f32_single(core, "0x80000000", [0.0])
    assert vals[0] == float(-2147483648)

    vals = _eval_f32_single(core, "0x1.8p2", [0.0])
    assert vals[0] == 6.0

    vals = _eval_f32_single(core, "-0.0", [0.0])
    assert vals[0] == 0.0 and math.copysign(1.0, float(vals[0])) == -1.0

    # Runtime expression combined with literals
    got = _eval_f32_single(core, "x 023 + 0x1.8p2 +", [1.0, 2.0])
    np.testing.assert_array_equal(got, np.array([30.0, 31.0], dtype=np.float32))


def test_d9_rounding_modes_and_negative_zero(core: vs.Core) -> None:
    """D9/X3/Y4: round is half-to-even (banker's rounding), trunc toward zero, floor toward -inf; -0.0 preserved."""
    xs = [-2.5, -1.5, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 1.5, 2.5]

    r = _eval_f32_single(core, "x round", xs)
    np.testing.assert_array_equal(r, np.array([-2.0, -2.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 2.0, 2.0], dtype=np.float32))
    # X3/Y4: sign of -0.0 preserved on round(-0.5), round(-0.25), and round(-0.0)
    assert math.copysign(1.0, float(r[2])) == -1.0
    assert math.copysign(1.0, float(r[3])) == -1.0
    assert math.copysign(1.0, float(r[4])) == -1.0
    assert math.copysign(1.0, float(r[5])) == 1.0

    t = _eval_f32_single(core, "x trunc", xs)
    np.testing.assert_array_equal(t, np.array([-2.0, -1.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32))

    f = _eval_f32_single(core, "x floor", xs)
    np.testing.assert_array_equal(f, np.array([-3.0, -2.0, -1.0, -1.0, -0.0, 0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32))

    ce = _eval_f32_single(core, "x ceil", xs)
    np.testing.assert_array_equal(ce, np.array([-2.0, -1.0, -0.0, -0.0, -0.0, 0.0, 1.0, 1.0, 2.0, 3.0], dtype=np.float32))
    assert math.copysign(1.0, float(ce[2])) == -1.0
    assert math.copysign(1.0, float(ce[3])) == -1.0
    assert math.copysign(1.0, float(ce[4])) == -1.0
    assert math.copysign(1.0, float(ce[5])) == 1.0

    # Constant-folded versions must match runtime exactly!
    for op, expected in [
        ("-2.5 round", -2.0),
        ("-1.5 round", -2.0),
        ("1.5 round", 2.0),
        ("2.5 round", 2.0),
        ("-0.5 round", 0.0),
        ("0.5 round", 0.0),
        ("-2.5 trunc", -2.0),
        ("-2.5 floor", -3.0),
        ("-2.5 ceil", -2.0),
        ("0.25 ceil", 1.0),
    ]:
        assert float(_eval_f32_single(core, op, [0.0])[0]) == expected


def test_d3_modulo_unfused_and_sign(core: vs.Core) -> None:
    """D3/R3: `%` uses `a - trunc(a / b) * b` (unfused), sign follows dividend, constant == runtime."""
    xs = [7.5, -7.5, 7.5, -7.5, 1e10, 0.0]
    ys = [3.0, 3.0, -3.0, -3.0, 3.0, 3.0]
    got = _eval_f32_single(core, "x y %", xs, ys)
    np.testing.assert_array_equal(got, np.array([1.5, -1.5, 1.5, -1.5, 0.0, 0.0], dtype=np.float32))
    assert float(_eval_f32_single(core, "1e10 3 %", [0.0])[0]) == 0.0
    assert float(_eval_f32_single(core, "-7.5 3 %", [0.0])[0]) == -1.5


def test_d4_int_output_clamping_and_nan_to_peak(core: vs.Core) -> None:
    """D4: Integer outputs round-half-even and clamp [0, peak]; NaN -> peak, +inf -> peak, -inf -> 0."""
    xs = [-10.0, -0.6, -0.4, 0.0, 0.5, 1.5, 2.5, 254.5, 255.0, 300.0, float("nan"), float("inf"), float("-inf")]
    cx = make_f32_clip(core, np.array([xs], dtype=np.float32))
    out8 = core.zexpr.Expr([cx], ["x"], format=vs.GRAY8)
    got8 = plane_array(out8.get_frame(0), 0)[0]
    # Note: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, 254.5 -> 254 (round-half-to-even), NaN -> 255, +inf -> 255, -inf -> 0
    np.testing.assert_array_equal(
        got8,
        np.array([0, 0, 0, 0, 0, 2, 2, 254, 255, 255, 255, 255, 0], dtype=np.uint8),
    )

    out10 = core.zexpr.Expr([cx], ["x"], format=vs.GRAY10)
    got10 = plane_array(out10.get_frame(0), 0)[0]
    assert got10[10] == 1023  # NaN -> 10-bit peak
    assert got10[11] == 1023  # +inf -> peak
    assert got10[12] == 0     # -inf -> 0

    out16 = core.zexpr.Expr([cx], ["x"], format=vs.GRAY16)
    got16 = plane_array(out16.get_frame(0), 0)[0]
    assert got16[10] == 65535  # NaN -> 16-bit peak
    assert got16[11] == 65535  # +inf -> peak
    assert got16[12] == 0      # -inf -> 0


def test_d5_pow_mandatory_lowering_and_neg_base(core: vs.Core) -> None:
    """D5: Rational/integer powers (incl. negative bases & chained powers) work identically at const & runtime."""
    xs = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    # Even integer power on negative base
    p2 = _eval_f32_single(core, "x 2 pow", xs)
    np.testing.assert_array_equal(p2, np.array([9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0], dtype=np.float32))

    # Odd integer power on negative base
    p3 = _eval_f32_single(core, "x 3 pow", xs)
    np.testing.assert_array_equal(p3, np.array([-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0], dtype=np.float32))

    # Chained powers: (x^2)^3 == x^6
    p6 = _eval_f32_single(core, "x 2 pow 3 pow", xs)
    np.testing.assert_allclose(p6, np.array([729.0, 64.0, 1.0, 0.0, 1.0, 64.0, 729.0], dtype=np.float32), rtol=0, atol=0)

    # Half-integer power `x 0.5 pow` == `sqrt(x)`
    pos = [0.0, 1.0, 4.0, 9.0, 16.0]
    p_half = _eval_f32_single(core, "x 0.5 pow", pos)
    np.testing.assert_array_equal(p_half, np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    # 0 ** 0 == 1
    p0 = _eval_f32_single(core, "x 0 pow", [-5.0, 0.0, 5.0])
    np.testing.assert_array_equal(p0, np.array([1.0, 1.0, 1.0], dtype=np.float32))

    # V8: odd N = 65 expands `x * (x ** 64)` and `32.5` expands `sqrt(x ** 65)`, matching std.Expr.
    c_q = make_f32_clip(core, np.array([[0.25, 0.5, 0.75, 1.0]], dtype=np.float32))
    assert_same_clip(core.zexpr.Expr([c_q], ["x 4.4 * 65 pow"]), core.std.Expr([c_q], ["x 4.4 * 65 pow"]))
    assert_same_clip(core.zexpr.Expr([c_q], ["x 4.4 * 32.5 pow"]), core.std.Expr([c_q], ["x 4.4 * 32.5 pow"]))

    # V9: `tan` near +-pi/2 poles stays within 1e-5 relative accuracy.
    pole_xs = [-1.569, -1.56, 1.56, 1.569]
    got_tan = _eval_f32_single(core, "x tan", pole_xs)
    ref_tan = np.tan(np.array(pole_xs, dtype=np.float64))
    assert np.all(np.abs(got_tan.astype(np.float64) - ref_tan) <= 1e-5 * np.maximum(1.0, np.abs(ref_tan)))

    # V11: `swap0` is a valid no-op on non-empty stack.
    np.testing.assert_array_equal(_eval_f32_single(core, "x 1 swap0 -", [0.25, 2.0]), np.array([-0.75, 1.0], dtype=np.float32))

    # V1: extreme negative relative offset does not overflow or crash.
    c8 = core.std.BlankClip(width=32, height=4, format=vs.GRAY8, color=[100])
    assert np.all(plane_array(core.zexpr.Expr([c8], ["x[-2147483648,0] x max"]).get_frame(0), 0) == 100)


def test_d10_d12_d15_truthiness_logic_and_bitwise(core: vs.Core) -> None:
    """D10/D12/D15: Truthiness is `> 0.0`; ordered NaN handling; bitwise rounds half-to-even to signed i32."""
    xs = [-2.0, -0.0, 0.0, 0.5, 2.0]
    ys = [-1.0, 0.0, 1.0, 0.0, 3.0]

    np.testing.assert_array_equal(_eval_f32_single(core, "x not", xs), np.array([1, 1, 1, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(_eval_f32_single(core, "x y and", xs, ys), np.array([0, 0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_array_equal(_eval_f32_single(core, "x y or", xs, ys), np.array([0, 0, 1, 1, 1], dtype=np.float32))
    np.testing.assert_array_equal(_eval_f32_single(core, "x y xor", xs, ys), np.array([0, 0, 1, 1, 0], dtype=np.float32))
    np.testing.assert_array_equal(_eval_f32_single(core, "x 10 20 ?", xs), np.array([20, 20, 20, 10, 10], dtype=np.float32))

    # D15 constant twins for ordered NaN comparisons & select.
    assert float(_eval_f32_single(core, "0 0 / 0 >", [0.0])[0]) == 0.0
    assert float(_eval_f32_single(core, "0 0 / 10 20 ?", [0.0])[0]) == 20.0

    # D10: bitwise ops convert float operands via round-half-to-even (`cvtps2dq` / `fcvtns`).
    bx = [5.0, 6.0, -1.0, 0.0, 2.5, 3.5]
    by = [3.0, 3.0, 15.0, 7.0, 0.0, 0.0]
    np.testing.assert_array_equal(_eval_f32_single(core, "x y bitand", bx, by), np.array([1.0, 2.0, 15.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(_eval_f32_single(core, "x y bitor", bx, by), np.array([7.0, 7.0, -1.0, 7.0, 2.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(_eval_f32_single(core, "x y bitxor", bx, by), np.array([6.0, 5.0, -16.0, 7.0, 2.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(_eval_f32_single(core, "x bitnot", bx), np.array([-6.0, -7.0, 0.0, -1.0, -3.0, -5.0], dtype=np.float32))


def test_sgn_empty_and_fewer_exprs(core: vs.Core) -> None:
    """sgn returns -1/0/1; empty expr copies plane, fewer exprs repeat last."""
    xs = [-5.0, -0.0, 0.0, 5.0]
    np.testing.assert_array_equal(_eval_f32_single(core, "x sgn", xs), np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32))

    # Empty expr copies Y plane, modifies U and V
    yuv = make_clip(core, width=32, height=16, fmt=vs.YUV420P8, length=2, seed=55)
    out = core.zexpr.Expr([yuv], ["", "255 x -"])
    f_in = yuv.get_frame(0)
    f_out = out.get_frame(0)
    np.testing.assert_array_equal(plane_array(f_out, 0), plane_array(f_in, 0))
    np.testing.assert_array_equal(plane_array(f_out, 1), 255 - plane_array(f_in, 1))
    np.testing.assert_array_equal(plane_array(f_out, 2), 255 - plane_array(f_in, 2))


def test_d15_extra_clip_length_clamping(core: vs.Core) -> None:
    """D15: Output length equals clip 0's length; shorter secondary clips clamp to their last frame."""
    c0 = make_clip(core, width=16, height=8, fmt=vs.GRAY8, length=5, seed=1)
    c1 = make_clip(core, width=16, height=8, fmt=vs.GRAY8, length=2, seed=2)
    out = core.zexpr.Expr([c0, c1], ["x y + 2 /"])
    assert out.num_frames == 5
    # Frame 4 should use c0[4] and c1[1] (clamped last frame)
    a0 = plane_array(c0.get_frame(4), 0).astype(np.float32)
    a1 = plane_array(c1.get_frame(1), 0).astype(np.float32)
    expected = np.rint((a0 + a1) * 0.5).astype(np.uint8)
    np.testing.assert_array_equal(plane_array(out.get_frame(4), 0), expected)


def test_neg_lerp_and_polyval_ops(core: vs.Core) -> None:
    """neg, lerp (constant & dynamic t), polyvalN / polyevalN across float, opt=1, Select, and PropExpr."""
    # neg: sign flip incl. +-0.0, double negation, and opt=1 integer
    xs = [-3.5, -0.0, 0.0, 2.25]
    ng = _eval_f32_single(core, "x neg", xs)
    np.testing.assert_array_equal(ng, np.array([3.5, 0.0, -0.0, -2.25], dtype=np.float32))
    assert math.copysign(1.0, float(ng[1])) == 1.0
    assert math.copysign(1.0, float(ng[2])) == -1.0
    np.testing.assert_array_equal(_eval_f32_single(core, "x neg neg", xs), np.array(xs, dtype=np.float32))

    c8 = make_clip(core, width=17, height=4, fmt=vs.GRAY8, length=1, seed=7)
    out_neg_opt1 = core.zexpr.Expr([c8], ["x neg 255 +"], opt=1)
    np.testing.assert_array_equal(plane_array(out_neg_opt1.get_frame(0), 0), 255 - plane_array(c8.get_frame(0), 0))

    # lerp: exact endpoints (incl. 0.1 0.7 1 lerp == 0.7), constant t, dynamic 3-clip t, extrapolation
    for expr, expected in [
        ("10 20 0 lerp", 10.0),
        ("10 20 1 lerp", 20.0),
        ("10 20 0.5 lerp", 15.0),
        ("10 20 0.25 lerp", 12.5),
        ("20 10 0.25 lerp", 17.5),
        ("10 20 2 lerp", 30.0),
        ("10 20 -1 lerp", 0.0),
        ("0.1 0.7 1 lerp", np.float32(0.7)),
    ]:
        assert _eval_f32_single(core, expr, [0.0])[0] == np.float32(expected)

    a_arr = np.linspace(-1.0, 2.0, 33, dtype=np.float32).reshape(1, -1)
    b_arr = np.linspace(3.0, -0.5, 33, dtype=np.float32).reshape(1, -1)
    t_arr = np.linspace(-0.25, 1.25, 33, dtype=np.float32).reshape(1, -1)
    ca, cb, ct = make_f32_clip(core, a_arr), make_f32_clip(core, b_arr), make_f32_clip(core, t_arr)
    got_var = plane_array(core.zexpr.Expr([ca, cb, ct], ["x y z lerp"]).get_frame(0), 0)
    ref_var = a_arr * (1.0 - t_arr) + b_arr * t_arr
    np.testing.assert_allclose(got_var, ref_var, rtol=1e-6, atol=1e-6)
    got_cst = plane_array(core.zexpr.Expr([ca, cb], ["x y 0.3 lerp"]).get_frame(0), 0)
    ref_cst = a_arr * np.float32(0.7) + b_arr * np.float32(0.3)
    np.testing.assert_allclose(got_cst, ref_cst, rtol=1e-6, atol=1e-6)

    # polyvalN / polyevalN: degree 0, 1, 2, 3, 5, 7, 11 (covering Horner & odd-even Horner paths)
    for expr, expected in [
        ("5 0 polyval0", 5.0),
        ("2 3 4 polyval1", 11.0),
        ("3 2 1 2 polyval2", 17.0),
        ("1 0 -1 3 polyval2", 8.0),
        ("2 -3 0 5 0.5 polyval3", 4.5),
        ("2 -3 0 5 0.5 polyeval3", 4.5),
        ("7 3 2 1 2 polyval2 +", 24.0),
    ]:
        assert _eval_f32_single(core, expr, [0.0])[0] == np.float32(expected)

    px = np.linspace(-0.9, 0.9, 33, dtype=np.float32).reshape(1, -1)
    cpx = make_f32_clip(core, px)
    for deg in (0, 1, 2, 3, 4, 5, 6, 7, 11):
        coeffs = [float(((i % 5) - 2) * 0.25 + 0.1) for i in range(deg + 1)]
        op = "polyeval" if deg % 2 == 1 else "polyval"
        expr = " ".join(f"{c}" for c in coeffs) + f" x {op}{deg}"
        got = plane_array(core.zexpr.Expr([cpx], [expr]).get_frame(0), 0)[0]
        ref = np.polyval(np.array(coeffs, dtype=np.float64), px[0].astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)

    # PropExpr and Select support the new operators too
    base = core.std.BlankClip(width=8, height=8, format=vs.GRAY8, length=1).std.SetFrameProps(A=0.5, B=2.5)
    pe = core.zexpr.PropExpr([base], NegA="x.A neg", CeilB="x.B ceil", LerpAB="x.A x.B 0.5 lerp", Poly="2 3 1 x.A polyval2")
    props = pe.get_frame(0).props
    assert props["NegA"] == pytest.approx(-0.5)
    assert props["CeilB"] == pytest.approx(3.0)
    assert props["LerpAB"] == pytest.approx(1.5)
    assert props["Poly"] == pytest.approx(3.0)


def test_vsjetpack_idiom_operators(core: vs.Core) -> None:
    """linstep, smoothstep, difflimit, absdiff, copysign, minabs, maxabs across float, u8/u16 fast paths, opt=1, and PropExpr."""
    # linstep & smoothstep (const & dynamic edges, plus exact 0.0 / 1.0 endpoints on tricky spans like 16..235)
    xs = np.linspace(-10.0, 100.0, 65, dtype=np.float32).reshape(1, -1)
    cx = make_f32_clip(core, xs)
    t_ref = np.clip((xs - np.float32(16.0)) / np.float32(59.0), 0.0, 1.0)
    s_ref = t_ref * t_ref * (np.float32(3.0) - np.float32(2.0) * t_ref)
    np.testing.assert_allclose(plane_array(core.zexpr.Expr([cx], ["16 75 x linstep"]).get_frame(0), 0), t_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(plane_array(core.zexpr.Expr([cx], ["16 75 x smoothstep"]).get_frame(0), 0), s_ref, rtol=1e-6, atol=1e-6)
    lo_c = make_f32_clip(core, np.full_like(xs, 16.0))
    hi_c = make_f32_clip(core, np.full_like(xs, 75.0))
    np.testing.assert_allclose(plane_array(core.zexpr.Expr([cx, lo_c, hi_c], ["y z x linstep"]).get_frame(0), 0), t_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(plane_array(core.zexpr.Expr([cx, lo_c, hi_c], ["y z x smoothstep"]).get_frame(0), 0), s_ref, rtol=1e-6, atol=1e-6)
    edge_x = make_f32_clip(core, np.array([[0.0, 16.0, 235.0, 255.0]], dtype=np.float32))
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([edge_x], ["16 235 x linstep"]).get_frame(0), 0)[0],
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([edge_x], ["16 235 x smoothstep"]).get_frame(0), 0)[0],
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([edge_x], ["235 16 x linstep"]).get_frame(0), 0)[0],
        np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    # E2: constant-expression bounds (folded in lower.zig Phase A before linstep expansion)
    ce_x = make_f32_clip(core, np.array([[0.0, 48.0 * 257.0, 167.0 * 257.0, 65535.0]], dtype=np.float32))
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([ce_x], ["48 257 * 167 257 * x linstep"]).get_frame(0), 0)[0],
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([ce_x], ["48 257 * 167 257 * x smoothstep"]).get_frame(0), 0)[0],
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    # A1: all-constant linstep/smoothstep folds must match runtime x bit-for-bit
    for op in ("linstep", "smoothstep"):
        for lo_s, hi_s, v_s in (("77", "199", "124"), ("241", "81", "199"), ("0.2", "0.7", "0.5")):
            vx = make_f32_clip(core, np.array([[float(v_s)]], dtype=np.float32))
            got_const = plane_array(core.zexpr.Expr([vx], [f"{lo_s} {hi_s} {v_s} {op}"]).get_frame(0), 0)
            got_rt = plane_array(core.zexpr.Expr([vx], [f"{lo_s} {hi_s} x {op}"]).get_frame(0), 0)
            np.testing.assert_array_equal(got_const.view(np.uint32), got_rt.view(np.uint32))

    # copysign: magnitude + sign bit (incl. +-0.0)
    mags = [-3.5, 3.5, -0.0, 0.0, 2.0, -2.0]
    sgns = [1.0, -1.0, 0.0, -0.0, -0.0, 0.0]
    got_cs = _eval_f32_single(core, "x y copysign", mags, sgns)
    expected_cs = np.copysign(np.array(mags, dtype=np.float32), np.array(sgns, dtype=np.float32))
    np.testing.assert_array_equal(got_cs.view(np.uint32), expected_cs.view(np.uint32))

    # difflimit, absdiff, minabs, maxabs on float + u8/u16 fast paths
    d1 = np.array([[-5.0, -2.0, 0.0, 2.0, 5.0, -5.0, 5.0, 3.0]], dtype=np.float32)
    d2 = np.array([[-3.0, -3.0, 2.0, 3.0, 3.0, 3.0, -3.0, -3.0]], dtype=np.float32)
    cd1, cd2 = make_f32_clip(core, d1), make_f32_clip(core, d2)
    ref_dl = np.clip(d1, np.minimum(0.0, d2), np.maximum(0.0, d2))
    np.testing.assert_array_equal(plane_array(core.zexpr.Expr([cd1, cd2], ["x y difflimit"]).get_frame(0), 0), ref_dl)
    np.testing.assert_array_equal(plane_array(core.zexpr.Expr([cd1, cd2], ["x y absdiff"]).get_frame(0), 0), np.abs(d1 - d2))
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([cd1, cd2], ["x y minabs"]).get_frame(0), 0),
        np.where(np.abs(d1) < np.abs(d2), d1, d2),
    )
    np.testing.assert_array_equal(
        plane_array(core.zexpr.Expr([cd1, cd2], ["x y maxabs"]).get_frame(0), 0),
        np.where(np.abs(d1) > np.abs(d2), d1, d2),
    )

    # Native u8 / u16 / i16 parity
    for fmt in (vs.GRAY8, vs.GRAY16):
        c0 = make_clip(core, width=65, height=4, fmt=fmt, length=1, seed=901)
        c1 = make_clip(core, width=65, height=4, fmt=fmt, length=1, seed=902)
        c2 = make_clip(core, width=65, height=4, fmt=fmt, length=1, seed=903)
        mid = 128 if fmt == vs.GRAY8 else 32768
        assert_same_clip(core.zexpr.Expr([c0, c1], ["x y absdiff"]), core.zexpr.Expr([c0, c1], ["x y - abs"]))
        assert_same_clip(
            core.zexpr.Expr([c0, c1, c2], [f"x y z - y {mid} - difflimit -"]),
            core.zexpr.Expr([c0, c1, c2], [f"y z - D1! y {mid} - D2! D1@ D2@ xor x x D1@ abs D2@ abs < D1@ D2@ ? - ?"]),
        )
        assert_same_clip(
            core.zexpr.Expr([c0, c1, c2], [f"y {mid} - z {mid} - minabs x +"]),
            core.zexpr.Expr([c0, c1, c2], [f"y {mid} - Y! z {mid} - Z! Y@ abs Z@ abs < Y@ Z@ ? x +"]),
        )

    # PropExpr support
    base = core.std.BlankClip(width=8, height=8, format=vs.GRAY8, length=1).std.SetFrameProps(A=3.0, B=4.0, C=-2.0)
    pe = core.zexpr.PropExpr(
        [base],
        H="x.A x.B hypot",
        LS="0 10 x.B linstep",
        AD="x.A x.B absdiff",
        CS="x.A x.C copysign",
        MA="x.A x.C minabs",
        MX="x.A x.C maxabs",
        DL="x.B x.A difflimit",
        E2="x.A exp2",
        L2="x.B log2",
        RS="x.B rsqrt",
    )
    pr = pe.get_frame(0).props
    assert pr["H"] == pytest.approx(5.0)
    assert pr["LS"] == pytest.approx(0.4)
    assert pr["AD"] == pytest.approx(1.0)
    assert pr["CS"] == pytest.approx(-3.0)
    assert pr["MA"] == pytest.approx(-2.0)
    assert pr["MX"] == pytest.approx(3.0)
    assert pr["DL"] == pytest.approx(3.0)
    assert pr["E2"] == pytest.approx(8.0)
    assert pr["L2"] == pytest.approx(2.0)
    assert pr["RS"] == pytest.approx(0.5)
