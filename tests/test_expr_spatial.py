"""Phase 3 spatial access tests: x[dx,dy][:c|:m], x[][:c|:m], B1/D6/D7/Y16/Y27."""

from __future__ import annotations

import numpy as np
import pytest
import vapoursynth as vs

from helpers import make_clip, make_f32_clip, plane_array


def _clamp_idx(i: int, dim: int) -> int:
    if dim <= 1:
        return 0
    return max(0, min(dim - 1, i))


def _mirror_idx(i: int, dim: int) -> int:
    if dim <= 1:
        return 0
    if i < 0:
        i = -1 - i
    if i >= dim:
        i = 2 * dim - 1 - i
    return max(0, min(dim - 1, i))


def _oracle_rel(arr: np.ndarray, dx: int, dy: int, mode: str) -> np.ndarray:
    h, w = arr.shape
    out = np.empty_like(arr)
    eff_dx = max(-w, min(w, dx)) if mode == "m" else dx
    eff_dy = max(-h, min(h, dy)) if mode == "m" else dy
    fn = _mirror_idx if mode == "m" else _clamp_idx
    for y in range(h):
        sy = fn(y + eff_dy, h)
        for x in range(w):
            sx = fn(x + eff_dx, w)
            out[y, x] = arr[sy, sx]
    return out


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65])
@pytest.mark.parametrize("mode,suffix", [("c", ":c"), ("m", ":m")])
def test_rel_horizontal_offsets_all_widths(core: vs.Core, width: int, mode: str, suffix: str) -> None:
    src = make_clip(core, width=width, height=5, fmt=vs.GRAYS, length=1, seed=101)
    src_arr = plane_array(src.get_frame(0), 0)
    offsets = [-width - 5, -width, -17, -9, -8, -7, -4, -3, -1, 1, 3, 4, 7, 8, 9, 17, width, width + 5]
    for dx in offsets:
        expr = f"x[{dx},0]{suffix}"
        got = plane_array(core.zexpr.Expr([src], [expr]).get_frame(0), 0)
        expected = _oracle_rel(src_arr, dx, 0, mode)
        np.testing.assert_array_equal(got, expected, err_msg=f"width={width} expr={expr}")


@pytest.mark.parametrize("height", [1, 2, 7])
@pytest.mark.parametrize("mode,suffix", [("c", ":c"), ("m", ":m")])
def test_rel_vertical_offsets_y27(core: vs.Core, height: int, mode: str, suffix: str) -> None:
    src = make_clip(core, width=19, height=height, fmt=vs.GRAY16, length=1, seed=202)
    src_arr = plane_array(src.get_frame(0), 0)
    offsets = [-2 * height - 1, -height - 1, -height, -1, 1, height, height + 1, 2 * height + 1]
    for dy in offsets:
        expr = f"x[0,{dy}]{suffix}"
        got = plane_array(core.zexpr.Expr([src], [expr]).get_frame(0), 0)
        expected = _oracle_rel(src_arr, 0, dy, mode)
        np.testing.assert_array_equal(got, expected, err_msg=f"height={height} expr={expr}")


def test_rel_2d_blur_u8_u16_f32_and_boundary_arg(core: vs.Core) -> None:
    for fmt in [vs.GRAY8, vs.GRAY10, vs.GRAY16, vs.GRAYS]:
        src = make_clip(core, width=25, height=9, fmt=fmt, length=1, seed=303)
        src_f = plane_array(src.get_frame(0), 0).astype(np.float32)
        for boundary, mode in [(0, "c"), (1, "m")]:
            expr = "x[-1,-1] x[0,-1] + x[1,-1] + x[-1,0] + x + x[1,0] + x[-1,1] + x[0,1] + x[1,1] + 9 /"
            got = plane_array(core.zexpr.Expr([src], [expr], boundary=boundary).get_frame(0), 0)
            acc = np.zeros_like(src_f)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    acc = acc + _oracle_rel(src_f, dx, dy, mode)
            expected = acc / np.float32(9.0)
            if fmt == vs.GRAYS:
                np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)
            else:
                peak = (1 << src.format.bits_per_sample) - 1
                exp_i = np.rint(np.clip(expected, 0, peak)).astype(got.dtype)
                np.testing.assert_array_equal(got, exp_i)


def test_abs_access_d7_y16_and_nonfinite_coords(core: vs.Core) -> None:
    w, h = 17, 5
    src = make_clip(core, width=w, height=h, fmt=vs.GRAYS, length=1, seed=404)
    src_arr = plane_array(src.get_frame(0), 0)

    # Y16: suffix-less `x[]` ALWAYS clamps even when boundary=1!
    got_default = plane_array(core.zexpr.Expr([src], ["X 5 - Y 3 - x[]"], boundary=1).get_frame(0), 0)
    got_clamp = plane_array(core.zexpr.Expr([src], ["X 5 - Y 3 - x[]:c"], boundary=1).get_frame(0), 0)
    np.testing.assert_array_equal(got_default, got_clamp)

    # Explicit `:m` mirrors out-of-range coordinates.
    got_mirror = plane_array(core.zexpr.Expr([src], ["X 2 - Y 1 - x[]:m"]).get_frame(0), 0)
    expected_m = _oracle_rel(src_arr, -2, -1, "m")
    np.testing.assert_array_equal(got_mirror, expected_m)

    # D7 / U6: NaN, -inf, +inf coordinates are 100% bounds-safe & ISA-identical.
    coords = np.array(
        [[np.nan, -np.inf, np.inf, -1000.0, 1000.0, 0.0, 3.4, 3.6, 16.0]],
        dtype=np.float32,
    )
    coord_clip = make_f32_clip(core, coords)
    src_1row = make_clip(core, width=coords.shape[1], height=1, fmt=vs.GRAYS, length=1, seed=405)
    src_1row_arr = plane_array(src_1row.get_frame(0), 0)[0]

    # `:c`: NaN & -inf -> col 0, +inf -> col w-1
    got_c = plane_array(core.zexpr.Expr([src_1row, coord_clip], ["y 0 x[]:c"]).get_frame(0), 0)[0]
    assert got_c[0] == src_1row_arr[0]  # NaN -> 0
    assert got_c[1] == src_1row_arr[0]  # -inf -> 0
    assert got_c[2] == src_1row_arr[-1]  # +inf -> w-1
    assert got_c[3] == src_1row_arr[0]
    assert got_c[4] == src_1row_arr[-1]

    # `:m`: NaN & -inf -> clamp to -w -> reflect to w-1; +inf -> clamp to 2*w-1 -> reflect to 0 (always in bounds!).
    got_m = plane_array(core.zexpr.Expr([src_1row, coord_clip], ["y 0 x[]:m"]).get_frame(0), 0)[0]
    assert got_m[0] == src_1row_arr[-1]  # NaN -> -w -> w-1
    assert got_m[1] == src_1row_arr[-1]  # -inf -> -w -> w-1
    assert got_m[2] == src_1row_arr[0]   # +inf -> 2w-1 -> 0

    # Also test u8 and u16 dynamic gathers (never over-reading past end of last row).
    for fmt in [vs.GRAY8, vs.GRAY16]:
        u_src = make_clip(core, width=17, height=5, fmt=fmt, length=1, seed=406)
        u_arr = plane_array(u_src.get_frame(0), 0)
        got_u = plane_array(core.zexpr.Expr([u_src], ["width 1 - X - height 1 - Y - x[]:c"]).get_frame(0), 0)
        np.testing.assert_array_equal(got_u, u_arr[::-1, ::-1])


@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY10, vs.GRAY16, vs.GRAY32, vs.GRAYH, vs.GRAYS])
@pytest.mark.parametrize("width", [1, 17, 65])
def test_wide_2d_rotate_gathers_all_formats(core: vs.Core, fmt: vs.PresetVideoFormat, width: int) -> None:
    """2D affine/rotate-style dynamic gathers across all sample formats and AVX-512/NEON interior+tail widths."""
    height = 7
    src = make_clip(core, width=width, height=height, fmt=fmt, length=1, seed=(int(fmt) & 0xFFFF) + width)
    src_arr = plane_array(src.get_frame(0), 0)
    # 2D non-separable coordinate transform exercising full 16-lane AVX-512 gathers (both vpgatherdd & reg gather).
    expr = "X 0.8 * Y 0.6 * - 2 + Y 0.6 * X 0.2 * + x[]:c"
    got = plane_array(core.zexpr.Expr([src], [expr]).get_frame(0), 0)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    sx_f = xx * np.float32(0.8) - yy * np.float32(0.6) + np.float32(2.0)
    sy_f = yy * np.float32(0.6) + xx * np.float32(0.2)
    sx = np.clip(np.rint(sx_f).astype(np.int64), 0, width - 1)
    sy = np.clip(np.rint(sy_f).astype(np.int64), 0, height - 1)
    expected = src_arr[sy, sx]
    if fmt == vs.GRAY32:
        # opt=0 GRAY32 loads signed i32 -> f32 and clamps to [0, 2147483520].
        expected = np.clip(np.rint(np.clip(expected.view(np.int32).astype(np.float32), 0.0, 2147483520.0)), 0, 2147483520).astype(np.uint32)
    np.testing.assert_array_equal(got, expected)
