"""Phase 5 tests: Frame properties (D11), variables, coordinates, stack ops (incl. sort9/25/65), and concurrency."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import vapoursynth as vs

from helpers import assert_same_clip, make_clip, make_f32_clip, plane_array


def test_d11_frame_properties_all_types_and_missing(core: vs.Core) -> None:
    """D11: Read ptInt, ptFloat, ptData (unsigned u8 first byte!), empty ptData -> NaN, missing -> NaN."""
    base = core.std.BlankClip(width=8, height=2, format=vs.GRAYS, length=2)

    def add_props(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        np.asarray(out[0])[:] = float(n + 1)
        out.props["MyInt"] = 42 + n
        out.props["MyFloat"] = 1.5 + n
        out.props["HighByte"] = bytes([0xFF, 0x01])  # unsigned 255, NOT -1!
        out.props["AsciiData"] = b"A"                # 65
        out.props["EmptyData"] = b""                 # NaN
        return out

    clip = base.std.ModifyFrame(base, add_props)

    # Int + Float + unsigned high byte + ASCII byte
    out = core.zexpr.Expr([clip], ["x x.MyInt + x.MyFloat + x.HighByte + x.AsciiData +"])
    for n in range(2):
        got = plane_array(out.get_frame(n), 0)
        expected = (n + 1) + (42 + n) + (1.5 + n) + 255.0 + 65.0
        np.testing.assert_array_equal(got, np.full((2, 8), expected, dtype=np.float32))

    # Missing property and empty ptData evaluate to NaN
    out_missing = core.zexpr.Expr([clip], ["x.DoesNotExist"])
    assert np.all(np.isnan(plane_array(out_missing.get_frame(0), 0)))

    out_empty = core.zexpr.Expr([clip], ["x.EmptyData"])
    assert np.all(np.isnan(plane_array(out_empty.get_frame(0), 0)))


def test_variables_ssa_reassignment_and_coords(core: vs.Core) -> None:
    """Named variables (`a!`, `a@`), reassignment, `N`, `X`, `Y`, `width`, `height`, `pi`."""
    c = make_clip(core, width=19, height=7, fmt=vs.GRAYS, length=3, seed=42)
    expr = "x 2 * a! a@ 1 + a! a@ X + Y 10 * + N 100 * + width + height + pi +"
    out = core.zexpr.Expr([c], [expr])
    for n in range(3):
        got = plane_array(out.get_frame(n), 0)
        src = plane_array(c.get_frame(n), 0)
        ys, xs = np.meshgrid(np.arange(7, dtype=np.float32), np.arange(19, dtype=np.float32), indexing="ij")
        expected = (src * np.float32(2.0) + np.float32(1.0)) + xs + ys * np.float32(10.0) + np.float32(n * 100) + np.float32(19) + np.float32(7) + np.float32(np.pi)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-5)


def test_stack_ops_and_sorting_networks(core: vs.Core) -> None:
    """dupN, dropN, swapN, sort9 (optimal), sort25 (optimal), and sort65 (Batcher odd-even mergesort)."""
    c = make_clip(core, width=17, height=5, fmt=vs.GRAY8, length=1, seed=99)

    # 3x3 median via sort9 (smallest on top: drop4 -> median -> swap1 drop -> ... or drop4 then keep 1 drop4)
    med9 = core.zexpr.Expr(
        [c],
        ["x[-1,-1] x[0,-1] x[1,-1] x[-1,0] x x[1,0] x[-1,1] x[0,1] x[1,1] sort9 drop4 swap4 drop4"],
    )
    arr = plane_array(c.get_frame(0), 0)
    padded = np.pad(arr, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    expected_med9 = np.median(windows.reshape(5, 17, 9), axis=-1).astype(np.uint8)
    np.testing.assert_array_equal(plane_array(med9.get_frame(0), 0), expected_med9)

    # sort5 on runtime x-dependent values (cannot constant-fold): smallest ends on top, largest on bottom.
    top_val = plane_array(core.zexpr.Expr([c], ["x 5 + x 1 + x 4 + x 2 + x 3 + sort5 a! drop4 a@"]).get_frame(0), 0)
    bot_val = plane_array(core.zexpr.Expr([c], ["x 5 + x 1 + x 4 + x 2 + x 3 + sort5 drop4"]).get_frame(0), 0)
    np.testing.assert_array_equal(top_val, np.minimum(arr.astype(np.int32) + 1, 255).astype(np.uint8))
    np.testing.assert_array_equal(bot_val, np.minimum(arr.astype(np.int32) + 5, 255).astype(np.uint8))

    # sort25 on runtime `(x + i) % 25`: smallest (top) is always 0, largest (bottom) is always 24.
    tokens25 = [f"x {i} + 25 %" for i in range(25)]
    got25_min = plane_array(core.zexpr.Expr([c], [" ".join(tokens25) + " sort25 a! drop24 a@"]).get_frame(0), 0)
    got25_max = plane_array(core.zexpr.Expr([c], [" ".join(tokens25) + " sort25 drop24"]).get_frame(0), 0)
    assert np.all(got25_min == 0)
    assert np.all(got25_max == 24)

    # sort65 (Batcher odd-even mergesort path, N > 64) on runtime `(x + i) % 65 + 1`: min == 1, max == 65.
    vals65 = " ".join(f"x {(i * 37) % 65} + 65 % 1 +" for i in range(65))
    min65 = plane_array(core.zexpr.Expr([c], [f"{vals65} sort65 a! drop64 a@"]).get_frame(0), 0)
    max65 = plane_array(core.zexpr.Expr([c], [f"{vals65} sort65 drop64"]).get_frame(0), 0)
    assert np.all(min65 == 1)
    assert np.all(max65 == 65)

    # Multi-clip src26+ syntax (over 26 clips!)
    clips = [make_clip(core, width=8, height=4, fmt=vs.GRAY8, length=1, seed=i) for i in range(28)]
    got_28 = core.zexpr.Expr(clips, ["src0 src27 + 2 /"])
    ref_28 = core.zexpr.Expr([clips[0], clips[27]], ["x y + 2 /"])
    assert_same_clip(got_28, ref_28)


def test_multithreaded_concurrent_get_frame(core: vs.Core) -> None:
    """ThreadPoolExecutor stress test verifying .Parallel filter thread safety across 32 frames."""
    c0 = make_clip(core, width=64, height=32, fmt=vs.YUV420P8, length=32, seed=7)
    c1 = make_clip(core, width=64, height=32, fmt=vs.YUV420P8, length=32, seed=8)
    out = core.zexpr.Expr(
        [c0, c1],
        ["x[-1,0] x + x[1,0] + y + 4 /", "x y + 2 /", "255 x -"],
    )
    serial = [plane_array(out.get_frame(n), 0) for n in range(32)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(lambda idx: plane_array(out.get_frame(idx), 0), n) for n in range(32)]
        parallel = [f.result() for f in futures]

    for n in range(32):
        np.testing.assert_array_equal(parallel[n], serial[n])
