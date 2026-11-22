"""Phase 6 tests: `zexpr.Select` contracts (B2, D8, X2, Y6, Y17, Y19)."""

from __future__ import annotations

import numpy as np
import pytest
import vapoursynth as vs

from helpers import make_clip, plane_array


def _with_props(core: vs.Core, clip: vs.VideoNode, fn) -> vs.VideoNode:
    def apply(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        for k, v in fn(n).items():
            out.props[k] = v
        return out

    return clip.std.ModifyFrame(clip, apply)


def test_select_per_plane_and_clip_index_zero_b2(core: vs.Core) -> None:
    """B2/Y6: Clip index 0 works cleanly (non-null frameData slot) and per-plane selection mixes planes."""
    c0 = core.std.BlankClip(width=16, height=8, format=vs.YUV420P8, color=[10, 20, 30], length=4)
    c1 = core.std.BlankClip(width=16, height=8, format=vs.YUV420P8, color=[110, 120, 130], length=4)
    c2 = core.std.BlankClip(width=16, height=8, format=vs.YUV420P8, color=[210, 220, 230], length=4)

    # Frame 0: Y=0, U=1, V=2; Frame 1: Y=2, U=0, V=1; Frame 2: Y=0, U=0, V=0
    p0 = _with_props(core, c0, lambda n: {"Y": [0, 2, 0, 1][n], "U": [1, 0, 0, 2][n], "V": [2, 1, 0, 0][n]})
    sel = core.zexpr.Select([c0, c1, c2], [p0], ["x.Y", "x.U", "x.V"])

    expected = [
        (10, 120, 230),
        (210, 20, 130),
        (10, 20, 30),
        (110, 220, 30),
    ]
    for n, (ey, eu, ev) in enumerate(expected):
        f = sel.get_frame(n)
        assert np.all(plane_array(f, 0) == ey)
        assert np.all(plane_array(f, 1) == eu)
        assert np.all(plane_array(f, 2) == ev)


def test_select_d8_clamping_half_away_rounding_and_nan(core: vs.Core) -> None:
    """D8/X2: Select rounds half-away-from-zero (`0.5 -> 1`, `1.5 -> 2`, `2.5 -> 3`), clamps NaN/neg -> 0 and +inf/oob -> last."""
    clips = [
        core.std.BlankClip(width=8, height=4, format=vs.GRAY8, color=[i * 50], length=9)
        for i in range(4)
    ]
    vals = [0.5, 1.5, 2.5, -5.0, 99.0, float("nan"), float("inf"), float("-inf"), 0.49]
    p = _with_props(core, clips[0], lambda n: {"Idx": vals[n]})
    sel = core.zexpr.Select(clips, [p], ["x.Idx"])
    expected_idxs = [1, 2, 3, 0, 3, 0, 3, 0, 0]
    for n, ei in enumerate(expected_idxs):
        assert int(plane_array(sel.get_frame(n), 0)[0, 0]) == ei * 50

    # D9: `round` inside Select uses round-half-to-even (`2.5 round -> 2`, `0.5 round -> 0`).
    sel_rnd = core.zexpr.Select(clips, [p], ["x.Idx round"])
    assert int(plane_array(sel_rnd.get_frame(0), 0)[0, 0]) == 0 * 50  # 0.5 -> 0
    assert int(plane_array(sel_rnd.get_frame(2), 0)[0, 0]) == 2 * 50  # 2.5 -> 2

    # Missing property defaults to 0.0 -> clip 0
    sel_missing = core.zexpr.Select(clips, [clips[0]], ["x.NoSuchProp"])
    assert int(plane_array(sel_missing.get_frame(0), 0)[0, 0]) == 0


def test_select_v6_clip_dimensions_on_subsampled_planes(core: vs.Core) -> None:
    """V6: Select exposes full clip width/height (not subsampled plane dims) on every plane."""
    c0 = core.std.BlankClip(width=16, height=8, format=vs.YUV420P8, color=[10, 20, 30], length=1)
    c1 = core.std.BlankClip(width=16, height=8, format=vs.YUV420P8, color=[110, 120, 130], length=1)
    for expr in ("width 10 >", "height 6 >"):
        sel = core.zexpr.Select([c0, c1], [c0], [expr])
        f = sel.get_frame(0)
        assert np.all(plane_array(f, 0) == 110)
        assert np.all(plane_array(f, 1) == 120)
        assert np.all(plane_array(f, 2) == 130)


def test_select_argmin_argmax_argsort_and_short_prop_src_y19(core: vs.Core) -> None:
    """Select supports argminN/argmaxN/argsortN and clamps shorter prop_src to its last frame (Y19)."""
    clips = [
        core.std.BlankClip(width=8, height=4, format=vs.GRAY8, color=[i * 60], length=4)
        for i in range(4)
    ]
    # Shorter prop_src (length=2) clamps frames 2, 3 to frame 1!
    p_short = _with_props(
        core,
        core.std.BlankClip(width=8, height=4, format=vs.GRAY8, length=2),
        lambda n: {"a": 10.0 if n == 0 else 5.0, "b": 2.0 if n == 0 else 20.0, "c": 7.0},
    )
    sel_min = core.zexpr.Select(clips, [p_short], ["x.a x.b x.c argmin3"])
    sel_max = core.zexpr.Select(clips, [p_short], ["x.a x.b x.c argmax3"])
    # Frame 0: (10, 2, 7) -> argmin=1, argmax=0. Frames 1..3: (5, 20, 7) -> argmin=0, argmax=1
    assert int(plane_array(sel_min.get_frame(0), 0)[0, 0]) == 60
    assert int(plane_array(sel_min.get_frame(3), 0)[0, 0]) == 0
    assert int(plane_array(sel_max.get_frame(0), 0)[0, 0]) == 0
    assert int(plane_array(sel_max.get_frame(3), 0)[0, 0]) == 60

    # argsort3 pushes indices of descending-sorted values so stack top holds the index of the smallest value
    # and stack bottom holds the index of the largest value (matching akarin.Select).
    # For (10, 2, 7): descending indices are [0 (10), 2 (7), 1 (2)] -> top is 1, bottom is 0.
    sel_as_top = core.zexpr.Select(clips, [p_short], ["x.a x.b x.c argsort3 a! drop2 a@"])
    sel_as_bot = core.zexpr.Select(clips, [p_short], ["x.a x.b x.c argsort3 drop2"])
    assert int(plane_array(sel_as_top.get_frame(0), 0)[0, 0]) == 1 * 60
    assert int(plane_array(sel_as_bot.get_frame(0), 0)[0, 0]) == 0 * 60


def test_select_validation_y17(core: vs.Core) -> None:
    """Y17: Reject mismatched clip_src, variable-format prop_src, pixel/X/Y access, and empty expr."""
    c0 = make_clip(core, width=16, height=8, fmt=vs.GRAY8, length=4)
    c_len2 = make_clip(core, width=16, height=8, fmt=vs.GRAY8, length=2)
    c_fmt = make_clip(core, width=16, height=8, fmt=vs.GRAY16, length=4)
    var_fmt = core.std.Splice([c0, c_fmt], mismatch=True)

    with pytest.raises(vs.Error):
        core.zexpr.Select([c0, c_len2], [c0], ["0"])
    with pytest.raises(vs.Error):
        core.zexpr.Select([c0, c_fmt], [c0], ["0"])
    with pytest.raises(vs.Error):
        core.zexpr.Select([c0], [var_fmt], ["0"])
    with pytest.raises(vs.Error):
        core.zexpr.Select([c0], [c0], ["x"])
    with pytest.raises(vs.Error):
        core.zexpr.Select([c0], [c0], ["X"])
    with pytest.raises(vs.Error):
        core.zexpr.Select([c0], [c0], [""])
