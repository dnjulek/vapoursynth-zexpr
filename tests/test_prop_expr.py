"""Phase 6 tests: `zexpr.PropExpr` contracts (akarin dict + cranexpr kwargs, Y17, Y18, Y19, Y20, atomic swap)."""

from __future__ import annotations

import pytest
import vapoursynth as vs


def _with_props(core: vs.Core, clip: vs.VideoNode, fn) -> vs.VideoNode:
    def apply(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        for k, v in fn(n).items():
            out.props[k] = v
        return out

    return clip.std.ModifyFrame(clip, apply)


def test_prop_expr_kwargs_atomic_swap_and_delete(core: vs.Core) -> None:
    """Cranexpr **kwargs style: atomic property swap, arithmetic, int vs float storage, empty string deletes key."""
    base = core.std.BlankClip(width=8, height=4, format=vs.GRAY8, length=3)
    src = _with_props(core, base, lambda n: {"A": 10 + n, "B": 20 + n, "DropMe": 99})

    out = core.zexpr.PropExpr(
        [src],
        A="x.B",
        B="x.A",
        Sum="x.A x.B +",
        Half="x.A 0.5 *",
        FrameIdx="N 100 +",
        DropMe="",
    )

    for n in range(3):
        props = out.get_frame(n).props
        assert props["A"] == 20 + n and isinstance(props["A"], int)
        assert props["B"] == 10 + n and isinstance(props["B"], int)
        assert props["Sum"] == 30 + 2 * n and isinstance(props["Sum"], int)
        assert props["FrameIdx"] == 100 + n and isinstance(props["FrameIdx"], int)
        if (10 + n) % 2 == 0:
            assert props["Half"] == (10 + n) // 2 and isinstance(props["Half"], int)
        else:
            assert props["Half"] == pytest.approx((10 + n) * 0.5) and isinstance(props["Half"], float)
        assert "DropMe" not in props


def test_prop_expr_akarin_dict_exact_i64_f64_y18_and_arrays(core: vs.Core) -> None:
    """Y18: Akarin `dict=lambda: {...}` passes i64/f64 constants through exactly without f32 loss, and cycles arrays."""
    base = core.std.BlankClip(width=8, height=4, format=vs.GRAY8, length=4)
    big_i64 = (1 << 55) + 123456789  # Would lose low bits if round-tripped through f32!
    precise_f64 = 1.0000000000001235

    out = core.zexpr.PropExpr(
        [base],
        dict=lambda: {
            "BigInt": big_i64,
            "PreciseFloat": precise_f64,
            "CycledInt": [10, 20, 30],
            "CycledExpr": ["N 1 +", "N 10 +"],
            "dict": "42",  # Y20: reserved kwarg name works via `dict=`!
        },
    )

    for n in range(4):
        props = out.get_frame(n).props
        assert props["BigInt"] == big_i64 and isinstance(props["BigInt"], int)
        assert props["PreciseFloat"] == precise_f64 and isinstance(props["PreciseFloat"], float)
        assert props["CycledInt"] == [10, 20, 30][n % 3]
        assert props["CycledExpr"] == (n + 1 if n % 2 == 0 else n + 10)
        assert props["dict"] == 42


def test_prop_expr_multi_clip_and_short_clip_y19(core: vs.Core) -> None:
    """Y19: Secondary shorter clips clamp to their last frame."""
    c0 = _with_props(core, core.std.BlankClip(width=8, height=4, format=vs.GRAY8, length=4), lambda n: {"V": n})
    c1 = _with_props(core, core.std.BlankClip(width=8, height=4, format=vs.GRAY8, length=2), lambda n: {"V": 100 + n})
    out = core.zexpr.PropExpr([c0, c1], Combined="x.V y.V +")
    assert [out.get_frame(n).props["Combined"] for n in range(4)] == [100, 102, 103, 104]


def test_prop_expr_validation_y17_and_y20_error_paths(core: vs.Core) -> None:
    """Y17/Y20: Reject both dict+kwargs, empty dict, zero bindings, callback exception, bad expr, and non-string kwarg."""
    c = core.std.BlankClip(width=8, height=4, format=vs.GRAY8, length=2)

    # Zero bindings
    with pytest.raises(vs.Error):
        core.zexpr.PropExpr([c])

    # Empty dict
    with pytest.raises(vs.Error):
        core.zexpr.PropExpr([c], dict=lambda: {})

    # Both dict and kwargs
    with pytest.raises(vs.Error):
        core.zexpr.PropExpr([c], dict=lambda: {"A": "1"}, B="2")

    # Non-string kwarg (cranexpr style requires string expressions)
    with pytest.raises(vs.Error):
        core.zexpr.PropExpr([c], A=123)

    # Bad expression inside dict (exercises Y20 cleanup of function + both maps!)
    with pytest.raises(vs.Error):
        core.zexpr.PropExpr([c], dict=lambda: {"A": "x +"})

    # Callback raises Python error (exercises Y20 cleanup!)
    def boom():
        raise ValueError("callback failed")

    with pytest.raises(vs.Error):
        core.zexpr.PropExpr([c], dict=boom)
