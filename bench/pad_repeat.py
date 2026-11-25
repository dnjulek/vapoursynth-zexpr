# Repeat-border padding (vs-jetpack vstools/utils/misc.py) across YUV420P8 (0..2), YUV420P16 (3..5), YUV420PS (6..8).
# Exercises frame/row LICM (hoisted corners & edge loads), contiguous `abs_row_ptrs` row loads, and 7 nested `?` selects.
#
# Run from repo root, one output at a time (0/3/6=zexpr, 1/4/7=akarin [no SDE], 2/5/8=cranexpr):
#   vspipe -p -o 0 --end 1999 bench/pad_repeat.py --    # YUV420P8  zexpr (-o 1 akarin, -o 2 cranexpr)
#   vspipe -p -o 3 --end 999  bench/pad_repeat.py --    # YUV420P16 zexpr (-o 4 akarin, -o 5 cranexpr)
#   vspipe -p -o 6 --end 999  bench/pad_repeat.py --    # YUV420PS  zexpr (-o 7 akarin, -o 8 cranexpr)
# Single thread: add  -a threads=1
# Output check (p8): for o in 0 1 2; do vspipe -o $o --end 9 bench/pad_repeat.py - | sha256sum; done
# Needs: zig build -Doptimize=ReleaseFast; pip install VapourSynth numpy vapoursynth-akarin (+ vapoursynth-cranexpr for -o 2/5/8)

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from common import run_case


def _pad_expr(pad: int, w: int, h: int) -> str:
    r = w - 1 - pad
    b = h - 1 - pad
    return (
        f"X {pad} < L! Y {pad} < T! X {r} > R! Y {b} > B! T@ B@ or L@ R@ or and "
        f"L@ T@ {pad} {pad} x[] {pad} {b} x[] ? T@ {r} {pad} x[] {r} {b} x[] ? ? "
        f"L@ {pad} Y x[] R@ {r} Y x[] T@ X {pad} x[] B@ X {b} x[] x ? ? ? ? ?"
    )


_Y = _pad_expr(16, 1920, 1080)
_UV = _pad_expr(8, 960, 540)

run_case(globals(), [_Y, _UV, _UV])
