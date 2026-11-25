# 2-clip nonlinear FineSharp kernel (vs-jetpack vsrgtools/sharp.py) across YUV420P8 (0..2), YUV420P16 (3..5), YUV420PS (6..8).
# Exercises 2D u8x8 LUT (p8) and 2x-interleaved transcendental + division + select DAG (p16/ps).
#
# Run from repo root, one output at a time (0/3/6=zexpr, 1/4/7=akarin [no SDE], 2/5/8=cranexpr):
#   vspipe -p -o 0 --end 1999 bench/fine_sharp.py --    # YUV420P8  zexpr (-o 1 akarin, -o 2 cranexpr)
#   vspipe -p -o 3 --end 499  bench/fine_sharp.py --    # YUV420P16 zexpr (-o 4 akarin, -o 5 cranexpr)
#   vspipe -p -o 6 --end 499  bench/fine_sharp.py --    # YUV420PS  zexpr (-o 7 akarin, -o 8 cranexpr)
# Single thread: add  -a threads=1
# Output check (p8): for o in 0 1 2; do vspipe -o $o --end 9 bench/fine_sharp.py - | sha256sum; done
# Needs: zig build -Doptimize=ReleaseFast; pip install VapourSynth numpy vapoursynth-akarin (+ vapoursynth-cranexpr for -o 2/5/8)

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from common import run_case


def _fine_sharp(range_size: str) -> str:
    return (
        f"x y = x dup x y - {range_size} 256 / / dup dup dup abs 1.49 / 0.7861635 pow "
        "swap3 abs 0.01 + / swap dup * dup 0.01 + / * * 2.0 * + ?"
    )


run_case(
    globals(),
    {
        "p8": _fine_sharp("256"),
        "p16": _fine_sharp("65536"),
        "ps": _fine_sharp("1.0"),
    },
    num_clips=2,
)
