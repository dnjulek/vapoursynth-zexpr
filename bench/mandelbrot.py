# 23-iteration Mandelbrot fractal zoomer (translated from AviSynth+ Expr examples)
# across YUV420P8 (0..2), YUV420P16 (3..5), YUV420PS (6..8).
# Exercises frame-invariant `1.01 N pow` hoisting, `X`/`Y`/`width`/`height` coordinate math,
# deep unrolled 23-step FMA/min chains, and neutral chroma plane fills.
#
# Run from repo root, one output at a time (0/3/6=zexpr, 1/4/7=akarin [no SDE], 2/5/8=cranexpr):
#   vspipe -p -o 0 --end 499 bench/mandelbrot.py --    # YUV420P8  zexpr (-o 1 akarin, -o 2 cranexpr)
#   vspipe -p -o 3 --end 199 bench/mandelbrot.py --    # YUV420P16 zexpr (-o 4 akarin, -o 5 cranexpr)
#   vspipe -p -o 6 --end 199 bench/mandelbrot.py --    # YUV420PS  zexpr (-o 7 akarin, -o 8 cranexpr)
# Single thread: add  -a threads=1
# Output check (p8): for o in 0 1 2; do vspipe -o $o --end 9 bench/mandelbrot.py - | sha256sum; done
# Needs: zig build -Doptimize=ReleaseFast; pip install VapourSynth numpy vapoursynth-akarin (+ vapoursynth-cranexpr for -o 2/5/8)

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from common import run_case

_STEP = "X@ dup * Y@ dup * - A@ + T! X@ Y@ 2 * * B@ + 2 min Y! T@ 2 min X! "
_B = _STEP + _STEP
_C = _B + _B
_BODY = (
    "X width / sxr! "
    "Y height / syr! "
    "N frameno! "
    "sxr@ 3 * 2 - -1.2947627 - 1.01 frameno@ pow / -1.2947627 + dup A! X! "
    "syr@ 2 * 1 - 0.4399695 - 1.01 frameno@ pow / 0.4399695 + dup B! Y! "
    + _C * 5
    + _B
    + _STEP
    + "X@ dup * Y@ dup * + 4 < 0 "
)

run_case(
    globals(),
    {
        "p8": [_BODY + "255 ?", "128", "128"],
        "p16": [_BODY + "65535 ?", "32768", "32768"],
        "ps": [_BODY + "1 ?", "0", "0"],
    },
)
