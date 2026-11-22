# 5x5 8-direction Kirsch edge detector (boundary=1) across YUV420P8 (0..2), YUV420P16 (3..5), YUV420PS (6..8).
# Expression from cranexpr's bench/ex-kirsch.py (https://github.com/sgt0/cranexpr); exercises 5x5 mirror spatial loads and deep DAGs.
#
# Run from repo root, one output at a time (0/3/6=zexpr, 1/4/7=akarin [no SDE], 2/5/8=cranexpr):
#   vspipe -p -o 0 --end 499 bench/kirsch.py --    # YUV420P8  zexpr (-o 1 akarin, -o 2 cranexpr)
#   vspipe -p -o 3 --end 499 bench/kirsch.py --    # YUV420P16 zexpr (-o 4 akarin, -o 5 cranexpr)
#   vspipe -p -o 6 --end 499 bench/kirsch.py --    # YUV420PS  zexpr (-o 7 akarin, -o 8 cranexpr)
# Single thread: add  -a threads=1
# Output check (p8): for o in 0 1 2; do vspipe -o $o --end 9 bench/kirsch.py - | sha256sum; done
# Needs: zig build -Doptimize=ReleaseFast; pip install VapourSynth numpy vapoursynth-akarin (+ vapoursynth-cranexpr for -o 2/5/8)

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from common import run_case

EXPR = """
x[-2,-2] 9 * x[-1,-2] 9 * x[0,-2] 9 * x[1,-2] 9 * x[2,-2] 9 * x[-2,-1] 9 *
x[-1,-1] 5 * x[0,-1] 5 * x[1,-1] 5 * x[2,-1] 9 * x[-2,0] -7 * x[-1,0] -3 *
x[1,0] -3 * x[2,0] -7 * x[-2,1] -7 * x[-1,1] -3 * x[0,1] -3 * x[1,1] -3 * x[2,1]
-7 * x[-2,2] -7 * x[-1,2] -7 * x[0,2] -7 * x[1,2] -7 * x[2,2] -7 * + + + + + + +
+ + + + + + + + + + + + + + + + abs x[-2,-2] 9 * x[-1,-2] 9 * x[0,-2] 9 *
x[1,-2] 9 * x[2,-2] -7 * x[-2,-1] 9 * x[-1,-1] 5 * x[0,-1] 5 * x[1,-1] -3 *
x[2,-1] -7 * x[-2,0] 9 * x[-1,0] 5 * x[1,0] -3 * x[2,0] -7 * x[-2,1] 9 * x[-1,1]
-3 * x[0,1] -3 * x[1,1] -3 * x[2,1] -7 * x[-2,2] -7 * x[-1,2] -7 * x[0,2] -7 *
x[1,2] -7 * x[2,2] -7 * + + + + + + + + + + + + + + + + + + + + + + + abs
x[-2,-2] 9 * x[-1,-2] 9 * x[0,-2] -7 * x[1,-2] -7 * x[2,-2] -7 * x[-2,-1] 9 *
x[-1,-1] 5 * x[0,-1] -3 * x[1,-1] -3 * x[2,-1] -7 * x[-2,0] 9 * x[-1,0] 5 *
x[1,0] -3 * x[2,0] -7 * x[-2,1] 9 * x[-1,1] 5 * x[0,1] -3 * x[1,1] -3 * x[2,1]
-7 * x[-2,2] 9 * x[-1,2] 9 * x[0,2] -7 * x[1,2] -7 * x[2,2] -7 * + + + + + + + +
+ + + + + + + + + + + + + + + abs x[-2,-2] -7 * x[-1,-2] -7 * x[0,-2] -7 *
x[1,-2] -7 * x[2,-2] -7 * x[-2,-1] 9 * x[-1,-1] -3 * x[0,-1] -3 * x[1,-1] -3 *
x[2,-1] -7 * x[-2,0] 9 * x[-1,0] 5 * x[1,0] -3 * x[2,0] -7 * x[-2,1] 9 * x[-1,1]
5 * x[0,1] 5 * x[1,1] -3 * x[2,1] -7 * x[-2,2] 9 * x[-1,2] 9 * x[0,2] 9 * x[1,2]
9 * x[2,2] -7 * + + + + + + + + + + + + + + + + + + + + + + + abs x[-2,-2] -7 *
x[-1,-2] -7 * x[0,-2] -7 * x[1,-2] -7 * x[2,-2] -7 * x[-2,-1] -7 * x[-1,-1] -3 *
x[0,-1] -3 * x[1,-1] -3 * x[2,-1] -7 * x[-2,0] -7 * x[-1,0] -3 * x[1,0] -3 *
x[2,0] -7 * x[-2,1] 9 * x[-1,1] 5 * x[0,1] 5 * x[1,1] 5 * x[2,1] 9 * x[-2,2] 9 *
x[-1,2] 9 * x[0,2] 9 * x[1,2] 9 * x[2,2] 9 * + + + + + + + + + + + + + + + + + +
+ + + + + abs x[-2,-2] -7 * x[-1,-2] -7 * x[0,-2] -7 * x[1,-2] -7 * x[2,-2] -7 *
x[-2,-1] -7 * x[-1,-1] -3 * x[0,-1] -3 * x[1,-1] -3 * x[2,-1] 9 * x[-2,0] -7 *
x[-1,0] -3 * x[1,0] 5 * x[2,0] 9 * x[-2,1] -7 * x[-1,1] -3 * x[0,1] 5 * x[1,1] 5
* x[2,1] 9 * x[-2,2] -7 * x[-1,2] 9 * x[0,2] 9 * x[1,2] 9 * x[2,2] 9 * + + + + +
+ + + + + + + + + + + + + + + + + + abs x[-2,-2] -7 * x[-1,-2] -7 * x[0,-2] -7 *
x[1,-2] 9 * x[2,-2] 9 * x[-2,-1] -7 * x[-1,-1] -3 * x[0,-1] -3 * x[1,-1] 5 *
x[2,-1] 9 * x[-2,0] -7 * x[-1,0] -3 * x[1,0] 5 * x[2,0] 9 * x[-2,1] -7 * x[-1,1]
-3 * x[0,1] -3 * x[1,1] 5 * x[2,1] 9 * x[-2,2] -7 * x[-1,2] -7 * x[0,2] -7 *
x[1,2] 9 * x[2,2] 9 * + + + + + + + + + + + + + + + + + + + + + + + abs x[-2,-2]
-7 * x[-1,-2] 9 * x[0,-2] 9 * x[1,-2] 9 * x[2,-2] 9 * x[-2,-1] -7 * x[-1,-1] -3
* x[0,-1] 5 * x[1,-1] 5 * x[2,-1] 9 * x[-2,0] -7 * x[-1,0] -3 * x[1,0] 5 *
x[2,0] 9 * x[-2,1] -7 * x[-1,1] -3 * x[0,1] -3 * x[1,1] -3 * x[2,1] 9 * x[-2,2]
-7 * x[-1,2] -7 * x[0,2] -7 * x[1,2] -7 * x[2,2] -7 * + + + + + + + + + + + + +
+ + + + + + + + + + abs max max max max max max max
"""

run_case(globals(), EXPR, boundary=1)
