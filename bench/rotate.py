# Bicubic clip rotation (boundary=1) across YUV420P8 (0..2), YUV420P16 (3..5), YUV420PS (6..8).
# Expression from cranexpr's bench/rotate-clip.py (https://github.com/sgt0/cranexpr); exercises src0[] dynamic gathers and deep stack.
#
# Run from repo root, one output at a time (0/3/6=zexpr, 1/4/7=akarin [no SDE], 2/5/8=cranexpr):
#   vspipe -p -o 0 --end 199 bench/rotate.py --    # YUV420P8  zexpr (-o 1 akarin, -o 2 cranexpr)
#   vspipe -p -o 3 --end 199 bench/rotate.py --    # YUV420P16 zexpr (-o 4 akarin, -o 5 cranexpr)
#   vspipe -p -o 6 --end 199 bench/rotate.py --    # YUV420PS  zexpr (-o 7 akarin, -o 8 cranexpr)
# Single thread: add  -a threads=1
# Output check (p8): for o in 0 1 2; do vspipe -o $o --end 9 bench/rotate.py - > /tmp/o$o.raw; done && python -B bench/compare.py --dtype u8 --tol 1 /tmp/o0.raw /tmp/o1.raw /tmp/o2.raw
# Output check (ps): for o in 6 7 8; do vspipe -o $o --end 9 bench/rotate.py - > /tmp/o$o.raw; done && python -B bench/compare.py --dtype f32 --tol 3e-4 /tmp/o6.raw /tmp/o7.raw /tmp/o8.raw
# Needs: zig build -Doptimize=ReleaseFast; pip install VapourSynth numpy vapoursynth-akarin (+ vapoursynth-cranexpr for -o 2/5/8)

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from common import run_case

EXPR = """
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 N + 0.0174533 * swap51 drop width 2
/ swap50 drop height 2 / swap49 drop dup50 cos swap48 drop dup50 sin swap19 drop
X dup50 - swap41 drop Y dup49 - swap40 drop dup40 dup48 * dup40 dup20 * + swap47
drop dup40 -1 * dup19 * dup40 dup49 * + swap46 drop dup46 dup50 + swap18 drop
dup45 dup49 + swap17 drop dup17 floor swap43 drop dup16 floor swap42 drop dup17
dup43 - swap45 drop dup16 dup42 - swap44 drop dup42 1 - swap5 drop dup42 swap8
drop dup42 1 + swap7 drop dup42 2 + swap6 drop dup41 1 - swap drop dup41 swap4
drop dup41 1 + swap3 drop dup41 2 + swap2 drop dup44 1 + abs swap61 drop dup60 2
pow swap60 drop dup60 3 pow swap59 drop 7 dup59 * swap53 drop -12 dup60 * swap55
drop 0 dup53 + dup55 + 5.33333 + 6 / swap58 drop -2.33333 dup59 * swap52 drop 12
dup60 * swap54 drop -20 dup61 * swap56 drop 0 dup52 + dup54 + dup56 + 10.6667 +
6 / swap57 drop dup60 1 < dup58 dup62 2 < dup59 0 ? ? swap13 drop dup44 abs
swap61 drop dup60 2 pow swap60 drop dup60 3 pow swap59 drop 7 dup59 * swap53
drop -12 dup60 * swap55 drop 0 dup53 + dup55 + 5.33333 + 6 / swap58 drop
-2.33333 dup59 * swap52 drop 12 dup60 * swap54 drop -20 dup61 * swap56 drop 0
dup52 + dup54 + dup56 + 10.6667 + 6 / swap57 drop dup60 1 < dup58 dup62 2 <
dup59 0 ? ? swap16 drop dup44 1 - abs swap61 drop dup60 2 pow swap60 drop dup60
3 pow swap59 drop 7 dup59 * swap53 drop -12 dup60 * swap55 drop 0 dup53 + dup55
+ 5.33333 + 6 / swap58 drop -2.33333 dup59 * swap52 drop 12 dup60 * swap54 drop
-20 dup61 * swap56 drop 0 dup52 + dup54 + dup56 + 10.6667 + 6 / swap57 drop
dup60 1 < dup58 dup62 2 < dup59 0 ? ? swap15 drop dup44 2 - abs swap61 drop
dup60 2 pow swap60 drop dup60 3 pow swap59 drop 7 dup59 * swap53 drop -12 dup60
* swap55 drop 0 dup53 + dup55 + 5.33333 + 6 / swap58 drop -2.33333 dup59 *
swap52 drop 12 dup60 * swap54 drop -20 dup61 * swap56 drop 0 dup52 + dup54 +
dup56 + 10.6667 + 6 / swap57 drop dup60 1 < dup58 dup62 2 < dup59 0 ? ? swap14
drop dup43 1 + abs swap61 drop dup60 2 pow swap60 drop dup60 3 pow swap59 drop 7
dup59 * swap53 drop -12 dup60 * swap55 drop 0 dup53 + dup55 + 5.33333 + 6 /
swap58 drop -2.33333 dup59 * swap52 drop 12 dup60 * swap54 drop -20 dup61 *
swap56 drop 0 dup52 + dup54 + dup56 + 10.6667 + 6 / swap57 drop dup60 1 < dup58
dup62 2 < dup59 0 ? ? swap9 drop dup43 abs swap61 drop dup60 2 pow swap60 drop
dup60 3 pow swap59 drop 7 dup59 * swap53 drop -12 dup60 * swap55 drop 0 dup53 +
dup55 + 5.33333 + 6 / swap58 drop -2.33333 dup59 * swap52 drop 12 dup60 * swap54
drop -20 dup61 * swap56 drop 0 dup52 + dup54 + dup56 + 10.6667 + 6 / swap57 drop
dup60 1 < dup58 dup62 2 < dup59 0 ? ? swap12 drop dup43 1 - abs swap61 drop
dup60 2 pow swap60 drop dup60 3 pow swap59 drop 7 dup59 * swap53 drop -12 dup60
* swap55 drop 0 dup53 + dup55 + 5.33333 + 6 / swap58 drop -2.33333 dup59 *
swap52 drop 12 dup60 * swap54 drop -20 dup61 * swap56 drop 0 dup52 + dup54 +
dup56 + 10.6667 + 6 / swap57 drop dup60 1 < dup58 dup62 2 < dup59 0 ? ? swap11
drop dup43 2 - abs swap61 drop dup60 2 pow swap60 drop dup60 3 pow swap59 drop 7
dup59 * swap53 drop -12 dup60 * swap55 drop 0 dup53 + dup55 + 5.33333 + 6 /
swap58 drop -2.33333 dup59 * swap52 drop 12 dup60 * swap54 drop -20 dup61 *
swap56 drop 0 dup52 + dup54 + dup56 + 10.6667 + 6 / swap57 drop dup60 1 < dup58
dup62 2 < dup59 0 ? ? swap10 drop dup4 dup1 src0[] swap24 drop dup7 dup1 src0[]
swap36 drop dup6 dup1 src0[] swap32 drop dup5 dup1 src0[] swap28 drop dup4 dup4
src0[] swap27 drop dup7 dup4 src0[] swap39 drop dup6 dup4 src0[] swap35 drop
dup5 dup4 src0[] swap31 drop dup4 dup3 src0[] swap26 drop dup7 dup3 src0[]
swap38 drop dup6 dup3 src0[] swap34 drop dup5 dup3 src0[] swap30 drop dup4 dup2
src0[] swap25 drop dup7 dup2 src0[] swap37 drop dup6 dup2 src0[] swap33 drop
dup5 dup2 src0[] swap29 drop dup23 dup13 * dup36 dup17 * + dup32 dup16 * + dup28
dup15 * + swap20 drop dup26 dup13 * dup39 dup17 * + dup35 dup16 * + dup31 dup15
* + swap23 drop dup25 dup13 * dup38 dup17 * + dup34 dup16 * + dup30 dup15 * +
swap22 drop dup24 dup13 * dup37 dup17 * + dup33 dup16 * + dup29 dup15 * + swap21
drop dup19 dup9 * dup23 dup13 * + dup22 dup12 * + dup21 dup11 * + swap drop swap
drop swap drop swap drop swap drop swap drop swap drop swap drop swap drop swap
drop swap drop swap drop swap drop swap drop swap drop swap drop swap drop swap
drop swap drop swap drop swap drop swap drop swap drop swap drop swap drop swap
drop swap drop swap drop swap drop swap drop swap drop swap drop swap drop swap
drop swap drop swap drop swap drop swap drop swap drop swap drop swap drop swap
drop swap drop swap drop swap drop swap drop swap drop swap drop swap drop swap
drop swap drop swap drop swap drop swap drop swap drop swap drop swap drop swap
drop swap drop swap drop swap drop
"""

run_case(globals(), EXPR, boundary=1)
