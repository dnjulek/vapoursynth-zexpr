# Wide horizontal reach (|dx|=65) across YUV420P8 (0..2), YUV420P16 (3..5), YUV420PS (6..8).
# Exercises zexpr's arbitrary-reach SIMD interior (native u8 on p8, native u16 on p16, f32 SIMD on ps).
#
# Run from repo root, one output at a time (0/3/6=zexpr, 1/4/7=akarin [no SDE], 2/5/8=cranexpr):
#   vspipe -p -o 0 --end 4999 bench/wide-reach.py --    # YUV420P8  zexpr (-o 1 akarin, -o 2 cranexpr)
#   vspipe -p -o 3 --end 1999 bench/wide-reach.py --    # YUV420P16 zexpr (-o 4 akarin, -o 5 cranexpr)
#   vspipe -p -o 6 --end 1999 bench/wide-reach.py --    # YUV420PS  zexpr (-o 7 akarin, -o 8 cranexpr)
# Single thread: add  -a threads=1
# Output check (p8): for o in 0 1 2; do vspipe -o $o --end 9 bench/wide-reach.py - | sha256sum; done
# Needs: zig build -Doptimize=ReleaseFast; pip install VapourSynth numpy vapoursynth-akarin (+ vapoursynth-cranexpr for -o 2/5/8)

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from common import run_case

run_case(globals(), "x[-65,0] x[65,0] max x min")
