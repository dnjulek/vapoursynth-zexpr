# 3-clip closest-to-reference pixel selector (vs-jetpack vsrgtools/blur.py side_box_blur) across YUV420P8 (0..2), YUV420P16 (3..5), YUV420PS (6..8).
# Exercises 3-clip `absdiff` comparison and conditional select on native u8/i32 interior and f32 SIMD paths.
#
# Run from repo root, one output at a time (0/3/6=zexpr, 1/4/7=akarin [no SDE], 2/5/8=cranexpr):
#   vspipe -p -o 0 --end 3999 bench/side_box_pick.py --    # YUV420P8  zexpr (-o 1 akarin, -o 2 cranexpr)
#   vspipe -p -o 3 --end 1999 bench/side_box_pick.py --    # YUV420P16 zexpr (-o 4 akarin, -o 5 cranexpr)
#   vspipe -p -o 6 --end 1999 bench/side_box_pick.py --    # YUV420PS  zexpr (-o 7 akarin, -o 8 cranexpr)
# Single thread: add  -a threads=1
# Output check (p8): for o in 0 1 2; do vspipe -o $o --end 9 bench/side_box_pick.py - | sha256sum; done
# Needs: zig build -Doptimize=ReleaseFast; pip install VapourSynth numpy vapoursynth-akarin (+ vapoursynth-cranexpr for -o 2/5/8)

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from common import run_case

run_case(globals(), "x z - abs y z - abs < x y ?", num_clips=3)
