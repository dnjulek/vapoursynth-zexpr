# Adaptive dark/bright luma mask (vs-jetpack vsmasktools/spat_funcs.py adg_mask) across YUV420P8 (0..2), YUV420P16 (3..5), YUV420PS (6..8).
# Exercises frame-property loading (`x.PAverage`), degree-5 polynomial evaluation, and 2x-interleaved `pow` (never mis-cached as static LUT).
#
# Run from repo root, one output at a time (0/3/6=zexpr, 1/4/7=akarin [no SDE], 2/5/8=cranexpr):
#   vspipe -p -o 0 --end 1999 bench/adg_mask.py --    # YUV420P8  zexpr (-o 1 akarin, -o 2 cranexpr)
#   vspipe -p -o 3 --end 499  bench/adg_mask.py --    # YUV420P16 zexpr (-o 4 akarin, -o 5 cranexpr)
#   vspipe -p -o 6 --end 499  bench/adg_mask.py --    # YUV420PS  zexpr (-o 7 akarin, -o 8 cranexpr)
# Single thread: add  -a threads=1
# Output check (p8): for o in 0 1 2; do vspipe -o $o --end 9 bench/adg_mask.py - | sha256sum; done
# Needs: zig build -Doptimize=ReleaseFast; pip install VapourSynth numpy vapoursynth-akarin (+ vapoursynth-cranexpr for -o 2/5/8)

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from common import run_case


def _adg_mask(peak: str) -> str:
    return (
        f"x {peak} / 0 0.999 clamp V! "
        "1 V@ V@ V@ V@ V@ 18.188 * 45.47 - * 36.624 + * 9.466 - * 1.124 + * - "
        f"x.PAverage 2 pow 10 * pow 0 1 clamp {peak} *"
    )


run_case(
    globals(),
    {
        "p8": _adg_mask("255"),
        "p16": _adg_mask("65535"),
        "ps": _adg_mask("1.0"),
    },
    props={"PAverage": 0.5},
)
