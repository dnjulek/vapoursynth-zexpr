"""In-process speed probe used by run.sh to size each vspipe run (no vspipe launch per probe).

Usage: python bench/probe.py SCRIPT SECONDS FMT:THREADS:OUT[,OUT...] ...

For each FMT:THREADS:OUTS spec it evaluates SCRIPT with `fmt` and `threads` set (as vspipe's
`-a fmt=... -a threads=...` would), then renders each listed output for about SECONDS after one
warm-up frame and prints "OUT FPS" (or "OUT FAIL"). Only the sizing comes from here; run.sh takes the
reported timings from vspipe.
"""

from __future__ import annotations

import os
import sys
import time

import vapoursynth as vs


def main() -> int:
    script, seconds, specs = os.path.abspath(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]
    os.chdir(os.path.dirname(script))  # vspipe runs scripts from their own directory
    with open(script, encoding="utf-8") as fh:
        code = compile(fh.read(), script, "exec")
    for spec in specs:
        fmt, threads, outs = spec.split(":")
        vs.clear_outputs()
        try:
            exec(code, {"__name__": "__vapoursynth__", "__file__": script, "fmt": fmt, "threads": threads})
        except Exception as e:  # noqa: BLE001 - report every output of this spec as failed
            print(f"probe: {os.path.basename(script)} fmt={fmt}: {e}", file=sys.stderr)
            for o in outs.split(","):
                print(o, "FAIL", flush=True)
            continue
        for o in outs.split(","):
            try:
                clip = vs.get_output(int(o)).clip
                clip.get_frame(0)
                n = 0
                start = time.perf_counter()
                for _ in clip[1:].frames(close=True):
                    n += 1
                    if time.perf_counter() - start >= seconds:
                        break
                print(o, f"{n / (time.perf_counter() - start):.2f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"probe: {os.path.basename(script)} -o {o}: {e}", file=sys.stderr)
                print(o, "FAIL", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
