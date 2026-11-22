#!/usr/bin/env python3
"""Compare raw vspipe output files (output 0 vs outputs 1, 2, ...) within a tolerance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DTYPES = {
    "u8": np.uint8,
    "u16": np.uint16,
    "f16": np.float16,
    "f32": np.float32,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare raw vspipe outputs against output 0 within a tolerance.")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="f32")
    ap.add_argument("--tol", type=float, default=1e-4, help="Maximum allowed absolute difference")
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    if len(args.files) < 2:
        ap.error("Need at least 2 raw files to compare")

    dt = DTYPES[args.dtype]
    ref = np.fromfile(args.files[0], dtype=dt)
    if ref.size == 0:
        print(f"ERROR: empty file {args.files[0]}", file=sys.stderr)
        return 1

    ok = True
    for other_path in args.files[1:]:
        other = np.fromfile(other_path, dtype=dt)
        if other.size != ref.size:
            print(f"FAIL {other_path}: size {other.size} != {ref.size}", file=sys.stderr)
            ok = False
            continue
        if np.issubdtype(dt, np.floating):
            nan_r = np.isnan(ref)
            nan_o = np.isnan(other)
            if not np.array_equal(nan_r, nan_o):
                print(f"FAIL {other_path}: NaN positions differ", file=sys.stderr)
                ok = False
                continue
            mask = ~nan_r
            diff = np.abs(ref[mask].astype(np.float64) - other[mask].astype(np.float64))
        else:
            diff = np.abs(ref.astype(np.int64) - other.astype(np.int64))
        max_diff = float(diff.max()) if diff.size else 0.0
        n_diff = int(np.count_nonzero(diff > 0))
        status = "OK" if max_diff <= args.tol else "FAIL"
        print(f"{status:4s} {args.files[0].name} vs {other_path.name}: max_abs_diff={max_diff:.6g} (tol={args.tol:g}, differing={n_diff}/{ref.size})")
        if max_diff > args.tol:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
