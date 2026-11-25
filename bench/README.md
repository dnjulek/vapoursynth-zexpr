# Benchmarks

Standalone `vspipe` benchmarks comparing `zexpr.Expr`, `akarin.Expr`, and `cranexpr.Expr` on 1920×1080 random-content `YUV420` inputs (8 distinct frames looped and cached via `std.Loop`; float luma is in `[0, 1]` and float chroma in `[-0.5, 0.5]`).

Every script registers up to **9 outputs** covering 3 formats × 3 plugins (`-o N`). `cranexpr` is optional: its outputs exist only when `vapoursynth-cranexpr` is installed, and `run.sh` compares `zexpr` with `akarin` unless run with `CRANEXPR=1`.

| Format | `zexpr` | `akarin` (**no SDE**) | `cranexpr` |
|---|---|---|---|
| `YUV420P8` (`p8`) | `-o 0` | `-o 1` | `-o 2` |
| `YUV420P16` (`p16`) | `-o 3` | `-o 4` | `-o 5` |
| `YUV420PS` (`ps`) | `-o 6` | `-o 7` | `-o 8` |

## Prerequisites

Build `zexpr` in `ReleaseFast` with Zig 0.17.1, and install VapourSynth and the two reference plugins from PyPI into the environment that runs `vspipe`:

```sh
zig build -Doptimize=ReleaseFast                  # -> zig-out/lib/libzexpr.so
pip install VapourSynth numpy vapoursynth-akarin
pip install vapoursynth-cranexpr                 # optional: outputs 2/5/8 and CRANEXPR=1
```

VapourSynth autoloads the pip-installed `akarin` and `cranexpr`; `common.py` loads `zexpr` from `zig-out/lib/libzexpr.so`. Environment variables select other builds (relative paths are taken from the repository root): `ZEXPR_LIB` for another `zexpr` build, and `AKARIN_LIB` / `CRANEXPR_LIB` for locally built reference plugins when the pip packages are not installed. A script stops with an error when an autoloaded plugin differs from the library its variable names, since VapourSynth cannot load a second copy. If `vapoursynth-zexpr` is pip-installed, VapourSynth autoloads it and the benchmarks time that copy: `run.sh` prints a warning above the table, saying whether a local `zig-out` build exists and is not being measured. To time a local build, `pip uninstall vapoursynth-zexpr`.

## Running

Run **one benchmark at a time** with nothing else active (no parallel runs, builds, fuzzers, or SDE/QEMU sessions):

```sh
# Single case (default thread count)
vspipe -p -o 0 --end 1999 bench/median3x3.py --   # YUV420P8  zexpr (-o 1 akarin, -o 2 cranexpr)
vspipe -p -o 3 --end 999  bench/median3x3.py --   # YUV420P16 zexpr (-o 4 akarin, -o 5 cranexpr)
vspipe -p -o 6 --end 999  bench/median3x3.py --   # YUV420PS  zexpr (-o 7 akarin, -o 8 cranexpr)

# Single-threaded
vspipe -p -a threads=1 -o 0 --end 1999 bench/median3x3.py --

# Sequential runner across all cases and formats (or a subset)
bash bench/run.sh                                  # zexpr vs akarin; p8 at default threads, p16/ps single-threaded
CRANEXPR=1 bash bench/run.sh                       # also time cranexpr
THREADS=1 bash bench/run.sh                        # one thread for every cell
THREADS=auto bash bench/run.sh                     # VapourSynth default threads for every cell
FORMATS="p8" bash bench/run.sh
THREADS=1 FORMATS="p8 ps" bash bench/run.sh median3x3.py gamma.py
```

`run.sh` picks threads per format: `p8` runs at VapourSynth's default thread count, while `p16` and `ps` run single-threaded, because at full thread counts their 2–4× larger frames saturate DRAM bandwidth and all three plugins converge on the same memory-bound fps. Each row of the table names its thread count (`p16, 1T`). Run lengths are sized on the fly: `bench/probe.py` first times every output of a script in-process for ~0.25 s (one Python process per thread count, no `vspipe` launch), then each plugin gets **one** `vspipe` run sized to last about `BENCH_SECONDS` (default **1.5 s**). `run.sh` passes `-a fmt=p8|p16|ps`, so a script builds only that format's outputs and generates its random source frames during script evaluation, before `vspipe` starts timing; the reported fps therefore covers only the expression. A full default table takes about 3 minutes on a 16-thread desktop CPU. `FRAMES=N` forces a fixed count instead, and `BENCH_LOG=FILE` appends each measured run's frames, seconds and fps. Single `vspipe` commands use VapourSynth's default threads unless given `-a threads=N`.

Note: `run_case` constructs all nine outputs when a script loads, so in a custom script where any of the three plugins rejects an expression, `run.sh` reports `FAIL` for all three cells of that row.

## Verifying Output Parity

Integer cases with exact arithmetic match bit-for-bit across all three plugins:

```sh
for o in 0 1 2; do vspipe -o $o --end 9 bench/median3x3.py - | sha256sum; done
```

Float cases and cases with polynomial/FMA/reciprocal rounding differences can be checked with `bench/compare.py`:

```sh
for o in 6 7 8; do vspipe -o $o --end 9 bench/kirsch.py - > /tmp/o$o.raw; done
python -B bench/compare.py --dtype f32 --tol 3e-5 /tmp/o6.raw /tmp/o7.raw /tmp/o8.raw
```

Measured parity tolerances (`--end 9`, `zexpr` vs `akarin` / `cranexpr`; `=` means bit-identical):

| Case | `p8` (LSB) | `p16` (LSB) | `ps` (max abs diff) |
|---|---:|---:|---:|
| `median3x3`, `wide-reach` | `=` | `=` | `=` |
| `box3x3` | `=` | `=` | `<= 2e-7` |
| `gamma`, `exp` | `=` | `<= 1` | `<= 5e-7` |
| `kirsch` | `=` | `=` | `<= 3e-5` |
| `rotate` | `<= 1` | `<= 17` | `<= 3e-4` |
| `mandelbrot` | `<= 0.00005%` boundary flips | `<= 0.00005%` boundary flips | `<= 0.00005%` boundary flips |
| `morpho`, `deblock`, `pad_repeat`, `rp_mode19`, `rg_mode2`, `side_box_pick` | `=` | `=` | `=` |
| `fine_sharp` | `=` | `<= 1` | `<= 1e-4` |
| `adg_mask` | `=` | `<= 1` | `<= 6e-7` |
| `sobel` | `=` | `=` | `<= 1e-6` (`akarin`; `cranexpr` `=`) |

## Cases

| Script | Clips | Expression / Workload | Exercised Paths in `zexpr` |
|---|---|---|---|
| `deblock.py` | 1 | `X 8 % 7 % Y 8 % 7 % and 0 peak ?` (vs-jetpack `deblock_qed` mask) | Exact `X % 8`, hoisted periodic `X % C`, row-invariant `u8`/`u16` pre-pack (`p8`/`p16`), `f32` SIMD (`ps`) |
| `pad_repeat.py` | 1 | Repeat-border padding (`vstools/utils/misc.py`, 16px Y / 8px UV) | Frame/row LICM (hoisted corners & edge loads), contiguous `abs_row_ptrs` row loads, 7 nested `?` selects |
| `rg_mode2.py` | 1 | 3×3 RemoveGrain mode 2 (`vsrgtools`, 8-neighbor `sort8` clamp) | Native `u8`/`u16` SIMD sorting network (`p8`/`p16`), `f32` SIMD (`ps`) |
| `rp_mode19.py` | 2 | 2-clip 3×3 Repair mode 19 (`vsrgtools`, 8-neighbor `absdiff` min + clamp) | Native `u8`/`u16` SIMD interior (`p8`/`p16`), `f32` SIMD (`ps`) |
| `gamma.py` | 1 | `2.2 pow` (`p8`/`p16` scaled; `ps`: `x abs 2.2 pow`) | SIMD in-register 1D `u8` LUT (`p8`), 1D `u16` LUT (`p16`), SIMD `pow` (`ps`) |
| `kirsch.py` | 1 | 5×5 8-direction Kirsch (`boundary=1`) | 5×5 mirror spatial + deep DAG |
| `morpho.py` | 1 | 3×3 morphological gradient: 9-tap `max` − 9-tap `min` (vs-jetpack `Morpho.gradient`; the inner/outer hats are one `min`/`max` chain fewer) | Native `u8`/`u16` saturating subtract (`p8`/`p16`), `f32` SIMD (`ps`) |
| `median3x3.py` | 1 | 3×3 `sort9` median | Native `u8` SIMD interior (`p8`), native `u16` SIMD interior (`p16`), `f32` SIMD (`ps`) |
| `fine_sharp.py` | 2 | 2-clip nonlinear sharpening (`vsrgtools/sharp.py`, `pow` + div + `?`) | 2D `u8x8` LUT (`p8`), 2×-interleaved transcendental + div DAG (`p16`/`ps`) |
| `side_box_pick.py` | 3 | 3-clip closest-to-reference pixel selector (`vsrgtools/blur.py`) | Native `u8`/`i32` 3-clip `absdiff` + select (`p8`/`p16`), `f32` SIMD (`ps`) |
| `sobel.py` | 1 | 3×3 Sobel `\|Gx\| + \|Gy\|` | `i16` 2×-unrolled interior (`p8`), `f32` 2×-unrolled SIMD (`p16`/`ps`) |
| `adg_mask.py` | 1 | Adaptive dark/bright mask (`vsmasktools`, `x.PAverage` prop + degree-5 poly + `pow`) | Frame-property loading (`x.PAverage`), Horner polynomial, 2×-interleaved `pow` (not static LUT) |
| `exp.py` | 1 | `exp` (`p8`/`p16` scaled; `ps`: `x exp`) | 1D LUT (`p8`/`p16`), zero-spill Estrin polynomial + const-mem FMA (`ps`) |
| `rotate.py` | 1 | Bicubic clip rotation (`boundary=1`) | `src0[]` dynamic gathers + deep stack |
| `mandelbrot.py` | 1 | 23-iteration Mandelbrot fractal zoomer (`N`/`X`/`Y`/`width`/`height` + neutral chroma) | Frame-invariant `1.01 N pow` hoisting, deep 23-step FMA/`min` chains, row-invariant chroma fills |
| `box3x3.py` | 1 | 9-tap box sum, `9 /` | Spatial 3×3, `f32` SIMD (`9` not power-of-two) |
| `wide-reach.py` | 1 | `x[-65,0] x[65,0] max x min` | Wide horizontal reach (`>64` px) + native `u8`/`u16` interior (`p8`/`p16`), `f32` SIMD (`ps`) |

## Files

- `common.py` — shared plugin loader (`setup`), cached 8-frame random `YUV420` generator (`random_source`), and `run_case` (`-o 0..8`).
- `compare.py` — raw frame output tolerance checker.
- `probe.py` — in-process speed probe `run.sh` uses to size each `vspipe` run (timings themselves always come from `vspipe`).
- `run.sh` — sequential `vspipe` runner printing a comparison table across selected scripts and `FORMATS`.
