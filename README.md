# vapoursynth-zexpr

A high-performance VapourSynth expression plugin written in Zig and JIT-compiled with AsmJit.

---

## Filters

### `zexpr.Expr`

```python
zexpr.Expr(
  clips: vs.VideoNode | Sequence[vs.VideoNode],
  expr: str | list[str],
  format: int | None = None,
  opt: Literal[0, 1] = 0,
  boundary: Literal[0, 1] = 0,
) -> vs.VideoNode
```

Per-pixel RPN expression filter: one expression per plane (the last repeats, `""` copies `src0`'s plane), `opt=1` keeps integer math in wrapping `i32`, and `boundary` picks clamp (`0`) or mirror (`1`) for `clip[dx,dy]`.

### `zexpr.Select`

```python
zexpr.Select(
  clip_src: vs.VideoNode | Sequence[vs.VideoNode],
  prop_src: vs.VideoNode | Sequence[vs.VideoNode],
  expr: str | list[str],
) -> vs.VideoNode
```

Per-plane clip selector: evaluates each plane's expression on `prop_src` frame properties and returns the `clip_src` frame at the rounded, clamped index.

### `zexpr.PropExpr`

```python
zexpr.PropExpr(
  clips: vs.VideoNode | Sequence[vs.VideoNode],
  dict: Callable[[], Mapping[str, str | int | float]] | None = None,
  **kwargs: str,
) -> vs.VideoNode
```

Atomic frame-property editor in `akarin` style (`dict=lambda: {...}`, exact `int`/`float` constants, `""` deletes) or `cranexpr` style (`Prop="x.A 2 *"` keyword arguments).

---

## Expression Language

### Inputs, Spatial Access, Properties, Variables & Built-ins

| Syntax | Modes | Description |
| --- | --- | --- |
| `x`, `y`, `z`, `a`..`w` | `Expr` | Shorthand clip pixel loads (`src0`..`src25`) |
| `srcN` | `Expr` | Arbitrary 0-based clip index (`0`..`65535`) |
| `clip[dx,dy]`, `:c`, `:m` | `Expr` | Static relative pixel access with default `boundary` or explicit clamp (`:c`) / mirror (`:m`) |
| `clip[]`, `clip[]:c`, `clip[]:m` | `Expr` | Dynamic absolute pixel access (`absX absY clip[]`) with clamp (`:c`, default) or mirror (`:m`) |
| `clip.Prop` | All | Frame property read (`int`, `float`, or first byte of `data`/string; missing → `NaN` in `Expr`, `0.0` in `Select`/`PropExpr`) |
| `var!` / `var@` | All | Store / load named temporary variable (must be stored before load) |
| `X`, `Y` | `Expr` | Current pixel column / row (`0`-based) |
| `N` | All | Current frame number |
| `width`, `height` | All | Current plane dimensions in `Expr`; full clip dimensions in `Select`/`PropExpr` |
| `pi` | All | `3.14159265...` (`f32`) |

### Numeric Literals

- **Decimal integers & floats**: `42`, `-7`, `0.5`, `.5`, `1e-6`, `-0` (leading-zero integers such as `023` are base-10, not octal)
- **Hexadecimal integers**: `0xFF`, `0xFFFFFFFF` (wraps `[0x80000000, 0xFFFFFFFF]` to signed `i32` under `opt=1`)
- **C99 hexadecimal floats**: `0x1.8p2`, `0x1.fffffep+30`

### Operators

| Category | Operators |
| --- | --- |
| **Arithmetic** | `+`, `-`, `*`, `/`, `%`, `pow`, `**`, `abs`, `neg`, `absdiff`, `sqrt`, `rsqrt`, `hypot`, `sgn`, `copysign`, `lerp`, `polyvalN` / `polyevalN` (`0..1024`) |
| **Clamping, Shaping & Rounding** | `min`, `max`, `minabs`, `maxabs`, `clip`, `clamp`, `difflimit`, `linstep`, `smoothstep`, `trunc`, `floor`, `ceil`, `round` |
| **Transcendentals** | `exp`, `exp2`, `log`, `log2`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` (inline SIMD polynomials; shared `sin`/`cos` range reduction) |
| **Comparison** | `<`, `<=`, `>`, `>=`, `=` |
| **Logic & Conditional** | `and`, `or`, `xor`, `not`, `?` (`cond true_val false_val ?`, truthy iff `> 0`) |
| **Bitwise** | `bitand`, `bitor`, `bitxor`, `bitnot` (32-bit `i32`; rounds `f32` inputs to `i32`) |
| **Stack Manipulation** | `dup`, `dupN`, `swap`, `swapN`, `drop`, `dropN` |
| **Sorting & Rank Order** | `sortN` (`1..1024`; optimal sorting networks up to `sort64`, Batcher odd-even mergesort above) |
| **Selection (`Select` / `PropExpr`)** | `argminN`, `argmaxN`, `argsortN` (`1..1024`, stable index sort) |

### Operators Beyond `akarin.Expr`

| Syntax | Semantics |
| --- | --- |
| `x neg` | Negation (`-x`) |
| `x sgn` | Sign (`-1` if `x < 0`, `1` if `x > 0`, `0` otherwise) |
| `x ceil` | Ceiling (smallest integer `>= x`) |
| `a b absdiff` | Absolute difference `\|a - b\|` (uses native `u8`/`u16` `max - min` fast path) |
| `mag sgn copysign` | Magnitude of `mag` with the sign bit of `sgn` |
| `a b hypot` | Fast Euclidean norm `sqrt(a*a + b*b)` |
| `x rsqrt` | Fast reciprocal square root `1 / sqrt(clamp(x, FLT_MIN, FLT_MAX))` |
| `a b minabs` | Smaller magnitude: `\|a\| < \|b\| ? a : b` (returns `b` on ties) |
| `a b maxabs` | Larger magnitude: `\|a\| > \|b\| ? a : b` (returns `b` on ties) |
| `d1 d2 difflimit` | Signed difference limiter `clamp(d1, min(0, d2), max(0, d2))` |
| `a b t lerp` | Linear interpolation `a * (1 - t) + b * t` (exact at `t = 0` and `t = 1`) |
| `lo hi x linstep` | Clamped linear ramp `clamp((x - lo) / (hi - lo), 0, 1)` (supports reversed `lo > hi`) |
| `lo hi x smoothstep` | Smooth Hermite ramp: `t = linstep(lo, hi, x); t * t * (3 - 2 * t)` |
| `c_N ... c_0 x polyvalN` | Polynomial `c_N*x^N + ... + c_1*x + c_0` (`0 <= N <= 1024`, highest power first; alias `polyevalN`) |
| `x exp2` | Base-2 exponential `2^x` (exact at integers, clamped like `exp`) |
| `x log2` | Base-2 logarithm `log2(x)` (exact at powers of 2, `FLT_MIN`-clamped like `log`) |
| `x tan` | Tangent `tan(x)` |
| `x atan` | Arctangent `atan(x)` in `[-pi/2, pi/2]` |
| `x asin` | Arcsine `asin(x)` in `[-pi/2, pi/2]` (`NaN` for `\|x\| > 1`) |
| `x acos` | Arccosine `acos(x)` in `[0, pi]` (`NaN` for `\|x\| > 1`) |
| `y x atan2` | Two-argument arctangent `atan2(y, x)` in `[-pi, pi]` (`0` at `(0, 0)`) |

---

## Installation & Usage

Install from PyPI (autoloads into VapourSynth R75+):

```sh
pip install vapoursynth-zexpr
```

Or build from source with Zig 0.17.1:

```sh
zig build -Doptimize=ReleaseFast
```

Usage in VapourSynth:

```python
import vapoursynth as vs

core = vs.core

# Gamma curve (automatic 1D LUT on u8/u16, inline SIMD on f32)
out = core.zexpr.Expr([clip], ["x 255 / 2.2 pow 255 *"])

# 3x3 median (native u8/u16 SIMD interior fast path)
med = core.zexpr.Expr([clip], [
    "x[-1,-1] x[0,-1] x[1,-1] x[-1,0] x x[1,0] x[-1,1] x[0,1] x[1,1] sort9 drop4 swap4 drop4"
])

# Exact 32-bit integer arithmetic
u32_out = core.zexpr.Expr([clip32], ["x 0x00FF00FF bitxor 1 +"], opt=1)
```