//! Target-parameterized pure-Zig scalar interpreter and semantic oracle.
//!
//! Single semantic source of truth per target ISA tier (`use_fma`, `isa_family`).
//! Shared arithmetic helpers (`minF32`, `maxF32`, `roundEvenF32`, `cvtRoundF32ToI32`,
//! `cvtTruncF32ToI32`, `modF32`, and polynomial transcendentals) are reused directly
//! by `lower.zig` and `optimize.zig`.

const std = @import("std");
const ast = @import("ast.zig");
const cpu_mod = @import("../jit/cpu.zig");

pub const IsaFamily = cpu_mod.IsaFamily;

pub const TargetProfile = struct {
    use_fma: bool,
    isa_family: IsaFamily,

    pub fn fromCpu(cpu: cpu_mod.CpuInfo) TargetProfile {
        return .{
            .use_fma = cpu.has_fma,
            .isa_family = switch (cpu.isa) {
                .neon => .aarch64,
                .scalar => IsaFamily.host(),
                else => .x86,
            },
        };
    }

    pub fn hostDefault() TargetProfile {
        const cpu = cpu_mod.detectWithOverride(null);
        return fromCpu(cpu);
    }
};

/// True iff `v` is an IEEE-754 binary32 signaling NaN (quiet bit 22 == 0).
inline fn isSignalingNanF32(v: f32) bool {
    const bits: u32 = @bitCast(v);
    return (bits & 0x7F80_0000) == 0x7F80_0000 and (bits & 0x007F_FFFF) != 0 and (bits & 0x0040_0000) == 0;
}

inline fn quietNanF32(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    return @bitCast(bits | 0x0040_0000);
}

/// ISA-native `min(a, b)` matching `minps` on x86 and `fminnm` on AArch64 (U2, W2).
/// Never uses Zig's `@min`.
pub inline fn minF32(profile: TargetProfile, a: f32, b: f32) f32 {
    return switch (profile.isa_family) {
        .x86 => if (a < b) a else b,
        .aarch64 => blk: {
            // IEEE-754-2008 minNum / AArch64 fminnm: signaling NaN propagates as quieted NaN;
            // quiet NaN loses to the non-NaN operand.
            if (isSignalingNanF32(a)) break :blk quietNanF32(a);
            if (isSignalingNanF32(b)) break :blk quietNanF32(b);
            if (std.math.isNan(a)) break :blk b;
            if (std.math.isNan(b)) break :blk a;
            if (a == 0.0 and b == 0.0) {
                const sa: u32 = @bitCast(a);
                const sb: u32 = @bitCast(b);
                break :blk if (((sa | sb) & 0x8000_0000) != 0) -0.0 else 0.0;
            }
            break :blk if (a < b) a else b;
        },
    };
}

/// ISA-native `max(a, b)` matching `maxps` on x86 and `fmaxnm` on AArch64 (U2, W2).
/// Never uses Zig's `@max`.
pub inline fn maxF32(profile: TargetProfile, a: f32, b: f32) f32 {
    return switch (profile.isa_family) {
        .x86 => if (a > b) a else b,
        .aarch64 => blk: {
            if (isSignalingNanF32(a)) break :blk quietNanF32(a);
            if (isSignalingNanF32(b)) break :blk quietNanF32(b);
            if (std.math.isNan(a)) break :blk b;
            if (std.math.isNan(b)) break :blk a;
            if (a == 0.0 and b == 0.0) {
                const sa: u32 = @bitCast(a);
                const sb: u32 = @bitCast(b);
                break :blk if (((sa & sb) & 0x8000_0000) != 0) -0.0 else 0.0;
            }
            break :blk if (a > b) a else b;
        },
    };
}

/// Exact IEEE round-to-nearest-even in `f32`, preserving `-0.0`, `|x| >= 2^23`, `±inf`, and `NaN` (D9, X2).
pub inline fn roundEvenF32(x: f32) f32 {
    if (std.math.isNan(x) or @abs(x) >= 8388608.0) return x;
    const s = std.math.copysign(@as(f32, 8388608.0), x);
    const r = (x + s) - s;
    return std.math.copysign(r, x);
}

/// ISA-native half-to-even `f32 -> i32` matching `cvtps2dq` (x86) and `fcvtns` (AArch64) (X2, T7).
pub inline fn cvtRoundF32ToI32(profile: TargetProfile, x: f32) i32 {
    return cvtRoundedF32ToI32(profile, roundEvenF32(x));
}

/// ISA-native truncating `f32 -> i32` matching `cvttps2dq` (x86) and `fcvtzs` (AArch64) (X2, T7).
pub inline fn cvtTruncF32ToI32(profile: TargetProfile, x: f32) i32 {
    return cvtRoundedF32ToI32(profile, if (std.math.isNan(x)) x else @trunc(x));
}

inline fn cvtRoundedF32ToI32(profile: TargetProfile, r: f32) i32 {
    return switch (profile.isa_family) {
        .x86 => blk: {
            if (std.math.isNan(r) or r < -2147483648.0 or r >= 2147483648.0) {
                break :blk std.math.minInt(i32);
            }
            break :blk @intFromFloat(r);
        },
        .aarch64 => blk: {
            if (std.math.isNan(r)) break :blk 0;
            if (r <= -2147483648.0) break :blk std.math.minInt(i32);
            if (r >= 2147483648.0) break :blk std.math.maxInt(i32);
            break :blk @intFromFloat(r);
        },
    };
}

/// Shared unfused D3 modulo helper (`a - trunc(a / b) * b`). Never calls `@rem` or `@mod` (Q3).
pub inline fn modF32(a: f32, b: f32) f32 {
    const q = @trunc(a / b);
    return a - q * b;
}

/// FMA-or-separate `a * b + c` matching `UniOpVVVV::kMAddF32`.
pub inline fn fmaF32(profile: TargetProfile, a: f32, b: f32, c: f32) f32 {
    return if (profile.use_fma) @mulAdd(f32, a, b, c) else a * b + c;
}

/// FMA-or-separate `a * b - c` matching `UniOpVVVV::kMSubF32`.
pub inline fn fmsF32(profile: TargetProfile, a: f32, b: f32, c: f32) f32 {
    return if (profile.use_fma) @mulAdd(f32, a, b, -c) else a * b - c;
}

/// FMA-or-separate `c - a * b` matching `UniOpVVVV::kNMAddF32`.
pub inline fn fnmaF32(profile: TargetProfile, a: f32, b: f32, c: f32) f32 {
    return if (profile.use_fma) @mulAdd(f32, -a, b, c) else c - a * b;
}

/// FMA-or-separate `-(a * b) - c` matching `UniOpVVVV::kNMSubF32`.
pub inline fn fnmsF32(profile: TargetProfile, a: f32, b: f32, c: f32) f32 {
    return if (profile.use_fma) @mulAdd(f32, -a, b, -c) else -(a * b) - c;
}

// ============================================================================
// Fast SIMD-matched polynomial transcendentals (identical sequence in math.zig)
// ============================================================================

pub const exp_hi_clamp: f32 = 88.3762626647949;
pub const exp_lo_clamp: f32 = -88.3762626647949;
pub const log2e_f32: f32 = 1.4426950408889634;
pub const ln2_hi_f32: f32 = 0.693145751953125; // 0x3f317200 (exact when * k for |k| <= 128)
pub const ln2_lo_f32: f32 = 1.428606765330187e-6; // 0x35bfbe8e
pub const flt_min_f32: f32 = @bitCast(@as(u32, 0x0080_0000));
pub const flt_max_f32: f32 = std.math.floatMax(f32);

/// Clamped Cody-Waite `exp(x)` with single `(k + 127) << 23` injection and Estrin polynomial (U3, W1).
pub fn expApprox(profile: TargetProfile, x: f32) f32 {
    const xc = maxF32(profile, minF32(profile, x, exp_hi_clamp), exp_lo_clamp);
    const ki = cvtRoundF32ToI32(profile, xc * log2e_f32);
    const k: f32 = @floatFromInt(ki);
    var r = fnmaF32(profile, k, ln2_hi_f32, xc);
    r = fnmaF32(profile, k, ln2_lo_f32, r);

    // Estrin's scheme for degree-6 Taylor polynomial (3 FMA levels vs 6 in Horner).
    const r2 = r * r;
    const r4 = r2 * r2;
    const p01 = 1.0 + r;
    const p23 = fmaF32(profile, r, 1.6666667e-1, 0.5);
    const p45 = fmaF32(profile, r, 8.3333338e-3, 4.1666668e-2);
    const p03 = fmaF32(profile, p23, r2, p01);
    const p46 = fmaF32(profile, r2, 1.3888889e-3, p45);
    const p = fmaF32(profile, p46, r4, p03);

    const scale_bits: u32 = @bitCast((ki +% 127) << 23);
    const scale: f32 = @bitCast(scale_bits);
    return p * scale;
}

// Clamped exact-at-integer deg-5 minimax `exp2(x)` (`max_rel <= 2.9e-7`, bit-exact at integers k in [-126, 127]).
pub const exp2_hi_clamp: f32 = @bitCast(@as(u32, 0x42FE_FFFF)); // 127.49999237060547 (< 127.5)
pub const exp2_lo_clamp: f32 = @bitCast(@as(u32, 0xC2FD_FFFF)); // -126.99999237060547 (rounds to -127 -> 0.0)
pub const exp2_c1: f32 = 0.6931471824645996;
pub const exp2_c2: f32 = 0.2402234971523285;
pub const exp2_c3: f32 = 0.05550333485007286;
pub const exp2_c4: f32 = 0.009666292928159237;
pub const exp2_c5: f32 = 0.0013400345342233777;

pub fn exp2Approx(profile: TargetProfile, x: f32) f32 {
    const xc = maxF32(profile, minF32(profile, x, exp2_hi_clamp), exp2_lo_clamp);
    const ki = cvtRoundF32ToI32(profile, xc);
    const k: f32 = @floatFromInt(ki);
    const r = xc - k;

    var p = fmaF32(profile, r, exp2_c5, exp2_c4);
    p = fmaF32(profile, p, r, exp2_c3);
    p = fmaF32(profile, p, r, exp2_c2);
    p = fmaF32(profile, p, r, exp2_c1);
    p = fmaF32(profile, p, r, 1.0);

    const scale_bits: u32 = @bitCast((ki +% 127) << 23);
    const scale: f32 = @bitCast(scale_bits);
    return p * scale;
}

// 4-term minimax odd polynomial for `log2((1+u)/(1-u)) / u` on `|u| <= 3 - 2*sqrt(2)` (max_abs <= 2.3e-7, exact at 2^k).
pub const log2_u_c0: f32 = log_u_c0 * log2e_f32;
pub const log2_u_c1: f32 = log_u_c1 * log2e_f32;
pub const log2_u_c2: f32 = log_u_c2 * log2e_f32;
pub const log2_u_c3: f32 = log_u_c3 * log2e_f32;

pub fn log2Approx(profile: TargetProfile, x: f32) f32 {
    const xc = maxF32(profile, x, flt_min_f32);
    const xi: i32 = @bitCast(xc);
    const adj = xi +% 0x004A_FB0D;
    const e: f32 = @floatFromInt((adj >> 23) -% 127);
    const m_bits: u32 = @bitCast((adj & 0x007F_FFFF) +% 0x3F35_04F3);
    const m: f32 = @bitCast(m_bits);

    const u = (m - 1.0) / (m + 1.0);
    const u_sq = u * u;
    var p = fmaF32(profile, u_sq, log2_u_c3, log2_u_c2);
    p = fmaF32(profile, p, u_sq, log2_u_c1);
    p = fmaF32(profile, p, u_sq, log2_u_c0);
    return fmaF32(profile, p, u, e);
}

pub inline fn copysignF32(mag: f32, sgn: f32) f32 {
    const a_bits: u32 = @bitCast(mag);
    const b_bits: u32 = @bitCast(sgn);
    return @bitCast((a_bits & 0x7FFF_FFFF) | (b_bits & 0x8000_0000));
}

// 4-term minimax odd polynomial for `ln((1+u)/(1-u)) / u` on `|u| <= 3 - 2*sqrt(2)` (max_rel <= 2.3e-7).
pub const log_u_c0: f32 = 2.0;
pub const log_u_c1: f32 = 0.6666679382324219;
pub const log_u_c2: f32 = 0.3997650742530823;
pub const log_u_c3: f32 = 0.298920601606369;

/// Fast-clamped `log(x)` with `kMaxF32(x, FLT_MIN)` and 3-FMA minimax odd polynomial (D13, U3).
pub fn logApprox(profile: TargetProfile, x: f32) f32 {
    const xc = maxF32(profile, x, flt_min_f32);
    const xi: i32 = @bitCast(xc);
    // Shift by (0x3F800000 - 0x3F3504F3) so mantissa falls in [sqrt(0.5), sqrt(2)).
    const adj = xi +% 0x004A_FB0D;
    const e: f32 = @floatFromInt((adj >> 23) -% 127);
    const m_bits: u32 = @bitCast((adj & 0x007F_FFFF) +% 0x3F35_04F3);
    const m: f32 = @bitCast(m_bits);

    const u = (m - 1.0) / (m + 1.0);
    const u_sq = u * u;
    var p = fmaF32(profile, u_sq, log_u_c3, log_u_c2);
    p = fmaF32(profile, p, u_sq, log_u_c1);
    p = fmaF32(profile, p, u_sq, log_u_c0);
    const ln_m = p * u;
    return fmaF32(profile, e, ln2_hi_f32, fmaF32(profile, e, ln2_lo_f32, ln_m));
}

// 4-part Cody-Waite pi decomposition (15+ trailing zero mantissa bits in p1..p3 -> exact k*pi for |x| <= 1e5).
pub const inv_pi_f32: f32 = 0.3183098861837907;
pub const pi1_f32: f32 = @bitCast(@as(u32, 0x4049_0000)); // 3.140625
pub const pi2_f32: f32 = @bitCast(@as(u32, 0x3A7D_8000)); // 9.670257568359375e-4
pub const pi3_f32: f32 = @bitCast(@as(u32, 0x3528_8000)); // 6.277114152908325e-7
pub const pi4_f32: f32 = @bitCast(@as(u32, 0x2F05_A0B5)); // 1.215420125655342e-10

inline fn reducePi(profile: TargetProfile, x: f32) struct { r: f32, r2: f32, sign_mask: u32 } {
    const ki_s = cvtRoundF32ToI32(profile, x * inv_pi_f32);
    const k: f32 = @floatFromInt(ki_s);
    var r = fnmaF32(profile, k, pi1_f32, x);
    r = fnmaF32(profile, k, pi2_f32, r);
    r = fnmaF32(profile, k, pi3_f32, r);
    r = fnmaF32(profile, k, pi4_f32, r);
    const ki: u32 = @bitCast(ki_s);
    return .{ .r = r, .r2 = r * r, .sign_mask = (ki & 1) << 31 };
}

inline fn sinPoly(profile: TargetProfile, r: f32, r2: f32) f32 {
    const r4 = r2 * r2;
    const p01 = fmaF32(profile, r2, 8.3333338e-3, -1.6666667e-1);
    const p23 = fmaF32(profile, r2, 2.7557319e-6, -1.9841270e-4);
    const p24 = fmaF32(profile, r4, -2.5052108e-8, p23);
    const p = fmaF32(profile, p24, r4, p01);
    return fmaF32(profile, p * r2, r, r);
}

inline fn cosPoly(profile: TargetProfile, r2: f32) f32 {
    const r4 = r2 * r2;
    const p01 = fmaF32(profile, r2, 4.1666668e-2, -0.5);
    const p23 = fmaF32(profile, r2, 2.4801587e-5, -1.3888889e-3);
    const p24 = fmaF32(profile, r4, -2.7557319e-7, p23);
    const p = fmaF32(profile, p24, r4, p01);
    return fmaF32(profile, p, r2, 1.0);
}

pub fn sinApprox(profile: TargetProfile, x: f32) f32 {
    const x_bits: u32 = @bitCast(x);
    const x_sign = x_bits & 0x8000_0000;
    const ax: f32 = @bitCast(x_bits & 0x7FFF_FFFF);
    const red = reducePi(profile, ax);
    const s = sinPoly(profile, red.r, red.r2);
    const bits: u32 = @bitCast(s);
    return @bitCast(bits ^ red.sign_mask ^ x_sign);
}

pub fn cosApprox(profile: TargetProfile, x: f32) f32 {
    const ax: f32 = @abs(x);
    const red = reducePi(profile, ax);
    const c = cosPoly(profile, red.r2);
    const bits: u32 = @bitCast(c);
    return @bitCast(bits ^ red.sign_mask);
}

// 4-part Cody-Waite pi/2 decomposition for `tan` (<= 8 significant bits in half_pi1..3
// so k * half_pi{1..3} stays exact without FMA for k <= 63662, i.e. |x| <= 1e5, K2).
pub const inv_half_pi_f32: f32 = 2.0 / std.math.pi;
pub const half_pi1_f32: f32 = @bitCast(@as(u32, 0x3FC9_0000)); // 1.5703125
pub const half_pi2_f32: f32 = @bitCast(@as(u32, 0x39FD_0000)); // 4.825592041015625e-4 (8 bits)
pub const half_pi3_f32: f32 = @bitCast(@as(u32, 0x35AA_0000)); // 1.2665987014770508e-6 (8 bits)
pub const half_pi4_f32: f32 = @bitCast(@as(u32, 0x3088_5A31)); // 9.92093629470503e-10

pub fn tanApprox(profile: TargetProfile, x: f32) f32 {
    const x_bits: u32 = @bitCast(x);
    const x_sign = x_bits & 0x8000_0000;
    const ax: f32 = @bitCast(x_bits & 0x7FFF_FFFF);
    const ki_s = cvtRoundF32ToI32(profile, ax * inv_half_pi_f32);
    const k: f32 = @floatFromInt(ki_s);
    var r = fnmaF32(profile, k, half_pi1_f32, ax);
    r = fnmaF32(profile, k, half_pi2_f32, r);
    r = fnmaF32(profile, k, half_pi3_f32, r);
    r = fnmaF32(profile, k, half_pi4_f32, r);
    const r2 = r * r;
    const s = sinPoly(profile, r, r2);
    const c = cosPoly(profile, r2);
    const ki: u32 = @bitCast(ki_s);
    const is_odd = (ki & 1) != 0;
    const num = if (is_odd) c else s;
    const den = if (is_odd) s else c;
    const odd_sign: u32 = (ki & 1) << 31;
    const q = num / den;
    const q_bits: u32 = @bitCast(q);
    return @bitCast(q_bits ^ odd_sign ^ x_sign);
}

// Cephes 2-breakpoint `atan(x)` constants (`|z| <= tan(pi/8)`, 4-coeff Horner, max_abs_err <= 1.4e-7).
pub const atan_tan_pi8_f32: f32 = 0.4142135623730950;
pub const atan_tan_3pi8_f32: f32 = 2.4142135623730950;
pub const atan_c0_f32: f32 = -3.33329491539e-1;
pub const atan_c1_f32: f32 = 1.99777106478e-1;
pub const atan_c2_f32: f32 = -1.38776856032e-1;
pub const atan_c3_f32: f32 = 8.05374449538e-2;

pub fn atanApprox(profile: TargetProfile, x: f32) f32 {
    const x_bits: u32 = @bitCast(x);
    const x_sign = x_bits & 0x8000_0000;
    const ax: f32 = @bitCast(x_bits & 0x7FFF_FFFF);
    const m_hi = ax > atan_tan_3pi8_f32;
    const m_mid = ax > atan_tan_pi8_f32;
    const num = if (m_hi) @as(f32, -1.0) else if (m_mid) ax - 1.0 else ax;
    const den = if (m_hi) ax else if (m_mid) ax + 1.0 else @as(f32, 1.0);
    const z = num / den;
    const z2 = z * z;
    var p = fmaF32(profile, z2, atan_c3_f32, atan_c2_f32);
    p = fmaF32(profile, p, z2, atan_c1_f32);
    p = fmaF32(profile, p, z2, atan_c0_f32);
    const r = fmaF32(profile, p * z2, z, z);
    const off: f32 = if (m_hi) std.math.pi * 0.5 else if (m_mid) std.math.pi * 0.25 else 0.0;
    const res_bits: u32 = @bitCast(off + r);
    return @bitCast(res_bits ^ x_sign);
}

// Direct Cephes-style 4-coeff minimax polynomial for `(asin(sqrt(z))/sqrt(z) - 1) / z` on `[0, 0.25]`
// (zero divisions, 1 sqrt, 3-FMA Horner completely hidden behind sqrt; max_abs_err <= 4.4e-7 asin, <= 5.9e-7 acos).
pub const asin_c0_f32: f32 = 0.16666563;
pub const asin_c1_f32: f32 = 0.07513281;
pub const asin_c2_f32: f32 = 0.042074867;
pub const asin_c3_f32: f32 = 0.04546545;

inline fn sqrtRawF32(z: f32) f32 {
    if (std.math.isNan(z) or z < 0.0) return std.math.nan(f32);
    return @sqrt(z);
}

pub fn asinApprox(profile: TargetProfile, x: f32) f32 {
    const x_bits: u32 = @bitCast(x);
    const x_sign = x_bits & 0x8000_0000;
    const ax: f32 = @bitCast(x_bits & 0x7FFF_FFFF);
    const hi = ax > 0.5;
    const z = if (hi) fnmaF32(profile, ax, 0.5, 0.5) else ax * ax;
    const s = sqrtRawF32(z);
    var p = fmaF32(profile, z, asin_c3_f32, asin_c2_f32);
    p = fmaF32(profile, p, z, asin_c1_f32);
    p = fmaF32(profile, p, z, asin_c0_f32);
    const t = if (hi) s else ax;
    const r = fmaF32(profile, t * z, p, t);
    const res = if (hi) (std.math.pi * 0.5) - (r + r) else r;
    const res_bits: u32 = @bitCast(res);
    return @bitCast(res_bits ^ x_sign);
}

pub fn acosApprox(profile: TargetProfile, x: f32) f32 {
    const x_bits: u32 = @bitCast(x);
    const ax: f32 = @bitCast(x_bits & 0x7FFF_FFFF);
    const hi = ax > 0.5;
    const z = if (hi) fnmaF32(profile, ax, 0.5, 0.5) else ax * ax;
    const s = sqrtRawF32(z);
    var p = fmaF32(profile, z, asin_c3_f32, asin_c2_f32);
    p = fmaF32(profile, p, z, asin_c1_f32);
    p = fmaF32(profile, p, z, asin_c0_f32);
    const t = if (hi) s else x;
    const r = fmaF32(profile, t * z, p, t);
    const r2 = r + r;
    const x_neg = (x_bits & 0x8000_0000) != 0;
    const hi_val = if (x_neg) std.math.pi - r2 else r2;
    return if (hi) hi_val else (std.math.pi * 0.5) - r;
}

pub fn atan2Approx(profile: TargetProfile, y: f32, x: f32) f32 {
    const ax = @abs(x);
    const ay = @abs(y);
    const mx = maxF32(profile, ax, ay);
    const mn = minF32(profile, ax, ay);
    // Avoid 0/0 when both are 0: mask result to 0 when mx == 0.
    const ratio = if (mx > 0.0) mn / mx else 0.0;
    const r2 = ratio * ratio;

    // SLEEF-style degree-15 minimax polynomial for atan(t) on [0, 1].
    var p: f32 = 0.0028236389625817537;
    p = fmaF32(profile, p, r2, -0.015956902876496315);
    p = fmaF32(profile, p, r2, 0.042504988610744476);
    p = fmaF32(profile, p, r2, -0.07489009201526642);
    p = fmaF32(profile, p, r2, 0.1063479334115982);
    p = fmaF32(profile, p, r2, -0.1420273631811142);
    p = fmaF32(profile, p, r2, 0.19992695748806);
    p = fmaF32(profile, p, r2, -0.33333101868629456);
    var res = fmaF32(profile, p * r2, ratio, ratio);

    // If |y| > |x|, map via pi/2 - atan(|x|/|y|).
    if (ay > ax) {
        res = (std.math.pi * 0.5) - res;
    }
    // If x < 0, map via pi - res.
    if (x < 0.0) {
        res = std.math.pi - res;
    }
    // If both x == 0 and y == 0, return 0.0.
    if (!(mx > 0.0)) {
        res = 0.0;
    }
    // Apply y's sign bit.
    const y_sign: u32 = @as(u32, @bitCast(y)) & 0x8000_0000;
    const res_bits: u32 = @bitCast(res);
    return @bitCast(res_bits ^ y_sign);
}

// ============================================================================
// Coordinate clamping / mirroring (D6, D7) & sample load/store (D4)
// ============================================================================

/// Shared compile-time relative offset clamp to `[-dim, dim]` (exact for both clamp and D6 mirror,
/// prevents `minInt(i32)` negation overflow and keeps reach bounded by plane dimensions).
pub inline fn clampRelOffset(d: i32, dim: i32) i32 {
    return std.math.clamp(d, -dim, dim);
}

pub inline fn clampCoord(idx: i32, dim: i32) usize {
    if (dim <= 1) return 0;
    if (idx <= 0) return 0;
    if (idx >= dim - 1) return @intCast(dim - 1);
    return @intCast(idx);
}

/// D6 single-reflection mirror: assumes `idx` already has `eff_offset = clamp(d, -dim, dim)`
/// or float-clamped `[-dim, 2*dim - 1]`.
pub inline fn mirrorCoord(idx: i32, dim: i32) usize {
    if (dim <= 1) return 0;
    var v = idx;
    if (v < 0) v = -1 - v;
    if (v >= dim) v = 2 * dim - 1 - v;
    return clampCoord(v, dim);
}

/// D7 bounds-safe float coordinate to index (`kMaxF32` first with `coord` first, U6).
pub inline fn absCoordToIndex(profile: TargetProfile, coord: f32, dim: i32, mode: ast.BoundaryMode) usize {
    if (dim <= 1) return 0;
    const dim_f: f32 = @floatFromInt(dim);
    switch (mode) {
        .clamp => {
            const hi = dim_f - 1.0;
            const c = minF32(profile, maxF32(profile, coord, 0.0), hi);
            return clampCoord(cvtRoundF32ToI32(profile, c), dim);
        },
        .mirror => {
            const lo = -dim_f;
            const hi = 2.0 * dim_f - 1.0;
            const c = minF32(profile, maxF32(profile, coord, lo), hi);
            return mirrorCoord(cvtRoundF32ToI32(profile, c), dim);
        },
    }
}

/// Zero-overhead D4 `NaN -> peak`, `+inf -> peak`, `-inf -> 0` integer store helper.
pub inline fn floatToIntSample(profile: TargetProfile, fmt: ast.SampleFormat, v: f32) u32 {
    const peak = fmt.peakF32();
    // Step 1: min(v, peak) with v first -> NaN & +inf become peak on BOTH x86 and AArch64!
    const t = minF32(profile, v, peak);
    // Step 2: max(t, 0.0) -> -inf and negatives become 0.0 while peak stays peak.
    const clamped = maxF32(profile, t, 0.0);
    const iv = cvtRoundF32ToI32(profile, clamped);
    if (iv <= 0) return 0;
    if (fmt.bits_per_sample == 32) return @intCast(iv);
    const peak_i: i32 = @intCast(fmt.peakInt());
    if (iv >= peak_i) return @intCast(peak_i);
    return @intCast(iv);
}

pub inline fn loadSample(ptr: [*]const u8, stride: isize, fmt: ast.SampleFormat, x: usize, y: usize) f32 {
    const row_ptr = offsetRow(ptr, stride, y);
    if (fmt.is_float) {
        if (fmt.bits_per_sample == 16) {
            const p: [*]const f16 = @ptrCast(@alignCast(row_ptr));
            return @floatCast(p[x]);
        }
        const p: [*]const f32 = @ptrCast(@alignCast(row_ptr));
        return p[x];
    }
    if (fmt.bits_per_sample <= 8) {
        return @floatFromInt(row_ptr[x]);
    }
    if (fmt.bits_per_sample <= 16) {
        const p: [*]const u16 = @ptrCast(@alignCast(row_ptr));
        return @floatFromInt(p[x]);
    }
    const p: [*]const i32 = @ptrCast(@alignCast(row_ptr));
    return @floatFromInt(p[x]);
}

pub inline fn loadSampleI32(ptr: [*]const u8, stride: isize, fmt: ast.SampleFormat, x: usize, y: usize) i32 {
    const row_ptr = offsetRow(ptr, stride, y);
    if (fmt.bits_per_sample <= 8) {
        return row_ptr[x];
    }
    if (fmt.bits_per_sample <= 16) {
        const p: [*]const u16 = @ptrCast(@alignCast(row_ptr));
        return p[x];
    }
    const p: [*]const i32 = @ptrCast(@alignCast(row_ptr));
    return p[x];
}

pub inline fn storeSample(profile: TargetProfile, ptr: [*]u8, stride: isize, fmt: ast.SampleFormat, x: usize, y: usize, val: EvalVal) void {
    const row_ptr = offsetRowMut(ptr, stride, y);
    if (fmt.is_float) {
        const fv = val.asF32();
        if (fmt.bits_per_sample == 16) {
            const p: [*]f16 = @ptrCast(@alignCast(row_ptr));
            p[x] = @floatCast(fv);
            return;
        }
        const p: [*]f32 = @ptrCast(@alignCast(row_ptr));
        p[x] = fv;
        return;
    }
    if (fmt.bits_per_sample == 32) {
        const p: [*]i32 = @ptrCast(@alignCast(row_ptr));
        p[x] = switch (val) {
            .f => |fv| @intCast(floatToIntSample(profile, fmt, fv)),
            .i => |iv| iv,
        };
        return;
    }
    const sample_u16: u16 = switch (val) {
        .f => |fv| @intCast(floatToIntSample(profile, fmt, fv)),
        .i => |iv| blk: {
            if (iv <= 0) break :blk 0;
            const peak_i: i32 = @intCast(fmt.peakInt());
            if (iv >= peak_i) break :blk @intCast(peak_i);
            break :blk @intCast(iv);
        },
    };
    if (fmt.bits_per_sample <= 8) {
        row_ptr[x] = @intCast(sample_u16);
    } else {
        const p: [*]u16 = @ptrCast(@alignCast(row_ptr));
        p[x] = sample_u16;
    }
}

inline fn offsetRow(base: [*]const u8, stride: isize, y: usize) [*]const u8 {
    const byte_off = @as(isize, @intCast(y)) * stride;
    if (byte_off >= 0) return base + @as(usize, @intCast(byte_off));
    return base - @as(usize, @intCast(-byte_off));
}

inline fn offsetRowMut(base: [*]u8, stride: isize, y: usize) [*]u8 {
    const byte_off = @as(isize, @intCast(y)) * stride;
    if (byte_off >= 0) return base + @as(usize, @intCast(byte_off));
    return base - @as(usize, @intCast(-byte_off));
}

// ============================================================================
// Pure scalar evaluator for unary/binary/ternary ops & whole planes
// ============================================================================

pub const EvalVal = union(enum) {
    f: f32,
    i: i32,

    pub inline fn asF32(self: EvalVal) f32 {
        return switch (self) {
            .f => |v| v,
            .i => |v| @floatFromInt(v),
        };
    }
};

/// AArch64 `frintz`/`frintm`/`frintn` return a quiet NaN for a signaling NaN (M2).
inline fn roundNanF32(profile: TargetProfile, a: f32) f32 {
    return switch (profile.isa_family) {
        .aarch64 => quietNanF32(a),
        .x86 => a,
    };
}

pub fn evalUnaryF32(profile: TargetProfile, op: ast.UnaryOp, a: f32) f32 {
    return switch (op) {
        .neg => -a,
        .abs => @abs(a),
        .sqrt => @sqrt(maxF32(profile, a, 0.0)), // D2 runtime: sqrt(max(x, 0.0))
        .rsqrt => 1.0 / @sqrt(minF32(profile, maxF32(profile, a, flt_min_f32), flt_max_f32)),
        .sgn => blk: {
            const pos: f32 = if (a > 0.0) 1.0 else 0.0;
            const neg: f32 = if (a < 0.0) 1.0 else 0.0;
            break :blk pos - neg;
        },
        .trunc => if (std.math.isNan(a)) roundNanF32(profile, a) else @trunc(a),
        .floor => if (std.math.isNan(a)) roundNanF32(profile, a) else @floor(a),
        .ceil => if (std.math.isNan(a)) roundNanF32(profile, a) else @ceil(a),
        .round => if (std.math.isNan(a)) roundNanF32(profile, a) else roundEvenF32(a),
        .not => if (a <= 0.0) 1.0 else 0.0, // D12 / D15: ordered v <= 0.0 (NaN -> 0.0)
        .bitnot => @floatFromInt(~cvtRoundF32ToI32(profile, a)),
        .exp => expApprox(profile, a),
        .exp2 => exp2Approx(profile, a),
        .log => logApprox(profile, a),
        .log2 => log2Approx(profile, a),
        .sin => sinApprox(profile, a),
        .cos => cosApprox(profile, a),
        .tan => tanApprox(profile, a),
        .atan => atanApprox(profile, a),
        .asin => asinApprox(profile, a),
        .acos => acosApprox(profile, a),
        .to_f32 => a,
    };
}

/// `v > 0.0` evaluated on the IEEE-754 bit pattern so LLVM on AArch64 cannot fold
/// `(a > 0.0) or (b > 0.0)` into `fmaxnm(a, b) > 0.0`, which turns a signaling NaN
/// into qNaN and returns false instead of matching ordered SIMD comparisons (K1).
inline fn gtZeroBits(v: f32) bool {
    const bits: i32 = @bitCast(v);
    return bits > 0 and bits <= 0x7F80_0000;
}

pub fn evalBinaryF32(profile: TargetProfile, op: ast.BinaryOp, a: f32, b: f32) f32 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
        .mod => modF32(a, b),
        .min => minF32(profile, a, b),
        .max => maxF32(profile, a, b),
        .pow => exp2Approx(profile, log2Approx(profile, a) * b),
        .atan2 => atan2Approx(profile, a, b),
        .copysign => copysignF32(a, b),
        .lt => if (a < b) 1.0 else 0.0,
        .le => if (a <= b) 1.0 else 0.0,
        .gt => if (a > b) 1.0 else 0.0,
        .ge => if (a >= b) 1.0 else 0.0,
        .eq => if (a == b) 1.0 else 0.0,
        .and_op => if (gtZeroBits(a) and gtZeroBits(b)) 1.0 else 0.0,
        .or_op => if (gtZeroBits(a) or gtZeroBits(b)) 1.0 else 0.0,
        .xor_op => if (gtZeroBits(a) != gtZeroBits(b)) 1.0 else 0.0,
        .bitand => @floatFromInt(cvtRoundF32ToI32(profile, a) & cvtRoundF32ToI32(profile, b)),
        .bitor => @floatFromInt(cvtRoundF32ToI32(profile, a) | cvtRoundF32ToI32(profile, b)),
        .bitxor => @floatFromInt(cvtRoundF32ToI32(profile, a) ^ cvtRoundF32ToI32(profile, b)),
    };
}

pub fn evalTernaryF32(profile: TargetProfile, op: ast.TernaryOp, a: f32, b: f32, c: f32) f32 {
    return switch (op) {
        .select => if (a > 0.0) b else c, // D15: ordered a > 0 (NaN -> c)
        .fmadd => fmaF32(profile, a, b, c),
        .fmsub => fmsF32(profile, a, b, c),
        .fnmadd => fnmaF32(profile, a, b, c),
        .fnmsub => fnmsF32(profile, a, b, c),
        .linstep => minF32(profile, maxF32(profile, (c - a) / (b - a), 0.0), 1.0),
    };
}

pub fn evalUnaryVal(profile: TargetProfile, op: ast.UnaryOp, a: EvalVal, opt_int: bool) EvalVal {
    if (opt_int) {
        if (a == .i) {
            const iv = a.i;
            switch (op) {
                .neg => return .{ .i = 0 -% iv },
                .abs => return .{ .i = if (iv < 0) 0 -% iv else iv },
                .sgn => return .{ .i = if (iv > 0) 1 else if (iv < 0) -1 else 0 },
                .not => return .{ .i = if (iv <= 0) 1 else 0 },
                .bitnot => return .{ .i = ~iv },
                else => {},
            }
        } else {
            // D10 / akarin `opt=1`: `not` and `bitnot` always produce `i32` even on float inputs.
            switch (op) {
                .not => return .{ .i = if (a.f <= 0.0) 1 else 0 },
                .bitnot => return .{ .i = ~cvtRoundF32ToI32(profile, a.f) },
                else => {},
            }
        }
    }
    return .{ .f = evalUnaryF32(profile, op, a.asF32()) };
}

pub fn evalBinaryVal(profile: TargetProfile, op: ast.BinaryOp, a: EvalVal, b: EvalVal, opt_int: bool) EvalVal {
    if (opt_int) {
        if (a == .i and b == .i) {
            const ia = a.i;
            const ib = b.i;
            switch (op) {
                .add => return .{ .i = ia +% ib },
                .sub => return .{ .i = ia -% ib },
                .mul => return .{ .i = ia *% ib },
                .min => return .{ .i = if (ia < ib) ia else ib },
                .max => return .{ .i = if (ia > ib) ia else ib },
                .lt => return .{ .i = if (ia < ib) 1 else 0 },
                .le => return .{ .i = if (ia <= ib) 1 else 0 },
                .gt => return .{ .i = if (ia > ib) 1 else 0 },
                .ge => return .{ .i = if (ia >= ib) 1 else 0 },
                .eq => return .{ .i = if (ia == ib) 1 else 0 },
                .and_op => return .{ .i = if (ia > 0 and ib > 0) 1 else 0 },
                .or_op => return .{ .i = if (ia > 0 or ib > 0) 1 else 0 },
                .xor_op => return .{ .i = if ((ia > 0) != (ib > 0)) 1 else 0 },
                .bitand => return .{ .i = ia & ib },
                .bitor => return .{ .i = ia | ib },
                .bitxor => return .{ .i = ia ^ ib },
                else => {},
            }
        } else {
            // D10 / akarin `opt=1`: comparisons, logic, and bitwise ops always produce `i32`.
            const fa = a.asF32();
            const fb = b.asF32();
            switch (op) {
                .lt => return .{ .i = if (fa < fb) 1 else 0 },
                .le => return .{ .i = if (fa <= fb) 1 else 0 },
                .gt => return .{ .i = if (fa > fb) 1 else 0 },
                .ge => return .{ .i = if (fa >= fb) 1 else 0 },
                .eq => return .{ .i = if (fa == fb) 1 else 0 },
                .and_op => return .{ .i = if (gtZeroBits(fa) and gtZeroBits(fb)) 1 else 0 },
                .or_op => return .{ .i = if (gtZeroBits(fa) or gtZeroBits(fb)) 1 else 0 },
                .xor_op => return .{ .i = if (gtZeroBits(fa) != gtZeroBits(fb)) 1 else 0 },
                .bitand => return .{ .i = cvtRoundF32ToI32(profile, fa) & cvtRoundF32ToI32(profile, fb) },
                .bitor => return .{ .i = cvtRoundF32ToI32(profile, fa) | cvtRoundF32ToI32(profile, fb) },
                .bitxor => return .{ .i = cvtRoundF32ToI32(profile, fa) ^ cvtRoundF32ToI32(profile, fb) },
                else => {},
            }
        }
    }
    return .{ .f = evalBinaryF32(profile, op, a.asF32(), b.asF32()) };
}

pub fn evalTernaryVal(profile: TargetProfile, op: ast.TernaryOp, a: EvalVal, b: EvalVal, c: EvalVal, opt_int: bool) EvalVal {
    if (opt_int and op == .select and b == .i and c == .i) {
        const cond = switch (a) {
            .i => |iv| iv > 0,
            .f => |fv| fv > 0.0,
        };
        return .{ .i = if (cond) b.i else c.i };
    }
    return .{ .f = evalTernaryF32(profile, op, a.asF32(), b.asF32(), c.asF32()) };
}

pub const PlaneContext = struct {
    dst: [*]u8,
    dst_stride: isize,
    dst_fmt: ast.SampleFormat,
    srcs: []const [*]const u8,
    src_strides: []const isize,
    src_fmts: []const ast.SampleFormat,
    width: usize,
    height: usize,
    frame_n: i32,
    props: []const f32,
};

pub const ScalarContext = struct {
    width: i32,
    height: i32,
    frame_n: i32,
    props: []const f32,
};

/// Evaluate a scalar expression (for `Select` / `PropExpr` / constant tests).
pub fn evalScalar(
    allocator: std.mem.Allocator,
    prog: *const ast.Program,
    profile: TargetProfile,
    ctx: ScalarContext,
) std.mem.Allocator.Error!f32 {
    const vals = try allocator.alloc(EvalVal, prog.nodes.items.len);
    defer allocator.free(vals);

    const sort_results = try allocator.alloc(?[]u16, prog.arg_sorts.items.len);
    defer {
        for (sort_results) |maybe_buf| {
            if (maybe_buf) |buf| allocator.free(buf);
        }
        allocator.free(sort_results);
    }
    @memset(sort_results, null);

    for (prog.nodes.items, 0..) |node, idx| {
        vals[idx] = switch (node) {
            .const_f32 => |v| .{ .f = v },
            .const_i32 => |v| .{ .i = v },
            .load_prop => |pid| .{ .f = if (pid < ctx.props.len) ctx.props[pid] else std.math.nan(f32) },
            .frame_n => if (prog.opt_int) .{ .i = ctx.frame_n } else .{ .f = @floatFromInt(ctx.frame_n) },
            .plane_width => if (prog.opt_int) .{ .i = ctx.width } else .{ .f = @floatFromInt(ctx.width) },
            .plane_height => if (prog.opt_int) .{ .i = ctx.height } else .{ .f = @floatFromInt(ctx.height) },
            .coord_x, .coord_y, .load_clip, .load_rel, .load_abs => .{ .f = 0.0 },
            .unary => |u| evalUnaryVal(profile, u.op, vals[u.a], prog.opt_int),
            .binary => |b| evalBinaryVal(profile, b.op, vals[b.a], vals[b.b], prog.opt_int),
            .ternary => |t| evalTernaryVal(profile, t.op, vals[t.a], vals[t.b], vals[t.c], prog.opt_int),
            .arg_extrema => |ae| blk: {
                const arg_slice = prog.extra_args.items[ae.args_start..][0..ae.count];
                var best_idx: usize = 0;
                var best_val = vals[arg_slice[0]].asF32();
                for (arg_slice[1..], 1..) |arg_node, i| {
                    const v = vals[arg_node].asF32();
                    const better = switch (ae.kind) {
                        .argmin => v < best_val,
                        .argmax => v > best_val,
                    };
                    if (better) {
                        best_val = v;
                        best_idx = i;
                    }
                }
                break :blk .{ .f = @floatFromInt(best_idx) };
            },
            .arg_sort_elem => |as| blk: {
                if (sort_results[as.sort_id] == null) {
                    const spec = prog.arg_sorts.items[as.sort_id];
                    const arg_slice = prog.extra_args.items[spec.args_start..][0..spec.count];
                    const idxs = try allocator.alloc(u16, spec.count);
                    for (idxs, 0..) |*slot, i| slot.* = @intCast(i);
                    const SortCtx = struct {
                        args: []const ast.NodeIdx,
                        node_vals: []const EvalVal,
                        fn lessThan(self: @This(), lhs: u16, rhs: u16) bool {
                            // Descending sort matching akarin (`stack[off+l] > stack[off+r]`).
                            return self.node_vals[self.args[lhs]].asF32() > self.node_vals[self.args[rhs]].asF32();
                        }
                    };
                    std.mem.sort(u16, idxs, SortCtx{ .args = arg_slice, .node_vals = vals }, SortCtx.lessThan);
                    sort_results[as.sort_id] = idxs;
                }
                break :blk .{ .f = @floatFromInt(sort_results[as.sort_id].?[as.rank]) };
            },
        };
    }

    return vals[prog.root].asF32();
}

/// Evaluate a full plane iteratively using the exact same lowered/optimized DAG as the JIT.
pub fn evalPlane(
    allocator: std.mem.Allocator,
    prog: *const ast.Program,
    profile: TargetProfile,
    ctx: PlaneContext,
) std.mem.Allocator.Error!void {
    const vals = try allocator.alloc(EvalVal, prog.nodes.items.len);
    defer allocator.free(vals);

    const w_i32: i32 = @intCast(ctx.width);
    const h_i32: i32 = @intCast(ctx.height);

    for (0..ctx.height) |y| {
        const y_i32: i32 = @intCast(y);
        for (0..ctx.width) |x| {
            const x_i32: i32 = @intCast(x);
            for (prog.nodes.items, 0..) |node, idx| {
                vals[idx] = switch (node) {
                    .const_f32 => |v| .{ .f = v },
                    .const_i32 => |v| .{ .i = v },
                    .load_clip => |clip| blk: {
                        const fmt = ctx.src_fmts[clip];
                        if (prog.opt_int and !fmt.is_float) {
                            break :blk .{ .i = loadSampleI32(ctx.srcs[clip], ctx.src_strides[clip], fmt, x, y) };
                        }
                        break :blk .{ .f = loadSample(ctx.srcs[clip], ctx.src_strides[clip], fmt, x, y) };
                    },
                    .load_rel => |rel| blk: {
                        const fmt = ctx.src_fmts[rel.clip];
                        const eff_dx = clampRelOffset(rel.dx, w_i32);
                        const eff_dy = clampRelOffset(rel.dy, h_i32);
                        const sx: usize = switch (rel.mode) {
                            .clamp => clampCoord(x_i32 + eff_dx, w_i32),
                            .mirror => mirrorCoord(x_i32 + eff_dx, w_i32),
                        };
                        const sy: usize = switch (rel.mode) {
                            .clamp => clampCoord(y_i32 + eff_dy, h_i32),
                            .mirror => mirrorCoord(y_i32 + eff_dy, h_i32),
                        };
                        if (prog.opt_int and !fmt.is_float) {
                            break :blk .{ .i = loadSampleI32(ctx.srcs[rel.clip], ctx.src_strides[rel.clip], fmt, sx, sy) };
                        }
                        break :blk .{ .f = loadSample(ctx.srcs[rel.clip], ctx.src_strides[rel.clip], fmt, sx, sy) };
                    },
                    .load_abs => |abs_acc| blk: {
                        const fmt = ctx.src_fmts[abs_acc.clip];
                        const sx = absCoordToIndex(profile, vals[abs_acc.x].asF32(), w_i32, abs_acc.mode);
                        const sy = absCoordToIndex(profile, vals[abs_acc.y].asF32(), h_i32, abs_acc.mode);
                        if (prog.opt_int and !fmt.is_float) {
                            break :blk .{ .i = loadSampleI32(ctx.srcs[abs_acc.clip], ctx.src_strides[abs_acc.clip], fmt, sx, sy) };
                        }
                        break :blk .{ .f = loadSample(ctx.srcs[abs_acc.clip], ctx.src_strides[abs_acc.clip], fmt, sx, sy) };
                    },
                    .load_prop => |pid| .{ .f = if (pid < ctx.props.len) ctx.props[pid] else std.math.nan(f32) },
                    .coord_x => if (prog.opt_int) .{ .i = x_i32 } else .{ .f = @floatFromInt(x_i32) },
                    .coord_y => if (prog.opt_int) .{ .i = y_i32 } else .{ .f = @floatFromInt(y_i32) },
                    .frame_n => if (prog.opt_int) .{ .i = ctx.frame_n } else .{ .f = @floatFromInt(ctx.frame_n) },
                    .plane_width => if (prog.opt_int) .{ .i = w_i32 } else .{ .f = @floatFromInt(w_i32) },
                    .plane_height => if (prog.opt_int) .{ .i = h_i32 } else .{ .f = @floatFromInt(h_i32) },
                    .unary => |u| evalUnaryVal(profile, u.op, vals[u.a], prog.opt_int),
                    .binary => |b| evalBinaryVal(profile, b.op, vals[b.a], vals[b.b], prog.opt_int),
                    .ternary => |t| evalTernaryVal(profile, t.op, vals[t.a], vals[t.b], vals[t.c], prog.opt_int),
                    .arg_extrema, .arg_sort_elem => .{ .f = 0.0 },
                };
            }
            storeSample(profile, ctx.dst, ctx.dst_stride, ctx.dst_fmt, x, y, vals[prog.root]);
        }
    }
}

test "minF32 and maxF32 match x86 and AArch64 ISA rules on NaN, +-0, and +-inf" {
    const x86: TargetProfile = .{ .use_fma = false, .isa_family = .x86 };
    const a64: TargetProfile = .{ .use_fma = true, .isa_family = .aarch64 };
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);

    // x86: second operand wins on NaN and +-0 ties.
    try std.testing.expectEqual(@as(f32, 5.0), minF32(x86, nan, 5.0));
    try std.testing.expect(std.math.isNan(minF32(x86, 5.0, nan)));
    try std.testing.expectEqual(@as(f32, 5.0), maxF32(x86, nan, 5.0));
    try std.testing.expect(std.math.isNan(maxF32(x86, 5.0, nan)));
    try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(minF32(x86, 0.0, -0.0))));
    try std.testing.expectEqual(@as(u32, 0x0000_0000), @as(u32, @bitCast(minF32(x86, -0.0, 0.0))));

    // AArch64 fminnm/fmaxnm: quiet NaN loses to non-NaN; signaling NaN propagates as NaN; -0 < +0.
    const snan: f32 = @bitCast(@as(u32, 0x7FA0_0001));
    try std.testing.expectEqual(@as(f32, 5.0), minF32(a64, nan, 5.0));
    try std.testing.expectEqual(@as(f32, 5.0), minF32(a64, 5.0, nan));
    try std.testing.expect(std.math.isNan(minF32(a64, snan, 5.0)));
    try std.testing.expect(std.math.isNan(minF32(a64, 5.0, snan)));
    try std.testing.expect(std.math.isNan(maxF32(a64, snan, 5.0)));
    try std.testing.expect(std.math.isNan(maxF32(a64, 5.0, snan)));
    try std.testing.expectEqual(inf, minF32(a64, nan, inf));
    try std.testing.expectEqual(-inf, maxF32(a64, -inf, nan));
    try std.testing.expect(std.math.isNan(minF32(a64, nan, nan)));
    try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(minF32(a64, 0.0, -0.0))));
    try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(minF32(a64, -0.0, 0.0))));
    try std.testing.expectEqual(@as(u32, 0x0000_0000), @as(u32, @bitCast(maxF32(a64, -0.0, 0.0))));
}

test "roundEvenF32 and cvtF32ToI32 match X2 edge cases" {
    const x86: TargetProfile = .{ .use_fma = false, .isa_family = .x86 };
    const a64: TargetProfile = .{ .use_fma = true, .isa_family = .aarch64 };
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);

    try std.testing.expectEqual(@as(f32, 0.0), roundEvenF32(0.5));
    try std.testing.expectEqual(@as(f32, 2.0), roundEvenF32(1.5));
    try std.testing.expectEqual(@as(f32, 2.0), roundEvenF32(2.5));
    try std.testing.expectEqual(@as(f32, -2.0), roundEvenF32(-1.5));
    try std.testing.expectEqual(@as(f32, -2.0), roundEvenF32(-2.5));
    // Negative zero preservation for -0.3, -0.5, -0.0 (X2/X3)
    try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(roundEvenF32(-0.3))));
    try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(roundEvenF32(-0.5))));
    try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(roundEvenF32(-0.0))));
    try std.testing.expect(std.math.isNan(roundEvenF32(nan)));
    try std.testing.expectEqual(inf, roundEvenF32(inf));

    // x86 vs AArch64 float->i32 on NaN, +-inf, +-2^31.
    try std.testing.expectEqual(std.math.minInt(i32), cvtRoundF32ToI32(x86, nan));
    try std.testing.expectEqual(std.math.minInt(i32), cvtRoundF32ToI32(x86, inf));
    try std.testing.expectEqual(std.math.minInt(i32), cvtRoundF32ToI32(x86, 2147483648.0));
    try std.testing.expectEqual(std.math.minInt(i32), cvtRoundF32ToI32(x86, -2147483648.0));

    try std.testing.expectEqual(@as(i32, 0), cvtRoundF32ToI32(a64, nan));
    try std.testing.expectEqual(std.math.maxInt(i32), cvtRoundF32ToI32(a64, inf));
    try std.testing.expectEqual(std.math.minInt(i32), cvtRoundF32ToI32(a64, -inf));
    try std.testing.expectEqual(std.math.maxInt(i32), cvtRoundF32ToI32(a64, 2147483648.0));
    try std.testing.expectEqual(std.math.minInt(i32), cvtRoundF32ToI32(a64, -2147483648.0));
}

test "D3 modF32 and D4 NaN->peak integer store" {
    try std.testing.expectEqual(@as(f32, 0.0), modF32(0.7, 0.1));
    try std.testing.expectEqual(@as(f32, 0.0), modF32(2.0, 0.2));
    try std.testing.expectEqual(@as(f32, 0.0), modF32(1e10, 3.0));
    // `-0 3 % == +0.0` (Y4)
    try std.testing.expectEqual(@as(u32, 0x0000_0000), @as(u32, @bitCast(modF32(-0.0, 3.0))));

    const fmt8: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 8 };
    const fmt10: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 10 };
    const fmt32: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 32 };
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);

    inline for ([_]TargetProfile{
        .{ .use_fma = false, .isa_family = .x86 },
        .{ .use_fma = true, .isa_family = .aarch64 },
    }) |prof| {
        try std.testing.expectEqual(@as(u32, 255), floatToIntSample(prof, fmt8, nan));
        try std.testing.expectEqual(@as(u32, 255), floatToIntSample(prof, fmt8, inf));
        try std.testing.expectEqual(@as(u32, 0), floatToIntSample(prof, fmt8, -inf));
        try std.testing.expectEqual(@as(u32, 1023), floatToIntSample(prof, fmt10, nan));
        try std.testing.expectEqual(@as(u32, 2147483520), floatToIntSample(prof, fmt32, nan));
        try std.testing.expectEqual(@as(u32, 2147483520), floatToIntSample(prof, fmt32, inf));
        try std.testing.expectEqual(@as(u32, 2147483520), floatToIntSample(prof, fmt32, 2147483648.0));
        try std.testing.expectEqual(@as(u32, 0), floatToIntSample(prof, fmt32, -inf));
        try std.testing.expectEqual(@as(u32, 0), floatToIntSample(prof, fmt32, -5.0));
    }
}

test "polynomial transcendentals meet §6 finite-domain accuracy bounds vs f64 libm" {
    inline for ([_]TargetProfile{
        .{ .use_fma = false, .isa_family = .x86 },
        .{ .use_fma = true, .isa_family = .x86 },
        .{ .use_fma = true, .isa_family = .aarch64 },
    }) |prof| {
        // exp on [-87, 88]
        var i: i32 = -870;
        while (i <= 880) : (i += 1) {
            const x: f32 = @as(f32, @floatFromInt(i)) * 0.1;
            const got: f64 = expApprox(prof, x);
            const ref = @exp(@as(f64, x));
            const rel = @abs(got - ref) / ref;
            try std.testing.expect(rel <= 2e-6);
        }
        // exp NaN / overflow / underflow
        try std.testing.expect(expApprox(prof, std.math.nan(f32)) > 2.4e38);
        try std.testing.expectEqual(@as(f32, 0.0), expApprox(prof, -88.38));

        // log on [FLT_MIN, 1e37]
        var e: i32 = -125;
        while (e <= 122) : (e += 1) {
            const base = std.math.ldexp(@as(f32, 1.0), e);
            for ([_]f32{ 1.0, 1.15, 1.414, 1.75, 1.99 }) |m| {
                const x = base * m;
                const got: f64 = logApprox(prof, x);
                const ref = @log(@as(f64, x));
                const err = @abs(got - ref);
                try std.testing.expect(err <= 2e-6 * @max(1.0, @abs(ref)));
            }
        }
        // log <= 0 and NaN clamp to FLT_MIN -> ~-87.3365
        try std.testing.expect(@abs(logApprox(prof, 0.0) - (-87.33654)) < 1e-3);
        try std.testing.expect(@abs(logApprox(prof, std.math.nan(f32)) - (-87.33654)) < 1e-3);

        // sin & cos on |x| <= 1e5
        for ([_]f32{ -1e5, -12345.67, -100.0, -3.14159, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 3.14159, 100.0, 99999.25 }) |x| {
            const s_got: f64 = sinApprox(prof, x);
            const c_got: f64 = cosApprox(prof, x);
            try std.testing.expect(@abs(s_got - @sin(@as(f64, x))) <= 2e-6);
            try std.testing.expect(@abs(c_got - @cos(@as(f64, x))) <= 2e-6);
        }
        // sin(-0.0) == -0.0
        try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(sinApprox(prof, -0.0))));

        // tan across full domain |x| <= 1e5 including near +-pi/2 poles where |cos(x)| >= 1e-3 (V9, K2)
        for ([_]f32{
            -1e5, -99999.25, -75432.125, -51475.5, -10.0, -1.569, -1.56,   -1.0,      -0.25,    0.0,
            0.25, 1.0,       1.56,       1.569,    10.0,  100.0,  51475.5, 75432.125, 99999.25, 1e5,
        }) |x| {
            const ref = @tan(@as(f64, x));
            const got: f64 = tanApprox(prof, x);
            try std.testing.expect(@abs(got - ref) <= 1e-5 * @max(1.0, @abs(ref)));
        }
        var ti: i32 = -500;
        while (ti <= 500) : (ti += 1) {
            const x: f32 = @as(f32, @floatFromInt(ti)) * 0.01;
            if (@abs(@cos(@as(f64, x))) < 1e-3) continue;
            const ref = @tan(@as(f64, x));
            const got: f64 = tanApprox(prof, x);
            try std.testing.expect(@abs(got - ref) <= 1e-5 * @max(1.0, @abs(ref)));
        }
        // Wide sweep across |x| <= 1e5 including the k > 2^15 region (51,472 < |x| <= 1e5) without FMA (K2)
        var wi: i32 = -2000;
        while (wi <= 2000) : (wi += 1) {
            const x: f32 = @as(f32, @floatFromInt(wi)) * 49.97;
            if (@abs(@cos(@as(f64, x))) < 1e-3) continue;
            const ref = @tan(@as(f64, x));
            const got: f64 = tanApprox(prof, x);
            try std.testing.expect(@abs(got - ref) <= 1e-5 * @max(1.0, @abs(ref)));
        }

        // K1: runtime signaling NaN truthiness in oracle logical ops (guards AArch64 ReleaseFast fmaxnm fold;
        // doNotOptimizeAway prevents compile-time constant folding so the test fails if the fold returns, M3).
        const snan: f32 = @bitCast(@as(u32, 0x7FA0_0001));
        const neg_snan: f32 = @bitCast(@as(u32, 0xFFA0_0001));
        var bad_vals = [_]f32{ snan, neg_snan };
        var pos_val: f32 = 1e10;
        std.mem.doNotOptimizeAway(&bad_vals);
        std.mem.doNotOptimizeAway(&pos_val);
        for (bad_vals) |bad| {
            try std.testing.expectEqual(@as(f32, 1.0), evalBinaryF32(prof, .or_op, bad, pos_val));
            try std.testing.expectEqual(@as(f32, 1.0), evalBinaryF32(prof, .or_op, pos_val, bad));
            try std.testing.expectEqual(@as(f32, 0.0), evalBinaryF32(prof, .and_op, bad, pos_val));
            try std.testing.expectEqual(@as(f32, 1.0), evalBinaryF32(prof, .xor_op, bad, pos_val));
            try std.testing.expectEqual(@as(i32, 1), evalBinaryVal(prof, .or_op, .{ .f = bad }, .{ .f = pos_val }, true).i);
            try std.testing.expectEqual(@as(i32, 0), evalBinaryVal(prof, .and_op, .{ .f = bad }, .{ .f = pos_val }, true).i);
            try std.testing.expectEqual(@as(i32, 1), evalBinaryVal(prof, .xor_op, .{ .f = bad }, .{ .f = pos_val }, true).i);
        }

        // atan2 across all 4 quadrants and axes
        for ([_]f32{ -3.0, -1.0, -0.25, -0.0, 0.0, 0.25, 1.0, 3.0 }) |y| {
            for ([_]f32{ -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0 }) |x| {
                const got: f64 = atan2Approx(prof, y, x);
                const ref: f64 = if (x == 0.0 and y == 0.0) 0.0 else std.math.atan2(@as(f64, y), @as(f64, x));
                try std.testing.expect(@abs(got - ref) <= 2e-6);
            }
        }

        // atan across wide range, breakpoints, +-0, +-1, +-inf, NaN
        for ([_]f32{
            -1e6, -100.0, -2.5,       -2.4142136, -1.0, -0.5,      -0.41421356, -0.1,  -0.0,
            0.0,  0.1,    0.41421356, 0.5,        1.0,  2.4142136, 2.5,         100.0, 1e6,
        }) |x| {
            const got: f64 = atanApprox(prof, x);
            const ref = std.math.atan(@as(f64, x));
            try std.testing.expect(@abs(got - ref) <= 2e-6);
        }
        try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(atanApprox(prof, -0.0))));
        try std.testing.expect(@abs(@as(f64, atanApprox(prof, std.math.inf(f32))) - std.math.pi * 0.5) <= 2e-6);
        try std.testing.expect(@abs(@as(f64, atanApprox(prof, -std.math.inf(f32))) + std.math.pi * 0.5) <= 2e-6);
        try std.testing.expect(std.math.isNan(atanApprox(prof, std.math.nan(f32))));

        // asin & acos across [-1, 1], +-0, +-0.5, +-1, |x| > 1 -> NaN, NaN -> NaN
        var ai: i32 = -200;
        while (ai <= 200) : (ai += 1) {
            const x: f32 = @as(f32, @floatFromInt(ai)) * 0.005;
            const ga: f64 = asinApprox(prof, x);
            const gc: f64 = acosApprox(prof, x);
            try std.testing.expect(@abs(ga - std.math.asin(@as(f64, x))) <= 2e-6);
            try std.testing.expect(@abs(gc - std.math.acos(@as(f64, x))) <= 2e-6);
        }
        try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(asinApprox(prof, -0.0))));
        try std.testing.expectEqual(@as(f32, std.math.pi * 0.5), asinApprox(prof, 1.0));
        try std.testing.expectEqual(@as(f32, -std.math.pi * 0.5), asinApprox(prof, -1.0));
        try std.testing.expectEqual(@as(f32, 0.0), acosApprox(prof, 1.0));
        try std.testing.expectEqual(@as(f32, std.math.pi), acosApprox(prof, -1.0));
        for ([_]f32{ -1.5, 1.5, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) }) |bad| {
            try std.testing.expect(std.math.isNan(asinApprox(prof, bad)));
            try std.testing.expect(std.math.isNan(acosApprox(prof, bad)));
        }

        // exp2 & log2: exact at integers / powers of 2, < 2e-6 across domain
        for ([_]f32{ -126.0, -10.0, -1.0, 0.0, 1.0, 3.0, 8.0, 20.0 }) |k| {
            try std.testing.expectEqual(std.math.exp2(k), exp2Approx(prof, k));
        }
        for ([_]f32{ -20.3, -1.7, -0.5, -0.1, 0.1, 0.5, 1.7, 20.3 }) |x| {
            const ref = std.math.exp2(@as(f64, x));
            try std.testing.expect(@abs(@as(f64, exp2Approx(prof, x)) - ref) <= 2e-6 * ref);
        }
        try std.testing.expectEqual(@as(f32, 0.0), exp2Approx(prof, -200.0));
        for ([_]f32{ 0.0625, 0.25, 0.5, 1.0, 2.0, 8.0, 1024.0 }) |p2| {
            try std.testing.expectEqual(std.math.log2(p2), log2Approx(prof, p2));
        }
        for ([_]f32{ 1e-6, 0.1, 0.707, 1.414, 3.0, 100.0, 1e6 }) |x| {
            const ref = std.math.log2(@as(f64, x));
            try std.testing.expect(@abs(@as(f64, log2Approx(prof, x)) - ref) <= 2e-6 * @max(1.0, @abs(ref)));
        }
        try std.testing.expectEqual(@as(f32, 8.0), evalBinaryF32(prof, .pow, 2.0, 3.0));
        try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(copysignF32(0.0, -0.0))));
        try std.testing.expectEqual(@as(f32, -3.5), copysignF32(3.5, -2.0));
        try std.testing.expectEqual(@as(f32, 3.5), copysignF32(-3.5, 0.0));

        // ceil preserves -0.0 on (-1, 0] and handles NaN/+-inf
        try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(evalUnaryF32(prof, .ceil, -0.5))));
        try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(evalUnaryF32(prof, .ceil, -0.0))));
        try std.testing.expectEqual(@as(f32, 1.0), evalUnaryF32(prof, .ceil, 0.25));
        try std.testing.expectEqual(@as(f32, -1.0), evalUnaryF32(prof, .ceil, -1.5));
        try std.testing.expect(std.math.isNan(evalUnaryF32(prof, .ceil, std.math.nan(f32))));
        try std.testing.expectEqual(std.math.inf(f32), evalUnaryF32(prof, .ceil, std.math.inf(f32)));
        try std.testing.expectEqual(-std.math.inf(f32), evalUnaryF32(prof, .ceil, -std.math.inf(f32)));
    }
}
