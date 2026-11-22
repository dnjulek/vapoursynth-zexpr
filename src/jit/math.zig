//! SIMD transcendental emitters (`exp`, `log`, `sin`, `cos`, `tan`, `atan2`)
//! matching `src/expr/interp.zig` bit-for-bit on every ISA tier.

const std = @import("std");
const interp = @import("../expr/interp.zig");
const aj = @import("asmjit.zig");
const pixel = @import("pixel.zig");

/// `dst = a * b + c` (FMA on FMA tiers, separate mul+add on non-FMA tiers).
inline fn emitFma(b: *aj.Builder, dst: aj.Reg, a: aj.Reg, b_reg: aj.Reg, c: aj.Reg) void {
    b.vecOp4(.madd_f32, dst, a, b_reg, c);
}

/// `dst = c - a * b` (FMA on FMA tiers, separate mul+sub on non-FMA tiers).
inline fn emitFnma(b: *aj.Builder, dst: aj.Reg, a: aj.Reg, b_reg: aj.Reg, c: aj.Reg) void {
    b.vecOp4(.nmadd_f32, dst, a, b_reg, c);
}

/// Clamped Cody-Waite `exp(x)` with single `(k + 127) << 23` injection, Estrin polynomial,
/// and RIP-relative const-memory FMA operands (zero constant stack spills!).
/// When `use_const_mem` is false (small low-pressure kernels like standalone `x exp`), keeps
/// clamp/add constants hoisted in vector registers; when true (inside `pow` or deep DAGs),
/// folds them as RIP-relative memory operands to avoid register spills (G4).
pub fn emitExp(b: *aj.Builder, dst: aj.Reg, x: aj.Reg, use_const_mem: bool) void {
    const xc = b.newVec();
    if (use_const_mem) {
        b.vecMinCF32(xc, x, interp.exp_hi_clamp);
        b.vecMaxCF32(xc, xc, interp.exp_lo_clamp);
    } else {
        b.vecOp3(.min_f32, xc, x, b.constF32Splat(interp.exp_hi_clamp));
        b.vecOp3(.max_f32, xc, xc, b.constF32Splat(interp.exp_lo_clamp));
    }

    const ki = b.newVec();
    b.vecMulCF32(ki, xc, interp.log2e_f32);
    b.vecOp2(.cvt_round_f32_to_i32, ki, ki);
    const k = b.newVec();
    b.vecOp2(.cvt_i32_to_f32, k, ki);

    const r = b.newVec();
    b.vecFnmaBCF32(r, k, interp.ln2_hi_f32, xc);
    b.vecFnmaBCF32(r, k, interp.ln2_lo_f32, r);

    // Estrin's scheme matching interp.expApprox bit-for-bit (3 FMA levels, 0 spills).
    const r2 = b.newVec();
    const r4 = b.newVec();
    b.vecOp3(.mul_f32, r2, r, r);
    b.vecOp3(.mul_f32, r4, r2, r2);

    const p01 = b.newVec();
    const p23 = b.newVec();
    const p45 = b.newVec();
    if (use_const_mem) {
        b.vecAddCF32(p01, r, 1.0);
    } else {
        b.vecOp3(.add_f32, p01, b.constF32Splat(1.0), r);
    }
    b.vecFmaCCF32(p23, r, 1.6666667e-1, 0.5);
    b.vecFmaCCF32(p45, r, 8.3333338e-3, 4.1666668e-2);

    const p03 = b.newVec();
    const p46 = b.newVec();
    emitFma(b, p03, p23, r2, p01);
    // p46 = r2 * 1.3888889e-3 + p45 = p45 - r2 * (-1.3888889e-3)
    b.vecFnmaBCF32(p46, r2, -1.3888889e-3, p45);

    const p = b.newVec();
    emitFma(b, p, p46, r4, p03);

    b.vecOp3(.add_i32, ki, ki, b.constI32Splat(127));
    b.vecOp2i(.sll_u32, ki, ki, 23);
    b.vecOp3(.mul_f32, dst, p, ki);
}

/// Exact-at-integer minimax `exp2(x)` matching `interp.exp2Approx` bit-for-bit.
/// Saves `mul(log2e)` and two Cody-Waite `fnma` subtractions vs `emitExp`.
pub fn emitExp2(b: *aj.Builder, dst: aj.Reg, x: aj.Reg, use_const_mem: bool) void {
    const xc = b.newVec();
    if (use_const_mem) {
        b.vecMinCF32(xc, x, interp.exp2_hi_clamp);
        b.vecMaxCF32(xc, xc, interp.exp2_lo_clamp);
    } else {
        b.vecOp3(.min_f32, xc, x, b.constF32Splat(interp.exp2_hi_clamp));
        b.vecOp3(.max_f32, xc, xc, b.constF32Splat(interp.exp2_lo_clamp));
    }

    const ki = b.newVec();
    b.vecOp2(.cvt_round_f32_to_i32, ki, xc);
    const k = b.newVec();
    b.vecOp2(.cvt_i32_to_f32, k, ki);

    const r = b.newVec();
    b.vecOp3(.sub_f32, r, xc, k);

    // Exact-at-integer deg-5 Horner (5 FMAs, 0 r2/r4 muls, max_rel <= 2.9e-7).
    const p = b.newVec();
    b.vecFmaCCF32(p, r, interp.exp2_c5, interp.exp2_c4);
    b.vecFmaBCF32(p, p, r, interp.exp2_c3);
    b.vecFmaBCF32(p, p, r, interp.exp2_c2);
    b.vecFmaBCF32(p, p, r, interp.exp2_c1);
    b.vecFmaBCF32(p, p, r, 1.0);

    b.vecOp3(.add_i32, ki, ki, b.constI32Splat(127));
    b.vecOp2i(.sll_u32, ki, ki, 23);
    b.vecOp3(.mul_f32, dst, p, ki);
}

/// Fast-clamped `log2(x)` with `u = (m - 1) / (m + 1)` + 3-FMA odd minimax + `e` (exact at 2^k).
pub fn emitLog2(b: *aj.Builder, dst: aj.Reg, x: aj.Reg, use_const_mem: bool) void {
    const c_adj = b.constI32Splat(0x004A_FB0D);
    const c_mant_mask = b.constI32Splat(0x007F_FFFF);
    const c_mant_base = b.constI32Splat(0x3F35_04F3);
    const c_127 = b.constI32Splat(127);

    const xc = b.newVec();
    if (use_const_mem) {
        b.vecMaxCF32(xc, x, interp.flt_min_f32);
    } else {
        b.vecOp3(.max_f32, xc, x, b.constF32Splat(interp.flt_min_f32));
    }

    const adj = b.newVec();
    b.vecOp3(.add_i32, adj, xc, c_adj);

    const e = b.newVec();
    b.vecOp2i(.sra_i32, e, adj, 23);
    b.vecOp3(.sub_i32, e, e, c_127);
    b.vecOp2(.cvt_i32_to_f32, e, e);

    const m = b.newVec();
    b.vecOp3(.and_u32, m, adj, c_mant_mask);
    b.vecOp3(.add_i32, m, m, c_mant_base);

    const num = b.newVec();
    const den = b.newVec();
    if (use_const_mem) {
        b.vecSubCF32(num, m, 1.0);
        b.vecAddCF32(den, m, 1.0);
    } else {
        const c_one = b.constF32Splat(1.0);
        b.vecOp3(.sub_f32, num, m, c_one);
        b.vecOp3(.add_f32, den, m, c_one);
    }
    const u = b.newVec();
    b.vecOp3(.div_f32, u, num, den);

    const u_sq = b.newVec();
    b.vecOp3(.mul_f32, u_sq, u, u);

    const p = b.newVec();
    b.vecFmaCCF32(p, u_sq, interp.log2_u_c3, interp.log2_u_c2);
    b.vecFmaBCF32(p, p, u_sq, interp.log2_u_c1);
    b.vecFmaBCF32(p, p, u_sq, interp.log2_u_c0);
    emitFma(b, dst, p, u, e);
}

/// Fast-clamped `log(x)` with `kMaxF32(x, FLT_MIN)` and 3-FMA minimax odd polynomial (D13, U3).
pub fn emitLog(b: *aj.Builder, dst: aj.Reg, x: aj.Reg, use_const_mem: bool) void {
    const c_adj = b.constI32Splat(0x004A_FB0D);
    const c_mant_mask = b.constI32Splat(0x007F_FFFF);
    const c_mant_base = b.constI32Splat(0x3F35_04F3);
    const c_127 = b.constI32Splat(127);

    const xc = b.newVec();
    if (use_const_mem) {
        b.vecMaxCF32(xc, x, interp.flt_min_f32);
    } else {
        b.vecOp3(.max_f32, xc, x, b.constF32Splat(interp.flt_min_f32));
    }

    const adj = b.newVec();
    b.vecOp3(.add_i32, adj, xc, c_adj);

    const e = b.newVec();
    b.vecOp2i(.sra_i32, e, adj, 23);
    b.vecOp3(.sub_i32, e, e, c_127);
    b.vecOp2(.cvt_i32_to_f32, e, e);

    const m = b.newVec();
    b.vecOp3(.and_u32, m, adj, c_mant_mask);
    b.vecOp3(.add_i32, m, m, c_mant_base);

    const num = b.newVec();
    const den = b.newVec();
    if (use_const_mem) {
        b.vecSubCF32(num, m, 1.0);
        b.vecAddCF32(den, m, 1.0);
    } else {
        const c_one = b.constF32Splat(1.0);
        b.vecOp3(.sub_f32, num, m, c_one);
        b.vecOp3(.add_f32, den, m, c_one);
    }
    const u = b.newVec();
    b.vecOp3(.div_f32, u, num, den);

    const u_sq = b.newVec();
    b.vecOp3(.mul_f32, u_sq, u, u);

    const p = b.newVec();
    b.vecFmaCCF32(p, u_sq, interp.log_u_c3, interp.log_u_c2);
    b.vecFmaBCF32(p, p, u_sq, interp.log_u_c1);
    b.vecFmaBCF32(p, p, u_sq, interp.log_u_c0);
    b.vecOp3(.mul_f32, p, p, u);

    b.vecFnmaBCF32(p, e, -interp.ln2_lo_f32, p);
    b.vecFnmaBCF32(dst, e, -interp.ln2_hi_f32, p);
}

pub const TrigState = struct {
    r: aj.Reg,
    r2: aj.Reg,
    r4: aj.Reg,
    sign_mask: aj.Reg,
    x_sign: aj.Reg,
};

/// Prepare shared Cody-Waite range reduction for `sin`/`cos`/`tan` on `x` (enables shared `sincos`
/// with RIP-relative constant memory operands and zero stack spills!).
pub fn emitTrigPrepare(b: *aj.Builder, x: aj.Reg) TrigState {
    const c_sign = b.constI32Splat(@bitCast(@as(u32, 0x8000_0000)));
    const c_abs = b.constI32Splat(0x7FFF_FFFF);
    const x_sign = b.newVec();
    const ax = b.newVec();
    b.vecOp3(.and_u32, x_sign, x, c_sign);
    b.vecOp3(.and_u32, ax, x, c_abs);

    const ki = b.newVec();
    b.vecMulCF32(ki, ax, interp.inv_pi_f32);
    b.vecOp2(.cvt_round_f32_to_i32, ki, ki);
    const k = b.newVec();
    b.vecOp2(.cvt_i32_to_f32, k, ki);

    const r = b.newVec();
    b.vecFnmaBCF32(r, k, interp.pi1_f32, ax);
    b.vecFnmaBCF32(r, k, interp.pi2_f32, r);
    b.vecFnmaBCF32(r, k, interp.pi3_f32, r);
    b.vecFnmaBCF32(r, k, interp.pi4_f32, r);

    const r2 = b.newVec();
    b.vecOp3(.mul_f32, r2, r, r);
    const r4 = b.newVec();
    b.vecOp3(.mul_f32, r4, r2, r2);

    const sign_mask = b.newVec();
    b.vecOp2i(.sll_u32, sign_mask, ki, 31);
    return .{ .r = r, .r2 = r2, .r4 = r4, .sign_mask = sign_mask, .x_sign = x_sign };
}

fn emitSinPoly(b: *aj.Builder, dst: aj.Reg, r: aj.Reg, r2: aj.Reg, r4: aj.Reg) void {
    const p01 = b.newVec();
    const p23 = b.newVec();
    b.vecFmaCCF32(p01, r2, 8.3333338e-3, -1.6666667e-1);
    b.vecFmaCCF32(p23, r2, 2.7557319e-6, -1.9841270e-4);

    const p24 = b.newVec();
    b.vecFnmaBCF32(p24, r4, 2.5052108e-8, p23); // p23 - r4 * (+2.5052108e-8) == r4 * (-2.5052108e-8) + p23

    const p = b.newVec();
    emitFma(b, p, p24, r4, p01);
    b.vecOp3(.mul_f32, p, p, r2);
    emitFma(b, dst, p, r, r);
}

fn emitCosPoly(b: *aj.Builder, dst: aj.Reg, r2: aj.Reg, r4: aj.Reg) void {
    const p01 = b.newVec();
    const p23 = b.newVec();
    b.vecFmaCCF32(p01, r2, 4.1666668e-2, -0.5);
    b.vecFmaCCF32(p23, r2, 2.4801587e-5, -1.3888889e-3);

    const p24 = b.newVec();
    b.vecFnmaBCF32(p24, r4, 2.7557319e-7, p23); // p23 - r4 * (+2.7557319e-7) == r4 * (-2.7557319e-7) + p23

    const p = b.newVec();
    emitFma(b, p, p24, r4, p01);
    b.vecFmaBCF32(dst, p, r2, 1.0);
}

pub fn emitSinFromState(b: *aj.Builder, dst: aj.Reg, st: TrigState) void {
    const s = b.newVec();
    emitSinPoly(b, s, st.r, st.r2, st.r4);
    b.vecOp3(.xor_u32, s, s, st.sign_mask);
    b.vecOp3(.xor_u32, dst, s, st.x_sign);
}

pub fn emitCosFromState(b: *aj.Builder, dst: aj.Reg, st: TrigState) void {
    const c = b.newVec();
    emitCosPoly(b, c, st.r2, st.r4);
    b.vecOp3(.xor_u32, dst, c, st.sign_mask);
}

pub fn emitSin(b: *aj.Builder, dst: aj.Reg, x: aj.Reg) void {
    emitSinFromState(b, dst, emitTrigPrepare(b, x));
}

pub fn emitCos(b: *aj.Builder, dst: aj.Reg, x: aj.Reg) void {
    emitCosFromState(b, dst, emitTrigPrepare(b, x));
}

/// V9: Mod-(pi/2) Cody-Waite `tan(x)` matching `interp.tanApprox` bit-for-bit across all quadrants
/// including near `+-pi/2` poles (`|r| <= pi/4`, single division).
pub fn emitTan(b: *aj.Builder, dst: aj.Reg, x: aj.Reg) void {
    const c_sign = b.constI32Splat(@bitCast(@as(u32, 0x8000_0000)));
    const c_abs = b.constI32Splat(0x7FFF_FFFF);
    const x_sign = b.newVec();
    const ax = b.newVec();
    b.vecOp3(.and_u32, x_sign, x, c_sign);
    b.vecOp3(.and_u32, ax, x, c_abs);

    const ki = b.newVec();
    b.vecMulCF32(ki, ax, interp.inv_half_pi_f32);
    b.vecOp2(.cvt_round_f32_to_i32, ki, ki);
    const k = b.newVec();
    b.vecOp2(.cvt_i32_to_f32, k, ki);

    const r = b.newVec();
    b.vecFnmaBCF32(r, k, interp.half_pi1_f32, ax);
    b.vecFnmaBCF32(r, k, interp.half_pi2_f32, r);
    b.vecFnmaBCF32(r, k, interp.half_pi3_f32, r);
    b.vecFnmaBCF32(r, k, interp.half_pi4_f32, r);

    const r2 = b.newVec();
    const r4 = b.newVec();
    b.vecOp3(.mul_f32, r2, r, r);
    b.vecOp3(.mul_f32, r4, r2, r2);

    const s = b.newVec();
    const c = b.newVec();
    emitSinPoly(b, s, r, r2, r4);
    emitCosPoly(b, c, r2, r4);

    const odd_sign = b.newVec();
    b.vecOp2i(.sll_u32, odd_sign, ki, 31);
    const odd_mask = b.newVec();
    b.vecOp2i(.sra_i32, odd_mask, odd_sign, 31);

    const num = b.newVec();
    const den = b.newVec();
    pixel.emitSelectF32ByMask(b, num, s, c, odd_mask);
    pixel.emitSelectF32ByMask(b, den, c, s, odd_mask);

    b.vecOp3(.div_f32, num, num, den);
    b.vecOp3(.xor_u32, num, num, odd_sign);
    b.vecOp3(.xor_u32, dst, num, x_sign);
}

/// Cephes 2-breakpoint `atan(x)` (`|z| <= tan(pi/8)`, 4-coeff Horner, 1 division, 0 spills)
/// matching `interp.atanApprox` bit-for-bit on every ISA tier.
pub fn emitAtan(b: *aj.Builder, dst: aj.Reg, x: aj.Reg) void {
    const c_sign = b.constI32Splat(@bitCast(@as(u32, 0x8000_0000)));
    const c_abs = b.constI32Splat(0x7FFF_FFFF);
    const c_one = b.constF32Splat(1.0);
    const x_sign = b.newVec();
    const ax = b.newVec();
    b.vecOp3(.and_u32, x_sign, x, c_sign);
    b.vecOp3(.and_u32, ax, x, c_abs);

    const m_hi = b.newVec();
    const m_mid = b.newVec();
    b.vecCmpCF32(.cmp_gt_f32, m_hi, ax, interp.atan_tan_3pi8_f32);
    b.vecCmpCF32(.cmp_gt_f32, m_mid, ax, interp.atan_tan_pi8_f32);

    const ax_m1 = b.newVec();
    const ax_p1 = b.newVec();
    b.vecOp3(.sub_f32, ax_m1, ax, c_one);
    b.vecOp3(.add_f32, ax_p1, ax, c_one);

    const num = b.newVec();
    const den = b.newVec();
    pixel.emitSelectF32ByMask(b, num, ax, ax_m1, m_mid);
    pixel.emitSelectF32ByMask(b, num, num, b.constF32Splat(-1.0), m_hi);
    pixel.emitSelectF32ByMask(b, den, c_one, ax_p1, m_mid);
    pixel.emitSelectF32ByMask(b, den, den, ax, m_hi);

    const z = b.newVec();
    b.vecOp3(.div_f32, z, num, den);
    const z2 = b.newVec();
    b.vecOp3(.mul_f32, z2, z, z);

    const p = b.newVec();
    b.vecFmaCCF32(p, z2, interp.atan_c3_f32, interp.atan_c2_f32);
    b.vecFmaBCF32(p, p, z2, interp.atan_c1_f32);
    b.vecFmaBCF32(p, p, z2, interp.atan_c0_f32);

    b.vecOp3(.mul_f32, p, p, z2);
    emitFma(b, p, p, z, z);

    // pi/4 + pi/4 == pi/2 bit-exact in f32 -> saves 1 hoisted const reg and 1 blendv.
    const c_pi4 = b.constF32Splat(std.math.pi * 0.25);
    const off1 = b.newVec();
    const off2 = b.newVec();
    b.vecAndF32(off1, m_mid, c_pi4);
    b.vecAndF32(off2, m_hi, c_pi4);
    b.vecOp3(.add_f32, off1, off1, off2);
    b.vecOp3(.add_f32, p, off1, p);
    b.vecOp3(.xor_u32, dst, p, x_sign);
}

/// Direct division-free Cephes `asin(x)` (4-coeff Horner hidden behind sqrt, 0 divisions)
/// matching `interp.asinApprox` bit-for-bit on every ISA tier.
pub fn emitAsin(b: *aj.Builder, dst: aj.Reg, x: aj.Reg) void {
    const c_sign = b.constI32Splat(@bitCast(@as(u32, 0x8000_0000)));
    const c_abs = b.constI32Splat(0x7FFF_FFFF);
    const c_half = b.constF32Splat(0.5);
    const x_sign = b.newVec();
    const ax = b.newVec();
    b.vecOp3(.and_u32, x_sign, x, c_sign);
    b.vecOp3(.and_u32, ax, x, c_abs);

    const hi = b.newVec();
    b.vecOp3(.cmp_gt_f32, hi, ax, c_half);

    const z_lo = b.newVec();
    const z_hi = b.newVec();
    b.vecOp3(.mul_f32, z_lo, ax, ax);
    b.vecFnmaBCF32(z_hi, ax, 0.5, c_half);

    const z = b.newVec();
    pixel.emitSelectF32ByMask(b, z, z_lo, z_hi, hi);
    const s = b.newVec();
    b.vecOp2(.sqrt_f32, s, z);

    const p = b.newVec();
    b.vecFmaCCF32(p, z, interp.asin_c3_f32, interp.asin_c2_f32);
    b.vecFmaBCF32(p, p, z, interp.asin_c1_f32);
    b.vecFmaBCF32(p, p, z, interp.asin_c0_f32);

    const t = b.newVec();
    pixel.emitSelectF32ByMask(b, t, ax, s, hi);
    const r = b.newVec();
    b.vecOp3(.mul_f32, r, t, z);
    emitFma(b, r, r, p, t);

    const r2 = b.newVec();
    b.vecOp3(.add_f32, r2, r, r);
    const hi_val = b.newVec();
    b.vecOp3(.sub_f32, hi_val, b.constF32Splat(std.math.pi * 0.5), r2);
    const res = b.newVec();
    pixel.emitSelectF32ByMask(b, res, r, hi_val, hi);
    b.vecOp3(.xor_u32, dst, res, x_sign);
}

/// Direct division-free Cephes `acos(x)` (4-coeff Horner hidden behind sqrt, 0 divisions)
/// matching `interp.acosApprox` bit-for-bit on every ISA tier.
pub fn emitAcos(b: *aj.Builder, dst: aj.Reg, x: aj.Reg) void {
    const c_abs = b.constI32Splat(0x7FFF_FFFF);
    const c_half = b.constF32Splat(0.5);
    const ax = b.newVec();
    b.vecOp3(.and_u32, ax, x, c_abs);

    const hi = b.newVec();
    b.vecOp3(.cmp_gt_f32, hi, ax, c_half);

    const z_lo = b.newVec();
    const z_hi = b.newVec();
    b.vecOp3(.mul_f32, z_lo, ax, ax);
    b.vecFnmaBCF32(z_hi, ax, 0.5, c_half);

    const z = b.newVec();
    pixel.emitSelectF32ByMask(b, z, z_lo, z_hi, hi);
    const s = b.newVec();
    b.vecOp2(.sqrt_f32, s, z);

    const p = b.newVec();
    b.vecFmaCCF32(p, z, interp.asin_c3_f32, interp.asin_c2_f32);
    b.vecFmaBCF32(p, p, z, interp.asin_c1_f32);
    b.vecFmaBCF32(p, p, z, interp.asin_c0_f32);

    const t = b.newVec();
    pixel.emitSelectF32ByMask(b, t, x, s, hi);
    const r = b.newVec();
    b.vecOp3(.mul_f32, r, t, z);
    emitFma(b, r, r, p, t);

    const r2 = b.newVec();
    b.vecOp3(.add_f32, r2, r, r);
    const pi_m_r2 = b.newVec();
    b.vecOp3(.sub_f32, pi_m_r2, b.constF32Splat(std.math.pi), r2);
    const x_neg = b.newVec();
    b.vecOp2i(.sra_i32, x_neg, x, 31);
    const hi_val = b.newVec();
    pixel.emitSelectF32ByMask(b, hi_val, r2, pi_m_r2, x_neg);

    const lo_val = b.newVec();
    b.vecOp3(.sub_f32, lo_val, b.constF32Splat(std.math.pi * 0.5), r);
    pixel.emitSelectF32ByMask(b, dst, lo_val, hi_val, hi);
}

pub fn emitAtan2(b: *aj.Builder, dst: aj.Reg, y: aj.Reg, x: aj.Reg) void {
    const zero = b.vecZero();
    const c_sign = b.constI32Splat(@bitCast(@as(u32, 0x8000_0000)));
    const c_pi = b.constF32Splat(std.math.pi);
    const c_half_pi = b.constF32Splat(std.math.pi * 0.5);

    const ax = b.newVec();
    const ay = b.newVec();
    b.vecOp2(.abs_f32, ax, x);
    b.vecOp2(.abs_f32, ay, y);

    const mx = b.newVec();
    const mn = b.newVec();
    b.vecOp3(.max_f32, mx, ax, ay);
    b.vecOp3(.min_f32, mn, ax, ay);

    const mx_gt_0 = b.newVec();
    b.vecOp3(.cmp_gt_f32, mx_gt_0, mx, zero);

    const ratio = b.newVec();
    b.vecOp3(.div_f32, ratio, mn, mx);
    b.vecAndF32(ratio, ratio, mx_gt_0);

    const r2 = b.newVec();
    b.vecOp3(.mul_f32, r2, ratio, ratio);

    const p = b.newVec();
    b.vecFmaCCF32(p, r2, 0.0028236389625817537, -0.015956902876496315);
    b.vecFmaBCF32(p, p, r2, 0.042504988610744476);
    b.vecFmaBCF32(p, p, r2, -0.07489009201526642);
    b.vecFmaBCF32(p, p, r2, 0.1063479334115982);
    b.vecFmaBCF32(p, p, r2, -0.1420273631811142);
    b.vecFmaBCF32(p, p, r2, 0.19992695748806);
    b.vecFmaBCF32(p, p, r2, -0.33333101868629456);
    b.vecOp3(.mul_f32, p, p, r2);
    const res = b.newVec();
    emitFma(b, res, p, ratio, ratio);

    // If |y| > |x|: res = pi/2 - res
    const swap_mask = b.newVec();
    b.vecOp3(.cmp_gt_f32, swap_mask, ay, ax);
    const swapped = b.newVec();
    b.vecOp3(.sub_f32, swapped, c_half_pi, res);
    pixel.emitSelectF32ByMask(b, res, res, swapped, swap_mask);

    // If x < 0: res = pi - res
    const x_lt_0 = b.newVec();
    b.vecOp3(.cmp_lt_f32, x_lt_0, x, zero);
    const flipped = b.newVec();
    b.vecOp3(.sub_f32, flipped, c_pi, res);
    pixel.emitSelectF32ByMask(b, res, res, flipped, x_lt_0);

    // If !(mx > 0): res = 0.0
    b.vecAndF32(res, res, mx_gt_0);

    // Apply y's sign bit.
    const y_sign = b.newVec();
    b.vecOp3(.and_u32, y_sign, y, c_sign);
    b.vecOp3(.xor_u32, dst, res, y_sign);
}
