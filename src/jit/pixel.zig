//! Pixel load/store helpers and the single rounding helper (`emitRoundF32`)
//! enforcing D4 (`NaN -> peak` integer stores) and X3/Y4 (SSE2 `-0.0` sign preservation).

const std = @import("std");
const ast = @import("../expr/ast.zig");
const aj = @import("asmjit.zig");

pub const RoundMode = enum {
    trunc,
    floor,
    ceil,
    round_even,
};

/// Single codegen rounding helper used by `trunc`/`floor`/`ceil`/`round`, `%`'s internal `trunc`,
/// and transcendental range reduction.
///
/// On SSE2 baseline (level 1) ONLY, ORs `src`'s sign bit back into the result so `-0.3`,
/// `-0.5`, `-0.0`, `-0 3 %`, and `sin(-0.0)` match SSE4.1+/NEON and the oracle bit-for-bit (X3, Y4).
/// Costs 0 extra instructions on v2+, AVX2+, AVX-512, and AArch64 NEON.
pub fn emitRoundF32(b: *aj.Builder, dst: aj.Reg, src: aj.Reg, mode: RoundMode) void {
    const op: aj.VecOp2 = switch (mode) {
        .trunc => .trunc_f32,
        .floor => .floor_f32,
        .ceil => .ceil_f32,
        .round_even => .round_even_f32,
    };

    if (b.cpu.isa == .sse2) {
        const sign_mask = b.constI32Splat(@bitCast(@as(u32, 0x8000_0000)));
        const saved_sign = b.newVec();
        b.vecOp3(.and_u32, saved_sign, src, sign_mask);
        b.vecOp2(op, dst, src);
        b.vecOp3(.or_u32, dst, dst, saved_sign);
    } else {
        b.vecOp2(op, dst, src);
    }
}

/// Emit `dst = (mask != 0) ? true_val : false_val` in FP domain (`vblendvps` on AVX2, `vpternlogd` on AVX-512, `bsl/bit/bif` on NEON).
pub fn emitSelectF32ByMask(b: *aj.Builder, dst: aj.Reg, false_val: aj.Reg, true_val: aj.Reg, mask: aj.Reg) void {
    b.vecBlendVF32(dst, false_val, true_val, mask);
}

/// Emit `dst = (mask != 0) ? true_val : false_val` where `mask` is all-1s / all-0s per lane.
pub fn emitSelectByMask(b: *aj.Builder, dst: aj.Reg, false_val: aj.Reg, true_val: aj.Reg, mask: aj.Reg) void {
    b.vecOp4(.blendv_u8, dst, false_val, true_val, mask);
}

/// Load `f32_lanes` pixels from `mem` and convert to `f32`.
pub fn emitLoadF32(b: *aj.Builder, dst: aj.Reg, mem: aj.Mem, fmt: ast.SampleFormat) void {
    if (fmt.is_float) {
        if (fmt.bits_per_sample == 16) {
            b.vecCvtF16ToF32(dst, mem);
        } else {
            b.vecLoad(.u_vec, dst, mem);
        }
        return;
    }

    if (fmt.bits_per_sample <= 16) {
        b.vecLoadUintToF32(dst, mem, fmt.bytesPerSample());
        return;
    }
    b.vecLoad(.u_vec, dst, mem);
    b.vecOp2(.cvt_i32_to_f32, dst, dst);
}

/// Load `f32_lanes` integer pixels from `mem` into `i32` (`opt=1`; zero-extended for u8/u16, raw signed i32 for 32-bit).
pub fn emitLoadI32(b: *aj.Builder, dst: aj.Reg, mem: aj.Mem, fmt: ast.SampleFormat) void {
    std.debug.assert(!fmt.is_float);
    if (fmt.bits_per_sample <= 8) {
        b.vecLoad(.cvt_u8_to_u32, dst, mem);
    } else if (fmt.bits_per_sample <= 16) {
        b.vecLoad(.cvt_u16_to_u32, dst, mem);
    } else {
        b.vecLoad(.u_vec, dst, mem);
    }
}

/// Store `f32_lanes` `f32` values to `mem`, using D4's zero-overhead `min(v, peak)`-first
/// order so `NaN -> peak`, `+inf -> peak`, `-inf -> 0` identically on x86 and AArch64.
pub fn emitStoreF32(b: *aj.Builder, mem: aj.Mem, src: aj.Reg, fmt: ast.SampleFormat) void {
    if (fmt.is_float) {
        if (fmt.bits_per_sample == 16) {
            b.vecCvtF32ToF16(mem, src);
        } else {
            b.vecStore(.u_vec, mem, src);
        }
        return;
    }

    const peak = b.constF32Splat(fmt.peakF32());
    const tmp = b.newVec();
    // D4 Step 1: min(src, peak) with src first -> NaN & +inf become peak on both x86 and AArch64.
    b.vecOp3(.min_f32, tmp, src, peak);
    // D4 Step 2: AVX-512 unsigned narrowing (vpmovusdb/vpmovusdw) and 32-bit integer stores
    // (no saturating pack) require >= 0 before i32 conversion.
    // On SSE2/v2/v3/NEON u8/u16, signed-to-unsigned packs already saturate negative i32 (incl. INT32_MIN) to 0!
    if (b.cpu.has_avx512 or fmt.bits_per_sample == 32) {
        b.vecOp3(.max_f32, tmp, tmp, b.vecZero());
    }
    b.vecOp2(.cvt_round_f32_to_i32, tmp, tmp);
    if (fmt.bits_per_sample <= 8) {
        b.vecStoreU8FromI32(mem, tmp);
    } else if (fmt.bits_per_sample <= 16) {
        b.vecStoreU16FromI32(mem, tmp);
    } else {
        b.vecStore(.u_vec, mem, tmp);
    }
}

/// Store `f32_lanes` `i32` values (`opt=1`) clamped to `[0, peak]` into `u8`/`u16`,
/// stored raw (unclamped wrapping signed `i32`, matching `akarin`) into 32-bit integer,
/// or converted to `f32`/`f16` for float outputs.
pub fn emitStoreI32(b: *aj.Builder, mem: aj.Mem, src: aj.Reg, fmt: ast.SampleFormat) void {
    if (fmt.is_float) {
        const tmp = b.newVec();
        b.vecOp2(.cvt_i32_to_f32, tmp, src);
        emitStoreF32(b, mem, tmp, fmt);
        return;
    }
    if (fmt.bits_per_sample == 32) {
        b.vecStore(.u_vec, mem, src);
        return;
    }

    if (!b.cpu.has_avx512 and fmt.bits_per_sample == 16) {
        // SSE/AVX2 `vpackusdw` and NEON `sqxtun` saturate signed i32 directly to [0, 65535].
        b.vecStoreU16FromI32(mem, src);
        return;
    }

    const peak_i = b.constI32Splat(@intCast(fmt.peakInt()));
    const tmp = b.newVec();
    b.vecOp3(.min_i32, tmp, src, peak_i);
    if (b.cpu.has_avx512) {
        b.vecOp3(.max_i32, tmp, tmp, b.vecZero());
    }
    if (fmt.bits_per_sample <= 8) {
        b.vecStoreU8FromI32(mem, tmp);
    } else {
        b.vecStoreU16FromI32(mem, tmp);
    }
}

/// Store `2 * f32_lanes` `i32` values (`opt=1`) to `mem0` and `mem1`.
pub fn emitStoreI32x2(b: *aj.Builder, mem0: aj.Mem, mem1: aj.Mem, src0: aj.Reg, src1: aj.Reg, fmt: ast.SampleFormat, use_pair_mem: bool) void {
    if (fmt.is_float) {
        const f0 = b.newVec();
        const f1 = b.newVec();
        b.vecOp2(.cvt_i32_to_f32, f0, src0);
        b.vecOp2(.cvt_i32_to_f32, f1, src1);
        emitStoreF32x2(b, mem0, mem1, f0, f1, fmt, use_pair_mem);
        return;
    }
    if (fmt.bits_per_sample == 32) {
        if (use_pair_mem) {
            b.vecStoreUVecx2(mem0, src0, src1);
        } else {
            b.vecStore(.u_vec, mem0, src0);
            b.vecStore(.u_vec, mem1, src1);
        }
        return;
    }
    if (fmt.bits_per_sample == 16) {
        // SSE/AVX2/AVX-512 `vpackusdw` and NEON `kPacksI32_U16` saturate signed i32 directly to [0, 65535].
        b.vecStoreU16FromI32x2(mem0, src0, src1);
        return;
    }

    const peak_i = b.constI32Splat(@intCast(fmt.peakInt()));
    const t0 = b.newVec();
    const t1 = b.newVec();
    b.vecOp3(.min_i32, t0, src0, peak_i);
    b.vecOp3(.min_i32, t1, src1, peak_i);
    if (b.cpu.has_avx512 and fmt.bits_per_sample <= 8) {
        const z = b.vecZero();
        b.vecOp3(.max_i32, t0, t0, z);
        b.vecOp3(.max_i32, t1, t1, z);
    }
    if (fmt.bits_per_sample <= 8) {
        b.vecStoreU8FromI32x2(mem0, t0, t1);
    } else {
        b.vecStoreU16FromI32x2(mem0, t0, t1);
    }
}

/// Store `2 * f32_lanes` `f32` values (`src0`, `src1`) to `mem0` and `mem1`,
/// using paired `vpackssdw`/`vpackuswb`/`vpackusdw` for `u8`/`u16` outputs,
/// paired `fcvtn`+`fcvtn2`+`str q` for NEON `f16`, and optional `stp` for NEON `f32`/`u32`.
pub fn emitStoreF32x2(b: *aj.Builder, mem0: aj.Mem, mem1: aj.Mem, src0: aj.Reg, src1: aj.Reg, fmt: ast.SampleFormat, use_pair_mem: bool) void {
    if (fmt.is_float) {
        if (fmt.bits_per_sample == 16) {
            if (b.cpu.isa == .neon) {
                b.vecCvtF32ToF16x2(mem0, src0, src1);
            } else {
                b.vecCvtF32ToF16(mem0, src0);
                b.vecCvtF32ToF16(mem1, src1);
            }
        } else if (use_pair_mem) {
            b.vecStoreUVecx2(mem0, src0, src1);
        } else {
            b.vecStore(.u_vec, mem0, src0);
            b.vecStore(.u_vec, mem1, src1);
        }
        return;
    }

    const peak = b.constF32Splat(fmt.peakF32());
    const t0 = b.newVec();
    const t1 = b.newVec();
    b.vecOp3(.min_f32, t0, src0, peak);
    b.vecOp3(.min_f32, t1, src1, peak);
    if ((b.cpu.has_avx512 and fmt.bits_per_sample <= 8) or fmt.bits_per_sample == 32) {
        const z = b.vecZero();
        b.vecOp3(.max_f32, t0, t0, z);
        b.vecOp3(.max_f32, t1, t1, z);
    }
    b.vecOp2(.cvt_round_f32_to_i32, t0, t0);
    b.vecOp2(.cvt_round_f32_to_i32, t1, t1);
    if (fmt.bits_per_sample <= 8) {
        b.vecStoreU8FromI32x2(mem0, t0, t1);
    } else if (fmt.bits_per_sample <= 16) {
        b.vecStoreU16FromI32x2(mem0, t0, t1);
    } else if (use_pair_mem) {
        b.vecStoreUVecx2(mem0, t0, t1);
    } else {
        b.vecStore(.u_vec, mem0, t0);
        b.vecStore(.u_vec, mem1, t1);
    }
}
