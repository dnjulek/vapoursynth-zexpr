//! Provably-in-range unsigned (`u8`/`u16`, zero unpack/pack) and signed (`i16`/`i32`, 2x unrolled
//! with paired store packing and NEON widening `uaddl`/`uaddw`/`usubl`/`usubw`) SIMD interior fast
//! paths for integer clips (Y1, Y2).
//!
//! Optional final `/ 2^k` uses exact round-half-even
//! `(v + 2^(k-1) - 1 + ((v >> k) & 1)) >> k` guarded by `analysis.zig`'s Y1 check.

const std = @import("std");
const ast = @import("../expr/ast.zig");
const analysis_mod = @import("../expr/analysis.zig");
const interp = @import("../expr/interp.zig");
const aj = @import("asmjit.zig");
const pixel = @import("pixel.zig");

const RegSlot = struct {
    reg: aj.Reg,
    owned: bool,
};

fn constU8Splat(b: *aj.Builder, val: u8) aj.Reg {
    const packed_u32: u32 = (@as(u32, val) << 24) | (@as(u32, val) << 16) | (@as(u32, val) << 8) | val;
    return b.constI32Splat(@bitCast(packed_u32));
}

fn constU16Splat(b: *aj.Builder, val: u16) aj.Reg {
    const packed_u32: u32 = (@as(u32, val) << 16) | val;
    return b.constI32Splat(@bitCast(packed_u32));
}

fn constI16Splat(b: *aj.Builder, val: i16) aj.Reg {
    return constU16Splat(b, @bitCast(val));
}

fn acquireVec(b: *aj.Builder, pool: *std.ArrayList(aj.Reg)) aj.Reg {
    if (!b.cpu.has32VecRegs() and pool.items.len > 0) return pool.pop().?;
    return b.newVec();
}

fn releaseUse(
    allocator: std.mem.Allocator,
    pool: *std.ArrayList(aj.Reg),
    rem_uses: []u32,
    slots: []?RegSlot,
    idx: ast.NodeIdx,
) std.mem.Allocator.Error!void {
    if (rem_uses[idx] == 0) return;
    rem_uses[idx] -= 1;
    if (rem_uses[idx] == 0) {
        if (slots[idx]) |s| {
            if (s.owned) {
                try pool.append(allocator, s.reg);
                slots[idx] = null;
            }
        }
    }
}

// ============================================================================
// Parameterized native unsigned fast paths (`u8` and `u16`)
// ============================================================================

const UnsignedWidth = enum {
    u8,
    u16,

    fn Int(comptime self: UnsignedWidth) type {
        return if (self == .u8) u8 else u16;
    }

    fn shift(comptime self: UnsignedWidth) u8 {
        return @intFromBool(self == .u16);
    }

    fn elemBytes(comptime self: UnsignedWidth) i32 {
        return if (self == .u8) 1 else 2;
    }

    fn splat(comptime self: UnsignedWidth, b: *aj.Builder, val: self.Int()) aj.Reg {
        return if (self == .u8) constU8Splat(b, val) else constU16Splat(b, val);
    }

    /// The `<base>_u8` / `<base>_u16` op (`min`, `max`, `adds`, `subs`).
    fn op3(comptime self: UnsignedWidth, comptime base: []const u8) aj.VecOp3 {
        return @field(aj.VecOp3, base ++ "_" ++ @tagName(self));
    }

    fn subOp(comptime self: UnsignedWidth) aj.VecOp3 {
        return if (self == .u8) .sub_u8 else .sub_i16;
    }
};

/// Memory operand of a `.load_clip`/`.load_rel` source pixel at horizontal pixel offset `disp_px`,
/// for source samples of `1 << shift` bytes.
fn srcLoadMem(
    comptime shift: u8,
    node: ast.Node,
    w: i32,
    disp_px: i32,
    src_ptrs: []const ?aj.Reg,
    resolver: anytype,
) aj.Mem {
    const scale: i32 = 1 << shift;
    return switch (node) {
        .load_clip => |clip| resolver.rowMem(src_ptrs[clip].?, shift, disp_px * scale),
        .load_rel => |rel| blk: {
            const row_ptr = resolver.rowPtrFor(rel.clip, rel.dy, rel.mode);
            const eff_dx: i32 = interp.clampRelOffset(rel.dx, w);
            break :blk resolver.rowMem(row_ptr, shift, (eff_dx + disp_px) * scale);
        },
        else => unreachable,
    };
}

/// Emit native `u8` (`vec_bytes` px/iter) or `u16` (`vec_bytes / 2` px/iter) SIMD evaluation at
/// horizontal pixel offset `col_disp`, folding single-use loads into AVX2/AVX-512 memory operands.
fn emitUnsignedInteriorAt(
    comptime W: UnsignedWidth,
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    width: usize,
    col_disp: i32,
    dst_fmt: ast.SampleFormat,
    dst_ptr: aj.Reg,
    src_row_ptrs: []const ?aj.Reg,
    row_ptr_resolver: anytype,
    free_vecs: *std.ArrayList(aj.Reg),
) std.mem.Allocator.Error!void {
    const eval_root = prog.root;
    const n_nodes = prog.nodes.items.len;
    const slots = try allocator.alloc(?RegSlot, n_nodes);
    defer allocator.free(slots);
    @memset(slots, null);

    // Detect `abs(a - b)` patterns so `sub` inside `abs` is emitted as `max(a, b) - min(a, b)`.
    const skip_sub = try allocator.alloc(bool, n_nodes);
    defer allocator.free(skip_sub);
    @memset(skip_sub, false);

    const rem_uses = try allocator.alloc(u32, n_nodes);
    defer allocator.free(rem_uses);
    @memset(rem_uses, 0);
    rem_uses[eval_root] = 1;
    var ri: usize = eval_root + 1;
    while (ri > 0) {
        ri -= 1;
        if (rem_uses[ri] == 0) continue;
        switch (prog.nodes.items[ri]) {
            .unary => |u| {
                const child = prog.nodes.items[u.a];
                if (u.op == .abs and child == .binary and child.binary.op == .sub and info.use_counts.items[u.a] == 1) {
                    skip_sub[u.a] = true;
                    rem_uses[child.binary.a] += 1;
                    rem_uses[child.binary.b] += 1;
                } else {
                    rem_uses[u.a] += 1;
                }
            },
            .binary => |bin| {
                rem_uses[bin.a] += 1;
                rem_uses[bin.b] += 1;
            },
            else => {},
        }
    }

    // On AVX2/AVX-512, fold single-use loads directly into VEX/EVEX memory operands (zero alignment req).
    const folded_load = try allocator.alloc(bool, n_nodes);
    defer allocator.free(folded_load);
    @memset(folded_load, false);
    if (b.cpu.vec_bytes >= 32) {
        for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
            if (rem_uses[idx] == 0 or skip_sub[idx]) continue;
            if (node != .binary) continue;
            const bin = node.binary;
            const is_comm = switch (bin.op) {
                .min, .max, .add, .bitand, .bitor, .bitxor => true,
                .sub => false,
                else => continue,
            };
            const b_is_load = rem_uses[bin.b] == 1 and !folded_load[bin.b] and
                (prog.nodes.items[bin.b] == .load_clip or prog.nodes.items[bin.b] == .load_rel);
            if (b_is_load and bin.a != bin.b) {
                folded_load[bin.b] = true;
            } else if (is_comm) {
                const a_is_load = rem_uses[bin.a] == 1 and !folded_load[bin.a] and
                    (prog.nodes.items[bin.a] == .load_clip or prog.nodes.items[bin.a] == .load_rel);
                if (a_is_load and bin.a != bin.b) {
                    folded_load[bin.a] = true;
                }
            }
        }
    }

    const w_i32: i32 = @intCast(width);

    for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
        if (rem_uses[idx] == 0 or skip_sub[idx] or folded_load[idx]) continue;
        switch (node) {
            .const_f32 => |v| {
                const uv: W.Int() = @intFromFloat(v);
                slots[idx] = .{ .reg = W.splat(b, uv), .owned = false };
            },
            .const_i32 => |v| {
                const uv: W.Int() = @intCast(v);
                slots[idx] = .{ .reg = W.splat(b, uv), .owned = false };
            },
            .load_clip, .load_rel => {
                const dst = acquireVec(b, free_vecs);
                b.vecLoad(.u_vec, dst, srcLoadMem(W.shift(), node, w_i32, col_disp, src_row_ptrs, row_ptr_resolver));
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .unary => |u| {
                // Only `abs(a - b)` reaches here: `max(a, b) - min(a, b)`.
                const sb = prog.nodes.items[u.a].binary;
                const a_reg = slots[sb.a].?.reg;
                const b_reg = slots[sb.b].?.reg;
                const mx = acquireVec(b, free_vecs);
                const mn = acquireVec(b, free_vecs);
                b.vecOp3(W.op3("max"), mx, a_reg, b_reg);
                b.vecOp3(W.op3("min"), mn, a_reg, b_reg);
                b.vecOp3(W.subOp(), mx, mx, mn);
                try free_vecs.append(allocator, mn);
                try releaseUse(allocator, free_vecs, rem_uses, slots, sb.a);
                try releaseUse(allocator, free_vecs, rem_uses, slots, sb.b);
                slots[idx] = .{ .reg = mx, .owned = true };
            },
            .binary => |bin| {
                const dst = acquireVec(b, free_vecs);
                const vop: aj.VecOp3 = switch (bin.op) {
                    .min => W.op3("min"),
                    .max => W.op3("max"),
                    .add => W.op3("adds"),
                    .sub => W.op3("subs"),
                    .bitand => .and_u32,
                    .bitor => .or_u32,
                    .bitxor => .xor_u32,
                    else => unreachable,
                };
                if (folded_load[bin.b]) {
                    const a_reg = slots[bin.a].?.reg;
                    const mem = srcLoadMem(W.shift(), prog.nodes.items[bin.b], w_i32, col_disp, src_row_ptrs, row_ptr_resolver);
                    b.vecOp3Mem(vop, dst, a_reg, mem);
                } else if (folded_load[bin.a]) {
                    const b_reg = slots[bin.b].?.reg;
                    const mem = srcLoadMem(W.shift(), prog.nodes.items[bin.a], w_i32, col_disp, src_row_ptrs, row_ptr_resolver);
                    b.vecOp3Mem(vop, dst, b_reg, mem);
                } else {
                    const a_reg = slots[bin.a].?.reg;
                    const b_reg = slots[bin.b].?.reg;
                    b.vecOp3(vop, dst, a_reg, b_reg);
                }
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.a);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.b);
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            else => unreachable,
        }
    }

    var res = slots[eval_root].?.reg;
    var res_owned = slots[eval_root].?.owned;
    if (W == .u16 and dst_fmt.bits_per_sample < 16) {
        const clamped = acquireVec(b, free_vecs);
        b.vecOp3(.min_u16, clamped, res, constU16Splat(b, @intCast(dst_fmt.peakInt())));
        if (res_owned) try free_vecs.append(allocator, res);
        res = clamped;
        res_owned = true;
    }
    b.vecStore(.u_vec, row_ptr_resolver.rowMem(dst_ptr, W.shift(), col_disp * W.elemBytes()), res);
    if (res_owned) try free_vecs.append(allocator, res);
}

/// The native integer interior bodies `codegen` dispatches to (`u8`/`u16` native, `i16`/`i32` signed).
pub const InteriorKind = enum { u8, u16, i16, i32 };

/// Emit one interior iteration of `kind` at `renv`'s current column, 2x unrolled when `unroll2`.
/// `renv` supplies `dst_ptr`, `src_row_ptrs`, `use_running_ptrs` and `rowMem` (codegen's `BodyEnv`).
pub fn emitInteriorBody(
    kind: InteriorKind,
    unroll2: bool,
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    width: usize,
    dst_fmt: ast.SampleFormat,
    renv: anytype,
) std.mem.Allocator.Error!void {
    switch (kind) {
        .u8 => try emitUnsignedBody(.u8, unroll2, allocator, b, prog, info, width, .{ .is_float = false, .bits_per_sample = 8 }, renv),
        .u16 => try emitUnsignedBody(.u16, unroll2, allocator, b, prog, info, width, dst_fmt, renv),
        .i16 => try emitSignedBody(.i16, unroll2, allocator, b, prog, info, width, dst_fmt, renv),
        .i32 => try emitSignedBody(.i32, unroll2, allocator, b, prog, info, width, dst_fmt, renv),
    }
}

/// Native `u8` (`vec_bytes` px/iter) or `u16` (`vec_bytes / 2` px/iter) SIMD evaluation, zero unpack/pack;
/// `unroll2` emits a second vector one `vec_bytes` further on, sharing the free-register pool.
fn emitUnsignedBody(
    comptime W: UnsignedWidth,
    unroll2: bool,
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    width: usize,
    dst_fmt: ast.SampleFormat,
    renv: anytype,
) std.mem.Allocator.Error!void {
    var free_vecs: std.ArrayList(aj.Reg) = .empty;
    defer free_vecs.deinit(allocator);
    try emitUnsignedInteriorAt(W, allocator, b, prog, info, width, 0, dst_fmt, renv.dst_ptr, renv.src_row_ptrs, renv, &free_vecs);
    if (unroll2) {
        const step_px: i32 = @divExact(@as(i32, @intCast(b.cpu.vec_bytes)), W.elemBytes());
        try emitUnsignedInteriorAt(W, allocator, b, prog, info, width, step_px, dst_fmt, renv.dst_ptr, renv.src_row_ptrs, renv, &free_vecs);
    }
}

// ============================================================================
// Parameterized signed integer fast paths (`i16` for u8, `i32` for u9..u16)
// ============================================================================

const SignedWidth = enum {
    i16,
    i32,

    fn Int(comptime self: SignedWidth) type {
        return if (self == .i16) i16 else i32;
    }

    fn UnsignedInt(comptime self: SignedWidth) type {
        return if (self == .i16) u16 else u32;
    }

    fn Shift(comptime self: SignedWidth) type {
        return std.math.Log2Int(self.Int());
    }

    fn srcShift(comptime self: SignedWidth) u8 {
        return @intFromBool(self == .i32);
    }

    fn splat(comptime self: SignedWidth, b: *aj.Builder, val: self.Int()) aj.Reg {
        return if (self == .i16) constI16Splat(b, val) else b.constI32Splat(val);
    }

    fn loadCvtOp(comptime self: SignedWidth) aj.VecLoad {
        return if (self == .i16) .cvt_u8_to_u16 else .cvt_u16_to_u32;
    }

    /// The `<base>_i16` / `<base>_i32` op (`add`, `sub`, `mul`, `min`, `max`, `cmp_gt`, `cmp_eq`).
    fn op3(comptime self: SignedWidth, comptime base: []const u8) aj.VecOp3 {
        return @field(aj.VecOp3, base ++ "_" ++ @tagName(self));
    }

    /// The `<base>_i16` / `<base>_i32` op (`neg`, `abs`).
    fn op2(comptime self: SignedWidth, comptime base: []const u8) aj.VecOp2 {
        return @field(aj.VecOp2, base ++ "_" ++ @tagName(self));
    }

    fn sllOp(comptime self: SignedWidth) aj.VecOp2i {
        return if (self == .i16) .sll_u16 else .sll_u32;
    }

    fn sraOp(comptime self: SignedWidth) aj.VecOp2i {
        return if (self == .i16) .sra_i16 else .sra_i32;
    }

    fn allowsRematOnAvx512(comptime self: SignedWidth) bool {
        // On AVX-512, rematerializing `vpmovzxwd` (`i32`) eliminates 115 stack spills in 5x5 Kirsch `p16`,
        // whereas rematerializing `vpmovzxbw` (`i16`) saturates Zen 4 FP2 without reducing spills.
        return self == .i32;
    }

    fn planEvalRoot(comptime self: SignedWidth, info: *const analysis_mod.Analysis) ast.NodeIdx {
        return if (self == .i16) info.int16_plan.eval_root else info.int32_plan.eval_root;
    }

    fn planFinalShift(comptime self: SignedWidth, info: *const analysis_mod.Analysis) u5 {
        return if (self == .i16) info.int16_plan.final_div_shift else info.int32_plan.final_div_shift;
    }

    fn neonLoadWidenX2(comptime self: SignedWidth, b: *aj.Builder, d0: aj.Reg, d1: aj.Reg, mem: aj.Mem) void {
        if (self == .i16) b.vecLoadU8ToU16x2(d0, d1, mem) else b.vecLoadU16ToU32x2(d0, d1, mem);
    }

    fn neonUaddl(comptime self: SignedWidth, b: *aj.Builder, d0: aj.Reg, d1: aj.Reg, a: aj.Reg, rhs: aj.Reg) void {
        if (self == .i16) b.vecUaddlU8ToI16x2(d0, d1, a, rhs) else b.vecUaddlU16ToI32x2(d0, d1, a, rhs);
    }

    fn neonUsubl(comptime self: SignedWidth, b: *aj.Builder, d0: aj.Reg, d1: aj.Reg, a: aj.Reg, rhs: aj.Reg) void {
        if (self == .i16) b.vecUsublU8ToI16x2(d0, d1, a, rhs) else b.vecUsublU16ToI32x2(d0, d1, a, rhs);
    }

    fn neonUaddw(comptime self: SignedWidth, b: *aj.Builder, d0: aj.Reg, d1: aj.Reg, a0: aj.Reg, a1: aj.Reg, rhs: aj.Reg) void {
        if (self == .i16) b.vecUaddwU8ToI16x2(d0, d1, a0, a1, rhs) else b.vecUaddwU16ToI32x2(d0, d1, a0, a1, rhs);
    }

    fn neonUsubw(comptime self: SignedWidth, b: *aj.Builder, d0: aj.Reg, d1: aj.Reg, a0: aj.Reg, a1: aj.Reg, rhs: aj.Reg) void {
        if (self == .i16) b.vecUsubwU8ToI16x2(d0, d1, a0, a1, rhs) else b.vecUsubwU16ToI32x2(d0, d1, a0, a1, rhs);
    }
};

fn exactInt(comptime W: SignedWidth, node: ast.Node) ?W.Int() {
    return switch (node) {
        .const_f32 => |v| blk: {
            if (!std.math.isFinite(v) or @trunc(v) != v) break :blk null;
            if (W == .i16) {
                if (v < @as(f32, std.math.minInt(i16)) or v > @as(f32, std.math.maxInt(i16))) break :blk null;
            } else {
                if (v < -2147483648.0 or v >= 2147483648.0) break :blk null;
            }
            break :blk @intFromFloat(v);
        },
        .const_i32 => |v| if (W == .i16)
            (if (v >= std.math.minInt(i16) and v <= std.math.maxInt(i16)) @intCast(v) else null)
        else
            v,
        else => null,
    };
}

fn pow2Shift(comptime W: SignedWidth, val: W.Int()) ?W.Shift() {
    if (val <= 0) return null;
    const u: W.UnsignedInt() = @intCast(val);
    if (!std.math.isPowerOfTwo(u)) return null;
    return @intCast(@ctz(u));
}

fn emitMulSigned(
    comptime W: SignedWidth,
    b: *aj.Builder,
    dst: aj.Reg,
    lhs_idx: ast.NodeIdx,
    rhs_idx: ast.NodeIdx,
    lhs_reg: aj.Reg,
    rhs_reg: aj.Reg,
    nodes: []const ast.Node,
) void {
    const min_int = std.math.minInt(W.Int());
    const operands = [2]struct { idx: ast.NodeIdx, other_reg: aj.Reg }{
        .{ .idx = rhs_idx, .other_reg = lhs_reg },
        .{ .idx = lhs_idx, .other_reg = rhs_reg },
    };
    for (operands) |op| {
        if (exactInt(W, nodes[op.idx])) |iv| {
            if (pow2Shift(W, iv)) |sh| {
                if (sh == 0) {
                    b.vecOp2(.mov, dst, op.other_reg);
                } else if (sh == 1) {
                    // On Zen 3/4, `vpaddw`/`vpaddd` runs on all 4 FP pipes whereas shifts compete on FP2.
                    b.vecOp3(W.op3("add"), dst, op.other_reg, op.other_reg);
                } else {
                    b.vecOp2i(W.sllOp(), dst, op.other_reg, sh);
                }
                return;
            }
            if (iv < 0 and iv > min_int) {
                if (pow2Shift(W, -iv)) |sh| {
                    if (sh == 0) {
                        b.vecOp2(W.op2("neg"), dst, op.other_reg);
                    } else if (sh == 1) {
                        b.vecOp3(W.op3("add"), dst, op.other_reg, op.other_reg);
                        b.vecOp2(W.op2("neg"), dst, dst);
                    } else {
                        b.vecOp2i(W.sllOp(), dst, op.other_reg, sh);
                        b.vecOp2(W.op2("neg"), dst, dst);
                    }
                    return;
                }
            }
        }
    }
    b.vecOp3(W.op3("mul"), dst, lhs_reg, rhs_reg);
}

fn isBool01Node(node: ast.Node) bool {
    return switch (node) {
        .unary => |u| u.op == .not,
        .binary => |bin| switch (bin.op) {
            .lt, .le, .gt, .ge, .eq, .and_op, .or_op, .xor_op => true,
            else => false,
        },
        else => false,
    };
}

fn isFusableBool(nodes: []const ast.Node, rem_uses: []const u32, idx: ast.NodeIdx) bool {
    if (rem_uses[idx] != 1) return false;
    return isBool01Node(nodes[idx]);
}

fn markFusableBoolTree(nodes: []const ast.Node, rem_uses: []const u32, fused: []bool, idx: ast.NodeIdx) void {
    if (!isFusableBool(nodes, rem_uses, idx)) return;
    fused[idx] = true;
    switch (nodes[idx]) {
        .binary => |bin| switch (bin.op) {
            .and_op, .or_op, .xor_op => {
                markFusableBoolTree(nodes, rem_uses, fused, bin.a);
                markFusableBoolTree(nodes, rem_uses, fused, bin.b);
            },
            else => {},
        },
        else => {},
    }
}

fn computeRemUses(allocator: std.mem.Allocator, nodes: []const ast.Node, eval_root: ast.NodeIdx) std.mem.Allocator.Error![]u32 {
    const rem_uses = try allocator.alloc(u32, nodes.len);
    @memset(rem_uses, 0);
    rem_uses[eval_root] = 1;
    var ri: usize = eval_root + 1;
    while (ri > 0) {
        ri -= 1;
        if (rem_uses[ri] == 0) continue;
        switch (nodes[ri]) {
            .unary => |u| rem_uses[u.a] += 1,
            .binary => |bin| {
                rem_uses[bin.a] += 1;
                rem_uses[bin.b] += 1;
            },
            .ternary => |t| {
                rem_uses[t.a] += 1;
                rem_uses[t.b] += 1;
                rem_uses[t.c] += 1;
            },
            else => {},
        }
    }
    return rem_uses;
}

fn computeFusedBool(allocator: std.mem.Allocator, nodes: []const ast.Node, eval_root: ast.NodeIdx, rem_uses: []const u32) std.mem.Allocator.Error![]bool {
    const fused_bool = try allocator.alloc(bool, nodes.len);
    @memset(fused_bool, false);
    for (nodes[0 .. eval_root + 1], 0..) |node, idx| {
        if (rem_uses[idx] == 0) continue;
        if (node == .ternary and node.ternary.op == .select) {
            markFusableBoolTree(nodes, rem_uses, fused_bool, node.ternary.a);
        }
    }
    return fused_bool;
}

/// Emit the all-ones/zero lane mask of `a <op> rhs` for a comparison `op` (`gt`/`lt`/`eq`/`le`/`ge`).
fn emitCmpMask(comptime W: SignedWidth, b: *aj.Builder, dst: aj.Reg, op: ast.BinaryOp, a: aj.Reg, rhs: aj.Reg) void {
    switch (op) {
        .gt, .le => b.vecOp3(W.op3("cmp_gt"), dst, a, rhs),
        .lt, .ge => b.vecOp3(W.op3("cmp_gt"), dst, rhs, a),
        .eq => b.vecOp3(W.op3("cmp_eq"), dst, a, rhs),
        else => unreachable,
    }
    if (op == .le or op == .ge) b.vecOp2(.not_u32, dst, dst);
}

fn logicOp(op: ast.BinaryOp) aj.VecOp3 {
    return switch (op) {
        .and_op => .and_u32,
        .or_op => .or_u32,
        .xor_op => .xor_u32,
        else => unreachable,
    };
}

/// Finish `fmadd`/`fmsub`/`fnmadd`/`fnmsub` on the product already in `dst` with addend `c`.
fn emitFmaTail(comptime W: SignedWidth, b: *aj.Builder, op: ast.TernaryOp, dst: aj.Reg, c: aj.Reg) void {
    switch (op) {
        .fmadd => b.vecOp3(W.op3("add"), dst, dst, c),
        .fmsub => b.vecOp3(W.op3("sub"), dst, dst, c),
        .fnmadd => b.vecOp3(W.op3("sub"), dst, c, dst),
        .fnmsub => {
            b.vecOp2(W.op2("neg"), dst, dst);
            b.vecOp3(W.op3("sub"), dst, dst, c);
        },
        .select, .linstep => unreachable,
    }
}

fn emitCmpOrLogicMaskSigned(
    comptime W: SignedWidth,
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    nodes: []const ast.Node,
    slots: []?RegSlot,
    rem_uses: []u32,
    fused: []const bool,
    free_vecs: *std.ArrayList(aj.Reg),
    idx: ast.NodeIdx,
) std.mem.Allocator.Error!aj.Reg {
    const mask = acquireVec(b, free_vecs);
    if (!fused[idx]) {
        b.vecOp3(W.op3("cmp_gt"), mask, slots[idx].?.reg, b.vecZero());
        try releaseUse(allocator, free_vecs, rem_uses, slots, idx);
        return mask;
    }
    switch (nodes[idx]) {
        .unary => |u| {
            // `.not`: `!(a > 0)`, i.e. `a <= 0`.
            emitCmpMask(W, b, mask, .le, slots[u.a].?.reg, b.vecZero());
            try releaseUse(allocator, free_vecs, rem_uses, slots, u.a);
            return mask;
        },
        .binary => |bin| switch (bin.op) {
            .gt, .lt, .eq, .le, .ge => {
                emitCmpMask(W, b, mask, bin.op, slots[bin.a].?.reg, slots[bin.b].?.reg);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.a);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.b);
                return mask;
            },
            .and_op, .or_op, .xor_op => {
                const ma = try emitCmpOrLogicMaskSigned(W, allocator, b, nodes, slots, rem_uses, fused, free_vecs, bin.a);
                const mb = try emitCmpOrLogicMaskSigned(W, allocator, b, nodes, slots, rem_uses, fused, free_vecs, bin.b);
                b.vecOp3(logicOp(bin.op), mask, ma, mb);
                try free_vecs.append(allocator, ma);
                try free_vecs.append(allocator, mb);
                return mask;
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

/// Build a Sethi-Ullman / minimum-register-pressure topological schedule for large integer stencils
/// on 32-register ISAs (AArch64 NEON and AVX-512, e.g. 5x5 Kirsch `p8`/`p16`), or RPN order otherwise.
fn buildIntSchedule(
    allocator: std.mem.Allocator,
    prog: *const ast.Program,
    eval_root: ast.NodeIdx,
    rem_uses: []const u32,
    fused_bool: []const bool,
    use_su: bool,
) std.mem.Allocator.Error!std.ArrayList(ast.NodeIdx) {
    const n_nodes = prog.nodes.items.len;
    var schedule: std.ArrayList(ast.NodeIdx) = .empty;
    errdefer schedule.deinit(allocator);
    try schedule.ensureTotalCapacity(allocator, eval_root + 1);

    if (!use_su) {
        for (0..eval_root + 1) |i| {
            if (rem_uses[i] == 0 or fused_bool[i]) continue;
            try schedule.append(allocator, @intCast(i));
        }
        return schedule;
    }

    const su = try allocator.alloc(u32, n_nodes);
    defer allocator.free(su);
    @memset(su, 1);
    for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
        if (rem_uses[idx] == 0) continue;
        su[idx] = switch (node) {
            .const_f32, .const_i32 => 0,
            .load_clip, .load_rel => 1,
            .unary => |u| @max(1, su[u.a]),
            .binary => |bin| blk: {
                const sa = su[bin.a];
                const sb = su[bin.b];
                break :blk if (sa == sb) sa + 1 else @max(sa, sb);
            },
            .ternary => |t| blk: {
                var vals = [3]u32{ su[t.a], su[t.b], su[t.c] };
                if (vals[0] < vals[1]) std.mem.swap(u32, &vals[0], &vals[1]);
                if (vals[1] < vals[2]) std.mem.swap(u32, &vals[1], &vals[2]);
                if (vals[0] < vals[1]) std.mem.swap(u32, &vals[0], &vals[1]);
                break :blk @max(vals[0], @max(vals[1] + 1, vals[2] + 2));
            },
            else => 1,
        };
    }

    const Visit = enum(u8) { unvisited, expanded, emitted };
    const vis = try allocator.alloc(Visit, n_nodes);
    defer allocator.free(vis);
    @memset(vis, .unvisited);

    const Frame = struct { idx: ast.NodeIdx, post: bool };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(allocator);
    var deps: std.ArrayList(ast.NodeIdx) = .empty;
    defer deps.deinit(allocator);
    var fused_walk: std.ArrayList(ast.NodeIdx) = .empty;
    defer fused_walk.deinit(allocator);

    if (rem_uses[eval_root] > 0 and !fused_bool[eval_root]) {
        try stack.append(allocator, .{ .idx = eval_root, .post = false });
    }
    while (stack.items.len > 0) {
        const top = stack.pop().?;
        const idx = top.idx;
        if (top.post) {
            if (vis[idx] != .emitted) {
                vis[idx] = .emitted;
                try schedule.append(allocator, idx);
            }
            continue;
        }
        if (vis[idx] != .unvisited) continue;
        vis[idx] = .expanded;
        try stack.append(allocator, .{ .idx = idx, .post = true });

        deps.clearRetainingCapacity();
        switch (prog.nodes.items[idx]) {
            .unary => |u| try deps.append(allocator, u.a),
            .binary => |bin| {
                try deps.append(allocator, bin.a);
                try deps.append(allocator, bin.b);
            },
            .ternary => |t| {
                if (t.op == .select and fused_bool[t.a]) {
                    fused_walk.clearRetainingCapacity();
                    try fused_walk.append(allocator, t.a);
                    while (fused_walk.items.len > 0) {
                        const fidx = fused_walk.pop().?;
                        if (!fused_bool[fidx]) {
                            try deps.append(allocator, fidx);
                            continue;
                        }
                        switch (prog.nodes.items[fidx]) {
                            .unary => |fu| try fused_walk.append(allocator, fu.a),
                            .binary => |fbin| {
                                try fused_walk.append(allocator, fbin.a);
                                try fused_walk.append(allocator, fbin.b);
                            },
                            else => {},
                        }
                    }
                } else {
                    try deps.append(allocator, t.a);
                }
                try deps.append(allocator, t.b);
                try deps.append(allocator, t.c);
            },
            else => {},
        }

        var write_i: usize = 0;
        for (deps.items) |d| {
            if (vis[d] == .unvisited and rem_uses[d] > 0 and !fused_bool[d]) {
                deps.items[write_i] = d;
                write_i += 1;
            }
        }
        const slice = deps.items[0..write_i];
        var si: usize = 1;
        while (si < slice.len) : (si += 1) {
            const key = slice[si];
            const key_su = su[key];
            var sj: usize = si;
            while (sj > 0 and su[slice[sj - 1]] > key_su) : (sj -= 1) {
                slice[sj] = slice[sj - 1];
            }
            slice[sj] = key;
        }
        for (slice) |d| {
            try stack.append(allocator, .{ .idx = d, .post = false });
        }
    }
    return schedule;
}

fn resolveSigned(
    comptime W: SignedWidth,
    b: *aj.Builder,
    nodes: []const ast.Node,
    slots: []const ?RegSlot,
    remat: []const bool,
    w: i32,
    disp: i32,
    ptrs: []const ?aj.Reg,
    resolver: anytype,
    nidx: ast.NodeIdx,
) aj.Reg {
    if (!remat[nidx]) return slots[nidx].?.reg;
    const d = b.newVec();
    b.vecLoad(W.loadCvtOp(), d, srcLoadMem(W.srcShift(), nodes[nidx], w, disp, ptrs, resolver));
    return d;
}

/// Evaluate one signed `i16` (`2 * f32_lanes` u8 px) or `i32` (`f32_lanes` u9..u16 px) vector
/// at horizontal pixel offset `col_disp`, returning the result register.
fn evalSignedVector(
    comptime W: SignedWidth,
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    width: usize,
    col_disp: i32,
    src_row_ptrs: []const ?aj.Reg,
    row_ptr_resolver: anytype,
    free_vecs: *std.ArrayList(aj.Reg),
) std.mem.Allocator.Error!aj.Reg {
    const n_nodes = prog.nodes.items.len;
    const eval_root = W.planEvalRoot(info);
    const final_shift = W.planFinalShift(info);

    const slots = try allocator.alloc(?RegSlot, n_nodes);
    defer allocator.free(slots);
    @memset(slots, null);

    const rem_uses = try computeRemUses(allocator, prog.nodes.items, eval_root);
    defer allocator.free(rem_uses);

    const fused_bool = try computeFusedBool(allocator, prog.nodes.items, eval_root, rem_uses);
    defer allocator.free(fused_bool);

    const w_i32: i32 = @intCast(width);
    const use_su = b.cpu.has32VecRegs() and info.live_node_count > 48;
    const remat_load = try allocator.alloc(bool, n_nodes);
    defer allocator.free(remat_load);
    @memset(remat_load, false);
    if (use_su and (b.cpu.isa == .neon or W.allowsRematOnAvx512())) {
        var live_loads: usize = 0;
        for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
            if (rem_uses[idx] > 0 and idx != eval_root and (node == .load_clip or node == .load_rel)) {
                live_loads += 1;
            }
        }
        if (live_loads > 12) {
            for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
                if (rem_uses[idx] > 0 and idx != eval_root and (node == .load_clip or node == .load_rel)) {
                    remat_load[idx] = true;
                }
            }
            for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
                // A bare (unfused) select condition is read straight from its slot by
                // emitCmpOrLogicMaskSigned, so a load used as a condition must stay materialized.
                if (rem_uses[idx] > 0 and node == .ternary and node.ternary.op == .select) {
                    remat_load[node.ternary.a] = false;
                }
                if (!fused_bool[idx]) continue;
                switch (node) {
                    .unary => |u| remat_load[u.a] = false,
                    .binary => |bin| {
                        remat_load[bin.a] = false;
                        remat_load[bin.b] = false;
                    },
                    else => {},
                }
            }
        }
    }

    var schedule = try buildIntSchedule(allocator, prog, eval_root, rem_uses, fused_bool, use_su);
    defer schedule.deinit(allocator);

    for (schedule.items) |idx| {
        if (remat_load[idx]) continue;
        const node = prog.nodes.items[idx];
        switch (node) {
            .const_f32 => |v| {
                const iv: W.Int() = @intFromFloat(v);
                slots[idx] = .{ .reg = W.splat(b, iv), .owned = false };
            },
            .const_i32 => |v| {
                const iv: W.Int() = @intCast(v);
                slots[idx] = .{ .reg = W.splat(b, iv), .owned = false };
            },
            .load_clip, .load_rel => {
                const dst = acquireVec(b, free_vecs);
                b.vecLoad(W.loadCvtOp(), dst, srcLoadMem(W.srcShift(), node, w_i32, col_disp, src_row_ptrs, row_ptr_resolver));
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .unary => |u| {
                const a_reg = resolveSigned(W, b, prog.nodes.items, slots, remat_load, w_i32, col_disp, src_row_ptrs, row_ptr_resolver, u.a);
                const dst = acquireVec(b, free_vecs);
                switch (u.op) {
                    .neg => b.vecOp2(W.op2("neg"), dst, a_reg),
                    .abs => b.vecOp2(W.op2("abs"), dst, a_reg),
                    .not => {
                        emitCmpMask(W, b, dst, .le, a_reg, b.vecZero());
                        b.vecOp3(.and_u32, dst, dst, W.splat(b, 1));
                    },
                    else => unreachable,
                }
                try releaseUse(allocator, free_vecs, rem_uses, slots, u.a);
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .binary => |bin| {
                const a_reg = resolveSigned(W, b, prog.nodes.items, slots, remat_load, w_i32, col_disp, src_row_ptrs, row_ptr_resolver, bin.a);
                const b_reg = resolveSigned(W, b, prog.nodes.items, slots, remat_load, w_i32, col_disp, src_row_ptrs, row_ptr_resolver, bin.b);
                const dst = acquireVec(b, free_vecs);
                switch (bin.op) {
                    .add => b.vecOp3(W.op3("add"), dst, a_reg, b_reg),
                    .sub => b.vecOp3(W.op3("sub"), dst, a_reg, b_reg),
                    .mul => emitMulSigned(W, b, dst, bin.a, bin.b, a_reg, b_reg, prog.nodes.items),
                    .min => b.vecOp3(W.op3("min"), dst, a_reg, b_reg),
                    .max => b.vecOp3(W.op3("max"), dst, a_reg, b_reg),
                    .gt, .lt, .eq, .le, .ge => {
                        emitCmpMask(W, b, dst, bin.op, a_reg, b_reg);
                        b.vecOp3(.and_u32, dst, dst, W.splat(b, 1));
                    },
                    .and_op, .or_op, .xor_op => {
                        if (isBool01Node(prog.nodes.items[bin.a]) and isBool01Node(prog.nodes.items[bin.b])) {
                            b.vecOp3(logicOp(bin.op), dst, a_reg, b_reg);
                        } else {
                            const ma = acquireVec(b, free_vecs);
                            const mb = acquireVec(b, free_vecs);
                            b.vecOp3(W.op3("cmp_gt"), ma, a_reg, b.vecZero());
                            b.vecOp3(W.op3("cmp_gt"), mb, b_reg, b.vecZero());
                            b.vecOp3(logicOp(bin.op), dst, ma, mb);
                            b.vecOp3(.and_u32, dst, dst, W.splat(b, 1));
                            try free_vecs.append(allocator, ma);
                            try free_vecs.append(allocator, mb);
                        }
                    },
                    else => unreachable,
                }
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.a);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.b);
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .ternary => |t| {
                var dst: aj.Reg = undefined;
                if (t.op == .select) {
                    const mask = try emitCmpOrLogicMaskSigned(W, allocator, b, prog.nodes.items, slots, rem_uses, fused_bool, free_vecs, t.a);
                    const b_is_zero = exactInt(W, prog.nodes.items[t.b]) == 0;
                    const c_is_zero = exactInt(W, prog.nodes.items[t.c]) == 0;
                    if (c_is_zero) {
                        dst = acquireVec(b, free_vecs);
                        b.vecOp3(.and_u32, dst, resolveSigned(W, b, prog.nodes.items, slots, remat_load, w_i32, col_disp, src_row_ptrs, row_ptr_resolver, t.b), mask);
                        try free_vecs.append(allocator, mask);
                    } else if (b_is_zero) {
                        dst = acquireVec(b, free_vecs);
                        b.vecOp3(.andn_u32, dst, mask, resolveSigned(W, b, prog.nodes.items, slots, remat_load, w_i32, col_disp, src_row_ptrs, row_ptr_resolver, t.c));
                        try free_vecs.append(allocator, mask);
                    } else {
                        const reuse_mask = b.cpu.prefersMaskDstReuse();
                        dst = if (reuse_mask) mask else acquireVec(b, free_vecs);
                        pixel.emitSelectByMask(
                            b,
                            dst,
                            resolveSigned(W, b, prog.nodes.items, slots, remat_load, w_i32, col_disp, src_row_ptrs, row_ptr_resolver, t.c),
                            resolveSigned(W, b, prog.nodes.items, slots, remat_load, w_i32, col_disp, src_row_ptrs, row_ptr_resolver, t.b),
                            mask,
                        );
                        if (!reuse_mask) try free_vecs.append(allocator, mask);
                    }
                    try releaseUse(allocator, free_vecs, rem_uses, slots, t.b);
                    try releaseUse(allocator, free_vecs, rem_uses, slots, t.c);
                } else {
                    dst = acquireVec(b, free_vecs);
                    const a_reg = resolveSigned(W, b, prog.nodes.items, slots, remat_load, w_i32, col_disp, src_row_ptrs, row_ptr_resolver, t.a);
                    const b_reg = resolveSigned(W, b, prog.nodes.items, slots, remat_load, w_i32, col_disp, src_row_ptrs, row_ptr_resolver, t.b);
                    const c_reg = resolveSigned(W, b, prog.nodes.items, slots, remat_load, w_i32, col_disp, src_row_ptrs, row_ptr_resolver, t.c);
                    emitMulSigned(W, b, dst, t.a, t.b, a_reg, b_reg, prog.nodes.items);
                    emitFmaTail(W, b, t.op, dst, c_reg);
                    try releaseUse(allocator, free_vecs, rem_uses, slots, t.a);
                    try releaseUse(allocator, free_vecs, rem_uses, slots, t.b);
                    try releaseUse(allocator, free_vecs, rem_uses, slots, t.c);
                }
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            else => unreachable,
        }
    }

    var res = slots[eval_root].?.reg;
    if (final_shift != 0) {
        // Exact round-half-to-even division by 2^k:
        // `(v + (1 << (k - 1)) - 1 + ((v >> k) & 1)) >> k`
        const out = acquireVec(b, free_vecs);
        const lsb = acquireVec(b, free_vecs);
        b.vecOp2i(W.sraOp(), lsb, res, final_shift);
        b.vecOp3(.and_u32, lsb, lsb, W.splat(b, 1));
        const bias: W.Int() = @intCast((@as(i64, 1) << (final_shift - 1)) - 1);
        if (bias != 0) {
            b.vecOp3(W.op3("add"), out, res, W.splat(b, bias));
            b.vecOp3(W.op3("add"), out, out, lsb);
        } else {
            b.vecOp3(W.op3("add"), out, res, lsb);
        }
        b.vecOp2i(W.sraOp(), out, out, final_shift);
        try free_vecs.append(allocator, lsb);
        if (slots[eval_root].?.owned) try free_vecs.append(allocator, res);
        res = out;
    }
    return res;
}

fn setPair(lanes: [2][]?RegSlot, idx: ast.NodeIdx, regs: [2]aj.Reg, owned: bool) void {
    for (lanes, regs) |sl, r| sl[idx] = .{ .reg = r, .owned = owned };
}

/// `d[i] = op(lanes[i][a], lanes[i][rhs])` for both lanes of a NEON pair, lane 0 first.
fn emitPairOp3(b: *aj.Builder, op: aj.VecOp3, d: [2]aj.Reg, lanes: [2][]?RegSlot, a: ast.NodeIdx, rhs: ast.NodeIdx) void {
    for (d, lanes) |dl, sl| b.vecOp3(op, dl, sl[a].?.reg, sl[rhs].?.reg);
}

/// NEON lockstep pair evaluator (`i16` 16 px / `i32` 8 px) using single 16B `ldr q` loads and
/// widening `uaddl`/`uaddw`/`usubl`/`usubw` to eliminate `ushll` instructions and halve load count.
fn evalSignedPairNeon(
    comptime W: SignedWidth,
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    width: usize,
    src_row_ptrs: []const ?aj.Reg,
    row_ptr_resolver: anytype,
) std.mem.Allocator.Error![2]aj.Reg {
    const n_nodes = prog.nodes.items.len;
    const eval_root = W.planEvalRoot(info);
    const final_shift = W.planFinalShift(info);

    const slots0 = try allocator.alloc(?RegSlot, n_nodes);
    defer allocator.free(slots0);
    @memset(slots0, null);
    const slots1 = try allocator.alloc(?RegSlot, n_nodes);
    defer allocator.free(slots1);
    @memset(slots1, null);
    const raw_slots = try allocator.alloc(?aj.Reg, n_nodes);
    defer allocator.free(raw_slots);
    @memset(raw_slots, null);

    const rem_uses = try computeRemUses(allocator, prog.nodes.items, eval_root);
    defer allocator.free(rem_uses);

    const fused_bool = try computeFusedBool(allocator, prog.nodes.items, eval_root, rem_uses);
    defer allocator.free(fused_bool);

    // Identify `.load_clip`/`.load_rel` nodes that can stay as a single 16B raw vector because
    // every consumer can consume them directly via `uaddl`/`uaddw`/`usubl`/`usubw`.
    const raw_load = try allocator.alloc(bool, n_nodes);
    defer allocator.free(raw_load);
    @memset(raw_load, false);
    for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
        if (rem_uses[idx] == 0 or idx == eval_root) continue;
        if (node == .load_clip or node == .load_rel) raw_load[idx] = true;
    }
    for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
        if (rem_uses[idx] == 0) continue;
        if (fused_bool[idx]) {
            switch (node) {
                .unary => |u| raw_load[u.a] = false,
                .binary => |bin| {
                    raw_load[bin.a] = false;
                    raw_load[bin.b] = false;
                },
                else => {},
            }
            continue;
        }
        switch (node) {
            .unary => |u| raw_load[u.a] = false,
            .binary => |bin| switch (bin.op) {
                .add, .sub => {},
                .mul => {
                    const rhs_is_2 = exactInt(W, prog.nodes.items[bin.b]) == 2 and bin.a != bin.b;
                    const lhs_is_2 = exactInt(W, prog.nodes.items[bin.a]) == 2 and bin.a != bin.b;
                    if (!rhs_is_2) raw_load[bin.a] = false;
                    if (!lhs_is_2) raw_load[bin.b] = false;
                },
                else => {
                    raw_load[bin.a] = false;
                    raw_load[bin.b] = false;
                },
            },
            .ternary => |t| {
                raw_load[t.a] = false;
                raw_load[t.b] = false;
                raw_load[t.c] = false;
            },
            else => {},
        }
    }
    var changed = true;
    while (changed) {
        changed = false;
        for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
            if (rem_uses[idx] == 0 or fused_bool[idx]) continue;
            if (node == .binary and node.binary.op == .sub) {
                const bin = node.binary;
                if (raw_load[bin.a] and !raw_load[bin.b]) {
                    raw_load[bin.a] = false;
                    changed = true;
                }
            }
        }
    }

    var dummy_pool: std.ArrayList(aj.Reg) = .empty;
    defer dummy_pool.deinit(allocator);
    const rem_uses0 = try allocator.dupe(u32, rem_uses);
    defer allocator.free(rem_uses0);
    const rem_uses1 = try allocator.dupe(u32, rem_uses);
    defer allocator.free(rem_uses1);
    const w_i32: i32 = @intCast(width);
    var schedule = try buildIntSchedule(allocator, prog, eval_root, rem_uses, fused_bool, b.cpu.isa == .neon and info.live_node_count > 48);
    defer schedule.deinit(allocator);

    const lanes = [2][]?RegSlot{ slots0, slots1 };
    for (schedule.items) |idx| {
        const node = prog.nodes.items[idx];
        switch (node) {
            .const_f32 => |v| {
                const reg = W.splat(b, @intFromFloat(v));
                setPair(lanes, idx, .{ reg, reg }, false);
            },
            .const_i32 => |v| {
                const reg = W.splat(b, @intCast(v));
                setPair(lanes, idx, .{ reg, reg }, false);
            },
            .load_clip, .load_rel => {
                const mem = srcLoadMem(W.srcShift(), node, w_i32, 0, src_row_ptrs, row_ptr_resolver);
                if (raw_load[idx]) {
                    const r = b.newVec();
                    b.vecLoad(.u_vec, r, mem);
                    raw_slots[idx] = r;
                } else {
                    const d = [2]aj.Reg{ b.newVec(), b.newVec() };
                    W.neonLoadWidenX2(b, d[0], d[1], mem);
                    setPair(lanes, idx, d, true);
                }
            },
            .unary => |u| {
                const d = [2]aj.Reg{ b.newVec(), b.newVec() };
                switch (u.op) {
                    .neg => for (d, lanes) |dl, sl| b.vecOp2(W.op2("neg"), dl, sl[u.a].?.reg),
                    .abs => for (d, lanes) |dl, sl| b.vecOp2(W.op2("abs"), dl, sl[u.a].?.reg),
                    .not => {
                        const c1 = W.splat(b, 1);
                        for (d, lanes) |dl, sl| {
                            emitCmpMask(W, b, dl, .le, sl[u.a].?.reg, b.vecZero());
                            b.vecOp3(.and_u32, dl, dl, c1);
                        }
                    },
                    else => unreachable,
                }
                setPair(lanes, idx, d, true);
            },
            .binary => |bin| {
                const d = [2]aj.Reg{ b.newVec(), b.newVec() };
                switch (bin.op) {
                    .add => {
                        if (raw_load[bin.a] and raw_load[bin.b]) {
                            W.neonUaddl(b, d[0], d[1], raw_slots[bin.a].?, raw_slots[bin.b].?);
                        } else if (raw_load[bin.b]) {
                            W.neonUaddw(b, d[0], d[1], slots0[bin.a].?.reg, slots1[bin.a].?.reg, raw_slots[bin.b].?);
                        } else if (raw_load[bin.a]) {
                            W.neonUaddw(b, d[0], d[1], slots0[bin.b].?.reg, slots1[bin.b].?.reg, raw_slots[bin.a].?);
                        } else {
                            emitPairOp3(b, W.op3("add"), d, lanes, bin.a, bin.b);
                        }
                    },
                    .sub => {
                        if (raw_load[bin.a] and raw_load[bin.b]) {
                            W.neonUsubl(b, d[0], d[1], raw_slots[bin.a].?, raw_slots[bin.b].?);
                        } else if (raw_load[bin.b]) {
                            W.neonUsubw(b, d[0], d[1], slots0[bin.a].?.reg, slots1[bin.a].?.reg, raw_slots[bin.b].?);
                        } else {
                            emitPairOp3(b, W.op3("sub"), d, lanes, bin.a, bin.b);
                        }
                    },
                    .mul => {
                        if (raw_load[bin.a]) {
                            W.neonUaddl(b, d[0], d[1], raw_slots[bin.a].?, raw_slots[bin.a].?);
                        } else if (raw_load[bin.b]) {
                            W.neonUaddl(b, d[0], d[1], raw_slots[bin.b].?, raw_slots[bin.b].?);
                        } else for (d, lanes) |dl, sl| {
                            emitMulSigned(W, b, dl, bin.a, bin.b, sl[bin.a].?.reg, sl[bin.b].?.reg, prog.nodes.items);
                        }
                    },
                    .min => emitPairOp3(b, W.op3("min"), d, lanes, bin.a, bin.b),
                    .max => emitPairOp3(b, W.op3("max"), d, lanes, bin.a, bin.b),
                    .gt, .lt, .eq, .le, .ge => {
                        const c1 = W.splat(b, 1);
                        for (d, lanes) |dl, sl| {
                            emitCmpMask(W, b, dl, bin.op, sl[bin.a].?.reg, sl[bin.b].?.reg);
                            b.vecOp3(.and_u32, dl, dl, c1);
                        }
                    },
                    .and_op, .or_op, .xor_op => {
                        const vop = logicOp(bin.op);
                        if (isBool01Node(prog.nodes.items[bin.a]) and isBool01Node(prog.nodes.items[bin.b])) {
                            emitPairOp3(b, vop, d, lanes, bin.a, bin.b);
                        } else {
                            const c1 = W.splat(b, 1);
                            for (d, lanes) |dl, sl| {
                                const ma = b.newVec();
                                const mb = b.newVec();
                                b.vecOp3(W.op3("cmp_gt"), ma, sl[bin.a].?.reg, b.vecZero());
                                b.vecOp3(W.op3("cmp_gt"), mb, sl[bin.b].?.reg, b.vecZero());
                                b.vecOp3(vop, dl, ma, mb);
                                b.vecOp3(.and_u32, dl, dl, c1);
                            }
                        }
                    },
                    else => unreachable,
                }
                setPair(lanes, idx, d, true);
            },
            .ternary => |t| {
                if (t.op == .select) {
                    dummy_pool.clearRetainingCapacity();
                    const m0 = try emitCmpOrLogicMaskSigned(W, allocator, b, prog.nodes.items, slots0, rem_uses0, fused_bool, &dummy_pool, t.a);
                    dummy_pool.clearRetainingCapacity();
                    const m1 = try emitCmpOrLogicMaskSigned(W, allocator, b, prog.nodes.items, slots1, rem_uses1, fused_bool, &dummy_pool, t.a);
                    const m = [2]aj.Reg{ m0, m1 };
                    var d = m;
                    if (exactInt(W, prog.nodes.items[t.c]) == 0) {
                        d = .{ b.newVec(), b.newVec() };
                        for (d, lanes, m) |dl, sl, ml| b.vecOp3(.and_u32, dl, sl[t.b].?.reg, ml);
                    } else if (exactInt(W, prog.nodes.items[t.b]) == 0) {
                        d = .{ b.newVec(), b.newVec() };
                        for (d, lanes, m) |dl, sl, ml| b.vecOp3(.andn_u32, dl, ml, sl[t.c].?.reg);
                    } else {
                        for (lanes, m) |sl, ml| pixel.emitSelectByMask(b, ml, sl[t.c].?.reg, sl[t.b].?.reg, ml);
                    }
                    setPair(lanes, idx, d, true);
                } else {
                    const d = [2]aj.Reg{ b.newVec(), b.newVec() };
                    for (d, lanes) |dl, sl| emitMulSigned(W, b, dl, t.a, t.b, sl[t.a].?.reg, sl[t.b].?.reg, prog.nodes.items);
                    for (d, lanes) |dl, sl| emitFmaTail(W, b, t.op, dl, sl[t.c].?.reg);
                    setPair(lanes, idx, d, true);
                }
            },
            else => unreachable,
        }
    }

    var res0 = slots0[eval_root].?.reg;
    var res1 = slots1[eval_root].?.reg;
    if (final_shift != 0) {
        const bias: W.Int() = @intCast((@as(i64, 1) << (final_shift - 1)) - 1);
        const c1 = W.splat(b, 1);
        const lsb0 = b.newVec();
        const lsb1 = b.newVec();
        const out0 = b.newVec();
        const out1 = b.newVec();
        b.vecOp2i(W.sraOp(), lsb0, res0, final_shift);
        b.vecOp2i(W.sraOp(), lsb1, res1, final_shift);
        b.vecOp3(.and_u32, lsb0, lsb0, c1);
        b.vecOp3(.and_u32, lsb1, lsb1, c1);
        if (bias != 0) {
            const cb = W.splat(b, bias);
            b.vecOp3(W.op3("add"), out0, res0, cb);
            b.vecOp3(W.op3("add"), out1, res1, cb);
            b.vecOp3(W.op3("add"), out0, out0, lsb0);
            b.vecOp3(W.op3("add"), out1, out1, lsb1);
        } else {
            b.vecOp3(W.op3("add"), out0, res0, lsb0);
            b.vecOp3(W.op3("add"), out1, res1, lsb1);
        }
        b.vecOp2i(W.sraOp(), out0, out0, final_shift);
        b.vecOp2i(W.sraOp(), out1, out1, final_shift);
        res0 = out0;
        res1 = out1;
    }
    return .{ res0, res1 };
}

fn emitScaledStoreU8FromI16(
    b: *aj.Builder,
    dst_ptr: aj.Reg,
    base_off: i32,
    res: aj.Reg,
    scale_op: analysis_mod.FinalScaleOp,
    c_reg: aj.Reg,
    row_ptr_resolver: anytype,
) void {
    const lo = b.newVec();
    const hi = b.newVec();
    b.vecOp2(.cvt_i16_lo_to_i32, lo, res);
    b.vecOp2(.cvt_i16_hi_to_i32, hi, res);
    b.vecOp2(.cvt_i32_to_f32, lo, lo);
    b.vecOp2(.cvt_i32_to_f32, hi, hi);
    switch (scale_op) {
        .mul => {
            b.vecOp3(.mul_f32, lo, lo, c_reg);
            b.vecOp3(.mul_f32, hi, hi, c_reg);
        },
        .div => {
            b.vecOp3(.div_f32, lo, lo, c_reg);
            b.vecOp3(.div_f32, hi, hi, c_reg);
        },
        .none => unreachable,
    }
    const lanes_i32: i32 = @intCast(b.cpu.f32_lanes);
    pixel.emitStoreF32x2(
        b,
        row_ptr_resolver.rowMem(dst_ptr, 0, base_off),
        row_ptr_resolver.rowMem(dst_ptr, 0, base_off + lanes_i32),
        lo,
        hi,
        .{ .is_float = false, .bits_per_sample = 8 },
        false,
    );
}

/// Signed `i16` (u8 output, `2 * f32_lanes` px/vector) or `i32` (u9..u16 output, `f32_lanes` px/vector)
/// interior body. `unroll2` evaluates a second vector (NEON: as one interleaved pair) and packs both
/// results together via `vecStoreU8FromI16x2` / `emitStoreI32x2` (or scaled f32 pairs when
/// `final_scale_op != .none`).
fn emitSignedBody(
    comptime W: SignedWidth,
    unroll2: bool,
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    width: usize,
    dst_fmt: ast.SampleFormat,
    renv: anytype,
) std.mem.Allocator.Error!void {
    const vec_px: i32 = @intCast(if (W == .i16) b.cpu.f32_lanes * 2 else b.cpu.f32_lanes);
    const n: usize = if (unroll2) 2 else 1;
    var res: [2]aj.Reg = undefined;
    if (unroll2 and b.cpu.isa == .neon) {
        res = try evalSignedPairNeon(W, allocator, b, prog, info, width, renv.src_row_ptrs, renv);
    } else {
        var free_vecs: std.ArrayList(aj.Reg) = .empty;
        defer free_vecs.deinit(allocator);
        for (res[0..n], 0..) |*r, i| {
            const col_disp: i32 = @intCast(i);
            r.* = try evalSignedVector(W, allocator, b, prog, info, width, col_disp * vec_px, renv.src_row_ptrs, renv, &free_vecs);
        }
    }
    const dst_ptr = renv.dst_ptr;
    switch (W) {
        .i16 => if (info.int16_plan.final_scale_op == .none) {
            const dst_mem = renv.rowMem(dst_ptr, 0, 0);
            if (unroll2) b.vecStoreU8FromI16x2(dst_mem, res[0], res[1]) else b.vecStoreU8FromI16(dst_mem, res[0]);
        } else {
            const c_reg = b.constF32Splat(info.int16_plan.final_scale_c);
            for (res[0..n], 0..) |r, i| {
                const base_off: i32 = @intCast(i);
                emitScaledStoreU8FromI16(b, dst_ptr, base_off * vec_px, r, info.int16_plan.final_scale_op, c_reg, renv);
            }
        },
        .i32 => {
            const dst_mem0 = renv.rowMem(dst_ptr, 1, 0);
            const dst_mem1 = renv.rowMem(dst_ptr, 1, vec_px * 2);
            if (info.int32_plan.final_scale_op == .none) {
                if (unroll2) {
                    pixel.emitStoreI32x2(b, dst_mem0, dst_mem1, res[0], res[1], dst_fmt, renv.use_running_ptrs);
                } else {
                    pixel.emitStoreI32(b, dst_mem0, res[0], dst_fmt);
                }
            } else {
                var f: [2]aj.Reg = undefined;
                for (f[0..n]) |*v| v.* = b.newVec();
                for (f[0..n], res[0..n]) |v, r| b.vecOp2(.cvt_i32_to_f32, v, r);
                const c_reg = b.constF32Splat(info.int32_plan.final_scale_c);
                const op: aj.VecOp3 = switch (info.int32_plan.final_scale_op) {
                    .mul => .mul_f32,
                    .div => .div_f32,
                    .none => unreachable,
                };
                for (f[0..n]) |v| b.vecOp3(op, v, v, c_reg);
                if (unroll2) {
                    pixel.emitStoreF32x2(b, dst_mem0, dst_mem1, f[0], f[1], dst_fmt, renv.use_running_ptrs);
                } else {
                    pixel.emitStoreF32(b, dst_mem0, f[0], dst_fmt);
                }
            }
        },
    }
}

test "u8 and i16 fast paths match f32 oracle across u8 inputs including Y1 and Y2 edge cases" {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");
    const cpu_mod = @import("cpu.zig");
    const codegen = @import("codegen.zig");

    const allocator = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    const profile = interp.TargetProfile.fromCpu(cpu);
    const fmt8 = ast.SampleFormat{ .is_float = false, .bits_per_sample = 8 };
    const fmts = [_]ast.SampleFormat{ fmt8, fmt8 };

    var rt = try aj.Runtime.init();
    defer rt.deinit();

    const cases = [_]struct { expr: []const u8, width: usize, expect_u8: bool, expect_i16: bool }{
        .{ .expr = "x 128 * 127 + 2 /", .width = 32, .expect_u8 = false, .expect_i16 = false },
        .{ .expr = "x[-1,0] x + x[1,0] + x[0,-1] + 4 /", .width = 32, .expect_u8 = false, .expect_i16 = true },
        .{ .expr = "x[-1,0] x + x[1,0] + x[0,-1] + 4 /", .width = 64, .expect_u8 = false, .expect_i16 = true },
        .{ .expr = "x[-1,0] x + x[1,0] + x[0,-1] + 4 /", .width = 129, .expect_u8 = false, .expect_i16 = true },
        .{ .expr = "x y - abs", .width = 65, .expect_u8 = true, .expect_i16 = true },
        .{ .expr = "255 x -", .width = 64, .expect_u8 = true, .expect_i16 = true },
        .{ .expr = "x[-1,-1] x[0,-1] x[1,-1] x[-1,0] x x[1,0] x[-1,1] x[0,1] x[1,1] sort9 drop4 swap4 drop4", .width = 65, .expect_u8 = true, .expect_i16 = true },
    };

    for (cases) |tc| {
        var prog = try parser.parse(allocator, tc.expr, .{ .num_clips = 2 });
        defer prog.deinit(allocator);
        try lower_mod.lower(allocator, &prog, profile);
        try optimize_mod.optimize(allocator, &prog, profile, .{});

        var info = try analysis_mod.analyze(allocator, &prog, &fmts, fmt8, @intCast(tc.width), 3, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
        defer info.deinit(allocator);
        try std.testing.expectEqual(tc.expect_u8, info.u8_plan.eligible);
        try std.testing.expectEqual(tc.expect_i16, info.int16_plan.eligible);

        var compiled = try codegen.compilePlane(allocator, &rt, cpu, &prog, &info, .{
            .width = tc.width,
            .height = 3,
            .src_fmts = &fmts,
            .dst_fmt = fmt8,
        });
        defer compiled.deinit(allocator, &rt);

        const stride = std.mem.alignForward(usize, tc.width, 64);
        const s0 = try allocator.alignedAlloc(u8, .@"64", stride * 3);
        defer allocator.free(s0);
        const s1 = try allocator.alignedAlloc(u8, .@"64", stride * 3);
        defer allocator.free(s1);
        const d_jit = try allocator.alignedAlloc(u8, .@"64", stride * 3);
        defer allocator.free(d_jit);
        const d_ref = try allocator.alignedAlloc(u8, .@"64", stride * 3);
        defer allocator.free(d_ref);

        for (0..3) |y| {
            for (0..tc.width) |x| {
                s0[y * stride + x] = if (x == 0 and y == 0) 255 else @intCast((y * 73 + x * 19 + 5) & 255);
                s1[y * stride + x] = @intCast((y * 41 + x * 29 + 11) & 255);
            }
        }
        const srcs = [_][*]const u8{ s0.ptr, s1.ptr };
        const strides = [_]isize{ @intCast(stride), @intCast(stride) };

        compiled.kernel.func(d_jit.ptr, @intCast(stride), &srcs, &strides, 0, null);
        try interp.evalPlane(allocator, &prog, profile, .{
            .dst = d_ref.ptr,
            .dst_stride = @intCast(stride),
            .dst_fmt = fmt8,
            .srcs = &srcs,
            .src_strides = &strides,
            .src_fmts = &fmts,
            .width = tc.width,
            .height = 3,
            .frame_n = 0,
            .props = &.{},
        });

        for (0..3) |y| {
            for (0..tc.width) |x| {
                try std.testing.expectEqual(d_ref[y * stride + x], d_jit[y * stride + x]);
            }
        }
    }
}

test "native u16 fast path matches f32 oracle across all host ISA tiers, u16, and unvalidated u10 inputs" {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");
    const cpu_mod = @import("cpu.zig");
    const codegen = @import("codegen.zig");

    const allocator = std.testing.allocator;
    var cpus: [4]cpu_mod.CpuInfo = undefined;
    var n_cpus: usize = 0;
    switch (cpu_mod.IsaFamily.host()) {
        .x86 => {
            const host = cpu_mod.detectWithOverride(null);
            cpus[n_cpus] = cpu_mod.detectWithOverride(1);
            n_cpus += 1;
            if (@backingInt(host.isa) >= @backingInt(cpu_mod.Isa.v2)) {
                cpus[n_cpus] = cpu_mod.detectWithOverride(2);
                n_cpus += 1;
            }
            if (@backingInt(host.isa) >= @backingInt(cpu_mod.Isa.v3)) {
                cpus[n_cpus] = cpu_mod.detectWithOverride(3);
                n_cpus += 1;
            }
            if (@backingInt(host.isa) >= @backingInt(cpu_mod.Isa.v4)) {
                cpus[n_cpus] = cpu_mod.detectWithOverride(4);
                n_cpus += 1;
            }
        },
        .aarch64 => {
            cpus[n_cpus] = cpu_mod.detectWithOverride(null);
            n_cpus += 1;
        },
    }

    var rt = try aj.Runtime.init();
    defer rt.deinit();

    const fmts_to_test = [_]ast.SampleFormat{
        .{ .is_float = false, .bits_per_sample = 16 },
        .{ .is_float = false, .bits_per_sample = 10 },
    };

    const cases = [_]struct { expr: []const u8, width: usize, expect_u16: bool, opt_int: bool = false }{
        .{ .expr = "x y - abs", .width = 1, .expect_u16 = true },
        .{ .expr = "x y - abs", .width = 17, .expect_u16 = true },
        .{ .expr = "x y - abs", .width = 65, .expect_u16 = true },
        .{ .expr = "65535 x -", .width = 33, .expect_u16 = true },
        .{ .expr = "x[-1,-1] x[0,-1] x[1,-1] x[-1,0] x x[1,0] x[-1,1] x[0,1] x[1,1] sort9 drop4 swap4 drop4", .width = 65, .expect_u16 = true },
        .{ .expr = "x[-65,0] x[65,0] max x min", .width = 193, .expect_u16 = true },
        .{ .expr = "x y +", .width = 32, .expect_u16 = true },
        .{ .expr = "x y -", .width = 33, .expect_u16 = true },
        .{ .expr = "x y + 1023 min", .width = 33, .expect_u16 = true },
        .{ .expr = "x y - 0 max", .width = 33, .expect_u16 = true },
        .{ .expr = "x y bitxor", .width = 33, .expect_u16 = true, .opt_int = true },
        .{ .expr = "x y bitand 1023 bitor", .width = 33, .expect_u16 = true, .opt_int = true },
        .{ .expr = "x y bitxor", .width = 33, .expect_u16 = true, .opt_int = false },
        .{ .expr = "x y + 2 /", .width = 32, .expect_u16 = false },
        .{ .expr = "x y 4 - x 4 + clamp", .width = 33, .expect_u16 = true },
        .{ .expr = "x[-1,-1] x[0,-1] + x[1,-1] + x[-1,0] + x + x[1,0] + x[-1,1] + x[0,1] + x[1,1] + 9 /", .width = 35, .expect_u16 = false },
        .{ .expr = "x 4 * x[0,-1] x[-1,0] x[1,0] x[0,1] + + + 2 * + x[-1,-1] x[1,-1] x[-1,1] x[1,1] + + + + 16 /", .width = 35, .expect_u16 = false },
        .{ .expr = "x y = x 100 > and 65535 *", .width = 33, .expect_u16 = false },
        // Bare-load select condition in a > 48-node kernel (32-reg i32 load remat must keep the condition load).
        .{ .expr = "x[-2,-2] 7 x[-1,-2] x[0,-2] + x[1,-2] + x[2,-2] + x[-2,-1] + x[-1,-1] + x[0,-1] + x[1,-1] + x[2,-1] + x[-2,0] + x[-1,0] + x[0,0] + x[1,0] + x[2,0] + x[-2,1] + x[-1,1] + x[0,1] + x[1,1] + x[2,1] + x[-2,2] + x[-1,2] + x[0,2] + x[1,2] + x[2,2] + ?", .width = 300, .expect_u16 = false },
    };

    for (cpus[0..n_cpus]) |cpu| {
        const profile = interp.TargetProfile.fromCpu(cpu);

        for (fmts_to_test) |fmt| {
            const fmts = [_]ast.SampleFormat{ fmt, fmt };
            for (cases) |tc| {
                var prog = try parser.parse(allocator, tc.expr, .{ .num_clips = 2, .opt_int = tc.opt_int });
                defer prog.deinit(allocator);
                try lower_mod.lower(allocator, &prog, profile);
                try optimize_mod.optimize(allocator, &prog, profile, .{});

                var info = try analysis_mod.analyze(allocator, &prog, &fmts, fmt, @intCast(tc.width), 3, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
                defer info.deinit(allocator);
                try std.testing.expectEqual(tc.expect_u16, info.u16_plan.eligible);

                var compiled = try codegen.compilePlane(allocator, &rt, cpu, &prog, &info, .{
                    .width = tc.width,
                    .height = 3,
                    .src_fmts = &fmts,
                    .dst_fmt = fmt,
                });
                defer compiled.deinit(allocator, &rt);

                const stride = std.mem.alignForward(usize, tc.width * 2, 64);
                const s0 = try allocator.alignedAlloc(u8, .@"64", stride * 3);
                defer allocator.free(s0);
                const s1 = try allocator.alignedAlloc(u8, .@"64", stride * 3);
                defer allocator.free(s1);
                const d_jit = try allocator.alignedAlloc(u8, .@"64", stride * 3);
                defer allocator.free(d_jit);
                const d_ref = try allocator.alignedAlloc(u8, .@"64", stride * 3);
                defer allocator.free(d_ref);
                @memset(s0, 0);
                @memset(s1, 0);

                for (0..3) |y| {
                    const r0: [*]u16 = @ptrCast(@alignCast(s0.ptr + y * stride));
                    const r1: [*]u16 = @ptrCast(@alignCast(s1.ptr + y * stride));
                    for (0..tc.width) |x| {
                        // Include full 0..65535 range (even on u10) to prove zero u16 wrap & exact peak clamp!
                        r0[x] = if (x == 0 and y == 0) 65535 else if (x == 1 and y == 0) 0 else @intCast((y * 19381 + x * 4099 + 17) & 65535);
                        r1[x] = if (x == 0 and y == 0) 0 else @intCast((y * 7919 + x * 9973 + 31) & 65535);
                    }
                }
                const srcs = [_][*]const u8{ s0.ptr, s1.ptr };
                const strides = [_]isize{ @intCast(stride), @intCast(stride) };

                compiled.kernel.func(d_jit.ptr, @intCast(stride), &srcs, &strides, 0, null);
                try interp.evalPlane(allocator, &prog, profile, .{
                    .dst = d_ref.ptr,
                    .dst_stride = @intCast(stride),
                    .dst_fmt = fmt,
                    .srcs = &srcs,
                    .src_strides = &strides,
                    .src_fmts = &fmts,
                    .width = tc.width,
                    .height = 3,
                    .frame_n = 0,
                    .props = &.{},
                });

                for (0..3) |y| {
                    const ref_row: [*]const u16 = @ptrCast(@alignCast(d_ref.ptr + y * stride));
                    const jit_row: [*]const u16 = @ptrCast(@alignCast(d_jit.ptr + y * stride));
                    for (0..tc.width) |x| {
                        try std.testing.expectEqual(ref_row[x], jit_row[x]);
                    }
                }
            }
        }
    }
}
