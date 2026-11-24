//! Provably-in-range `u8` (16/32/64 px/iter) and `i16` (8/16/32 px/iter, 2x unrolled
//! to 16/32/64 px/iter with paired store packing) SIMD interior fast paths for `u8` clips (Y1, Y2).
//!
//! Optional final `/ 2^k` in `i16` uses exact round-half-even
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

fn exactIntI16(node: ast.Node) ?i16 {
    return switch (node) {
        .const_f32 => |v| blk: {
            if (!std.math.isFinite(v) or @trunc(v) != v) break :blk null;
            if (v < @as(f32, std.math.minInt(i16)) or v > @as(f32, std.math.maxInt(i16))) break :blk null;
            break :blk @intFromFloat(v);
        },
        .const_i32 => |v| if (v >= std.math.minInt(i16) and v <= std.math.maxInt(i16)) @intCast(v) else null,
        else => null,
    };
}

fn pow2ShiftI16(val: i16) ?u4 {
    if (val <= 0) return null;
    const u: u16 = @intCast(val);
    if (!std.math.isPowerOfTwo(u)) return null;
    return @intCast(@ctz(u));
}

/// Emit native `u8` SIMD evaluation (`vec_bytes` = 16/32/64 pixels per iteration, zero unpack/pack).
pub fn emitU8InteriorBody(
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    width: usize,
    dst_ptr: aj.Reg,
    x_gp: aj.Reg,
    src_row_ptrs: []const ?aj.Reg,
    row_ptr_resolver: anytype,
) std.mem.Allocator.Error!void {
    _ = x_gp;
    const eval_root = prog.root;
    const n_nodes = prog.nodes.items.len;
    const slots = try allocator.alloc(?RegSlot, n_nodes);
    defer allocator.free(slots);
    @memset(slots, null);

    // Detect `abs(a - b)` patterns so `sub` inside `abs` is emitted as `max_u8(a, b) - min_u8(a, b)`.
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

    var free_vecs: std.ArrayList(aj.Reg) = .empty;
    defer free_vecs.deinit(allocator);

    const acquireVec = struct {
        fn run(builder: *aj.Builder, pool: *std.ArrayList(aj.Reg)) aj.Reg {
            if (pool.items.len > 0) return pool.pop().?;
            return builder.newVec();
        }
    }.run;

    const releaseUse = struct {
        fn run(alloc: std.mem.Allocator, pool: *std.ArrayList(aj.Reg), rem: []u32, slot_arr: []?RegSlot, idx: ast.NodeIdx) std.mem.Allocator.Error!void {
            if (rem[idx] == 0) return;
            rem[idx] -= 1;
            if (rem[idx] == 0) {
                if (slot_arr[idx]) |s| {
                    if (s.owned) {
                        try pool.append(alloc, s.reg);
                        slot_arr[idx] = null;
                    }
                }
            }
        }
    }.run;

    const w_i32: i32 = @intCast(width);

    for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
        if (rem_uses[idx] == 0 or skip_sub[idx]) continue;
        switch (node) {
            .const_f32 => |v| {
                const uv: u8 = @intFromFloat(v);
                slots[idx] = .{ .reg = constU8Splat(b, uv), .owned = false };
            },
            .const_i32 => |v| {
                const uv: u8 = @intCast(v);
                slots[idx] = .{ .reg = constU8Splat(b, uv), .owned = false };
            },
            .load_clip => |clip| {
                const dst = acquireVec(b, &free_vecs);
                b.vecLoad(.u_vec, dst, row_ptr_resolver.rowMem(src_row_ptrs[clip].?, 0, 0));
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .load_rel => |rel| {
                const dst = acquireVec(b, &free_vecs);
                const row_ptr = row_ptr_resolver.rowPtrFor(rel.clip, rel.dy, rel.mode);
                const eff_dx: i32 = interp.clampRelOffset(rel.dx, w_i32);
                b.vecLoad(.u_vec, dst, row_ptr_resolver.rowMem(row_ptr, 0, eff_dx));
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .unary => |u| {
                // Only `abs(a - b)` reaches here: `max_u8(a, b) - min_u8(a, b)`.
                const sb = prog.nodes.items[u.a].binary;
                const a_reg = slots[sb.a].?.reg;
                const b_reg = slots[sb.b].?.reg;
                const mx = acquireVec(b, &free_vecs);
                const mn = acquireVec(b, &free_vecs);
                b.vecOp3(.max_u8, mx, a_reg, b_reg);
                b.vecOp3(.min_u8, mn, a_reg, b_reg);
                b.vecOp3(.sub_u8, mx, mx, mn);
                try free_vecs.append(allocator, mn);
                try releaseUse(allocator, &free_vecs, rem_uses, slots, sb.a);
                try releaseUse(allocator, &free_vecs, rem_uses, slots, sb.b);
                slots[idx] = .{ .reg = mx, .owned = true };
            },
            .binary => |bin| {
                const a_reg = slots[bin.a].?.reg;
                const b_reg = slots[bin.b].?.reg;
                const dst = acquireVec(b, &free_vecs);
                switch (bin.op) {
                    .min => b.vecOp3(.min_u8, dst, a_reg, b_reg),
                    .max => b.vecOp3(.max_u8, dst, a_reg, b_reg),
                    .add => b.vecOp3(.adds_u8, dst, a_reg, b_reg),
                    .sub => b.vecOp3(.subs_u8, dst, a_reg, b_reg),
                    .bitand => b.vecOp3(.and_u32, dst, a_reg, b_reg),
                    .bitor => b.vecOp3(.or_u32, dst, a_reg, b_reg),
                    .bitxor => b.vecOp3(.xor_u32, dst, a_reg, b_reg),
                    else => unreachable,
                }
                try releaseUse(allocator, &free_vecs, rem_uses, slots, bin.a);
                try releaseUse(allocator, &free_vecs, rem_uses, slots, bin.b);
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            else => unreachable,
        }
    }

    const res = slots[eval_root].?.reg;
    b.vecStore(.u_vec, row_ptr_resolver.rowMem(dst_ptr, 0, 0), res);
}

/// Emit native `u16` SIMD evaluation (`vec_bytes / 2` = 8/16/32 pixels per iteration, zero unpack/pack).
pub fn emitU16InteriorBody(
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    width: usize,
    dst_fmt: ast.SampleFormat,
    dst_ptr: aj.Reg,
    x_gp: aj.Reg,
    src_row_ptrs: []const ?aj.Reg,
    row_ptr_resolver: anytype,
) std.mem.Allocator.Error!void {
    _ = x_gp;
    const eval_root = prog.root;
    const n_nodes = prog.nodes.items.len;
    const slots = try allocator.alloc(?RegSlot, n_nodes);
    defer allocator.free(slots);
    @memset(slots, null);

    // Detect `abs(a - b)` patterns so `sub` inside `abs` is emitted as `max_u16(a, b) - min_u16(a, b)`.
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

    const w_i32: i32 = @intCast(width);

    for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
        if (rem_uses[idx] == 0 or skip_sub[idx]) continue;
        switch (node) {
            .const_f32 => |v| {
                const uv: u16 = @intFromFloat(v);
                slots[idx] = .{ .reg = constU16Splat(b, uv), .owned = false };
            },
            .const_i32 => |v| {
                const uv: u16 = @intCast(v);
                slots[idx] = .{ .reg = constU16Splat(b, uv), .owned = false };
            },
            .load_clip => |clip| {
                const dst = b.newVec();
                b.vecLoad(.u_vec, dst, row_ptr_resolver.rowMem(src_row_ptrs[clip].?, 1, 0));
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .load_rel => |rel| {
                const dst = b.newVec();
                const row_ptr = row_ptr_resolver.rowPtrFor(rel.clip, rel.dy, rel.mode);
                const eff_dx: i32 = interp.clampRelOffset(rel.dx, w_i32);
                b.vecLoad(.u_vec, dst, row_ptr_resolver.rowMem(row_ptr, 1, eff_dx * 2));
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .unary => |u| {
                // Only `abs(a - b)` reaches here: `max_u16(a, b) - min_u16(a, b)`.
                const sb = prog.nodes.items[u.a].binary;
                const a_reg = slots[sb.a].?.reg;
                const b_reg = slots[sb.b].?.reg;
                const mx = b.newVec();
                const mn = b.newVec();
                b.vecOp3(.max_u16, mx, a_reg, b_reg);
                b.vecOp3(.min_u16, mn, a_reg, b_reg);
                b.vecOp3(.sub_i16, mx, mx, mn);
                slots[idx] = .{ .reg = mx, .owned = true };
            },
            .binary => |bin| {
                const a_reg = slots[bin.a].?.reg;
                const b_reg = slots[bin.b].?.reg;
                const dst = b.newVec();
                switch (bin.op) {
                    .min => b.vecOp3(.min_u16, dst, a_reg, b_reg),
                    .max => b.vecOp3(.max_u16, dst, a_reg, b_reg),
                    .add => b.vecOp3(.adds_u16, dst, a_reg, b_reg),
                    .sub => b.vecOp3(.subs_u16, dst, a_reg, b_reg),
                    .bitand => b.vecOp3(.and_u32, dst, a_reg, b_reg),
                    .bitor => b.vecOp3(.or_u32, dst, a_reg, b_reg),
                    .bitxor => b.vecOp3(.xor_u32, dst, a_reg, b_reg),
                    else => unreachable,
                }
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            else => unreachable,
        }
    }

    var res = slots[eval_root].?.reg;
    if (dst_fmt.bits_per_sample < 16) {
        const clamped = b.newVec();
        b.vecOp3(.min_u16, clamped, res, constU16Splat(b, @intCast(dst_fmt.peakInt())));
        res = clamped;
    }
    b.vecStore(.u_vec, row_ptr_resolver.rowMem(dst_ptr, 1, 0), res);
}

fn isFusableBoolI16(nodes: []const ast.Node, rem_uses: []const u32, idx: ast.NodeIdx) bool {
    if (rem_uses[idx] != 1) return false;
    return switch (nodes[idx]) {
        .unary => |u| u.op == .not,
        .binary => |bin| switch (bin.op) {
            .lt, .le, .gt, .ge, .eq, .and_op, .or_op, .xor_op => true,
            else => false,
        },
        else => false,
    };
}

fn markFusableBoolTreeI16(nodes: []const ast.Node, rem_uses: []const u32, fused: []bool, idx: ast.NodeIdx) void {
    if (!isFusableBoolI16(nodes, rem_uses, idx)) return;
    fused[idx] = true;
    switch (nodes[idx]) {
        .binary => |bin| switch (bin.op) {
            .and_op, .or_op, .xor_op => {
                markFusableBoolTreeI16(nodes, rem_uses, fused, bin.a);
                markFusableBoolTreeI16(nodes, rem_uses, fused, bin.b);
            },
            else => {},
        },
        else => {},
    }
}

fn emitCmpOrLogicMaskI16(
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    nodes: []const ast.Node,
    slots: []?RegSlot,
    rem_uses: []u32,
    fused: []const bool,
    free_vecs: *std.ArrayList(aj.Reg),
    idx: ast.NodeIdx,
) std.mem.Allocator.Error!aj.Reg {
    const acquireVec = struct {
        fn run(builder: *aj.Builder, pool: *std.ArrayList(aj.Reg)) aj.Reg {
            if (pool.items.len > 0) return pool.pop().?;
            return builder.newVec();
        }
    }.run;
    const releaseUse = struct {
        fn run(alloc: std.mem.Allocator, pool: *std.ArrayList(aj.Reg), rem: []u32, slot_arr: []?RegSlot, node_idx: ast.NodeIdx) std.mem.Allocator.Error!void {
            if (rem[node_idx] == 0) return;
            rem[node_idx] -= 1;
            if (rem[node_idx] == 0) {
                if (slot_arr[node_idx]) |s| {
                    if (s.owned) {
                        try pool.append(alloc, s.reg);
                        slot_arr[node_idx] = null;
                    }
                }
            }
        }
    }.run;

    const mask = acquireVec(b, free_vecs);
    if (!fused[idx]) {
        b.vecOp3(.cmp_gt_i16, mask, slots[idx].?.reg, b.vecZero());
        try releaseUse(allocator, free_vecs, rem_uses, slots, idx);
        return mask;
    }
    switch (nodes[idx]) {
        .unary => |u| {
            // `.not`: `!(a > 0)` in i16.
            b.vecOp3(.cmp_gt_i16, mask, slots[u.a].?.reg, b.vecZero());
            b.vecOp2(.not_u32, mask, mask);
            try releaseUse(allocator, free_vecs, rem_uses, slots, u.a);
            return mask;
        },
        .binary => |bin| switch (bin.op) {
            .gt => {
                b.vecOp3(.cmp_gt_i16, mask, slots[bin.a].?.reg, slots[bin.b].?.reg);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.a);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.b);
                return mask;
            },
            .lt => {
                b.vecOp3(.cmp_gt_i16, mask, slots[bin.b].?.reg, slots[bin.a].?.reg);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.a);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.b);
                return mask;
            },
            .eq => {
                b.vecOp3(.cmp_eq_i16, mask, slots[bin.a].?.reg, slots[bin.b].?.reg);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.a);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.b);
                return mask;
            },
            .le => {
                b.vecOp3(.cmp_gt_i16, mask, slots[bin.a].?.reg, slots[bin.b].?.reg);
                b.vecOp2(.not_u32, mask, mask);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.a);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.b);
                return mask;
            },
            .ge => {
                b.vecOp3(.cmp_gt_i16, mask, slots[bin.b].?.reg, slots[bin.a].?.reg);
                b.vecOp2(.not_u32, mask, mask);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.a);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.b);
                return mask;
            },
            .and_op, .or_op, .xor_op => {
                const ma = try emitCmpOrLogicMaskI16(allocator, b, nodes, slots, rem_uses, fused, free_vecs, bin.a);
                const mb = try emitCmpOrLogicMaskI16(allocator, b, nodes, slots, rem_uses, fused, free_vecs, bin.b);
                switch (bin.op) {
                    .and_op => b.vecOp3(.and_u32, mask, ma, mb),
                    .or_op => b.vecOp3(.or_u32, mask, ma, mb),
                    .xor_op => b.vecOp3(.xor_u32, mask, ma, mb),
                    else => unreachable,
                }
                try free_vecs.append(allocator, ma);
                try free_vecs.append(allocator, mb);
                return mask;
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

/// Evaluate one `i16` vector (`2 * f32_lanes` pixels) at horizontal offset `x_gp + col_disp`, returning the result register.
fn evalI16Vector(
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    width: usize,
    x_gp: aj.Reg,
    col_disp: i32,
    src_row_ptrs: []const ?aj.Reg,
    row_ptr_resolver: anytype,
    free_vecs: *std.ArrayList(aj.Reg),
) std.mem.Allocator.Error!aj.Reg {
    _ = x_gp;
    const final_shift = info.int16_plan.final_div_shift;
    const eval_root: ast.NodeIdx = if (info.int16_plan.eval_root != ast.invalid_node)
        info.int16_plan.eval_root
    else
        prog.root;

    const n_nodes = prog.nodes.items.len;
    const slots = try allocator.alloc(?RegSlot, n_nodes);
    defer allocator.free(slots);
    @memset(slots, null);

    const rem_uses = try allocator.alloc(u32, n_nodes);
    defer allocator.free(rem_uses);
    @memset(rem_uses, 0);
    rem_uses[eval_root] = 1;
    var ri: usize = eval_root + 1;
    while (ri > 0) {
        ri -= 1;
        if (rem_uses[ri] == 0) continue;
        switch (prog.nodes.items[ri]) {
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

    const acquireVec = struct {
        fn run(builder: *aj.Builder, pool: *std.ArrayList(aj.Reg)) aj.Reg {
            if (pool.items.len > 0) return pool.pop().?;
            return builder.newVec();
        }
    }.run;

    const releaseUse = struct {
        fn run(alloc: std.mem.Allocator, pool: *std.ArrayList(aj.Reg), rem: []u32, slot_arr: []?RegSlot, idx: ast.NodeIdx) std.mem.Allocator.Error!void {
            if (rem[idx] == 0) return;
            rem[idx] -= 1;
            if (rem[idx] == 0) {
                if (slot_arr[idx]) |s| {
                    if (s.owned) {
                        try pool.append(alloc, s.reg);
                        slot_arr[idx] = null;
                    }
                }
            }
        }
    }.run;

    const emitMulI16 = struct {
        fn run(builder: *aj.Builder, dst: aj.Reg, lhs_idx: ast.NodeIdx, rhs_idx: ast.NodeIdx, lhs_reg: aj.Reg, rhs_reg: aj.Reg, nodes: []const ast.Node) void {
            if (exactIntI16(nodes[rhs_idx])) |iv| {
                if (pow2ShiftI16(iv)) |sh| {
                    if (sh == 0) {
                        builder.vecOp2(.mov, dst, lhs_reg);
                    } else {
                        builder.vecOp2i(.sll_u16, dst, lhs_reg, sh);
                    }
                    return;
                }
            }
            if (exactIntI16(nodes[lhs_idx])) |iv| {
                if (pow2ShiftI16(iv)) |sh| {
                    if (sh == 0) {
                        builder.vecOp2(.mov, dst, rhs_reg);
                    } else {
                        builder.vecOp2i(.sll_u16, dst, rhs_reg, sh);
                    }
                    return;
                }
            }
            builder.vecOp3(.mul_i16, dst, lhs_reg, rhs_reg);
        }
    }.run;

    const fused_bool = try allocator.alloc(bool, n_nodes);
    defer allocator.free(fused_bool);
    @memset(fused_bool, false);
    for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
        if (rem_uses[idx] == 0) continue;
        if (node == .ternary and node.ternary.op == .select) {
            markFusableBoolTreeI16(prog.nodes.items, rem_uses, fused_bool, node.ternary.a);
        }
    }

    const w_i32: i32 = @intCast(width);

    for (prog.nodes.items[0 .. eval_root + 1], 0..) |node, idx| {
        if (rem_uses[idx] == 0 or fused_bool[idx]) continue;
        switch (node) {
            .const_f32 => |v| {
                const iv: i16 = @intFromFloat(v);
                slots[idx] = .{ .reg = constI16Splat(b, iv), .owned = false };
            },
            .const_i32 => |v| {
                const iv: i16 = @intCast(v);
                slots[idx] = .{ .reg = constI16Splat(b, iv), .owned = false };
            },
            .load_clip => |clip| {
                const dst = acquireVec(b, free_vecs);
                b.vecLoad(.cvt_u8_to_u16, dst, row_ptr_resolver.rowMem(src_row_ptrs[clip].?, 0, col_disp));
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .load_rel => |rel| {
                const dst = acquireVec(b, free_vecs);
                const row_ptr = row_ptr_resolver.rowPtrFor(rel.clip, rel.dy, rel.mode);
                const eff_dx: i32 = interp.clampRelOffset(rel.dx, w_i32);
                b.vecLoad(.cvt_u8_to_u16, dst, row_ptr_resolver.rowMem(row_ptr, 0, eff_dx + col_disp));
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .unary => |u| {
                const a_reg = slots[u.a].?.reg;
                const dst = acquireVec(b, free_vecs);
                switch (u.op) {
                    .neg => b.vecOp2(.neg_i16, dst, a_reg),
                    .abs => b.vecOp2(.abs_i16, dst, a_reg),
                    .not => {
                        b.vecOp3(.cmp_gt_i16, dst, a_reg, b.vecZero());
                        b.vecOp2(.not_u32, dst, dst);
                        b.vecOp3(.and_u32, dst, dst, constI16Splat(b, 1));
                    },
                    else => unreachable,
                }
                try releaseUse(allocator, free_vecs, rem_uses, slots, u.a);
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .binary => |bin| {
                const a_reg = slots[bin.a].?.reg;
                const b_reg = slots[bin.b].?.reg;
                const dst = acquireVec(b, free_vecs);
                switch (bin.op) {
                    .add => b.vecOp3(.add_i16, dst, a_reg, b_reg),
                    .sub => b.vecOp3(.sub_i16, dst, a_reg, b_reg),
                    .mul => emitMulI16(b, dst, bin.a, bin.b, a_reg, b_reg, prog.nodes.items),
                    .min => b.vecOp3(.min_i16, dst, a_reg, b_reg),
                    .max => b.vecOp3(.max_i16, dst, a_reg, b_reg),
                    .gt => {
                        b.vecOp3(.cmp_gt_i16, dst, a_reg, b_reg);
                        b.vecOp3(.and_u32, dst, dst, constI16Splat(b, 1));
                    },
                    .lt => {
                        b.vecOp3(.cmp_gt_i16, dst, b_reg, a_reg);
                        b.vecOp3(.and_u32, dst, dst, constI16Splat(b, 1));
                    },
                    .eq => {
                        b.vecOp3(.cmp_eq_i16, dst, a_reg, b_reg);
                        b.vecOp3(.and_u32, dst, dst, constI16Splat(b, 1));
                    },
                    .le => {
                        b.vecOp3(.cmp_gt_i16, dst, a_reg, b_reg);
                        b.vecOp2(.not_u32, dst, dst);
                        b.vecOp3(.and_u32, dst, dst, constI16Splat(b, 1));
                    },
                    .ge => {
                        b.vecOp3(.cmp_gt_i16, dst, b_reg, a_reg);
                        b.vecOp2(.not_u32, dst, dst);
                        b.vecOp3(.and_u32, dst, dst, constI16Splat(b, 1));
                    },
                    .and_op, .or_op, .xor_op => {
                        const ma = acquireVec(b, free_vecs);
                        const mb = acquireVec(b, free_vecs);
                        b.vecOp3(.cmp_gt_i16, ma, a_reg, b.vecZero());
                        b.vecOp3(.cmp_gt_i16, mb, b_reg, b.vecZero());
                        switch (bin.op) {
                            .and_op => b.vecOp3(.and_u32, dst, ma, mb),
                            .or_op => b.vecOp3(.or_u32, dst, ma, mb),
                            .xor_op => b.vecOp3(.xor_u32, dst, ma, mb),
                            else => unreachable,
                        }
                        b.vecOp3(.and_u32, dst, dst, constI16Splat(b, 1));
                        try free_vecs.append(allocator, ma);
                        try free_vecs.append(allocator, mb);
                    },
                    else => unreachable,
                }
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.a);
                try releaseUse(allocator, free_vecs, rem_uses, slots, bin.b);
                slots[idx] = .{ .reg = dst, .owned = true };
            },
            .ternary => |t| {
                const dst = acquireVec(b, free_vecs);
                if (t.op == .select) {
                    const mask = try emitCmpOrLogicMaskI16(allocator, b, prog.nodes.items, slots, rem_uses, fused_bool, free_vecs, t.a);
                    const b_is_zero = exactIntI16(prog.nodes.items[t.b]) == 0;
                    const c_is_zero = exactIntI16(prog.nodes.items[t.c]) == 0;
                    if (c_is_zero) {
                        b.vecOp3(.and_u32, dst, slots[t.b].?.reg, mask);
                    } else if (b_is_zero) {
                        b.vecOp3(.andn_u32, dst, mask, slots[t.c].?.reg);
                    } else {
                        pixel.emitSelectByMask(b, dst, slots[t.c].?.reg, slots[t.b].?.reg, mask);
                    }
                    try free_vecs.append(allocator, mask);
                    try releaseUse(allocator, free_vecs, rem_uses, slots, t.b);
                    try releaseUse(allocator, free_vecs, rem_uses, slots, t.c);
                } else {
                    const a_reg = slots[t.a].?.reg;
                    const b_reg = slots[t.b].?.reg;
                    const c_reg = slots[t.c].?.reg;
                    emitMulI16(b, dst, t.a, t.b, a_reg, b_reg, prog.nodes.items);
                    switch (t.op) {
                        .fmadd => b.vecOp3(.add_i16, dst, dst, c_reg),
                        .fmsub => b.vecOp3(.sub_i16, dst, dst, c_reg),
                        .fnmadd => b.vecOp3(.sub_i16, dst, c_reg, dst),
                        .fnmsub => {
                            b.vecOp2(.neg_i16, dst, dst);
                            b.vecOp3(.sub_i16, dst, dst, c_reg);
                        },
                        .select, .linstep => unreachable,
                    }
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
        // Exact round-half-to-even division by 2^k in i16:
        // `(v + (1 << (k - 1)) - 1 + ((v >> k) & 1)) >> k`
        const out = acquireVec(b, free_vecs);
        const lsb = acquireVec(b, free_vecs);
        b.vecOp2i(.sra_i16, lsb, res, final_shift);
        b.vecOp3(.and_u32, lsb, lsb, constI16Splat(b, 1));
        const bias: i16 = @intCast((@as(i32, 1) << (final_shift - 1)) - 1);
        if (bias != 0) {
            b.vecOp3(.add_i16, out, res, constI16Splat(b, bias));
            b.vecOp3(.add_i16, out, out, lsb);
        } else {
            b.vecOp3(.add_i16, out, res, lsb);
        }
        b.vecOp2i(.sra_i16, out, out, final_shift);
        try free_vecs.append(allocator, lsb);
        if (slots[eval_root].?.owned) try free_vecs.append(allocator, res);
        res = out;
    }
    return res;
}

pub fn emitInteriorBody(
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    width: usize,
    dst_ptr: aj.Reg,
    x_gp: aj.Reg,
    src_row_ptrs: []const ?aj.Reg,
    row_ptr_resolver: anytype,
) std.mem.Allocator.Error!void {
    var free_vecs: std.ArrayList(aj.Reg) = .empty;
    defer free_vecs.deinit(allocator);
    const res = try evalI16Vector(allocator, b, prog, info, width, x_gp, 0, src_row_ptrs, row_ptr_resolver, &free_vecs);
    b.vecStoreU8FromI16(row_ptr_resolver.rowMem(dst_ptr, 0, 0), res);
}

/// 2x unrolled `i16` interior body (`4 * f32_lanes` pixels per iteration: 16 on 128b, 32 on AVX2, 64 on AVX-512)
/// packing both `i16` results together via `vecStoreU8FromI16x2` (1 pack + 1 permute + 1 32B store on AVX2).
pub fn emitInteriorBodyX2(
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    width: usize,
    dst_ptr: aj.Reg,
    x_gp: aj.Reg,
    src_row_ptrs: []const ?aj.Reg,
    row_ptr_resolver: anytype,
) std.mem.Allocator.Error!void {
    var free_vecs: std.ArrayList(aj.Reg) = .empty;
    defer free_vecs.deinit(allocator);
    const half_step: i32 = @intCast(b.cpu.f32_lanes * 2);
    const res0 = try evalI16Vector(allocator, b, prog, info, width, x_gp, 0, src_row_ptrs, row_ptr_resolver, &free_vecs);
    const res1 = try evalI16Vector(allocator, b, prog, info, width, x_gp, half_step, src_row_ptrs, row_ptr_resolver, &free_vecs);
    b.vecStoreU8FromI16x2(row_ptr_resolver.rowMem(dst_ptr, 0, 0), res0, res1);
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
