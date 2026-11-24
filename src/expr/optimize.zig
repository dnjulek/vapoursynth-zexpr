//! Optimizer passes (`src/expr/optimize.zig`) strictly separated into:
//! 1. Exact rewrites (bit-identical for all inputs incl. -0.0, +-inf, NaN):
//!    constant folding, exact identities (omitting non-exact `!(a<b)`/`!c?` per Y5), CSE, DCE.
//! 2. Documented `std.Expr`-style value-changing rewrites:
//!    `x / C -> x * (1/C)` when `isFinite(1/C)`, FMA fusion (`use_count(mul) == 1`, keeping
//!    `Op.mod` atomic per R3), `exp(log)`/`log(exp)`, special-value algebraic identities (T5).

const std = @import("std");
const ast = @import("ast.zig");
const interp = @import("interp.zig");
const lower_mod = @import("lower.zig");

pub const OptOptions = struct {
    enable_value_changing: bool = true,
    enable_fma: bool = true,
};

fn nodeConstVal(node: ast.Node) ?interp.EvalVal {
    return switch (node) {
        .const_f32 => |v| .{ .f = v },
        .const_i32 => |v| .{ .i = v },
        else => null,
    };
}

fn constNode(val: interp.EvalVal) ast.Node {
    return switch (val) {
        .f => |v| .{ .const_f32 = v },
        .i => |v| .{ .const_i32 = v },
    };
}

/// True iff `c` is a finite normal power of two whose reciprocal `1/c` is also normal
/// (so `x / c == x * (1 / c)` is bit-exact for every float32 input including subnormals/zeros/NaN).
fn isExactPow2Divisor(c: f32) bool {
    const bits: u32 = @bitCast(c);
    const mant = bits & 0x007F_FFFF;
    const exp = (bits >> 23) & 0xFF;
    // Normal power of two with exponent in [2, 253] guarantees 1/c has exponent in [1, 252] (also normal).
    return mant == 0 and exp >= 2 and exp <= 253;
}

const CseKey = struct {
    tag: u8,
    a: u32,
    b: u32,
    c: u32,
    extra: u32,
};

fn nodeCseKey(node: ast.Node) ?CseKey {
    return switch (node) {
        .const_f32 => |v| .{ .tag = 1, .a = @bitCast(v), .b = 0, .c = 0, .extra = 0 },
        .const_i32 => |v| .{ .tag = 2, .a = @bitCast(v), .b = 0, .c = 0, .extra = 0 },
        .load_clip => |clip| .{ .tag = 3, .a = clip, .b = 0, .c = 0, .extra = 0 },
        .load_rel => |r| .{ .tag = 4, .a = r.clip, .b = @bitCast(r.dx), .c = @bitCast(r.dy), .extra = @backingInt(r.mode) },
        .load_abs => |la| .{ .tag = 5, .a = la.clip, .b = la.x, .c = la.y, .extra = @backingInt(la.mode) },
        .load_prop => |pid| .{ .tag = 6, .a = pid, .b = 0, .c = 0, .extra = 0 },
        .coord_x => .{ .tag = 7, .a = 0, .b = 0, .c = 0, .extra = 0 },
        .coord_y => .{ .tag = 8, .a = 0, .b = 0, .c = 0, .extra = 0 },
        .frame_n => .{ .tag = 9, .a = 0, .b = 0, .c = 0, .extra = 0 },
        .plane_width => .{ .tag = 10, .a = 0, .b = 0, .c = 0, .extra = 0 },
        .plane_height => .{ .tag = 11, .a = 0, .b = 0, .c = 0, .extra = 0 },
        .unary => |u| .{ .tag = 12, .a = @backingInt(u.op), .b = u.a, .c = 0, .extra = 0 },
        .binary => |b| .{ .tag = 13, .a = @backingInt(b.op), .b = b.a, .c = b.b, .extra = 0 },
        .ternary => |t| .{ .tag = 14, .a = @backingInt(t.op), .b = t.a, .c = t.b, .extra = t.c },
        .arg_extrema, .arg_sort_elem => null,
    };
}

const KnownType = enum(u8) {
    unknown,
    f32,
    i32,
};

fn nodeKnownType(nodes: []const ast.Node, idx: ast.NodeIdx) KnownType {
    return switch (nodes[idx]) {
        .const_f32, .load_prop, .arg_extrema, .arg_sort_elem => .f32,
        .const_i32, .coord_x, .coord_y, .frame_n, .plane_width, .plane_height => .i32,
        .load_clip, .load_rel, .load_abs => .unknown,
        .unary => |u| switch (u.op) {
            .not, .bitnot => .i32,
            .neg, .abs, .sgn => nodeKnownType(nodes, u.a),
            else => .f32,
        },
        .binary => |b| switch (b.op) {
            .lt, .le, .gt, .ge, .eq, .and_op, .or_op, .xor_op, .bitand, .bitor, .bitxor => .i32,
            .add, .sub, .mul, .min, .max => blk: {
                const ta = nodeKnownType(nodes, b.a);
                const tb = nodeKnownType(nodes, b.b);
                if (ta == .f32 or tb == .f32) break :blk .f32;
                if (ta == .i32 and tb == .i32) break :blk .i32;
                break :blk .unknown;
            },
            else => .f32,
        },
        .ternary => |t| if (t.op == .select) blk: {
            const tb = nodeKnownType(nodes, t.b);
            const tc = nodeKnownType(nodes, t.c);
            if (tb == .f32 or tc == .f32) break :blk .f32;
            if (tb == .i32 and tc == .i32) break :blk .i32;
            break :blk .unknown;
        } else .f32,
    };
}

/// Run optimizer passes on a lowered `ast.Program`.
pub fn optimize(
    allocator: std.mem.Allocator,
    prog: *ast.Program,
    profile: interp.TargetProfile,
    opts: OptOptions,
) std.mem.Allocator.Error!void {
    // Pass 1: Algebraic / identity / constant-fold + GVN/CSE fixpoint (up to 4 sweeps).
    for (0..4) |_| {
        const changed = try rewriteAndCsePass(allocator, prog, profile, opts);
        if (!changed) break;
    }

    // Pass 2: FMA fusion when target has FMA and `use_count(mul) == 1` (never touches `Op.mod`, R3).
    if (opts.enable_value_changing and opts.enable_fma and profile.use_fma and !prog.opt_int) {
        try fuseFmaPass(allocator, prog);
    }

    // Pass 3: Dead code elimination (compact to live nodes in topological order).
    try dcePass(allocator, prog);
}

fn rewriteAndCsePass(
    allocator: std.mem.Allocator,
    prog: *ast.Program,
    profile: interp.TargetProfile,
    opts: OptOptions,
) std.mem.Allocator.Error!bool {
    const old_nodes = prog.nodes.items;
    var new_nodes: std.ArrayList(ast.Node) = .empty;
    errdefer new_nodes.deinit(allocator);
    try new_nodes.ensureTotalCapacity(allocator, old_nodes.len);

    const remap = try allocator.alloc(ast.NodeIdx, old_nodes.len);
    defer allocator.free(remap);

    var cse: std.AutoHashMapUnmanaged(CseKey, ast.NodeIdx) = .empty;
    defer cse.deinit(allocator);

    var changed = false;

    const intern = struct {
        fn run(alloc: std.mem.Allocator, nodes: *std.ArrayList(ast.Node), map: *std.AutoHashMapUnmanaged(CseKey, ast.NodeIdx), node: ast.Node) std.mem.Allocator.Error!ast.NodeIdx {
            if (nodeCseKey(node)) |key| {
                if (map.get(key)) |existing| return existing;
                const idx: ast.NodeIdx = @intCast(nodes.items.len);
                try nodes.append(alloc, node);
                try map.put(alloc, key, idx);
                return idx;
            }
            const idx: ast.NodeIdx = @intCast(nodes.items.len);
            try nodes.append(alloc, node);
            return idx;
        }
    }.run;

    for (old_nodes, 0..) |node, old_idx| {
        const out_idx: ast.NodeIdx = switch (node) {
            .const_f32, .const_i32, .load_clip, .load_rel, .load_prop, .coord_x, .coord_y, .frame_n, .plane_width, .plane_height, .arg_extrema, .arg_sort_elem => try intern(allocator, &new_nodes, &cse, node),
            .load_abs => |la| try intern(allocator, &new_nodes, &cse, .{ .load_abs = .{
                .clip = la.clip,
                .x = remap[la.x],
                .y = remap[la.y],
                .mode = la.mode,
            } }),
            .unary => |u| blk: {
                const a = remap[u.a];
                if (nodeConstVal(new_nodes.items[a])) |ca| {
                    changed = true;
                    const folded = lower_mod.foldConstUnary(profile, u.op, ca, prog.opt_int);
                    break :blk try intern(allocator, &new_nodes, &cse, constNode(folded));
                }
                // Exact identities: abs(abs(x)) -> abs(x), to_f32(to_f32(x)) -> to_f32(x), neg(neg(x)) -> x.
                if ((u.op == .abs or u.op == .to_f32) and new_nodes.items[a] == .unary and new_nodes.items[a].unary.op == u.op) {
                    changed = true;
                    break :blk a;
                }
                if (u.op == .neg and new_nodes.items[a] == .unary and new_nodes.items[a].unary.op == .neg) {
                    changed = true;
                    break :blk new_nodes.items[a].unary.a;
                }
                if (u.op == .abs and new_nodes.items[a] == .binary and new_nodes.items[a].binary.op == .copysign) {
                    changed = true;
                    break :blk try intern(allocator, &new_nodes, &cse, .{ .unary = .{ .op = .abs, .a = new_nodes.items[a].binary.a } });
                }
                // Value-changing inverse cancellations (Q2.1): exp(log(x)) -> x, log(exp(x)) -> x, exp2(log2(x)) -> x, log2(exp2(x)) -> x.
                if (opts.enable_value_changing and !prog.opt_int) {
                    if (u.op == .exp and new_nodes.items[a] == .unary and new_nodes.items[a].unary.op == .log) {
                        changed = true;
                        break :blk new_nodes.items[a].unary.a;
                    }
                    if (u.op == .log and new_nodes.items[a] == .unary and new_nodes.items[a].unary.op == .exp) {
                        changed = true;
                        break :blk new_nodes.items[a].unary.a;
                    }
                    if (u.op == .exp2 and new_nodes.items[a] == .unary and new_nodes.items[a].unary.op == .log2) {
                        changed = true;
                        break :blk new_nodes.items[a].unary.a;
                    }
                    if (u.op == .log2 and new_nodes.items[a] == .unary and new_nodes.items[a].unary.op == .exp2) {
                        changed = true;
                        break :blk new_nodes.items[a].unary.a;
                    }
                }
                break :blk try intern(allocator, &new_nodes, &cse, .{ .unary = .{ .op = u.op, .a = a } });
            },
            .binary => |b| blk: {
                const a = remap[b.a];
                const rhs = remap[b.b];
                const ca = nodeConstVal(new_nodes.items[a]);
                const cb = nodeConstVal(new_nodes.items[rhs]);

                if (ca != null and cb != null) {
                    changed = true;
                    const folded = lower_mod.foldConstBinary(profile, b.op, ca.?, cb.?, prog.opt_int);
                    break :blk try intern(allocator, &new_nodes, &cse, constNode(folded));
                }

                // Exact identities (all bit-exact for every float32 input):
                if (prog.opt_int) {
                    // Under opt=1, `/` always promotes to f32. Rewrite constant divisor to `* const_f32(1/C)`
                    // (preserving `.f32` type via `.const_f32`, never simplifying `/ 1` to `a`).
                    if (b.op == .div and cb != null) {
                        const div_c = cb.?.asF32();
                        if (isExactPow2Divisor(div_c)) {
                            changed = true;
                            const inv_node = try intern(allocator, &new_nodes, &cse, .{ .const_f32 = 1.0 / div_c });
                            break :blk try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .mul, .a = a, .b = inv_node } });
                        }
                        if (opts.enable_value_changing) {
                            const inv = 1.0 / div_c;
                            if (std.math.isFinite(inv)) {
                                changed = true;
                                const inv_node = try intern(allocator, &new_nodes, &cse, .{ .const_f32 = inv });
                                break :blk try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .mul, .a = a, .b = inv_node } });
                            }
                        }
                    }
                } else {
                    if (b.op == .mul) {
                        if (cb != null and cb.?.asF32() == 1.0) {
                            changed = true;
                            break :blk a;
                        }
                        if (ca != null and ca.?.asF32() == 1.0) {
                            changed = true;
                            break :blk rhs;
                        }
                    }
                    if (b.op == .div and cb != null) {
                        const div_c = cb.?.asF32();
                        if (div_c == 1.0) {
                            changed = true;
                            break :blk a;
                        }
                        // Exact power-of-two reciprocal rewrite (always exact).
                        if (isExactPow2Divisor(div_c)) {
                            changed = true;
                            const inv_node = try intern(allocator, &new_nodes, &cse, .{ .const_f32 = 1.0 / div_c });
                            break :blk try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .mul, .a = a, .b = inv_node } });
                        }
                        // Documented std.Expr-style reciprocal rewrite: x / C -> x * (1 / C) when isFinite(1 / C).
                        if (opts.enable_value_changing) {
                            const inv = 1.0 / div_c;
                            if (std.math.isFinite(inv)) {
                                changed = true;
                                const inv_node = try intern(allocator, &new_nodes, &cse, .{ .const_f32 = inv });
                                break :blk try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .mul, .a = a, .b = inv_node } });
                            }
                        }
                    }

                    // Documented value-changing special-value algebraic identities (Q2.2, T5):
                    if (opts.enable_value_changing) {
                        switch (b.op) {
                            .add => {
                                if (cb != null and cb.?.asF32() == 0.0) {
                                    changed = true;
                                    break :blk a;
                                }
                                if (ca != null and ca.?.asF32() == 0.0) {
                                    changed = true;
                                    break :blk rhs;
                                }
                            },
                            .sub => {
                                if (cb != null and cb.?.asF32() == 0.0) {
                                    changed = true;
                                    break :blk a;
                                }
                                if (a == rhs) {
                                    changed = true;
                                    break :blk try intern(allocator, &new_nodes, &cse, .{ .const_f32 = 0.0 });
                                }
                            },
                            .mul => {
                                if ((cb != null and cb.?.asF32() == 0.0) or (ca != null and ca.?.asF32() == 0.0)) {
                                    changed = true;
                                    break :blk try intern(allocator, &new_nodes, &cse, .{ .const_f32 = 0.0 });
                                }
                                // Constant-factor folding: `(x * C1) * C2 -> x * (C1 * C2)` when finite & non-zero.
                                if (cb != null and new_nodes.items[a] == .binary and new_nodes.items[a].binary.op == .mul) {
                                    const inner = new_nodes.items[a].binary;
                                    const inner_ca = nodeConstVal(new_nodes.items[inner.a]);
                                    const inner_cb = nodeConstVal(new_nodes.items[inner.b]);
                                    const c2 = cb.?.asF32();
                                    if (inner_cb != null) {
                                        const prod = inner_cb.?.asF32() * c2;
                                        if (std.math.isFinite(prod) and prod != 0.0) {
                                            changed = true;
                                            const p_node = try intern(allocator, &new_nodes, &cse, .{ .const_f32 = prod });
                                            break :blk try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .mul, .a = inner.a, .b = p_node } });
                                        }
                                    } else if (inner_ca != null) {
                                        const prod = inner_ca.?.asF32() * c2;
                                        if (std.math.isFinite(prod) and prod != 0.0) {
                                            changed = true;
                                            const p_node = try intern(allocator, &new_nodes, &cse, .{ .const_f32 = prod });
                                            break :blk try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .mul, .a = inner.b, .b = p_node } });
                                        }
                                    }
                                }
                                if (ca != null and new_nodes.items[rhs] == .binary and new_nodes.items[rhs].binary.op == .mul) {
                                    const inner = new_nodes.items[rhs].binary;
                                    const inner_ca = nodeConstVal(new_nodes.items[inner.a]);
                                    const inner_cb = nodeConstVal(new_nodes.items[inner.b]);
                                    const c2 = ca.?.asF32();
                                    if (inner_cb != null) {
                                        const prod = inner_cb.?.asF32() * c2;
                                        if (std.math.isFinite(prod) and prod != 0.0) {
                                            changed = true;
                                            const p_node = try intern(allocator, &new_nodes, &cse, .{ .const_f32 = prod });
                                            break :blk try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .mul, .a = inner.a, .b = p_node } });
                                        }
                                    } else if (inner_ca != null) {
                                        const prod = inner_ca.?.asF32() * c2;
                                        if (std.math.isFinite(prod) and prod != 0.0) {
                                            changed = true;
                                            const p_node = try intern(allocator, &new_nodes, &cse, .{ .const_f32 = prod });
                                            break :blk try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .mul, .a = inner.b, .b = p_node } });
                                        }
                                    }
                                }
                            },
                            .div => {
                                if (ca != null and ca.?.asF32() == 0.0) {
                                    changed = true;
                                    break :blk try intern(allocator, &new_nodes, &cse, .{ .const_f32 = 0.0 });
                                }
                            },
                            .min, .max => {
                                if (a == rhs) {
                                    changed = true;
                                    break :blk a;
                                }
                            },
                            else => {},
                        }
                    }
                }

                break :blk try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = b.op, .a = a, .b = rhs } });
            },
            .ternary => |t| blk: {
                const a = remap[t.a];
                const b = remap[t.b];
                const c = remap[t.c];
                const ca = nodeConstVal(new_nodes.items[a]);
                const cb = nodeConstVal(new_nodes.items[b]);
                const cc = nodeConstVal(new_nodes.items[c]);

                if (ca != null and cb != null and cc != null) {
                    changed = true;
                    const folded = lower_mod.foldConstTernary(profile, t.op, ca.?, cb.?, cc.?, prog.opt_int);
                    break :blk try intern(allocator, &new_nodes, &cse, constNode(folded));
                }
                // Exact select identities: `const ? a : b` and `c ? x : x -> x`.
                if (t.op == .select) {
                    if (ca) |cond_val| {
                        const cond = switch (cond_val) {
                            .f => |fv| fv > 0.0,
                            .i => |iv| iv > 0,
                        };
                        const kept = if (cond) b else c;
                        const dropped = if (cond) c else b;
                        if (!prog.opt_int) {
                            changed = true;
                            break :blk kept;
                        }
                        // V4: under `opt=1`, preserve `.f32` result type if the discarded branch is `.f32`.
                        const tk = nodeKnownType(new_nodes.items, kept);
                        const td = nodeKnownType(new_nodes.items, dropped);
                        if (td == .i32 or tk == .f32) {
                            changed = true;
                            break :blk kept;
                        }
                        if (td == .f32) {
                            changed = true;
                            break :blk try intern(allocator, &new_nodes, &cse, .{ .unary = .{ .op = .to_f32, .a = kept } });
                        }
                        // Mixed clip format unknown at optimize time: keep select so analysis types it accurately.
                    }
                    if (b == c) {
                        changed = true;
                        break :blk b;
                    }
                    // Signed diff-limit collapse (contrasharpening_dehalo / sbr idiom):
                    // `select(xor(d1, d2), base, base +/- select(|d1| < |d2|, d1, d2))`
                    //   -> `base +/- clamp(d1, min(0, d2), max(0, d2))` (4 min/max ops, enables i16 SIMD).
                    if (opts.enable_value_changing and !prog.opt_int and
                        new_nodes.items[a] == .binary and new_nodes.items[a].binary.op == .xor_op)
                    {
                        const x_a = new_nodes.items[a].binary.a;
                        const x_b = new_nodes.items[a].binary.b;
                        var combine_op: ?ast.BinaryOp = null;
                        var base_node: ast.NodeIdx = ast.invalid_node;
                        var inner_idx: ast.NodeIdx = ast.invalid_node;
                        if (cb != null and cb.?.asF32() == 0.0) {
                            inner_idx = c;
                        } else if (new_nodes.items[c] == .binary) {
                            const cb_node = new_nodes.items[c].binary;
                            if (cb_node.op == .add) {
                                if (cb_node.a == b) {
                                    combine_op = .add;
                                    base_node = b;
                                    inner_idx = cb_node.b;
                                } else if (cb_node.b == b) {
                                    combine_op = .add;
                                    base_node = b;
                                    inner_idx = cb_node.a;
                                }
                            } else if (cb_node.op == .sub and cb_node.a == b) {
                                combine_op = .sub;
                                base_node = b;
                                inner_idx = cb_node.b;
                            }
                        }
                        if (inner_idx != ast.invalid_node and new_nodes.items[inner_idx] == .ternary and
                            new_nodes.items[inner_idx].ternary.op == .select)
                        {
                            const in_t = new_nodes.items[inner_idx].ternary;
                            if (new_nodes.items[in_t.a] == .binary and
                                (new_nodes.items[in_t.a].binary.op == .lt or new_nodes.items[in_t.a].binary.op == .le))
                            {
                                const cmp = new_nodes.items[in_t.a].binary;
                                if (new_nodes.items[cmp.a] == .unary and new_nodes.items[cmp.a].unary.op == .abs and
                                    new_nodes.items[cmp.b] == .unary and new_nodes.items[cmp.b].unary.op == .abs)
                                {
                                    const d1 = new_nodes.items[cmp.a].unary.a;
                                    const d2 = new_nodes.items[cmp.b].unary.a;
                                    const branches_ok = (in_t.b == d1 and in_t.c == d2);
                                    const xor_ok = (x_a == d1 and x_b == d2) or (x_a == d2 and x_b == d1);
                                    if (branches_ok and xor_ok) {
                                        changed = true;
                                        const zero_n = try intern(allocator, &new_nodes, &cse, .{ .const_f32 = 0.0 });
                                        const lo_n = try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .min, .a = zero_n, .b = d2 } });
                                        const hi_n = try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .max, .a = zero_n, .b = d2 } });
                                        const min_n = try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .min, .a = d1, .b = hi_n } });
                                        const clamped = try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = .max, .a = min_n, .b = lo_n } });
                                        if (combine_op) |op| {
                                            break :blk try intern(allocator, &new_nodes, &cse, .{ .binary = .{ .op = op, .a = base_node, .b = clamped } });
                                        }
                                        break :blk clamped;
                                    }
                                }
                            }
                        }
                    }
                }
                break :blk try intern(allocator, &new_nodes, &cse, .{ .ternary = .{ .op = t.op, .a = a, .b = b, .c = c } });
            },
        };
        remap[old_idx] = out_idx;
    }

    for (prog.extra_args.items) |*arg| {
        arg.* = remap[arg.*];
    }
    prog.root = remap[prog.root];

    if (new_nodes.items.len != old_nodes.len) changed = true;
    prog.nodes.deinit(allocator);
    prog.nodes = new_nodes;
    return changed;
}

fn computeUseCounts(allocator: std.mem.Allocator, prog: *const ast.Program) std.mem.Allocator.Error![]u32 {
    const n = prog.nodes.items.len;
    const counts = try allocator.alloc(u32, n);
    @memset(counts, 0);
    if (prog.root < n) {
        counts[prog.root] = 1;
        var i: usize = n;
        while (i > 0) {
            i -= 1;
            if (counts[i] == 0) continue;
            switch (prog.nodes.items[i]) {
                .load_abs => |la| {
                    counts[la.x] += 1;
                    counts[la.y] += 1;
                },
                .unary => |u| counts[u.a] += 1,
                .binary => |b| {
                    counts[b.a] += 1;
                    counts[b.b] += 1;
                },
                .ternary => |t| {
                    counts[t.a] += 1;
                    counts[t.b] += 1;
                    counts[t.c] += 1;
                },
                .arg_extrema => |ae| {
                    for (prog.extra_args.items[ae.args_start..][0..ae.count]) |arg| counts[arg] += 1;
                },
                .arg_sort_elem => |as| {
                    const spec = prog.arg_sorts.items[as.sort_id];
                    for (prog.extra_args.items[spec.args_start..][0..spec.count]) |arg| counts[arg] += 1;
                },
                else => {},
            }
        }
    }
    return counts;
}

/// FMA fusion pass (B5, R3): fuses `add`/`sub` with single-use `mul` into `fmadd`/`fmsub`/`fnmadd`/`fnmsub`.
/// `Op.mod` is atomic and is never touched.
fn fuseFmaPass(allocator: std.mem.Allocator, prog: *ast.Program) std.mem.Allocator.Error!void {
    const uses = try computeUseCounts(allocator, prog);
    defer allocator.free(uses);

    const nodes = prog.nodes.items;
    for (nodes, 0..) |*node, idx| {
        if (uses[idx] == 0) continue;
        if (node.* != .binary) continue;
        const bin = node.binary;
        switch (bin.op) {
            .add => {
                if (uses[bin.a] == 1 and nodes[bin.a] == .binary and nodes[bin.a].binary.op == .mul) {
                    const m = nodes[bin.a].binary;
                    node.* = .{ .ternary = .{ .op = .fmadd, .a = m.a, .b = m.b, .c = bin.b } };
                } else if (uses[bin.b] == 1 and nodes[bin.b] == .binary and nodes[bin.b].binary.op == .mul) {
                    const m = nodes[bin.b].binary;
                    node.* = .{ .ternary = .{ .op = .fmadd, .a = m.a, .b = m.b, .c = bin.a } };
                }
            },
            .sub => {
                if (uses[bin.a] == 1 and nodes[bin.a] == .binary and nodes[bin.a].binary.op == .mul) {
                    const m = nodes[bin.a].binary;
                    node.* = .{ .ternary = .{ .op = .fmsub, .a = m.a, .b = m.b, .c = bin.b } };
                } else if (uses[bin.b] == 1 and nodes[bin.b] == .binary and nodes[bin.b].binary.op == .mul) {
                    const m = nodes[bin.b].binary;
                    node.* = .{ .ternary = .{ .op = .fnmadd, .a = m.a, .b = m.b, .c = bin.a } };
                } else if (uses[bin.a] == 1 and nodes[bin.a] == .unary and nodes[bin.a].unary.op == .neg) {
                    const inner = nodes[bin.a].unary.a;
                    if (uses[inner] == 1 and nodes[inner] == .binary and nodes[inner].binary.op == .mul) {
                        const m = nodes[inner].binary;
                        node.* = .{ .ternary = .{ .op = .fnmsub, .a = m.a, .b = m.b, .c = bin.b } };
                    }
                }
            },
            else => {},
        }
    }
}

fn dcePass(allocator: std.mem.Allocator, prog: *ast.Program) std.mem.Allocator.Error!void {
    const uses = try computeUseCounts(allocator, prog);
    defer allocator.free(uses);

    const old_nodes = prog.nodes.items;
    var new_nodes: std.ArrayList(ast.Node) = .empty;
    errdefer new_nodes.deinit(allocator);

    const remap = try allocator.alloc(ast.NodeIdx, old_nodes.len);
    defer allocator.free(remap);
    @memset(remap, ast.invalid_node);

    for (old_nodes, 0..) |node, old_idx| {
        if (uses[old_idx] == 0) continue;
        const new_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
        const remapped: ast.Node = switch (node) {
            .load_abs => |la| .{ .load_abs = .{ .clip = la.clip, .x = remap[la.x], .y = remap[la.y], .mode = la.mode } },
            .unary => |u| .{ .unary = .{ .op = u.op, .a = remap[u.a] } },
            .binary => |b| .{ .binary = .{ .op = b.op, .a = remap[b.a], .b = remap[b.b] } },
            .ternary => |t| .{ .ternary = .{ .op = t.op, .a = remap[t.a], .b = remap[t.b], .c = remap[t.c] } },
            else => node,
        };
        try new_nodes.append(allocator, remapped);
        remap[old_idx] = new_idx;
    }

    for (prog.extra_args.items) |*arg| {
        if (arg.* < remap.len and remap[arg.*] != ast.invalid_node) {
            arg.* = remap[arg.*];
        }
    }
    prog.root = remap[prog.root];

    prog.nodes.deinit(allocator);
    prog.nodes = new_nodes;
}

test "exact rewrites are bit-identical and R3 prevents FMA fusion on %" {
    const parser = @import("parser.zig");
    const allocator = std.testing.allocator;
    const fma_prof: interp.TargetProfile = .{ .use_fma = true, .isa_family = .x86 };

    const edge_vals = [_]f32{ -10.5, -0.0, 0.0, 0.7, 2.0, 1e10, std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
    const exact_exprs = [_][]const u8{
        "x 1 * 1 / 4 / abs abs",
        "x 0 < not", // Y5: must NOT rewrite to x >= 0!
        "x not 10 20 ?", // Y5: must NOT rewrite to x 20 10 ?!
        "x 3 %",
    };

    const fmt = ast.SampleFormat{ .is_float = true, .bits_per_sample = 32 };
    const fmts = [_]ast.SampleFormat{fmt};

    for (exact_exprs) |e| {
        var unopt = try parser.parse(allocator, e, .{ .num_clips = 1 });
        defer unopt.deinit(allocator);
        try lower_mod.lower(allocator, &unopt, fma_prof);

        var opt = try parser.parse(allocator, e, .{ .num_clips = 1 });
        defer opt.deinit(allocator);
        try lower_mod.lower(allocator, &opt, fma_prof);
        try optimize(allocator, &opt, fma_prof, .{ .enable_value_changing = false, .enable_fma = false });

        var src = edge_vals;
        var dst_u: [edge_vals.len]f32 = @splat(0);
        var dst_o: [edge_vals.len]f32 = @splat(0);
        const srcs = [_][*]const u8{@ptrCast(&src)};
        const strides = [_]isize{@sizeOf(@TypeOf(src))};

        try interp.evalPlane(allocator, &unopt, fma_prof, .{
            .dst = @ptrCast(&dst_u),
            .dst_stride = strides[0],
            .dst_fmt = fmt,
            .srcs = &srcs,
            .src_strides = &strides,
            .src_fmts = &fmts,
            .width = edge_vals.len,
            .height = 1,
            .frame_n = 0,
            .props = &.{},
        });
        try interp.evalPlane(allocator, &opt, fma_prof, .{
            .dst = @ptrCast(&dst_o),
            .dst_stride = strides[0],
            .dst_fmt = fmt,
            .srcs = &srcs,
            .src_strides = &strides,
            .src_fmts = &fmts,
            .width = edge_vals.len,
            .height = 1,
            .frame_n = 0,
            .props = &.{},
        });

        for (dst_u, dst_o) |u, o| {
            if (std.math.isNan(u)) {
                try std.testing.expect(std.math.isNan(o));
            } else {
                try std.testing.expectEqual(@as(u32, @bitCast(u)), @as(u32, @bitCast(o)));
            }
        }
    }

    // R3 regression: `x 3 %` with FMA enabled must stay atomic Op.mod and yield 0.0 on 1e10 (never 256).
    {
        var p = try parser.parse(allocator, "x 3 %", .{ .num_clips = 1 });
        defer p.deinit(allocator);
        try lower_mod.lower(allocator, &p, fma_prof);
        try optimize(allocator, &p, fma_prof, .{ .enable_value_changing = true, .enable_fma = true });
        try std.testing.expect(p.nodes.items[p.root] == .binary and p.nodes.items[p.root].binary.op == .mod);
    }

    // FMA fusion creates ternary fmadd for `x 0.5 * 0.25 +`.
    {
        var p = try parser.parse(allocator, "x 0.5 * 0.25 +", .{ .num_clips = 1 });
        defer p.deinit(allocator);
        try lower_mod.lower(allocator, &p, fma_prof);
        try optimize(allocator, &p, fma_prof, .{ .enable_value_changing = true, .enable_fma = true });
        try std.testing.expect(p.nodes.items[p.root] == .ternary and p.nodes.items[p.root].ternary.op == .fmadd);
    }

    // Constant-factor folding: `(x * 2.49) * 1.4` collapses to a single mul.
    {
        var p = try parser.parse(allocator, "x 2.49 * 1.4 *", .{ .num_clips = 1 });
        defer p.deinit(allocator);
        try lower_mod.lower(allocator, &p, fma_prof);
        try optimize(allocator, &p, fma_prof, .{ .enable_value_changing = true, .enable_fma = true });
        try std.testing.expect(p.nodes.items[p.root] == .binary and p.nodes.items[p.root].binary.op == .mul);
        try std.testing.expect(p.nodes.items[p.nodes.items[p.root].binary.a] == .load_clip);
    }

    // Signed diff-limit collapse (contrasharpening_dehalo / sbr): exact on all finite combinations.
    {
        const pairs = [_][2]f32{
            .{ 5.0, 2.0 },   .{ 2.0, 5.0 },  .{ -5.0, -2.0 }, .{ -2.0, -5.0 },
            .{ 5.0, -2.0 },  .{ -5.0, 2.0 }, .{ 0.0, 5.0 },   .{ 5.0, 0.0 },
            .{ 0.0, -5.0 },  .{ -5.0, 0.0 }, .{ 0.0, 0.0 },   .{ 3.0, 3.0 },
            .{ -3.0, -3.0 },
        };
        const base_f: [pairs.len]f32 = @splat(100.0);
        var d1_f: [pairs.len]f32 = undefined;
        var d2_f: [pairs.len]f32 = undefined;
        for (pairs, 0..) |pr, i| {
            d1_f[i] = pr[0];
            d2_f[i] = pr[1];
        }
        const fmts3 = [_]ast.SampleFormat{ fmt, fmt, fmt };
        const srcs3 = [_][*]const u8{ @ptrCast(&base_f), @ptrCast(&d1_f), @ptrCast(&d2_f) };
        const strides3 = [_]isize{ @sizeOf(@TypeOf(base_f)), @sizeOf(@TypeOf(d1_f)), @sizeOf(@TypeOf(d2_f)) };
        for ([_][]const u8{
            "y z xor x x y abs z abs < y z ? + ?",
            "y z xor x x y abs z abs < y z ? - ?",
            "y z xor 0 y abs z abs < y z ? ?",
        }) |expr| {
            var unopt = try parser.parse(allocator, expr, .{ .num_clips = 3 });
            defer unopt.deinit(allocator);
            try lower_mod.lower(allocator, &unopt, fma_prof);

            var opt = try parser.parse(allocator, expr, .{ .num_clips = 3 });
            defer opt.deinit(allocator);
            try lower_mod.lower(allocator, &opt, fma_prof);
            try optimize(allocator, &opt, fma_prof, .{ .enable_value_changing = true, .enable_fma = true });

            // Collapsed AST must no longer contain xor or select.
            for (opt.nodes.items) |n| {
                try std.testing.expect(!(n == .binary and n.binary.op == .xor_op));
                try std.testing.expect(!(n == .ternary and n.ternary.op == .select));
            }

            var dst_u: [pairs.len]f32 = @splat(0);
            var dst_o: [pairs.len]f32 = @splat(0);
            try interp.evalPlane(allocator, &unopt, fma_prof, .{
                .dst = @ptrCast(&dst_u),
                .dst_stride = strides3[0],
                .dst_fmt = fmt,
                .srcs = &srcs3,
                .src_strides = &strides3,
                .src_fmts = &fmts3,
                .width = pairs.len,
                .height = 1,
                .frame_n = 0,
                .props = &.{},
            });
            try interp.evalPlane(allocator, &opt, fma_prof, .{
                .dst = @ptrCast(&dst_o),
                .dst_stride = strides3[0],
                .dst_fmt = fmt,
                .srcs = &srcs3,
                .src_strides = &strides3,
                .src_fmts = &fmts3,
                .width = pairs.len,
                .height = 1,
                .frame_n = 0,
                .props = &.{},
            });
            try std.testing.expectEqualSlices(f32, &dst_u, &dst_o);
        }
    }
}
