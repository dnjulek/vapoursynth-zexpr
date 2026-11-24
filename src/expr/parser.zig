//! Iterative RPN stack parser building a flat-arena DAG (`ast.Program`).

const std = @import("std");
const ast = @import("ast.zig");
const builtins_mod = @import("builtins.zig");
const token = @import("token.zig");
const lexer = @import("lexer.zig");
const sortnet = @import("sortnet.zig");

pub const ParseOptions = struct {
    mode: ast.ExprMode = .expr,
    num_clips: u16 = 1,
    default_boundary: ast.BoundaryMode = .clamp,
    opt_int: bool = false,
    /// Optional built-in resolution context for compile-time literal emission in `Expr` (D6).
    builtin_ctx: ?*const builtins_mod.ResolveContext = null,
};

pub const ParseError = token.TokenError || builtins_mod.ResolveError || std.mem.Allocator.Error || error{
    EmptyExpression,
    StackUnderflow,
    StackNotEmpty,
    ClipOutOfRange,
    UninitializedVariable,
    UnsupportedInMode,
    SortTooLarge,
    PolyvalTooLarge,
    TooManyNodes,
};

const max_nodes: usize = 2_000_000;
const max_polyval_degree: u32 = 1024;

pub fn parse(allocator: std.mem.Allocator, source: []const u8, opts: ParseOptions) ParseError!ast.Program {
    var prog: ast.Program = .{
        .opt_int = opts.opt_int,
        .mode = opts.mode,
    };
    errdefer prog.deinit(allocator);

    var stack: std.ArrayList(ast.NodeIdx) = .empty;
    defer stack.deinit(allocator);

    var vars: std.StringHashMapUnmanaged(ast.NodeIdx) = .empty;
    defer vars.deinit(allocator);

    var net_pairs: std.ArrayList(sortnet.Pair) = .empty;
    defer net_pairs.deinit(allocator);

    var lex = lexer.Lexer.init(source);
    while (try lex.next()) |tok| {
        if (prog.nodes.items.len >= max_nodes) return error.TooManyNodes;

        switch (tok.kind) {
            .literal => |lit| {
                const node: ast.Node = if (opts.opt_int and lit.i32_val != null)
                    .{ .const_i32 = lit.i32_val.? }
                else
                    .{ .const_f32 = lit.f32_val };
                const idx = try prog.appendNode(allocator, node);
                try stack.append(allocator, idx);
            },
            .pi => {
                const idx = try prog.appendNode(allocator, .{ .const_f32 = std.math.pi });
                try stack.append(allocator, idx);
            },
            .builtin_const => |spec| {
                switch (spec.target) {
                    .clip => |cid| if (cid >= opts.num_clips) return error.ClipOutOfRange,
                    .dst => {},
                }
                _ = try prog.internBuiltin(allocator, spec);
                if (opts.builtin_ctx) |ctx| {
                    const val = try builtins_mod.resolveOne(spec, ctx.src_metas, ctx.dst_meta);
                    const node: ast.Node = if (ctx.opt_int and val.i32_val != null)
                        .{ .const_i32 = val.i32_val.? }
                    else
                        .{ .const_f32 = val.f32Val() };
                    const idx = try prog.appendNode(allocator, node);
                    try stack.append(allocator, idx);
                } else if (opts.mode == .expr) {
                    // Validation / spec-collection parse for Expr (never compiled or evaluated).
                    const idx = try prog.appendNode(allocator, .{ .const_f32 = 0.0 });
                    try stack.append(allocator, idx);
                } else {
                    // Select / PropExpr: intern into props as a `.builtin` slot reusing `.load_prop` (D2/F1).
                    const prop_id = try prog.internBuiltinProp(allocator, spec);
                    const idx = try prog.appendNode(allocator, .{ .load_prop = prop_id });
                    try stack.append(allocator, idx);
                }
            },
            .frame_n => {
                const idx = try prog.appendNode(allocator, .frame_n);
                try stack.append(allocator, idx);
            },
            .plane_width => {
                const idx = try prog.appendNode(allocator, .plane_width);
                try stack.append(allocator, idx);
            },
            .plane_height => {
                const idx = try prog.appendNode(allocator, .plane_height);
                try stack.append(allocator, idx);
            },
            .coord_x => {
                if (opts.mode != .expr) return error.UnsupportedInMode;
                const idx = try prog.appendNode(allocator, .coord_x);
                try stack.append(allocator, idx);
            },
            .coord_y => {
                if (opts.mode != .expr) return error.UnsupportedInMode;
                const idx = try prog.appendNode(allocator, .coord_y);
                try stack.append(allocator, idx);
            },
            .clip_load => |clip| {
                if (opts.mode != .expr) return error.UnsupportedInMode;
                if (clip >= opts.num_clips) return error.ClipOutOfRange;
                const idx = try prog.appendNode(allocator, .{ .load_clip = clip });
                try stack.append(allocator, idx);
            },
            .clip_rel => |rel| {
                if (opts.mode != .expr) return error.UnsupportedInMode;
                if (rel.clip >= opts.num_clips) return error.ClipOutOfRange;
                const mode = rel.mode orelse opts.default_boundary;
                const idx = try prog.appendNode(allocator, .{ .load_rel = .{
                    .clip = rel.clip,
                    .dx = rel.dx,
                    .dy = rel.dy,
                    .mode = mode,
                } });
                try stack.append(allocator, idx);
            },
            .clip_abs => |abs_acc| {
                if (opts.mode != .expr) return error.UnsupportedInMode;
                if (abs_acc.clip >= opts.num_clips) return error.ClipOutOfRange;
                if (stack.items.len < 2) return error.StackUnderflow;
                const y_node = stack.pop().?;
                const x_node = stack.pop().?;
                // D7 / Y16: suffix-less `x[]` always clamps regardless of `boundary`.
                const mode = abs_acc.mode orelse .clamp;
                const idx = try prog.appendNode(allocator, .{ .load_abs = .{
                    .clip = abs_acc.clip,
                    .x = x_node,
                    .y = y_node,
                    .mode = mode,
                } });
                try stack.append(allocator, idx);
            },
            .prop_load => |prop| {
                if (prop.clip >= opts.num_clips) return error.ClipOutOfRange;
                const prop_id = try prog.internProp(allocator, prop.clip, prop.name);
                const idx = try prog.appendNode(allocator, .{ .load_prop = prop_id });
                try stack.append(allocator, idx);
            },
            .var_store => |name| {
                if (stack.items.len < 1) return error.StackUnderflow;
                const val = stack.pop().?;
                try vars.put(allocator, name, val);
            },
            .var_load => |name| {
                const val = vars.get(name) orelse return error.UninitializedVariable;
                try stack.append(allocator, val);
            },
            .unary => |op| {
                if (stack.items.len < 1) return error.StackUnderflow;
                const a = stack.pop().?;
                const idx = try prog.appendNode(allocator, .{ .unary = .{ .op = op, .a = a } });
                try stack.append(allocator, idx);
            },
            .binary => |op| {
                if (stack.items.len < 2) return error.StackUnderflow;
                const b = stack.pop().?;
                const a = stack.pop().?;
                const idx = try prog.appendNode(allocator, .{ .binary = .{ .op = op, .a = a, .b = b } });
                try stack.append(allocator, idx);
            },
            .ternary_select => {
                if (stack.items.len < 3) return error.StackUnderflow;
                const false_val = stack.pop().?;
                const true_val = stack.pop().?;
                const cond = stack.pop().?;
                const idx = try prog.appendNode(allocator, .{ .ternary = .{
                    .op = .select,
                    .a = cond,
                    .b = true_val,
                    .c = false_val,
                } });
                try stack.append(allocator, idx);
            },
            .clip_op => {
                // `x lo hi clip` -> `max(min(x, hi), lo)` so `x 3 1 clip = 3`.
                if (stack.items.len < 3) return error.StackUnderflow;
                const hi = stack.pop().?;
                const lo = stack.pop().?;
                const x = stack.pop().?;
                const min_node = try prog.appendNode(allocator, .{ .binary = .{ .op = .min, .a = x, .b = hi } });
                const max_node = try prog.appendNode(allocator, .{ .binary = .{ .op = .max, .a = min_node, .b = lo } });
                try stack.append(allocator, max_node);
            },
            .lerp_op => {
                // `a b t lerp`:
                // - constant float `t`: `a * (1 - t) + b * t` -> `vmulps` + `vfmadd231ps` (7 cyc, exact at t=0,1).
                // - dynamic `t`: `(a - a * t) + b * t` -> `vfnmadd213ps` + `vfmadd231ps` (0 extra consts, exact at t=0,1).
                if (stack.items.len < 3) return error.StackUnderflow;
                const t = stack.pop().?;
                const b = stack.pop().?;
                const a = stack.pop().?;
                if (!opts.opt_int and prog.nodes.items[t] == .const_f32) {
                    const one_minus_t = try prog.appendNode(allocator, .{ .const_f32 = 1.0 - prog.nodes.items[t].const_f32 });
                    const a_scaled = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = a, .b = one_minus_t } });
                    const b_scaled = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = b, .b = t } });
                    const res = try prog.appendNode(allocator, .{ .binary = .{ .op = .add, .a = b_scaled, .b = a_scaled } });
                    try stack.append(allocator, res);
                } else {
                    const at = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = a, .b = t } });
                    const a_rem = try prog.appendNode(allocator, .{ .binary = .{ .op = .sub, .a = a, .b = at } });
                    const bt = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = b, .b = t } });
                    const res = try prog.appendNode(allocator, .{ .binary = .{ .op = .add, .a = a_rem, .b = bt } });
                    try stack.append(allocator, res);
                }
            },
            .hypot_op => {
                // `a b hypot` -> `sqrt(a*a + b*b)` (fuses to `mul + fmadd + sqrt` on FMA tiers).
                if (stack.items.len < 2) return error.StackUnderflow;
                var b = stack.pop().?;
                var a = stack.pop().?;
                if (opts.opt_int) {
                    a = try prog.appendNode(allocator, .{ .unary = .{ .op = .to_f32, .a = a } });
                    b = try prog.appendNode(allocator, .{ .unary = .{ .op = .to_f32, .a = b } });
                }
                const a2 = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = a, .b = a } });
                const b2 = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = b, .b = b } });
                const sum = try prog.appendNode(allocator, .{ .binary = .{ .op = .add, .a = a2, .b = b2 } });
                const res = try prog.appendNode(allocator, .{ .unary = .{ .op = .sqrt, .a = sum } });
                try stack.append(allocator, res);
            },
            .linstep_op, .smoothstep_op => {
                // `lo hi x linstep`    -> `clamp((x - lo) / (hi - lo), 0, 1)` (expanded in `lower.zig` Phase B
                //                         after constant folding so both literal and constant-expr bounds get exact endpoints).
                // `lo hi x smoothstep` -> `t = linstep(lo, hi, x); t * t * (3 - 2 * t)`
                if (stack.items.len < 3) return error.StackUnderflow;
                const x = stack.pop().?;
                const hi = stack.pop().?;
                const lo = stack.pop().?;
                const t = try prog.appendNode(allocator, .{ .ternary = .{ .op = .linstep, .a = lo, .b = hi, .c = x } });
                if (tok.kind == .linstep_op) {
                    try stack.append(allocator, t);
                } else {
                    const two = try prog.appendNode(allocator, .{ .const_f32 = 2.0 });
                    const three = try prog.appendNode(allocator, .{ .const_f32 = 3.0 });
                    const t2 = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = t, .b = t } });
                    const two_t = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = two, .b = t } });
                    const hermite = try prog.appendNode(allocator, .{ .binary = .{ .op = .sub, .a = three, .b = two_t } });
                    const res = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = t2, .b = hermite } });
                    try stack.append(allocator, res);
                }
            },
            .difflimit_op => {
                // `d1 d2 difflimit` -> `clamp(d1, min(0, d2), max(0, d2))` (4 min/max ops, native i16).
                if (stack.items.len < 2) return error.StackUnderflow;
                const d2 = stack.pop().?;
                const d1 = stack.pop().?;
                const zero = try prog.appendNode(allocator, if (opts.opt_int) .{ .const_i32 = 0 } else .{ .const_f32 = 0.0 });
                const lo = try prog.appendNode(allocator, .{ .binary = .{ .op = .min, .a = zero, .b = d2 } });
                const hi = try prog.appendNode(allocator, .{ .binary = .{ .op = .max, .a = zero, .b = d2 } });
                const c_hi = try prog.appendNode(allocator, .{ .binary = .{ .op = .min, .a = d1, .b = hi } });
                const res = try prog.appendNode(allocator, .{ .binary = .{ .op = .max, .a = c_hi, .b = lo } });
                try stack.append(allocator, res);
            },
            .absdiff_op => {
                // `a b absdiff` -> `abs(a - b)` (maps to native u8/u16 `max_u - min_u`).
                if (stack.items.len < 2) return error.StackUnderflow;
                const b = stack.pop().?;
                const a = stack.pop().?;
                const diff = try prog.appendNode(allocator, .{ .binary = .{ .op = .sub, .a = a, .b = b } });
                const res = try prog.appendNode(allocator, .{ .unary = .{ .op = .abs, .a = diff } });
                try stack.append(allocator, res);
            },
            .minabs_op, .maxabs_op => {
                // `a b minabs` -> `|a| < |b| ? a : b`; `a b maxabs` -> `|a| > |b| ? a : b`.
                if (stack.items.len < 2) return error.StackUnderflow;
                const b = stack.pop().?;
                const a = stack.pop().?;
                const abs_a = try prog.appendNode(allocator, .{ .unary = .{ .op = .abs, .a = a } });
                const abs_b = try prog.appendNode(allocator, .{ .unary = .{ .op = .abs, .a = b } });
                const cmp_op: ast.BinaryOp = if (tok.kind == .minabs_op) .lt else .gt;
                const cond = try prog.appendNode(allocator, .{ .binary = .{ .op = cmp_op, .a = abs_a, .b = abs_b } });
                const res = try prog.appendNode(allocator, .{ .ternary = .{ .op = .select, .a = cond, .b = a, .c = b } });
                try stack.append(allocator, res);
            },
            .polyval => |degree_u32| {
                // `c_N ... c_1 c_0 x polyvalN` -> `c_N * x^N + ... + c_1 * x + c_0`.
                // Benchmarked hybrid:
                // - degree 3..5: odd-even Horner (`p_even(x^2) + x * p_odd(x^2)`) cuts FMA chain depth
                //   while keeping all coefficients in vector registers.
                // - degree 1..2 and >= 6: standard Horner minimizes total uops and register pressure
                //   (keeps up to degree-11 coefficients register-resident with 0 spills).
                if (degree_u32 > max_polyval_degree) return error.PolyvalTooLarge;
                const degree: usize = degree_u32;
                const needed = degree + 2;
                if (stack.items.len < needed) return error.StackUnderflow;
                const x = stack.pop().?;
                const coeff_start = stack.items.len - (degree + 1);
                if (degree == 0) {
                    const c0 = stack.pop().?;
                    try stack.append(allocator, c0);
                } else if (degree >= 3 and degree <= 5) {
                    const m = degree + 1;
                    const x2 = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = x, .b = x } });
                    var even_acc: ?ast.NodeIdx = null;
                    var odd_acc: ?ast.NodeIdx = null;
                    for (0..m) |k| {
                        const coeff = stack.items[coeff_start + k];
                        const power = degree - k;
                        if ((power & 1) == 0) {
                            if (even_acc) |acc| {
                                const prod = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = acc, .b = x2 } });
                                even_acc = try prog.appendNode(allocator, .{ .binary = .{ .op = .add, .a = prod, .b = coeff } });
                            } else {
                                even_acc = coeff;
                            }
                        } else {
                            if (odd_acc) |acc| {
                                const prod = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = acc, .b = x2 } });
                                odd_acc = try prog.appendNode(allocator, .{ .binary = .{ .op = .add, .a = prod, .b = coeff } });
                            } else {
                                odd_acc = coeff;
                            }
                        }
                    }
                    stack.shrinkRetainingCapacity(coeff_start);
                    const odd_term = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = odd_acc.?, .b = x } });
                    const res = try prog.appendNode(allocator, .{ .binary = .{ .op = .add, .a = odd_term, .b = even_acc.? } });
                    try stack.append(allocator, res);
                } else {
                    var acc = stack.items[coeff_start];
                    for (stack.items[coeff_start + 1 ..]) |coeff| {
                        const prod = try prog.appendNode(allocator, .{ .binary = .{ .op = .mul, .a = acc, .b = x } });
                        acc = try prog.appendNode(allocator, .{ .binary = .{ .op = .add, .a = prod, .b = coeff } });
                    }
                    stack.shrinkRetainingCapacity(coeff_start);
                    try stack.append(allocator, acc);
                }
            },
            .dup => |n| {
                const depth: usize = n;
                if (stack.items.len <= depth) return error.StackUnderflow;
                const val = stack.items[stack.items.len - 1 - depth];
                try stack.append(allocator, val);
            },
            .swap => |n| {
                const depth: usize = n;
                if (stack.items.len <= depth) return error.StackUnderflow;
                if (depth > 0) {
                    const top_idx = stack.items.len - 1;
                    std.mem.swap(ast.NodeIdx, &stack.items[top_idx], &stack.items[top_idx - depth]);
                }
            },
            .drop => |n| {
                const count: usize = n;
                if (stack.items.len < count) return error.StackUnderflow;
                stack.shrinkRetainingCapacity(stack.items.len - count);
            },
            .sort => |n| {
                if (n > sortnet.max_sort_n) return error.SortTooLarge;
                const count: usize = n;
                if (stack.items.len < count) return error.StackUnderflow;
                if (count >= 2) {
                    net_pairs.clearRetainingCapacity();
                    try sortnet.appendNetwork(allocator, &net_pairs, count);
                    if (prog.nodes.items.len + net_pairs.items.len * 2 > max_nodes) {
                        return error.TooManyNodes;
                    }
                    // `at(0)` is top of stack (receives smallest); `at(count-1)` receives largest.
                    const top = stack.items.len - 1;
                    for (net_pairs.items) |p| {
                        const ia = top - p.a;
                        const ib = top - p.b;
                        const va = stack.items[ia];
                        const vb = stack.items[ib];
                        const lo = try prog.appendNode(allocator, .{ .binary = .{ .op = .min, .a = va, .b = vb } });
                        const hi = try prog.appendNode(allocator, .{ .binary = .{ .op = .max, .a = va, .b = vb } });
                        stack.items[ia] = lo;
                        stack.items[ib] = hi;
                    }
                }
            },
            .argmin, .argmax => |n| {
                if (opts.mode == .expr) return error.UnsupportedInMode;
                if (n > sortnet.max_sort_n) return error.SortTooLarge;
                const count: usize = n;
                if (stack.items.len < count) return error.StackUnderflow;
                const start_stack = stack.items.len - count;
                const args_start: u32 = @intCast(prog.extra_args.items.len);
                try prog.extra_args.appendSlice(allocator, stack.items[start_stack..]);
                stack.shrinkRetainingCapacity(start_stack);
                const kind: ast.ArgExtremaKind = if (tok.kind == .argmin) .argmin else .argmax;
                const idx = try prog.appendNode(allocator, .{ .arg_extrema = .{
                    .kind = kind,
                    .args_start = args_start,
                    .count = @intCast(count),
                } });
                try stack.append(allocator, idx);
            },
            .argsort => |n| {
                if (opts.mode == .expr) return error.UnsupportedInMode;
                if (n > sortnet.max_sort_n) return error.SortTooLarge;
                const count: usize = n;
                if (stack.items.len < count) return error.StackUnderflow;
                const start_stack = stack.items.len - count;
                const args_start: u32 = @intCast(prog.extra_args.items.len);
                try prog.extra_args.appendSlice(allocator, stack.items[start_stack..]);
                stack.shrinkRetainingCapacity(start_stack);

                const sort_id: u32 = @intCast(prog.arg_sorts.items.len);
                try prog.arg_sorts.append(allocator, .{
                    .args_start = args_start,
                    .count = @intCast(count),
                });
                // Push ranks 0..count-1 in order so rank `count-1` (smallest value's index) ends on top,
                // matching akarin's stable descending sort placement.
                try stack.ensureUnusedCapacity(allocator, count);
                for (0..count) |rank| {
                    const idx = try prog.appendNode(allocator, .{ .arg_sort_elem = .{
                        .sort_id = sort_id,
                        .rank = @intCast(rank),
                    } });
                    stack.appendAssumeCapacity(idx);
                }
            },
        }
    }

    if (stack.items.len == 0) return error.EmptyExpression;
    if (stack.items.len > 1) return error.StackNotEmpty;
    prog.root = stack.items[0];
    return prog;
}

test "parser handles basic, spatial, var, sort289, sort1025, and 5000-op chain" {
    const allocator = std.testing.allocator;

    // Basic RPN + variables
    {
        var p = try parse(allocator, "x 2 * t! t@ t@ +", .{ .num_clips = 1 });
        defer p.deinit(allocator);
        try std.testing.expect(p.root != ast.invalid_node);
    }

    // Suffix-less `x[]` always clamps even with default_boundary == .mirror (Y16).
    {
        var p = try parse(allocator, "X Y x[]", .{ .num_clips = 1, .default_boundary = .mirror });
        defer p.deinit(allocator);
        try std.testing.expectEqual(ast.BoundaryMode.clamp, p.nodes.items[p.root].load_abs.mode);
    }

    // Relative access uses default_boundary unless suffixed.
    {
        var p = try parse(allocator, "x[1,0]", .{ .num_clips = 1, .default_boundary = .mirror });
        defer p.deinit(allocator);
        try std.testing.expectEqual(ast.BoundaryMode.mirror, p.nodes.items[p.root].load_rel.mode);
    }

    // Select rejects X/Y and pixel loads (Y17).
    try std.testing.expectError(error.UnsupportedInMode, parse(allocator, "X", .{ .mode = .select, .num_clips = 1 }));
    try std.testing.expectError(error.UnsupportedInMode, parse(allocator, "x", .{ .mode = .select, .num_clips = 1 }));
    try std.testing.expectError(error.UnsupportedInMode, parse(allocator, "x.A x.B argmin2", .{ .mode = .expr, .num_clips = 1 }));

    // Uninitialized variable, stack underflow, unconsumed items, clip OOB.
    try std.testing.expectError(error.UninitializedVariable, parse(allocator, "foo@", .{}));
    try std.testing.expectError(error.StackUnderflow, parse(allocator, "x +", .{}));
    try std.testing.expectError(error.StackNotEmpty, parse(allocator, "x x", .{}));
    try std.testing.expectError(error.EmptyExpression, parse(allocator, "   ", .{}));
    try std.testing.expectError(error.ClipOutOfRange, parse(allocator, "y", .{ .num_clips = 1 }));

    // sort289 succeeds; sort1025 rejected (Z3).
    {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(allocator);
        for (0..289) |_| try buf.appendSlice(allocator, "x ");
        try buf.appendSlice(allocator, "sort289 drop288");
        var p = try parse(allocator, buf.items, .{ .num_clips = 1 });
        defer p.deinit(allocator);
    }
    try std.testing.expectError(error.SortTooLarge, parse(allocator, "x sort1025", .{ .num_clips = 1 }));

    // New operators: neg, ceil, atan, asin, acos, lerp (const & dynamic), polyvalN / polyevalN.
    {
        for ([_][]const u8{
            "x neg",
            "x ceil",
            "x atan",
            "x asin",
            "x acos",
            "x y 0.3 lerp",
            "x y z lerp",
            "5 x polyval0",
            "2 3 x polyval1",
            "3 2 1 x polyval2",
            "4 3 2 1 x polyval3",
            "6 5 4 3 2 1 x polyeval5",
            "8 7 6 5 4 3 2 1 x polyval7",
        }) |expr| {
            var p = try parse(allocator, expr, .{ .num_clips = 3 });
            p.deinit(allocator);
        }
        // Polyval degree limit: one below (1023), at (1024), and above (1025).
        for ([_]usize{ 1023, 1024 }) |deg| {
            var buf: std.ArrayList(u8) = .empty;
            defer buf.deinit(allocator);
            for (0..deg + 1) |_| try buf.appendSlice(allocator, "1 ");
            try buf.print(allocator, "x polyval{d}", .{deg});
            var p = try parse(allocator, buf.items, .{ .num_clips = 1 });
            p.deinit(allocator);
        }
        try std.testing.expectError(error.PolyvalTooLarge, parse(allocator, "x polyval1025", .{ .num_clips = 1 }));
        try std.testing.expectError(error.StackUnderflow, parse(allocator, "2 1 x polyval2", .{ .num_clips = 1 }));
        try std.testing.expectError(error.StackUnderflow, parse(allocator, "x y lerp", .{ .num_clips = 2 }));
    }

    // 5,000-op deep chain succeeds without stack overflow (Y26).
    {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(allocator);
        try buf.appendSlice(allocator, "x");
        for (0..5000) |_| try buf.appendSlice(allocator, " 1 +");
        var p = try parse(allocator, buf.items, .{ .num_clips = 1 });
        defer p.deinit(allocator);
        try std.testing.expect(p.nodes.items.len > 10_000);
    }

    // Built-in constants: parser-level literal resolution (D1/D6), prop slots in Select/PropExpr (D2), C4 rejection.
    {
        const u8_meta: builtins_mod.ClipMeta = .{ .is_float = false, .bits_per_sample = 8, .color_family = .gray, .plane_idx = 0, .is_full_range = false };
        const f32_meta: builtins_mod.ClipMeta = .{ .is_float = true, .bits_per_sample = 32, .color_family = .gray, .plane_idx = 0, .is_full_range = true };
        const u32_meta: builtins_mod.ClipMeta = .{ .is_float = false, .bits_per_sample = 32, .color_family = .gray, .plane_idx = 0, .is_full_range = false };

        // D1: under opt=1, integer target emits const_i32, float target emits const_f32 even for 1.0.
        const ctx_u8: builtins_mod.ResolveContext = .{ .src_metas = &.{u8_meta}, .dst_meta = u8_meta, .opt_int = true };
        var p_u8 = try parse(allocator, "range_max", .{ .num_clips = 1, .opt_int = true, .builtin_ctx = &ctx_u8 });
        defer p_u8.deinit(allocator);
        try std.testing.expectEqual(ast.Node{ .const_i32 = 255 }, p_u8.nodes.items[p_u8.root]);
        try std.testing.expectEqual(@as(usize, 0), p_u8.props.items.len);

        const ctx_f32: builtins_mod.ResolveContext = .{ .src_metas = &.{f32_meta}, .dst_meta = f32_meta, .opt_int = true };
        var p_f32 = try parse(allocator, "range_max", .{ .num_clips = 1, .opt_int = true, .builtin_ctx = &ctx_f32 });
        defer p_f32.deinit(allocator);
        try std.testing.expectEqual(ast.Node{ .const_f32 = 1.0 }, p_f32.nodes.items[p_f32.root]);

        // C4: >16-bit integer target returns UnsupportedIntBitDepthConstant.
        const ctx_u32: builtins_mod.ResolveContext = .{ .src_metas = &.{u32_meta}, .dst_meta = u32_meta, .opt_int = false };
        try std.testing.expectError(
            error.UnsupportedIntBitDepthConstant,
            parse(allocator, "neutral", .{ .num_clips = 1, .builtin_ctx = &ctx_u32 }),
        );
        try std.testing.expectError(error.ClipOutOfRange, parse(allocator, "y:neutral", .{ .num_clips = 1 }));

        // D2: Select / PropExpr intern built-ins into props as `.builtin` and emit `.load_prop`.
        var p_sel = try parse(allocator, "x:range_max x:range_max + dst:range_min +", .{ .mode = .select, .num_clips = 1 });
        defer p_sel.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 2), p_sel.props.items.len);
        try std.testing.expect(p_sel.props.items[0] == .builtin);
        try std.testing.expect(p_sel.props.items[1] == .builtin);
        try std.testing.expectEqual(@as(usize, 2), p_sel.builtins.items.len);
    }
}

test "parser OOM and remap failure sweep" {
    const Runner = struct {
        fn run(allocator: std.mem.Allocator) anyerror!void {
            const exprs = [_][]const u8{
                "x y + t! t@ x[-1,0]:m + x.PropA + X Y src0[]:m + x 1 2 3 4 sort5 drop4 + x y 0.3 lerp + 1 2 3 4 5 6 x polyval5 + neutral + dst:range_max +",
                "x.A x.B x.C argsort3 drop2 x:neutral + dst:range_max +",
            };
            var p0 = try parse(allocator, exprs[0], .{ .num_clips = 2, .default_boundary = .mirror });
            p0.deinit(allocator);
            var p1 = try parse(allocator, exprs[1], .{ .mode = .select, .num_clips = 2 });
            p1.deinit(allocator);
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Runner.run, .{});

    // Also exercise remap/resize failures directly.
    var fail_idx: usize = 0;
    while (fail_idx < 128) : (fail_idx += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{
            .resize_fail_index = fail_idx,
        });
        var p = parse(failing.allocator(), "x 1 2 3 4 5 6 7 8 sort9 drop8 x.PropA + t! t@", .{ .num_clips = 1 }) catch |err| switch (err) {
            error.OutOfMemory => continue,
            else => return err,
        };
        p.deinit(failing.allocator());
        if (failing.deallocations == failing.allocations) break;
    }
}

fn fuzzParseCallback(_: void, smith: *std.testing.Smith) anyerror!void {
    var input_buf: [256]u8 = undefined;
    const len = smith.slice(&input_buf);
    const mode_idx = smith.valueRangeAtMost(u8, 0, 2);
    const mode: ast.ExprMode = switch (mode_idx) {
        0 => .expr,
        1 => .select,
        else => .prop_expr,
    };
    const boundary: ast.BoundaryMode = if (smith.value(bool)) .mirror else .clamp;
    const opt_int = smith.value(bool);
    const num_clips = smith.valueRangeAtMost(u16, 1, 8);

    var p = parse(std.testing.allocator, input_buf[0..len], .{
        .mode = mode,
        .num_clips = num_clips,
        .default_boundary = boundary,
        .opt_int = opt_int,
    }) catch return;
    p.deinit(std.testing.allocator);
}

test "parser fuzz entry" {
    try std.testing.fuzz({}, fuzzParseCallback, .{
        .corpus = &.{
            "x 2 * 1 +",
            "x[-1,0]:m y[1,1]:c + 2 /",
            "X Y x[]:m",
            "x 0xFFFFFFFF bitand",
            "x y z sort3 drop2",
            "x.A x.B argmin2",
        },
    });
}
